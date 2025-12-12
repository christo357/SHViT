import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from timm.models import create_model
from data import samplers  # noqa: F401
from data.datasets import build_dataset
from engine import evaluate
import utils  # noqa: F401 - needed by your codebase
import model  # noqa: F401 - registers SHViT models

from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
class GeometricCorruptionDataset(Dataset):
    """
    Wraps an existing dataset and applies geometric / color changes
    to the already-transformed image tensor (C, H, W).

    mode:
      - 'rotation': param = angle in degrees
      - 'color':    param = 'clean' or 'grayscale'
      - 'crop':     param = crop scale (0 < s <= 1.0)
    """

    def __init__(self, base_dataset, mode: str, param=None):
        self.base_dataset = base_dataset
        self.mode = mode
        self.param = param

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img is tensor (C, H, W)
        img = self.apply_geometric(img)
        return img, label

    def apply_geometric(self, img: torch.Tensor) -> torch.Tensor:
        if self.mode == "rotation":
            angle = float(self.param) if self.param is not None else 0.0
            if angle != 0.0:
                img = TF.rotate(
                    img,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                )
            return img

        elif self.mode == "color":
            if self.param == "grayscale":
                gray = img.mean(dim=0, keepdim=True)
                img = gray.repeat(3, 1, 1)
            # 'clean' = no change
            return img

        elif self.mode == "crop":
            # Center crop by a given scale, then upsample back to (H, W)
            scale = float(self.param) if self.param is not None else 1.0
            scale = max(min(scale, 1.0), 0.01)  # clip to (0,1]
            C, H, W = img.shape

            crop_h = max(1, int(H * scale))
            crop_w = max(1, int(W * scale))
            crop_h = min(crop_h, H)
            crop_w = min(crop_w, W)

            start_y = (H - crop_h) // 2
            start_x = (W - crop_w) // 2

            cropped = img[:, start_y:start_y + crop_h, start_x:start_x + crop_w]
            # Upsample back to original resolution
            cropped_up = Fnn.interpolate(
                cropped.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return cropped_up

        else:
            # No change
            return img


def build_eval_dataset_and_num_classes(args) -> Tuple[torch.utils.data.Dataset, int]:
    """
    Uses your build_dataset() and dataset naming conventions.
    """

    # Mirror your robustness script's eval_args setup
    eval_args = argparse.Namespace(**vars(args))
    eval_args.data_set = args.dataset
    eval_args.input_size = 224
    eval_args.color_jitter = 0.4
    eval_args.aa = "rand-m9-mstd0.5-inc1"
    eval_args.train_interpolation = "bicubic"
    eval_args.reprob = 0.25
    eval_args.remode = "pixel"
    eval_args.recount = 1
    eval_args.resplit = False
    eval_args.ThreeAugment = False
    eval_args.finetune = ""

    # Build validation dataset
    dataset_val, _ = build_dataset(is_train=False, args=eval_args)

    # Number of classes (same logic as robustness script)
    if args.dataset == "CIFAR":
        nb_classes = 100
    elif args.dataset == "EUROSAT":
        nb_classes = 10
    elif args.dataset == "MEDMNIST":
        try:
            import medmnist
            from medmnist import INFO

            info = INFO[args.medmnist_dataset]
            nb_classes = len(info["label"])
        except Exception:
            nb_classes = 9
    else:
        nb_classes = 1000

    return dataset_val, nb_classes


def load_state_dict_flex(net: torch.nn.Module, checkpoint: dict, model_key: str = "model"):
    if model_key in checkpoint:
        state_dict = checkpoint[model_key]
    else:
        state_dict = checkpoint

    model_dict = net.state_dict()
    loaded = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            loaded[k] = v
        else:
            skipped.append(k)

    model_dict.update(loaded)
    net.load_state_dict(model_dict)

    print(f"Loaded {len(loaded)} / {len(state_dict)} keys into model.")
    if skipped:
        print("Skipped keys (name or shape mismatch), first few:")
        for name in skipped[:10]:
            print(f"  - {name}")


def load_model_for_dataset(model_name: str, nb_classes: int, args, device: torch.device):
    checkpoint_dir = Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}_frac1.0"
    if args.dataset == "CIFAR":
        checkpoint_path = checkpoint_dir / "checkpoint_99.pth"
    else:
        checkpoint_path = checkpoint_dir / "checkpoint_29.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading {model_name} checkpoint from {checkpoint_path}")

    net = create_model(
        model_name,
        num_classes=nb_classes,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    try:
        net.load_state_dict(checkpoint["model"], strict=True)
        print("Loaded checkpoint with strict=True.")
    except RuntimeError as e:
        print("Strict load failed with:")
        print(e)
        print("Falling back to flexible shape-matched loading...")
        load_state_dict_flex(net, checkpoint, model_key="model")

    net.to(device)
    net.eval()
    return net


def evaluate_on_dataset(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    """
    Wraps engine.evaluate() to compute Acc@1 on a dataset.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    stats = evaluate(loader, model, device)
    return float(stats["acc1"])


def plot_multi_model_curve(
    x_values: List[float],
    acc_by_model: Dict[str, List[float]],
    x_label: str,
    title: str,
    output_path: Path,
    x_tick_labels: Optional[List[str]] = None,
):
    """
    Plot multiple models on the same curve plot.

    - Different line styles / markers per model.
    - Same x-axis (e.g., rotation angle, color condition index, crop scale).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "d", "v", "x", "P", "*"]

    all_accs = []

    for i, (model_name, accs) in enumerate(acc_by_model.items()):
        style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        ax.plot(
            x_values,
            accs,
            linestyle=style,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=model_name,
        )
        all_accs.extend(accs)

    if x_tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    if len(all_accs) > 0:
        y_min = min(all_accs) - 5
        y_max = max(all_accs) + 5
        ax.set_ylim(y_min, y_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()

def _tensor_to_image(img: torch.Tensor):
    """
    Convert CxHxW tensor (possibly normalized) to HxWxC in [0,1] for plotting.
    """
    img = img.detach().cpu().float()
    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img_min = img.min()
    img = img - img_min
    img_max = img.max()
    if img_max > 0:
        img = img / img_max
    return img.numpy()


def _save_row_figure(
    imgs: List[torch.Tensor],
    titles: List[str],
    main_title: str,
    out_path: Path,
):
    assert len(imgs) == len(titles)
    n = len(imgs)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(2.8 * n, 3),
        gridspec_kw={"wspace": 0.02},
    )
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(_tensor_to_image(img))
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    fig.suptitle(
        main_title,
        fontsize=14,
        fontweight="bold",
        y=0.98,
        va="bottom",
    )

    plt.subplots_adjust(
        top=0.9,
        bottom=0.01,
        left=0.01,
        right=0.99,
        wspace=0.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample grid to {out_path}")


def save_geometric_sample_grids(args, dataset_val):
    """
    For each sample index in [0, num_sample_images),
    create three figures:

      <dataset>_rotation_sample<n>.png
      <dataset>_crop_sample<n>.png
      <dataset>_color_sample<n>.png

    Each figure is a row: [Original | variants...].
    """
    sample_dir = Path(getattr(args, "sample_output_dir", "outputs/geometric_sample"))
    sample_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(getattr(args, "num_sample_images", 1), len(dataset_val))
    dataset_name = args.dataset.lower()

    rotation_angles = [30, 60, 90, 120, 150, 180]
    crop_scales = [0.25, 0.5, 0.75, 1.0]
    color_modes = ["grayscale"]  # we show Original + Grayscale

    for idx in range(num_samples):
        base_img, _ = dataset_val[idx]  # C,H,W

        # ---------------- Rotation row ----------------
        rot_imgs = [base_img]
        rot_titles = ["Original"]
        rot_ds_dummy = GeometricCorruptionDataset(dataset_val, mode="rotation", param=0)

        for ang in rotation_angles:
            rot_ds_dummy.param = ang
            img_rot = rot_ds_dummy.apply_geometric(base_img.clone())
            rot_imgs.append(img_rot)
            rot_titles.append(f"{ang}°")

        out_path = sample_dir / f"{dataset_name}_rotation_sample{idx+1}.png"
        _save_row_figure(
            rot_imgs,
            rot_titles,
            main_title=f"Rotation Examples – {args.dataset}",
            out_path=out_path,
        )

        # ---------------- Crop row ----------------
        crop_imgs = [base_img]
        crop_titles = ["Original"]
        crop_ds_dummy = GeometricCorruptionDataset(dataset_val, mode="crop", param=1.0)

        for s in crop_scales:
            crop_ds_dummy.param = s
            img_cropped = crop_ds_dummy.apply_geometric(base_img.clone())
            crop_imgs.append(img_cropped)
            crop_titles.append(f"{s:.2f}x")

        out_path = sample_dir / f"{dataset_name}_crop_sample{idx+1}.png"
        _save_row_figure(
            crop_imgs,
            crop_titles,
            main_title=f"Crop Examples – {args.dataset}",
            out_path=out_path,
        )

        # ---------------- Color row ----------------
        color_imgs = [base_img]
        color_titles = ["Original"]
        color_ds_dummy = GeometricCorruptionDataset(dataset_val, mode="color", param=None)

        for mode in color_modes:
            color_ds_dummy.param = mode
            img_color = color_ds_dummy.apply_geometric(base_img.clone())
            title = "Grayscale" if mode == "grayscale" else mode
            color_imgs.append(img_color)
            color_titles.append(title)

        out_path = sample_dir / f"{dataset_name}_color_sample{idx+1}.png"
        _save_row_figure(
            color_imgs,
            color_titles,
            main_title=f"Color Examples – {args.dataset}",
            out_path=out_path,
        )

def run_rotation_test(
    args,
    dataset_val,
    models: Dict[str, torch.nn.Module],
    device,
):
    """
    Rotation invariance:
      angles = [30, 60, 90, 120, 150, 180]
    Returns:
      angles, acc_by_model (dict: model_name -> list of accs)
    """
    angles = [30, 60, 90, 120, 150, 180]
    acc_by_model: Dict[str, List[float]] = {m: [] for m in models.keys()}

    print("\n" + "=" * 80)
    print(f"Rotation invariance test on {args.dataset}")
    print("=" * 80)

    for angle in angles:
        print(f"\nRotation: {angle} degrees")

        rot_ds = GeometricCorruptionDataset(
            base_dataset=dataset_val,
            mode="rotation",
            param=angle,
        )

        for name, net in models.items():
            acc = evaluate_on_dataset(
                rot_ds, net, device, args.batch_size, args.num_workers
            )
            acc_by_model[name].append(acc)
            print(f"  {name:25s} Acc@1: {acc:.2f}%")

    return angles, acc_by_model


def run_color_test(
    args,
    dataset_val,
    models: Dict[str, torch.nn.Module],
    device,
):
    """
    Color invariance:
      conditions = ["clean", "grayscale"]
    Returns:
      x_vals (indices), cond_labels, acc_by_model (dict: model_name -> list)
    """
    conditions = ["clean", "grayscale"]
    x_vals = list(range(len(conditions)))
    acc_by_model: Dict[str, List[float]] = {m: [] for m in models.keys()}

    print("\n" + "=" * 80)
    print(f"Color / grayscale invariance test on {args.dataset}")
    print("=" * 80)

    for cond in conditions:
        print(f"\nCondition: {cond}")

        color_ds = GeometricCorruptionDataset(
            base_dataset=dataset_val,
            mode="color",
            param=cond,
        )

        for name, net in models.items():
            acc = evaluate_on_dataset(
                color_ds, net, device, args.batch_size, args.num_workers
            )
            acc_by_model[name].append(acc)
            print(f"  {name:25s} Acc@1: {acc:.2f}%")

    return x_vals, conditions, acc_by_model


def run_crop_test(
    args,
    dataset_val,
    models: Dict[str, torch.nn.Module],
    device,
):
    """
    Crop invariance:
      scales = [0.25, 0.5, 0.75, 1.0]
    (Center crop to scale*H/W, then upsample back to H/W.)

    Returns:
      scales, acc_by_model
    """
    scales = [0.25, 0.5, 0.75, 1.0]
    acc_by_model: Dict[str, List[float]] = {m: [] for m in models.keys()}

    print("\n" + "=" * 80)
    print(f"Crop invariance test on {args.dataset}")
    print("=" * 80)

    for scale in scales:
        print(f"\nCrop scale: {scale}x")

        crop_ds = GeometricCorruptionDataset(
            base_dataset=dataset_val,
            mode="crop",
            param=scale,
        )

        for name, net in models.items():
            acc = evaluate_on_dataset(
                crop_ds, net, device, args.batch_size, args.num_workers
            )
            acc_by_model[name].append(acc)
            print(f"  {name:25s} Acc@1: {acc:.2f}%")

    return scales, acc_by_model

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Geometric & Color Invariance Analysis", add_help=True
    )

    # Old-style two-model args (kept for backward compatibility)
    parser.add_argument("--baseline-model", default="mobilenetv2_100", type=str)
    parser.add_argument("--shvit-model", default="shvit_s2", type=str)

    # NEW: multi-model arg – overrides baseline/shvit if provided
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="List of model names to evaluate. If set, overrides baseline/shvit.",
    )

    # Dataset settings (same as robustness script)
    parser.add_argument(
        "--dataset",
        default="CIFAR",
        type=str,
        choices=["CIFAR", "EUROSAT", "MEDMNIST"],
    )
    parser.add_argument(
        "--medmnist-dataset",
        default="pathmnist",
        type=str,
        help="MedMNIST subset name, if using MEDMNIST",
    )
    parser.add_argument(
        "--data-path",
        default="dataset",
        type=str,
        help="Root directory for data (same as training script).",
    )

    # Checkpoints (frac1.0) – match BASE_OUTPUT from training script
    parser.add_argument(
        "--checkpoint-dir",
        default="learning_curve_results",
        type=str,
        help="Directory containing <model>_<DATASET>_frac1.0 checkpoints.",
    )

    # Evaluation settings
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    # Output
    parser.add_argument(
        "--output-dir",
        default="outputs/geometric_variations",
        type=str,
    )

    parser.add_argument(
        "--sample-output-dir",
        default="outputs/geometric_sample",
        type=str,
        help="Directory to save geometric sample images.",
    )
    parser.add_argument(
        "--num-sample-images",
        default=4,
        type=int,
        help="Number of sample images to visualize per geometric setting.",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    args.data_set = args.dataset  # some parts of your code may expect this
    args.data_path = args.data_path

    # Decide which models to use
    if args.models is not None and len(args.models) > 0:
        model_names = args.models
    else:
        # fallback to the original two-model interface
        model_names = [args.baseline_model, args.shvit_model]

    # 1) Build eval dataset and infer num classes
    dataset_val, nb_classes = build_eval_dataset_and_num_classes(args)

    # Optional: save visualization grids
    # save_geometric_sample_grids(args, dataset_val)

    # 2) Load all requested models from frac1.0 checkpoints
    models: Dict[str, torch.nn.Module] = {}
    for mname in model_names:
        models[mname] = load_model_for_dataset(
            mname, nb_classes, args, device
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Rotation ----------------
    angles, acc_rot_by_model = run_rotation_test(
        args, dataset_val, models, device
    )
    plot_multi_model_curve(
        x_values=angles,
        acc_by_model=acc_rot_by_model,
        x_label="Rotation (degrees)",
        title=f"{args.dataset}: Rotation Invariance",
        output_path=out_dir / f"{args.dataset}_rotation_multi.png",
        x_tick_labels=[str(a) for a in angles],
    )

    # ---------------- Color / grayscale ----------------
    x_vals, cond_labels, acc_color_by_model = run_color_test(
        args, dataset_val, models, device
    )
    plot_multi_model_curve(
        x_values=x_vals,
        acc_by_model=acc_color_by_model,
        x_label="Color condition",
        title=f"{args.dataset}: Color / Grayscale Invariance",
        output_path=out_dir / f"{args.dataset}_color_multi.png",
        x_tick_labels=cond_labels,
    )

    # ---------------- Crop ----------------
    scales, acc_crop_by_model = run_crop_test(
        args, dataset_val, models, device
    )
    plot_multi_model_curve(
        x_values=scales,
        acc_by_model=acc_crop_by_model,
        x_label="Crop scale (relative area)",
        title=f"{args.dataset}: Crop Invariance",
        output_path=out_dir / f"{args.dataset}_crop_multi.png",
        x_tick_labels=[f"{s:.2f}x" for s in scales],
    )

    print("\n✓ Geometric (rotation, color, crop) invariance analysis complete!")


if __name__ == "__main__":
    main()