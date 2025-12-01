"""
Geometric & Color Invariance Analysis

Evaluates how robust models are to:
  - Rotations (0, 15, 30, 45 degrees)
  - Color changes (RGB vs grayscale)
  - Resizing / resolution changes (0.5x, 0.75x, 1.0x)

Uses:
  - SHViT model: shvit_s2
  - Baseline: deit_tiny

Checkpoints:
  results/deit_tiny_<DATASET>_frac1.0/checkpoint_99.pth (CIFAR)
  results/shvit_s2_<DATASET>_frac1.0/checkpoint_99.pth (CIFAR)
  For EUROSAT/MEDMNIST, uses checkpoint_29.pth by default (mirrors robustness script).
  
  
to run:
python analyze_geometric_invariance.py \
  --dataset CIFAR \
  --data-path dataset/ \
  --checkpoint-dir results \
  --device cuda
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from timm.models import create_model
from data.datasets import build_dataset
from engine import evaluate
import utils  # noqa: F401 - needed by your codebase
import model  # noqa: F401 - registers SHViT models

from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


# ------------------------------------------------------------------
# Geometric corruption wrapper
# ------------------------------------------------------------------


class GeometricCorruptionDataset(Dataset):
    """
    Wraps an existing dataset and applies geometric / color changes
    to the already-transformed image tensor (C, H, W).
    """

    def __init__(self, base_dataset, mode: str, param=None):
        """
        mode:
          - 'rotation': param = angle in degrees
          - 'color': param = 'clean' or 'grayscale'
          - 'resize': param = scale factor (e.g., 0.5)
        """
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
                # Convert to grayscale then replicate to 3 channels
                # img: (C, H, W)
                gray = img.mean(dim=0, keepdim=True)
                img = gray.repeat(3, 1, 1)
            # 'clean' = no change
            return img

        elif self.mode == "resize":
            scale = float(self.param) if self.param is not None else 1.0
            C, H, W = img.shape
            # Downscale to (H*scale, W*scale) then upsample back to (H, W)
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            small = Fnn.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            resized = Fnn.interpolate(
                small,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return resized

        else:
            # No change
            return img


# ------------------------------------------------------------------
# Dataset + model helpers (reuse your robustness setup)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------


def plot_two_model_curve(
    x_values: List[float],
    acc_baseline: List[float],
    acc_shvit: List[float],
    x_label: str,
    title: str,
    output_path: Path,
    x_tick_labels: Optional[List[str]] = None,
):
    """
    Plot baseline (deit_tiny) as dashed line, shvit_s2 as solid line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x_values,
        acc_baseline,
        "--o",
        linewidth=2,
        markersize=6,
        label="deit_tiny (baseline)",
    )
    ax.plot(
        x_values,
        acc_shvit,
        "-o",
        linewidth=2,
        markersize=6,
        label="shvit_s2",
    )

    if x_tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    all_accs = acc_baseline + acc_shvit
    if len(all_accs) > 0:
        y_min = min(all_accs) - 5
        y_max = max(all_accs) + 5
        ax.set_ylim(y_min, y_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


# ------------------------------------------------------------------
# Specific tests
# ------------------------------------------------------------------


def run_rotation_test(
    args, dataset_val, baseline_model, shvit_model, device
):
    """
    Rotation invariance:
      angles = [0, 15, 30, 45]
    """
    angles = [0, 15, 30, 45]
    acc_baseline = []
    acc_shvit = []

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

        acc_b = evaluate_on_dataset(
            rot_ds, baseline_model, device, args.batch_size, args.num_workers
        )
        acc_s = evaluate_on_dataset(
            rot_ds, shvit_model, device, args.batch_size, args.num_workers
        )

        print(f"  deit_tiny  Acc@1: {acc_b:.2f}%")
        print(f"  shvit_s2   Acc@1: {acc_s:.2f}%")

        acc_baseline.append(acc_b)
        acc_shvit.append(acc_s)

    return angles, acc_baseline, acc_shvit


def run_color_test(
    args, dataset_val, baseline_model, shvit_model, device
):
    """
    Color invariance:
      conditions = ["clean", "grayscale"]
    """
    conditions = ["clean", "grayscale"]
    x_vals = list(range(len(conditions)))
    acc_baseline = []
    acc_shvit = []

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

        acc_b = evaluate_on_dataset(
            color_ds, baseline_model, device, args.batch_size, args.num_workers
        )
        acc_s = evaluate_on_dataset(
            color_ds, shvit_model, device, args.batch_size, args.num_workers
        )

        print(f"  deit_tiny  Acc@1: {acc_b:.2f}%")
        print(f"  shvit_s2   Acc@1: {acc_s:.2f}%")

        acc_baseline.append(acc_b)
        acc_shvit.append(acc_s)

    return x_vals, conditions, acc_baseline, acc_shvit


def run_resize_test(
    args, dataset_val, baseline_model, shvit_model, device
):
    """
    Resize / resolution invariance:
      scales = [0.5, 0.75, 1.0]
    (Downscale to scale*H/W, then upsample back to H/W.)
    """
    scales = [0.25, 0.5, 0.75, 1.0]
    acc_baseline = []
    acc_shvit = []

    print("\n" + "=" * 80)
    print(f"Resize / scale invariance test on {args.dataset}")
    print("=" * 80)

    for scale in scales:
        print(f"\nScale factor: {scale}x")

        resize_ds = GeometricCorruptionDataset(
            base_dataset=dataset_val,
            mode="resize",
            param=scale,
        )

        acc_b = evaluate_on_dataset(
            resize_ds, baseline_model, device, args.batch_size, args.num_workers
        )
        acc_s = evaluate_on_dataset(
            resize_ds, shvit_model, device, args.batch_size, args.num_workers
        )

        print(f"  deit_tiny  Acc@1: {acc_b:.2f}%")
        print(f"  shvit_s2   Acc@1: {acc_s:.2f}%")

        acc_baseline.append(acc_b)
        acc_shvit.append(acc_s)

    return scales, acc_baseline, acc_shvit


# ------------------------------------------------------------------
# Argument parsing & main
# ------------------------------------------------------------------


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Geometric & Color Invariance Analysis", add_help=True
    )

    # Models – match what you trained
    parser.add_argument("--baseline-model", default="deit_tiny_patch16_224", type=str)
    parser.add_argument("--shvit-model", default="shvit_s2", type=str)

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

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    args.data_set = args.dataset  # some parts of your code may expect this
    args.data_path = args.data_path

    # 1) Build eval dataset and infer num classes
    dataset_val, nb_classes = build_eval_dataset_and_num_classes(args)

    # 2) Load baseline and shvit models from frac1.0 checkpoints
    baseline_model = load_model_for_dataset(
        args.baseline_model, nb_classes, args, device
    )
    shvit_model = load_model_for_dataset(
        args.shvit_model, nb_classes, args, device
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Rotation ----------------
    angles, acc_b_rot, acc_s_rot = run_rotation_test(
        args, dataset_val, baseline_model, shvit_model, device
    )
    plot_two_model_curve(
        x_values=angles,
        acc_baseline=acc_b_rot,
        acc_shvit=acc_s_rot,
        x_label="Rotation (degrees)",
        title=f"{args.dataset}: Rotation Invariance",
        output_path=out_dir / f"{args.dataset}_rotation.png",
        x_tick_labels=[str(a) for a in angles],
    )

    # ---------------- Color / grayscale ----------------
    x_vals, cond_labels, acc_b_color, acc_s_color = run_color_test(
        args, dataset_val, baseline_model, shvit_model, device
    )
    plot_two_model_curve(
        x_values=x_vals,
        acc_baseline=acc_b_color,
        acc_shvit=acc_s_color,
        x_label="Color condition",
        title=f"{args.dataset}: Color / Grayscale Invariance",
        output_path=out_dir / f"{args.dataset}_color.png",
        x_tick_labels=cond_labels,
    )

    # ---------------- Resize / scale ----------------
    scales, acc_b_resize, acc_s_resize = run_resize_test(
        args, dataset_val, baseline_model, shvit_model, device
    )
    plot_two_model_curve(
        x_values=scales,
        acc_baseline=acc_b_resize,
        acc_shvit=acc_s_resize,
        x_label="Resize scale (downscale factor)",
        title=f"{args.dataset}: Resize / Scale Invariance",
        output_path=out_dir / f"{args.dataset}_resize.png",
        x_tick_labels=[f"{s:.2f}x" for s in scales],
    )

    print("\n✓ Geometric & color invariance analysis complete!")


if __name__ == "__main__":
    main()