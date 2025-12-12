import argparse
import json
from pathlib import Path
from typing import Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from timm.models import create_model
from data.datasets import build_dataset
from engine import evaluate
import utils  # noqa: F401
import model  # noqa: F401


# ---------------------------------------------------------------------------
# Corruption dataset
# ---------------------------------------------------------------------------

class CorruptionDataset(Dataset):
    """Apply corruptions to a dataset"""

    def __init__(self, base_dataset, corruption_type, severity):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img_corrupted = self.apply_corruption(img, self.corruption_type, self.severity)
        return img_corrupted, label

    def apply_corruption(self, img, corruption_type, severity):
        """Apply corruption to image tensor (C, H, W) in [0, 1]"""

        if corruption_type == "gaussian_noise":
            noise = torch.randn_like(img) * (0.05 * severity)
            return torch.clamp(img + noise, 0, 1)

        elif corruption_type == "shot_noise":
            noisy = torch.poisson(img * (100 / severity)) / (100 / severity)
            return torch.clamp(noisy, 0, 1)

        elif corruption_type == "impulse_noise":
            mask = torch.rand_like(img)
            salt_pepper = torch.where(
                mask < 0.01 * severity,
                torch.ones_like(img),
                torch.where(
                    mask > 1 - 0.01 * severity,
                    torch.zeros_like(img),
                    img,
                ),
            )
            return salt_pepper

        elif corruption_type == "gaussian_blur":
            kernel_size = 2 * severity + 1
            padding = severity
            blurred = img.clone()
            for c in range(img.shape[0]):
                blurred[c] = torch.nn.functional.avg_pool2d(
                    img[c].unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ).squeeze()
            return blurred

        elif corruption_type == "brightness":
            factor = 0.1 + (0.2 * severity)
            return torch.clamp(img + factor, 0, 1)

        elif corruption_type == "contrast":
            factor = 0.4 + (0.2 * severity)
            mean = img.mean(dim=[1, 2], keepdim=True)
            return torch.clamp((img - mean) * factor + mean, 0, 1)

        elif corruption_type == "fog":
            fog_intensity = 0.1 * severity
            fog = torch.ones_like(img) * 0.7
            return torch.clamp(
                img * (1 - fog_intensity) + fog * fog_intensity, 0, 1
            )

        elif corruption_type == "jpeg_compression":
            quantize_factor = severity * 8
            quantized = torch.round(img * quantize_factor) / quantize_factor
            return torch.clamp(quantized, 0, 1)

        elif corruption_type == "pixelate":
            scale_factor = max(1, 6 - severity)
            C, H, W = img.shape
            small_h, small_w = H // scale_factor, W // scale_factor
            small = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(small_h, small_w),
                mode="nearest",
            )
            pixelated = torch.nn.functional.interpolate(
                small,
                size=(H, W),
                mode="nearest",
            ).squeeze(0)
            return pixelated

        else:
            return img


# ---------------------------------------------------------------------------
# Dataset + model helpers
# ---------------------------------------------------------------------------

def build_eval_dataset_and_num_classes(args):
    """
    Build validation dataset and infer number of classes.
    Mirrors your geometric / robustness scripts.
    """
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

    dataset_val, _ = build_dataset(is_train=False, args=eval_args)

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


def load_model_for_fraction(
    model_name: str,
    nb_classes: int,
    args,
    device: torch.device,
    fraction: float,
):
    """
    Load checkpoint for a given model and training fraction.
    Assumes directories like:
      <checkpoint_dir>/<model>_<DATASET>_frac<FRACTION>/checkpoint_*.pth
    """
    checkpoint_dir = (
        Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}_frac{fraction}"
    )

    if args.dataset == "CIFAR":
        checkpoint_path = checkpoint_dir / "checkpoint_99.pth"
    else:  # EUROSAT / MEDMNIST
        checkpoint_path = checkpoint_dir / "checkpoint_29.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"  Loading {model_name} checkpoint from {checkpoint_path}")

    net = create_model(
        model_name,
        num_classes=nb_classes,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(checkpoint["model"])
    net.to(device)
    net.eval()
    return net


def evaluate_on_dataset(
    dataset: Dataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
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


# ---------------------------------------------------------------------------
# Robustness evaluation for ONE model (severity 3 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_robustness_for_model(
    model_name: str,
    args,
    dataset_val,
    nb_classes: int,
    device: torch.device,
    severity_to_use: int = 3,
):
    """
    Evaluate clean + corrupted accuracy (single severity) for one model
    across:
      - training fractions
      - corruption types
    """
    fractions = args.fractions
    corruptions = args.corruptions
    severities = [severity_to_use]  # only severity 3

    results = {
        "model": model_name,
        "dataset": args.dataset,
        "fractions": fractions,
        "corruptions": corruptions,
        "severities": severities,  # [3]
        "clean": {},               # clean[frac] = acc1
        "corrupt": {},             # corrupt[corruption][severity][frac] = acc1
    }

    for corruption in corruptions:
        results["corrupt"][corruption] = {severity_to_use: {}}

    print("=" * 80)
    print(f"Robustness evaluation for model: {model_name} (severity {severity_to_use})")
    print("=" * 80)

    for frac in fractions:
        print(f"\n--- Fraction {frac * 100:.1f}% ---")
        try:
            net = load_model_for_fraction(
                model_name=model_name,
                nb_classes=nb_classes,
                args=args,
                device=device,
                fraction=frac,
            )
        except FileNotFoundError as e:
            print(f"  WARNING: {e}. Skipping this fraction.")
            continue

        # Clean
        print("  Evaluating on clean data...")
        acc_clean = evaluate_on_dataset(
            dataset_val, net, device, args.batch_size, args.num_workers
        )
        results["clean"][frac] = acc_clean
        print(f"    Clean Acc@1: {acc_clean:.2f}%")

        # Corruptions at severity 3 only
        for corruption in corruptions:
            print(f"  Corruption: {corruption}")
            sev = severity_to_use
            print(f"    Severity {sev}...", end=" ")
            corr_ds = CorruptionDataset(dataset_val, corruption, sev)
            acc_corr = evaluate_on_dataset(
                corr_ds, net, device, args.batch_size, args.num_workers
            )
            results["corrupt"][corruption][sev][frac] = acc_corr
            print(f"Acc@1: {acc_corr:.2f}%")

        del net
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Plotting (severity 3 only, multi-model)
# ---------------------------------------------------------------------------

def plot_multi_model_robustness(
    all_results: Dict[str, Dict],
    args,
    severity_to_plot: int = 3,
):
    """
    all_results: dict mapping model_name -> results_dict

    Plots:
      1) Clean accuracy vs training fraction (all models)
      2) Single robustness plot:
           - severity = severity_to_plot (3)
           - X-axis: training fraction (%)
           - Different color per corruption
           - Different line style per model
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the first model's metadata as reference
    first_model_name = next(iter(all_results.keys()))
    ref = all_results[first_model_name]

    fractions = ref["fractions"]
    corruptions = ref["corruptions"]
    severities = ref["severities"]
    frac_percent = [f * 100 for f in fractions]
    dataset_name = ref["dataset"]

    model_names = list(all_results.keys())

    # Line styles to differentiate models
    line_styles = ["-", "--", "-.", ":"]
    if len(model_names) > len(line_styles):
        print(
            f"WARNING: more models ({len(model_names)}) than available line styles "
            f"({len(line_styles)}). Styles will be reused."
        )

    # ------------------------------------------------------------------
    # 1) Clean accuracy vs training fraction (all models)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, mname in enumerate(model_names):
        res = all_results[mname]
        acc_clean = [res["clean"].get(f, np.nan) for f in fractions]

        ax.plot(
            frac_percent,
            acc_clean,
            line_styles[idx % len(line_styles)] + "o",
            linewidth=2,
            markersize=6,
            label=mname,
        )

    ax.set_xlabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Clean Accuracy vs Training Data Size\n{dataset_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    plt.tight_layout()
    clean_path = out_dir / f"{dataset_name}_clean_multi_model.png"
    plt.savefig(clean_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {clean_path}")

    # ------------------------------------------------------------------
    # 2) Combined robustness plot – severity 3 only
    # ------------------------------------------------------------------
    if severity_to_plot not in severities:
        print(
            f"WARNING: severity {severity_to_plot} not in results; "
            f"available severities = {severities}"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.get_cmap("tab10", len(corruptions))

    # For legend: we will build
    #  - one legend for corruption colors
    #  - one legend for model line styles
    corruption_handles = []
    corruption_labels = []

    for ci, corruption in enumerate(corruptions):
        color = cmap(ci)
        label_corr = corruption.replace("_", " ").title()

        # To show corruption in legend, we only need one dummy line per corruption
        corruption_handles.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=2)
        )
        corruption_labels.append(label_corr)

        for mi, mname in enumerate(model_names):
            res = all_results[mname]
            acc = [
                res["corrupt"][corruption][severity_to_plot].get(f, np.nan)
                for f in fractions
            ]

            ax.plot(
                frac_percent,
                acc,
                line_styles[mi % len(line_styles)] + "o",
                linewidth=2,
                markersize=5,
                color=color,
            )

    ax.set_xlabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Robustness vs Training Data Size (Severity {severity_to_plot})\n{dataset_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend 1: corruption (color)
    legend1 = ax.legend(
        corruption_handles,
        corruption_labels,
        title="Corruption",
        fontsize=9,
        loc="upper right",
    )
    ax.add_artist(legend1)

    # Legend 2: models (line style)
    model_style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2,
            label=mname,
        )
        for i, mname in enumerate(model_names)
    ]
    ax.legend(
        handles=model_style_handles,
        labels=model_names,
        title="Model",
        fontsize=9,
        loc="lower left",
    )

    plt.tight_layout()
    out_path = out_dir / f"{dataset_name}_robustness_sev{severity_to_plot}_multi_model.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def save_results_json(results: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {out_path}")


# ---------------------------------------------------------------------------
# Arg parser & main
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Multi-model Robustness vs Training Data Analysis (severity 3 only)",
        add_help=True,
    )

    # Models: list instead of single baseline + shvit only
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["deit_tiny_patch16_224", "shvit_s2"],
        help="List of model names to compare.",
    )

    # Dataset
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
        help="Root directory for data.",
    )

    # Training fractions
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.325, 0.55, 0.775, 1.0],
        help="Training data fractions to evaluate (0-1).",
    )

    # Corruptions (severity 3 only)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        type=str,
        default=["gaussian_noise", "gaussian_blur", "brightness", "contrast"],
        help="Corruption types to test (severity 3 only).",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        default="learning_curve_results",
        type=str,
        help="Where <model>_<DATASET>_fracX checkpoints are stored.",
    )

    # Eval
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    # Output
    parser.add_argument(
        "--output-dir",
        default="outputs/robustness_multi_sev1",
        type=str,
        help="Directory for plots + JSON.",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    severity_to_use = 1  # only severity 3

    device = torch.device(args.device)
    args.data_set = args.dataset
    args.data_path = args.data_path

    # Dataset + classes
    dataset_val, nb_classes = build_eval_dataset_and_num_classes(args)

    # Evaluate all models
    all_results: Dict[str, Dict] = {}
    out_dir = Path(args.output_dir)

    for model_name in args.models:
        res = evaluate_robustness_for_model(
            model_name=model_name,
            args=args,
            dataset_val=dataset_val,
            nb_classes=nb_classes,
            device=device,
            severity_to_use=severity_to_use,
        )
        all_results[model_name] = res
        save_results_json(
            res,
            out_dir / f"robustness_sev{severity_to_use}_{model_name}_{args.dataset}.json",
        )

    # Plots: clean + combined robustness (severity 3)
    plot_multi_model_robustness(
        all_results,
        args,
        severity_to_plot=severity_to_use,
    )

    print("\n✓ Multi-model robustness analysis (severity 3 only) complete!")


if __name__ == "__main__":
    main()
    
