"""
Qualitative Saliency Maps for SHViT vs DeiT (no external deps)

This script:
  - Loads two models (e.g., shvit_s2 and deit_tiny_patch16_224) with checkpoints.
  - Uses the CIFAR / EuroSAT validation set.
  - Applies Gaussian noise corruption.
  - Finds:
      (1) an example where Model A (e.g. SHViT) is robust but Model B (e.g. DeiT) fails,
      (2) an example where Model B is robust but Model A fails.
  - Computes gradient-based saliency maps (d logit / d input pixels) for:
      - Clean image (both models)
      - Corrupted image (both models)
  - Saves 4-panel figures for each case.

This gives you qualitative "where does the model look?" comparisons for the thesis.

python analyze_gradcam_compare.py \
  --model-a shvit_s2 \
  --ckpt-a results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth \
  --model-b deit_tiny_patch16_224 \
  --ckpt-b results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth \
  --dataset CIFAR \
  --data-path dataset/ \
  --severity 3 \
  --max-search 300 \
  --output-dir analysis/saliency_cifar
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from timm.models import create_model

from data.datasets import build_dataset
import model  # noqa: F401  # register SHViT models via import side-effect


# ------------------------------------------------------------------
# Basic corruption: Gaussian noise in image space
# ------------------------------------------------------------------

def apply_gaussian_noise(img: torch.Tensor, severity: int) -> torch.Tensor:
    """
    Apply Gaussian noise to an image tensor in [0, 1].

    img: [3, H, W] in [0, 1]
    severity: int in {1,...,5}
    """
    std = 0.05 * severity
    noise = torch.randn_like(img) * std
    out = torch.clamp(img + noise, 0.0, 1.0)
    return out


# ------------------------------------------------------------------
# Dataset & normalization utilities
# ------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_val_loader(args):
    """
    Build the validation DataLoader using your existing build_dataset function.
    """
    eval_args = argparse.Namespace(**vars(args))
    eval_args.data_set = args.dataset
    eval_args.input_size = 224
    eval_args.color_jitter = 0.4
    eval_args.aa = 'rand-m9-mstd0.5-inc1'
    eval_args.train_interpolation = 'bicubic'
    eval_args.reprob = 0.25
    eval_args.remode = 'pixel'
    eval_args.recount = 1
    eval_args.resplit = False
    eval_args.ThreeAugment = False
    eval_args.finetune = ''

    # Number of classes
    if args.dataset == 'CIFAR':
        nb_classes = 100
    elif args.dataset == 'EUROSAT':
        nb_classes = 10
    else:
        nb_classes = 1000  # extend if needed

    dataset_val, _ = build_dataset(is_train=False, args=eval_args)
    loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, nb_classes


def unnormalize_for_vis(img_norm: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor [3, H, W] back to [H, W, 3] in [0, 1] for visualization.
    Assumes ImageNet mean/std normalization.
    """
    img = img_norm.clone().cpu()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = torch.clamp(img, 0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()
    return img_np


# ------------------------------------------------------------------
# Prediction and saliency
# ------------------------------------------------------------------

def predict(model: nn.Module, x: torch.Tensor, device: torch.device):
    """
    x: [1, 3, H, W] normalized.
    Returns (pred_label, probs) on CPU.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
    return pred.cpu(), probs.cpu()


def compute_saliency_map(
    model: nn.Module,
    img_norm: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute gradient-based saliency:
      saliency = max_c | d logit[target_class] / d img[c, h, w] |

    img_norm: [3, H, W] normalized tensor (requires no grad initially)
    Returns [H, W] saliency in [0, 1].
    """
    model.eval()
    img = img_norm.unsqueeze(0).to(device)
    img.requires_grad_(True)

    model.zero_grad()
    logits = model(img)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    # score for the target class
    score = logits[0, target_class]
    score.backward()

    grad = img.grad[0].detach().cpu()  # [3, H, W]
    saliency = grad.abs().max(dim=0)[0]  # [H, W]

    # Normalize to [0, 1]
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)
    return saliency.numpy()


def overlay_saliency_on_rgb(rgb: np.ndarray, saliency: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an RGB overlay of saliency on top of rgb image.

    rgb: [H, W, 3] in [0, 1]
    saliency: [H, W] in [0, 1]
    """
    cmap = plt.get_cmap("jet")
    saliency_color = cmap(saliency)[..., :3]  # [H, W, 3], in [0,1]
    overlay = alpha * saliency_color + (1 - alpha) * rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# ------------------------------------------------------------------
# Search for interesting examples
# ------------------------------------------------------------------

def find_interesting_examples(
    model_a, model_b, loader, device, severity, max_search=200
):
    """
    Search for indices where:
      - idx_robust_a: Model A correct on clean & corrupted, Model B correct on clean but wrong on corrupted
      - idx_robust_b: symmetric for Model B.

    Returns (idx_robust_a, idx_robust_b). Either may be None if not found.
    """
    idx_robust_a = None
    idx_robust_b = None

    dataset = loader.dataset

    for idx in range(len(dataset)):
        if idx >= max_search:
            break

        img_norm, label = dataset[idx]
        label = int(label)

        # Clean prediction
        pred_a_clean, _ = predict(model_a, img_norm.unsqueeze(0), device)
        pred_b_clean, _ = predict(model_b, img_norm.unsqueeze(0), device)

        # We only care about cases where both are correct on the clean image
        if not (pred_a_clean.item() == label and pred_b_clean.item() == label):
            continue

        # Clean image for vis: unnormalize to [0,1]
        clean_rgb = unnormalize_for_vis(img_norm)  # [H,W,3]

        # Apply corruption in [0,1] space
        clean_t = torch.from_numpy(clean_rgb).permute(2, 0, 1)  # [3,H,W], in [0,1]
        corr_t = apply_gaussian_noise(clean_t, severity)
        corr_t = torch.clamp(corr_t, 0.0, 1.0)
        corr_rgb = corr_t.permute(1, 2, 0).numpy()

        # Re-normalize corrupted image
        corr_norm = (corr_t - IMAGENET_MEAN) / IMAGENET_STD

        pred_a_corr, _ = predict(model_a, corr_norm.unsqueeze(0), device)
        pred_b_corr, _ = predict(model_b, corr_norm.unsqueeze(0), device)

        # Case 1: A robust, B fails
        if (
            idx_robust_a is None
            and pred_a_corr.item() == label
            and pred_b_corr.item() != label
        ):
            idx_robust_a = idx

        # Case 2: B robust, A fails
        if (
            idx_robust_b is None
            and pred_b_corr.item() == label
            and pred_a_corr.item() != label
        ):
            idx_robust_b = idx

        if idx_robust_a is not None and idx_robust_b is not None:
            break

    return idx_robust_a, idx_robust_b


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_four_panel(
    clean_rgb,
    sal_a_clean,
    sal_b_clean,
    corr_rgb,
    sal_a_corr,
    sal_b_corr,
    title_prefix: str,
    out_path: Path,
    model_a_name: str,
    model_b_name: str,
    corruption_desc: str,
):
    """
    4-panel plot:

      Row 1: clean image with saliency overlays (A left, B right)
      Row 2: corrupted image with saliency overlays (A left, B right)
    """

    overlay_a_clean = overlay_saliency_on_rgb(clean_rgb, sal_a_clean)
    overlay_b_clean = overlay_saliency_on_rgb(clean_rgb, sal_b_clean)

    overlay_a_corr = overlay_saliency_on_rgb(corr_rgb, sal_a_corr)
    overlay_b_corr = overlay_saliency_on_rgb(corr_rgb, sal_b_corr)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(overlay_a_clean)
    axes[0, 0].set_title(f"{model_a_name} – clean")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_b_clean)
    axes[0, 1].set_title(f"{model_b_name} – clean")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay_a_corr)
    axes[1, 0].set_title(f"{model_a_name} – {corruption_desc}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_b_corr)
    axes[1, 1].set_title(f"{model_b_name} – {corruption_desc}")
    axes[1, 1].axis("off")

    plt.suptitle(title_prefix, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ------------------------------------------------------------------
# CLI & main
# ------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("SHViT vs DeiT saliency visualization", add_help=True)

    parser.add_argument("--model-a", type=str, default="shvit_s2",
                        help="First model name (e.g. shvit_s2)")
    parser.add_argument("--ckpt-a", type=str, required=True,
                        help="Checkpoint for model A (path to .pth)")

    parser.add_argument("--model-b", type=str, default="deit_tiny_patch16_224",
                        help="Second model name (e.g. deit_tiny_patch16_224)")
    parser.add_argument("--ckpt-b", type=str, required=True,
                        help="Checkpoint for model B (path to .pth)")

    parser.add_argument("--dataset", type=str, default="CIFAR",
                        choices=["CIFAR", "EUROSAT"],
                        help="Dataset to visualize on")
    parser.add_argument("--data-path", type=str, default="dataset/",
                        help="Root path to dataset")

    parser.add_argument("--severity", type=int, default=3,
                        help="Gaussian noise severity (1–5)")
    parser.add_argument("--max-search", type=int, default=200,
                        help="Max number of validation samples to search for interesting cases")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output-dir", type=str, default="analysis/saliency_examples",
                        help="Directory to save output figures")

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build validation loader and get num classes
    val_loader, nb_classes = build_val_loader(args)

    # Create models and load checkpoints
    print(f"Creating model A: {args.model_a}")
    model_a = create_model(args.model_a, num_classes=nb_classes, pretrained=False)
    print(f"Loading checkpoint A from {args.ckpt_a}")
    ckpt_a = torch.load(args.ckpt_a, map_location="cpu")
    model_a.load_state_dict(ckpt_a["model"])
    model_a.to(device)

    print(f"Creating model B: {args.model_b}")
    model_b = create_model(args.model_b, num_classes=nb_classes, pretrained=False)
    print(f"Loading checkpoint B from {args.ckpt_b}")
    ckpt_b = torch.load(args.ckpt_b, map_location="cpu")
    model_b.load_state_dict(ckpt_b["model"])
    model_b.to(device)

    # Find interesting examples
    print("\nSearching for interesting examples...")
    idx_robust_a, idx_robust_b = find_interesting_examples(
        model_a, model_b, val_loader, device,
        severity=args.severity,
        max_search=args.max_search,
    )
    print(f"Index where {args.model_a} robust, {args.model_b} fails: {idx_robust_a}")
    print(f"Index where {args.model_b} robust, {args.model_a} fails: {idx_robust_b}")

    dataset = val_loader.dataset

    def process_index(idx, tag, robust_model_name, failing_model_name):
        if idx is None:
            print(f"No example found for pattern '{tag}', skipping.")
            return

        img_norm, label = dataset[idx]
        label = int(label)

        # Clean image (norm → RGB)
        clean_rgb = unnormalize_for_vis(img_norm)

        # Corrupted image
        clean_t = torch.from_numpy(clean_rgb).permute(2, 0, 1)  # [3,H,W]
        corr_t = apply_gaussian_noise(clean_t, args.severity)
        corr_t = torch.clamp(corr_t, 0.0, 1.0)
        corr_rgb = corr_t.permute(1, 2, 0).numpy()

        # Renormalize corrupted
        corr_norm = (corr_t - IMAGENET_MEAN) / IMAGENET_STD

        # Saliency maps
        sal_a_clean = compute_saliency_map(model_a, img_norm, label, device)
        sal_b_clean = compute_saliency_map(model_b, img_norm, label, device)

        sal_a_corr = compute_saliency_map(model_a, corr_norm, label, device)
        sal_b_corr = compute_saliency_map(model_b, corr_norm, label, device)

        title = (
            f"{tag} – {args.dataset}, Gaussian noise severity {args.severity}\n"
            f"{robust_model_name} robust, {failing_model_name} misclassifies"
        )
        out_file = output_dir / f"saliency_{tag}_{args.dataset}_idx{idx}.png"

        plot_four_panel(
            clean_rgb,
            sal_a_clean,
            sal_b_clean,
            corr_rgb,
            sal_a_corr,
            sal_b_corr,
            title_prefix=title,
            out_path=out_file,
            model_a_name=args.model_a,
            model_b_name=args.model_b,
            corruption_desc=f"noise s={args.severity}",
        )

    # Case 1: Model A robust, Model B fails
    process_index(
        idx_robust_a,
        tag=f"{args.model_a}_robust_{args.model_b}_fails",
        robust_model_name=args.model_a,
        failing_model_name=args.model_b,
    )

    # Case 2: Model B robust, Model A fails
    process_index(
        idx_robust_b,
        tag=f"{args.model_b}_robust_{args.model_a}_fails",
        robust_model_name=args.model_b,
        failing_model_name=args.model_a,
    )

    print("\n✓ Done. Check saved PNGs in:", output_dir)


if __name__ == "__main__":
    main()