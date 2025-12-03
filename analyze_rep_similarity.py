"""
Representation Analysis:
How similar are SHViT's learned features to those of standard ViTs?

This script:
  - Loads two trained models (e.g., shvit_s2 and vit_tiny_patch16_224),
  - Extracts penultimate-layer features on the validation set,
  - Computes CKA similarity between their representations,
  - Computes class-mean cosine similarity,
  - Saves metrics and (optionally) raw features.

Usage example (CIFAR-100):

python analyze_rep_similarity.py \
  --model-a shvit_s2 \
  --ckpt-a results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth \
  --model-b deit_tiny_patch16_224 \
  --ckpt-b results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth \
  --dataset CIFAR \
  --data-path dataset/ \
  --output-dir outputs/rep_similarity_cifar
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timm.models import create_model

from data.datasets import build_dataset
import utils
import model  # noqa: F401  # register SHViT models via side-effects


# -------------------- Feature Hook -------------------- #

class PenultimateFeatureHook:
    """
    Attach to the classifier head and capture its INPUT (penultimate features).
    Works for most timm models & SHViT variants as long as they have a 'head',
    'fc', or 'classifier' module.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.features = []
        self.handle = None
        self._register_hook()

    def _classifier_module(self) -> nn.Module:
        # Try common classifier names
        for name in ["head", "fc", "classifier"]:
            if hasattr(self.model, name):
                mod = getattr(self.model, name)
                if isinstance(mod, nn.Module):
                    return mod

        # Fallback: last module in the model's children
        modules = list(self.model.children())
        if len(modules) == 0:
            raise RuntimeError("Model has no children modules to hook.")
        return modules[-1]

    def _hook_fn(self, module, inputs, output):
        # We want the input to the classifier head
        x = inputs[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        # [B, ...]
        x = x.detach().cpu()
        self.features.append(x)

    def _register_hook(self):
        cls_mod = self._classifier_module()
        self.handle = cls_mod.register_forward_hook(self._hook_fn)

    def clear(self):
        self.features = []

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_features(self) -> torch.Tensor:
        if not self.features:
            return torch.empty(0)
        return torch.cat(self.features, dim=0)


# -------------------- CKA Utilities -------------------- #

def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    """
    Center a Gram matrix: H K H where H = I - 1/n 1.
    """
    n = gram.size(0)
    unit = torch.ones((n, n), device=gram.device)
    identity = torch.eye(n, device=gram.device)
    H = identity - unit / n
    return H @ gram @ H


def _gram_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Linear kernel Gram matrix: K = X X^T
    X: [n, d]
    """
    return x @ x.t()


def compute_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute linear CKA between two representation matrices.
    x: [n, d1]
    y: [n, d2]
    """
    assert x.size(0) == y.size(0), "Number of samples must match for CKA."

    # Move to a common device (CPU is fine)
    device = torch.device("cpu")
    x = x.to(device)
    y = y.to(device)

    # Centered Gram matrices
    Kx = _center_gram(_gram_linear(x))
    Ky = _center_gram(_gram_linear(y))

    # HSIC(X, Y) = trace(Kx * Ky)
    hsic_xy = (Kx * Ky).sum()

    # HSIC(X, X) and HSIC(Y, Y)
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()

    # Numerical safety
    denom = torch.sqrt(hsic_xx * hsic_yy + 1e-12)
    cka = hsic_xy / denom
    return cka.item()


# -------------------- Feature Extraction -------------------- #

@torch.no_grad()
def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model on data_loader, capture penultimate features and labels.
    Returns:
      features: [N, D]
      labels:   [N]
    """
    model.eval()
    hook = PenultimateFeatureHook(model)

    all_labels = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        _ = model(images)  # forward pass; hook captures features
        all_labels.append(targets)

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    feats = hook.get_features()  # [N, ...]
    hook.remove()
    labels = torch.cat(all_labels, dim=0)

    # Flatten features per sample
    feats = feats.view(feats.size(0), -1)

    return feats, labels


def build_val_loader(args: argparse.Namespace) -> Tuple[DataLoader, int]:
    """
    Build validation DataLoader and return (dataloader, num_classes)
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
    elif args.dataset == 'MEDMNIST':
        try:
            import medmnist
            from medmnist import INFO
            info = INFO[args.medmnist_dataset]
            nb_classes = len(info['label'])
        except Exception:
            nb_classes = 9
    else:
        nb_classes = 1000

    dataset_val, _ = build_dataset(is_train=False, args=eval_args)

    data_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return data_loader, nb_classes


# -------------------- Class-Mean Similarity -------------------- #

def compute_class_mean_cosine(
    feats_a: torch.Tensor,
    feats_b: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Compute cosine similarity between per-class mean representations.

    Returns:
      {
        'mean_cosine': float,
        'per_class_cosine': {class_idx: value, ...}
      }
    """
    device = torch.device("cpu")

    feats_a = feats_a.to(device)
    feats_b = feats_b.to(device)
    labels = labels.to(device)

    per_class_cos = {}
    cos = nn.CosineSimilarity(dim=-1)

    for c in range(num_classes):
        idx = (labels == c)
        if idx.sum() == 0:
            continue

        mean_a = feats_a[idx].mean(dim=0, keepdim=True)  # [1, Da]
        mean_b = feats_b[idx].mean(dim=0, keepdim=True)  # [1, Db]

        # Project to same dim via PCA or just compute cosine after linear proj?
        # For simplicity, we'll compute cosine in a shared latent via SVD on concatenated
        # but here we can instead normalize and compute cosine using a small trick:
        # We map both to same dimension using a random projection (fixed seed) to avoid bias.
        # To keep it simple for you, we just pad/truncate to min dim.

        d_a = mean_a.size(-1)
        d_b = mean_b.size(-1)
        d = min(d_a, d_b)

        mean_a_ = mean_a[..., :d]
        mean_b_ = mean_b[..., :d]

        val = cos(mean_a_, mean_b_).item()
        per_class_cos[c] = val

    mean_cos = float(np.mean(list(per_class_cos.values()))) if per_class_cos else 0.0

    return {
        "mean_cosine": mean_cos,
        "per_class_cosine": per_class_cos,
    }


# -------------------- Main Logic -------------------- #

def run_representation_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)

    # Build val loader
    val_loader, nb_classes = build_val_loader(args)

    # Create both models
    print(f"Creating model A: {args.model_a}")
    model_a = create_model(
        args.model_a,
        num_classes=nb_classes,
        pretrained=False,
    )

    print(f"Creating model B: {args.model_b}")
    model_b = create_model(
        args.model_b,
        num_classes=nb_classes,
        pretrained=False,
    )

    # Load checkpoints if provided
    if args.ckpt_a:
        print(f"Loading checkpoint A from: {args.ckpt_a}")
        ckpt_a = torch.load(args.ckpt_a, map_location='cpu')
        model_a.load_state_dict(ckpt_a['model'])

    if args.ckpt_b:
        print(f"Loading checkpoint B from: {args.ckpt_b}")
        ckpt_b = torch.load(args.ckpt_b, map_location='cpu')
        model_b.load_state_dict(ckpt_b['model'])

    model_a.to(device)
    model_b.to(device)

    # Extract features
    print("\nExtracting features for Model A...")
    feats_a, labels = extract_features(model_a, val_loader, device, max_batches=args.max_batches)
    print(f"Model A features shape: {tuple(feats_a.shape)}")

    print("\nExtracting features for Model B...")
    feats_b, labels_b = extract_features(model_b, val_loader, device, max_batches=args.max_batches)
    print(f"Model B features shape: {tuple(feats_b.shape)}")

    # Sanity check
    assert feats_a.size(0) == feats_b.size(0) == labels.size(0) == labels_b.size(0), \
        "Mismatch in number of samples between models/labels."

    # Compute CKA
    print("\nComputing CKA similarity between representations...")
    cka_score = compute_cka(feats_a, feats_b)
    print(f"CKA similarity: {cka_score:.4f}")

    # Class-mean cosine similarity
    print("\nComputing class-mean cosine similarity...")
    class_mean_stats = compute_class_mean_cosine(
        feats_a, feats_b, labels, num_classes=nb_classes
    )
    print(f"Mean class-wise cosine similarity: {class_mean_stats['mean_cosine']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "num_classes": nb_classes,
        "num_samples": feats_a.size(0),
        "cka": cka_score,
        "class_mean_cosine": class_mean_stats["mean_cosine"],
        "per_class_cosine": class_mean_stats["per_class_cosine"],
    }

    out_json = output_dir / f"rep_similarity_{args.model_a}_vs_{args.model_b}_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics to: {out_json}")

    if args.save_features:
        out_npz = output_dir / f"rep_features_{args.model_a}_vs_{args.model_b}_{args.dataset}.npz"
        np.savez_compressed(
            out_npz,
            feats_a=feats_a.numpy(),
            feats_b=feats_b.numpy(),
            labels=labels.numpy(),
        )
        print(f"Saved raw features to: {out_npz}")

    return result


def get_args_parser():
    parser = argparse.ArgumentParser("Representation similarity analysis", add_help=True)

    # Models & checkpoints
    parser.add_argument("--model-a", type=str, default="shvit_s2",
                        help="First model name (e.g. shvit_s2)")
    parser.add_argument("--ckpt-a", type=str, default="",
                        help="Checkpoint for model A (path to .pth)")

    parser.add_argument("--model-b", type=str, default="vit_tiny_patch16_224",
                        help="Second model name (e.g. vit_tiny_patch16_224 or deit_tiny_patch16_224)")
    parser.add_argument("--ckpt-b", type=str, default="",
                        help="Checkpoint for model B (path to .pth)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR",
                        choices=["CIFAR", "EUROSAT", "MEDMNIST"],
                        help="Dataset type")
    parser.add_argument("--medmnist-dataset", type=str, default="pathmnist",
                        help="MedMNIST subset name (if using MEDMNIST)")
    parser.add_argument("--data-path", type=str, default="dataset/",
                        help="Root path to dataset (same as main.py)")

    # Loader / device
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # Analysis options
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches for quick debugging (None = all)")
    parser.add_argument("--save-features", action="store_true",
                        help="If set, save raw features as .npz")

    # Output
    parser.add_argument("--output-dir", type=str, default="analysis/rep_similarity",
                        help="Directory to store outputs")

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Set data_path in a way compatible with build_dataset
    # (Your build_dataset likely uses args.data_path and args.data_set)
    utils.init_distributed_mode(args)  # harmless for single GPU; uses env vars
    # We don't actually do DDP here, but init for consistency.

    print("=" * 80)
    print("Representation Analysis: SHViT vs ViT/DeiT")
    print("=" * 80)
    print(f"Model A: {args.model_a} (ckpt: {args.ckpt_a})")
    print(f"Model B: {args.model_b} (ckpt: {args.ckpt_b})")
    print(f"Dataset: {args.dataset} (data_path: {args.data_path})")
    print("=" * 80)

    run_representation_analysis(args)

    print("\n✓ Representation analysis complete!")


if __name__ == "__main__":
    main()