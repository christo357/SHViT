"""
Domain Shift with Feature Transfer + Fine-Tuning (Single Checkpoint per Model)

For each model:

  1. Load checkpoint trained on SOURCE_DATASET with frac1.0.
  2. Create a new model with num_classes = TARGET_NUM_CLASSES.
  3. Load all matching weights from the source checkpoint
     (backbone transfer; classifier head randomly initialized).
  4. Fine-tune on TARGET_DATASET (train split).
  5. Evaluate on TARGET_DATASET (val split) and record Acc@1.

Produces:
  - JSON: accuracies per model
  - Bar plot: Top-1 accuracy on target after fine-tuning, one bar per model.

Example:
  python analyze_domain_shift.py \
    --source-dataset CIFAR \
    --target-dataset EUROSAT \
    --models shvit_s2 deit_tiny_patch16_224 mobilenetv2_100 \
    --data-path dataset \
    --checkpoint-dir results \
    --ft-epochs 10 \
    --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from timm.models import create_model
from data.datasets import build_dataset
from engine import evaluate
import utils  # noqa: F401
import model  # noqa: F401  # registers SHViT models


# ---------------------------------------------------------------------
# Dataset / class helpers
# ---------------------------------------------------------------------


def get_nb_classes(dataset: str, medmnist_subset: str) -> int:
    dataset = dataset.upper()
    if dataset == "CIFAR":
        return 100
    elif dataset == "EUROSAT":
        return 10
    elif dataset == "MEDMNIST":
        try:
            import medmnist
            from medmnist import INFO

            info = INFO[medmnist_subset]
            return len(info["label"])
        except Exception:
            return 9
    else:
        return 1000


def build_train_val_for_dataset(
    dataset_name: str,
    medmnist_subset: str,
    data_path: str,
):
    """
    Build (train, val) datasets for a given dataset.
    """
    base_args = argparse.Namespace()
    base_args.data_set = dataset_name
    base_args.dataset = dataset_name
    base_args.data_path = data_path
    base_args.input_size = 224
    base_args.color_jitter = 0.4
    base_args.aa = "rand-m9-mstd0.5-inc1"
    base_args.train_interpolation = "bicubic"
    base_args.reprob = 0.25
    base_args.remode = "pixel"
    base_args.recount = 1
    base_args.resplit = False
    base_args.ThreeAugment = False
    base_args.finetune = ""

    train_dataset, _ = build_dataset(is_train=True, args=base_args)
    val_dataset, _ = build_dataset(is_train=False, args=base_args)
    nb_classes = get_nb_classes(dataset_name, medmnist_subset)
    return train_dataset, val_dataset, nb_classes


# ---------------------------------------------------------------------
# Checkpoint loading with flexible shape matching
# ---------------------------------------------------------------------


def load_state_dict_flex(net: nn.Module, checkpoint: dict, model_key: str = "model"):
    """
    Load as many matching keys as possible (by name and shape).
    Mismatched keys (e.g., classifier head) are skipped.
    """
    if model_key in checkpoint:
        state_dict = checkpoint[model_key]
    else:
        state_dict = checkpoint

    model_dict = net.state_dict()
    loaded, skipped = {}, []

    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            loaded[k] = v
        else:
            skipped.append(k)

    model_dict.update(loaded)
    net.load_state_dict(model_dict)

    print(f"    Loaded {len(loaded)} / {len(state_dict)} keys.")
    if skipped:
        print("    Skipped keys (name or shape mismatch), first few:")
        for name in skipped[:10]:
            print(f"      - {name}")


def init_model_for_transfer(
    model_name: str,
    source_dataset: str,
    nb_classes_target: int,
    checkpoint_dir: str,
    device: torch.device,
) -> nn.Module:
    """
    Create a model for the TARGET task (head size = target classes),
    then load as many weights as possible from SOURCE checkpoint
    trained with frac1.0.
    """
    src = source_dataset.upper()
    ckpt_root = Path(checkpoint_dir) / f"{model_name}_{src}_frac1.0"

    if src == "CIFAR":
        ckpt_path = ckpt_root / "checkpoint_99.pth"
    else:
        ckpt_path = ckpt_root / "checkpoint_29.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading source checkpoint: {ckpt_path}")

    net = create_model(
        model_name,
        num_classes=nb_classes_target,  # classifier head sized for TARGET task
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    load_state_dict_flex(net, checkpoint, model_key="model")

    net.to(device)
    return net


# ---------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n = 0

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = samples.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(1, n)


def eval_acc1(
    model: nn.Module,
    dataset_val,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    model.eval()
    loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    stats = evaluate(loader, model, device)
    return float(stats["acc1"])


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_bar_results(
    source_dataset: str,
    target_dataset: str,
    acc_by_model: Dict[str, float],
    output_path: Path,
):
    models = list(acc_by_model.keys())
    accs = [acc_by_model[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(models))
    ax.bar(x, accs)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")

    ax.set_ylabel("Top-1 Accuracy on target after fine-tune (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        f"Domain Shift (fine-tuned): {source_dataset} → {target_dataset}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved bar plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_domain_shift_finetune_simple(args):
    device = torch.device(args.device)

    src = args.source_dataset.upper()
    tgt = args.target_dataset.upper()

    print("=" * 80)
    print("Domain Shift with Feature Transfer + Fine-Tuning (Single Checkpoint)")
    print("=" * 80)
    print(f"Source dataset (pretraining): {src}")
    print(f"Target dataset (fine-tune & eval): {tgt}")
    print(f"Models: {args.models}")
    print(f"Fine-tune epochs: {args.ft_epochs}")
    print("=" * 80)

    # Build target train/val
    train_tgt, val_tgt, nb_classes_tgt = build_train_val_for_dataset(
        tgt, args.medmnist_dataset, args.data_path
    )

    train_loader = DataLoader(
        train_tgt,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    acc_by_model: Dict[str, float] = {}
    results = {
        "source_dataset": src,
        "target_dataset": tgt,
        "models": args.models,
        "ft_epochs": args.ft_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "acc1_after_ft": {},
    }

    for model_name in args.models:
        print(f"\n==================== Model: {model_name} ====================")

        try:
            net = init_model_for_transfer(
                model_name=model_name,
                source_dataset=src,
                nb_classes_target=nb_classes_tgt,
                checkpoint_dir=args.checkpoint_dir,
                device=device,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            acc_by_model[model_name] = float("nan")
            results["acc1_after_ft"][model_name] = float("nan")
            continue

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_acc = 0.0
        for epoch in range(args.ft_epochs):
            train_loss = train_one_epoch(net, criterion, optimizer, train_loader, device)
            val_acc = eval_acc1(
                net,
                val_tgt,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            best_acc = max(best_acc, val_acc)
            print(
                f"  [Epoch {epoch+1:02d}/{args.ft_epochs}] "
                f"train_loss={train_loss:.4f}, val_acc={val_acc:.2f}%, best={best_acc:.2f}%"
            )

        acc_by_model[model_name] = best_acc
        results["acc1_after_ft"][model_name] = best_acc
        print(f"  >>> Best Acc@1 on {tgt} after fine-tune: {best_acc:.2f}%")

        del net
        torch.cuda.empty_cache()

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"domain_shift_finetune_{src}_to_{tgt}_singlefrac.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to {json_path}")

    # Plot bar chart
    plot_path = out_dir / f"domain_shift_finetune_{src}_to_{tgt}_singlefrac.png"
    plot_bar_results(src, tgt, acc_by_model, plot_path)


# ---------------------------------------------------------------------
# Argparser
# ---------------------------------------------------------------------


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Domain Shift Fine-Tuning (Single Checkpoint per Model)", add_help=True
    )

    # Datasets
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="CIFAR",
        choices=["CIFAR", "EUROSAT", "MEDMNIST"],
        help="Dataset used for pretraining (where checkpoints come from).",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default="EUROSAT",
        choices=["CIFAR", "EUROSAT", "MEDMNIST"],
        help="Dataset used for fine-tuning & evaluation.",
    )
    parser.add_argument(
        "--medmnist-dataset",
        type=str,
        default="pathmnist",
        help="MedMNIST subset name (used if source/target is MEDMNIST).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset",
        help="Root directory for data.",
    )

    # Models
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["shvit_s2", "deit_tiny_patch16_224"],
        help="List of model names to evaluate.",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="learning_curve_results",
        help="Directory containing <model>_<SOURCE>_frac1.0 checkpoints.",
    )

    # Fine-tuning hyperparameters
    parser.add_argument("--ft-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/domain_shift_finetune",
        help="Where to save JSON + plots.",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    run_domain_shift_finetune_simple(args)


if __name__ == "__main__":
    main()