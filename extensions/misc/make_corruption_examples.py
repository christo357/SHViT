import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from data.datasets import build_dataset
import model  # noqa: F401  # to register SHViT etc.


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
        """
        Apply corruption to image tensor (C, H, W).
        Assumes img is roughly in [0,1]; we always clamp back to [0,1].
        """
        img = img.clone()

        if corruption_type == "gaussian_noise":
            noise = torch.randn_like(img) * (0.05 * severity)
            return torch.clamp(img + noise, 0, 1)

        elif corruption_type == "gaussian_blur":
            # simple box blur as in your robustness script
            kernel_size = 2 * severity + 1
            padding = severity
            blurred = img.clone()
            for c in range(img.shape[0]):
                blurred[c] = F.avg_pool2d(
                    img[c].unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ).squeeze()
            return blurred

        elif corruption_type == "motion_blur":
            # horizontal motion blur with length depending on severity
            k = 2 * severity + 1  # 3, 5, 7, ...
            C, H, W = img.shape
            kernel = torch.zeros(C, 1, k, k, device=img.device)
            kernel[:, 0, k // 2, :] = 1.0 / k  # horizontal line
            img_b = img.unsqueeze(0)  # (1, C, H, W)
            blurred = F.conv2d(img_b, kernel, padding=k // 2, groups=C)[0]
            return torch.clamp(blurred, 0, 1)

        elif corruption_type == "brightness":
            factor = 0.1 + (0.2 * severity)
            return torch.clamp(img + factor, 0, 1)

        elif corruption_type == "contrast":
            factor = 0.4 + (0.2 * severity)
            mean = img.mean(dim=[1, 2], keepdim=True)
            return torch.clamp((img - mean) * factor + mean, 0, 1)

        else:
            return img



def build_eval_dataset(args):
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
    return dataset_val


def _tensor_to_image(img: torch.Tensor):
    """
    Convert CxHxW tensor (possibly normalized) to HxWxC numpy image in [0,1]
    for visualization.
    """
    img = img.detach().cpu().float()
    img = img.permute(1, 2, 0)  # C,H,W -> H,W,C
    img_min = img.min()
    img = img - img_min
    img_max = img.max()
    if img_max > 0:
        img = img / img_max
    return img.numpy()


def make_corruption_panel(args):
    dataset_val = build_eval_dataset(args)

    # pick a sample index (can expose as arg)
    idx = args.sample_index
    base_img, _ = dataset_val[idx]  # C,H,W

    severity = args.severity

    # Create corrupted versions
    gaussian_noise = CorruptionDataset(dataset_val, "gaussian_noise", severity).apply_corruption(base_img, "gaussian_noise", severity)
    motion_blur   = CorruptionDataset(dataset_val, "motion_blur", severity).apply_corruption(base_img, "motion_blur", severity)
    brightness    = CorruptionDataset(dataset_val, "brightness", severity).apply_corruption(base_img, "brightness", severity)
    contrast      = CorruptionDataset(dataset_val, "contrast", severity).apply_corruption(base_img, "contrast", severity)

    imgs = [
        base_img,
        gaussian_noise,
        motion_blur,
        brightness,
        contrast,
    ]
    titles = [
        "Original",
        "Gaussian Noise",
        "Motion Blur",
        "Brightness",
        "Contrast",
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_corruption_examples_sev{severity}.png"

    fig, axes = plt.subplots(1, len(imgs), figsize=(12, 3))
    if len(imgs) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(_tensor_to_image(img))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Corruption Examples (Severity {severity}) - {args.dataset}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved panel to {out_path}")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Create corruption example panel (Gaussian noise, motion blur, brightness, contrast)",
        add_help=True,
    )
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
    parser.add_argument(
        "--output-dir",
        default="outputs/robustness",
        type=str,
        help="Where to save the corruption panel image.",
    )
    parser.add_argument(
        "--sample-index",
        default=10,
        type=int,
        help="Index of the validation sample to visualize.",
    )
    parser.add_argument(
        "--severity",
        default=3,
        type=int,
        help="Corruption severity to visualize (default: 3).",
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.data_set = args.dataset
    make_corruption_panel(args)


if __name__ == "__main__":
    main()