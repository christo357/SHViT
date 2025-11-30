"""
Robustness Analysis vs Training Data Size

Evaluates how training data size affects model robustness to corruptions.
Tests models trained on different data fractions against various corruptions.

Research Question: Does training with more data improve robustness to corruptions?
Or do architectural properties dominate over data quantity?
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

from timm.models import create_model
from data.datasets import build_dataset
from engine import evaluate
import utils
import model  # Register SHViT models


class CorruptionDataset(torch.utils.data.Dataset):
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
        
        if corruption_type == 'gaussian_noise':
            # Gaussian noise
            noise = torch.randn_like(img) * (0.05 * severity)
            return torch.clamp(img + noise, 0, 1)
        
        elif corruption_type == 'shot_noise':
            # Shot (Poisson) noise
            noisy = torch.poisson(img * (100 / severity)) / (100 / severity)
            return torch.clamp(noisy, 0, 1)
        
        elif corruption_type == 'impulse_noise':
            # Salt and pepper noise
            mask = torch.rand_like(img)
            salt_pepper = torch.where(mask < 0.01 * severity, 
                                     torch.ones_like(img), 
                                     torch.where(mask > 1 - 0.01 * severity,
                                               torch.zeros_like(img),
                                               img))
            return salt_pepper
        
        elif corruption_type == 'gaussian_blur':
            # Gaussian blur using averaging
            kernel_size = 2 * severity + 1
            padding = severity
            # Simple box blur approximation
            blurred = img.clone()
            for c in range(img.shape[0]):
                blurred[c] = torch.nn.functional.avg_pool2d(
                    img[c].unsqueeze(0).unsqueeze(0),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding
                ).squeeze()
            return blurred
        
        elif corruption_type == 'brightness':
            # Brightness shift
            factor = 0.1 + (0.2 * severity)
            return torch.clamp(img + factor, 0, 1)
        
        elif corruption_type == 'contrast':
            # Contrast change
            factor = 0.4 + (0.2 * severity)
            mean = img.mean(dim=[1, 2], keepdim=True)
            return torch.clamp((img - mean) * factor + mean, 0, 1)
        
        elif corruption_type == 'fog':
            # Fog effect (blend with gray)
            fog_intensity = 0.1 * severity
            fog = torch.ones_like(img) * 0.7
            return torch.clamp(img * (1 - fog_intensity) + fog * fog_intensity, 0, 1)
        
        elif corruption_type == 'jpeg_compression':
            # Simulated JPEG artifacts (quantization)
            quantize_factor = severity * 8
            quantized = torch.round(img * quantize_factor) / quantize_factor
            return torch.clamp(quantized, 0, 1)
        
        elif corruption_type == 'pixelate':
            # Pixelation
            scale_factor = max(1, 6 - severity)
            C, H, W = img.shape
            small_h, small_w = H // scale_factor, W // scale_factor
            small = torch.nn.functional.interpolate(
                img.unsqueeze(0), 
                size=(small_h, small_w), 
                mode='nearest'
            )
            pixelated = torch.nn.functional.interpolate(
                small, 
                size=(H, W), 
                mode='nearest'
            ).squeeze(0)
            return pixelated
        
        else:
            return img


@torch.no_grad()
def evaluate_robustness_vs_data(args):
    """Evaluate robustness across different training data fractions"""
    
    device = torch.device(args.device)
    
    # Corruption types and severities
    corruptions = args.corruptions
    severities = args.severities
    fractions = args.fractions
    
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'corruptions': corruptions,
        'severities': severities,
        'fractions': fractions,
        'results': {}
    }
    
    # Setup args for dataset loading
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
    
    # Get number of classes
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
        except:
            nb_classes = 9
    else:
        nb_classes = 1000
    
    # Build base dataset
    dataset_val, _ = build_dataset(is_train=False, args=eval_args)
    
    print("="*80)
    print("Robustness Analysis vs Training Data Size")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Training Fractions: {fractions}")
    print(f"Corruptions: {corruptions}")
    print(f"Severities: {severities}")
    print("="*80)
    
    # Evaluate each training fraction
    for fraction in fractions:
        print(f"\n{'='*80}")
        print(f"Evaluating Model Trained on {fraction*100:.1f}% of Data")
        print(f"{'='*80}")
        
        # Load model
        checkpoint_dir = Path(args.checkpoint_dir) / f"{args.model}_{args.dataset}_frac{fraction}"
        
        # Determine checkpoint based on dataset
        if args.dataset == 'CIFAR':
            checkpoint_path = checkpoint_dir / "checkpoint_99.pth"
        else:  # EUROSAT or MEDMNIST
            checkpoint_path = checkpoint_dir / "checkpoint_29.pth"
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            print("Skipping this fraction...")
            continue
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Create model
        model_net = create_model(
            args.model,
            num_classes=nb_classes,
            pretrained=False
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_net.load_state_dict(checkpoint['model'])
        model_net.to(device)
        model_net.eval()
        
        # Store results for this fraction
        fraction_results = {
            'clean': None,
            'corruptions': {}
        }
        
        # Evaluate on clean data
        print("\nEvaluating on clean data...")
        data_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )
        
        test_stats = evaluate(data_loader, model_net, device)
        fraction_results['clean'] = {
            'acc1': float(test_stats['acc1']),
            'acc5': float(test_stats['acc5']),
            'loss': float(test_stats['loss'])
        }
        print(f"Clean Acc@1: {test_stats['acc1']:.2f}%")
        
        # Evaluate on corruptions
        for corruption in corruptions:
            print(f"\nCorruption: {corruption}")
            fraction_results['corruptions'][corruption] = {}
            
            for severity in severities:
                print(f"  Severity {severity}...", end=' ')
                
                # Create corrupted dataset
                corrupted_dataset = CorruptionDataset(dataset_val, corruption, severity)
                data_loader = torch.utils.data.DataLoader(
                    corrupted_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False
                )
                
                # Evaluate
                test_stats = evaluate(data_loader, model_net, device)
                fraction_results['corruptions'][corruption][f'severity_{severity}'] = {
                    'acc1': float(test_stats['acc1']),
                    'acc5': float(test_stats['acc5']),
                    'loss': float(test_stats['loss'])
                }
                print(f"Acc@1: {test_stats['acc1']:.2f}%")
        
        results['results'][f'frac_{fraction}'] = fraction_results
        
        del model_net
        torch.cuda.empty_cache()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'robustness_vs_data_{args.model}_{args.dataset}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return results


def plot_robustness_analysis(results_file: str, output_dir: str):
    """Create visualization plots"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fractions = results['fractions']
    corruptions = results['corruptions']
    severities = results['severities']
    
    # Plot 1: Clean accuracy vs training fraction
    fig, ax = plt.subplots(figsize=(8, 6))
    
    clean_accs = []
    for frac in fractions:
        clean_acc = results['results'][f'frac_{frac}']['clean']['acc1']
        clean_accs.append(clean_acc)
    
    ax.plot([f*100 for f in fractions], clean_accs, 'o-', linewidth=2, markersize=8, color='green', label='Clean Data')
    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(f'Clean Accuracy vs Training Data Size\n{results["model"]} on {results["dataset"]}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clean_accuracy_vs_data.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'clean_accuracy_vs_data.png'}")
    plt.close()
    
    # Plot 2: Robustness curves for each corruption
    for corruption in corruptions:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Accuracy vs severity for each training fraction
        for frac in fractions:
            accs = []
            for sev in severities:
                acc = results['results'][f'frac_{frac}']['corruptions'][corruption][f'severity_{sev}']['acc1']
                accs.append(acc)
            axes[0].plot(severities, accs, 'o-', linewidth=2, markersize=6, label=f'{frac*100:.1f}% data')
        
        axes[0].set_xlabel('Corruption Severity', fontsize=11)
        axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
        axes[0].set_title(f'{corruption.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Right: Accuracy vs training fraction for each severity
        for sev in severities:
            accs = []
            for frac in fractions:
                acc = results['results'][f'frac_{frac}']['corruptions'][corruption][f'severity_{sev}']['acc1']
                accs.append(acc)
            axes[1].plot([f*100 for f in fractions], accs, 'o-', linewidth=2, markersize=6, label=f'Severity {sev}')
        
        axes[1].set_xlabel('Training Data (%)', fontsize=11)
        axes[1].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
        axes[1].set_title('Robustness vs Training Size', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{results["model"]} on {results["dataset"]}', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f'robustness_{corruption}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'robustness_{corruption}.png'}")
        plt.close()
    
    # Plot 3: Heatmap - mean accuracy across all corruptions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute mean accuracy across all corruptions for each (fraction, severity)
    heatmap_data = []
    for frac in fractions:
        row = []
        for sev in severities:
            accs = []
            for corruption in corruptions:
                acc = results['results'][f'frac_{frac}']['corruptions'][corruption][f'severity_{sev}']['acc1']
                accs.append(acc)
            row.append(np.mean(accs))
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=heatmap_data.max())
    
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f'Sev {s}' for s in severities])
    ax.set_yticks(range(len(fractions)))
    ax.set_yticklabels([f'{f*100:.1f}%' for f in fractions])
    
    ax.set_xlabel('Corruption Severity', fontsize=12)
    ax.set_ylabel('Training Data Fraction', fontsize=12)
    ax.set_title(f'Mean Accuracy Across All Corruptions\n{results["model"]} on {results["dataset"]}', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(fractions)):
        for j in range(len(severities)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'robustness_heatmap.png'}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print("\n1. Clean Accuracy:")
    for frac in fractions:
        clean_acc = results['results'][f'frac_{frac}']['clean']['acc1']
        print(f"   {frac*100:5.1f}% data: {clean_acc:5.2f}%")
    
    print("\n2. Robustness Drop (Clean - Corrupted Mean):")
    for frac in fractions:
        clean_acc = results['results'][f'frac_{frac}']['clean']['acc1']
        all_corrupt_accs = []
        for corruption in corruptions:
            for sev in severities:
                acc = results['results'][f'frac_{frac}']['corruptions'][corruption][f'severity_{sev}']['acc1']
                all_corrupt_accs.append(acc)
        mean_corrupt_acc = np.mean(all_corrupt_accs)
        drop = clean_acc - mean_corrupt_acc
        print(f"   {frac*100:5.1f}% data: {drop:5.2f}% drop (Clean: {clean_acc:.2f}% → Corrupted: {mean_corrupt_acc:.2f}%)")
    
    print("\n3. Best Corruption per Training Fraction:")
    for frac in fractions:
        best_corruption = None
        best_mean_acc = 0
        for corruption in corruptions:
            accs = [results['results'][f'frac_{frac}']['corruptions'][corruption][f'severity_{sev}']['acc1'] 
                   for sev in severities]
            mean_acc = np.mean(accs)
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_corruption = corruption
        print(f"   {frac*100:5.1f}% data: Most robust to {best_corruption} ({best_mean_acc:.2f}% mean)")
    
    print(f"\n{'='*80}")


def get_args_parser():
    parser = argparse.ArgumentParser('Robustness vs Training Data Analysis', add_help=False)
    
    # Model and dataset
    parser.add_argument('--model', default='shvit_s2', type=str)
    parser.add_argument('--dataset', default='CIFAR', type=str,
                        choices=['CIFAR', 'EUROSAT', 'MEDMNIST'])
    parser.add_argument('--medmnist-dataset', default='pathmnist', type=str)
    parser.add_argument('--data-path', default='/research/projects/mllab/vv382/', type=str)
    
    # Training fractions to evaluate
    parser.add_argument('--fractions', nargs='+', type=float, 
                        default=[0.1, 0.325, 0.55, 0.775, 1.0])
    
    # Corruptions to test
    parser.add_argument('--corruptions', nargs='+', type=str,
                        default=['gaussian_noise', 'gaussian_blur', 'brightness', 'contrast'],
                        help='Corruption types to test')
    parser.add_argument('--severities', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Corruption severity levels (1-5)')
    
    # Checkpoint settings
    parser.add_argument('--checkpoint-dir', default='learning_curve_results', type=str)
    
    # Evaluation
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Output
    parser.add_argument('--output-dir', default='analysis/robustness_vs_data', type=str)
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing results')
    parser.add_argument('--results-file', type=str,
                        help='Results file to plot (for --plot-only)')
    
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.plot_only:
        if not args.results_file:
            print("ERROR: --results-file required with --plot-only")
            return
        print("="*80)
        print("Plotting Existing Results")
        print("="*80)
        plot_robustness_analysis(args.results_file, args.output_dir)
    else:
        print("="*80)
        print("Robustness Analysis vs Training Data Size")
        print("="*80)
        results = evaluate_robustness_vs_data(args)
        
        # Generate plots
        output_file = Path(args.output_dir) / f'robustness_vs_data_{args.model}_{args.dataset}.json'
        plot_robustness_analysis(str(output_file), args.output_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
