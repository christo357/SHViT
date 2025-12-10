"""
Test How Patchify Stride Affects Domain Generalization

Compares SHViT with different patchify strides across domains:
- Stride 4 (2 conv layers): More patches, higher resolution
- Stride 8 (3 conv layers): Medium patches
- Stride 16 (4 conv layers): Original SHViT, fewer patches
- Stride 32 (5 conv layers): Very few patches

Research Question: Does patchify stride interact with domain characteristics 
(spatial complexity, dataset size, detail requirements)?

To run:
python analyze_patchify_stride.py 
    --model shvit_s2 
    --datasets CIFAR 
    --checkpoint-dir results/
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from timm.models import create_model
from data.datasets import build_dataset
from engine import evaluate, train_one_epoch
import utils
import model  # Register SHViT models

from model.shvit import SHViT, Conv2d_BN


class SHViTCustomStride(SHViT):
    """SHViT with configurable patchify stride"""
    
    def __init__(self, patch_stride=16, *args, **kwargs):
        self.patch_stride = patch_stride
        # Call parent __init__ but we'll override patch_embed
        super().__init__(*args, **kwargs)
        
    def _build_patch_embed(self, in_chans, embed_dim):
        """Build patch embedding with custom stride"""
        if self.patch_stride == 4:
            # 2 layers: 2^2 = 4
            return torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim // 4, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 4, embed_dim, 3, 2, 1)
            )
        elif self.patch_stride == 8:
            # 3 layers: 2^3 = 8
            return torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim // 8, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 4, embed_dim, 3, 2, 1)
            )
        elif self.patch_stride == 16:
            # 4 layers: 2^4 = 16 (original)
            return torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim // 8, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1)
            )
        elif self.patch_stride == 32:
            # 5 layers: 2^5 = 32
            return torch.nn.Sequential(
                Conv2d_BN(in_chans, embed_dim // 16, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 16, embed_dim // 8, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1), 
                torch.nn.ReLU(),
                Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1)
            )
        else:
            raise ValueError(f"Unsupported stride: {self.patch_stride}")


def modify_shvit_stride(base_model_name: str, patch_stride: int, num_classes: int):
    """Create SHViT with custom patchify stride"""
    
    # Get base config
    if base_model_name == 'shvit_s1':
        config = {
            'embed_dim': [128, 224, 320],
            'depth': [2, 4, 5],
            'partial_dim': [32, 48, 68],
            'types': ["i", "s", "s"]
        }
    elif base_model_name == 'shvit_s2':
        config = {
            'embed_dim': [128, 308, 448],
            'depth': [2, 4, 5],
            'partial_dim': [32, 66, 96],
            'types': ["i", "s", "s"]
        }
    elif base_model_name == 'shvit_s3':
        config = {
            'embed_dim': [192, 352, 448],
            'depth': [3, 5, 5],
            'partial_dim': [48, 75, 96],
            'types': ["i", "s", "s"]
        }
    elif base_model_name == 'shvit_s4':
        config = {
            'embed_dim': [224, 336, 448],
            'depth': [4, 7, 6],
            'partial_dim': [48, 72, 96],
            'types': ["i", "s", "s"]
        }
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Create model with custom stride
    model = SHViTCustomStride(
        patch_stride=patch_stride,
        num_classes=num_classes,
        distillation=False,
        **config
    )
    
    # Replace patch_embed with custom one
    model.patch_embed = model._build_patch_embed(3, config['embed_dim'][0])
    
    return model


@torch.no_grad()
def evaluate_stride_across_domains(args):
    """Evaluate different patchify strides across domains"""
    
    device = torch.device(args.device)
    results = {
        'strides': args.strides,
        'datasets': args.datasets,
        'results': {}
    }
    
    for stride in args.strides:
        print("\n" + "="*70)
        print(f"Testing Patchify Stride: {stride}×{stride}")
        print("="*70)
        
        results['results'][f'stride_{stride}'] = {}
        
        for dataset_name in args.datasets:
            print(f"\n--- Dataset: {dataset_name} ---")
            
            # Prepare args for this dataset
            test_args = argparse.Namespace(**vars(args))
            test_args.data_set = dataset_name
            test_args.input_size = 224
            test_args.color_jitter = 0.4
            test_args.aa = 'rand-m9-mstd0.5-inc1'
            test_args.train_interpolation = 'bicubic'
            test_args.reprob = 0.25
            test_args.remode = 'pixel'
            test_args.recount = 1
            test_args.resplit = False
            test_args.ThreeAugment = False
            test_args.finetune = ''
            
            # Get number of classes
            if dataset_name == 'CIFAR':
                nb_classes = 100
            elif dataset_name == 'EUROSAT':
                nb_classes = 10
            elif dataset_name == 'MEDMNIST':
                try:
                    import medmnist
                    from medmnist import INFO
                    info = INFO[args.medmnist_dataset]
                    nb_classes = len(info['label'])
                except:
                    nb_classes = 9
            else:
                nb_classes = 1000
            
            # Build dataset
            dataset_val, _ = build_dataset(is_train=False, args=test_args)
            data_loader = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )
            
            # Create model with this stride
            print(f"Creating {args.model} with stride {stride}...")
            model = modify_shvit_stride(args.model, stride, nb_classes)
            
            # Load checkpoint if exists
            checkpoint_path = f"{args.checkpoint_dir}/{args.model}_{dataset_name}_stride{stride}/checkpoint.pth"
            if Path(checkpoint_path).exists():
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
            else:
                print(f"WARNING: No checkpoint found at {checkpoint_path}")
                print("Evaluating untrained model (not recommended)")
            
            model.to(device)
            model.eval()
            
            # Evaluate
            test_stats = evaluate(data_loader, model, device)
            
            # Calculate spatial complexity metric
            img_size = 224  # Assuming 224x224 images
            num_patches = (img_size // stride) ** 2
            
            results['results'][f'stride_{stride}'][dataset_name] = {
                'acc1': float(test_stats['acc1']),
                'acc5': float(test_stats['acc5']),
                'loss': float(test_stats['loss']),
                'num_patches': num_patches,
                'dataset_size': len(dataset_val)
            }
            
            print(f"Acc@1: {test_stats['acc1']:.2f}%")
            print(f"Number of patches: {num_patches}")
            
            del model
            torch.cuda.empty_cache()
    
    # Analysis: correlation between stride and performance by domain
    print("\n" + "="*70)
    print("ANALYSIS: Stride Impact Across Domains")
    print("="*70)
    
    for dataset_name in args.datasets:
        print(f"\n{dataset_name}:")
        accs = []
        for stride in args.strides:
            acc = results['results'][f'stride_{stride}'][dataset_name]['acc1']
            num_patches = results['results'][f'stride_{stride}'][dataset_name]['num_patches']
            accs.append(acc)
            print(f"  Stride {stride:2d} ({num_patches:4d} patches): {acc:5.2f}% acc")
        
        # Calculate relative performance
        best_acc = max(accs)
        worst_acc = min(accs)
        variance = best_acc - worst_acc
        print(f"  → Variance: {variance:.2f}% (sensitivity to stride)")
        
        results['results'][dataset_name + '_variance'] = variance
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'patchify_stride_analysis.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    print("\nInterpretation Guide:")
    print("- High variance → domain sensitive to patchify stride")
    print("- Low variance → domain robust to stride changes")
    print("- Small datasets benefit from smaller strides (more patches)")
    print("- High-detail domains (medical) may need smaller strides")
    
    return results


def get_args_parser():
    parser = argparse.ArgumentParser('Patchify Stride Analysis', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='shvit_s1', type=str,
                        help='Base model: shvit_s1, shvit_s2, shvit_s3, shvit_s4')
    parser.add_argument('--strides', nargs='+', type=int, default=[8, 16, 32],
                        help='Patchify strides to test')
    
    # Dataset parameters
    parser.add_argument('--datasets', nargs='+', default=['CIFAR', 'EUROSAT', 'MEDMNIST'],
                        help='Datasets to evaluate')
    parser.add_argument('--data-path', default='/research/projects/mllab/vv382/', type=str)
    parser.add_argument('--medmnist-dataset', default='pathmnist', type=str)
    
    # Checkpoint directory
    parser.add_argument('--checkpoint-dir', default='stride_experiments', type=str,
                        help='Directory containing trained models')
    
    # Evaluation parameters
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    # Output
    parser.add_argument('--output-dir', default='analysis/stride_results', type=str)
    
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("="*70)
    print("Patchify Stride Analysis for Domain Generalization")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Strides to test: {args.strides}")
    print(f"Datasets: {args.datasets}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("="*70)
    
    results = evaluate_stride_across_domains(args)
    
    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("1. Compare variance across domains")
    print("2. Plot accuracy vs stride for each domain")
    print("3. Correlate with spatial complexity metrics")


if __name__ == '__main__':
    main()
