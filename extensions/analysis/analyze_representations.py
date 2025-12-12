import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from timm.models import create_model
from data.datasets import build_dataset
from model import build


class RepresentationExtractor(nn.Module):
    """Extract intermediate representations from a model"""
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to extract features"""
        def get_hook(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(get_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def forward(self, x):
        """Forward pass with feature extraction"""
        self.features = {}
        output = self.model(x)
        return output, self.features


def get_model_layers(model, model_type='shvit'):
    """Get layer names for feature extraction"""
    if model_type == 'shvit':
        # Extract from blocks1, blocks2, blocks3
        layers = []
        for i in range(1, 4):
            block_name = f'blocks{i}'
            if hasattr(model, block_name):
                block = getattr(model, block_name)
                # Add intermediate blocks
                for j, layer in enumerate(block):
                    layers.append(f'blocks{i}.{j}')
        layers.append('head')  # Final classification head
        return layers
    
    elif model_type == 'deit':
        # Extract from transformer blocks
        layers = []
        if hasattr(model, 'blocks'):
            for i in range(len(model.blocks)):
                layers.append(f'blocks.{i}')
        layers.append('head')
        return layers
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two feature matrices
    X, Y: [N, D] where N is number of samples, D is feature dimension
    """
    # Flatten spatial dimensions if present
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    
    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    # Compute CKA
    hsic = (K_X * K_Y).sum()
    norm_x = torch.sqrt((K_X * K_X).sum())
    norm_y = torch.sqrt((K_Y * K_Y).sum())
    
    cka = hsic / (norm_x * norm_y + 1e-10)
    return cka.item()


def compute_cca_similarity(X, Y, n_components=10):
    """
    Compute CCA similarity between two feature matrices
    X, Y: [N, D] numpy arrays
    """
    # Flatten spatial dimensions if present
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    
    # Reduce dimensions if needed
    n_components = min(n_components, X.shape[0], X.shape[1], Y.shape[1])
    
    if n_components < 2:
        return 0.0
    
    try:
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X, Y)
        
        # Compute mean correlation
        correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
        return np.mean(correlations)
    except:
        return 0.0


def compute_feature_statistics(features):
    """Compute statistics of features"""
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    stats = {
        'mean': features.mean().item(),
        'std': features.std().item(),
        'sparsity': (features.abs() < 0.01).float().mean().item(),
        'max': features.max().item(),
        'min': features.min().item(),
    }
    return stats


def extract_all_features(model, dataloader, device, layer_names, max_batches=50):
    """Extract features from all specified layers"""
    model.eval()
    extractor = RepresentationExtractor(model, layer_names)
    extractor.register_hooks()
    
    all_features = {name: [] for name in layer_names}
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Extracting features")):
            if batch_idx >= max_batches:
                break
                
            images = images.to(device)
            _, features = extractor(images)
            
            for name in layer_names:
                if name in features:
                    feat = features[name]
                    # Move to CPU to save GPU memory
                    all_features[name].append(feat.cpu())
    
    extractor.remove_hooks()
    
    # Concatenate all batches
    for name in all_features:
        if all_features[name]:
            all_features[name] = torch.cat(all_features[name], dim=0)
    
    return all_features


def align_layers(shvit_layers, deit_layers):
    """
    Create alignment between SHViT and DeiT layers for comparison
    This is approximate as architectures differ
    """
    # Simple strategy: compare layers at similar depths
    alignments = []
    
    # Compare early, middle, and late layers
    n_shvit = len(shvit_layers) - 1  # Exclude head
    n_deit = len(deit_layers) - 1
    
    # Early layers (first 25%)
    shvit_early = shvit_layers[:max(1, n_shvit // 4)]
    deit_early = deit_layers[:max(1, n_deit // 4)]
    for s in shvit_early:
        for d in deit_early:
            alignments.append((s, d, 'early'))
    
    # Middle layers (25-75%)
    shvit_mid = shvit_layers[n_shvit // 4: 3 * n_shvit // 4]
    deit_mid = deit_layers[n_deit // 4: 3 * n_deit // 4]
    for s in shvit_mid:
        for d in deit_mid:
            alignments.append((s, d, 'middle'))
    
    # Late layers (last 25%)
    shvit_late = shvit_layers[3 * n_shvit // 4:]
    deit_late = deit_layers[3 * n_deit // 4:]
    for s in shvit_late:
        for d in deit_late:
            alignments.append((s, d, 'late'))
    
    # Always compare heads
    alignments.append(('head', 'head', 'output'))
    
    return alignments


def visualize_cka_matrix(cka_matrix, shvit_layers, deit_layers, save_path):
    """Visualize CKA similarity matrix"""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cka_matrix, 
                xticklabels=deit_layers,
                yticklabels=shvit_layers,
                cmap='RdYlGn',
                vmin=0, vmax=1,
                annot=True if len(shvit_layers) < 10 else False,
                fmt='.2f',
                cbar_kws={'label': 'CKA Similarity'})
    
    plt.xlabel('DeiT Layers', fontsize=12)
    plt.ylabel('SHViT Layers', fontsize=12)
    plt.title('Representation Similarity (CKA)\nSHViT vs DeiT', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CKA matrix to {save_path}")


def visualize_feature_statistics(shvit_stats, deit_stats, save_path):
    """Visualize feature statistics comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['mean', 'std', 'sparsity', 'max', 'min']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        shvit_layers = list(shvit_stats.keys())
        deit_layers = list(deit_stats.keys())
        
        shvit_values = [shvit_stats[layer][metric] for layer in shvit_layers]
        deit_values = [deit_stats[layer][metric] for layer in deit_layers]
        
        x_shvit = np.arange(len(shvit_layers))
        x_deit = np.arange(len(deit_layers))
        
        ax.plot(x_shvit, shvit_values, 'o-', label='SHViT', linewidth=2, markersize=6)
        ax.plot(x_deit, deit_values, 's-', label='DeiT', linewidth=2, markersize=6)
        
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_title(f'Feature {metric.capitalize()}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle('Feature Statistics Comparison: SHViT vs DeiT', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature statistics to {save_path}")


def visualize_similarity_progression(cka_scores, cca_scores, alignments, save_path):
    """Visualize how similarity progresses through network depth"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Group by depth category
    depth_categories = ['early', 'middle', 'late', 'output']
    cka_by_depth = {cat: [] for cat in depth_categories}
    cca_by_depth = {cat: [] for cat in depth_categories}
    
    for idx, (s_layer, d_layer, depth) in enumerate(alignments):
        if idx < len(cka_scores):
            cka_by_depth[depth].append(cka_scores[idx])
        if idx < len(cca_scores):
            cca_by_depth[depth].append(cca_scores[idx])
    
    # CKA plot
    cka_means = [np.mean(cka_by_depth[cat]) if cka_by_depth[cat] else 0 
                 for cat in depth_categories]
    cka_stds = [np.std(cka_by_depth[cat]) if len(cka_by_depth[cat]) > 1 else 0 
                for cat in depth_categories]
    
    x = np.arange(len(depth_categories))
    ax1.bar(x, cka_means, yerr=cka_stds, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in depth_categories])
    ax1.set_ylabel('CKA Similarity', fontsize=12)
    ax1.set_xlabel('Network Depth', fontsize=12)
    ax1.set_title('CKA Similarity by Network Depth', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # CCA plot
    cca_means = [np.mean(cca_by_depth[cat]) if cca_by_depth[cat] else 0 
                 for cat in depth_categories]
    cca_stds = [np.std(cca_by_depth[cat]) if len(cca_by_depth[cat]) > 1 else 0 
                for cat in depth_categories]
    
    ax2.bar(x, cca_means, yerr=cca_stds, capsize=5, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.capitalize() for c in depth_categories])
    ax2.set_ylabel('CCA Similarity', fontsize=12)
    ax2.set_xlabel('Network Depth', fontsize=12)
    ax2.set_title('CCA Similarity by Network Depth', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Representation Similarity Progression: SHViT vs DeiT', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity progression to {save_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset_val, _ = build_dataset(is_train=False, args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load SHViT model
    print(f"Loading SHViT model from {args.shvit_checkpoint}...")
    shvit_model = create_model(
        args.shvit_model,
        num_classes=args.nb_classes,
        distillation=False,
        pretrained=False,
        fuse=False,
    )
    checkpoint = torch.load(args.shvit_checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        shvit_model.load_state_dict(checkpoint['model'])
    else:
        shvit_model.load_state_dict(checkpoint)
    shvit_model.to(device)
    shvit_model.eval()
    print(f"SHViT model loaded successfully")
    
    # Load DeiT model
    print(f"Loading DeiT model from {args.deit_checkpoint}...")
    deit_model = create_model(
        args.deit_model,
        num_classes=args.nb_classes,
        pretrained=False,
    )
    checkpoint = torch.load(args.deit_checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        deit_model.load_state_dict(checkpoint['model'])
    else:
        deit_model.load_state_dict(checkpoint)
    deit_model.to(device)
    deit_model.eval()
    print(f"DeiT model loaded successfully")
    
    # Get layer names
    shvit_layers = get_model_layers(shvit_model, 'shvit')
    deit_layers = get_model_layers(deit_model, 'deit')
    
    print(f"\nSHViT layers ({len(shvit_layers)}): {shvit_layers}")
    print(f"DeiT layers ({len(deit_layers)}): {deit_layers}")
    
    # Extract features
    print("\n" + "="*50)
    print("Extracting SHViT features...")
    shvit_features = extract_all_features(shvit_model, dataloader, device, shvit_layers, args.max_batches)
    
    print("Extracting DeiT features...")
    deit_features = extract_all_features(deit_model, dataloader, device, deit_layers, args.max_batches)
    
    # Compute feature statistics
    print("\n" + "="*50)
    print("Computing feature statistics...")
    shvit_stats = {}
    for layer_name, features in shvit_features.items():
        if features.numel() > 0:
            shvit_stats[layer_name] = compute_feature_statistics(features)
    
    deit_stats = {}
    for layer_name, features in deit_features.items():
        if features.numel() > 0:
            deit_stats[layer_name] = compute_feature_statistics(features)
    
    # Create layer alignments
    alignments = align_layers(shvit_layers, deit_layers)
    print(f"\nComparing {len(alignments)} layer pairs...")
    
    # Compute CKA and CCA similarities
    print("\n" + "="*50)
    print("Computing representation similarities...")
    cka_scores = []
    cca_scores = []
    results = []
    
    for shvit_layer, deit_layer, depth_cat in tqdm(alignments, desc="Computing similarities"):
        if shvit_layer in shvit_features and deit_layer in deit_features:
            shvit_feat = shvit_features[shvit_layer]
            deit_feat = deit_features[deit_layer]
            
            if shvit_feat.numel() > 0 and deit_feat.numel() > 0:
                # CKA
                cka = compute_cka(shvit_feat, deit_feat)
                cka_scores.append(cka)
                
                # CCA
                cca = compute_cca_similarity(
                    shvit_feat.numpy(), 
                    deit_feat.numpy(), 
                    n_components=min(10, shvit_feat.shape[0] // 2)
                )
                cca_scores.append(cca)
                
                results.append({
                    'shvit_layer': shvit_layer,
                    'deit_layer': deit_layer,
                    'depth_category': depth_cat,
                    'cka': cka,
                    'cca': cca
                })
    
    # Create CKA matrix for all layer pairs
    print("\n" + "="*50)
    print("Creating CKA matrix...")
    cka_matrix = np.zeros((len(shvit_layers), len(deit_layers)))
    for i, s_layer in enumerate(shvit_layers):
        for j, d_layer in enumerate(deit_layers):
            if s_layer in shvit_features and d_layer in deit_features:
                s_feat = shvit_features[s_layer]
                d_feat = deit_features[d_layer]
                if s_feat.numel() > 0 and d_feat.numel() > 0:
                    cka_matrix[i, j] = compute_cka(s_feat, d_feat)
    
    # Save results
    print("\n" + "="*50)
    print("Saving results...")
    
    # Save numerical results
    results_file = output_dir / 'similarity_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'shvit_stats': shvit_stats,
            'deit_stats': deit_stats,
            'summary': {
                'mean_cka': np.mean(cka_scores),
                'std_cka': np.std(cka_scores),
                'mean_cca': np.mean(cca_scores),
                'std_cca': np.std(cca_scores),
                'max_cka': np.max(cka_scores),
                'min_cka': np.min(cka_scores),
            }
        }, f, indent=2)
    print(f"Saved results to {results_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_cka_matrix(
        cka_matrix, 
        shvit_layers, 
        deit_layers, 
        output_dir / 'cka_matrix.png'
    )
    
    visualize_feature_statistics(
        shvit_stats,
        deit_stats,
        output_dir / 'feature_statistics.png'
    )
    
    visualize_similarity_progression(
        cka_scores,
        cca_scores,
        alignments,
        output_dir / 'similarity_progression.png'
    )
    
    # Print summary
    print("\n" + "="*50)
    print("REPRESENTATION ANALYSIS SUMMARY")
    print("="*50)
    print(f"Models compared: {args.shvit_model} vs {args.deit_model}")
    print(f"Dataset: {args.data_set}")
    print(f"Number of samples analyzed: {shvit_features[shvit_layers[0]].shape[0]}")
    print(f"\nSimilarity Metrics:")
    print(f"  CKA: {np.mean(cka_scores):.4f} ± {np.std(cka_scores):.4f}")
    print(f"  CCA: {np.mean(cca_scores):.4f} ± {np.std(cca_scores):.4f}")
    print(f"\nMax CKA similarity: {np.max(cka_scores):.4f}")
    print(f"Min CKA similarity: {np.min(cka_scores):.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Representation Analysis')
    
    # Model paths
    parser.add_argument('--shvit-checkpoint', type=str, required=True,
                        help='Path to SHViT checkpoint')
    parser.add_argument('--deit-checkpoint', type=str, required=True,
                        help='Path to DeiT checkpoint')
    parser.add_argument('--shvit-model', type=str, default='shvit_s2',
                        help='SHViT model name')
    parser.add_argument('--deit-model', type=str, default='deit_tiny_patch16_224',
                        help='DeiT model name')
    
    # Dataset parameters
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--data-set', type=str, default='CIFAR',
                        choices=['CIFAR', 'IMNET', 'EUROSAT', 'MEDMNIST'],
                        help='Dataset name')
    parser.add_argument('--medmnist-dataset', default='pathmnist', type=str,
                        help='MedMNIST dataset variant')
    parser.add_argument('--nb-classes', type=int, default=100,
                        help='Number of classes')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    
    # Data augmentation parameters (required by build_transform, not used for evaluation)
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint (not used in analysis)')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (not used in eval mode)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='AutoAugment policy (not used in eval mode)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (not used in eval mode)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (not used in eval mode)')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (not used in eval mode)')
    
    # Analysis parameters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument('--max-batches', type=int, default=50,
                        help='Maximum number of batches to process')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args)