import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_stride_analysis(results_file: str, output_dir: str):
    """Create visualizations from stride analysis results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strides = results['strides']
    datasets = results['datasets']
    
    # Plot 1: Accuracy vs Stride for each domain
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, dataset in enumerate(datasets):
        accs = []
        patches = []
        for stride in strides:
            data = results['results'][f'stride_{stride}'][dataset]
            accs.append(data['acc1'])
            patches.append(data['num_patches'])
        
        ax.plot(strides, accs, marker=markers[idx], color=colors[idx],
                label=dataset, linewidth=2, markersize=8)
    
    ax.set_xlabel('Patchify Stride', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Patchify Stride on Domain Generalization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(strides)
    ax.set_xticklabels([f'{s}×{s}' for s in strides])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stride_vs_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'stride_vs_accuracy.png'}")
    plt.close()
    
    # Plot 2: Number of patches vs Stride
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dataset_example = datasets[0]
    patches_list = []
    for stride in strides:
        patches_list.append(results['results'][f'stride_{stride}'][dataset_example]['num_patches'])
    
    ax.bar(range(len(strides)), patches_list, color='steelblue', alpha=0.7)
    ax.set_xlabel('Patchify Stride', fontsize=12)
    ax.set_ylabel('Number of Patches', fontsize=12)
    ax.set_title('Spatial Resolution: Patches per Image (224×224)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(strides)))
    ax.set_xticklabels([f'{s}×{s}' for s in strides])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add text labels
    for i, patches in enumerate(patches_list):
        ax.text(i, patches + 10, str(patches), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stride_vs_patches.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'stride_vs_patches.png'}")
    plt.close()
    
    # Plot 3: Stride Sensitivity by Domain
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variances = []
    for dataset in datasets:
        accs = [results['results'][f'stride_{s}'][dataset]['acc1'] for s in strides]
        variance = max(accs) - min(accs)
        variances.append(variance)
    
    bars = ax.bar(datasets, variances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(datasets)], alpha=0.7)
    ax.set_ylabel('Accuracy Variance (%)', fontsize=12)
    ax.set_title('Domain Sensitivity to Patchify Stride', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{var:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stride_sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'stride_sensitivity.png'}")
    plt.close()
    
    # Plot 4: Heatmap of accuracy across strides and domains
    fig, ax = plt.subplots(figsize=(10, 6))
    
    acc_matrix = []
    for dataset in datasets:
        accs = [results['results'][f'stride_{s}'][dataset]['acc1'] for s in strides]
        acc_matrix.append(accs)
    
    acc_matrix = np.array(acc_matrix)
    
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=acc_matrix.min()-5, vmax=acc_matrix.max())
    
    ax.set_xticks(range(len(strides)))
    ax.set_xticklabels([f'{s}×{s}' for s in strides])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    
    ax.set_xlabel('Patchify Stride', fontsize=12)
    ax.set_ylabel('Domain', fontsize=12)
    ax.set_title('Accuracy Heatmap: Stride × Domain', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(strides)):
            text = ax.text(j, i, f'{acc_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'stride_domain_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'stride_domain_heatmap.png'}")
    plt.close()
    
    # Print analysis
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n1. Optimal Stride by Domain:")
    for dataset in datasets:
        accs = [results['results'][f'stride_{s}'][dataset]['acc1'] for s in strides]
        best_idx = np.argmax(accs)
        best_stride = strides[best_idx]
        best_acc = accs[best_idx]
        print(f"   {dataset:12s}: {best_stride}×{best_stride} stride ({best_acc:.2f}% acc)")
    
    print("\n2. Stride Sensitivity Ranking:")
    sensitivity_pairs = [(dataset, max(accs) - min(accs)) 
                         for dataset, accs in [(d, [results['results'][f'stride_{s}'][d]['acc1'] for s in strides]) 
                         for d in datasets]]
    sensitivity_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for dataset, sensitivity in sensitivity_pairs:
        print(f"   {dataset:12s}: {sensitivity:.2f}% variance")
        if sensitivity > 5:
            print(f"                → High sensitivity to stride")
        elif sensitivity > 2:
            print(f"                → Moderate sensitivity")
        else:
            print(f"                → Low sensitivity (robust)")
    
    print("\n3. Key Insights:")
    print("   - Smaller strides (more patches) may help with:")
    print("     * Small datasets (less information loss)")
    print("     * High-detail domains (medical imaging)")
    print("   - Larger strides (fewer patches) may help with:")
    print("     * Computational efficiency")
    print("     * Avoiding overfitting on small datasets")
    print("     * Natural images with global structure")


def main():
    parser = argparse.ArgumentParser('Visualize Patchify Stride Analysis')
    parser.add_argument('--results-file', default='analysis/stride_results/patchify_stride_analysis.json',
                        type=str, help='Path to analysis results JSON')
    parser.add_argument('--output-dir', default='analysis/stride_results/plots',
                        type=str, help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Patchify Stride Analysis Visualization")
    print("="*70)
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    plot_stride_analysis(args.results_file, args.output_dir)
    
    print("\n Visualization complete!")


if __name__ == '__main__':
    main()
