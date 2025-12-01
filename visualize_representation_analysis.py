"""
Additional Visualization Script for Representation Analysis
Creates supplementary plots and detailed analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load analysis results"""
    results_file = Path(results_dir) / 'similarity_results.json'
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def plot_layer_pair_similarities(results, save_path):
    """Plot CKA and CCA for each layer pair"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    layer_pairs = [f"{r['shvit_layer']}\nvs\n{r['deit_layer']}" for r in results]
    cka_scores = [r['cka'] for r in results]
    cca_scores = [r['cca'] for r in results]
    
    x = np.arange(len(layer_pairs))
    width = 0.35
    
    ax.bar(x - width/2, cka_scores, width, label='CKA', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, cca_scores, width, label='CCA', color='coral', alpha=0.8)
    
    ax.set_xlabel('Layer Pairs', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title('Layer-wise Representation Similarity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_pairs, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer pair similarities to {save_path}")


def plot_depth_category_comparison(results, save_path):
    """Detailed comparison by depth category"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    categories = ['early', 'middle', 'late', 'output']
    
    for idx, cat in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        
        cat_results = [r for r in results if r['depth_category'] == cat]
        
        if not cat_results:
            ax.text(0.5, 0.5, f'No {cat} layers', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{cat.capitalize()} Layers', fontsize=12, fontweight='bold')
            continue
        
        pairs = [f"{r['shvit_layer']} vs {r['deit_layer']}" for r in cat_results]
        cka = [r['cka'] for r in cat_results]
        cca = [r['cca'] for r in cat_results]
        
        x = np.arange(len(pairs))
        width = 0.35
        
        ax.bar(x - width/2, cka, width, label='CKA', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, cca, width, label='CCA', color='coral', alpha=0.8)
        
        ax.set_title(f'{cat.capitalize()} Layers (avg CKA: {np.mean(cka):.3f})', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Similarity', fontsize=10)
    
    plt.suptitle('Representation Similarity by Network Depth', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved depth category comparison to {save_path}")


def plot_feature_stats_comparison(shvit_stats, deit_stats, save_path):
    """Create detailed feature statistics comparison"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    metrics = ['mean', 'std', 'sparsity', 'max', 'min']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get all layers
        shvit_layers = sorted(shvit_stats.keys())
        deit_layers = sorted(deit_stats.keys())
        
        # Create comparison
        x_pos = np.arange(max(len(shvit_layers), len(deit_layers)))
        
        shvit_vals = [shvit_stats[l][metric] for l in shvit_layers]
        deit_vals = [deit_stats[l][metric] for l in deit_layers]
        
        # Pad shorter list
        if len(shvit_vals) < len(deit_vals):
            shvit_vals.extend([np.nan] * (len(deit_vals) - len(shvit_vals)))
        elif len(deit_vals) < len(shvit_vals):
            deit_vals.extend([np.nan] * (len(shvit_vals) - len(deit_vals)))
        
        ax.plot(x_pos[:len(shvit_vals)], shvit_vals, 'o-', 
               label='SHViT', linewidth=2, markersize=8, color='steelblue')
        ax.plot(x_pos[:len(deit_vals)], deit_vals, 's-', 
               label='DeiT', linewidth=2, markersize=8, color='coral')
        
        ax.set_title(f'Feature {metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary statistics in last subplot
    ax = axes[-1]
    ax.axis('off')
    
    summary_text = "Feature Statistics Summary\n" + "="*40 + "\n\n"
    
    for metric in metrics:
        shvit_vals = [shvit_stats[l][metric] for l in shvit_stats.keys()]
        deit_vals = [deit_stats[l][metric] for l in deit_stats.keys()]
        
        summary_text += f"{metric.upper()}:\n"
        summary_text += f"  SHViT: {np.mean(shvit_vals):.4f} ± {np.std(shvit_vals):.4f}\n"
        summary_text += f"  DeiT:  {np.mean(deit_vals):.4f} ± {np.std(deit_vals):.4f}\n\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.suptitle('Detailed Feature Statistics Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed feature stats to {save_path}")


def plot_similarity_distribution(results, save_path):
    """Plot distribution of similarity scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    cka_scores = [r['cka'] for r in results]
    cca_scores = [r['cca'] for r in results]
    
    # CKA histogram
    ax1.hist(cka_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(cka_scores), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(cka_scores):.3f}')
    ax1.axvline(np.median(cka_scores), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(cka_scores):.3f}')
    ax1.set_xlabel('CKA Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of CKA Similarities', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # CCA histogram
    ax2.hist(cca_scores, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(cca_scores), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(cca_scores):.3f}')
    ax2.axvline(np.median(cca_scores), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(cca_scores):.3f}')
    ax2.set_xlabel('CCA Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of CCA Similarities', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Similarity Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity distributions to {save_path}")


def generate_summary_report(data, save_path):
    """Generate a text summary report"""
    results = data['results']
    summary = data['summary']
    
    report = []
    report.append("="*70)
    report.append("REPRESENTATION ANALYSIS SUMMARY REPORT")
    report.append("="*70)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL SIMILARITY METRICS")
    report.append("-"*70)
    report.append(f"Mean CKA:   {summary['mean_cka']:.4f} ± {summary['std_cka']:.4f}")
    report.append(f"Mean CCA:   {summary['mean_cca']:.4f} ± {summary['std_cca']:.4f}")
    report.append(f"Max CKA:    {summary['max_cka']:.4f}")
    report.append(f"Min CKA:    {summary['min_cka']:.4f}")
    report.append("")
    
    # Depth-wise analysis
    report.append("DEPTH-WISE ANALYSIS")
    report.append("-"*70)
    
    for cat in ['early', 'middle', 'late', 'output']:
        cat_results = [r for r in results if r['depth_category'] == cat]
        if cat_results:
            cat_cka = [r['cka'] for r in cat_results]
            cat_cca = [r['cca'] for r in cat_results]
            report.append(f"{cat.upper()} Layers:")
            report.append(f"  CKA: {np.mean(cat_cka):.4f} ± {np.std(cat_cka):.4f}")
            report.append(f"  CCA: {np.mean(cat_cca):.4f} ± {np.std(cat_cca):.4f}")
            report.append(f"  Count: {len(cat_results)} layer pairs")
            report.append("")
    
    # Most similar layers
    report.append("TOP 5 MOST SIMILAR LAYER PAIRS (CKA)")
    report.append("-"*70)
    sorted_results = sorted(results, key=lambda x: x['cka'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        report.append(f"{i}. {r['shvit_layer']} <-> {r['deit_layer']}")
        report.append(f"   CKA: {r['cka']:.4f}, CCA: {r['cca']:.4f}")
    report.append("")
    
    # Least similar layers
    report.append("TOP 5 LEAST SIMILAR LAYER PAIRS (CKA)")
    report.append("-"*70)
    for i, r in enumerate(sorted_results[-5:], 1):
        report.append(f"{i}. {r['shvit_layer']} <-> {r['deit_layer']}")
        report.append(f"   CKA: {r['cka']:.4f}, CCA: {r['cca']:.4f}")
    report.append("")
    
    # Interpretation
    report.append("INTERPRETATION")
    report.append("-"*70)
    mean_cka = summary['mean_cka']
    
    if mean_cka > 0.7:
        interpretation = "HIGH SIMILARITY: Models learn very similar representations."
    elif mean_cka > 0.4:
        interpretation = "MODERATE SIMILARITY: Models capture related features."
    else:
        interpretation = "LOW SIMILARITY: Models learn substantially different features."
    
    report.append(interpretation)
    report.append("")
    report.append("="*70)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved summary report to {save_path}")
    print("\n" + '\n'.join(report))


def main(args):
    results_dir = Path(args.results_dir)
    
    print("Loading results...")
    data = load_results(results_dir)
    
    results = data['results']
    shvit_stats = data['shvit_stats']
    deit_stats = data['deit_stats']
    
    print(f"Found {len(results)} layer pair comparisons")
    
    # Generate additional visualizations
    print("\nGenerating additional visualizations...")
    
    plot_layer_pair_similarities(
        results,
        results_dir / 'layer_pair_similarities.png'
    )
    
    plot_depth_category_comparison(
        results,
        results_dir / 'depth_category_detailed.png'
    )
    
    plot_feature_stats_comparison(
        shvit_stats,
        deit_stats,
        results_dir / 'feature_stats_detailed.png'
    )
    
    plot_similarity_distribution(
        results,
        results_dir / 'similarity_distributions.png'
    )
    
    generate_summary_report(
        data,
        results_dir / 'summary_report.txt'
    )
    
    print("\n" + "="*70)
    print("Additional visualizations complete!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Additional Representation Analysis Visualizations')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing similarity_results.json')
    args = parser.parse_args()
    main(args)
