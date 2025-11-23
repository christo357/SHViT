"""
Analyze Learning Curve Results

Extracts test accuracy from log files and creates visualization
showing model performance vs. training data size.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_best_accuracy(log_file):
    """Extract best test accuracy from log.txt"""
    best_acc = 0.0
    
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if 'test_acc1' in log_entry:
                    best_acc = max(best_acc, log_entry['test_acc1'])
            except:
                continue
    
    return best_acc if best_acc > 0 else None


def analyze_learning_curve(results_dir, model_name, dataset_name):
    """Analyze learning curve from multiple training runs"""
    
    results_dir = Path(results_dir)
    fractions = [0.1, 0.325, 0.55, 0.775, 1.0]
    
    results = []
    
    print("="*60)
    print("Learning Curve Analysis")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("="*60)
    
    for frac in fractions:
        exp_dir = results_dir / f"{model_name}_{dataset_name}_frac{frac}"
        log_file = exp_dir / "log.txt"
        
        if not log_file.exists():
            print(f"Warning: Log file not found for fraction {frac}")
            continue
        
        best_acc = extract_best_accuracy(log_file)
        
        if best_acc is not None:
            results.append({
                'fraction': frac,
                'percentage': frac * 100,
                'accuracy': best_acc,
                'num_samples': frac  # Relative to full dataset
            })
            print(f"Fraction {frac:5.1%} ({frac*100:5.1f}%): {best_acc:6.2f}% accuracy")
    
    if not results:
        print("No results found!")
        return None
    
    # Save results
    output_file = results_dir / 'learning_curve_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


def plot_learning_curve(results, output_path, model_name, dataset_name):
    """Create learning curve visualization"""
    
    if not results:
        return
    
    fractions = [r['fraction'] for r in results]
    percentages = [r['percentage'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curve
    ax.plot(percentages, accuracies, 'o-', linewidth=2, markersize=10, 
            color='#2ecc71', label=model_name)
    
    # Add data point labels
    for pct, acc in zip(percentages, accuracies):
        ax.annotate(f'{acc:.2f}%', 
                   xy=(pct, acc), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Styling
    ax.set_xlabel('Training Data Size (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Learning Curve: {model_name} on {dataset_name}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Set x-axis to show all data points
    ax.set_xticks(percentages)
    ax.set_xlim(0, 105)
    
    # Set y-axis range
    y_min = min(accuracies) - 5
    y_max = max(accuracies) + 5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()
    
    # Also create a log-scale version for better visualization of early improvement
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use log scale for x-axis
    ax.semilogx(percentages, accuracies, 'o-', linewidth=2, markersize=10,
               color='#e74c3c', label=model_name, base=10)
    
    for pct, acc in zip(percentages, accuracies):
        ax.annotate(f'{acc:.2f}%', 
                   xy=(pct, acc), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Training Data Size (%, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Learning Curve (Log Scale): {model_name} on {dataset_name}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    log_output = output_path.parent / f"{output_path.stem}_logscale.png"
    plt.savefig(log_output, dpi=300, bbox_inches='tight')
    print(f"Log-scale plot saved to {log_output}")
    plt.close()


def compute_data_efficiency_metrics(results):
    """Compute data efficiency metrics"""
    
    if len(results) < 2:
        return {}
    
    # Find accuracy at 10% and 100%
    acc_10 = next((r['accuracy'] for r in results if r['fraction'] == 0.1), None)
    acc_100 = next((r['accuracy'] for r in results if r['fraction'] == 1.0), None)
    
    metrics = {}
    
    if acc_10 and acc_100:
        metrics['accuracy_10pct'] = acc_10
        metrics['accuracy_100pct'] = acc_100
        metrics['accuracy_gap'] = acc_100 - acc_10
        metrics['data_efficiency_score'] = acc_10 / acc_100  # Higher is better
        
        print("\n" + "="*60)
        print("Data Efficiency Metrics")
        print("="*60)
        print(f"Accuracy at 10% data:  {acc_10:.2f}%")
        print(f"Accuracy at 100% data: {acc_100:.2f}%")
        print(f"Accuracy gap:          {acc_100 - acc_10:.2f}%")
        print(f"Data efficiency score: {acc_10/acc_100:.3f}")
        print(f"  (ratio of 10% to 100% accuracy - higher is better)")
        print("="*60)
    
    # Compute sample efficiency: how much data needed to reach 90% of final accuracy
    if acc_100:
        target_acc = 0.9 * acc_100
        for r in results:
            if r['accuracy'] >= target_acc:
                metrics['data_for_90pct_performance'] = r['percentage']
                print(f"\nReaches 90% of final performance at {r['percentage']:.1f}% of training data")
                break
    
    return metrics


def main():
    parser = argparse.ArgumentParser('Analyze Learning Curve Results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing learning curve results')
    parser.add_argument('--model', type=str, default='shvit_s4',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='CIFAR',
                       help='Dataset name')
    
    args = parser.parse_args()
    
    # Analyze results
    results = analyze_learning_curve(args.results_dir, args.model, args.dataset)
    
    if results:
        # Create plots
        output_path = Path(args.results_dir) / 'learning_curve.png'
        plot_learning_curve(results, output_path, args.model, args.dataset)
        
        # Compute efficiency metrics
        metrics = compute_data_efficiency_metrics(results)
        
        # Save metrics
        if metrics:
            metrics_file = Path(args.results_dir) / 'data_efficiency_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
