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
            except Exception:
                continue

    return best_acc if best_acc > 0 else None


def analyze_learning_curve(results_dir, model_name, dataset_name):
    """
    Analyze learning curve from multiple training runs for a single model.

    Looks for:
      results_dir / f"{model_name}_{dataset_name}_frac{frac}" / "log.txt"
    """
    results_dir = Path(results_dir)
    fractions = [0.1, 0.325, 0.55, 0.775, 1.0]

    results = []

    print("=" * 60)
    print("Learning Curve Analysis")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    for frac in fractions:
        exp_dir = results_dir / f"{model_name}_{dataset_name}_frac{frac}"
        log_file = exp_dir / "log.txt"

        if not log_file.exists():
            print(f"Warning: Log file not found for {model_name}, fraction {frac}, logfile:{log_file}")
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
        print(f"No results found for {model_name} on {dataset_name}!")
        return None

    # Sort by fraction just in case
    results = sorted(results, key=lambda r: r['fraction'])

    # Save results (per model + dataset)
    output_file = results_dir / f'learning_curve_results_{model_name}_{dataset_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def plot_learning_curves(all_results, output_path, dataset_name):
    """
    Plot learning curves for multiple models on the same dataset.

    all_results: dict[model_name] -> list of result dicts
    """
    if not all_results:
        print("No results to plot.")
        return

    # Define consistent colors for each model
    # Adjust / extend if you add more models
    model_names = list(all_results.keys())
    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    # ------------------ Linear x-axis plot ------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    y_vals_all = []

    for i, model_name in enumerate(model_names):
        results = all_results[model_name]
        fractions = [r['fraction'] for r in results]
        percentages = [r['percentage'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        y_vals_all.extend(accuracies)

        color = color_cycle[i % len(color_cycle)]

        ax.plot(
            percentages,
            accuracies,
            'o-',
            linewidth=2,
            markersize=8,
            label=model_name,
            color=color,
        )

        # Add data labels
        for pct, acc in zip(percentages, accuracies):
            ax.annotate(
                f'{acc:.2f}%',
                xy=(pct, acc),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
            )

    ax.set_xlabel('Training Data Size (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Learning Curves on {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # X ticks: union of all percentages
    all_percentages = sorted({
        r['percentage']
        for results in all_results.values()
        for r in results
    })
    ax.set_xticks(all_percentages)
    ax.set_xlim(0, 105)

    # Y axis range from all models
    y_min = min(y_vals_all) - 5
    y_max = max(y_vals_all) + 5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()

    # ------------------ Log-scale x-axis plot ------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        results = all_results[model_name]
        percentages = [r['percentage'] for r in results]
        accuracies = [r['accuracy'] for r in results]

        color = color_cycle[i % len(color_cycle)]

        ax.semilogx(
            percentages,
            accuracies,
            'o-',
            linewidth=2,
            markersize=8,
            label=model_name,
            # default base 10 is fine
        )

        for pct, acc in zip(percentages, accuracies):
            ax.annotate(
                f'{acc:.2f}%',
                xy=(pct, acc),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
            )

    ax.set_xlabel('Training Data Size (%, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Learning Curves (Log Scale) on {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    log_output = output_path.parent / f"{output_path.stem}_logscale.png"
    plt.savefig(log_output, dpi=300, bbox_inches='tight')
    print(f"Log-scale plot saved to {log_output}")
    plt.close()


def compute_data_efficiency_metrics(results):
    """Compute data efficiency metrics for a single model."""
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

        print("\n" + "=" * 60)
        print("Data Efficiency Metrics")
        print("=" * 60)
        print(f"Accuracy at 10% data:  {acc_10:.2f}%")
        print(f"Accuracy at 100% data: {acc_100:.2f}%")
        print(f"Accuracy gap:          {acc_100 - acc_10:.2f}%")
        print(f"Data efficiency score: {acc_10/acc_100:.3f}")
        print("  (ratio of 10% to 100% accuracy - higher is better)")
        print("=" * 60)

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
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        default='results/', 
        help='Directory containing learning curve results (experiment roots)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR',
        help='Dataset name (used in folder names and output filenames)',
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    dataset_name = args.dataset

    # Models to plot together
    MODELS = ['shvit_s1', 'shvit_s2', 'shvit_s3' , 'deit_tiny_patch16_224', 'mobilenetv2_100']

    all_results = {}

    for model_name in MODELS:
        results = analyze_learning_curve(results_dir, model_name, dataset_name)
        if results:
            all_results[model_name] = results

            # Compute and save per-model metrics
            metrics = compute_data_efficiency_metrics(results)
            if metrics:
                metrics_file = results_dir / f"data_efficiency_metrics_{model_name}_{dataset_name}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\nMetrics saved to {metrics_file}")

    # Plot all models together, save in outputs/generalization/ with dataset name
    if all_results:
        gen_dir = Path("outputs") / "generalization"
        gen_dir.mkdir(parents=True, exist_ok=True)
        output_path = gen_dir / f"{dataset_name}_learning_curve.png"
        plot_learning_curves(all_results, output_path, dataset_name)
    else:
        print("No results for any model; nothing to plot.")


if __name__ == '__main__':
    main()