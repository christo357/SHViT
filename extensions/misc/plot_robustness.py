import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_results_json(json_path: Path) -> Dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def collect_results_from_dir(
    input_dir: Path,
    dataset: str,
    severity: int,
    models: List[str] = None,
) -> Dict[str, Dict]:
    """
    Load robustness JSONs for a given dataset & severity.

    If `models` is provided, we look for:
        robustness_sev<severity>_<model>_<dataset>.json
    Otherwise, auto-discover all matching JSONs in the directory.

    Returns:
        all_results: dict[model_name] -> results_dict
    """
    all_results: Dict[str, Dict] = {}

    if models:
        json_paths: List[Path] = []
        for m in models:
            jp = input_dir / f"robustness_sev{severity}_{m}_{dataset}.json"
            if not jp.exists():
                print(f"[WARN] JSON not found for model {m}: {jp}")
                continue
            json_paths.append(jp)
    else:
        pattern = f"robustness_sev{severity}_*_{dataset}.json"
        json_paths = list(input_dir.glob(pattern))
        if not json_paths:
            raise FileNotFoundError(f"No JSON files matching {pattern} in {input_dir}")

    for jp in json_paths:
        data = load_results_json(jp)
        model_name = data["model"]
        all_results[model_name] = data
        print(f"[INFO] Loaded results for model {model_name} from {jp}")

    if not all_results:
        raise RuntimeError("No valid robustness JSONs were loaded.")

    return all_results



def plot_multi_model_robustness_subplots_by_corruption(
    all_results: Dict[str, Dict],
    severity_to_plot: int,
    output_path: Path,
):
    """
    Make a grid of subplots:

      - One subplot per corruption type (including "Clean")
      - X-axis: training fraction (%)
      - Different color per model (consistent across subplots)
      - Shared x/y axes

    all_results: dict mapping model_name -> results_dict
    """

    # Use first model as reference for metadata
    first_model_name = next(iter(all_results.keys()))
    ref = all_results[first_model_name]

    fractions   = ref["fractions"]      # e.g. [0.1, 0.325, ...]
    corruptions = ref["corruptions"]    # e.g. ["gaussian_noise", ...]
    severities  = ref["severities"]     # e.g. [3]
    dataset     = ref["dataset"]
    frac_percent = [f * 100 for f in fractions]

    if severity_to_plot not in severities:
        raise ValueError(
            f"Requested severity {severity_to_plot}, "
            f"but available severities = {severities}"
        )

    model_names = list(all_results.keys())

    # We treat "clean" as an additional pseudo-corruption
    corr_list = ["clean"] + corruptions
    n_corr = len(corr_list)

    # ---- Create grid of subplots ----
    # Try up to 3 columns; adjust rows automatically.
    n_cols = 3
    n_rows = int(np.ceil(n_corr / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.4 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    # Color mapping – one color per model, consistent across all subplots
    cmap_models = plt.cm.get_cmap("Set2", len(model_names))
    model_colors = [cmap_models(i) for i in range(len(model_names))]

    for ci, corr in enumerate(corr_list):
        ax = axes[ci]

        for mi, mname in enumerate(model_names):
            res = all_results[mname]

            if corr == "clean":
                # JSON stores dict keys as strings
                acc = [res["clean"].get(str(f), np.nan) for f in fractions]
            else:
                sev_key = str(severity_to_plot)
                acc = [
                    res["corrupt"][corr][sev_key].get(str(f), np.nan)
                    for f in fractions
                ]

            ax.plot(
                frac_percent,
                acc,
                "-o",
                linewidth=2,
                markersize=5,
                color=model_colors[mi],
                label=mname,
            )

        # Per-panel title
        if corr == "clean":
            title = "Clean"
        else:
            title = corr.replace("_", " ").title()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    # Remove any unused axes (if grid has extra cells)
    for j in range(n_corr, len(axes)):
        fig.delaxes(axes[j])

    # Shared axis labels
    # X label on bottom row
    for ax in axes[(n_rows - 1) * n_cols : n_rows * n_cols]:
        if ax in fig.axes:
            ax.set_xlabel("Training Data (%)", fontsize=11, fontweight="bold")

    # Y label on leftmost column
    for row in range(n_rows):
        idx = row * n_cols
        if idx < len(axes) and axes[idx] in fig.axes:
            axes[idx].set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")

    # Figure title
    fig.suptitle(
        f"Clean & Corrupted Robustness vs Training Data Size\n"
        f"{dataset} (Severity {severity_to_plot})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Global legend for models (colors)
    model_handles = [
        Line2D(
            [0],
            [0],
            color=model_colors[i],
            linewidth=2,
            marker="o",
            label=mname,
        )
        for i, mname in enumerate(model_names)
    ]
    fig.legend(
        handles=model_handles,
        labels=model_names,
        title="Model",
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(model_names),
    )

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved subplot robustness figure: {output_path}")

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Plot multi-model robustness vs training data size "
        "(subplots by corruption, including clean).",
        add_help=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR",
        choices=["CIFAR", "EUROSAT", "MEDMNIST"],
        help="Dataset name used in the JSON filenames.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Corruption severity to plot (must match JSON).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional list of model names. If omitted, auto-detects models from "
            "JSON files in --input-dir."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/robustness_multi",
        help="Directory where robustness JSON files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/robustness_multi",
        help="Directory to save the subplot figure.",
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    all_results = collect_results_from_dir(
        input_dir=input_dir,
        dataset=args.dataset,
        severity=args.severity,
        models=args.models,
    )

    out_path = (
        output_dir
        / f"{args.dataset}_robustness_sev{args.severity}_subplots_by_corruption.png"
    )

    plot_multi_model_robustness_subplots_by_corruption(
        all_results=all_results,
        severity_to_plot=args.severity,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()