"""Visualization module for paper figures."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_vmd_decomposition(modes: np.ndarray, prices, variable_names: List[str],
                           var_idx: int = 0, output_dir: str = "results/figures"):
    """Fig 1: VMD decomposition of copper price into K modes."""
    K = modes.shape[1]
    fig, axes = plt.subplots(K + 1, 1, figsize=(12, 2.5 * (K + 1)), sharex=True)

    dates = prices.index
    raw = prices.iloc[:, var_idx].values

    axes[0].plot(dates, raw, "k-", linewidth=0.8)
    axes[0].set_ylabel("Original")
    axes[0].set_title(f"VMD Decomposition of {variable_names[var_idx].title()} Price (K={K})")

    labels = ["Low-freq\n(Trend)", "Mid-low\n(Cycle)", "Mid\n(Seasonal)",
              "Mid-high\n(Short-term)", "High-freq\n(Noise)"]
    colors = sns.color_palette("viridis", K)
    for k in range(K):
        axes[k + 1].plot(dates, modes[var_idx, k, :], color=colors[k], linewidth=0.7)
        label = labels[k] if k < len(labels) else f"Mode {k+1}"
        axes[k + 1].set_ylabel(label)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_vmd_decomposition.pdf")
    plt.savefig(f"{output_dir}/fig1_vmd_decomposition.png")
    plt.close()


def plot_frequency_graphs(model, variable_names: List[str],
                          output_dir: str = "results/figures"):
    """Fig 2: Learned frequency-specific inter-commodity graphs."""
    graphs = model.get_learned_graphs()
    if not graphs:
        return

    K = len(graphs)
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 3.5))
    if K == 1:
        axes = [axes]

    titles = ["Low-freq (Trend)", "Mid-low (Cycle)", "Mid (Seasonal)",
              "Mid-high (Short)", "High-freq (Noise)"]

    for k, (ax, adj) in enumerate(zip(axes, graphs)):
        adj_np = adj.cpu().numpy()
        sns.heatmap(adj_np, ax=ax, cmap="YlOrRd", vmin=0, vmax=adj_np.max(),
                    xticklabels=variable_names, yticklabels=variable_names,
                    square=True, cbar_kws={"shrink": 0.8})
        ax.set_title(titles[k] if k < len(titles) else f"Band {k+1}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("Learned Frequency-Specific Inter-Commodity Graphs", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_frequency_graphs.pdf")
    plt.savefig(f"{output_dir}/fig2_frequency_graphs.png")
    plt.close()


def plot_attention_weights(attn_weights: np.ndarray, horizons: List[int],
                           output_dir: str = "results/figures"):
    """Fig 3: Mode attention weights across forecast horizons."""
    K = attn_weights.shape[1] if attn_weights.ndim > 1 else attn_weights.shape[0]
    mode_labels = [f"Mode {k+1}" for k in range(K)]

    fig, ax = plt.subplots(figsize=(8, 4))
    mean_attn = attn_weights.mean(axis=0) if attn_weights.ndim > 1 else attn_weights
    bars = ax.bar(mode_labels, mean_attn, color=sns.color_palette("viridis", K))
    ax.set_ylabel("Attention Weight")
    ax.set_title("Average Mode Importance Weights")
    ax.set_ylim(0, max(mean_attn) * 1.3)
    for bar, val in zip(bars, mean_attn):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_attention_weights.pdf")
    plt.savefig(f"{output_dir}/fig3_attention_weights.png")
    plt.close()


def plot_results_comparison(results: Dict, horizons: List[int],
                            output_dir: str = "results/figures"):
    """Fig 4: Bar chart comparing all models across horizons."""
    models = []
    for name, res in results.items():
        if "error" not in res:
            models.append(name)

    metrics = ["rmse", "mae", "da"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        x = np.arange(len(models))
        width = 0.18
        for hi, h in enumerate(horizons):
            vals = []
            for name in models:
                res = results[name]
                key = f"h{h}"
                if key in res and metric in res[key]:
                    vals.append(res[key][metric])
                else:
                    vals.append(0)
            offset = (hi - len(horizons) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=f"h={h}")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Model and Horizon")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_model_comparison.pdf")
    plt.savefig(f"{output_dir}/fig4_model_comparison.png")
    plt.close()


def plot_ablation_results(ablation_results: Dict, horizons: List[int],
                          output_dir: str = "results/figures"):
    """Fig 5: Ablation study results."""
    models = list(ablation_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE comparison
    for hi, h in enumerate(horizons):
        vals = []
        for name in models:
            res = ablation_results[name]
            key = f"h{h}"
            vals.append(res[key]["rmse"] if key in res else 0)
        axes[0].plot(models, vals, "o-", label=f"h={h}")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Ablation Study: RMSE")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    # DA comparison
    for hi, h in enumerate(horizons):
        vals = []
        for name in models:
            res = ablation_results[name]
            key = f"h{h}"
            vals.append(res[key]["da"] if key in res else 50)
        axes[1].plot(models, vals, "s-", label=f"h={h}")
    axes[1].set_ylabel("Directional Accuracy (%)")
    axes[1].set_title("Ablation Study: Directional Accuracy")
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_ablation.pdf")
    plt.savefig(f"{output_dir}/fig5_ablation.png")
    plt.close()


def generate_all_figures(config: dict, data: Dict, results: Dict,
                         ablation_results: Dict, output_dir: str = "results/figures",
                         model=None, attn_weights=None):
    """Generate all paper figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    horizons = data["horizons"]

    # Fig 1: VMD decomposition
    plot_vmd_decomposition(data["modes"], data["prices"],
                           data["variable_names"], var_idx=0, output_dir=output_dir)

    # Fig 2: Frequency graphs
    if model is not None:
        plot_frequency_graphs(model, data["variable_names"], output_dir)

    # Fig 3: Attention weights
    if attn_weights is not None:
        plot_attention_weights(attn_weights, horizons, output_dir)

    # Fig 4: Model comparison
    plot_results_comparison(results, horizons, output_dir)

    # Fig 5: Ablation
    if ablation_results:
        plot_ablation_results(ablation_results, horizons, output_dir)

    print(f"Figures saved to {output_dir}/")
