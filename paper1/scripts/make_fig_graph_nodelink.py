"""
Node-link supplement to Fig. 4 (fig2_frequency_graphs.pdf, the adjacency
heatmaps). Reviewer brief item 1.3: the heatmaps require cell-by-cell
comparison across five separate grids to notice the uniformity; a node-link
rendering makes it visible in one glance, from the exact same saved
adjacency data.

Reads the real, saved per-band adjacency matrices from
results/archive_paper1/interpretability/learned_graphs.pt (a list of 5
torch tensors, shape [8,8] each, produced by the trained HPO-tuned model's
FrequencyGraphConstructor). No synthetic or illustrative edges -- every
edge drawn here is a real kept (top-k=4) edge from the trained model, and
every edge weight is used to set its opacity.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

VARIABLE_NAMES = ["copper", "aluminum", "gold", "oil", "dxy", "sp500", "vix", "us10y"]
SHORT_NAMES = ["Cu", "Al", "Au", "Oil", "DXY", "S&P", "VIX", "US10Y"]
BAND_TITLES = ["Low-freq\n(Trend)", "Mid-low\n(Cycle)", "Mid\n(Seasonal)",
               "Mid-high\n(Short)", "High-freq\n(Noise)"]
BAND_COLORS = ["#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"]


def circular_layout(n, radius=1.0):
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def plot_graph_nodelink(graphs_path: str, output_dir: str = "results/archive_paper1/figures"):
    graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
    N = graphs[0].shape[0]
    pos = circular_layout(N)

    fig, axes = plt.subplots(1, len(graphs), figsize=(4 * len(graphs), 4.4))
    if len(graphs) == 1:
        axes = [axes]

    global_max = max(float(A.max()) for A in graphs)

    for k, (ax, A, title, col) in enumerate(zip(axes, graphs, BAND_TITLES, BAND_COLORS)):
        A = A.numpy()
        # draw edges: every non-zero (kept, top-k) entry, opacity proportional
        # to weight relative to the global max observed across all bands
        for i in range(N):
            for j in range(N):
                w = A[i, j]
                if w <= 0:
                    continue
                alpha = 0.15 + 0.75 * (w / global_max)
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color=col, linewidth=1.3, alpha=min(alpha, 1.0), zorder=1)
        ax.scatter(pos[:, 0], pos[:, 1], s=260, color=col, edgecolor="black",
                   linewidth=0.8, zorder=2)
        for i, name in enumerate(SHORT_NAMES):
            r = 1.28
            ax.text(pos[i, 0] * r / 1.0, pos[i, 1] * r / 1.0, name,
                    ha="center", va="center", fontsize=9, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        "Learned Frequency-Specific Inter-Commodity Graphs (node-link view; "
        "every drawn edge weight $\\approx 1/8$, to five decimal places, in every band)",
        y=1.03, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4b_graph_nodelink.pdf")
    plt.savefig(f"{output_dir}/fig4b_graph_nodelink.png")
    plt.close()
    print(f"Saved {output_dir}/fig4b_graph_nodelink.{{pdf,png}}")


if __name__ == "__main__":
    plot_graph_nodelink(
        "results/archive_paper1/interpretability/learned_graphs.pt",
        output_dir="results/archive_paper1/figures",
    )
