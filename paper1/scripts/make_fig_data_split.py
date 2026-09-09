"""
Data-split timeline figure (reviewer brief item 1.4). Renders the single
chronological train/validation/test split described in prose in Section
IV-A as a horizontal colored timeline, with the named regime events inside
each window annotated, so the split is inspectable at a glance.

Dates and event labels are transcribed directly from the paper's own
prose (Section IV-A "Split." paragraph); no new claims are introduced.
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SPLITS = [
    ("Train", 2010.0, 2020.0, "#B0B0B0", "2010–2019"),
    ("Validation", 2020.0, 2022.0, "#5EC962", "2020–2021"),
    ("Test", 2022.0, 2026.0, "#440154", "2022–2025"),
]

EVENTS = [
    (2020.2, "Validation", "COVID-19\nonset"),
    (2022.3, "Test", "supply\nshocks"),
    (2023.3, "Test", "rate\nhikes"),
    (2024.6, "Test", "green-transition\ndemand"),
]


def make_figure(output_dir: str = "results/archive_paper1/figures"):
    fig, ax = plt.subplots(figsize=(10, 2.6))

    bar_y, bar_h = 0.0, 1.0
    for name, start, end, color, daterange in SPLITS:
        ax.add_patch(mpatches.Rectangle((start, bar_y), end - start, bar_h,
                                         facecolor=color, edgecolor="black",
                                         linewidth=0.8, alpha=0.85))
        ax.text((start + end) / 2, bar_y + bar_h / 2 + 0.13, name,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if name != "Validation" else "black")
        ax.text((start + end) / 2, bar_y + bar_h / 2 - 0.20, daterange,
                ha="center", va="center", fontsize=10,
                color="white" if name != "Validation" else "black")

    # regime-event annotations, pointing down from above the bar
    for x, split_name, label in EVENTS:
        ax.annotate(label, xy=(x, bar_y + bar_h), xytext=(x, bar_y + bar_h + 0.55),
                    ha="center", va="bottom", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color="black", linewidth=0.7))

    ax.set_xlim(2010, 2026)
    ax.set_ylim(-0.35, bar_y + bar_h + 0.95)
    ax.set_xticks(range(2010, 2027, 2))
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Year")
    ax.set_title("Single Chronological Train / Validation / Test Split", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_data_split_timeline.pdf")
    plt.savefig(f"{output_dir}/fig_data_split_timeline.png")
    plt.close()
    print(f"Saved {output_dir}/fig_data_split_timeline.{{pdf,png}}")


if __name__ == "__main__":
    make_figure()
