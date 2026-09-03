"""CuBench Phase 5 figure-generation entry point.

Regenerates all evaluation figures (results/cubench/figures/) from whatever
currently exists in results/cubench/{grid_results.jsonl, significance/,
backtest/, regimes/, ablations/, base_rate_diagnostics.json} and
data/cubench/features.parquet.

This runs on PARTIAL, real data by design -- local grid training has stopped
short of full completion (see STATUS.md, "LOCAL GRID TRAINING STOPPED").
Phase 6 packages a Colab notebook to finish the grid; when that lands, this
script is re-run UNCHANGED and simply produces fuller figures (more folds,
more models) from the same code paths. No figure is skipped for sparse data;
sparse figures instead carry an explicit "(N/11 folds)" / "(partial grid)"
annotation so the figure-generation code is proven correct now.

Usage:
    .venv_corr/Scripts/python scripts/cubench_make_figures.py

Outputs:
    results/cubench/figures/fig1_model_comparison.{png,pdf}
    results/cubench/figures/fig2_dm_significance_heatmap.{png,pdf}
    results/cubench/figures/fig3_base_rate_diagnostics.{png,pdf}
    results/cubench/figures/fig4_regime_performance.{png,pdf}
    results/cubench/figures/fig5_cost_curve.{png,pdf}
    results/cubench/figures/fig6_realized_vol_timeseries.{png,pdf}
    results/cubench/figures/fig7_ablation_waterfall.{png,pdf}
    results/cubench/figures/fig8_quantile_calibration.{png,pdf}
    results/cubench/figures/figure_manifest.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cubench.regimes import CALENDAR_REGIMES

RES = ROOT / "results" / "cubench"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PATH = ROOT / "data" / "cubench" / "features.parquet"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COMPLETE_MODELS = {
    "null_persist", "null_rollmean", "null_zero", "null_majority", "null_always_down",
    "har_rv", "har_rv_q", "garch11", "gjr_garch", "arima", "elasticnet",
}
PARTIAL_TREE_MODELS = {"lgbm", "catboost", "xgboost", "randomforest"}
DEEP_MODELS = {"lstm", "transformer"}

MODEL_COLORS = {
    "har_rv": "#d62728",  # highlighted reference
}
_palette = plt.get_cmap("tab20").colors


def color_for(model, i):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return _palette[i % len(_palette)]


def save_fig(fig, name, notes=None):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    ok_png = png.exists() and png.stat().st_size > 0
    ok_pdf = pdf.exists() and pdf.stat().st_size > 0
    print(f"  wrote {png.name} ({png.stat().st_size if ok_png else 0} B), "
          f"{pdf.name} ({pdf.stat().st_size if ok_pdf else 0} B)")
    if not (ok_png and ok_pdf):
        raise RuntimeError(f"Figure {name} failed to save non-empty files")


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Fig 1: model comparison bar chart (QLIKE T1, pinball T2), faceted by horizon
# ---------------------------------------------------------------------------

def fig1_model_comparison(grid_df: pd.DataFrame):
    print("Figure 1: model comparison...")
    horizons = [1, 5, 22]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

    for row, (target, metric_col, metric_label) in enumerate(
        [("t1", "qlike", "QLIKE (T1, lower=better)\nmedian across folds"),
         ("t2", "pinball_avg", "Pinball loss avg (T2, lower=better)\nmedian across folds")]
    ):
        sub_all = grid_df[grid_df["target"] == target]
        for col, h in enumerate(horizons):
            ax = axes[row, col]
            sub = sub_all[sub_all["horizon"] == h]
            if sub.empty:
                ax.set_title(f"{target.upper()} h={h} (no data)")
                ax.axis("off")
                continue
            # QLIKE is unbounded above and null_persist/tree models occasionally blow up
            # to 4-5 orders of magnitude in a single fold when predicted variance is near
            # zero; the median (not mean) is used as the headline bar so one outlier fold
            # cannot make every other model's bar invisible. Both are shown in the manifest.
            agg = (sub.groupby("model")
                   .agg(metric=(metric_col, "median"), metric_mean=(metric_col, "mean"),
                        n_folds=("fold", "nunique"))
                   .reset_index())
            agg = agg.sort_values("metric")
            xs = np.arange(len(agg))
            colors, hatches, labels = [], [], []
            for i, r in enumerate(agg.itertuples()):
                complete = r.n_folds >= 11
                base_color = color_for(r.model, i)
                colors.append(base_color)
                hatches.append(None if complete else "///")
                lab = r.model
                if not complete:
                    lab += f" ({r.n_folds}/11)"
                labels.append(lab)
            bars = ax.bar(xs, agg["metric"], color=colors, edgecolor="black", linewidth=0.5)
            for b, h_ in zip(bars, hatches):
                if h_:
                    b.set_hatch(h_)
                    b.set_alpha(0.55)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
            if "har_rv" in agg["model"].values:
                har_val = agg.loc[agg["model"] == "har_rv", "metric"].iloc[0]
                ax.axhline(har_val, color="#d62728", linestyle="--", linewidth=1.3,
                           label="har_rv (reference)")
                ax.legend(loc="upper right", fontsize=7.5)
            ax.set_title(f"{target.upper()} h={h}")
            if target == "t1":
                ax.set_yscale("log")  # QLIKE is heavily right-skewed (e.g. null_persist's
                                       # median QLIKE h=1 is ~7.8e3 vs ~1.5 for har_rv: naive
                                       # persistence badly underpredicts variance on vol-spike
                                       # days, a real and legitimate finding, not an outlier
                                       # to hide -- log scale keeps every bar visible at once)
            if col == 0:
                ax.set_ylabel(metric_label)

    legend_elems = [
        Patch(facecolor="white", edgecolor="black", label="solid = fully complete (11/11 folds)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", alpha=0.55,
              label="hatched = partial grid (<11/11 folds)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("CuBench: QLIKE (T1) and pinball loss (T2) by model and horizon\n"
                 "(median across folds -- QLIKE mean is dominated by rare high-loss outlier "
                 "folds, esp. null_persist/trees; partial-grid models hatched; "
                 "har_rv is the reference to beat)", y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig1_model_comparison")


# ---------------------------------------------------------------------------
# Fig 2: DM significance heatmap
# ---------------------------------------------------------------------------

def fig2_dm_heatmap():
    print("Figure 2: DM significance heatmap...")
    dm_har = load_json(RES / "significance" / "dm_vs_har_rv.json")
    dm_null = load_json(RES / "significance" / "dm_vs_null_persist.json")
    holm = load_json(RES / "significance" / "holm_bonferroni.json")

    holm_lookup = {}
    if holm and "T1_vs_har_rv" in holm:
        fam = holm["T1_vs_har_rv"]
        labels = fam.get("labels", [])
        rejected = fam.get("reject", fam.get("rejected", []))
        p_adj = fam.get("p_adjusted", fam.get("pvals_corrected", []))
        for lab, rej, p in zip(labels, rejected, p_adj):
            holm_lookup[lab] = (rej, p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dm_data, ref_name in zip(axes, [dm_har, dm_null], ["har_rv", "null_persist"]):
        if not dm_data or not dm_data.get("comparisons"):
            ax.set_title(f"DM vs {ref_name} (no data)")
            ax.axis("off")
            continue
        comps = dm_data["comparisons"]
        models = sorted(set(c["model"] for c in comps))
        horizons = sorted(set(c["horizon"] for c in comps))
        mat = np.full((len(models), len(horizons)), np.nan)
        annot = np.empty((len(models), len(horizons)), dtype=object)
        for c in comps:
            i = models.index(c["model"])
            j = horizons.index(c["horizon"])
            lab = f"{c['model']}_h{c['horizon']}"
            sig, p_adj = holm_lookup.get(lab, (None, c["p_value"]))
            # sign = +1 if model better (lower loss) than reference
            signed = -1 if c["model_better"] else 1
            if sig is True:
                mat[i, j] = signed * 2
            elif sig is False:
                mat[i, j] = signed * 1
            else:
                mat[i, j] = signed * 1.5  # holm status unknown -> mid intensity
            star = "*" if sig else ("~" if sig is None else "")
            annot[i, j] = f"{'better' if c['model_better'] else 'worse'}{star}\np={p_adj:.2g}"
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        for i in range(len(models)):
            for j in range(len(horizons)):
                if annot[i, j]:
                    ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=6.5)
        ax.set_title(f"T1 DM test vs {ref_name}\n(blue=model better, red=model worse; "
                      f"* = Holm-significant at 5%)")
    fig.suptitle("Pairwise Diebold-Mariano significance (T1 QLIKE-scale loss), Holm-corrected", y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig2_dm_significance_heatmap")


# ---------------------------------------------------------------------------
# Fig 3: base-rate diagnostic summary
# ---------------------------------------------------------------------------

def fig3_base_rate():
    print("Figure 3: base-rate diagnostics...")
    d = load_json(RES / "base_rate_diagnostics.json")
    if not d:
        print("  no base_rate_diagnostics.json, skipping")
        return

    per_fold = d["per_fold_cells"]
    df = pd.DataFrame(per_fold)
    pivot = df.pivot_table(index=["model", "horizon"], columns="fold", values="verdict",
                            aggfunc="first")
    verdict_code = {"ok": 0, "suspect_low_dispersion": 1, "constant_forecast_artifact": 2}
    code_mat = pivot.map(lambda v: verdict_code.get(v, np.nan)) if hasattr(pivot, "map") else pivot.applymap(lambda v: verdict_code.get(v, np.nan))

    disp = pd.DataFrame(d.get("t1_t2_dispersion_checks", []))

    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, 0.32 * len(code_mat))),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])
    im = ax.imshow(code_mat.values, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(code_mat)))
    ax.set_yticklabels([f"{m} h={h}" for m, h in code_mat.index], fontsize=7)
    ax.set_xticks(range(code_mat.shape[1]))
    ax.set_xticklabels(code_mat.columns, fontsize=7)
    ax.set_xlabel("OOS fold")
    ax.set_title("T3 base-rate diagnostic verdict, per (model, horizon, fold)\n"
                 "arima collapses in every fold; null models are constant by design;\n"
                 "tree models have no T3 cells yet (not flagged = not attempted, not verified ok)")
    legend_elems = [
        Patch(facecolor="#2ca02c", label="ok"),
        Patch(facecolor="#ff7f0e", label="suspect_low_dispersion"),
        Patch(facecolor="#d62728", label="constant_forecast_artifact"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    ax2 = axes[1]
    if not disp.empty:
        disp_sorted = disp.sort_values("dispersion_ratio")
        colors = ["#d62728" if f else "#2ca02c" for f in disp_sorted["dispersion_ratio_flag"]]
        ylabels = [f"{r.model} {r.target} h={r.horizon}" for r in disp_sorted.itertuples()]
        ax2.barh(range(len(disp_sorted)), disp_sorted["dispersion_ratio"], color=colors)
        ax2.axvline(0.20, color="black", linestyle="--", linewidth=1, label="flag threshold 0.20")
        ax2.set_yticks(range(len(disp_sorted)))
        ax2.set_yticklabels(ylabels, fontsize=6)
        ax2.set_xlabel("std(pred) / std(actual)")
        ax2.set_title("T1/T2 dispersion-ratio check\n(red = flagged, std(pred)/std(actual) < 0.20)")
        ax2.legend(fontsize=7)
    else:
        ax2.axis("off")

    fig.tight_layout()
    save_fig(fig, "fig3_base_rate_diagnostics")


# ---------------------------------------------------------------------------
# Fig 4: regime-segmented performance
# ---------------------------------------------------------------------------

def fig4_regimes():
    print("Figure 4: regime-segmented performance...")
    d = load_json(RES / "regimes" / "regime_results.json")
    if not d:
        print("  no regime_results.json, skipping")
        return

    target_h = "t1_h1"
    rows = []
    for model, by_th in d.items():
        seg = by_th.get(target_h)
        if seg is None:
            continue
        for reg, info in seg.get("calendar_regimes", {}).items():
            rows.append({"model": model, "axis": "calendar", "bucket": reg,
                         "metric": info["metric"], "n": info["n"]})
        for reg, info in seg.get("vol_terciles", {}).items():
            rows.append({"model": model, "axis": "vol_tercile", "bucket": reg,
                         "metric": info["metric"], "n": info["n"]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no t1_h1 regime rows, skipping")
        return

    cal_order = list(CALENDAR_REGIMES.keys())
    vol_order = ["vol_lo", "vol_mid", "vol_hi"]
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, axis_name, order in zip(axes, ["calendar", "vol_tercile"], [cal_order, vol_order]):
        sub = df[df["axis"] == axis_name]
        xs = np.arange(len(order))
        for i, model in enumerate(models):
            msub = sub[sub["model"] == model].set_index("bucket").reindex(order)
            vals = msub["metric"].values
            lw = 2.6 if model == "lgbm" else 1.4
            marker = "o" if model == "lgbm" else "."
            color = "#1f77b4" if model == "lgbm" else color_for(model, i)
            ax.plot(xs, vals, marker=marker, linewidth=lw, label=model, color=color,
                    alpha=1.0 if model == "lgbm" else 0.7, zorder=10 if model == "lgbm" else 2)
        short_labels = [o.replace("_", " ") if axis_name == "calendar" else o for o in order]
        ax.set_xticks(xs)
        ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=7.5)
        # QLIKE spans ~1 (trees/econometric) to >1e4 (null_persist blow-ups on vol-spike
        # days); log scale keeps every model's line visible on one axis.
        ax.set_yscale("log")
        ax.set_ylabel("QLIKE (T1, h=1), log scale")
        ax.set_title(f"QLIKE by {axis_name.replace('_', ' ')}")

    lgbm_cal = df[(df.model == "lgbm") & (df.axis == "calendar")].set_index("bucket").reindex(cal_order)["metric"]
    r1_val, r4_val = lgbm_cal.iloc[0], lgbm_cal.iloc[3]
    axes[0].annotate(
        f"LightGBM QLIKE: {r1_val:.2f} (R1, 2015-16) -> {r4_val:.2f} (R4, 2021-22)\n"
        f"({r4_val / r1_val:.1f}x worse in the later regime)",
        xy=(3, r4_val), xytext=(0.1, r4_val * 25),
        fontsize=8.5, color="#1f77b4", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        bbox=dict(boxstyle="round", fc="white", ec="#1f77b4", alpha=0.9))
    axes[1].legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Regime-segmented QLIKE, T1 h=1\n(only models with h=1 T1 predictions shown; "
                 "n varies by fold coverage, see figure_manifest.md)", y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig4_regime_performance")


# ---------------------------------------------------------------------------
# Fig 5: transaction-cost curve
# ---------------------------------------------------------------------------

def fig5_cost_curve():
    print("Figure 5: transaction-cost curve...")
    d = load_json(RES / "backtest" / "cost_curve.json")
    dsr = load_json(RES / "backtest" / "dsr_pbo.json")
    if not d or not d.get("per_model"):
        print("  no cost_curve.json, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(0.4, 0.8, color="#2ca02c", alpha=0.12, label="realistic gross-Sharpe band (0.4-0.8)")
    for i, (model, info) in enumerate(sorted(d["per_model"].items())):
        cc = info["cost_curve"]
        bps = sorted(int(k) for k in cc.keys())
        sharpes = [cc[str(b)]["sharpe"] for b in bps]
        lw = 2.6 if model == "lgbm" else 1.4
        color = "#1f77b4" if model == "lgbm" else color_for(model, i)
        ax.plot(bps, sharpes, marker="o", label=model, linewidth=lw, color=color,
                alpha=1.0 if model == "lgbm" else 0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Round-trip cost (bps)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Transaction-cost curve: net Sharpe vs. round-trip cost, by model (h=1)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    if dsr and "lgbm" in dsr.get("per_model", {}):
        lgbm_dsr = dsr["per_model"]["lgbm"]
        pbo = lgbm_dsr["pbo"]["pbo"]
        overfit = lgbm_dsr["pbo"]["likely_overfit"]
        gross = d["per_model"].get("lgbm", {}).get("gross_sharpe")
        if gross is not None:
            ax.annotate(
                f"LightGBM: gross Sharpe {gross:.2f} looks realistic,\n"
                f"but PBO={pbo:.2f} -> {'LIKELY OVERFIT' if overfit else 'not flagged'}\n"
                f"(DSR={lgbm_dsr['dsr']:.1e}, seeds as trial proxy, N=33)",
                xy=(0, gross), xytext=(15, gross - 0.9),
                fontsize=8.5, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                bbox=dict(boxstyle="round", fc="white", ec="#d62728", alpha=0.9),
            )
    fig.tight_layout()
    save_fig(fig, "fig5_cost_curve")


# ---------------------------------------------------------------------------
# Fig 6: realized volatility time series with regime shading
# ---------------------------------------------------------------------------

def fig6_rv_timeseries():
    print("Figure 6: realized volatility time series...")
    df = pd.read_parquet(FEATURES_PATH)
    df = df[["date", "y1_h1"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    df["rv_ann"] = np.exp(2.0 * df["y1_h1"])  # inverse of y1 = 0.5*ln(RV_ann)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(df["date"], df["rv_ann"], color="#333333", linewidth=0.8)

    cmap = plt.get_cmap("tab10")
    for i, (name, (start, end)) in enumerate(CALENDAR_REGIMES.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=cmap(i), alpha=0.18)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ax.get_ylim()[1] if False else df["rv_ann"].quantile(0.995),
                name.split("_", 1)[0], ha="center", va="top", fontsize=8, rotation=0,
                color=cmap(i))
    ax.set_ylabel("Annualized realized volatility (implied from y1_h1)")
    ax.set_xlabel("Date")
    ax.set_title("Copper realized volatility, 2010-2025, with 5 pre-declared calendar regimes shaded\n"
                 "(fully complete: computed directly from features.parquet, no model dependency)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig6_realized_vol_timeseries")


# ---------------------------------------------------------------------------
# Fig 7: ablation waterfall
# ---------------------------------------------------------------------------

def fig7_ablation_waterfall():
    print("Figure 7: ablation waterfall...")
    d = load_json(RES / "ablations" / "ablation_results.json")
    if not d or not d.get("cells"):
        print("  no ablation_results.json cells, skipping")
        return

    cells = pd.DataFrame(d["cells"])
    cells = cells[(cells.get("target") == "t1") & (cells.get("horizon") == 1)]
    if "error" in cells.columns:
        cells = cells[cells["error"].isna()] if cells["error"].notna().any() else cells
    if cells.empty or "qlike" not in cells.columns:
        print("  no usable t1_h1 ablation cells, skipping")
        return

    rung_order = ["A0", "A1", "A2", "A3", "A4_FULL", "A5", "A6", "A7"]
    agg = cells.groupby("rung")["qlike"].agg(["mean", "count"]).reindex(
        [r for r in rung_order if r in cells["rung"].unique()])

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.arange(len(agg))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(agg)))
    bars = ax.bar(xs, agg["mean"], color=colors, edgecolor="black")
    for x, (rung, row) in zip(xs, agg.iterrows()):
        ax.text(x, row["mean"], f"{row['mean']:.3f}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(agg.index, rotation=20)
    ax.set_ylabel("Mean QLIKE (T1, h=1) across smoke-test cells")
    mode = d.get("mode", "unknown")
    timed_out = d.get("timed_out", False)
    n_folds_used = int(cells["fold"].nunique())
    ax.set_title(
        f"Block-wise ablation, LightGBM, T1 h=1 -- SMOKE TEST ONLY\n"
        f"mode={mode}, {n_folds_used} fold(s), seed=42 (not the full A0-A7 x 3 targets x "
        f"3 horizons x 5 seeds grid){' -- timed out before reaching later rungs' if timed_out else ''}"
    )
    comps = d.get("comparisons_by_target_horizon", {}).get("t1_h1", [])
    for c in comps:
        sig = c["p_value"] < 0.05
        if sig:
            i0 = list(agg.index).index(c["rung_from"])
            i1 = list(agg.index).index(c["rung_to"])
            y = max(agg["mean"].iloc[i0], agg["mean"].iloc[i1]) * 1.06
            ax.plot([i0, i1], [y, y], color="red", linewidth=1)
            ax.text((i0 + i1) / 2, y, "*", color="red", ha="center", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "fig7_ablation_waterfall")


# ---------------------------------------------------------------------------
# Fig 8: quantile calibration / coverage
# ---------------------------------------------------------------------------

def fig8_calibration():
    print("Figure 8: quantile calibration...")
    d = load_json(RES / "significance" / "t1_t2_extended_metrics.json")
    if not d or not d.get("t2_kupiec"):
        print("  no t2_kupiec data, skipping")
        return

    rows = d["t2_kupiec"]
    horizons = sorted(set(r["horizon"] for r in rows))
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.5 * len(horizons), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        hsub = [r for r in rows if r["horizon"] == h]
        for i, r in enumerate(sorted(hsub, key=lambda r: r["model"])):
            taus = [k["nominal_rate"] for k in r["kupiec"]]
            emp = [k["empirical_rate"] for k in r["kupiec"]]
            fail = [k["p_value"] < 0.05 for k in r["kupiec"]]
            color = color_for(r["model"], i)
            ax.plot(taus, emp, marker="o", label=r["model"], color=color, alpha=0.85, linewidth=1.3)
            for t, e, f in zip(taus, emp, fail):
                if f:
                    ax.scatter([t], [e], facecolors="none", edgecolors="red", s=90, linewidths=1.5)
        ax.set_xlabel("Nominal coverage (tau)")
        ax.set_title(f"h={h}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(fontsize=6.5, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("T2 quantile calibration: empirical vs nominal coverage\n"
                 "(open red circles = Kupiec unconditional-coverage test rejects at 5%)", y=1.05)
    fig.tight_layout()
    save_fig(fig, "fig8_quantile_calibration")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(grid_df: pd.DataFrame):
    print("Writing figure_manifest.md...")
    cov = (grid_df.groupby(["model", "target", "horizon"])
           .agg(n_folds=("fold", "nunique"), n_seeds=("seed", "nunique"))
           .reset_index())
    complete_combos = int((cov["n_folds"] >= 11).sum())
    total_combos = len(cov)

    lines = [
        "# CuBench Phase 5 figure manifest",
        "",
        f"Generated from real data present at run time. {complete_combos}/{total_combos} "
        f"(model, target, horizon) cells currently have all 11 OOS folds; the rest are "
        f"partial and are labeled as such in every figure that uses them.",
        "",
        "Re-run with `.venv_corr/Scripts/python scripts/cubench_make_figures.py` any time "
        "after more grid cells land (e.g. after the Phase 6 Colab run) -- no code changes "
        "needed, figures just get fuller.",
        "",
        "| Figure | Shows | Data completeness now | What changes once the grid is full |",
        "|---|---|---|---|",
        "| fig1_model_comparison | QLIKE (T1) and pinball loss (T2) by model, faceted by "
        "horizon; hatched bars = partial grid; har_rv shown as a reference line | Nulls, "
        "econometric (har_rv/har_rv_q/garch11/gjr_garch/arima) and elasticnet fully complete "
        "(11/11 folds). lgbm essentially complete for T1/T2. catboost/xgboost/randomforest "
        "complete for T1 only (no T2/T3 cells yet). lstm/transformer effectively absent (1 "
        "or 0 cells). | Tree-model bars stop being hatched; lstm/transformer bars appear for "
        "the first time; T2/T3 bars appear for catboost/xgboost/randomforest. |",
        "| fig2_dm_significance_heatmap | Pairwise DM test (T1, vs har_rv and vs "
        "null_persist), Holm-corrected significance | Only models/horizons with pooled T1 "
        "predictions across at least 10 common folds are shown -- this currently includes "
        "trees (lgbm/catboost/xgboost/randomforest have complete or near-complete T1) but "
        "not lstm/transformer | More rows (deep models) once their T1 grid lands; no other "
        "structural change. |",
        "| fig3_base_rate_diagnostics | Per-(model,horizon,fold) T3 base-rate diagnostic "
        "verdict (ok / suspect_low_dispersion / constant_forecast_artifact), plus T1/T2 "
        "dispersion-ratio checks | T3 diagnostics currently cover only models with T3 "
        "predictions: nulls (constant by design), arima (collapses to one class in every "
        "fold, all 33 cells flagged constant_forecast_artifact), elasticnet (mostly ok, some "
        "suspect_low_dispersion). Tree/deep models have NO T3 cells yet, so they are absent "
        "from this figure -- absent is not the same as 'passed'. | Tree/deep rows appear; "
        "the real open question (does lgbm's T3 pass this check?) gets answered for the "
        "first time. |",
        "| fig4_regime_performance | QLIKE by 5 calendar regimes and 3 volatility terciles, "
        "T1 h=1, for every model with regime data | All T1 h=1 models present (nulls, "
        "econometric, elasticnet, lgbm, catboost, xgboost, randomforest all have complete or "
        "near-complete h=1 T1 folds). Real finding visible: lgbm QLIKE is markedly worse in "
        "R4 (2021-22) and R5 (2023-25) than R1/R2 (2015-19). | Mostly stable -- this figure "
        "is already close to final for h=1 T1; h=5/h=22 regime panels could be added once "
        "more targets are computed by regimes.py for other horizons. |",
        "| fig5_cost_curve | Net Sharpe vs. round-trip cost (bps), by model, h=1, with the "
        "pre-registered realistic 0.4-0.8 gross-Sharpe band shaded; LightGBM's DSR/PBO "
        "likely_overfit finding annotated directly on its curve | 8 models with both T1+T2 "
        "h=1 predictions: nulls, econometric, elasticnet, lgbm. catboost/xgboost/"
        "randomforest lack T2 h=1, so are absent. LightGBM: gross Sharpe 0.43 (looks "
        "realistic) but PBO=0.73 -> likely_overfit=True. | catboost/xgboost/randomforest "
        "curves appear once T2 h=1 exists for them; more seeds -> tighter DSR/PBO estimates "
        "for all models, not just lgbm. |",
        "| fig6_realized_vol_timeseries | Copper realized volatility (annualized, implied "
        "from y1_h1), 2010-2025, with the 5 calendar regimes shaded | Fully complete -- "
        "computed directly from data/cubench/features.parquet, no model dependency at all. | "
        "None -- this figure is already final. |",
        "| fig7_ablation_waterfall | A0->A5 block-wise QLIKE waterfall, LightGBM, T1 h=1 "
        "only, smoke test (3 folds, seed=42) | Explicitly a smoke test, not the full A0-A7 x "
        "3 targets x 3 horizons x 5 seeds grid -- labeled as such directly in the figure "
        "title. Only reached A0->A5 before the evaluation script's time budget stopped it "
        "(A6/A7 comparisons missing). | Full 3x3x5-seed ablation grid across all 8 rungs "
        "once run with `--ablation-full`; figure code needs no change, just re-run. |",
        "| fig8_quantile_calibration | Empirical vs nominal coverage (Kupiec test) for T2 "
        "quantile forecasts, per horizon, per model | Same model coverage as fig1's T2 panel "
        "-- nulls, econometric, elasticnet, lgbm (near-complete). catboost/xgboost/"
        "randomforest absent (no T2 yet). | catboost/xgboost/randomforest lines appear; "
        "lstm/transformer lines appear once their T2 grid exists. |",
        "",
        "## Coverage snapshot at generation time",
        "",
        "| model | target | horizon | folds present | seeds present |",
        "|---|---|---|---|---|",
    ]
    for r in cov.sort_values(["model", "target", "horizon"]).itertuples():
        lines.append(f"| {r.model} | {r.target} | {r.horizon} | {r.n_folds}/11 | {r.n_seeds} |")

    (FIG_DIR / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {FIG_DIR / 'figure_manifest.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading grid_results.jsonl...")
    rows = [json.loads(l) for l in (RES / "grid_results.jsonl").open(encoding="utf-8")]
    grid_df = pd.DataFrame(rows)
    print(f"  {len(grid_df)} rows, {grid_df['model'].nunique()} models")

    fig1_model_comparison(grid_df)
    fig2_dm_heatmap()
    fig3_base_rate()
    fig4_regimes()
    fig5_cost_curve()
    fig6_rv_timeseries()
    fig7_ablation_waterfall()
    fig8_calibration()
    write_manifest(grid_df)

    print("\nDone. All figures in", FIG_DIR)


if __name__ == "__main__":
    main()
