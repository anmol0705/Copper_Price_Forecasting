"""Figure 6: the isotropic norm-collapse mechanism, plotted from the real 19-cell
robustness suite (Section~\\ref{sec:robustness} of the manuscript).

Deterministic: every value plotted is read directly from the raw experiment
artifacts under ``paper1/latest_results/`` --- the ``*_results.jsonl`` diagnostic
records and the ``.pt`` checkpoints' ``epoch`` field. Nothing is simulated,
smoothed or hand-entered.

Cell set: the *corrected* 19 (FINALEPOCH_FIX_RESULTS.md Sec. 5.2), i.e. the 16
cells diagnosed from their best-validation checkpoints plus the three cells
(EMD, full_model seed 44, pooled seed 45) re-diagnosed from final-epoch weights.
The two "bonus" re-runs (full_model seed 45 at E=45, pooled seed 44 at E=68) are
re-runs of configurations already represented in the 19 and are excluded, per
FINALEPOCH_FIX_RESULTS.md Sec. 5.1.

Epoch convention: ``E`` = epochs of training embodied in the diagnosed weights.
For a best-validation checkpoint, E = checkpoint["epoch"] - 19 (the trainer
stamps the loop's last epoch index and patience is 20, so the saved index is
epoch - 20; a full training pass precedes each evaluation, hence +1).
For a final-epoch checkpoint, E = diagnostic["final_epoch"] + 1.

Usage:  python scripts/make_fig6_collapse_mechanism.py
Output: results/archive_paper1/figures/fig6_collapse_mechanism.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
LR = ROOT / "latest_results"
OUT = ROOT / "results" / "archive_paper1" / "figures"

PATIENCE = 20

# (axis label, results jsonl, checkpoint dir, checkpoint-name builder from cell_key)
SOURCES = [
    ("k_sweep", LR / "k_sweep" / "k_sweep_results.jsonl",
     LR / "k_sweep" / "checkpoints", lambda ck: f"full_model_K{ck[0]}.pt"),
    ("decomposition", LR / "decomposition" / "decomposition_results.jsonl",
     LR / "decomposition" / "checkpoints", lambda ck: f"full_model_{ck[0]}.pt"),
    ("loss", LR / "loss_comparison" / "loss_comparison_results.jsonl",
     LR / "loss_comparison" / "checkpoints", lambda ck: f"full_model_{ck[0]}.pt"),
    ("multiseed", LR / "multiseed" / "multiseed_results.jsonl",
     LR / "multiseed" / "checkpoints", lambda ck: f"{ck[0]}_seed{ck[1]}.pt"),
]

# The three cells whose best-validation diagnosis was an early-stopping artifact,
# replaced by their final-epoch re-diagnosis.
FIXES = [
    (LR / "decomposition_finalepoch_fix" / "decomposition_results.jsonl"),
    (LR / "multiseed_finalepoch_fix" / "multiseed_results.jsonl"),
]
REPLACED = {("decomposition", "emd"), ("multiseed", "full_model|44"),
            ("multiseed", "pooled_graph_matched_dim|45")}
# Re-runs of configurations already present in the 19 -- not cells (Sec. 5.1).
BONUS = {"full_model|45", "pooled_graph_matched_dim|44"}


def cell_id(ck) -> str:
    return "|".join(str(x) for x in ck)


def read_diag(rec):
    """Mean cosine-to-init and mean RMS-ratio-to-init over the cell's bands."""
    mv = rec["diagnostic"]["adjacency_movement"]
    bands = mv["bands"]
    cos = mv["mean_cosine_similarity"]
    # Recompute the band mean rather than trusting a stored scalar.
    cos_recomputed = float(np.mean([b["cosine_similarity"] for b in bands]))
    assert abs(cos - cos_recomputed) < 1e-6, (cos, cos_recomputed)
    rms = float(np.mean([b["rms_ratio_to_init"] for b in bands]))
    grad = rec["diagnostic"]["gradient_magnitude"]["emb_grad_norm_mean"]
    return cos, rms, grad


def load_cells():
    cells = []          # the corrected 19
    prior = []          # the pre-correction (1-2 epoch) snapshots of the 3 fixed cells
    for axis, jsonl, ckdir, ckname in SOURCES:
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ck = rec["cell_key"]
            cid = cell_id(ck)
            epoch = torch.load(ckdir / ckname(ck), map_location="cpu",
                               weights_only=False)["epoch"]
            E = epoch - PATIENCE + 1
            cos, rms, grad = read_diag(rec)
            row = dict(axis=axis, cell=cid, E=E, cos=cos, rms=rms, grad=grad,
                       final_epoch=False)
            if (axis, cid) in REPLACED:
                prior.append(row)          # superseded snapshot, kept for the arrows
            else:
                cells.append(row)

    for jsonl in FIXES:
        axis = "decomposition" if "decomposition" in jsonl.parts[-2] else "multiseed"
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = cell_id(rec["cell_key"])
            if cid in BONUS:
                continue
            d = rec["diagnostic"]
            assert d["diagnosed_final_epoch"] is True, cid
            E = d["final_epoch"] + 1
            cos, rms, grad = read_diag(rec)
            cells.append(dict(axis=axis, cell=cid, E=E, cos=cos, rms=rms,
                              grad=grad, final_epoch=True))
    return cells, prior


STYLE = {
    "k_sweep":       ("#0072B2", "o", "Band count $K \\in \\{3,5,7,9\\}$"),
    "decomposition": ("#D55E00", "s", "Decomposition (VMD / EMD)"),
    "loss":          ("#009E73", "^", "Training loss (MSE / MAE / Huber)"),
    "multiseed":     ("#7F3F98", "D", "Seeds 42--46, both graph variants"),
}


def main():
    cells, prior = load_cells()
    assert len(cells) == 19, len(cells)
    assert len(prior) == 3, len(prior)
    cells.sort(key=lambda r: r["E"])

    E = np.array([c["E"] for c in cells], float)
    cos = np.array([c["cos"] for c in cells], float)
    rms = np.array([c["rms"] for c in cells], float)

    # Reported statistics, recomputed here so the figure and the text agree.
    from scipy.stats import spearmanr
    rho_cos = spearmanr(E, cos)
    rho_rms = spearmanr(E, rms)
    slope, icpt = np.polyfit(E, cos, 1)
    r2 = 1.0 - np.sum((cos - (icpt + slope * E)) ** 2) / np.sum((cos - cos.mean()) ** 2)
    print(f"n={len(cells)}  E range {E.min():.0f}-{E.max():.0f}")
    print(f"spearman(E, cos)      = {rho_cos.statistic:+.4f}  p={rho_cos.pvalue:.2e}")
    print(f"spearman(E, rms/init) = {rho_rms.statistic:+.4f}  p={rho_rms.pvalue:.2e}")
    print(f"cos fit = {icpt:.5f} {slope:+.6f} E   R^2={r2:.4f}")
    print(f"rms/init range {rms.max():.4g} -> {rms.min():.4g} "
          f"({np.log10(rms.max() / rms.min()):.1f} orders of magnitude)")
    print(f"cos range {cos.max():.4f} -> {cos.min():.4f} "
          f"(factor {cos.max() / cos.min():.2f})")
    for c in cells:
        print(f"  {c['axis']:<14s} {c['cell']:<28s} E={c['E']:>3d}  "
              f"cos={c['cos']:.5f}  rms/init={c['rms']:.4g}  grad={c['grad']:.3g}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.3))
    fig.subplots_adjust(wspace=0.30)
    orders = np.log10(rms.max() / rms.min())
    factor = cos.max() / cos.min()

    # ---- Panel A: magnitude ------------------------------------------------
    axL.set_yscale("log")
    for c in cells:
        col, mk, _ = STYLE[c["axis"]]
        axL.plot(c["E"], c["rms"], mk, color=col, ms=6.5, mew=0.8,
                 mec="white", zorder=3)
    axL.set_xlabel("Epochs of training embodied in the diagnosed checkpoint, $E$")
    axL.set_ylabel(r"Embedding RMS $/$ initialization RMS")
    axL.set_title(f"(a) Magnitude: {orders:.1f} orders of magnitude of decay",
                  loc="left", fontsize=11)
    axL.axhline(1.0, color="0.55", lw=1.0, ls=":", zorder=1)
    axL.text(38.5, 1.45, "initialization", color="0.35", fontsize=8.5,
             va="center", ha="right")
    axL.axhline(0.5, color="0.55", lw=1.0, ls="--", zorder=1)
    axL.text(38.5, 0.40, "collapse threshold (0.5)", color="0.35", fontsize=8.5,
             va="top", ha="right")
    axL.set_ylim(2e-8, 4.0)
    axL.grid(True, which="major", alpha=0.25, lw=0.6)

    # ---- Panel B: direction ------------------------------------------------
    for c in cells:
        col, mk, _ = STYLE[c["axis"]]
        axR.plot(c["E"], c["cos"], mk, color=col, ms=6.5, mew=0.8,
                 mec="white", zorder=3)
    xs = np.linspace(0, 38, 50)
    axR.plot(xs, icpt + slope * xs, color="0.35", lw=1.1, zorder=2,
             label=rf"$\cos = {icpt:.3f} - {abs(slope):.4f}\,E$   ($R^2={r2:.3f}$)")
    axR.set_xlabel("Epochs of training embodied in the diagnosed checkpoint, $E$")
    axR.set_ylabel("Cosine similarity to own random initialization")
    axR.set_title(f"(b) Direction: a single factor of {factor:.1f} over the same span",
                  loc="left", fontsize=11)
    axR.set_ylim(0.0, 1.06)
    axR.axhline(1.0, color="0.55", lw=1.0, ls=":", zorder=1)
    axR.grid(True, which="major", alpha=0.25, lw=0.6)
    axR.legend(loc="lower left", frameon=False, fontsize=8.5)

    # ---- The three within-run pairs: same run, two snapshots ---------------
    for p in prior:
        match = [c for c in cells if c["cell"] == p["cell"] and c["final_epoch"]]
        assert len(match) == 1, p["cell"]
        q = match[0]
        for ax, key in ((axL, "rms"), (axR, "cos")):
            ax.annotate("", xy=(q["E"], q[key]), xytext=(p["E"], p[key]),
                        arrowprops=dict(arrowstyle="-|>", color="0.45",
                                        lw=0.9, ls=(0, (4, 2)),
                                        shrinkA=3, shrinkB=4), zorder=2)
            ax.plot(p["E"], p[key], "o", ms=6.0, mfc="none", mec="0.35",
                    mew=1.0, zorder=3)

    axL.annotate("dashed arrows: the same three runs,\n"
                 "from their superseded $E=1$--$2$ snapshot\n"
                 "to their final-epoch re-diagnosis",
                 xy=(2.6, 0.66), xytext=(6.0, 2.0e-6), fontsize=8.2, color="0.30",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7))

    handles = [plt.Line2D([], [], color=c, marker=m, ls="none", ms=6.5,
                          mew=0.8, mec="white", label=lbl)
               for c, m, lbl in (STYLE[k] for k in
                                 ("k_sweep", "decomposition", "loss", "multiseed"))]
    handles.append(plt.Line2D([], [], color="0.35", marker="o", ls="none",
                              ms=6.0, mfc="none", mew=1.0,
                              label="Superseded 1--2-epoch snapshot of the same run"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.13), fontsize=9)

    fig.tight_layout(w_pad=3.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig6_collapse_mechanism.{ext}")
    print(f"\nwrote {OUT / 'fig6_collapse_mechanism.pdf'}")


if __name__ == "__main__":
    main()
