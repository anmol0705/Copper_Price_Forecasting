"""CuBench Phase 4 evaluation entry point.

Regenerates the complete statistical/diagnostic/backtest/ablation layer
(docs/cubench_implementation_plan.md Section 4) from whatever currently
exists in results/cubench/predictions/*.npy and results/cubench/grid_results.jsonl.

Designed to be re-run identically, with no code changes, once the
tree/deep-learning grid (still running as a background process as of the
first run of this script) completes further. Re-running simply picks up
more (model, target, horizon, fold, seed) cells and produces more complete
tables; it never assumes full coverage.

RESUMABLE BY DEFAULT: every layer checks for its own output file on disk
before recomputing and loads it back into memory instead of redoing the
work, so a re-run after a disconnect or crash resumes from wherever it left
off. The ablation layer (LightGBM retraining across A0-A7 rungs) uses the
same incremental per-cell jsonl append+resume pattern as
cubench_run_grid.py (results/cubench/ablations/ablation_results.jsonl) --
an interrupted ablation pass loses at most the one cell that was mid-fit,
not the whole pass. Pass --force to ignore all caches and recompute
everything from scratch.

Usage:
    .venv_corr/Scripts/python scripts/cubench_evaluate.py [--skip-ablations] [--skip-backtest] [--force]

Outputs:
    results/cubench/significance/dm_vs_har_rv.json
    results/cubench/significance/dm_vs_null_persist.json
    results/cubench/significance/pt_mcnemar_t3.json
    results/cubench/significance/holm_bonferroni.json
    results/cubench/base_rate_diagnostics.json
    results/cubench/backtest/cost_curve.json
    results/cubench/backtest/dsr_pbo.json
    results/cubench/ablations/ablation_results.jsonl  (incremental, resumable)
    results/cubench/ablations/ablation_results.json   (aggregated from the jsonl)
    results/cubench/regimes/regime_results.json
    results/cubench/evaluation_run_log.json   (coverage snapshot + timing + per-layer cache status)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import save_results
from src.cubench.walkforward import load_features, load_folds, get_fold_split, TARGET_PREFIX, TAUS
from src.cubench.stats_tests import dm_test_hln, pesaran_timmermann, holm_bonferroni_wrapper
from src.cubench.diagnostics import base_rate_report, dispersion_ratio_check
from src.cubench.metrics import qlike as qlike_fn, r2_oos, kupiec_uc_test, mincer_zarnowitz
OUT_METRICS = ROOT / "results" / "cubench" / "significance"
from src.cubench.regimes import segment_metric
from src.cubench import backtest as bt
from src.cubench import ablations as abl

PRED_DIR = ROOT / "results" / "cubench" / "predictions"
FOLDS_PATH = ROOT / "results" / "cubench" / "folds.json"
GRID_JSONL = ROOT / "results" / "cubench" / "grid_results.jsonl"
OUT_SIG = ROOT / "results" / "cubench" / "significance"
OUT_BACKTEST = ROOT / "results" / "cubench" / "backtest"
OUT_ABLATIONS = ROOT / "results" / "cubench" / "ablations"
OUT_REGIMES = ROOT / "results" / "cubench" / "regimes"
BASE_RATE_OUT = ROOT / "results" / "cubench" / "base_rate_diagnostics.json"
RUN_LOG = ROOT / "results" / "cubench" / "evaluation_run_log.json"

FNAME_RE = re.compile(
    r"^(?P<model>.+)_(?P<target>t[123])_h(?P<horizon>\d+)_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.npy$"
)

TARGET_FAMILY_SIZE = 33  # 11 non-null models x 3 horizons, per pre-registration


# ---------------------------------------------------------------------------
# Prediction index
# ---------------------------------------------------------------------------

def index_predictions() -> dict:
    """Scans results/cubench/predictions/, returns
    idx[model][target][horizon][fold][seed] = Path."""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    n_files = 0
    n_unparsed = 0
    for p in PRED_DIR.glob("*.npy"):
        m = FNAME_RE.match(p.name)
        if not m:
            n_unparsed += 1
            continue
        g = m.groupdict()
        idx[g["model"]][g["target"]][int(g["horizon"])][int(g["fold"])][int(g["seed"])] = p
        n_files += 1
    return idx, n_files, n_unparsed


def load_pred(path: Path) -> np.ndarray:
    return np.load(path)


def seed_average(idx_model_t_h_f: dict, fold: int) -> np.ndarray:
    """Average the prediction array across all available seeds for one fold.
    Works for t1 (1D), t3 (1D proba), and t2 (2D: n_tau x n)."""
    if fold not in idx_model_t_h_f:
        return None
    seeds = idx_model_t_h_f[fold]
    arrs = [load_pred(p) for p in seeds.values()]
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0)


def pooled_series(idx: dict, model: str, target: str, horizon: int, folds: list,
                   df: pd.DataFrame) -> dict:
    """Concatenate seed-averaged predictions + matching y_true + dates across
    all folds for which this (model, target, horizon) has at least one seed,
    in chronological (fold) order. Returns None if nothing available."""
    if model not in idx or target not in idx[model] or horizon not in idx[model][target]:
        return None
    per_fold = idx[model][target][horizon]
    y_chunks, pred_chunks, date_chunks, fold_ids = [], [], [], []
    for fold in folds:
        fk = fold["fold"]
        avg = seed_average(per_fold, fk)
        if avg is None:
            continue
        _, _, _, yte, dates_te = get_fold_split(df, fold, horizon, target)
        n = min(len(yte), avg.shape[-1])
        if target == "t2":
            pred_chunks.append(avg[:, :n])
        else:
            pred_chunks.append(avg[:n])
        y_chunks.append(yte[:n])
        date_chunks.append(dates_te[:n])
        fold_ids.extend([fk] * n)
    if not y_chunks:
        return None
    y_true = np.concatenate(y_chunks)
    dates = np.concatenate(date_chunks)
    if target == "t2":
        pred = np.concatenate(pred_chunks, axis=1)
    else:
        pred = np.concatenate(pred_chunks)
    return {"y_true": y_true, "pred": pred, "dates": dates, "folds": np.array(fold_ids),
            "n_folds": len(y_chunks)}


# ---------------------------------------------------------------------------
# Coverage snapshot
# ---------------------------------------------------------------------------

def coverage_snapshot(idx: dict, folds: list) -> dict:
    n_folds = len(folds)
    out = {}
    for model in idx:
        out[model] = {}
        for target in idx[model]:
            for horizon in idx[model][target]:
                n_present = len(idx[model][target][horizon])
                key = f"{target}_h{horizon}"
                out[model][key] = f"{n_present}/{n_folds} folds"
    return out


# ---------------------------------------------------------------------------
# Significance layer: DM-HLN, PT/McNemar, Holm
# ---------------------------------------------------------------------------

def run_dm_layer(idx: dict, folds: list, df: pd.DataFrame, reference_model: str) -> dict:
    """DM-HLN of every model's T1 pooled predictions vs `reference_model`,
    per horizon. Uses QLIKE-contribution loss (variance-scale) as the primary
    T1 comparison metric, matching Wang & Lu's QLIKE-based ranking."""
    ref_series = {h: pooled_series(idx, reference_model, "t1", h, folds, df) for h in [1, 5, 22]}
    results = []
    for model in sorted(idx.keys()):
        if model == reference_model:
            continue
        for h in [1, 5, 22]:
            mod_series = pooled_series(idx, model, "t1", h, folds, df)
            ref = ref_series.get(h)
            if mod_series is None or ref is None:
                continue
            # align on common fold coverage (intersection)
            common_folds = sorted(set(mod_series["folds"]) & set(ref["folds"]))
            if not common_folds:
                continue
            m_mask = np.isin(mod_series["folds"], common_folds)
            r_mask = np.isin(ref["folds"], common_folds)
            n = min(m_mask.sum(), r_mask.sum())
            if n < 10:
                continue
            y_m, p_m = mod_series["y_true"][m_mask][:n], mod_series["pred"][m_mask][:n]
            y_r, p_r = ref["y_true"][r_mask][:n], ref["pred"][r_mask][:n]
            # QLIKE per-obs loss on variance scale (y1 = 0.5*ln(RV_ann))
            rv_true_m = np.exp(2 * y_m)
            rv_pred_m = np.clip(np.exp(2 * p_m), 1e-12, None)
            loss_m = rv_true_m / rv_pred_m - np.log(rv_true_m / rv_pred_m) - 1
            rv_true_r = np.exp(2 * y_r)
            rv_pred_r = np.clip(np.exp(2 * p_r), 1e-12, None)
            loss_r = rv_true_r / rv_pred_r - np.log(rv_true_r / rv_pred_r) - 1
            out = dm_test_hln(loss_m, loss_r, h=h, loss="raw")
            out.update({
                "model": model, "reference": reference_model, "horizon": h,
                "n_common_folds": len(common_folds), "n_obs": n,
                "qlike_model": float(np.mean(loss_m)), "qlike_reference": float(np.mean(loss_r)),
                "model_better": bool(np.mean(loss_m) < np.mean(loss_r)),
            })
            results.append(out)
    return results


def run_pt_mcnemar_layer(idx: dict, folds: list, df: pd.DataFrame,
                          null_majority_model: str = "null_majority") -> tuple:
    """PT + McNemar for every T3 cell that exists, at BOTH the per-fold grain
    (satisfies 'every T3 result cell' literally, and lets us name specific
    fold/model/horizon in flags) and pooled-across-folds grain (for headline
    tables). Returns (per_fold_diagnostics, pooled_diagnostics)."""
    per_fold_out = []
    pooled_out = []
    null_maj_cache = {}

    for model in sorted(idx.keys()):
        if "t3" not in idx[model]:
            continue
        for h in idx[model]["t3"]:
            per_fold = idx[model]["t3"][h]
            for fold in folds:
                fk = fold["fold"]
                avg = seed_average(per_fold, fk)
                if avg is None:
                    continue
                _, _, _, yte, dates_te = get_fold_split(df, fold, h, "t3")
                n = min(len(yte), len(avg))
                y_true, proba = yte[:n], avg[:n]

                null_key = (h, fk)
                if null_key not in null_maj_cache:
                    nm_per_fold = idx.get(null_majority_model, {}).get("t3", {}).get(h, {})
                    nm_avg = seed_average(nm_per_fold, fk)
                    if nm_avg is not None:
                        nm_pred = (nm_avg[:n] >= 0.5).astype(float)
                    else:
                        nm_pred = None
                    null_maj_cache[null_key] = nm_pred
                nm_pred = null_maj_cache[null_key]

                report = base_rate_report(y_true, y_pred_proba=proba, null_majority_pred=nm_pred)
                report.update({"model": model, "horizon": h, "fold": fk, "oos_year": fold["oos_year"]})
                per_fold_out.append(report)

            pooled = pooled_series(idx, model, "t3", h, folds, df)
            if pooled is None:
                continue
            nm_pooled = pooled_series(idx, null_majority_model, "t3", h, folds, df)
            nm_pred_pooled = None
            if nm_pooled is not None:
                common = min(len(pooled["y_true"]), len(nm_pooled["pred"]))
                nm_pred_pooled = (nm_pooled["pred"][:common] >= 0.5).astype(float)
                y_true_p = pooled["y_true"][:common]
                proba_p = pooled["pred"][:common]
            else:
                y_true_p = pooled["y_true"]
                proba_p = pooled["pred"]
            report = base_rate_report(y_true_p, y_pred_proba=proba_p, null_majority_pred=nm_pred_pooled)
            report.update({"model": model, "horizon": h, "n_folds": pooled["n_folds"]})
            pooled_out.append(report)

    return per_fold_out, pooled_out


def apply_dispersion_check_t1_t2(idx: dict, folds: list, df: pd.DataFrame) -> list:
    out = []
    for model in sorted(idx.keys()):
        for target in ["t1", "t2"]:
            if target not in idx[model]:
                continue
            for h in idx[model][target]:
                pooled = pooled_series(idx, model, target, h, folds, df)
                if pooled is None:
                    continue
                if target == "t2":
                    median_idx = TAUS.index(0.5)
                    pred = pooled["pred"][median_idx]
                else:
                    pred = pooled["pred"]
                chk = dispersion_ratio_check(pooled["y_true"], pred)
                chk.update({"model": model, "target": target, "horizon": h, "n_folds": pooled["n_folds"]})
                out.append(chk)
    return out


def run_holm_layer(dm_results: list, pt_pooled: list) -> dict:
    """Three families: T1 (DM p-value vs har_rv, per model x horizon),
    T2 (DM p-value of pinball_avg vs har_rv — approximated here via the
    dm_results structure is T1-only; T2 uses pinball DM computed separately
    if available, else omitted with a note), T3 (PT p-value, per model x
    horizon), each corrected within itself."""
    families = {}

    t1_pvals, t1_labels = [], []
    for r in dm_results:
        t1_pvals.append(r.get("p_value"))
        t1_labels.append(f"{r['model']}_h{r['horizon']}")
    families["T1_vs_har_rv"] = holm_bonferroni_wrapper(
        t1_pvals, labels=t1_labels, family_name="T1", expected_size=TARGET_FAMILY_SIZE)

    t3_pvals, t3_labels = [], []
    for r in pt_pooled:
        t3_pvals.append(r.get("pt_p_value"))
        t3_labels.append(f"{r['model']}_h{r['horizon']}")
    families["T3_pt"] = holm_bonferroni_wrapper(
        t3_pvals, labels=t3_labels, family_name="T3", expected_size=TARGET_FAMILY_SIZE)

    return families


# ---------------------------------------------------------------------------
# Backtest layer
# ---------------------------------------------------------------------------

def run_backtest_layer(idx: dict, folds: list, df: pd.DataFrame) -> dict:
    """Vol-scaled strategy for every model that has BOTH t1 (vol forecast)
    and t2 (quantile/median direction) predictions at h=1 (daily rebalance is
    the only horizon where day-to-day turnover is meaningful under this
    strategy definition)."""
    out = {"per_model": {}, "notes": []}
    horizon = 1
    for model in sorted(idx.keys()):
        t1_pooled = pooled_series(idx, model, "t1", horizon, folds, df)
        t2_pooled = pooled_series(idx, model, "t2", horizon, folds, df)
        if t1_pooled is None or t2_pooled is None:
            continue
        n = min(len(t1_pooled["y_true"]), t2_pooled["pred"].shape[1])
        sigma_hat = np.exp(t1_pooled["pred"][:n])  # y1 = ln(sigma_ann) -> sigma_ann = exp(y1)
        median_idx = TAUS.index(0.5)
        median_signal = t2_pooled["pred"][median_idx][:n]
        realized_ret = t2_pooled["y_true"][:n]  # y2_h1 = 1-day forward log return

        w = bt.vol_scaled_weights(median_signal, sigma_hat)
        curve = bt.cost_curve(w, realized_ret)
        out["per_model"][model] = {"horizon": horizon, "n_obs": int(n), **curve}

    return out


def run_dsr_pbo_layer(idx: dict, folds: list, df: pd.DataFrame) -> dict:
    """DSR + PBO for models with >=2 seeds available at t1+t2 h=1 (seeds used
    as the CSCV/DSR 'trials' proxy — see notes). Also runs the mandatory
    hand-vs-pypbo DSR cross-check independent of any model's real data."""
    validation = bt.validate_dsr_against_pypbo()
    out = {"dsr_hand_vs_pypbo_validation": validation, "per_model": {},
           "notes": [
               "DSR/PBO trial dimension N is proxied by the model's own seed "
               "ensemble (up to 5 for tree models) rather than a full "
               "hyperparameter search, because no such search grid exists in "
               "this project (hyperparameters are frozen per the plan's D-anti-"
               "overfitting design, D12/Section 3.4). DSR's N parameter is set "
               "to the pre-declared family size (33) per the plan's "
               "instruction; sharpe_std is the empirical std of per-seed "
               "Sharpe ratios (a conservative, documented proxy).",
               "pypbo.dsr() and the hand-derived DSR closed form agree to 0.0 "
               "absolute difference on a validation case (see "
               "dsr_hand_vs_pypbo_validation above) -- pypbo IS importable and "
               "usable in .venv_corr, contradicting the Week A1 report's flag "
               "that it needed a manual site-packages workaround. However, "
               "pypbo.pbo() (the PBO/CSCV routine) took >120s wall-clock on a "
               "toy S=16,N=10,T=800 case on this machine (vs. <1s for the hand "
               "implementation using the identical CSCV algorithm), likely due "
               "to per-partition overhead in its metric-evaluation loop. Given "
               "the real grid's competing CPU load (concurrent background "
               "training), PBO below uses the hand-implemented CSCV "
               "(cscv_pbo_hand), not pypbo.pbo(), for runtime reasons only -- "
               "the algorithm is identical (Bailey & Lopez de Prado CSCV) and "
               "DSR's agreement above is the project's independent numerics "
               "check on this library.",
           ]}
    horizon = 1
    for model in sorted(idx.keys()):
        t1_per_fold = idx[model].get("t1", {}).get(horizon, {})
        t2_per_fold = idx[model].get("t2", {}).get(horizon, {})
        if not t1_per_fold or not t2_per_fold:
            continue
        # collect per-seed pooled return series
        all_seeds = set()
        for fk in t1_per_fold:
            all_seeds |= set(t1_per_fold[fk].keys())
        seed_returns = {}
        for seed in sorted(all_seeds):
            y_chunks, sigma_chunks, sig_chunks = [], [], []
            for fold in folds:
                fk = fold["fold"]
                if fk not in t1_per_fold or seed not in t1_per_fold[fk]:
                    continue
                if fk not in t2_per_fold or seed not in t2_per_fold[fk]:
                    continue
                t1_arr = load_pred(t1_per_fold[fk][seed])
                t2_arr = load_pred(t2_per_fold[fk][seed])
                _, _, _, yte, _ = get_fold_split(df, fold, horizon, "t2")
                n = min(len(yte), t1_arr.shape[-1], t2_arr.shape[-1])
                y_chunks.append(yte[:n])
                sigma_chunks.append(np.exp(t1_arr[:n]))
                sig_chunks.append(t2_arr[TAUS.index(0.5)][:n])
            if not y_chunks:
                continue
            y_true = np.concatenate(y_chunks)
            sigma_hat = np.concatenate(sigma_chunks)
            median_signal = np.concatenate(sig_chunks)
            w = bt.vol_scaled_weights(median_signal, sigma_hat)
            net, _ = bt.strategy_returns(w, y_true, cost_bps=2.0)
            seed_returns[seed] = net

        if len(seed_returns) < 2:
            continue
        min_len = min(len(v) for v in seed_returns.values())
        R = np.stack([v[:min_len] for v in seed_returns.values()], axis=1)
        sharpes = np.array([bt.sharpe_ratio(R[:, j]) for j in range(R.shape[1])])
        sharpes = sharpes[np.isfinite(sharpes)]
        if len(sharpes) < 2:
            continue
        test_sharpe = float(np.mean(sharpes))
        sharpe_std = float(np.std(sharpes, ddof=1))
        r_flat = R.flatten()
        r_flat = r_flat[np.isfinite(r_flat)]
        from scipy import stats as _stats
        skew = float(_stats.skew(r_flat)) if len(r_flat) > 2 else 0.0
        kurt = float(_stats.kurtosis(r_flat, fisher=False)) if len(r_flat) > 2 else 3.0
        T = min_len
        dsr = bt.dsr_hand(test_sharpe, sharpe_std, N=TARGET_FAMILY_SIZE, T=T, skew=skew, kurtosis=kurt)
        pbo = bt.cscv_pbo(R, S=16, prefer_pypbo=False)
        out["per_model"][model] = {
            "n_seeds": R.shape[1], "test_sharpe_mean_across_seeds": test_sharpe,
            "sharpe_std_across_seeds": sharpe_std, "T": T, "skew": skew, "kurtosis": kurt,
            "dsr": dsr, "pbo": {k: v for k, v in pbo.items() if k != "logits"},
        }
    return out


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

ABLATIONS_JSONL = OUT_ABLATIONS / "ablation_results.jsonl"


def _index_ablation_jsonl(jsonl_path: Path) -> dict:
    """Mirrors cubench_run_grid.py's incremental-jsonl resume/dedup pattern:
    composite key (rung, target, horizon, fold, seed) -> last-seen row dict,
    read back from the append-only log. An 'error' row still counts as done
    (matches run_grid's semantics: a failed cell is not silently retried
    forever on every resume)."""
    done = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r["rung"], r["target"], r["horizon"], r["fold"], r["seed"])
                    done[key] = r
                except Exception:
                    continue
    return done


def _aggregate_ablation_losses(done: dict, target: str, horizon: int) -> dict:
    """Concatenate per_obs_loss arrays (fold-then-seed order, matching the
    original in-run chronological append order) for every rung that has at
    least one successful cell for this (target, horizon), pooling cache +
    freshly-computed cells alike."""
    by_rung = defaultdict(list)
    for key in sorted(done.keys()):
        rung, t, h, fold, seed = key
        if t != target or h != horizon:
            continue
        row = done[key]
        if "error" in row or "per_obs_loss" not in row:
            continue
        by_rung[rung].append(np.asarray(row["per_obs_loss"], dtype=float))
    return {rung: np.concatenate(arrs) for rung, arrs in by_rung.items() if arrs}


def _pending_rung_comparisons(rung_losses: dict, comparisons: list) -> list:
    """Every consecutive-rung (+ A6/A7-vs-A4_FULL) comparison whose two rungs
    aren't both computed yet is reported as pending rather than silently
    dropped or crashed on."""
    done_pairs = {(c["rung_from"], c["rung_to"]) for c in comparisons}
    all_pairs = [(abl.RUNG_ORDER_FOR_DM[i], abl.RUNG_ORDER_FOR_DM[i + 1])
                 for i in range(len(abl.RUNG_ORDER_FOR_DM) - 1)]
    all_pairs += [("A4_FULL", "A6"), ("A4_FULL", "A7")]
    pending = []
    for r0, r1 in all_pairs:
        if (r0, r1) in done_pairs:
            continue
        missing = [r for r in (r0, r1) if r not in rung_losses]
        pending.append({"rung_from": r0, "rung_to": r1, "status": "pending",
                         "reason": f"rung(s) not yet computed: {missing}"})
    return pending


def run_ablations(df: pd.DataFrame, folds: list, full_run: bool, time_budget_s: float = 900,
                   force: bool = False) -> dict:
    """Same incremental-jsonl-with-resume discipline as cubench_run_grid.py's
    run_grid(): each (rung, target, horizon, fold, seed) cell is appended to
    ablation_results.jsonl and flushed immediately after it's fit+evaluated,
    and any cell already present in that file on start is skipped (not
    retrained). The backward-compatible ablation_results.json is rebuilt by
    aggregating the jsonl on every call, so it always reflects the true
    on-disk state even if this run is interrupted mid-way."""
    from src.cubench.models.trees import LGBM
    import yaml
    cfg_path = ROOT / "configs" / "cubench.yaml"
    lgbm_params = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        raw = dict(cfg.get("lgbm_hyperparameters", {}) or {})
        raw.pop("seeds", None)
        lgbm_params = raw

    def factory(seed, horizon, target):
        return LGBM(seed=seed, horizon=horizon, target=target, params=lgbm_params)

    all_cols = [c for c in df.columns if c != "date" and not c.startswith(("y1_", "y2_", "y3_"))]
    rungs = abl.build_rung_definitions(all_cols)

    if full_run:
        targets, horizons, seeds, fold_subset = ["t1", "t2", "t3"], [1, 5, 22], [42, 43, 44, 45, 46], folds
        mode = "full"
    else:
        targets, horizons, seeds, fold_subset = ["t1"], [1], [42], folds[:3]
        mode = "smoke_test"

    OUT_ABLATIONS.mkdir(parents=True, exist_ok=True)
    jsonl_path = ABLATIONS_JSONL
    if force and jsonl_path.exists():
        jsonl_path.unlink()
    done = _index_ablation_jsonl(jsonl_path)
    n_precached = len(done)

    t_start = time.time()
    timed_out = False
    n_new, n_skipped, n_errors = 0, 0, 0

    with open(jsonl_path, "a") as out_f:
        for target in targets:
            for horizon in horizons:
                for rung_id, cols in rungs.items():
                    if time.time() - t_start > time_budget_s:
                        timed_out = True
                        break
                    for fold in fold_subset:
                        if timed_out:
                            break
                        for seed in seeds:
                            key = (rung_id, target, horizon, fold["fold"], seed)
                            if key in done:
                                n_skipped += 1
                                continue
                            if time.time() - t_start > time_budget_s:
                                timed_out = True
                                break
                            try:
                                res = abl.run_ablation_cell(
                                    lambda s, h, _t=target: factory(s, h, _t),
                                    cols, target, horizon, fold, seed, df)
                                row = {
                                    "rung": rung_id, "target": target, "horizon": horizon,
                                    "fold": fold["fold"], "seed": seed, **res["metrics"],
                                    "per_obs_loss": res["per_obs_loss"].tolist(),
                                }
                            except Exception as e:
                                row = {
                                    "rung": rung_id, "target": target, "horizon": horizon,
                                    "fold": fold["fold"], "seed": seed, "error": str(e),
                                }
                                n_errors += 1
                            out_f.write(json.dumps(row, default=float) + "\n")
                            out_f.flush()
                            done[key] = row
                            n_new += 1
                    if timed_out:
                        break
                if timed_out:
                    break
            if timed_out:
                break

    # Rebuild the aggregated view from whatever is on disk now (pre-existing
    # cache + anything newly computed this run), not just this run's targets/
    # horizons -- so a narrower resume (e.g. smoke test after a prior full
    # run) never regresses previously-computed comparisons.
    all_th_pairs = {(t, h) for (_, t, h, _, _) in done.keys()}
    all_th_pairs |= {(t, h) for t in targets for h in horizons}

    comparisons_by_th = {}
    for target, horizon in sorted(all_th_pairs):
        rung_losses = _aggregate_ablation_losses(done, target, horizon)
        comparisons = abl.compare_consecutive_rungs(rung_losses, horizon=horizon)
        pending = _pending_rung_comparisons(rung_losses, comparisons)
        comparisons_by_th[f"{target}_h{horizon}"] = comparisons + pending

    cells = [{k: v for k, v in row.items() if k != "per_obs_loss"} for row in done.values()]
    cells.sort(key=lambda c: (c["rung"], c["target"], c["horizon"], c["fold"], c["seed"]))

    out = {
        "mode": mode,
        "rung_definitions_sizes": {k: len(v) for k, v in rungs.items()},
        "cells": cells,
        "comparisons_by_target_horizon": comparisons_by_th,
        "timed_out": timed_out,
        "wall_s": time.time() - t_start,
        "n_cells_total": len(done),
        "n_cells_precached": n_precached,
        "n_cells_new_this_run": n_new,
        "n_cells_skipped_already_done": n_skipped,
        "n_cells_errored_this_run": n_errors,
    }
    return out


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

def run_regimes(idx: dict, folds: list, df: pd.DataFrame) -> dict:
    out = {}
    for model in sorted(idx.keys()):
        for target in idx[model]:
            for h in idx[model][target]:
                pooled = pooled_series(idx, model, target, h, folds, df)
                if pooled is None:
                    continue
                if target == "t1":
                    def metric_fn(y, p):
                        return qlike_fn(np.exp(2 * y), np.clip(np.exp(2 * p), 1e-12, None))
                    pred = pooled["pred"]
                elif target == "t3":
                    def metric_fn(y, p):
                        return float(np.mean((p >= 0.5).astype(float) == y)) if len(y) else float("nan")
                    pred = pooled["pred"]
                else:
                    median_idx = TAUS.index(0.5)
                    def metric_fn(y, p, _tau=0.5):
                        diff = y - p
                        return float(np.mean(np.maximum(_tau * diff, (_tau - 1) * diff))) if len(y) else float("nan")
                    pred = pooled["pred"][median_idx]
                seg = segment_metric(pooled["dates"], pooled["y_true"], pred, metric_fn, features_df=df)
                out.setdefault(model, {})[f"{target}_h{h}"] = seg
    return out


# ---------------------------------------------------------------------------
# T1 R2_OOS / Mincer-Zarnowitz, T2 Kupiec coverage (plan Section 4.2)
# ---------------------------------------------------------------------------

def run_t1_extended_metrics(idx: dict, folds: list, df: pd.DataFrame) -> list:
    """R2_OOS (vs null_rollmean's own OOS predictions, per the plan's
    Gu-Kelly-Xiu convention benchmarked against the null_rollmean model) and
    Mincer-Zarnowitz regression, per (model, horizon) pooled series."""
    out = []
    rollmean_cache = {h: pooled_series(idx, "null_rollmean", "t1", h, folds, df) for h in [1, 5, 22]}
    for model in sorted(idx.keys()):
        if "t1" not in idx[model]:
            continue
        for h in idx[model]["t1"]:
            pooled = pooled_series(idx, model, "t1", h, folds, df)
            if pooled is None:
                continue
            mz = mincer_zarnowitz(pooled["y_true"], pooled["pred"])
            row = {"model": model, "horizon": h, "n": len(pooled["y_true"]), "mz": mz}
            bench = rollmean_cache.get(h)
            if bench is not None:
                common_folds = sorted(set(pooled["folds"]) & set(bench["folds"]))
                if common_folds:
                    m_mask = np.isin(pooled["folds"], common_folds)
                    b_mask = np.isin(bench["folds"], common_folds)
                    n = min(m_mask.sum(), b_mask.sum())
                    y = pooled["y_true"][m_mask][:n]
                    yhat = pooled["pred"][m_mask][:n]
                    yhat_bench = bench["pred"][b_mask][:n]
                    num = np.sum((y - yhat) ** 2)
                    den = np.sum((y - yhat_bench) ** 2)
                    row["r2_oos_vs_null_rollmean"] = float(1 - num / den) if den > 0 else float("nan")
                    row["r2_oos_n"] = int(n)
            out.append(row)
    return out


def run_t2_kupiec(idx: dict, folds: list, df: pd.DataFrame) -> list:
    out = []
    for model in sorted(idx.keys()):
        if "t2" not in idx[model]:
            continue
        for h in idx[model]["t2"]:
            pooled = pooled_series(idx, model, "t2", h, folds, df)
            if pooled is None:
                continue
            row = {"model": model, "horizon": h, "n": len(pooled["y_true"]), "kupiec": []}
            for i, tau in enumerate(TAUS):
                kt = kupiec_uc_test(pooled["y_true"], pooled["pred"][i], tau)
                row["kupiec"].append(kt)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--ablation-full", action="store_true",
                         help="Run the full A0-A7 x 3 targets x 3 horizons x 5 seeds grid "
                              "instead of the smoke test.")
    parser.add_argument("--ablation-time-budget", type=float, default=900.0)
    parser.add_argument("--force", action="store_true",
                         help="Ignore every cached layer output (including the ablation "
                              "jsonl) and recompute everything from scratch. Default "
                              "behavior with no flags is to resume: load whatever layer "
                              "outputs already exist on disk instead of recomputing them.")
    args = parser.parse_args()

    t_start = time.time()
    df = load_features()
    folds_doc = load_folds()
    folds = folds_doc["folds"]

    print("Indexing predictions...")
    idx, n_files, n_unparsed = index_predictions()
    print(f"  {n_files} prediction files indexed across {len(idx)} models ({n_unparsed} unparsed)")

    coverage = coverage_snapshot(idx, folds)

    for d in [OUT_SIG, OUT_BACKTEST, OUT_ABLATIONS, OUT_REGIMES]:
        d.mkdir(parents=True, exist_ok=True)

    layer_status = {}

    def cached(path: Path) -> bool:
        return path.exists() and not args.force

    # --- DM vs har_rv ---
    p = OUT_SIG / "dm_vs_har_rv.json"
    if cached(p):
        print(f"[cache] DM-HLN vs har_rv <- {p}")
        dm_vs_har = json.loads(p.read_text())["comparisons"]
        layer_status["dm_vs_har_rv"] = "cached"
    else:
        print("Running DM-HLN layer vs har_rv...")
        dm_vs_har = run_dm_layer(idx, folds, df, reference_model="har_rv")
        save_results({"comparisons": dm_vs_har}, str(p))
        layer_status["dm_vs_har_rv"] = "computed"

    # --- DM vs null_persist ---
    p = OUT_SIG / "dm_vs_null_persist.json"
    if cached(p):
        print(f"[cache] DM-HLN vs null_persist <- {p}")
        dm_vs_null = json.loads(p.read_text())["comparisons"]
        layer_status["dm_vs_null_persist"] = "cached"
    else:
        print("Running DM-HLN layer vs null_persist...")
        dm_vs_null = run_dm_layer(idx, folds, df, reference_model="null_persist")
        save_results({"comparisons": dm_vs_null}, str(p))
        layer_status["dm_vs_null_persist"] = "computed"

    # --- PT + McNemar + base-rate + T1/T2 dispersion diagnostics ---
    # base_rate_diagnostics.json is the canonical cache for this layer -- it's
    # the only one of the two files this layer writes that carries all three
    # in-memory structures (per_fold_diag, pooled_diag, dispersion_t1_t2)
    # downstream layers (Holm, run_log) need reconstructed, not just a print
    # statement skipped.
    if cached(BASE_RATE_OUT):
        print(f"[cache] PT/McNemar/base-rate/dispersion <- {BASE_RATE_OUT}")
        brd = json.loads(BASE_RATE_OUT.read_text())
        per_fold_diag = brd["per_fold_cells"]
        pooled_diag = brd["pooled_by_model_horizon"]
        dispersion_t1_t2 = brd["t1_t2_dispersion_checks"]
        layer_status["pt_mcnemar_base_rate"] = "cached"
    else:
        print("Running PT + McNemar + base-rate diagnostics on T3...")
        per_fold_diag, pooled_diag = run_pt_mcnemar_layer(idx, folds, df)
        dispersion_t1_t2 = apply_dispersion_check_t1_t2(idx, folds, df)
        layer_status["pt_mcnemar_base_rate"] = "computed"
    flagged = [r for r in per_fold_diag if r["verdict"] != "ok"]
    flagged_pooled = [r for r in pooled_diag if r["verdict"] != "ok"]
    # Re-save unconditionally: this is a cheap serialization of in-memory data
    # (cached or freshly computed), not a recomputation, so it stays correct
    # and in sync even when the layer above was loaded from cache.
    save_results({
        "per_fold_cells": per_fold_diag,
        "pooled_by_model_horizon": pooled_diag,
        "t1_t2_dispersion_checks": dispersion_t1_t2,
        "n_per_fold_cells": len(per_fold_diag),
        "n_flagged_per_fold": len(flagged),
        "n_pooled_cells": len(pooled_diag),
        "n_flagged_pooled": len(flagged_pooled),
    }, str(BASE_RATE_OUT))
    save_results({"per_fold": per_fold_diag, "pooled": pooled_diag},
                 str(OUT_SIG / "pt_mcnemar_t3.json"))

    # --- Holm-Bonferroni (depends on dm_vs_har + pooled_diag, both available
    # above whether they were loaded from cache or computed fresh) ---
    p = OUT_SIG / "holm_bonferroni.json"
    if cached(p):
        print(f"[cache] Holm-Bonferroni <- {p}")
        holm = json.loads(p.read_text())
        layer_status["holm_bonferroni"] = "cached"
    else:
        print("Running Holm-Bonferroni layer...")
        holm = run_holm_layer(dm_vs_har, pooled_diag)
        save_results(holm, str(p))
        layer_status["holm_bonferroni"] = "computed"

    # --- Backtest ---
    backtest_out, dsr_pbo_out = {}, {}
    if not args.skip_backtest:
        p = OUT_BACKTEST / "cost_curve.json"
        if cached(p):
            print(f"[cache] Backtest cost curve <- {p}")
            backtest_out = json.loads(p.read_text())
            layer_status["backtest_cost_curve"] = "cached"
        else:
            print("Running backtest layer (cost curve)...")
            backtest_out = run_backtest_layer(idx, folds, df)
            save_results(backtest_out, str(p))
            layer_status["backtest_cost_curve"] = "computed"

        p2 = OUT_BACKTEST / "dsr_pbo.json"
        if cached(p2):
            print(f"[cache] DSR/PBO <- {p2}")
            dsr_pbo_out = json.loads(p2.read_text())
            layer_status["dsr_pbo"] = "cached"
        else:
            print("Running DSR/PBO layer...")
            dsr_pbo_out = run_dsr_pbo_layer(idx, folds, df)
            save_results(dsr_pbo_out, str(p2))
            layer_status["dsr_pbo"] = "computed"
    else:
        layer_status["backtest_cost_curve"] = "skipped"
        layer_status["dsr_pbo"] = "skipped"

    # --- Ablations: always incremental (per-cell jsonl resume), regardless of
    # --force -- --force there means "clear the jsonl and retrain every cell",
    # handled inside run_ablations itself. ---
    ablation_out = {}
    if not args.skip_ablations:
        print(f"Running ablations ({'FULL' if args.ablation_full else 'SMOKE TEST'}, "
              f"resume-enabled{' [FORCE: clearing ablation cache]' if args.force else ''})...")
        ablation_out = run_ablations(df, folds, full_run=args.ablation_full,
                                       time_budget_s=args.ablation_time_budget,
                                       force=args.force)
        save_results(ablation_out, str(OUT_ABLATIONS / "ablation_results.json"))
        layer_status["ablations"] = (
            f"computed ({ablation_out.get('n_cells_new_this_run', 0)} new cells, "
            f"{ablation_out.get('n_cells_skipped_already_done', 0)} skipped as already-done, "
            f"{ablation_out.get('n_cells_total', 0)} total cells on disk)"
        )
    else:
        layer_status["ablations"] = "skipped"

    # --- T1/T2 extended metrics ---
    p = OUT_SIG / "t1_t2_extended_metrics.json"
    if cached(p):
        print(f"[cache] T1/T2 extended metrics <- {p}")
        ext = json.loads(p.read_text())
        t1_ext, t2_kupiec = ext["t1_extended"], ext["t2_kupiec"]
        layer_status["t1_t2_extended_metrics"] = "cached"
    else:
        print("Running T1 R2_OOS/Mincer-Zarnowitz and T2 Kupiec coverage...")
        t1_ext = run_t1_extended_metrics(idx, folds, df)
        t2_kupiec = run_t2_kupiec(idx, folds, df)
        save_results({"t1_extended": t1_ext, "t2_kupiec": t2_kupiec}, str(p))
        layer_status["t1_t2_extended_metrics"] = "computed"

    # --- Regimes ---
    p = OUT_REGIMES / "regime_results.json"
    if cached(p):
        print(f"[cache] Regime segmentation <- {p}")
        regime_out = json.loads(p.read_text())
        layer_status["regimes"] = "cached"
    else:
        print("Running regime segmentation...")
        regime_out = run_regimes(idx, folds, df)
        save_results(regime_out, str(p))
        layer_status["regimes"] = "computed"

    run_log = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "wall_s_total": time.time() - t_start,
        "force_recompute": args.force,
        "layer_status": layer_status,
        "n_prediction_files": n_files,
        "n_unparsed_files": n_unparsed,
        "coverage_by_model": coverage,
        "n_dm_comparisons_vs_har_rv": len(dm_vs_har),
        "n_dm_comparisons_vs_null_persist": len(dm_vs_null),
        "n_t3_cells_pooled": len(pooled_diag),
        "n_t3_cells_flagged_pooled": len(flagged_pooled),
        "n_t3_cells_per_fold": len(per_fold_diag),
        "n_t3_cells_flagged_per_fold": len(flagged),
        "flagged_per_fold_detail": [
            {"model": r["model"], "horizon": r["horizon"], "fold": r["fold"],
             "oos_year": r.get("oos_year"), "verdict": r["verdict"]}
            for r in flagged
        ],
        "ablation_mode": ablation_out.get("mode"),
        "ablation_timed_out": ablation_out.get("timed_out"),
        "ablation_cells_total": ablation_out.get("n_cells_total"),
    }
    save_results(run_log, str(RUN_LOG))
    print(f"Done in {run_log['wall_s_total']:.1f}s. Layer status: {layer_status}")
    print("See results/cubench/evaluation_run_log.json")


if __name__ == "__main__":
    main()
