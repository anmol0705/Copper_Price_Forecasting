"""Post-hoc, zero-GPU analysis of VMD-MFGNN / baseline predictions.

WHY THIS SCRIPT EXISTS
-----------------------
A raw DA (directional accuracy) number like "48.5%" or a negative R^2 is
UNINTERPRETABLE on its own for financial return prediction. Two things are
missing from `results/all_results.json` as currently produced by
`run_all_experiments()`/`run_baseline()` (src/trainer.py):

  1. A null/benchmark DA to compare against. 50% is NOT the correct null if
     the true return series has a skewed up/down day frequency (e.g. if
     copper went up 53% of test days, a coin-flip model would show ~53% DA
     purely from the base rate, not skill). The correct null is the
     unconditional up-day frequency of the TEST set's true returns.
  2. A significance test on directional accuracy (Pesaran-Timmermann), and a
     Holm-Bonferroni correction across the many DM/PT comparisons already
     being run (results/significance_table.json alone has 6 baselines x 4
     horizons = 24 comparisons; adding PT tests roughly doubles that count),
     since testing that many hypotheses without correction inflates the
     false-positive rate.

This gap was flagged as M3/M4 in the independent quant-researcher review
referenced in STATUS.md's "Independent Expert Review" section.

This script is 100% POST-HOC and read-only w.r.t. the training pipeline: it
loads only what `run_baseline()` / `run_all_experiments()` already save to
`results/predictions/{model}_{h}.npy` and `{model}_{h}_true.npy` (raw 1D
NumPy arrays: model predictions and matching ground-truth forward log
returns for the test split -- see src/trainer.py's `run_baseline()`, lines
~308-340, and the VMD-MFGNN save block in `run_all_experiments()`, lines
~424-437). It does NOT import torch, does NOT touch src/trainer.py,
src/utils.py, src/data_pipeline.py, or any model code, and does NOT
retrain or re-run inference on anything. It can be run entirely on CPU,
locally or pasted into a Colab cell, once `results/` exists.

WHAT THIS SCRIPT ADDS (not already in the pipeline)
-----------------------------------------------------
1. Random-walk / zero-forecast null baseline per horizon: RMSE/MAE of a
   constant-zero prediction (== std / mean_abs of true test returns), plus
   the unconditional up-day frequency (the correct DA null).
2. Historical-mean baseline: prediction = the TRAINING-period mean return.
   IMPORTANT CAVEAT (read before trusting this baseline): the pipeline does
   NOT currently persist the training-set true targets anywhere under
   results/ -- only TEST predictions/true values are saved. This script
   therefore looks for an OPTIONAL `--train-targets-dir` pointing at a
   directory of `{h}_true.npy` files (one array of training-period true
   returns per horizon) that the human must save separately from Colab,
   e.g.:
       for h in horizons:
           ys = np.concatenate([y[:, i].numpy() for _, y in
                                 data["train_loader"]])   # i = index of h
           np.save(f"{train_targets_dir}/{h}_true.npy", ys)
   If that directory/files are not supplied, this script explicitly SKIPS
   the historical-mean baseline for the affected horizons (reports
   `null` with a `"reason"` string) rather than silently approximating it
   from the test set itself, which would leak future information into a
   supposedly training-only baseline.
3. Every discovered model's DA reported against the CORRECT null
   (`da - up_day_frequency`), not just against 50.
4. Pesaran-Timmermann (1992) test of directional predictive ability,
   implemented from scratch in NumPy (see `pesaran_timmermann_test()`).
5. Holm-Bonferroni step-down correction applied jointly across every DM-test
   entry already in `results/significance_table.json` (if present) AND
   every PT-test entry computed here, saved as `results/holm_bonferroni.json`
   (also embedded in the main report).
6. Lightweight economic sanity check: a sign(prediction) long/short strategy,
   annualized return/vol/Sharpe/max-drawdown/hit-rate, and the breakeven
   per-round-trip transaction cost (in bps) at which Sharpe hits zero,
   turnover-adjusted (see `economic_sanity_check()` docstring for the exact
   assumptions -- this is a lightweight, illustrative check only, NOT a
   backtest, per the M4 recommendation in the original quant review).

USAGE
-----
    python scripts/post_hoc_analysis.py --results-dir results
    python scripts/post_hoc_analysis.py --results-dir results --train-targets-dir results/train_targets
    python scripts/post_hoc_analysis.py --results-dir results --output results/post_hoc_analysis.json

Or import and call directly (e.g. from a Colab cell):
    from post_hoc_analysis import run_post_hoc_analysis
    report = run_post_hoc_analysis("results")
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# The 4 locked horizons for this project (see vmd-mfgnn-protocol/SKILL.md).
# Kept as a literal default (not re-derived from a config file) so this
# script has zero dependency on the training pipeline's config loader;
# override with --horizons if a different set was actually used.
DEFAULT_HORIZONS = [1, 5, 10, 22]

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------
# 1. Discovery / loading
# --------------------------------------------------------------------------

def discover_predictions(pred_dir: Path, horizons: List[int]) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Scan `pred_dir` for `{name}_{h}.npy` / `{name}_{h}_true.npy` pairs.

    Returns {horizon: {model_name: (pred_array, true_array)}}.

    Model names may themselves contain underscores (e.g.
    "no_vmd_raw_price_matched_dim"), so we can't split on "_" -- instead,
    for each known horizon h we glob the literal suffix "_{h}.npy" (which
    never collides with "_{h}_true.npy", since that ends in "_true.npy" not
    "_{h}.npy") and strip that suffix to recover the model name.
    """
    out: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {h: {} for h in horizons}
    if not pred_dir.exists():
        logger.warning(f"Predictions directory {pred_dir} does not exist.")
        return out

    for h in horizons:
        suffix = f"_{h}.npy"
        for pred_path in sorted(pred_dir.glob(f"*{suffix}")):
            name = pred_path.name[: -len(suffix)]
            true_path = pred_dir / f"{name}_{h}_true.npy"
            if not true_path.exists():
                logger.warning(f"Found {pred_path.name} but no matching "
                                f"{true_path.name}; skipping.")
                continue
            pred = np.load(pred_path).flatten()
            true = np.load(true_path).flatten()
            if pred.shape != true.shape:
                logger.warning(f"{name} h={h}: pred shape {pred.shape} != "
                                f"true shape {true.shape}; skipping.")
                continue
            out[h][name] = (pred, true)
    return out


def load_train_targets(train_targets_dir: Optional[Path], horizons: List[int]) -> Dict[int, Optional[np.ndarray]]:
    """Load optional training-period true-return arrays for the
    historical-mean baseline. See module docstring, item 2, for the exact
    file convention expected: `{train_targets_dir}/{h}_true.npy`.
    """
    out: Dict[int, Optional[np.ndarray]] = {h: None for h in horizons}
    if train_targets_dir is None:
        return out
    train_targets_dir = Path(train_targets_dir)
    for h in horizons:
        p = train_targets_dir / f"{h}_true.npy"
        if p.exists():
            out[h] = np.load(p).flatten()
        else:
            logger.warning(f"--train-targets-dir given but {p} not found; "
                            f"historical-mean baseline will be skipped for h={h}.")
    return out


# --------------------------------------------------------------------------
# 2. Null baselines
# --------------------------------------------------------------------------

def random_walk_baseline(true: np.ndarray) -> Dict[str, float]:
    """Zero-forecast ("no change" / martingale) baseline for one horizon.

    Prediction is a constant 0 for every test sample, so:
      RMSE = sqrt(mean(true^2))   (== std(true) when mean(true) ~= 0)
      MAE  = mean(|true|)
    DA against a constant-zero prediction is not meaningful (sign(0) is
    ambiguous / not a directional call), so instead we report the
    unconditional up-day frequency of the true returns -- the correct null
    for interpreting any model's DA (see module docstring).
    """
    true = np.asarray(true).flatten()
    up_day_frequency = float((true > 0).mean() * 100.0)
    down_day_frequency = float((true < 0).mean() * 100.0)
    zero_day_frequency = float((true == 0).mean() * 100.0)
    rmse = float(np.sqrt(np.mean(true ** 2)))
    mae = float(np.mean(np.abs(true)))
    return {
        "rmse": rmse,
        "mae": mae,
        "up_day_frequency_pct": up_day_frequency,
        "down_day_frequency_pct": down_day_frequency,
        "zero_day_frequency_pct": zero_day_frequency,
        "n": int(len(true)),
        "note": "DA is not meaningful for a constant-zero prediction; "
                "up_day_frequency_pct is the correct DA null to compare "
                "every real model's DA against.",
    }


def historical_mean_baseline(test_true: np.ndarray, train_true: Optional[np.ndarray]) -> Dict[str, float]:
    """Historical-mean baseline: prediction = mean of TRAINING-period true
    returns, applied constantly to every test sample. See module docstring
    item 2 for why `train_true` may be None (not persisted by the pipeline
    by default).
    """
    if train_true is None:
        return {
            "available": False,
            "reason": "Training-period true targets are not saved by "
                      "run_baseline()/run_all_experiments() -- only test "
                      "predictions/true values land under "
                      "results/predictions/. Pass --train-targets-dir "
                      "pointing at pre-saved {h}_true.npy files (training "
                      "split) to enable this baseline. See this script's "
                      "module docstring for the exact snippet to save them "
                      "from a Colab session.",
        }
    test_true = np.asarray(test_true).flatten()
    train_mean = float(np.mean(train_true))
    pred = np.full_like(test_true, train_mean, dtype=float)
    rmse = float(np.sqrt(np.mean((test_true - pred) ** 2)))
    mae = float(np.mean(np.abs(test_true - pred)))
    da = float(np.mean(np.sign(test_true) == np.sign(pred)) * 100.0) if train_mean != 0 else float("nan")
    return {
        "available": True,
        "train_mean_return": train_mean,
        "n_train": int(len(train_true)),
        "rmse": rmse,
        "mae": mae,
        "da": da,
    }


# --------------------------------------------------------------------------
# 3. Pesaran-Timmermann directional-accuracy test
# --------------------------------------------------------------------------

def pesaran_timmermann_test(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Pesaran & Timmermann (1992) test of directional predictive ability.

    H0: the sign of `pred` and the sign of `true` are statistically
    independent (i.e. the model has no genuine directional skill beyond
    what the two series' own up/down base rates would produce by chance).
    H1: they co-move more (or less) often than independence would predict.

    Let n = sample size, p_y = P(true > 0), p_x = P(pred > 0) (empirical
    frequencies), p_hat = observed proportion of matching signs (== DA/100).
    Under independence, the expected proportion of sign matches is
        p_star = p_y*p_x + (1-p_y)*(1-p_x)
    The PT statistic is
        PT = (p_hat - p_star) / sqrt(var(p_hat) - var(p_star))
    which is asymptotically standard normal under H0. See Pesaran, M.H. and
    Timmermann, A. (1992), "A Simple Nonparametric Test of Predictive
    Performance", Journal of Business & Economic Statistics, 10(4), 461-465.
    """
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    n = len(true)
    if n < 2:
        return {"pt_stat": np.nan, "p_value": np.nan, "p_hat": np.nan,
                "p_star": np.nan, "n": n}

    ind_true = (true > 0).astype(float)
    ind_pred = (pred > 0).astype(float)
    p_y = ind_true.mean()
    p_x = ind_pred.mean()
    p_hat = float(np.mean(ind_true == ind_pred))
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)

    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = (
        ((2 * p_y - 1) ** 2) * (p_x * (1 - p_x)) / n
        + ((2 * p_x - 1) ** 2) * (p_y * (1 - p_y)) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / (n ** 2)
    )
    denom = var_p_hat - var_p_star
    if not np.isfinite(denom) or denom <= 0:
        # Degenerate case (e.g. p_y or p_x at/near 0 or 1, or a tiny sample)
        # -- the asymptotic variance estimate is not usable.
        return {"pt_stat": np.nan, "p_value": np.nan, "p_hat": p_hat * 100.0,
                "p_star": p_star * 100.0, "n": n}

    pt_stat = float((p_hat - p_star) / np.sqrt(denom))
    p_value = float(2 * (1 - stats.norm.cdf(np.abs(pt_stat))))
    return {"pt_stat": pt_stat, "p_value": p_value, "p_hat": p_hat * 100.0,
            "p_star": p_star * 100.0, "n": n}


# --------------------------------------------------------------------------
# 4. Holm-Bonferroni correction
# --------------------------------------------------------------------------

def holm_bonferroni_correction(named_pvalues: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Holm's step-down multiple-comparison correction.

    `named_pvalues`: {comparison_key: raw_p_value}. NaN p-values are passed
    through unmodified (excluded from the correction family, since they
    represent a degenerate test, not a real hypothesis to adjust).

    Returns {comparison_key: {"p_raw":..., "rank":..., "p_holm":...,
    "significant_at_0.05": bool}}, sorted ascending by raw p-value internally
    (rank 1 = smallest p-value); `p_holm` is the standard Holm-adjusted
    p-value: p_holm_(i) = max_{j<=i}( (m - j + 1) * p_(j) ), enforced
    monotone non-decreasing and clipped to [0, 1], where m is the number of
    non-NaN comparisons in the family.
    """
    valid = [(k, v) for k, v in named_pvalues.items() if v is not None and np.isfinite(v)]
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)

    result: Dict[str, Dict[str, float]] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(valid, start=1):
        adj = (m - i + 1) * p
        running_max = max(running_max, adj)
        p_holm = float(min(running_max, 1.0))
        result[k] = {
            "p_raw": float(p),
            "rank": i,
            "p_holm": p_holm,
            "significant_at_0.05": bool(p_holm < 0.05),
        }
    # NaN / missing entries: pass through, not part of the corrected family.
    for k, v in named_pvalues.items():
        if k not in result:
            result[k] = {
                "p_raw": None if v is None else float(v),
                "rank": None,
                "p_holm": None,
                "significant_at_0.05": False,
                "note": "excluded from Holm family (NaN/degenerate p-value)",
            }
    result["_meta"] = {"family_size_m": m}
    return result


# --------------------------------------------------------------------------
# 5. Economic sanity check
# --------------------------------------------------------------------------

def economic_sanity_check(pred: np.ndarray, true: np.ndarray, horizon: int) -> Dict[str, float]:
    """Lightweight sign(prediction) long/short strategy sanity check.

    NOT a backtest -- a standalone, self-contained calculation directly from
    the saved pred/true arrays (per the task's M4-style lightweight-check
    recommendation), with the following simplifying assumptions stated
    explicitly:

      - Position each period is sign(pred) in {-1, 0, +1} (0 if pred == 0
        exactly). Strategy per-period return = position * true (true is the
        realized forward log return over `horizon` days for that sample).
      - Test samples are treated as one "period" each for annualization,
        using periods_per_year = TRADING_DAYS_PER_YEAR / horizon (i.e.
        assumes non-overlapping horizon-day rebalancing; the real test set
        is actually a rolling/overlapping window, so this annualization is
        an approximation -- flagged, not hidden).
      - No transaction costs in the primary Sharpe/return calculation.
      - Turnover = fraction of periods where the position differs from the
        previous period's position (a trade occurs). Breakeven per-round-trip
        cost (bps) is the cost level at which the turnover-adjusted mean
        return hits exactly zero:
            mean_return - turnover * (cost_bps / 10000) = 0
            => breakeven_bps = mean_return / turnover * 10000
        (if turnover == 0, breakeven is undefined -- reported as NaN).
    """
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    position = np.sign(pred)
    strategy_returns = position * true

    n = len(strategy_returns)
    periods_per_year = TRADING_DAYS_PER_YEAR / horizon

    mean_ret = float(np.mean(strategy_returns))
    std_ret = float(np.std(strategy_returns, ddof=1)) if n > 1 else float("nan")

    ann_return = mean_ret * periods_per_year
    ann_vol = std_ret * np.sqrt(periods_per_year) if np.isfinite(std_ret) else float("nan")
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if (std_ret and std_ret > 0) else float("nan")

    equity_curve = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    max_drawdown = float(np.min(drawdown)) if n > 0 else float("nan")

    hit_rate = float(np.mean(strategy_returns > 0) * 100.0) if n > 0 else float("nan")

    turnover = float(np.mean(np.abs(np.diff(position)) > 0)) if n > 1 else 0.0
    if turnover > 0:
        breakeven_bps = float(mean_ret / turnover * 10000.0)
    else:
        breakeven_bps = float("nan")

    return {
        "n": int(n),
        "mean_return_per_period": mean_ret,
        "std_return_per_period": std_ret,
        "periods_per_year_assumed": periods_per_year,
        "annualized_return_pct": ann_return * 100.0,
        "annualized_vol_pct": ann_vol * 100.0 if np.isfinite(ann_vol) else float("nan"),
        "sharpe_ratio_annualized": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate_pct": hit_rate,
        "turnover_fraction": turnover,
        "breakeven_cost_bps_per_round_trip": breakeven_bps,
        "note": "Lightweight sanity check only, no transaction costs in the "
                "primary Sharpe figure, non-overlapping-period annualization "
                "approximation -- see economic_sanity_check() docstring for "
                "full assumptions.",
    }


# --------------------------------------------------------------------------
# 6. Orchestration
# --------------------------------------------------------------------------

def run_post_hoc_analysis(results_dir: str = "results",
                           train_targets_dir: Optional[str] = None,
                           horizons: Optional[List[int]] = None,
                           output_path: Optional[str] = None) -> Dict:
    """Run the full post-hoc analysis and return the report dict (also
    printed to console and saved as JSON to `output_path`).
    """
    results_dir = Path(results_dir)
    pred_dir = results_dir / "predictions"
    horizons = horizons or DEFAULT_HORIZONS
    output_path = Path(output_path) if output_path else results_dir / "post_hoc_analysis.json"

    by_horizon = discover_predictions(pred_dir, horizons)
    train_targets = load_train_targets(Path(train_targets_dir) if train_targets_dir else None, horizons)

    report: Dict = {
        "horizons": horizons,
        "null_baselines": {},
        "models": {},
        "holm_bonferroni": {},
    }

    all_pvalues: Dict[str, float] = {}

    # Load existing DM-test table if present, fold its p-values into the
    # Holm-Bonferroni family too.
    sig_table_path = results_dir / "significance_table.json"
    dm_table = {}
    if sig_table_path.exists():
        with open(sig_table_path) as f:
            dm_table = json.load(f)
        for key, entry in dm_table.items():
            p = entry.get("p_value")
            if p is not None:
                all_pvalues[f"DM::{key}"] = p
    else:
        logger.warning(f"{sig_table_path} not found; Holm-Bonferroni family "
                        f"will only include this script's PT-test p-values.")

    print("\n" + "=" * 100)
    print("POST-HOC ANALYSIS: null baselines, PT-test, economic sanity check")
    print("=" * 100)

    for h in horizons:
        models_h = by_horizon.get(h, {})
        if not models_h:
            logger.warning(f"No prediction files found for horizon {h} in {pred_dir}")
            continue

        # True test-return array should be identical across models for a
        # given horizon (same test split) -- take the first one found, but
        # warn if they actually disagree (a real sanity check in itself).
        any_true = next(iter(models_h.values()))[1]
        for name, (_, true_arr) in models_h.items():
            if not np.array_equal(true_arr, any_true):
                logger.warning(f"h={h}: {name}'s true-return array differs "
                                f"from other models' -- test sets may not "
                                f"be aligned across models!")

        rw = random_walk_baseline(any_true)
        hm = historical_mean_baseline(any_true, train_targets.get(h))
        report["null_baselines"][f"h{h}"] = {
            "random_walk_zero_forecast": rw,
            "historical_mean": hm,
        }

        print(f"\n--- Horizon {h} ---")
        print(f"  Null baselines: RW RMSE={rw['rmse']:.6f} MAE={rw['mae']:.6f} "
              f"| up_day_freq={rw['up_day_frequency_pct']:.1f}% "
              f"down_day_freq={rw['down_day_frequency_pct']:.1f}% (n={rw['n']})")
        if hm.get("available"):
            print(f"  Historical-mean baseline: RMSE={hm['rmse']:.6f} "
                  f"MAE={hm['mae']:.6f} DA={hm['da']:.1f}% "
                  f"(train_mean_return={hm['train_mean_return']:.6f}, n_train={hm['n_train']})")
        else:
            print(f"  Historical-mean baseline: SKIPPED ({hm['reason'][:80]}...)")

        for name, (pred, true) in models_h.items():
            da = float(np.mean(np.sign(true) == np.sign(pred)) * 100.0)
            da_vs_null = da - rw["up_day_frequency_pct"]
            da_vs_50 = da - 50.0
            pt = pesaran_timmermann_test(pred, true)
            econ = economic_sanity_check(pred, true, h)

            pt_key = f"PT::{name}_h{h}"
            all_pvalues[pt_key] = pt["p_value"]

            report["models"].setdefault(name, {})[f"h{h}"] = {
                "da_pct": da,
                "da_minus_up_day_frequency": da_vs_null,
                "da_minus_50": da_vs_50,
                "pesaran_timmermann": pt,
                "economic_sanity_check": econ,
            }

            pt_str = (f"PT={pt['pt_stat']:.3f} p={pt['p_value']:.4f}"
                      if np.isfinite(pt["pt_stat"]) else "PT=NaN (degenerate)")
            sharpe_str = (f"{econ['sharpe_ratio_annualized']:.3f}"
                          if np.isfinite(econ["sharpe_ratio_annualized"]) else "NaN")
            print(f"  {name:<32} DA={da:6.2f}%  DA-null={da_vs_null:+6.2f}pp  "
                  f"DA-50={da_vs_50:+6.2f}pp  {pt_str}  "
                  f"Sharpe={sharpe_str}  breakeven={econ['breakeven_cost_bps_per_round_trip']:.1f}bps")

    holm = holm_bonferroni_correction(all_pvalues)
    report["holm_bonferroni"] = holm
    report["dm_test_table_loaded"] = dm_table

    n_sig_raw = sum(1 for k, v in all_pvalues.items() if np.isfinite(v) and v < 0.05)
    n_sig_holm = sum(1 for k, v in holm.items() if isinstance(v, dict) and v.get("significant_at_0.05"))
    print(f"\n--- Holm-Bonferroni correction across {holm.get('_meta', {}).get('family_size_m', 0)} "
          f"DM+PT comparisons ---")
    print(f"  Significant at raw p<0.05: {n_sig_raw}")
    print(f"  Significant at Holm-adjusted p<0.05: {n_sig_holm}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nFull report saved to {output_path}")
    print("=" * 100)

    return report


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return obj
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-GPU post-hoc analysis of VMD-MFGNN/baseline "
                    "predictions: null baselines, Pesaran-Timmermann DA "
                    "test, Holm-Bonferroni correction, economic sanity check."
    )
    parser.add_argument("--results-dir", type=str, default="results",
                         help="Path to the results/ directory produced by "
                             "the Colab run (must contain a predictions/ "
                             "subdirectory). Default: 'results'.")
    parser.add_argument("--train-targets-dir", type=str, default=None,
                         help="Optional path to a directory of "
                             "{horizon}_true.npy training-period true-return "
                             "arrays, needed to enable the historical-mean "
                             "baseline (not persisted by the pipeline by "
                             "default -- see module docstring).")
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                         help=f"Horizons to analyze. Default: {DEFAULT_HORIZONS}")
    parser.add_argument("--output", type=str, default=None,
                         help="Output JSON path. Default: "
                             "<results-dir>/post_hoc_analysis.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_post_hoc_analysis(
        results_dir=args.results_dir,
        train_targets_dir=args.train_targets_dir,
        horizons=args.horizons,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
