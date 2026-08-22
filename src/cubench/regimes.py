"""Regime segmentation — Axis 1 (expanding-quantile volatility terciles, d_regime_lo/mid/hi)
and Axis 2 (5 pre-declared calendar regimes R1-R5, 2015-2025).

Full spec: docs/cubench_implementation_plan.md Section 4.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_REGIMES = {
    "R1_post_supercycle_bust": ("2015-01-01", "2016-12-31"),
    "R2_synchronized_growth_trade_war": ("2017-01-01", "2019-12-31"),
    "R3_covid_shock_rebound": ("2020-01-01", "2020-12-31"),
    "R4_stimulus_rally_2022_shock": ("2021-01-01", "2022-12-31"),
    "R5_energy_transition_destocking": ("2023-01-01", "2025-12-31"),
}


def assign_calendar_regime(dates: np.ndarray) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    out = np.array([None] * len(dates), dtype=object)
    for name, (start, end) in CALENDAR_REGIMES.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        out[mask.to_numpy()] = name
    return out


def assign_vol_tercile(df: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    """Reuses the already-computed d_regime_lo/mid/hi one-hot columns from
    features.parquet (per the plan: 'reuse them, don't recompute'). Returns
    a categorical array aligned to `dates`."""
    sub = df.set_index("date").loc[pd.to_datetime(pd.Series(dates))]
    out = np.array([None] * len(sub), dtype=object)
    if "d_regime_lo" in sub.columns:
        out[sub["d_regime_lo"].to_numpy() == 1] = "vol_lo"
    if "d_regime_mid" in sub.columns:
        out[sub["d_regime_mid"].to_numpy() == 1] = "vol_mid"
    if "d_regime_hi" in sub.columns:
        out[sub["d_regime_hi"].to_numpy() == 1] = "vol_hi"
    return out


def segment_metric(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                    metric_fn, features_df: pd.DataFrame = None) -> dict:
    """Recompute a headline metric within each calendar regime and (if
    features_df supplied) each volatility tercile.

    metric_fn(y_true_slice, y_pred_slice) -> float
    """
    dates = pd.to_datetime(pd.Series(dates))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {"calendar_regimes": {}, "vol_terciles": {}}

    cal = assign_calendar_regime(dates.to_numpy())
    for name in CALENDAR_REGIMES:
        mask = (cal == name)
        n = int(np.sum(mask))
        val = metric_fn(y_true[mask], y_pred[mask]) if n > 1 else float("nan")
        out["calendar_regimes"][name] = {"n": n, "metric": val}

    if features_df is not None:
        vol = assign_vol_tercile(features_df, dates.to_numpy())
        for name in ["vol_lo", "vol_mid", "vol_hi"]:
            mask = (vol == name)
            n = int(np.sum(mask))
            val = metric_fn(y_true[mask], y_pred[mask]) if n > 1 else float("nan")
            out["vol_terciles"][name] = {"n": n, "metric": val}

    return out
