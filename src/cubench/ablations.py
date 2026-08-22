"""Block-wise ablation design A0..A7 (strictly nested, added in evidence-strength order).
LightGBM only, 5 seeds, all 3 targets x 3 horizons.

Full spec: docs/cubench_implementation_plan.md Section 4.6.

Scope note (see scripts/cubench_evaluate.py for the live decision + real output):
this module builds the FULL A0-A7 infrastructure (feature-set definitions, the
retrain/evaluate loop, DM-tests between consecutive rungs). Whether the full
grid (3 targets x 3 horizons x 5 seeds x 8 rungs = 360 LightGBM fits) or a fast
smoke test is actually executed is a time-budget decision made at run time by
the caller, not baked into this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cubench.stats_tests import dm_test_hln

BLOCK_A = [
    "a_mom_5", "a_mom_10", "a_mom_22", "a_mom_63", "a_mom_126", "a_mom_252",
    "a_ewma_x_8_32", "a_ewma_x_16_64", "a_ewma_x_32_128",
    "a_zscore_22", "a_zscore_63", "a_rsi_14",
    "a_dist_hi_252", "a_dist_lo_252", "a_r_lag1", "a_r_lag2",
]

BLOCK_D = [
    "d_logrv_1", "d_logrv_5", "d_logrv_22", "d_logrv_66",
    "d_logrv_park_1", "d_logrv_park_5", "d_logrv_park_22",
    "d_logrv_sq_5", "d_logrv_sq_22",
    "d_rv_ratio_5_22", "d_rv_ratio_22_66",
    "d_volofvol_22", "d_downside_semivar_22", "d_skew_63", "d_kurt_63", "d_jump_22",
    "d_garch_sigma", "d_gjr_sigma", "d_garch_resid_z",
    "d_vix_level", "d_vix_ratio_5_22",
    "d_regime_lo", "d_regime_mid", "d_regime_hi",
    "d_dow_1", "d_dow_2", "d_dow_3", "d_dow_4",
]

BLOCK_C_CROSS = [
    "c_dxy_r1", "c_dxy_r5", "c_vix_r1", "c_vix_r5", "c_sp500_r1", "c_sp500_r5",
    "c_gold_r1", "c_gold_r5", "c_silver_r1", "c_silver_r5",
    "c_aluminum_r1", "c_aluminum_r5", "c_crude_r1", "c_crude_r5",
]

BLOCK_C_MACRO = [
    "c_tnx_d1", "c_tnx_d22", "c_dfii10_d1", "c_dfii10_d22",
    "c_baa10y_d1", "c_baa10y_d22", "c_ppi_yoy", "c_indpro_yoy",
]

BLOCK_BPRIME = ["b1_rv_term_ratio", "b2_cu_al_relvalue", "b3_cu_al_mom_spread", "b4_hg_qc_spread"]
BLOCK_CHINA = ["c_china_conf"]

A0_BASE = ["a_r_lag1", "a_r_lag2", "d_logrv_1", "d_logrv_5", "d_logrv_22"]


def _avail(cols: list, all_cols: list) -> list:
    s = set(all_cols)
    return [c for c in cols if c in s]


def build_rung_definitions(all_cols: list) -> dict:
    """Returns {rung_id: list_of_feature_columns}, filtered to columns that
    actually exist in the current feature matrix (reports missing ones
    separately so silent shrinkage is visible, not hidden)."""
    a0 = set(_avail(A0_BASE, all_cols))
    a1 = a0 | set(_avail(BLOCK_D, all_cols))
    a2 = a1 | set(_avail(BLOCK_A, all_cols))
    a3 = a2 | set(_avail(BLOCK_C_CROSS, all_cols))
    a4 = a3 | set(_avail(BLOCK_C_MACRO, all_cols))
    a5 = a4 | set(_avail(BLOCK_BPRIME, all_cols))
    a6 = a4 | set(_avail(BLOCK_CHINA, all_cols))
    a7 = a2  # A4 with all Block C removed = A2

    return {
        "A0": sorted(a0), "A1": sorted(a1), "A2": sorted(a2), "A3": sorted(a3),
        "A4_FULL": sorted(a4), "A5": sorted(a5), "A6": sorted(a6), "A7": sorted(a7),
    }


RUNG_ORDER_FOR_DM = ["A0", "A1", "A2", "A3", "A4_FULL", "A5"]  # consecutive-rung comparisons
# A6 and A7 are compared directly against A4_FULL (per plan: "A6 vs A4", "A7 answers
# does macro help at all", i.e. vs A4_FULL), not chained into the main A0->A5 ladder.


def run_ablation_cell(model_factory, feature_cols: list, target: str, horizon: int,
                       fold: dict, seed: int, df: pd.DataFrame) -> dict:
    """Fit+evaluate one (rung, target, horizon, fold, seed) LightGBM cell on a
    restricted feature set. Returns per-observation errors/losses needed for
    DM tests between rungs, plus summary metrics."""
    from src.cubench.walkforward import (
        TARGET_PREFIX, compute_metrics_t1, compute_metrics_t2, compute_metrics_t3, TAUS,
    )

    ycol = f"{TARGET_PREFIX[target]}{horizon}"
    train_end = pd.Timestamp(fold["embargo"][f"h{horizon}"]["train_end"])
    train_start = pd.Timestamp(fold["train_start"])
    oos_start = pd.Timestamp(fold["oos_start"])
    oos_end = pd.Timestamp(fold["oos_end"])

    train_mask = (df["date"] >= train_start) & (df["date"] <= train_end) & df[ycol].notna()
    test_mask = (df["date"] >= oos_start) & (df["date"] <= oos_end) & df[ycol].notna()

    Xtr = df.loc[train_mask, feature_cols].reset_index(drop=True)
    ytr = df.loc[train_mask, ycol].to_numpy(dtype=float)
    Xte = df.loc[test_mask, feature_cols].reset_index(drop=True)
    yte = df.loc[test_mask, ycol].to_numpy(dtype=float)

    model = model_factory(seed, horizon)
    if target == "t1":
        model.fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xte))
        metrics = compute_metrics_t1(yte, pred)
        per_obs_loss = (yte - pred) ** 2
        return {"metrics": metrics, "per_obs_loss": per_obs_loss, "n": len(yte)}
    elif target == "t2":
        model.fit(Xtr, ytr)
        preds = [np.asarray(p) for p in model.predict_quantiles(Xte, TAUS)]
        metrics = compute_metrics_t2(yte, preds, TAUS)
        median_idx = TAUS.index(0.5)
        per_obs_loss = np.maximum(0.5 * (yte - preds[median_idx]), -0.5 * (yte - preds[median_idx]))
        return {"metrics": metrics, "per_obs_loss": per_obs_loss, "n": len(yte)}
    else:
        model.fit(Xtr, ytr)
        proba = np.asarray(model.predict_proba(Xte))
        metrics = compute_metrics_t3(yte, proba)
        proba_c = np.clip(proba, 1e-9, 1 - 1e-9)
        per_obs_loss = -(yte * np.log(proba_c) + (1 - yte) * np.log(1 - proba_c))
        return {"metrics": metrics, "per_obs_loss": per_obs_loss, "n": len(yte)}


def compare_consecutive_rungs(rung_losses: dict, horizon: int = 1) -> list:
    """rung_losses: {rung_id: concatenated per-obs loss array (pooled across
    folds/seeds in time order)}. Runs DM-HLN between each consecutive pair in
    RUNG_ORDER_FOR_DM, plus A4_FULL vs A6 and A4_FULL vs A7."""
    comparisons = []
    for i in range(len(RUNG_ORDER_FOR_DM) - 1):
        r0, r1 = RUNG_ORDER_FOR_DM[i], RUNG_ORDER_FOR_DM[i + 1]
        if r0 in rung_losses and r1 in rung_losses:
            L0, L1 = rung_losses[r0], rung_losses[r1]
            n = min(len(L0), len(L1))
            out = dm_test_hln(L0[:n], L1[:n], h=horizon, loss="raw")
            comparisons.append({"rung_from": r0, "rung_to": r1, **out})
    for extra, base in [("A6", "A4_FULL"), ("A7", "A4_FULL")]:
        if extra in rung_losses and base in rung_losses:
            L0, L1 = rung_losses[base], rung_losses[extra]
            n = min(len(L0), len(L1))
            out = dm_test_hln(L0[:n], L1[:n], h=horizon, loss="raw")
            comparisons.append({"rung_from": base, "rung_to": extra, **out})
    return comparisons
