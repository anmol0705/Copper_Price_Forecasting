"""The 7 leakage-safety gates (docs/cubench_implementation_plan.md Section 2.5).
Pre-registered gate: ALL must pass before any model trains.

Run: .venv_corr/Scripts/python.exe -m pytest tests/test_leakage.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cubench import features as feat_mod
from src.cubench import folds as folds_mod
from src.cubench import garch as garch_mod
from src.cubench import pit as pit_mod
from src.cubench import targets as targets_mod
from src.cubench.data import EXPECTED_FRED_FIRST_OBS, PUBLICATION_LAG_DAYS

RNG_SEED = 20260824


# --------------------------------------------------------------------------------- #
# Shared, session-scoped fixtures -- avoid rebuilding the (GARCH-dominated, ~12s)
# feature matrix more times than necessary.
# --------------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def raw_panel():
    return feat_mod.build_raw_panel()


@pytest.fixture(scope="session")
def baseline_features(raw_panel):
    feat, meta = feat_mod.build_feature_matrix(panel=raw_panel)
    return feat, meta


@pytest.fixture(scope="session")
def baseline_targets(raw_panel):
    return targets_mod.build_all_targets(raw_panel)


# --------------------------------------------------------------------------------- #
# 1. test_causal_perturbation -- the decisive test.
# --------------------------------------------------------------------------------- #

def test_causal_perturbation(raw_panel, baseline_features):
    """For 30 randomly-chosen indices t: rebuild the full feature matrix from a raw
    panel in which all rows after t have been replaced with NaN; assert every
    feature's value at t matches the unperturbed build to tight numerical tolerance
    (floating-point-identical up to optimizer/BLAS-order noise; any feature that
    changes has look-ahead). This is the single most important test in the project.
    """
    baseline, _ = baseline_features
    n = len(raw_panel)
    rng = np.random.default_rng(RNG_SEED)
    # Avoid the very first handful of rows (trivially all-NaN for both builds anyway)
    # and the very last row (perturbation would be a no-op).
    candidate_idx = np.arange(60, n - 1)
    test_idx = rng.choice(candidate_idx, size=30, replace=False)

    feature_cols = [c for c in baseline.columns if c != "date"]
    failures = []

    for t in test_idx:
        perturbed_panel = feat_mod.null_rows_after(raw_panel, t)
        perturbed_feat, _ = feat_mod.build_feature_matrix(
            panel=perturbed_panel, use_garch_cache=False,
        )
        base_row = baseline.iloc[t]
        pert_row = perturbed_feat.iloc[t]
        for col in feature_cols:
            b = base_row[col]
            p = pert_row[col]
            both_nan = pd.isna(b) and pd.isna(p)
            if both_nan:
                continue
            if pd.isna(b) != pd.isna(p):
                failures.append((t, col, b, p))
                continue
            if not np.isclose(b, p, rtol=1e-6, atol=1e-9):
                failures.append((t, col, b, p))

    assert not failures, (
        f"{len(failures)} causal-perturbation failures (look-ahead detected): "
        f"{failures[:20]}{'...' if len(failures) > 20 else ''}"
    )


# --------------------------------------------------------------------------------- #
# 2. test_pit_monthly_join
# --------------------------------------------------------------------------------- #

def test_pit_monthly_join(raw_panel):
    """For PPIACO: assert every daily row's joined reference month satisfies
    month_end + lag <= daily_date. Assert the naive reindex().ffill() join produces a
    DIFFERENT series than the PIT join (proving the PIT join does real work)."""
    monthly = feat_mod.load_monthly_series("ppiaco")
    daily_calendar = raw_panel[["date"]].copy()

    pit = pit_mod.pit_join_monthly(daily_calendar, monthly, "PPIACO")
    naive = pit_mod._naive_reindex_ffill_join(daily_calendar, monthly, "PPIACO")

    # -- Availability constraint: for every row where PIT gives a non-null value, its
    # available_from date (recomputed) must be <= the daily row's date.
    monthly_avail = pit_mod.compute_available_from(monthly, "PPIACO")
    lag = PUBLICATION_LAG_DAYS["PPIACO"]
    for _, r in monthly_avail.iterrows():
        expected_avail = (r["date"] + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=lag)
        assert r["available_from"] == expected_avail

    valid = pit.dropna(subset=["PPIACO"])
    assert (valid["available_from"] <= valid["date"]).all(), (
        "PIT join produced a row where available_from is AFTER the daily date -- "
        "this would mean the join used data before it was actually released."
    )

    # -- Divergence from the naive join: the naive approach must NOT be a no-op
    # equivalent of the PIT join (if it were, the PIT machinery wouldn't be doing
    # anything -- see pit.py's own inline sanity check for the exact mechanism).
    pit_vals = pit["PPIACO"].values
    naive_vals = np.asarray(naive)
    both_notnull = ~(pd.isna(pit_vals) & pd.isna(naive_vals))
    comparable = ~pd.isna(pit_vals) & ~pd.isna(naive_vals)
    n_diff = int(np.sum(~np.isclose(pit_vals[comparable], naive_vals[comparable], equal_nan=True)))
    assert n_diff > 0, (
        "PIT join and naive reindex().ffill() join produced IDENTICAL values "
        "everywhere -- the PIT join is a no-op, which means it is not enforcing "
        "point-in-time availability."
    )


# --------------------------------------------------------------------------------- #
# 3. test_fred_coverage_assertions
# --------------------------------------------------------------------------------- #

@pytest.mark.parametrize("series_id,csv_name", [
    ("DFII10", "fred_real_yield_10y.csv"),
    ("BAA10Y", "fred_baa_credit_spread.csv"),
    ("PPIACO", "fred_ppi_all_commodities.csv"),
    ("INDPRO", "fred_industrial_production.csv"),
])
def test_fred_coverage_assertions(series_id, csv_name):
    """Hard-assert first-observation dates against the plan's expected values (Section
    2.5 test 3 / Section 1.4). Fails loudly on regression -- the HY OAS incident (full
    history -> 754 obs from 2023-08-21) proves FRED coverage can silently shrink."""
    df = pd.read_csv(feat_mod.RAW_DIR / csv_name, parse_dates=["date"])
    first_obs = df["date"].min()

    expected = {
        "DFII10": "2010-01-05", "BAA10Y": "2010-01-05",
        "PPIACO": "2009-01-01", "INDPRO": "2009-01-01",
    }[series_id]
    assert first_obs <= pd.Timestamp(expected), (
        f"{series_id} FRED coverage regression: expected first obs on/before "
        f"{expected}, got {first_obs.date()}. This is the failure mode that hit "
        f"BAMLH0A0HYM2 in April 2026."
    )
    # Cross-check against the single-source-of-truth registry in data.py.
    assert EXPECTED_FRED_FIRST_OBS[series_id] == expected


# --------------------------------------------------------------------------------- #
# 4. test_no_target_in_features
# --------------------------------------------------------------------------------- #

def test_no_target_in_features(baseline_features, baseline_targets):
    """Assert no feature column name matches the target-construction regex, and that
    |corr(feature_t, target_t)| < 0.99 for every feature (a near-perfect correlation
    at horizon 0 would indicate an accidental target leak into the feature set)."""
    feat, _ = baseline_features
    feature_cols = [c for c in feat.columns if c != "date"]

    for col in feature_cols:
        assert not re.match(targets_mod.TARGET_COLUMN_REGEX, col), (
            f"Feature column {col!r} matches the target-naming pattern "
            f"{targets_mod.TARGET_COLUMN_REGEX!r} -- a target column has leaked "
            f"into the feature set."
        )

    # feat and baseline_targets are both built directly from the same raw_panel and
    # share its exact row order (positional RangeIndex) -- concat by position, not by
    # a label-based .join() (feat's index becomes 'date' timestamps, baseline_targets'
    # stays the default integer RangeIndex; joining those by label would silently
    # produce all-NaN and make every correlation check vacuously skip).
    assert len(feat) == len(baseline_targets)
    joined = pd.concat(
        [feat.reset_index(drop=True), baseline_targets.reset_index(drop=True)], axis=1
    )
    bad = []
    for col in feature_cols:
        for tcol in baseline_targets.columns:
            sub = joined[[col, tcol]].dropna()
            if len(sub) < 30:
                continue
            corr = sub[col].corr(sub[tcol])
            if pd.notna(corr) and abs(corr) >= 0.99:
                bad.append((col, tcol, corr))
    assert not bad, f"Suspiciously near-perfect feature/target correlation(s): {bad}"


# --------------------------------------------------------------------------------- #
# 5. test_garch_causality
# --------------------------------------------------------------------------------- #

def test_garch_causality(raw_panel, baseline_features):
    """Assert d_garch_sigma[t] (and d_gjr_sigma[t]) are unchanged when r[t+1:] is
    nulled -- the GARCH-specific instance of the causal-perturbation property, run
    with fewer trials than test_causal_perturbation since it only needs to isolate
    the GARCH columns."""
    baseline, _ = baseline_features
    n = len(raw_panel)
    rng = np.random.default_rng(RNG_SEED + 1)
    test_idx = rng.choice(np.arange(600, n - 1), size=6, replace=False)

    garch_cols = ["d_garch_sigma", "d_gjr_sigma", "d_garch_resid_z"]
    failures = []
    for t in test_idx:
        perturbed_panel = feat_mod.null_rows_after(raw_panel, t)
        perturbed_feat, _ = feat_mod.build_feature_matrix(
            panel=perturbed_panel, use_garch_cache=False,
        )
        for col in garch_cols:
            b = baseline.iloc[t][col]
            p = perturbed_feat.iloc[t][col]
            if pd.isna(b) and pd.isna(p):
                continue
            if pd.isna(b) != pd.isna(p) or not np.isclose(b, p, rtol=1e-6, atol=1e-9):
                failures.append((t, col, b, p))

    assert not failures, f"GARCH causality violated: {failures}"


# --------------------------------------------------------------------------------- #
# 6. test_embargo_disjointness
# --------------------------------------------------------------------------------- #

def test_embargo_disjointness(raw_panel):
    """For every walk-forward fold and every horizon h, assert
    max(train_target_end_date) < min(oos_feature_date), where train_target_end_date
    accounts for the h-day forward span of the target (i.e. the embargoed training
    window's last row's target reaches at most `embargo_train_end + h` trading days,
    which must still fall strictly before the first OOS trading day)."""
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(raw_panel["date"]).unique()))
    folds = folds_mod.make_folds(cal)
    assert len(folds) == 11

    for fold in folds:
        oos_start = pd.Timestamp(fold["oos_start"])
        for hkey, info in fold["embargo"].items():
            h = int(hkey[1:])
            embargo_train_end = pd.Timestamp(info["train_end"])
            pos = cal.get_loc(embargo_train_end)
            # The last training target at `embargo_train_end` spans forward h trading
            # days -- its target reaches cal[pos + h] (if within range).
            target_end_pos = pos + h
            if target_end_pos < len(cal):
                train_target_end_date = cal[target_end_pos]
            else:
                train_target_end_date = cal[-1]
            assert train_target_end_date < oos_start, (
                f"Fold {fold['fold']} h={h}: train target end "
                f"{train_target_end_date.date()} is not strictly before OOS start "
                f"{oos_start.date()} -- embargo insufficient, targets overlap OOS."
            )


# --------------------------------------------------------------------------------- #
# 7. test_no_nan_leak
# --------------------------------------------------------------------------------- #

def test_no_nan_leak():
    """Assert forward-fill limits are <=5 trading days on daily cross-market series
    (existing project convention), and statically assert features.py never uses
    backward-fill or an unlimited forward-fill anywhere (both are look-ahead vectors:
    bfill trivially so; an unlimited ffill can silently smuggle a stale value across
    an arbitrarily long gap, including gaps that span the train/OOS boundary)."""
    assert feat_mod.FFILL_LIMIT_DAYS <= 5

    src = Path(feat_mod.__file__).read_text()
    assert ".bfill(" not in src, "features.py must never use backward-fill (look-ahead)."
    assert "method='bfill'" not in src and 'method="bfill"' not in src

    # Every ffill call must carry an explicit limit.
    for m in re.finditer(r"\.ffill\(([^)]*)\)", src):
        args = m.group(1)
        assert "limit=" in args, (
            f"Found an unlimited .ffill() call in features.py: '.ffill({args})' -- "
            f"every forward-fill must carry an explicit limit=<=5 per project convention."
        )
