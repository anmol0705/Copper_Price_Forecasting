"""Feature engineering -- Blocks A (trend/momentum, 16 feat), B' (curve proxy, 4 feat,
ablation-only), C (macro/cross-market, 22 feat + 1 ablation-only), D (volatility/regime,
28 feat per the literal table in Section 2.4 -- see NOTE below).

Full spec: docs/cubench_implementation_plan.md Section 2.

NOTE on feature count: the plan's Section 2.4 header says "26 features" for Block D but
its own feature table sums to 28 (4+3+2+2+1+1+2+1+1+1+1+2+3+4 = 28). This module
implements the table literally (28 Block-D features), since the table is the operative
spec and the "26" is a summary miscount. Reported as a discrepancy, not silently
resolved. Total feature count is therefore ~70 (16 + 4 + 22 + 28), not exactly 68 --
the plan itself calls "~68" an approximate target throughout.

Every rolling/expanding computation here is strictly causal: `.rolling(k)` (default
right-closed, includes t, excludes future), `.expanding(min_periods=...)`, `.shift(k)`
for lags, `.ewm(span=..., adjust=False)` for EWMAs. No `center=True` anywhere, no
whole-series `.fit()`/global standardization. This is mechanically checked by
tests/test_leakage.py::test_causal_perturbation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cubench import garch as garch_mod
from src.cubench.pit import pit_join_monthly

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/cubench/raw")

# Maps internal series name -> (raw CSV filename, value column name in that CSV).
# All are Close-only single-value CSVs except copper (full OHLC).
CLOSE_SERIES_FILES = {
    "aluminum": ("yf_aluminum.csv", "ALI=F"),
    "gold": ("yf_gold.csv", "GC=F"),
    "silver": ("yf_silver.csv", "SI=F"),
    "crude_oil": ("yf_crude_oil.csv", "CL=F"),
    "sp500": ("yf_sp500.csv", "^GSPC"),
    "us10y_yield": ("yf_us10y_yield.csv", "^TNX"),
    "dxy": ("yf_dxy.csv", "DX-Y.NYB"),
    "vix": ("yf_vix.csv", "^VIX"),
    "clp_usd_fx": ("yf_clp_usd_fx.csv", "CLP=X"),
    "qc_emini_copper": ("yf_qc_emini_copper.csv", "QC=F"),
    "dfii10": ("fred_real_yield_10y.csv", "DFII10"),
    "baa10y": ("fred_baa_credit_spread.csv", "BAA10Y"),
}

MONTHLY_SERIES_FILES = {
    "ppiaco": ("fred_ppi_all_commodities.csv", "PPIACO"),
    "indpro": ("fred_industrial_production.csv", "INDPRO"),
    "china_pmi_proxy": ("fred_china_pmi_proxy_oecd_business_confidence.csv", "BSCICP03CNM665S"),
}

FFILL_LIMIT_DAYS = 5  # per plan's existing forward-fill convention (test_no_nan_leak)


# --------------------------------------------------------------------------------- #
# Raw panel loading
# --------------------------------------------------------------------------------- #

def load_copper_ohlc() -> pd.DataFrame:
    """Copper's own OHLC panel -- this defines the daily trading calendar for the
    whole project (per the resolved decision: panel starts 2010-01-04, aluminum's
    later start is handled via NaN, never by truncating the calendar)."""
    df = pd.read_csv(RAW_DIR / "yf_copper.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
    return df[["date", "open", "high", "low", "close"]]


def load_close_series(name: str) -> pd.Series:
    fname, col = CLOSE_SERIES_FILES[name]
    df = pd.read_csv(RAW_DIR / fname, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.set_index("date")[col].rename(name)


def load_monthly_series(name: str) -> pd.DataFrame:
    fname, col = MONTHLY_SERIES_FILES[name]
    df = pd.read_csv(RAW_DIR / fname, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", col]]


def build_raw_panel() -> pd.DataFrame:
    """Assemble the single daily DataFrame, indexed by copper's trading calendar, that
    all feature blocks are computed from. Cross-market Close series are aligned onto
    copper's calendar and forward-filled up to FFILL_LIMIT_DAYS (causal -- only ever
    uses PAST values -- to bridge each market's own holiday calendar, never to fill a
    genuine data gap of unknown origin). Aluminum is deliberately NOT filled before its
    2014-05-06 start; NaN there is correct and intentional (see module docstring of the
    caller / STATUS.md decision), not a ffill target.
    """
    panel = load_copper_ohlc()
    calendar = panel[["date"]].copy()

    for name in CLOSE_SERIES_FILES:
        s = load_close_series(name)
        aligned = s.reindex(calendar["date"])
        # ffill(limit=5) bridges each market's own holiday calendar using only past
        # values (causal). For aluminum specifically this does NOT synthesize values
        # before its true 2014-05-06 start -- reindex() leaves those rows NaN and
        # there is nothing before them to ffill from.
        aligned = aligned.ffill(limit=FFILL_LIMIT_DAYS)
        panel[name] = aligned.values

    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------------- #

def log_return(close: pd.Series, k: int = 1) -> pd.Series:
    return np.log(close / close.shift(k))


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# --------------------------------------------------------------------------------- #
# Block A -- trend / momentum (16 features)
# --------------------------------------------------------------------------------- #

def build_block_a(panel: pd.DataFrame) -> pd.DataFrame:
    c = panel["close"]
    r = log_return(c, 1)
    out = pd.DataFrame(index=panel.index)

    for k in (5, 10, 22, 63, 126, 252):
        out[f"a_mom_{k}"] = np.log(c / c.shift(k))

    for f, s in ((8, 32), (16, 64), (32, 128)):
        ewma_f = c.ewm(span=f, adjust=False, min_periods=f).mean()
        ewma_s = c.ewm(span=s, adjust=False, min_periods=s).mean()
        out[f"a_ewma_x_{f}_{s}"] = ewma_f / ewma_s - 1.0

    for k in (22, 63):
        roll_mean = c.rolling(k).mean()
        roll_std = c.rolling(k).std()
        out[f"a_zscore_{k}"] = (c - roll_mean) / roll_std

    out["a_rsi_14"] = wilder_rsi(c, 14)

    roll_hi_252 = panel["high"].rolling(252).max()
    roll_lo_252 = panel["low"].rolling(252).min()
    out["a_dist_hi_252"] = np.log(c / roll_hi_252)
    out["a_dist_lo_252"] = np.log(c / roll_lo_252)

    out["a_r_lag1"] = r
    out["a_r_lag2"] = r.shift(1)

    return out


# --------------------------------------------------------------------------------- #
# Block D -- volatility / regime (built before B' because B' and C need RV/returns)
# --------------------------------------------------------------------------------- #

def _variance_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    r = log_return(c, 1)
    park = (1.0 / (4.0 * np.log(2.0))) * np.log(h / l) ** 2
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    sq = r ** 2
    gk_floored = gk.clip(lower=1e-10)
    return pd.DataFrame({"r": r, "park": park.clip(lower=1e-10), "gk": gk_floored, "sq": sq.clip(lower=1e-10)})


def build_block_d(panel: pd.DataFrame, vix: pd.Series, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Returns (Block D features, garch_meta dict with fit diagnostics)."""
    vp = _variance_proxies(panel)
    r, park, gk, sq = vp["r"], vp["park"], vp["gk"], vp["sq"]
    out = pd.DataFrame(index=panel.index)

    def logrv(series: pd.Series, k: int) -> pd.Series:
        return np.log(series.rolling(k).mean())

    for k in (1, 5, 22, 66):
        out[f"d_logrv_{k}"] = logrv(gk, k)
    for k in (1, 5, 22):
        out[f"d_logrv_park_{k}"] = logrv(park, k)
    for k in (5, 22):
        out[f"d_logrv_sq_{k}"] = logrv(sq, k)

    out["d_rv_ratio_5_22"] = out["d_logrv_5"] - out["d_logrv_22"]
    out["d_rv_ratio_22_66"] = out["d_logrv_22"] - out["d_logrv_66"]

    out["d_volofvol_22"] = out["d_logrv_1"].rolling(22).std()

    downside_sq = sq.where(r < 0, 0.0)
    out["d_downside_semivar_22"] = downside_sq.rolling(22).mean()

    out["d_skew_63"] = r.rolling(63).skew()
    out["d_kurt_63"] = r.rolling(63).kurt()

    out["d_jump_22"] = (sq.rolling(22).mean() - gk.rolling(22).mean()).clip(lower=0.0)

    # ---- GARCH / GJR-GARCH ----
    r_indexed = pd.Series(r.values, index=panel["date"].values)
    r_clean = r_indexed.dropna()

    garch_res = garch_mod.fit_expanding_monthly_refit(
        r_clean, spec_name="garch11_normal", o=0, dist="normal", use_cache=use_cache,
    )
    gjr_res = garch_mod.fit_expanding_monthly_refit(
        r_clean, spec_name="gjr_garch_t", o=1, dist="t", use_cache=use_cache,
    )

    date_index = pd.DatetimeIndex(panel["date"].values)
    out["d_garch_sigma"] = garch_res.sigma.reindex(date_index).values
    out["d_gjr_sigma"] = gjr_res.sigma.reindex(date_index).values
    garch_resid_z_aligned = pd.Series(r.values, index=date_index) / out["d_garch_sigma"].replace(0.0, np.nan).values
    out["d_garch_resid_z"] = garch_resid_z_aligned.values

    garch_meta = {
        "garch11_normal": {
            "n_refits": garch_res.n_refits,
            "n_convergence_failures": garch_res.n_convergence_failures,
        },
        "gjr_garch_t": {
            "n_refits": gjr_res.n_refits,
            "n_convergence_failures": gjr_res.n_convergence_failures,
        },
    }

    # ---- VIX features ----
    # `vix` is panel["vix"] -- already row-aligned 1:1 with `panel` (shares its
    # RangeIndex, same order), so no reindexing by date is needed or correct here:
    # `vix` was never date-indexed, so a .reindex(DatetimeIndex(...)) against it would
    # find no matching labels and silently produce all-NaN (caught during the real
    # end-to-end build -- see STATUS/report notes).
    vix_aligned = vix.reset_index(drop=True)
    out["d_vix_level"] = np.log(vix_aligned).values
    vix_ratio = np.log(vix_aligned.rolling(5).mean() / vix_aligned.rolling(22).mean())
    out["d_vix_ratio_5_22"] = vix_ratio.values

    # ---- Vol regime: expanding-quantile terciles of d_logrv_22, computed on [0:t] only ----
    logrv22 = out["d_logrv_22"]
    q33 = logrv22.expanding(min_periods=504).quantile(0.33)
    q67 = logrv22.expanding(min_periods=504).quantile(0.67)
    regime_lo = (logrv22 <= q33).astype(float)
    regime_hi = (logrv22 > q67).astype(float)
    regime_mid = 1.0 - regime_lo - regime_hi
    # Where the expanding quantiles themselves are still NaN (insufficient history),
    # the regime one-hots must also be NaN, not spuriously 0/1.
    insufficient = q33.isna() | q67.isna() | logrv22.isna()
    regime_lo = regime_lo.where(~insufficient)
    regime_mid = regime_mid.where(~insufficient)
    regime_hi = regime_hi.where(~insufficient)
    out["d_regime_lo"] = regime_lo
    out["d_regime_mid"] = regime_mid
    out["d_regime_hi"] = regime_hi

    # ---- Day-of-week dummies (Monday reference) ----
    dow = pd.to_datetime(panel["date"]).dt.dayofweek.values  # 0=Mon..4=Fri
    for i, name in zip((1, 2, 3, 4), ("tue", "wed", "thu", "fri")):
        out[f"d_dow_{i}"] = (dow == i).astype(float)

    return out, garch_meta


# --------------------------------------------------------------------------------- #
# Block B' -- curve proxy (4 features, ablation-only)
# --------------------------------------------------------------------------------- #

def build_block_b_prime(panel: pd.DataFrame, block_d: pd.DataFrame) -> pd.DataFrame:
    c = panel["close"]
    al = panel["aluminum"]
    out = pd.DataFrame(index=panel.index)

    # b1: RV term ratio RV5/RV22 (level ratio, not the log-difference used in Block D's
    # d_rv_ratio -- Section 2.2/1.5 defines b1 as the RV5/RV22 ratio itself).
    gk = _variance_proxies(panel)["gk"]
    rv5 = gk.rolling(5).mean()
    rv22 = gk.rolling(22).mean()
    out["b1_rv_term_ratio"] = rv5 / rv22

    # b2: 60-day z-score of log(HG=F / ALI=F)
    log_ratio = np.log(c / al)
    roll_mean = log_ratio.rolling(60).mean()
    roll_std = log_ratio.rolling(60).std()
    out["b2_cu_al_relvalue"] = (log_ratio - roll_mean) / roll_std

    # b3: 22-day copper momentum - 22-day aluminum momentum
    cu_mom_22 = np.log(c / c.shift(22))
    al_mom_22 = np.log(al / al.shift(22))
    out["b3_cu_al_mom_spread"] = cu_mom_22 - al_mom_22

    # b4: log(HG=F close) - log(QC=F close) -- QC=F included per the Week A1 judgment
    # call documented in docs/cubench_data_availability.md (Discrepancy 2).
    qc = panel["qc_emini_copper"]
    out["b4_hg_qc_spread"] = np.log(c) - np.log(qc)

    return out


# --------------------------------------------------------------------------------- #
# Block C -- macro / cross-market (22 features + 1 ablation-only)
# --------------------------------------------------------------------------------- #

DAILY_RETURN_SERIES = ("dxy", "vix", "sp500", "gold", "silver", "aluminum", "crude_oil")
YIELD_SPREAD_SERIES = ("us10y_yield", "dfii10", "baa10y")

# Internal series name -> feature-name short code used in the plan's c_<x>_* naming.
_C_NAME_MAP = {
    "dxy": "dxy", "vix": "vix", "sp500": "sp500", "gold": "gold",
    "silver": "silver", "aluminum": "aluminum", "crude_oil": "crude",
    "us10y_yield": "tnx", "dfii10": "dfii10", "baa10y": "baa10y",
}


def build_block_c(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (headline Block C features [22], ablation-only extras [china PMI proxy])."""
    out = pd.DataFrame(index=panel.index)

    for name in DAILY_RETURN_SERIES:
        code = _C_NAME_MAP[name]
        s = panel[name]
        out[f"c_{code}_r1"] = np.log(s / s.shift(1))
        out[f"c_{code}_r5"] = np.log(s / s.shift(5))

    for name in YIELD_SPREAD_SERIES:
        code = _C_NAME_MAP[name]
        s = panel[name]
        out[f"c_{code}_d1"] = 100.0 * (s - s.shift(1))
        out[f"c_{code}_d22"] = 100.0 * (s - s.shift(22))

    # ---- Monthly, point-in-time joined ----
    daily_calendar = panel[["date"]].copy()

    # NOTE: pit_join_monthly's `series_id` parameter is used BOTH as the
    # PUBLICATION_LAG lookup key and as the value column name in monthly_df, so the
    # YoY-transformed column must keep the raw series' name (PPIACO/INDPRO/
    # BSCICP03CNM665S) -- the publication lag is a property of the release schedule
    # of the raw series, unaffected by the YoY transform applied downstream of it.
    ppi = load_monthly_series("ppiaco")
    ppi["PPIACO"] = ppi["PPIACO"].pct_change(12) * 100.0
    ppi_pit = pit_join_monthly(daily_calendar, ppi[["date", "PPIACO"]], "PPIACO")
    out["c_ppi_yoy"] = ppi_pit["PPIACO"].values

    indpro = load_monthly_series("indpro")
    indpro["INDPRO"] = indpro["INDPRO"].pct_change(12) * 100.0
    indpro_pit = pit_join_monthly(daily_calendar, indpro[["date", "INDPRO"]], "INDPRO")
    out["c_indpro_yoy"] = indpro_pit["INDPRO"].values

    # ---- Ablation-only: China business-confidence (PMI proxy) ----
    china = load_monthly_series("china_pmi_proxy")
    china["BSCICP03CNM665S"] = china["BSCICP03CNM665S"].pct_change(12) * 100.0
    china_pit = pit_join_monthly(daily_calendar, china[["date", "BSCICP03CNM665S"]], "BSCICP03CNM665S")
    ablation = pd.DataFrame(index=panel.index)
    ablation["c_china_conf"] = china_pit["BSCICP03CNM665S"].values

    return out, ablation


# --------------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------------- #

FEATURE_BLOCK_PREFIXES = {
    "A": "a_", "Bprime": "b", "C": "c_", "D": "d_",
}


def build_feature_matrix(include_ablation_only: bool = True, panel: pd.DataFrame | None = None,
                          use_garch_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the full causal feature matrix (Blocks A, B', C, D [+ ablation-only extras])
    on copper's trading calendar starting 2010-01-04. Returns (DataFrame indexed by
    `date`, metadata dict incl. GARCH fit diagnostics). Does NOT include targets --
    see targets.py; targets are joined by the caller (scripts/cubench_build_data.py-
    style build pipeline), never inside this module, so nothing here can accidentally
    treat a target column as a feature.

    `panel` may be supplied pre-built (e.g. by tests/test_leakage.py::test_causal_perturbation,
    which injects a copy of the real panel with all rows after some index t nulled out)
    -- when omitted, the real raw panel is loaded from data/cubench/raw/ via
    build_raw_panel(). `use_garch_cache=False` forces a fresh GARCH fit (used by the
    perturbation test, where the return series differs from the real one on every call
    so the cache would never hit anyway, and skipping the cache write avoids polluting
    the real cache file with perturbed-series fits).
    """
    if panel is None:
        panel = build_raw_panel()

    block_a = build_block_a(panel)
    block_d, garch_meta = build_block_d(panel, panel["vix"], use_cache=use_garch_cache)
    block_bprime = build_block_b_prime(panel, block_d)
    block_c, block_c_ablation = build_block_c(panel)

    feat = pd.concat([panel[["date"]], block_a, block_bprime, block_c, block_d], axis=1)
    if include_ablation_only:
        feat = pd.concat([feat, block_c_ablation], axis=1)

    feat = feat.sort_values("date").reset_index(drop=True)

    meta = {
        "n_rows": len(feat),
        "date_range": [str(feat["date"].min().date()), str(feat["date"].max().date())],
        "garch": garch_meta,
        "block_a_cols": list(block_a.columns),
        "block_bprime_cols": list(block_bprime.columns),
        "block_c_cols": list(block_c.columns),
        "block_c_ablation_cols": list(block_c_ablation.columns) if include_ablation_only else [],
        "block_d_cols": list(block_d.columns),
    }
    return feat, meta


def null_rows_after(panel: pd.DataFrame, t: int) -> pd.DataFrame:
    """Return a copy of `panel` with every column except 'date' set to NaN for all rows
    with positional index > t. Used by tests/test_leakage.py::test_causal_perturbation
    to prove no feature at row t depends on any row after t."""
    perturbed = panel.copy()
    value_cols = [c for c in perturbed.columns if c != "date"]
    perturbed.loc[perturbed.index > t, value_cols] = np.nan
    return perturbed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feat, meta = build_feature_matrix()
    print(f"Feature matrix: {feat.shape}, date range {meta['date_range']}")
    print(f"Columns: {feat.shape[1] - 1} features (excl. date)")
