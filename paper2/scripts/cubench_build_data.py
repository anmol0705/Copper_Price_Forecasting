"""Week A1 deliverable: pull everything that passed the S1-S4 re-verification spikes
into data/cubench/raw/ as CSVs, plus data_source_log.json (mirroring the format/
conventions of data/raw_correlation_check/data_source_log.json).

Run: .venv_corr/Scripts/python.exe scripts/cubench_build_data.py
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.cubench.data import (
    fetch_fred_series_curl,
    fetch_yfinance_ohlc,
    fetch_yfinance_close,
    verify_fred_coverage,
    write_source_log,
    DataDownloadError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/cubench/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

START, END = "2010-01-01", "2025-12-31"

# --- Block P: copper price core (full OHLC) ---
OHLC_TICKERS = {
    "copper": "HG=F",
}

# --- Block C-cross: cross-market daily (Close only per plan Section 1.3) ---
CLOSE_ONLY_TICKERS = {
    "aluminum": "ALI=F",
    "gold": "GC=F",
    "silver": "SI=F",
    "crude_oil": "CL=F",
    "sp500": "^GSPC",
    "us10y_yield": "^TNX",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
}

# --- Block C-macro conditional (Section 1.4) ---
# CLP=X passed S4 (4164 obs, 0.22% gap < 2% threshold) -> include.
# QC=F did NOT meet the literal <2% gap rule (3.55% gap) though its obs count (4024)
# and gap profile are effectively identical to HG=F's own (3.57% gap, same COMEX
# holiday calendar) -- see docs/cubench_data_availability.md for the full discussion.
# Pulled here regardless so the raw data is snapshotted; the *feature* decision
# (whether b4_hg_qc_spread is built from it) is deferred to Week A2 features.py.
CLOSE_ONLY_CONDITIONAL_TICKERS = {
    "clp_usd_fx": "CLP=X",
    "qc_emini_copper": "QC=F",
}

# --- Block C-macro: FRED series (daily, curl) ---
FRED_DAILY_SERIES = {
    "real_yield_10y": "DFII10",
    "baa_credit_spread": "BAA10Y",
}

# --- Block C-macro: FRED series (monthly, curl) ---
FRED_MONTHLY_SERIES = {
    "ppi_all_commodities": "PPIACO",
    "industrial_production": "INDPRO",
}

# --- Robustness-appendix-only FRED series (not headline, per D4) ---
FRED_ROBUSTNESS_SERIES = {
    "high_yield_oas": "BAMLH0A0HYM2",  # 2023-08-21+ only, 2023-2025 robustness appendix
}

# --- Ablation-only FRED series (per D7) ---
FRED_ABLATION_SERIES = {
    "china_pmi_proxy_oecd_business_confidence": "BSCICP03CNM665S",
}


def main():
    log = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {"start": START, "end": END},
        "yfinance": {},
        "fred": {},
        "failures": {},
    }

    # --- yfinance: full OHLC (Block P) ---
    for name, ticker in OHLC_TICKERS.items():
        try:
            df = fetch_yfinance_ohlc(ticker, start=START, end=END)
            out_path = RAW_DIR / f"yf_{name}.csv"
            df.to_csv(out_path)
            log["yfinance"][name] = {
                "ticker": ticker, "fields": "OHLCV", "rows": len(df),
                "first_date": str(df.index.min().date()), "last_date": str(df.index.max().date()),
                "path": str(out_path),
            }
            logger.info(f"[OK] {name} ({ticker}): {len(df)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({ticker}): {e}")

    # --- yfinance: close-only (Block C-cross + conditional Block C-macro) ---
    for name, ticker in {**CLOSE_ONLY_TICKERS, **CLOSE_ONLY_CONDITIONAL_TICKERS}.items():
        try:
            s = fetch_yfinance_close(ticker, start=START, end=END)
            out_path = RAW_DIR / f"yf_{name}.csv"
            s.to_frame(name=ticker).to_csv(out_path)
            log["yfinance"][name] = {
                "ticker": ticker, "fields": "Close", "rows": len(s),
                "first_date": str(s.index.min().date()), "last_date": str(s.index.max().date()),
                "path": str(out_path),
                "conditional": name in CLOSE_ONLY_CONDITIONAL_TICKERS,
            }
            logger.info(f"[OK] {name} ({ticker}): {len(s)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({ticker}): {e}")

    # --- FRED: daily ---
    for name, series_id in FRED_DAILY_SERIES.items():
        try:
            df = fetch_fred_series_curl(series_id, cosd="2009-01-01", coed=END)
            first_obs = verify_fred_coverage(series_id, df)
            out_path = RAW_DIR / f"fred_{name}.csv"
            df.to_csv(out_path, index=False)
            log["fred"][name] = {
                "series_id": series_id, "frequency": "daily", "rows": len(df),
                "first_date": first_obs, "last_date": str(df["date"].max().date()),
                "path": str(out_path), "coverage_verified": True,
            }
            logger.info(f"[OK] {name} ({series_id}): {len(df)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({series_id}): {e}")

    # --- FRED: monthly (publication-lagged in pit.py, not here) ---
    for name, series_id in FRED_MONTHLY_SERIES.items():
        try:
            df = fetch_fred_series_curl(series_id, cosd="2009-01-01", coed=END)
            first_obs = verify_fred_coverage(series_id, df)
            out_path = RAW_DIR / f"fred_{name}.csv"
            df.to_csv(out_path, index=False)
            log["fred"][name] = {
                "series_id": series_id, "frequency": "monthly", "rows": len(df),
                "first_date": first_obs, "last_date": str(df["date"].max().date()),
                "path": str(out_path), "coverage_verified": True,
                "publication_lag_days_applied_at_join": "see src/cubench/pit.py::PUBLICATION_LAG_DAYS (not applied to this raw snapshot)",
            }
            logger.info(f"[OK] {name} ({series_id}): {len(df)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({series_id}): {e}")

    # --- FRED: robustness-appendix-only (HY OAS, truncated) ---
    for name, series_id in FRED_ROBUSTNESS_SERIES.items():
        try:
            df = fetch_fred_series_curl(series_id, cosd="2009-01-01", coed=END)
            out_path = RAW_DIR / f"fred_{name}.csv"
            df.to_csv(out_path, index=False)
            log["fred"][name] = {
                "series_id": series_id, "frequency": "daily", "rows": len(df),
                "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
                "path": str(out_path), "coverage_verified": False,
                "note": "EXCLUDED from headline model (D4) -- 2023-2025 robustness-appendix use only. "
                        "Confirmed truncated per prior investigation.",
            }
            logger.info(f"[OK, robustness-only] {name} ({series_id}): {len(df)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({series_id}): {e}")

    # --- FRED: ablation-only (China PMI proxy) ---
    for name, series_id in FRED_ABLATION_SERIES.items():
        try:
            df = fetch_fred_series_curl(series_id, cosd="2009-01-01", coed=END)
            out_path = RAW_DIR / f"fred_{name}.csv"
            df.to_csv(out_path, index=False)
            log["fred"][name] = {
                "series_id": series_id, "frequency": "monthly", "rows": len(df),
                "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date()),
                "path": str(out_path), "coverage_verified": False,
                "note": "ABLATION-ONLY (D7) -- not in headline model.",
            }
            logger.info(f"[OK, ablation-only] {name} ({series_id}): {len(df)} rows -> {out_path}")
        except DataDownloadError as e:
            log["failures"][name] = str(e)
            logger.error(f"[FAIL] {name} ({series_id}): {e}")

    log["china_pmi_genuine"] = "NOT_OBTAINABLE_FREE"
    log["china_pmi_fallback"] = "FRED BSCICP03CNM665S (OECD business confidence proxy, monthly) -- ablation-only"
    log["block_b_contract_month_audit"] = {
        "status": "TIER_3_CONFIRMED",
        "results_csv": "results/cubench/data_audit/comex_contract_month_coverage.csv",
        "summary_json": "results/cubench/data_audit/comex_contract_month_coverage_summary.json",
    }

    write_source_log(log, path=str(RAW_DIR / "data_source_log.json"))
    n_ok = len(log["yfinance"]) + len(log["fred"])
    n_fail = len(log["failures"])
    logger.info(f"Done. {n_ok} series pulled successfully, {n_fail} failures.")
    if log["failures"]:
        logger.warning(f"Failures: {log['failures']}")


if __name__ == "__main__":
    main()
