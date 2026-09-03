"""Unified downloader for CuBench: yfinance pulls (with retry/backoff) and FRED pulls
via `curl` shelled out through `subprocess` (NOT `requests`/`pandas_datareader`/`fredapi`
-- see the verified environment hazard below).

Verified environment hazard (carried forward from
notebooks/correlation_feasibility_analysis.ipynb and confirmed again during the Week A1
S3 spike, docs/cubench_implementation_plan.md Section 1.1): Python's `requests` library
reliably times out against `fred.stlouisfed.org` on this machine (TLS/schannel issue),
while `curl` succeeds against the identical URL. All FRED access in CuBench therefore
shells out to `curl`. (Yahoo Finance's endpoints do NOT have this problem -- `requests`
works fine against query1.finance.yahoo.com, confirmed during the S2 spike -- so the
yfinance path below uses the yfinance library directly, which itself uses `requests`
internally without issue.)

Defensive posture (per plan Section 1.1 / Risk R8): every download function raises
loudly on failure or partial/empty data rather than silently returning a partial frame.
This mirrors src/data_pipeline.py::DataDownloader.download()'s existing convention.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRED_URL_TEMPLATE = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={cosd}&coed={coed}"
)

# Deliberately conservative over-estimates of the true BLS/Fed/OECD release calendars
# (plan Section 2.3). Used by pit.py, defined here alongside the FRED series they gate.
PUBLICATION_LAG_DAYS = {
    "PPIACO": 30,
    "INDPRO": 22,
    "BSCICP03CNM665S": 45,
}

# Expected first-observation dates, hard-asserted per plan Section 2.5 test 3 /
# Section 1.4. A regression here (a series silently truncated by FRED, as happened to
# BAMLH0A0HYM2 in April 2026) must raise, not degrade silently (Risk R2).
EXPECTED_FRED_FIRST_OBS = {
    "DFII10": "2010-01-05",
    "BAA10Y": "2010-01-05",
    "PPIACO": "2009-01-01",
    "INDPRO": "2009-01-01",
}


class DataDownloadError(RuntimeError):
    """Raised when a download fails or returns partial/empty/malformed data.

    Per the plan's defensive posture, CuBench download functions never silently
    return partial data -- they raise loudly instead, so a broken data source is
    caught immediately rather than propagating into features/models.
    """


def fetch_fred_series_curl(
    series_id: str,
    cosd: str = "2009-01-01",
    coed: str = "2025-12-31",
    timeout_s: int = 30,
    retries: int = 3,
) -> pd.DataFrame:
    """Pull one FRED series via `curl` (never `requests`) and return a DataFrame with
    columns ['date', series_id], sorted by date, with rows where the value is FRED's
    missing-value sentinel "." dropped.

    Raises DataDownloadError if curl fails, returns a non-200, or returns an empty/
    malformed CSV.
    """
    url = FRED_URL_TEMPLATE.format(series_id=series_id, cosd=cosd, coed=coed)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["curl", "-s", "-w", "\n__HTTP_STATUS__:%{http_code}", url],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as e:
            raise DataDownloadError(
                "`curl` executable not found on PATH. CuBench requires shelling out "
                "to curl for FRED access (requests times out on this machine)."
            ) from e
        except subprocess.TimeoutExpired as e:
            last_err = e
            logger.warning(f"curl timed out pulling {series_id} (attempt {attempt}/{retries})")
            time.sleep(1.0 * attempt)
            continue

        stdout = result.stdout
        if "__HTTP_STATUS__:" not in stdout:
            last_err = RuntimeError(f"curl produced no status marker; stderr={result.stderr!r}")
            time.sleep(1.0 * attempt)
            continue
        body, _, status_part = stdout.rpartition("__HTTP_STATUS__:")
        http_status = status_part.strip()

        if http_status != "200":
            last_err = RuntimeError(f"HTTP {http_status} for {series_id}: stderr={result.stderr!r}")
            time.sleep(1.0 * attempt)
            continue

        body = body.strip()
        if not body:
            last_err = RuntimeError(f"Empty response body for {series_id}")
            time.sleep(1.0 * attempt)
            continue

        try:
            df = pd.read_csv(StringIO(body))
        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)
            continue

        if df.shape[1] != 2:
            last_err = RuntimeError(
                f"Expected 2 columns (observation_date, {series_id}) for {series_id}, "
                f"got {df.shape[1]}: {df.columns.tolist()}"
            )
            time.sleep(1.0 * attempt)
            continue

        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])
        # FRED encodes missing values as "." in the raw CSV.
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=[series_id]).sort_values("date").reset_index(drop=True)

        if len(df) == 0:
            last_err = RuntimeError(f"{series_id}: all rows were missing/non-numeric after parse")
            time.sleep(1.0 * attempt)
            continue

        return df

    raise DataDownloadError(
        f"Failed to fetch FRED series {series_id} via curl after {retries} attempts: {last_err}"
    )


def verify_fred_coverage(series_id: str, df: pd.DataFrame, expected_first_obs: Optional[str] = None) -> str:
    """Assert the first-observation date of a freshly-pulled FRED series against an
    expected value (plan Section 2.5, test `test_fred_coverage_assertions`; Section 1.4).

    This is the concrete implementation of Risk R2's mitigation (b): FRED coverage has
    silently shrunk before (BAMLH0A0HYM2 went from full history to 754 obs from
    2023-08-21 in April 2026) and must never be allowed to regress unnoticed.

    Raises DataDownloadError if the actual first-obs date is LATER than expected
    (i.e. coverage has regressed/truncated). An actual first-obs date EARLIER than
    expected is fine (conservative surprise) and is logged, not raised.

    Returns the actual first-obs date as an ISO string.
    """
    if expected_first_obs is None:
        expected_first_obs = EXPECTED_FRED_FIRST_OBS.get(series_id)
    if len(df) == 0:
        raise DataDownloadError(f"{series_id}: cannot verify coverage on an empty frame")

    actual_first = df["date"].min()
    actual_first_str = actual_first.strftime("%Y-%m-%d")

    if expected_first_obs is not None:
        expected_dt = pd.Timestamp(expected_first_obs)
        if actual_first > expected_dt:
            raise DataDownloadError(
                f"FRED coverage regression detected for {series_id}: expected first "
                f"obs on or before {expected_first_obs}, got {actual_first_str}. "
                f"This is the failure mode that hit BAMLH0A0HYM2 in April 2026 (full "
                f"history -> 754 obs from 2023-08-21). Do not silently proceed -- "
                f"either the series is genuinely truncated (apply the pre-decided "
                f"fallback per plan Section 1.4) or EXPECTED_FRED_FIRST_OBS needs a "
                f"deliberate, disclosed update."
            )
        logger.info(f"{series_id}: coverage OK, first obs {actual_first_str} "
                    f"(expected <= {expected_first_obs})")
    else:
        logger.info(f"{series_id}: first obs {actual_first_str} (no expected value registered)")

    return actual_first_str


def fetch_yfinance_ohlc(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
    backoff_s: float = 2.0,
) -> pd.DataFrame:
    """Pull full OHLC(V) history for a ticker via yfinance, with retry/backoff.

    Raises DataDownloadError on failure or if the result is empty / missing required
    OHLC columns / has any null Open/High/Low/Close values.
    """
    import yfinance as yf

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        except Exception as e:
            last_err = e
            logger.warning(f"yfinance download failed for {ticker} (attempt {attempt}/{retries}): {e}")
            time.sleep(backoff_s * attempt)
            continue

        if df is None or len(df) == 0:
            last_err = RuntimeError(f"yfinance returned empty data for {ticker}")
            time.sleep(backoff_s * attempt)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        required = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            last_err = RuntimeError(f"{ticker}: missing columns {missing_cols}")
            time.sleep(backoff_s * attempt)
            continue

        if df[required].isnull().any().any():
            n_null = int(df[required].isnull().any(axis=1).sum())
            last_err = RuntimeError(f"{ticker}: {n_null} rows have null OHLC values")
            time.sleep(backoff_s * attempt)
            continue

        df.index.name = "date"
        return df

    raise DataDownloadError(
        f"Failed to fetch usable OHLC data for {ticker} via yfinance after "
        f"{retries} attempts: {last_err}"
    )


def fetch_yfinance_close(
    ticker: str,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
    backoff_s: float = 2.0,
) -> pd.Series:
    """Pull Close-only history for a ticker via yfinance (for series where only the
    daily close is used -- Block C-cross, Block C-macro yfinance legs). Raises
    DataDownloadError on failure/empty result.
    """
    df = fetch_yfinance_ohlc(ticker, start=start, end=end, retries=retries, backoff_s=backoff_s)
    return df["Close"].rename(ticker)


def write_source_log(log: dict, path: str = "data/cubench/raw/data_source_log.json") -> None:
    """Write the data-source log in the same JSON convention as
    data/raw_correlation_check/data_source_log.json, extended with pull timestamp,
    row counts, and date ranges (per Week A1 task Section 6)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info(f"Wrote data source log to {path}")
