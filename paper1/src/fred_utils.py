"""FRED (Federal Reserve Economic Data) access for paper1's expanded feature set.

Ported from paper2/src/cubench/data.py and paper2/src/cubench/pit.py, which were
built, tested (tests/test_leakage.py), and run successfully against real FRED data as
part of the CuBench project in this same repository. Reused here rather than
reimplemented, since paper2 already found and worked around a real environment hazard
and a real data-source hazard described below.

Verified environment hazard (carried forward from paper2's own investigation,
docs/cubench_implementation_plan.md Section 1.1): Python's `requests` library reliably
times out against fred.stlouisfed.org on this machine (a TLS/schannel issue), while
`curl` succeeds against the identical URL. All FRED access here therefore shells out to
`curl` via `subprocess`, never `requests`/`pandas_datareader`/`fredapi`.

Verified data-source hazard (paper2's own finding, confirmed independently twice): FRED
silently shrinks coverage for some series without notice -- BAMLH0A0HYM2 (ICE BofA High
Yield OAS) went from full history back to 1996 down to only 754 observations starting
2023-08-21 sometime before April 2026, and an explicit `cosd=` start-date parameter does
NOT recover the missing history (it's a source-licensing restriction, not a query bug).
Because of this, paper1 uses BAA10Y (Moody's Baa - 10Y Treasury spread) as its credit
feature instead of BAMLH0A0HYM2 -- BAA10Y has no such restriction and its full history
back to 2009 was independently verified. `verify_fred_coverage()` below hard-asserts
against a silent regression happening again for any FRED series this project depends on.
"""
from __future__ import annotations

import logging
import subprocess
import time
from io import StringIO
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FRED_URL_TEMPLATE = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={cosd}&coed={coed}"
)

# Deliberately conservative over-estimates of the true BLS/Fed publication-lag
# calendars (erring long can only weaken a leakage-safety guarantee, never
# manufacture a leak). PPIACO and INDPRO per paper2's docs/cubench_implementation_plan.md
# Section 2.3, cross-checked against BLS's own release-schedule conventions.
PUBLICATION_LAG_DAYS = {
    "PPIACO": 30,
    "INDPRO": 22,
}

# Expected first-observation dates, hard-asserted on every fresh pull. A regression
# here (a series silently truncated by FRED, exactly as happened to BAMLH0A0HYM2) must
# raise loudly, not degrade the dataset silently.
EXPECTED_FRED_FIRST_OBS = {
    "DFII10": "2010-01-05",
    "BAA10Y": "2010-01-05",
    "PPIACO": "2009-01-01",
    "INDPRO": "2009-01-01",
}


class DataDownloadError(RuntimeError):
    """Raised when a FRED download fails or returns partial/empty/malformed data.
    Mirrors DataDownloader.download()'s existing fail-loudly convention in
    data_pipeline.py -- never silently return a partial series.
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
                "`curl` executable not found on PATH. FRED access requires shelling "
                "out to curl (requests times out on this machine)."
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
    """Hard-assert the first-observation date of a freshly-pulled FRED series against
    an expected value. Raises DataDownloadError if the actual first-obs date is LATER
    than expected (coverage has regressed/truncated, exactly the BAMLH0A0HYM2 failure
    mode). An actual first-obs date EARLIER than expected is fine (logged, not raised).
    Returns the actual first-obs date as an ISO string.
    """
    if expected_first_obs is None:
        expected_first_obs = EXPECTED_FRED_FIRST_OBS.get(series_id)
    actual_first = df["date"].min()
    actual_first_str = actual_first.strftime("%Y-%m-%d")
    if expected_first_obs is not None:
        expected_ts = pd.Timestamp(expected_first_obs)
        if actual_first > expected_ts:
            raise DataDownloadError(
                f"{series_id}: coverage regression detected. Expected first "
                f"observation on or before {expected_first_obs}, got {actual_first_str}. "
                f"This is the same failure mode that silently truncated BAMLH0A0HYM2 "
                f"to 754 rows in April 2026 -- do not proceed with a truncated series "
                f"without an explicit, disclosed decision to do so."
            )
        elif actual_first < expected_ts:
            logger.info(f"{series_id}: first obs {actual_first_str} is earlier than "
                        f"expected {expected_first_obs} (fine, conservative surprise).")
    return actual_first_str


def compute_available_from(monthly_df: pd.DataFrame, series_id: str, date_col: str = "date") -> pd.DataFrame:
    """Given a monthly DataFrame with a reference date column (FRED's period-start
    convention, e.g. 2015-03-01 for the March observation), compute the
    `available_from` timestamp: reference month END + PUBLICATION_LAG[series_id].
    """
    if series_id not in PUBLICATION_LAG_DAYS:
        raise KeyError(
            f"No PUBLICATION_LAG entry for {series_id!r}. Registered series: "
            f"{list(PUBLICATION_LAG_DAYS.keys())}. Refusing to guess a lag -- add it "
            f"to fred_utils.py::PUBLICATION_LAG_DAYS explicitly."
        )
    lag_days = PUBLICATION_LAG_DAYS[series_id]
    out = monthly_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    month_end = out[date_col] + pd.offsets.MonthEnd(0)
    out["available_from"] = month_end + pd.Timedelta(days=lag_days)
    out = out.sort_values("available_from").reset_index(drop=True)
    return out


def pit_join_monthly(
    daily_calendar: pd.DataFrame,
    monthly_df: pd.DataFrame,
    series_id: str,
    daily_date_col: str = "date",
    monthly_date_col: str = "date",
) -> pd.DataFrame:
    """Point-in-time join of a monthly macro series onto a daily calendar. Never
    replace this with `reindex().ffill()` -- that silently assumes a monthly value was
    known on its reference date itself, which is look-ahead leakage (paper2's own
    correlation-feasibility notebook made exactly this mistake and flagged it as an
    uncorrected risk in its own methodology notes; this function is the fix).

    `daily_calendar` must have a sorted datetime column `daily_date_col`. `monthly_df`
    must have a datetime column `monthly_date_col` with FRED's period-start reference
    dates and a value column named `series_id`. Returns `daily_calendar` with two new
    columns: `available_from` (debug/audit) and `series_id` (the point-in-time value,
    NaN before the series' first available_from).
    """
    daily = daily_calendar.copy()
    daily[daily_date_col] = pd.to_datetime(daily[daily_date_col])
    daily = daily.sort_values(daily_date_col).reset_index(drop=True)

    monthly = monthly_df.rename(columns={monthly_date_col: "date"}) if monthly_date_col != "date" else monthly_df.copy()
    monthly_with_avail = compute_available_from(monthly, series_id, date_col="date")

    merged = pd.merge_asof(
        daily,
        monthly_with_avail[["available_from", series_id]],
        left_on=daily_date_col,
        right_on="available_from",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged
