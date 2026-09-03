"""Point-in-time monthly-macro join.

Per docs/cubench_implementation_plan.md Section 2.3, this is flagged as *the single most
important leakage control in the whole project*. A monthly macro series (e.g. PPIACO)
is published well after its reference month ends -- e.g. the December PPI print is
released in late January, not on December 1st or December 31st. Naively reindexing a
monthly series onto a daily calendar and forward-filling (`reindex().ffill()`) silently
assumes the value was known on the reference date itself, which is look-ahead. The
correlation-feasibility notebook (notebooks/correlation_feasibility_analysis.ipynb) did
exactly this and flagged it in its own methodology notes as an uncorrected look-ahead
risk (Risk R3 in the implementation plan). CuBench fixes it here.

Procedure (mandatory, plan Section 2.3):
1. Each monthly observation has a `reference_date` (FRED's period start, e.g. 2015-03-01).
2. available_from = reference_month_end + PUBLICATION_LAG[series]  (conservative
   over-estimates: PPIACO +30 days, INDPRO +22 days, BSCICP03CNM665S +45 days).
3. Merge onto the daily calendar with
   `pandas.merge_asof(daily, monthly, left_on='date', right_on='available_from',
   direction='backward', allow_exact_matches=True)`.
4. Never use reindex().ffill() on the reference date.
5. Revisions are not modelled (FRED serves latest-vintage data) -- disclosed limitation,
   bounded by ablation A4-vs-A3 / A7 at the model-evaluation stage, not here.
"""
from __future__ import annotations

import pandas as pd

from src.cubench.data import PUBLICATION_LAG_DAYS


def compute_available_from(monthly_df: pd.DataFrame, series_id: str, date_col: str = "date") -> pd.DataFrame:
    """Given a monthly DataFrame with a reference date column (FRED's period-start
    convention, e.g. 2015-03-01 for the March observation), compute the
    `available_from` timestamp: reference month END + PUBLICATION_LAG[series_id].

    Returns a copy of monthly_df with an added 'available_from' column, sorted by it
    (merge_asof requires the right-side key to be sorted).
    """
    if series_id not in PUBLICATION_LAG_DAYS:
        raise KeyError(
            f"No PUBLICATION_LAG entry for {series_id!r}. Registered series: "
            f"{list(PUBLICATION_LAG_DAYS.keys())}. Refusing to guess a lag -- add it "
            f"to src/cubench/data.py::PUBLICATION_LAG_DAYS explicitly."
        )
    lag_days = PUBLICATION_LAG_DAYS[series_id]
    out = monthly_df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    # Reference month END = the last calendar day of the month containing `date_col`.
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
    """Point-in-time join of a monthly macro series onto a daily calendar.

    `daily_calendar` must have a sorted datetime column `daily_date_col` (e.g. copper's
    trading calendar). `monthly_df` must have a datetime column `monthly_date_col` with
    FRED's period-start reference dates and a value column named `series_id`.

    Returns `daily_calendar` with two new columns: `available_from` (debug/audit) and
    `series_id` (the point-in-time value, NaN before the series' first available_from).

    This function must NEVER be replaced with `reindex().ffill()` -- see module
    docstring and `tests/test_leakage.py::test_pit_monthly_join` (Week A2), which
    asserts the naive join produces a *different* (wrong) series, proving this function
    is doing real work, not a no-op.

    --- Inline sanity-check example (also exercised by the smoke test at the bottom of
    this module when run directly) ---

    Suppose PPIACO's December observation (reference_date=2015-12-01, value=101) is
    published with a conservative +30-day lag. Its available_from is
    2015-12-31 + 30 days = 2016-01-30. A trading day on 2016-01-15 must NOT see the
    December value yet (it should still see November's, available from ~2015-12-30); a
    trading day on 2016-01-30 or later must see it.

    The naive `reindex().ffill()` approach instead matches on the raw reference date,
    so as soon as a later reference-month row's date (e.g. 2016-01-01 for the January
    observation, value=102) is <= the daily date, it leaks that value -- weeks before
    it would actually be released. On 2016-01-15 the naive join already shows 102
    (January's own value, dated to the 1st of its reference month) while the correct
    PIT join still shows 100 (November's value, the latest one actually available by
    that date). This divergence is exactly what test_pit_monthly_join checks.
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


def _naive_reindex_ffill_join(
    daily_calendar: pd.DataFrame,
    monthly_df: pd.DataFrame,
    series_id: str,
    daily_date_col: str = "date",
    monthly_date_col: str = "date",
) -> pd.Series:
    """The INCORRECT/naive approach, implemented here ONLY so tests (and this module's
    own smoke check) can assert the PIT join produces a genuinely different result.
    Never use this for actual feature construction -- see module docstring.
    """
    daily = daily_calendar.copy()
    daily[daily_date_col] = pd.to_datetime(daily[daily_date_col])
    monthly = monthly_df.copy()
    monthly[monthly_date_col] = pd.to_datetime(monthly[monthly_date_col])
    s = monthly.set_index(monthly_date_col)[series_id].sort_index()
    reindexed = s.reindex(pd.to_datetime(daily[daily_date_col]), method="ffill")
    return reindexed.values


if __name__ == "__main__":
    # Minimal, self-contained sanity check matching the docstring example above.
    monthly = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=13, freq="MS"),  # Jan 2015 .. Jan 2016
        "PPIACO": [90 + i for i in range(13)],
    })
    daily = pd.DataFrame({"date": pd.bdate_range("2015-01-01", "2016-03-01")})

    pit = pit_join_monthly(daily, monthly, "PPIACO")
    naive = _naive_reindex_ffill_join(daily, monthly, "PPIACO")

    # December 2015 obs (value=101, reference_date=2015-12-01) -> available_from
    # = 2015-12-31 + 30d = 2016-01-30. January 2016 obs (value=102,
    # reference_date=2016-01-01) -> available_from = 2016-01-31 + 30d = 2016-03-02.
    check_date = pd.Timestamp("2016-01-15")  # before Dec's available_from -> must NOT see Dec/Jan value
    pit_val = pit.loc[pit["date"] == check_date, "PPIACO"].iloc[0]
    naive_val = naive[(daily["date"] == check_date).values][0]
    print(f"On {check_date.date()}: PIT-join value={pit_val}, naive-ffill value={naive_val}")
    # Naive reindex().ffill() matches on the raw reference date: since the January
    # reference date (2016-01-01) is already <= 2016-01-15, it leaks January's own
    # value (102) as if it were known on January 15th -- roughly six weeks before
    # it would actually be released. The PIT join correctly still shows November's
    # value (100), because December's is not available until 2016-01-30.
    assert naive_val == 102, "naive join sanity check itself broken -- expected it to leak the Jan value here"
    assert pit_val == 100, "PIT join should still show November's value on 2016-01-15"
    assert pit_val != naive_val, "PIT join and naive join must diverge -- if they match, PIT join is a no-op"
    print("Sanity check passed: naive reindex().ffill() leaks the January reference value "
          "on 2016-01-15 (~6 weeks early); the PIT join correctly still shows November's "
          "value, withholding December's until its available_from date of 2016-01-30.")
