"""Walk-forward fold generation -- expanding window, annual refit, 11 non-overlapping
OOS years (2015-2025), embargo E = h + 5 trading days purged from the END of the
training window (purging, not a gap inside the OOS block). Serializes to
results/cubench/folds.json.

Full spec: docs/cubench_implementation_plan.md Section 4.1.

Fold k (k=0..10): train on [2010-01-04, (2014+k)-12-31 - E], evaluate on
[(2015+k)-01-01, (2015+k)-12-31]. Embargo depends on h because y_h(t) spans t+1..t+h;
without purging the training window's last h targets would overlap the first OOS days.
Embargo is expressed in TRADING DAYS (removed from the training side only), computed
against copper's actual trading calendar -- not calendar days.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INITIAL_TRAIN_START = "2010-01-04"
INITIAL_TRAIN_END = "2014-12-31"
OOS_YEARS = list(range(2015, 2026))  # 2015..2025 inclusive, 11 years
HORIZONS = (1, 5, 22)


def embargo_trading_days(h: int) -> int:
    """E = h + 5 trading days, per plan Section 4.1."""
    return h + 5


def make_folds(trading_calendar: pd.DatetimeIndex) -> list[dict]:
    """Build the 11 expanding-window folds, one embargo variant per horizon h in
    {1,5,22}, against the REAL copper trading calendar (not a calendar-day
    approximation -- embargo is defined in trading days).

    `trading_calendar` must be a sorted, deduplicated DatetimeIndex of copper's actual
    trading dates (e.g. features.py's build_raw_panel()['date']).

    Returns a list of 11 dicts:
      {
        "fold": k,
        "oos_year": year,
        "train_start": iso date,
        "train_end_unembargoed": iso date (= Dec 31 of year (2014+k), or the last
            trading day on/before it),
        "oos_start": iso date, "oos_end": iso date,
        "embargo": {"h1": {"trading_days": 6, "train_end": iso}, "h5": {...}, "h22": {...}},
      }
    """
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(trading_calendar).unique()))
    if len(cal) == 0:
        raise ValueError("Empty trading calendar")

    folds = []
    train_start = pd.Timestamp(INITIAL_TRAIN_START)

    for k, oos_year in enumerate(OOS_YEARS):
        train_end_year = 2014 + k  # k=0 -> 2014, ..., k=10 -> 2024
        train_end_target = pd.Timestamp(f"{train_end_year}-12-31")
        # Snap to the last actual trading day on/before this date.
        train_end_candidates = cal[cal <= train_end_target]
        if len(train_end_candidates) == 0:
            raise ValueError(f"No trading days on/before {train_end_target} in calendar")
        train_end_unembargoed = train_end_candidates[-1]

        oos_start_target = pd.Timestamp(f"{oos_year}-01-01")
        oos_end_target = pd.Timestamp(f"{oos_year}-12-31")
        oos_candidates = cal[(cal >= oos_start_target) & (cal <= oos_end_target)]
        if len(oos_candidates) == 0:
            raise ValueError(f"No trading days in OOS year {oos_year}")
        oos_start = oos_candidates[0]
        oos_end = oos_candidates[-1]

        # Position of train_end_unembargoed within the calendar, for trading-day-count
        # embargo purging.
        train_end_pos = cal.get_loc(train_end_unembargoed)

        embargo_info = {}
        for h in HORIZONS:
            e = embargo_trading_days(h)
            purged_pos = train_end_pos - e
            if purged_pos < 0:
                raise ValueError(
                    f"Embargo of {e} trading days at fold {k} (h={h}) would purge before "
                    f"the start of the calendar -- initial training block too short."
                )
            embargo_train_end = cal[purged_pos]
            embargo_info[f"h{h}"] = {
                "trading_days": e,
                "train_end": str(embargo_train_end.date()),
            }

        folds.append({
            "fold": k,
            "oos_year": oos_year,
            "train_start": str(train_start.date()),
            "train_end_unembargoed": str(train_end_unembargoed.date()),
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "embargo": embargo_info,
        })

    return folds


def save_folds(folds: list[dict], path: str = "results/cubench/folds.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "scheme": "expanding_window_annual_refit",
            "initial_train_start": INITIAL_TRAIN_START,
            "initial_train_end_nominal": INITIAL_TRAIN_END,
            "oos_years": OOS_YEARS,
            "horizons": list(HORIZONS),
            "embargo_formula": "h + 5 trading days, purged from the END of the training window",
            "folds": folds,
        }, f, indent=2)
    logger.info(f"Wrote {len(folds)} folds to {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.cubench.features import build_raw_panel

    logging.basicConfig(level=logging.INFO)
    panel = build_raw_panel()
    folds = make_folds(panel["date"])
    save_folds(folds)
    for fd in folds:
        print(fd["fold"], fd["oos_year"], fd["train_start"], "->", fd["train_end_unembargoed"],
              "| OOS", fd["oos_start"], "->", fd["oos_end"],
              "| embargo h22 train_end:", fd["embargo"]["h22"]["train_end"])
