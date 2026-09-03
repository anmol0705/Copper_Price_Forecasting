"""GARCH(1,1)-normal and GJR-GARCH(1,1,1)-t, expanding-window fit with monthly refit,
content-hash cache (SHA-256 of return-series bytes + spec dict) at
data/cubench/cache/garch_sigma_{spec}.parquet.

Exact spec: docs/cubench_implementation_plan.md Section 2.4.

Expanding-window fit with monthly refit: at each month-start m, fit on r[0:m] only
(all data strictly before that month begins); then for every day t in month m, produce
sigma_t by FILTERING (not refitting) the fixed parameters forward over r[0:t]. This
mirrors the leakage discipline already established by VMDDecomposerExpanding in
src/data_pipeline.py (refit_interval-style expanding refit).

Caching: keyed by SHA-256 of (return series bytes, spec dict) -- NOT file size or
mtime. This project has a documented history of a file-size-based staleness bug
(todo.txt item 5); a content hash is the fix, and it is the only cache key used here.

Failure handling: if the optimizer fails to converge at a given refit point, the
previous month's fitted parameters are carried forward and a warning is logged. The
total convergence-failure count is tracked and returned to the caller so it can be
reported in the paper.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cubench/cache")


def _content_hash(returns: np.ndarray, spec: dict) -> str:
    """SHA-256 of (return series bytes, spec dict). Content-based, never file-size/mtime
    based -- see module docstring / todo.txt item 5's staleness bug."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(returns, dtype=np.float64).tobytes())
    h.update(json.dumps(spec, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


@dataclass
class GarchFitResult:
    sigma: pd.Series               # 1-step-ahead conditional sigma (return-scale, /100 undone), indexed by date
    resid_z: pd.Series             # standardized residuals r_t / sigma_t
    n_refits: int
    n_convergence_failures: int
    spec: dict
    content_hash: str


def _spec_cache_path(spec_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"garch_sigma_{spec_name}.parquet"


def _fit_one(am, starting_params=None):
    """Fit an arch_model instance, returning the fit result or None on failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = am.fit(disp="off", show_warning=False, starting_values=starting_params)
        except Exception as e:
            logger.warning(f"GARCH fit raised: {e}")
            return None
    if not np.isfinite(res.params.values).all():
        return None
    # arch's convergence flag: res.convergence_flag == 0 means success.
    if getattr(res, "convergence_flag", 0) != 0:
        return None
    return res


def fit_expanding_monthly_refit(
    returns: pd.Series,
    spec_name: str = "garch11_normal",
    o: int = 0,
    dist: str = "normal",
    use_cache: bool = True,
) -> GarchFitResult:
    """Fit GARCH(1,1)-normal (o=0, dist='normal') or GJR-GARCH(1,1,1)-t (o=1, dist='t')
    with an expanding-window fit, refitting once per calendar month on all data up to
    that month's start, then filtering (not refitting) forward through the month.

    `returns` must be a pandas Series of daily log returns (NOT yet scaled by 100 --
    scaling is applied internally, per the plan's exact spec), indexed by trading date,
    sorted ascending, with no NaNs.
    """
    from arch import arch_model

    returns = returns.dropna().sort_index()
    dates = returns.index
    r = returns.values.astype(np.float64)
    n = len(r)

    spec = {
        "spec_name": spec_name,
        "p": 1, "o": o, "q": 1,
        "dist": dist,
        "mean": "Constant",
        "vol": "GARCH",
        "n_obs": n,
        "first_date": str(dates[0].date()) if n else None,
        "last_date": str(dates[-1].date()) if n else None,
    }
    chash = _content_hash(r, spec)

    cache_path = _spec_cache_path(spec_name)
    if use_cache and cache_path.exists():
        try:
            meta = _read_cache_meta(spec_name)
            if meta is not None and meta.get("content_hash") == chash:
                cached = pd.read_parquet(cache_path)
                logger.info(f"GARCH cache hit for {spec_name} ({chash[:12]}...)")
                sigma = cached["sigma"].copy()
                sigma.index = pd.to_datetime(cached["date"])
                resid_z = cached["resid_z"].copy()
                resid_z.index = sigma.index
                return GarchFitResult(
                    sigma=sigma, resid_z=resid_z,
                    n_refits=int(meta.get("n_refits", 0)),
                    n_convergence_failures=int(meta.get("n_convergence_failures", 0)),
                    spec=spec, content_hash=chash,
                )
            else:
                logger.info(f"GARCH cache stale/mismatched for {spec_name}, refitting.")
        except Exception as e:
            logger.warning(f"Failed to read GARCH cache at {cache_path}, refitting: {e}")

    # ---- Real fit: expanding window, monthly refit, filter forward within month. ----
    r100 = r * 100.0  # arch's optimizer conditioning requires percent-scale returns.
    months = pd.Series(dates).dt.to_period("M")
    month_starts = sorted(months.unique())

    sigma_out = np.full(n, np.nan)
    resid_z_out = np.full(n, np.nan)

    last_good_params = None
    n_refits = 0
    n_failures = 0

    # Minimum history required before the first fit is attempted (avoid degenerate
    # fits on a handful of points).
    MIN_HISTORY = 60

    # Built ONCE on the full available array (whatever length this particular build
    # has -- the real 4023-row series, or a perturbation test's truncated one). Reused
    # for every day's `.fix(params, last_obs=day+1)` call -- see the filter-loop
    # comment below for why `last_obs` must be set explicitly per day.
    am_full = arch_model(r100, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)

    for month in month_starts:
        # Index range: rows in `returns` belonging to this calendar month.
        month_mask = (months == month).values
        month_idx = np.where(month_mask)[0]
        if len(month_idx) == 0:
            continue
        m_start = month_idx[0]

        if m_start < MIN_HISTORY:
            # Not enough history yet to fit anything meaningful; leave NaN for this
            # month (burn-in period, expected and reported, not hidden).
            continue

        # Fit on all data strictly BEFORE this month begins: r[0 : m_start].
        train = r100[:m_start]
        am = arch_model(train, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
        res = _fit_one(am, starting_params=last_good_params)
        n_refits += 1

        if res is None:
            n_failures += 1
            if last_good_params is None:
                # No prior fit to carry forward and this fit failed -- skip the month
                # (stays NaN); this can only happen very early in the burn-in period.
                logger.warning(
                    f"GARCH ({spec_name}) convergence failure at month {month} with no "
                    f"prior fit to carry forward -- month left unfilled."
                )
                continue
            logger.warning(
                f"GARCH ({spec_name}) convergence failure at month {month} -- carrying "
                f"forward previous month's fitted parameters."
            )
            params = last_good_params
        else:
            params = res.params
            last_good_params = params

        # Filter (not refit) the fixed parameters forward, one day at a time, using
        # arch's fixed-parameter filtering path with an EXPLICIT `last_obs=day+1` on
        # the single full-length array `am_full` (built once, above the month loop).
        #
        # Why per-day and not "filter the whole month in one call": arch's internal
        # variance backcast (the seed value that initializes the GARCH recursion) is
        # computed as tau = min(75, <effective sample length>), where the effective
        # sample length is the array length actually used by the call (equivalently,
        # `last_obs` when given explicitly, else the raw array length). If the whole
        # month were filtered in a single call whose implicit "last_obs" is the
        # month's own end position, then every day WITHIN that month would have its
        # backcast computed from a tau that depends on how many days are in the
        # month -- i.e. depends on information at the month's end, which is future
        # information relative to earlier days in the same month. This was confirmed
        # empirically (tests/test_leakage.py::test_causal_perturbation caught a
        # ~5e-6 relative discrepancy at an early burn-in index before this fix) and
        # confirmed structurally (arch source: `tau = min(75, resids.shape[0])` where
        # `resids` is the array AFTER first_obs/last_obs slicing). Passing an
        # EXPLICIT `last_obs=day+1` for each individual day makes tau -- and hence
        # every value in the recursion up to and including that day -- depend ONLY
        # on data up to that exact day, regardless of what (if anything) follows it
        # in the underlying array. Benchmarked at <1ms/call, ~4s total for the full
        # 2010-2025 panel across both GARCH specs -- not a performance concern.
        for day_pos in month_idx:
            try:
                fixed_res = am_full.fix(params, last_obs=int(day_pos) + 1)
            except Exception as e:
                logger.warning(f"GARCH ({spec_name}) filter-forward failed at day {day_pos} ({month}): {e}")
                n_failures += 1
                continue
            cond_vol = np.asarray(fixed_res.conditional_volatility)
            sigma_pct = cond_vol[day_pos]
            if not np.isnan(sigma_pct):
                sigma_out[day_pos] = sigma_pct / 100.0  # undo the *100 scaling
                resid_z_out[day_pos] = r[day_pos] / sigma_out[day_pos] if sigma_out[day_pos] != 0 else np.nan

    sigma = pd.Series(sigma_out, index=dates, name="sigma")
    resid_z = pd.Series(resid_z_out, index=dates, name="resid_z")

    if use_cache:
        out_df = pd.DataFrame({
            "date": dates, "sigma": sigma.values, "resid_z": resid_z.values,
        })
        out_df.attrs["content_hash"] = chash
        out_df.attrs["n_refits"] = n_refits
        out_df.attrs["n_convergence_failures"] = n_failures
        try:
            # parquet doesn't preserve .attrs; stash metadata in a sidecar json.
            out_df.to_parquet(cache_path, index=False)
            with open(str(cache_path) + ".meta.json", "w") as f:
                json.dump({
                    "content_hash": chash, "n_refits": n_refits,
                    "n_convergence_failures": n_failures, "spec": spec,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write GARCH cache at {cache_path}: {e}")

    return GarchFitResult(
        sigma=sigma, resid_z=resid_z, n_refits=n_refits,
        n_convergence_failures=n_failures, spec=spec, content_hash=chash,
    )


# Re-check cache validity using the sidecar meta file (parquet .attrs isn't persisted).
def _read_cache_meta(spec_name: str) -> Optional[dict]:
    p = Path(str(_spec_cache_path(spec_name)) + ".meta.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
