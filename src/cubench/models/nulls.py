"""Tier 0 statistical nulls: null_persist, null_rollmean, null_zero, null_majority,
null_always_down.

Full spec: docs/cubench_implementation_plan.md Section 3.1.

Design notes on leakage-safety / data availability:
- null_persist T1 uses the already-computed `d_logrv_{h}` feature (yesterday's backward
  realized-vol-proxy at matching horizon) as the forecast of the forward target -- this is
  a feature column present in X at prediction time, no extra data needed.
- null_persist T2 needs an empirical sample of daily returns. The feature matrix does not
  carry a raw daily-return column, only `a_r_lag1`/`a_r_lag2`. We use the training fold's
  own `a_r_lag1` column (trailing up to 252 obs) as a leakage-safe empirical sample of
  daily returns, scaled by sqrt(h) for the h-day cumulative quantile (documented
  approximation; iid scaling, no drift/autocorrelation adjustment).
- null_persist T3 uses `sign(a_r_lag1)` of the test row itself (most recent observed
  return as of prediction time -- an input feature, not a leak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _Base:
    def predict(self, X):
        raise NotImplementedError

    def predict_quantiles(self, X, taus):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class NullPersist(_Base):
    def __init__(self, seed: int = 0, horizon: int = 1):
        self.seed = seed
        self.horizon = horizon
        self._train_r_tail = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        col = "a_r_lag1" if "a_r_lag1" in X.columns else None
        if col is not None:
            tail = X[col].dropna().to_numpy()
            self._train_r_tail = tail[-252:] if len(tail) > 0 else np.array([0.0])
        else:
            self._train_r_tail = np.array([0.0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # d_logrv_h = ln(mean(RV, h)) is on the UNANNUALIZED daily-variance log scale,
        # while y1_h = ln(sqrt(252/h * sum(RV,h))) is annualized log-vol. Since
        # sum(RV,h) = h * mean(RV,h): y1_h = 0.5*ln(252/h) + 0.5*ln(h*exp(d_logrv_h))
        #           = 0.5*ln(252) + 0.5*d_logrv_h  (the h terms cancel exactly).
        # Verified empirically: 0.5*ln(252) + 0.5*mean(d_logrv_1) == mean(y1_h1) to 3dp.
        col = f"d_logrv_{self.horizon}"
        if col in X.columns:
            d = X[col].fillna(X[col].median()).to_numpy(dtype=float)
            return 0.5 * np.log(252.0) + 0.5 * d
        return np.zeros(len(X))

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        scale = np.sqrt(self.horizon)
        base_q = np.quantile(self._train_r_tail, taus)
        return np.stack([np.full(len(X), q * scale) for q in base_q], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        col = "a_r_lag1"
        if col in X.columns:
            r = X[col].fillna(0.0).to_numpy(dtype=float)
        else:
            r = np.zeros(len(X))
        return (r > 0).astype(float) * 0.9 + (r <= 0).astype(float) * 0.1


class NullRollmean(_Base):
    def __init__(self, seed: int = 0):
        self.seed = seed
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self._mean = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)


class NullZero(_Base):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        return self

    def predict_quantiles(self, X, taus):
        return np.zeros((len(taus), len(X)))

    def predict_proba(self, X):
        return np.ones(len(X))  # always-up


class NullMajority(_Base):
    def __init__(self, seed: int = 0, always_down: bool = False):
        self.seed = seed
        self.always_down = always_down
        self._majority = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if self.always_down:
            self._majority = 0.0
        else:
            vals, counts = np.unique(y[~np.isnan(y)], return_counts=True)
            self._majority = float(vals[np.argmax(counts)]) if len(vals) else 1.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 0.95 if self._majority >= 0.5 else 0.05
        return np.full(len(X), p)


def null_persist_factory(seed: int = 0, horizon: int = 1):
    return NullPersist(seed=seed, horizon=horizon)


def null_rollmean_factory(seed: int = 0, horizon: int = 1):
    return NullRollmean(seed=seed)


def null_zero_factory(seed: int = 0, horizon: int = 1):
    return NullZero(seed=seed)


def null_majority_factory(seed: int = 0, horizon: int = 1):
    return NullMajority(seed=seed, always_down=False)


def null_always_down_factory(seed: int = 0, horizon: int = 1):
    return NullMajority(seed=seed, always_down=True)
