"""Tier 2 linear ML: elasticnet (ElasticNetCV / QuantileRegressor / LogisticRegression
elasticnet), Pipeline-wrapped SimpleImputer(median, train-fold-only) + StandardScaler
(train-fold-only) to handle both missing-value burn-in (aluminum pre-2014, GARCH/regime
burn-in) and the leakage risk of fitting scalers on OOS data.

Full spec: docs/cubench_implementation_plan.md Section 3.3.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class _Base:
    def predict(self, X):
        raise NotImplementedError

    def predict_quantiles(self, X, taus):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class ElasticNetT1(_Base):
    def __init__(self, seed: int = 0, horizon: int = 1):
        self.seed = seed
        self.horizon = horizon
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.0], cv=3,
                                    max_iter=5000, random_state=seed, n_jobs=1)),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        self.pipe.fit(X.loc[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X)


class ElasticNetT2(_Base):
    def __init__(self, seed: int = 0, horizon: int = 1):
        self.seed = seed
        self.horizon = horizon
        self.taus = None
        self.pipes = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        Xf, yf = X.loc[mask], y[mask]
        self._Xf, self._yf = Xf, yf
        return self

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        out = []
        for tau in taus:
            if tau not in self.pipes:
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", QuantileRegressor(quantile=tau, alpha=1e-4, solver="highs")),
                ])
                pipe.fit(self._Xf, self._yf)
                self.pipes[tau] = pipe
            out.append(self.pipes[tau].predict(X))
        return np.stack(out, axis=0)


class ElasticNetT3(_Base):
    def __init__(self, seed: int = 0, horizon: int = 1):
        self.seed = seed
        self.horizon = horizon
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                          max_iter=2000, random_state=seed)),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        yf = y[mask].astype(int)
        Xf = X.loc[mask]
        if len(np.unique(yf)) < 2:
            self._const = float(yf.mean()) if len(yf) else 0.5
            self._fitted = False
        else:
            self.pipe.fit(Xf, yf)
            self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "_fitted", True):
            return np.full(len(X), self._const)
        return self.pipe.predict_proba(X)[:, 1]


def elasticnet_factory(target: str):
    def factory(seed: int = 0, horizon: int = 1):
        if target == "t1":
            return ElasticNetT1(seed=seed, horizon=horizon)
        elif target == "t2":
            return ElasticNetT2(seed=seed, horizon=horizon)
        else:
            return ElasticNetT3(seed=seed, horizon=horizon)
    return factory
