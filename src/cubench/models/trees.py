"""Tier 3 gradient-boosted trees: lgbm (PRIMARY model), xgboost, catboost, randomforest.
Fixed hyperparameters (not tuned per fold); single Optuna HPO study on 2010-2014 only,
frozen for all 11 OOS years; 5 seeds each. Hyperparameters are read from
configs/cubench.yaml so the frozen (possibly Optuna-adjusted) values live in one place.

Full spec: docs/cubench_implementation_plan.md Section 3.4.

Tree models pass NaN through natively (native missing-value routing) -- no imputation,
per the plan's explicit instruction that tree models handle the aluminum-pre-2014 /
burn-in NaNs structurally rather than via imputation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

_N_PARALLEL_QUANTILES = 6  # the 7 per-tau quantile boosters are independent fits;
# run them concurrently (single-threaded each) rather than sequentially -- this was a
# real throughput bottleneck found during the full-grid run (T2 cells were ~7x slower
# than T1/T3 with no parallelism benefit from n_jobs on any single booster).

warnings.filterwarnings("ignore")

TAUS_DEFAULT = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


class _Base:
    def predict(self, X):
        raise NotImplementedError

    def predict_quantiles(self, X, taus):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBM(_Base):
    def __init__(self, seed: int = 42, horizon: int = 1, target: str = "t1", params: dict = None):
        self.seed = seed
        self.horizon = horizon
        self.target = target
        self.params = dict(params or {})
        self.models = {}

    def _base_params(self, extra=None):
        p = dict(self.params)
        p["random_state"] = self.seed
        p.setdefault("n_jobs", 4)
        p["verbosity"] = -1
        if extra:
            p.update(extra)
        return p

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        Xf, yf = X.loc[mask], y[mask]
        if self.target == "t1":
            m = lgb.LGBMRegressor(objective="regression", **self._base_params())
            m.fit(Xf, yf)
            self.models["mean"] = m
        elif self.target == "t2":
            def _fit_one(tau):
                p = self._base_params(extra={"n_jobs": 1})
                m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **p)
                m.fit(Xf, yf)
                return tau, m
            results = Parallel(n_jobs=_N_PARALLEL_QUANTILES, prefer="threads")(
                delayed(_fit_one)(tau) for tau in TAUS_DEFAULT)
            self.models.update(dict(results))
        else:
            yfi = yf.astype(int)
            m = lgb.LGBMClassifier(objective="binary", **self._base_params())
            m.fit(Xf, yfi)
            self.models["clf"] = m
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.models["mean"].predict(X)

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        raw = np.stack([self.models[tau].predict(X) for tau in taus], axis=0)
        # quantile-crossing repair: isotonic sort across tau axis
        sorted_arr = np.sort(raw, axis=0)
        self.last_crossing_rate = float(np.mean(np.diff(raw, axis=0) < 0))
        return sorted_arr

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.models["clf"].predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGB(_Base):
    def __init__(self, seed: int = 42, horizon: int = 1, target: str = "t1", params: dict = None):
        self.seed = seed
        self.horizon = horizon
        self.target = target
        self.params = dict(params or {})
        self.models = {}

    def _base_params(self):
        p = dict(self.params)
        p["random_state"] = self.seed
        p.setdefault("n_jobs", 4)
        p["verbosity"] = 0
        return p

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        Xf, yf = X.loc[mask], y[mask]
        if self.target == "t1":
            m = xgb.XGBRegressor(objective="reg:squarederror", **self._base_params())
            m.fit(Xf, yf)
            self.models["mean"] = m
        elif self.target == "t2":
            def _fit_one(tau):
                p = dict(self._base_params())
                p["n_jobs"] = 1
                m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau, **p)
                m.fit(Xf, yf)
                return tau, m
            results = Parallel(n_jobs=_N_PARALLEL_QUANTILES, prefer="threads")(
                delayed(_fit_one)(tau) for tau in TAUS_DEFAULT)
            self.models.update(dict(results))
        else:
            yfi = yf.astype(int)
            m = xgb.XGBClassifier(objective="binary:logistic", **self._base_params())
            m.fit(Xf, yfi)
            self.models["clf"] = m
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.models["mean"].predict(X)

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        raw = np.stack([self.models[tau].predict(X) for tau in taus], axis=0)
        self.last_crossing_rate = float(np.mean(np.diff(raw, axis=0) < 0))
        return np.sort(raw, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.models["clf"].predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoost(_Base):
    def __init__(self, seed: int = 42, horizon: int = 1, target: str = "t1", params: dict = None):
        self.seed = seed
        self.horizon = horizon
        self.target = target
        self.params = dict(params or {})
        self.model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        Xf, yf = X.loc[mask].to_numpy(dtype=float), y[mask]
        p = dict(self.params)
        p["random_seed"] = self.seed
        p["verbose"] = False
        p["allow_writing_files"] = False
        p.setdefault("thread_count", 4)
        if self.target == "t1":
            self.model = CatBoostRegressor(loss_function="RMSE", **p)
            self.model.fit(Xf, yf)
        elif self.target == "t2":
            alphas = ",".join(str(t) for t in TAUS_DEFAULT)
            self.model = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **p)
            self.model.fit(Xf, yf)
        else:
            self.model = CatBoostClassifier(loss_function="Logloss", **p)
            self.model.fit(Xf, yf.astype(int))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X.to_numpy(dtype=float))).flatten()

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        raw = np.asarray(self.model.predict(X.to_numpy(dtype=float)))  # (n, ntau)
        raw = raw.T  # (ntau, n)
        self.last_crossing_rate = float(np.mean(np.diff(raw, axis=0) < 0))
        return np.sort(raw, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.to_numpy(dtype=float))[:, 1]


# ---------------------------------------------------------------------------
# Random Forest (+ hand-implemented quantile regression forest for T2)
# ---------------------------------------------------------------------------

class RandomForest(_Base):
    def __init__(self, seed: int = 42, horizon: int = 1, target: str = "t1", params: dict = None):
        self.seed = seed
        self.horizon = horizon
        self.target = target
        self.params = dict(params or {})
        self.model = None
        self._leaf_train_y = None
        self._Xtr_leaf_ids = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        # RF has no native NaN handling: median-impute (train-fold-only) as a documented
        # exception for this one tree model.
        Xf = X.loc[mask].copy()
        med = Xf.median()
        Xf = Xf.fillna(med)
        self._impute_med = med
        yf = y[mask]
        p = dict(self.params)
        p["random_state"] = self.seed
        p.setdefault("n_jobs", 4)
        if self.target == "t1":
            self.model = RandomForestRegressor(**p)
            self.model.fit(Xf, yf)
        elif self.target == "t2":
            self.model = RandomForestRegressor(**p)
            self.model.fit(Xf, yf)
            self._train_leaf_ids = self.model.apply(Xf)  # (n_train, n_trees)
            self._train_y = yf
        else:
            self.model = RandomForestClassifier(**p)
            self.model.fit(Xf, yf.astype(int))
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self._impute_med)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._impute(X))

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        Xi = self._impute(X)
        test_leaf_ids = self.model.apply(Xi)  # (n_test, n_trees)
        n_trees = test_leaf_ids.shape[1]
        out = np.zeros((len(taus), len(X)))
        for i in range(len(X)):
            weights = np.zeros(len(self._train_y))
            for t in range(n_trees):
                same_leaf = self._train_leaf_ids[:, t] == test_leaf_ids[i, t]
                cnt = same_leaf.sum()
                if cnt > 0:
                    weights[same_leaf] += 1.0 / cnt
            weights /= n_trees
            order = np.argsort(self._train_y)
            ys = self._train_y[order]
            ws = weights[order]
            cw = np.cumsum(ws)
            cw = cw / cw[-1] if cw[-1] > 0 else np.linspace(0, 1, len(cw))
            for j, tau in enumerate(taus):
                idx = np.searchsorted(cw, tau)
                idx = min(idx, len(ys) - 1)
                out[j, i] = ys[idx]
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._impute(X))[:, 1]


def _lgbm_params(cfg: dict) -> dict:
    keys = ["n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples",
            "subsample", "subsample_freq", "colsample_bytree", "reg_lambda", "reg_alpha"]
    return {k: cfg[k] for k in keys if k in cfg}


def _xgb_params(cfg: dict) -> dict:
    out = {
        "max_depth": cfg.get("max_depth", 5),
        "learning_rate": cfg.get("eta", 0.03),
        "n_estimators": cfg.get("n_estimators", 600),
        "subsample": cfg.get("subsample", 0.8),
        "colsample_bytree": cfg.get("colsample_bytree", 0.7),
        "min_child_weight": cfg.get("min_child_weight", 20),
        "reg_lambda": cfg.get("lambda", 1.0),
    }
    return out


def _cat_params(cfg: dict) -> dict:
    return {
        "depth": cfg.get("depth", 5),
        "learning_rate": cfg.get("learning_rate", 0.03),
        "iterations": cfg.get("iterations", 600),
        "l2_leaf_reg": cfg.get("l2_leaf_reg", 3),
    }


def _rf_params(cfg: dict) -> dict:
    return {
        "n_estimators": cfg.get("n_estimators", 500),
        "max_depth": cfg.get("max_depth", 12),
        "min_samples_leaf": cfg.get("min_samples_leaf", 20),
        "max_features": cfg.get("max_features", "sqrt"),
    }


def lgbm_factory(target: str, cfg: dict):
    p = _lgbm_params(cfg)
    def factory(seed: int = 42, horizon: int = 1):
        return LGBM(seed=seed, horizon=horizon, target=target, params=p)
    return factory


def xgboost_factory(target: str, cfg: dict):
    p = _xgb_params(cfg)
    def factory(seed: int = 42, horizon: int = 1):
        return XGB(seed=seed, horizon=horizon, target=target, params=p)
    return factory


def catboost_factory(target: str, cfg: dict):
    p = _cat_params(cfg)
    def factory(seed: int = 42, horizon: int = 1):
        return CatBoost(seed=seed, horizon=horizon, target=target, params=p)
    return factory


def randomforest_factory(target: str, cfg: dict):
    p = _rf_params(cfg)
    def factory(seed: int = 42, horizon: int = 1):
        return RandomForest(seed=seed, horizon=horizon, target=target, params=p)
    return factory
