"""Tier 1 econometric models: har_rv (headline benchmark), har_rv_q, garch11, gjr_garch,
arima. Full spec: docs/cubench_implementation_plan.md Section 3.2.

Design notes:
- har_rv / har_rv_q: OLS (Log-HAR) refit fresh on each fold's training window, using
  the already-computed leakage-safe `d_logrv_*`/`d_jump_22` feature columns as regressors.
  Newey-West(h+5) HAC standard errors via statsmodels `cov_type='HAC'`.
- garch11 / gjr_garch reuse the ALREADY-FITTED `d_garch_sigma` / `d_gjr_sigma` feature
  columns from Phase 2's garch.py (conditional daily vol, point-in-time safe). We do not
  refit an arch model here; "prediction" is a documented rescaling of that sigma to the
  target's log-RV / cumulative-return-quantile scale for horizon h, assuming the
  GARCH(1,1)/GJR conditional variance forecast is approximately constant over the short
  horizons h in {1,5,22} (a standard simplification for h-step aggregation of a
  near-unit-persistence GARCH process). Quantiles: Normal for GARCH, Student-t (df=8,
  a documented reasonable approximation absent an easily-accessible per-fold fitted df)
  for GJR.
- arima: fit ARIMA(p,0,q) once per fold (order by AIC over p,q in {0,1,2}) on a
  reconstructed daily-return series (d_logrv/target columns do not carry raw returns;
  we reconstruct r_t = a_r_lag1(t+1) within the training fold only -- leakage-safe since
  only training rows are used). Consistent with the plan's "refit per fold on that fold's
  training data only" (not per-day), the resulting h-step-ahead forecast from the end of
  the training window is used as a CONSTANT prediction across every OOS day in that fold.
  This is a known/expected weak-baseline behaviour explicitly flagged in the plan
  ("expect textbook behaviour: strong in-sample, weak OOS").
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class _Base:
    def predict(self, X):
        raise NotImplementedError

    def predict_quantiles(self, X, taus):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class HARRV(_Base):
    REGRESSORS = ["d_logrv_1", "d_logrv_5", "d_logrv_22"]

    def __init__(self, seed: int = 0, horizon: int = 1, quarterly: bool = False):
        self.seed = seed
        self.horizon = horizon
        self.quarterly = quarterly
        self.regressors = list(self.REGRESSORS)
        if quarterly:
            self.regressors += ["d_logrv_66", "d_jump_22"]
        self._res = None
        self._resid_std = 0.0

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X[self.regressors].copy()
        Z = Z.fillna(Z.median())
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        mask = ~np.isnan(y)
        Z = self._design(X.loc[mask])
        yv = y[mask]
        ols = sm.OLS(yv, Z)
        maxlags = self.horizon + 5
        self._res = ols.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        self._resid_std = float(np.std(self._res.resid))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._design(X)
        return np.asarray(self._res.predict(Z))

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        # HAR is fit on T1 (log RV) scale in this codebase's usage; when used for T2
        # (cumulative return) the caller fits a separate instance on y2. Quantiles here
        # are Normal(point_forecast, resid_std) generically applicable to whichever
        # target this instance was fit on.
        mu = self.predict(X)
        z = sstats.norm.ppf(taus)
        return np.stack([mu + zi * self._resid_std for zi in z], axis=0)


class GARCHFeature(_Base):
    """Rescales an already-fitted conditional-sigma feature column to the target scale."""

    def __init__(self, seed: int = 0, horizon: int = 1, sigma_col: str = "d_garch_sigma",
                 dist: str = "normal", t_df: float = 8.0):
        self.seed = seed
        self.horizon = horizon
        self.sigma_col = sigma_col
        self.dist = dist
        self.t_df = t_df

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self

    def _ann_sigma(self, X: pd.DataFrame) -> np.ndarray:
        s = X[self.sigma_col]
        s = s.fillna(s.median()).to_numpy(dtype=float)
        return s * np.sqrt(252.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ann_sigma = np.clip(self._ann_sigma(X), 1e-8, None)
        return np.log(ann_sigma)  # T1 scale: y1 = ln(ann RV) form used here

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        s = X[self.sigma_col]
        s = s.fillna(s.median()).to_numpy(dtype=float)
        h_scale = s * np.sqrt(self.horizon)
        if self.dist == "normal":
            q = sstats.norm.ppf(taus)
        else:
            adj = np.sqrt((self.t_df - 2) / self.t_df) if self.t_df > 2 else 1.0
            q = sstats.t.ppf(taus, df=self.t_df) * adj
        return np.stack([qi * h_scale for qi in q], axis=0)


class ARIMAModel(_Base):
    def __init__(self, seed: int = 0, horizon: int = 1):
        self.seed = seed
        self.horizon = horizon
        self._pred_t1 = 0.0
        self._pred_t3_proba = 0.5

    def _reconstruct_returns(self, X: pd.DataFrame) -> np.ndarray:
        if "a_r_lag1" not in X.columns:
            return np.array([])
        r = X["a_r_lag1"].shift(-1).dropna().to_numpy(dtype=float)
        return r

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        r = self._reconstruct_returns(X)
        if len(r) < 50:
            self._pred_t1, self._pred_t3_proba = 0.0, 0.5
            return self
        best_aic, best_res = np.inf, None
        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:
                    continue
                try:
                    res = ARIMA(r, order=(p, 0, q)).fit()
                    if res.aic < best_aic:
                        best_aic, best_res = res.aic, res
                except Exception:
                    continue
        if best_res is None:
            self._pred_t1, self._pred_t3_proba = float(np.log(np.std(r) * np.sqrt(252) + 1e-8)), 0.5
            return self
        fc = best_res.get_forecast(steps=self.horizon)
        mean_path = np.asarray(fc.predicted_mean)
        var_path = np.asarray(fc.var_pred_mean)
        cum_mean = float(np.sum(mean_path))
        cum_var = float(np.sum(var_path))
        ann_var = (252.0 / self.horizon) * cum_var
        self._pred_t1 = float(0.5 * np.log(max(ann_var, 1e-12)))
        self._pred_t3_proba = 0.9 if cum_mean > 0 else 0.1
        self._cum_mean = cum_mean
        self._cum_std = float(np.sqrt(max(cum_var, 1e-12)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._pred_t1)

    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray:
        mu = getattr(self, "_cum_mean", 0.0)
        sd = getattr(self, "_cum_std", 0.01)
        z = sstats.norm.ppf(taus)
        return np.stack([np.full(len(X), mu + zi * sd) for zi in z], axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._pred_t3_proba)


def har_rv_factory(seed: int = 0, horizon: int = 1):
    return HARRV(seed=seed, horizon=horizon, quarterly=False)


def har_rv_q_factory(seed: int = 0, horizon: int = 1):
    return HARRV(seed=seed, horizon=horizon, quarterly=True)


def garch11_factory(seed: int = 0, horizon: int = 1):
    return GARCHFeature(seed=seed, horizon=horizon, sigma_col="d_garch_sigma", dist="normal")


def gjr_garch_factory(seed: int = 0, horizon: int = 1):
    return GARCHFeature(seed=seed, horizon=horizon, sigma_col="d_gjr_sigma", dist="t", t_df=8.0)


def arima_factory(seed: int = 0, horizon: int = 1):
    return ARIMAModel(seed=seed, horizon=horizon)
