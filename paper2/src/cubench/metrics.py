"""Evaluation metrics — QLIKE, pinball loss, CRPS (7-quantile approx), Kupiec coverage
LR test, Mincer-Zarnowitz regression, R2_OOS (Gu-Kelly-Xiu convention), etc.

Full spec: docs/cubench_implementation_plan.md Section 4.2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats


def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    rv_true = np.asarray(rv_true, dtype=float)
    rv_pred = np.clip(np.asarray(rv_pred, dtype=float), 1e-12, None)
    ratio = rv_true / rv_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def crps_from_quantiles(y_true: np.ndarray, preds_by_tau: np.ndarray, taus: list) -> float:
    """CRPS approximated as 2 * mean over tau of pinball_loss(tau), which is
    the standard quantile-grid Riemann approximation to
    CRPS = 2 * integral_0^1 pinball_tau(y, qhat_tau) dtau."""
    losses = [pinball_loss(y_true, preds_by_tau[i], t) for i, t in enumerate(taus)]
    return float(2.0 * np.mean(losses))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    """Gu-Kelly-Xiu R2_OOS: 1 - sum((y-yhat)^2) / sum((y-ybar_train)^2).
    Denominator uses the TRAINING-set mean (not OOS mean) per the convention."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_train_mean) ** 2)
    if den <= 0:
        return float("nan")
    return float(1 - num / den)


def kupiec_uc_test(y_true: np.ndarray, y_quantile_pred: np.ndarray, tau: float) -> dict:
    """Kupiec (1995) unconditional coverage LR test for a single quantile tau.
    H0: empirical exceedance rate (y < qhat) equals tau."""
    y_true = np.asarray(y_true, dtype=float)
    y_quantile_pred = np.asarray(y_quantile_pred, dtype=float)
    n = len(y_true)
    hits = int(np.sum(y_true < y_quantile_pred))  # "exceedances" below quantile
    pi_hat = hits / n if n > 0 else float("nan")
    pi0 = tau
    # LR = -2 ln[ (1-pi0)^(n-x) pi0^x / (1-pihat)^(n-x) pihat^x ]
    if pi_hat in (0.0, 1.0) or n == 0:
        eps = 1e-6
        pi_hat_c = min(max(pi_hat, eps), 1 - eps)
    else:
        pi_hat_c = pi_hat
    x = hits
    ln_l0 = (n - x) * np.log(1 - pi0) + x * np.log(pi0) if 0 < pi0 < 1 else np.nan
    ln_l1 = (n - x) * np.log(1 - pi_hat_c) + x * np.log(pi_hat_c)
    lr = -2 * (ln_l0 - ln_l1)
    p_value = float(1 - _stats.chi2.cdf(lr, df=1)) if np.isfinite(lr) else float("nan")
    return {
        "tau": tau, "n": n, "hits": hits, "empirical_rate": pi_hat,
        "nominal_rate": pi0, "lr_stat": float(lr) if np.isfinite(lr) else float("nan"),
        "p_value": p_value,
    }


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """OLS y = a + b*yhat + e, plus a Wald joint test of (a,b) = (0,1)."""
    y = np.asarray(y_true, dtype=float)
    x = np.asarray(y_pred, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b = beta
        resid = y - X @ beta
        sigma2 = np.sum(resid ** 2) / (n - 2) if n > 2 else float("nan")
        XtX_inv = np.linalg.inv(X.T @ X)
        cov = sigma2 * XtX_inv
        # Wald test H0: (a,b) = (0,1)
        r = np.array([a - 0.0, b - 1.0])
        wald_stat = float(r @ np.linalg.inv(cov) @ r.T)
        p_value = float(1 - _stats.chi2.cdf(wald_stat, df=2))
        return {"a": float(a), "b": float(b), "wald_stat": wald_stat,
                "p_value": p_value, "n": n}
    except np.linalg.LinAlgError:
        return {"a": float("nan"), "b": float("nan"), "wald_stat": float("nan"),
                "p_value": float("nan"), "n": n}
