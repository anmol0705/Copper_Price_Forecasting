"""Statistical significance tests — dm_test_hln (Diebold-Mariano with Harvey-Leybourne-
Newbold small-sample correction, Student-t reference), pesaran_timmermann (hand-
implemented, with degenerate-constant-forecast guard), Holm-Bonferroni wrapper over
statsmodels.stats.multitest.multipletests.

This supersedes (does not modify) src/utils.py::diebold_mariano_test, which lacks the
HLN correction and uses a normal reference distribution — both wrong for the 11-fold,
h>1 CuBench setting.

Full spec: docs/cubench_implementation_plan.md Section 4.3.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy import stats as _stats
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction
# ---------------------------------------------------------------------------

def _autocov(d: np.ndarray, k: int) -> float:
    """Sample autocovariance at lag k (biased, /n denominator), matching the
    plan's gamma_k = autocov(d, k) used inside V = (gamma_0 + 2*sum gamma_k)/n."""
    n = len(d)
    dbar = d.mean()
    if k == 0:
        return float(np.sum((d - dbar) ** 2) / n)
    return float(np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / n)


def dm_test_hln(errors1: np.ndarray, errors2: np.ndarray, h: int = 1,
                 loss: str = "se") -> dict:
    """Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample
    correction, Student-t(n-1) reference distribution.

    Parameters
    ----------
    errors1, errors2 : arrays of *losses already computed* per observation for
        model 1 and model 2 (e.g. squared error, QLIKE contribution, pinball
        loss) UNLESS loss='se', in which case errors1/errors2 are raw forecast
        errors (y - yhat) and squared-error loss is applied internally.
    h : forecast horizon (bandwidth = h-1 lag truncation, per the plan).
    loss : 'se' (square errors1/errors2 internally) or 'raw' (errors1/errors2
        are already per-observation losses, e.g. QLIKE/pinball contributions;
        d_t = errors1 - errors2 directly).

    Returns dict with dm_stat (uncorrected), hln_stat, p_value (two-sided,
    Student-t(n-1)), n, h, mean_diff.
    """
    e1 = np.asarray(errors1, dtype=float).flatten()
    e2 = np.asarray(errors2, dtype=float).flatten()
    if len(e1) != len(e2):
        raise ValueError("errors1 and errors2 must have equal length")
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)
    if n < 2:
        return {"dm_stat": np.nan, "hln_stat": np.nan, "p_value": np.nan,
                "n": n, "h": h, "mean_diff": np.nan}

    if loss == "se":
        L1 = e1 ** 2
        L2 = e2 ** 2
    else:
        L1 = e1
        L2 = e2
    d = L1 - L2

    dbar = float(np.mean(d))
    gamma0 = _autocov(d, 0)
    gamma_sum = 0.0
    for k in range(1, max(h - 1, 0) + 1):
        if k >= n:
            break
        gamma_sum += 2.0 * _autocov(d, k)
    V = (gamma0 + gamma_sum) / n

    if V <= 0 or not np.isfinite(V):
        return {"dm_stat": np.nan, "hln_stat": np.nan, "p_value": np.nan,
                "n": n, "h": h, "mean_diff": dbar}

    dm_stat = dbar / math.sqrt(V)

    hln_mult_arg = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln_mult_arg < 0:
        hln_mult_arg = 0.0
    hln_stat = dm_stat * math.sqrt(hln_mult_arg)

    p_value = 2.0 * (1.0 - _stats.t.cdf(abs(hln_stat), df=n - 1))

    return {
        "dm_stat": float(dm_stat),
        "hln_stat": float(hln_stat),
        "p_value": float(p_value),
        "n": int(n),
        "h": int(h),
        "mean_diff": dbar,
    }


# ---------------------------------------------------------------------------
# Pesaran-Timmermann (1992) directional-accuracy test
# ---------------------------------------------------------------------------

def pesaran_timmermann(y_true_signs: np.ndarray, y_pred_signs: np.ndarray) -> dict:
    """Pesaran-Timmermann test of directional predictive ability.

    y_true_signs, y_pred_signs: arrays coercible to {0,1} "up" indicators
    (either already 0/1, or +1/-1, or booleans — anything where > a midpoint
    means "up"). Internally converted to 1[x > 0] semantics by the caller
    passing indicator arrays directly is safest; here we accept 0/1 or bool.

    Returns dict with p_hat, p_star, var_p_hat, var_p_star, pt_stat, p_value,
    degenerate (bool), flag (str or None).
    """
    y = np.asarray(y_true_signs).astype(float).flatten()
    x = np.asarray(y_pred_signs).astype(float).flatten()
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 2:
        return {"p_hat": np.nan, "p_star": np.nan, "var_p_hat": np.nan,
                "var_p_star": np.nan, "pt_stat": np.nan, "p_value": np.nan,
                "n": n, "degenerate": True, "flag": "degenerate_constant_forecast"}

    # Degenerate guard: constant predicted class (P_x in {0,1})
    p_x = float(np.mean(x))
    p_y = float(np.mean(y))
    if p_x <= 0.0 or p_x >= 1.0 or p_y <= 0.0 or p_y >= 1.0:
        return {"p_hat": float(np.mean(x == y)), "p_star": np.nan,
                "var_p_hat": np.nan, "var_p_star": np.nan, "pt_stat": np.nan,
                "p_value": np.nan, "n": n, "degenerate": True,
                "flag": "degenerate_constant_forecast"}

    p_hat = float(np.mean((x == y).astype(float)))
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)

    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = (
        (2 * p_y - 1) ** 2 * p_x * (1 - p_x) / n
        + (2 * p_x - 1) ** 2 * p_y * (1 - p_y) / n
        + 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n ** 2
    )

    denom = var_p_hat - var_p_star
    if denom <= 0 or not np.isfinite(denom):
        return {"p_hat": p_hat, "p_star": p_star, "var_p_hat": var_p_hat,
                "var_p_star": var_p_star, "pt_stat": np.nan, "p_value": np.nan,
                "n": n, "degenerate": True, "flag": "degenerate_constant_forecast"}

    pt_stat = (p_hat - p_star) / math.sqrt(denom)
    p_value = 2.0 * (1.0 - _stats.norm.cdf(abs(pt_stat)))

    return {
        "p_hat": p_hat, "p_star": p_star, "var_p_hat": var_p_hat,
        "var_p_star": var_p_star, "pt_stat": float(pt_stat),
        "p_value": float(p_value), "n": n, "degenerate": False, "flag": None,
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni wrapper with declared-family bookkeeping
# ---------------------------------------------------------------------------

def holm_bonferroni_wrapper(pvals: list, labels: Optional[list] = None,
                             family_name: str = "", expected_size: Optional[int] = None,
                             alpha: float = 0.05) -> dict:
    """Thin wrapper over statsmodels multipletests(method='holm') that also
    reports the declared family composition/size per plan Section 4.3.

    pvals: list of raw p-values (NaN entries are excluded from the correction
        but reported back in the output aligned to `labels`).
    labels: optional list of identifiers (e.g. "lgbm_h1") parallel to pvals.
    family_name: e.g. "T1", "T2", "T3".
    expected_size: the pre-declared family size (e.g. 33); a mismatch between
        len(pvals) and expected_size is reported, not silently ignored —
        this is expected under partial grid completion and must be visible.
    """
    pvals = list(pvals)
    n = len(pvals)
    if labels is None:
        labels = [f"test_{i}" for i in range(n)]

    valid_idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    result = {
        "family_name": family_name,
        "declared_family_size": expected_size,
        "actual_n_tests": n,
        "n_valid_pvalues": len(valid_idx),
        "family_size_matches_declaration": (expected_size is None or n == expected_size),
        "alpha": alpha,
        "results": [],
    }

    if len(valid_idx) == 0:
        for lbl, p in zip(labels, pvals):
            result["results"].append({
                "label": lbl, "p_raw": p, "p_holm": np.nan, "reject_holm": False,
            })
        return result

    valid_p = [pvals[i] for i in valid_idx]
    reject, p_holm, _, _ = multipletests(valid_p, alpha=alpha, method="holm")

    holm_map = {valid_idx[j]: (bool(reject[j]), float(p_holm[j])) for j in range(len(valid_idx))}
    for i, (lbl, p) in enumerate(zip(labels, pvals)):
        if i in holm_map:
            rej, ph = holm_map[i]
        else:
            rej, ph = False, np.nan
        result["results"].append({
            "label": lbl, "p_raw": p, "p_holm": ph, "reject_holm": rej,
        })
    return result
