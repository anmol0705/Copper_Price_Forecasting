"""base_rate_report() — the base-rate / constant-forecast diagnostic, applied to every
T3 result cell (and any T1/T2 cell with anomalously low prediction dispersion).
Extension of Cheung 2026 (arXiv:2607.12248), per the plan's pre-committed novelty framing.

Full spec: docs/cubench_implementation_plan.md Section 4.7.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _stats
from statsmodels.stats.contingency_tables import mcnemar

from src.cubench.stats_tests import pesaran_timmermann

SIGN_CONCENTRATION_FLAG = 0.95
DISPERSION_RATIO_FLAG = 0.20


def base_rate_report(y_true: np.ndarray, y_pred_proba: np.ndarray = None,
                      y_pred_class: np.ndarray = None,
                      null_majority_pred: np.ndarray = None) -> dict:
    """12-item base-rate/constant-forecast diagnostic for a single T3 cell.

    y_true: 0/1 array, actual "up" indicator.
    y_pred_proba: predicted probability of class 1 (used to derive y_pred_class
        at 0.5 threshold if y_pred_class not given directly).
    y_pred_class: 0/1 predicted class labels (overrides threshold derivation).
    null_majority_pred: 0/1 predictions from the null_majority baseline, for
        the McNemar comparison. If None, uses the training-set-implied
        majority class of y_true itself as a stand-in (documented as such).
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    if y_pred_class is not None:
        pred = np.asarray(y_pred_class, dtype=float).flatten()
    else:
        proba = np.asarray(y_pred_proba, dtype=float).flatten()
        pred = (proba >= 0.5).astype(float)

    mask = np.isfinite(y_true) & np.isfinite(pred)
    y_true, pred = y_true[mask], pred[mask]
    n = len(y_true)

    # 1. raw DA
    raw_da = float(np.mean(pred == y_true)) if n else float("nan")
    # 2. base rate (test-window positive rate)
    base_rate = float(np.mean(y_true)) if n else float("nan")
    # 3. majority-class accuracy
    majority_acc = float(max(base_rate, 1 - base_rate)) if n else float("nan")
    # 4. excess DA
    excess_da = raw_da - majority_acc if n else float("nan")
    # 5. predicted-positive rate
    pred_pos_rate = float(np.mean(pred)) if n else float("nan")
    # 6. sign-concentration + flag
    sign_concentration = float(max(pred_pos_rate, 1 - pred_pos_rate)) if n else float("nan")
    sign_concentration_flag = bool(sign_concentration > SIGN_CONCENTRATION_FLAG) if n else False
    # 7. binomial test of predicted signs against 0.5
    if n:
        k = int(np.sum(pred))
        binom_p = float(_stats.binomtest(k, n, 0.5).pvalue)
    else:
        binom_p = float("nan")
    # 8. dispersion ratio std(yhat)/std(y) — for T3 use proba if available, else pred
    disp_source = np.asarray(y_pred_proba, dtype=float).flatten()[mask] if y_pred_proba is not None else pred
    std_yhat = float(np.std(disp_source)) if n else float("nan")
    std_y = float(np.std(y_true)) if n else float("nan")
    dispersion_ratio = float(std_yhat / std_y) if (n and std_y > 0) else float("nan")
    dispersion_flag = bool(np.isfinite(dispersion_ratio) and dispersion_ratio < DISPERSION_RATIO_FLAG)
    # 9. confusion matrix
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    # 10. PT p-value + degenerate flag
    pt = pesaran_timmermann(y_true, pred)
    # 11. McNemar vs null_majority
    if null_majority_pred is not None:
        nm = np.asarray(null_majority_pred, dtype=float).flatten()[mask]
    else:
        nm = np.full(n, 1.0 if base_rate >= 0.5 else 0.0)
    b = int(np.sum((pred == 1) & (nm == 0)))  # model right-ish placeholder, real b/c below
    # McNemar needs correct/incorrect vs y_true for each of the two classifiers
    model_correct = (pred == y_true)
    null_correct = (nm == y_true)
    n01 = int(np.sum(~model_correct & null_correct))   # model wrong, null right
    n10 = int(np.sum(model_correct & ~null_correct))    # model right, null wrong
    table = [[int(np.sum(model_correct & null_correct)), n01],
             [n10, int(np.sum(~model_correct & ~null_correct))]]
    try:
        mc = mcnemar(table, exact=(n01 + n10 < 25), correction=True)
        mcnemar_stat, mcnemar_p = float(mc.statistic), float(mc.pvalue)
    except Exception:
        mcnemar_stat, mcnemar_p = float("nan"), float("nan")

    # 12. verdict
    if pt.get("degenerate") or sign_concentration_flag:
        verdict = "constant_forecast_artifact" if sign_concentration >= 0.999 else "suspect_low_dispersion"
        if pt.get("degenerate"):
            verdict = "constant_forecast_artifact"
    elif dispersion_flag:
        verdict = "suspect_low_dispersion"
    else:
        verdict = "ok"

    return {
        "n": n,
        "raw_da": raw_da,
        "base_rate": base_rate,
        "majority_class_accuracy": majority_acc,
        "excess_da": excess_da,
        "predicted_positive_rate": pred_pos_rate,
        "sign_concentration": sign_concentration,
        "sign_concentration_flag": sign_concentration_flag,
        "binomial_test_p": binom_p,
        "dispersion_ratio": dispersion_ratio,
        "dispersion_ratio_flag": dispersion_flag,
        "confusion_matrix": confusion,
        "pt_p_value": pt.get("p_value"),
        "pt_stat": pt.get("pt_stat"),
        "pt_degenerate": pt.get("degenerate"),
        "mcnemar_stat": mcnemar_stat,
        "mcnemar_p_value": mcnemar_p,
        "verdict": verdict,
    }


def dispersion_ratio_check(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Applies the dispersion-ratio check (item 8 of the T3 spec) to T1/T2
    continuous cells, per the plan's instruction to extend it there too."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    std_y = float(np.std(y_true)) if len(y_true) else float("nan")
    std_yhat = float(np.std(y_pred)) if len(y_pred) else float("nan")
    ratio = float(std_yhat / std_y) if (std_y and std_y > 0) else float("nan")
    return {
        "n": len(y_true),
        "std_y": std_y,
        "std_yhat": std_yhat,
        "dispersion_ratio": ratio,
        "dispersion_ratio_flag": bool(np.isfinite(ratio) and ratio < DISPERSION_RATIO_FLAG),
    }
