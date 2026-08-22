"""Target construction -- T1 (log annualized forward realized volatility, regression),
T2 (cumulative forward log return, quantile regression, 7 taus), T3 (direction,
classification/diagnostic). Horizons h in {1, 5, 22}.

Full spec: docs/cubench_implementation_plan.md Section 2.6.

These targets are FORWARD-looking by construction (they use data from t+1 ... t+h) --
that is correct and intentional for a target, but it means this module must never be
imported by features.py, and no target column may ever be treated as a feature
downstream. tests/test_leakage.py::test_no_target_in_features checks this mechanically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 22)
QUANTILE_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

TARGET_COLUMN_REGEX = r"^y[123]_h\d+(_q[\d.]+)?$"


def _gk_variance(panel: pd.DataFrame) -> pd.Series:
    """Garman-Klass variance proxy, floored at 1e-10 -- the primary RV series, per
    plan Section 2.4/2.6. `panel` must have columns open/high/low/close."""
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    return gk.clip(lower=1e-10)


def build_t1(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """y1_h = ln( sqrt( (252/h) * sum_{i=1..h} gk_{t+i} ) ), for i in 1..h -- i.e. the
    target at row t uses ONLY future rows t+1..t+h, computed via a forward-looking
    rolling window then shifted back by h so it aligns to row t."""
    gk = _gk_variance(panel)
    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        # rolling(h).sum() at index t+h covers gk[t+1..t+h] inclusive when shifted
        # back by h: fwd_sum(t) = sum_{i=1..h} gk[t+i] = rolling(h).sum().shift(-h) evaluated at t.
        fwd_sum = gk.rolling(h).sum().shift(-h)
        annualized_var = (252.0 / h) * fwd_sum
        out[f"y1_h{h}"] = np.log(np.sqrt(annualized_var.clip(lower=1e-12)))
    return out


def build_t2(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """y2_h = sum_{i=1..h} r_{t+i} -- cumulative forward log return. Quantile targets
    (per tau) are not separate columns here -- T2's *point* target is this cumulative
    return; the 7 quantile levels are properties of the MODEL's prediction (quantile
    regression heads trained against this single y2_h column at each tau), not 7
    separate target columns. This matches plan Section 2.6 exactly: "Predicted at
    tau in {...}" describes the model output, not the target construction.
    """
    r = np.log(panel["close"] / panel["close"].shift(1))
    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        fwd_sum = r.rolling(h).sum().shift(-h)
        out[f"y2_h{h}"] = fwd_sum
    return out


def build_t3(t2: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """y3_h = 1[y2_h > 0]."""
    out = pd.DataFrame(index=t2.index)
    for h in horizons:
        y2 = t2[f"y2_h{h}"]
        y3 = (y2 > 0).astype(float)
        y3 = y3.where(y2.notna())  # NaN where the forward window is incomplete
        out[f"y3_h{h}"] = y3
    return out


def build_all_targets(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """panel must have columns date/open/high/low/close (copper OHLC, same calendar as
    features.py's build_raw_panel()). Returns a DataFrame with y1_h*, y2_h*, y3_h*
    columns, indexed the same way as `panel`."""
    t1 = build_t1(panel, horizons)
    t2 = build_t2(panel, horizons)
    t3 = build_t3(t2, horizons)
    return pd.concat([t1, t2, t3], axis=1)
