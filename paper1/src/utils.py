import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 0.0, min_epochs: int = 0):
        """min_epochs: stopping is suppressed entirely (should_stop always
        returns False) until at least this many epochs have been observed,
        regardless of the patience counter. Default 0 preserves the exact
        prior behavior. Added after finding that several ablation-study
        variants early-stopped after 0-5 effective epochs of training (the
        best-validation epoch coinciding with initialization), confounding
        RMSE/MAE/DA comparisons with training length rather than
        architecture -- see main.tex's Ablation Study section."""
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = float("inf")
        self.num_seen = 0

    def should_stop(self, val_loss: float) -> bool:
        self.num_seen += 1
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.num_seen < self.min_epochs:
            return False
        return self.counter >= self.patience


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     mase_denom: Optional[float] = None) -> Dict[str, float]:
    """mase_denom: optional pre-computed scaling denominator for MASE (Hyndman
    & Koehler, 2006) -- the in-sample MAE of a one-step-ahead naive/persistence
    forecast on the TRAINING split, for this target's horizon. When omitted
    (the default), "mase" is NOT added to the returned dict, so every existing
    call site is unchanged unless it opts in. See compute_mase_denominator()
    for how to produce this value; it must come from the training split only
    (never test/val -- it is a scaling constant, not a test-set quantity)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        out = {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "smape": np.nan, "r2": np.nan, "da": np.nan}
        if mase_denom is not None:
            out["mase"] = np.nan
        return out

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.any() else np.nan
    # SMAPE: bounded [0, 200], symmetric in y_true/y_pred. Reported alongside
    # MAPE because MAPE is unstable on this data (return series near zero
    # blow MAPE toward infinity, hence the nonzero mask above) -- SMAPE's
    # denominator uses |y_true| + |y_pred| instead of |y_true| alone, so it
    # does not diverge the same way. Guard the 0/0 case (both y_true and
    # y_pred exactly zero) the same way MAPE guards its own division-by-zero,
    # by excluding those points from the mean.
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_nonzero = denom > 1e-8
    smape = (np.mean(200 * np.abs(y_true[smape_nonzero] - y_pred[smape_nonzero]) / denom[smape_nonzero])
             if smape_nonzero.any() else np.nan)
    r2 = r2_score(y_true, y_pred)
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    out = {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2, "da": da}
    # MASE (Hyndman & Koehler, 2006): scale-free ratio of this model's MAE to
    # the in-sample MAE of a naive one-step-ahead persistence forecast on the
    # TRAINING split (mase_denom). <1.0 beats that naive benchmark; >1.0 is
    # worse than it. Only added when the caller opts in via mase_denom, so
    # every pre-existing call site (which does not pass it) is unaffected.
    if mase_denom is not None:
        out["mase"] = mae / mase_denom if mase_denom > 1e-12 else np.nan
    return out


def compute_mase_denominator(train_targets: np.ndarray, lag: int = 1) -> float:
    """In-sample MAE of a one-step-ahead naive/persistence forecast on a
    TRAINING-split target series, i.e. mean(|y_t - y_{t-lag}|) over
    non-overlapping-appropriate, non-NaN training-split values -- the
    scale-free denominator MASE divides by (Hyndman & Koehler, 2006,
    "Another look at measures of forecast accuracy", IJF 22(4):679-688).
    `train_targets` must already be restricted to the training split only (no
    val/test leakage) and given in their natural time order (e.g. the
    per-horizon forward-return target series CopperDataset/RawPriceDataset
    build, sliced to the training indices, at their original daily spacing --
    i.e. NOT pre-subsampled to one point per horizon window).

    `lag` MUST equal the forecast horizon h when `train_targets` is a series
    of h-day-ahead forward log returns sampled at every day (the usual case
    here). This is because such a series is a *daily* rolling window over the
    same underlying h-day return, so adjacent daily entries overlap by h-1
    days: r_h(t) - r_h(t-1) telescopes to r_1(t+h-1) - r_1(t-1), a difference
    of two ordinary 1-day returns that does NOT scale with h and is thus not
    a meaningful "one step ahead in the h-day-return series" comparison.
    Using lag=h instead compares each h-day return to the h-day return of the
    immediately preceding, non-overlapping h-day window (the genuine "no
    change from the previous period" persistence forecast for that series),
    and its magnitude grows correctly with horizon. The default lag=1 is only
    correct for h=1 target series (or any series that is not built from
    overlapping windows), where lag=1 IS the previous period.

    The naive/persistence forecast this scales against ("no change from the
    previous period") is distinct from this project's zero-forecast baseline
    (which predicts a return of exactly zero).
    """
    train_targets = np.asarray(train_targets).flatten()
    train_targets = train_targets[~np.isnan(train_targets)]
    if len(train_targets) <= lag:
        return np.nan
    return float(np.mean(np.abs(train_targets[lag:] - train_targets[:-lag])))


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, horizon: int = 1) -> Dict[str, float]:
    """Diebold-Mariano test. H0: equal predictive accuracy. Uses squared errors."""
    e1, e2 = np.asarray(e1).flatten(), np.asarray(e2).flatten()
    d = e1**2 - e2**2
    n = len(d)
    d_bar = np.mean(d)
    # Newey-West HAC variance with bandwidth = horizon - 1
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, horizon):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0.0
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan}
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return {"dm_stat": dm_stat, "p_value": p_value}


def paired_seed_significance_test(values_a: list, values_b: list) -> Dict[str, float]:
    """Paired significance test across seeds (item 4: multi-seed robustness
    check). H0: the two variants' per-seed metric values (e.g. RMSE at a
    fixed horizon, one value per seed) come from the same distribution --
    i.e. whichever variant looks better on average is not a stable ranking,
    just seed noise.

    Uses a paired t-test (scipy.stats.ttest_rel) as the primary test (matches
    the paired-by-seed design: values_a[i] and values_b[i] are the same
    seed's full_model vs. pooled_graph result). Also reports the Wilcoxon
    signed-rank test as a distribution-free cross-check, since 5 seeds is a
    very small sample for a t-test's normality assumption -- with n=5 the
    Wilcoxon test's own asymptotic assumptions are weak too, so it is
    reported as a supplementary signal, not the primary claim.

    Returns: {"n": int, "mean_diff": float, "t_stat": float, "t_pvalue": float,
              "wilcoxon_stat": float, "wilcoxon_pvalue": float}.
    NaN fields indicate the test could not be computed (e.g. wilcoxon
    requires nonzero differences; n=5 all-equal differences would return NaN
    rather than crash).
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"values_a and values_b must be the same length (paired by seed), "
                          f"got {len(a)} vs {len(b)}")
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))

    if n < 2 or np.allclose(diff, diff[0]):
        t_stat, t_pvalue = float("nan"), float("nan")
    else:
        t_res = stats.ttest_rel(a, b)
        t_stat, t_pvalue = float(t_res.statistic), float(t_res.pvalue)

    try:
        w_res = stats.wilcoxon(a, b)
        wilcoxon_stat, wilcoxon_pvalue = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        # All-zero differences (identical values) -- wilcoxon is undefined.
        wilcoxon_stat, wilcoxon_pvalue = float("nan"), float("nan")

    return {
        "n": n, "mean_diff": mean_diff,
        "t_stat": t_stat, "t_pvalue": t_pvalue,
        "wilcoxon_stat": wilcoxon_stat, "wilcoxon_pvalue": wilcoxon_pvalue,
    }


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
