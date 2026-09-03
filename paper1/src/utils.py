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
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan, "da": np.nan}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.any() else np.nan
    r2 = r2_score(y_true, y_pred)
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "da": da}


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
