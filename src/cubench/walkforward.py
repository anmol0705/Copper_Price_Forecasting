"""Walk-forward execution engine. Takes a ModelAdapter, fold definitions, and a target
spec; emits per-fold predictions to results/cubench/predictions/.

ModelAdapter interface: fit(X, y), predict(X), predict_quantiles(X, taus), predict_proba(X).
Models implement only the methods relevant to their target type; the rest raise
NotImplementedError.

Full spec: docs/cubench_implementation_plan.md Section 3, Section 4.1.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "cubench" / "features.parquet"
FOLDS_PATH = ROOT / "results" / "cubench" / "folds.json"
PRED_DIR = ROOT / "results" / "cubench" / "predictions"

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGET_PREFIX = {"t1": "y1_h", "t2": "y2_h", "t3": "y3_h"}


@runtime_checkable
class ModelAdapter(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ModelAdapter": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_quantiles(self, X: pd.DataFrame, taus: list) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_folds() -> dict:
    return json.loads(FOLDS_PATH.read_text())


def feature_cols(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c != "date" and not (c.startswith("y1_") or c.startswith("y2_") or c.startswith("y3_"))
    ]


def get_fold_split(df: pd.DataFrame, fold: dict, horizon: int, target: str):
    """Leakage-safe train/test split for one fold/horizon/target.

    Train window: [train_start, embargo-purged train_end] for that horizon.
    Test window: [oos_start, oos_end], unmodified (no gap needed inside OOS).
    Rows with NaN target (burn-in or end-of-series lookahead unavailable) are dropped.
    """
    ycol = f"{TARGET_PREFIX[target]}{horizon}"
    fcols = feature_cols(df)
    train_end = pd.Timestamp(fold["embargo"][f"h{horizon}"]["train_end"])
    train_start = pd.Timestamp(fold["train_start"])
    oos_start = pd.Timestamp(fold["oos_start"])
    oos_end = pd.Timestamp(fold["oos_end"])

    train_mask = (df["date"] >= train_start) & (df["date"] <= train_end) & df[ycol].notna()
    test_mask = (df["date"] >= oos_start) & (df["date"] <= oos_end) & df[ycol].notna()

    Xtr = df.loc[train_mask, fcols].reset_index(drop=True)
    ytr = df.loc[train_mask, ycol].to_numpy(dtype=float)
    Xte = df.loc[test_mask, fcols].reset_index(drop=True)
    yte = df.loc[test_mask, ycol].to_numpy(dtype=float)
    dates_te = df.loc[test_mask, "date"].to_numpy()
    return Xtr, ytr, Xte, yte, dates_te


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def compute_metrics_t1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    # y1 = ln(sqrt(RV_annualized)) = 0.5 * ln(RV_ann)  =>  RV_ann = exp(2*y1)
    rv_true = np.exp(2 * y_true)
    rv_pred = np.clip(np.exp(2 * y_pred), 1e-12, None)
    qlike = float(np.mean(rv_true / rv_pred - np.log(rv_true / rv_pred) - 1))
    return {"rmse": rmse, "mae": mae, "qlike": qlike, "n": int(len(y_true))}


def compute_metrics_t2(y_true: np.ndarray, preds_by_tau: list, taus: list) -> dict:
    losses = {}
    for tau, pred in zip(taus, preds_by_tau):
        losses[f"pinball_{tau}"] = pinball_loss(y_true, np.asarray(pred), tau)
    losses["pinball_avg"] = float(np.mean([losses[f"pinball_{t}"] for t in taus]))
    stacked = np.stack([np.asarray(p) for p in preds_by_tau], axis=0)  # (ntau, n)
    diffs = np.diff(stacked, axis=0)
    losses["crossing_rate"] = float(np.mean(diffs < 0)) if diffs.size else 0.0
    losses["n"] = int(len(y_true))
    return losses


def compute_metrics_t3(y_true: np.ndarray, proba: np.ndarray) -> dict:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1 - 1e-9)
    pred = (proba >= 0.5).astype(int)
    acc = float(np.mean(pred == y_true))
    logloss = float(-np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))
    return {"accuracy": acc, "logloss": logloss, "n": int(len(y_true))}


def run_one(
    model_factory: Callable[[int, int], "ModelAdapter"],
    model_id: str,
    target: str,
    horizon: int,
    fold: dict,
    seed: int,
    df: pd.DataFrame,
    save: bool = True,
) -> dict:
    """Fit+predict one (model, target, horizon, fold, seed) cell, persist raw predictions.
    model_factory(seed, horizon) -> ModelAdapter."""
    Xtr, ytr, Xte, yte, dates_te = get_fold_split(df, fold, horizon, target)
    model = model_factory(seed, horizon)
    t0 = time.time()

    if target == "t2":
        model.fit(Xtr, ytr)
        preds = [np.asarray(p) for p in model.predict_quantiles(Xte, TAUS)]
        metrics = compute_metrics_t2(yte, preds, TAUS)
        # `preds` may already be isotonic-sorted (crossing repaired) by the model, which
        # would make metrics["crossing_rate"] trivially 0. Models that repair crossings
        # (lgbm/xgboost/catboost) stash the true PRE-repair rate on `last_crossing_rate`;
        # surface that as the calibration-honesty metric the plan asks for, keeping the
        # (repaired, ~0) rate as crossing_rate_post_repair for reference.
        if hasattr(model, "last_crossing_rate"):
            metrics["crossing_rate_post_repair"] = metrics["crossing_rate"]
            metrics["crossing_rate"] = model.last_crossing_rate
        arr = np.stack(preds, axis=0)
    elif target == "t3":
        model.fit(Xtr, ytr)
        proba = np.asarray(model.predict_proba(Xte))
        metrics = compute_metrics_t3(yte, proba)
        arr = proba
    else:
        model.fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xte))
        metrics = compute_metrics_t1(yte, pred)
        arr = pred

    dt = time.time() - t0
    fold_k = fold["fold"]
    if save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        fname = PRED_DIR / f"{model_id}_{target}_h{horizon}_fold{fold_k}_seed{seed}.npy"
        np.save(fname, arr)

    metrics.update({
        "model": model_id, "target": target, "horizon": horizon, "fold": fold_k,
        "oos_year": fold["oos_year"], "seed": seed, "train_n": int(len(ytr)),
        "test_n": int(len(yte)), "wall_s": dt,
    })
    return metrics


def run_grid(models: dict, targets: list, horizons: list, folds: list, df: pd.DataFrame,
             results_path: Path, append: bool = True) -> list:
    """Run models x targets x horizons x folds x seeds, writing incrementally (one JSON
    line per cell appended to results_path.with_suffix('.jsonl')) so progress survives
    interruption. models: {model_id: (factory, seeds)}."""
    jsonl_path = results_path.with_suffix(".jsonl")
    done = set()
    if append and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["target"], r["horizon"], r["fold"], r["seed"]))
                except Exception:
                    pass
    all_results = []
    with open(jsonl_path, "a") as out:
        for model_id, (factory, seeds) in models.items():
            for target in targets:
                for horizon in horizons:
                    for fold in folds:
                        for seed in seeds:
                            key = (model_id, target, horizon, fold["fold"], seed)
                            if key in done:
                                continue
                            try:
                                m = run_one(factory, model_id, target, horizon, fold, seed, df)
                            except Exception as e:
                                m = {
                                    "model": model_id, "target": target, "horizon": horizon,
                                    "fold": fold["fold"], "seed": seed, "error": str(e),
                                }
                            out.write(json.dumps(m, default=float) + "\n")
                            out.flush()
                            all_results.append(m)
    return all_results
