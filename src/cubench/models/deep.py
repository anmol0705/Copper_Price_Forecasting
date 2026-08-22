"""Tier 4 deep learning comparison points: lstm, transformer. Architecture ported from
`src/models/baselines.py::LSTMBaseline` / `TransformerBaseline` (READ-ONLY reference,
not imported/modified) with matching hidden_dim=64, num_heads=4, temporal_layers=2,
dropout=0.1 per configs/default.yaml. Only the input dimension (71 CuBench features
instead of Paper 1's 8 raw variables) and output head (one scalar per CuBench
target/horizon cell instead of Paper 1's multi-horizon head) differ.

Trained from scratch on CPU per fold under the CuBench walk-forward protocol, lookback=60,
batch_size=32, epochs<=100, EarlyStopper(patience=20) (reimplemented identically to
src/utils.py::EarlyStopper since that class has no import-time side effects), Adam
lr=1e-3, weight_decay=1e-5. 3 seeds (42,43,44).

Because these models do not consume one feature row per prediction but a rolling
lookback window, they bypass `walkforward.get_fold_split`/`run_one` (which are
row-independent) and instead build their own windows directly from the cached full
feature frame, respecting the exact same fold/embargo boundaries. Predictions are
still persisted to the same `results/cubench/predictions/*.npy` convention and
appended to the same run-log jsonl so aggregation is uniform across all tiers.

SCOPE NOTE (time-budget subsetting, per plan Section 3.5 / task brief allowance):
Full grid would be 2 models x 3 targets x 3 horizons x 11 folds x 3 seeds = 594 NN
trainings on CPU -- infeasible in one session. Deep models are run on targets {t1, t3}
(T2 quantile deep learning is skipped -- reason: no cheap way to get 7-quantile deep
output without materially more code/compute, and T1/T3 already establish the DL
comparison point the plan's Section 3.5 motivates) and horizon h=1 only (the horizon
with by far the most usable non-overlapping information and the cheapest window
count), across all 11 folds and all 3 seeds. This is documented, not silent.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.cubench import walkforward as wf

warnings.filterwarnings("ignore")

LOOKBACK = 60
BATCH_SIZE = 32
# REDUCED from the plan's epochs=100/patience=20 due to a real time-budget constraint
# discovered during this run: with 4 tree-model families and 2 linear-model processes
# already saturating all 12 CPU cores, a single LSTM fit at epochs=100/patience=20 did
# not produce output after >25 minutes (wall clock, not a hang -- CPU-starved). Per the
# task brief's explicit allowance ("reduce epochs if needed... document exactly what and
# why"), epochs/patience are cut here. torch's own intra-op thread pool is also capped
# so it does not itself try to grab all cores and worsen contention with the other
# families running concurrently.
MAX_EPOCHS = 40
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

torch.set_num_threads(2)


class EarlyStopper:
    """Reimplemented identically to src/utils.py::EarlyStopper (no cross-tier import
    needed; logic copied verbatim to keep src/cubench self-contained)."""

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


class LSTMNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN_DIM, NUM_LAYERS, batch_first=True,
                             dropout=DROPOUT if NUM_LAYERS > 1 else 0)
        self.head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, HIDDEN_DIM)
        self.pos_enc = nn.Parameter(torch.randn(1, LOOKBACK, HIDDEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(HIDDEN_DIM, NUM_HEADS, HIDDEN_DIM * 4,
                                            batch_first=True, dropout=DROPOUT)
        self.encoder = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def _build_windows(feat_arr: np.ndarray, dates: np.ndarray, target_by_date: dict):
    """feat_arr: (N, F) full-history imputed feature array aligned to `dates`.
    Returns (X_windows, y, window_end_dates) for every date whose target is available
    and which has >= LOOKBACK rows of history."""
    Xs, ys, ds = [], [], []
    for i in range(LOOKBACK - 1, len(dates)):
        d = dates[i]
        if d not in target_by_date:
            continue
        y = target_by_date[d]
        if np.isnan(y):
            continue
        Xs.append(feat_arr[i - LOOKBACK + 1: i + 1])
        ys.append(y)
        ds.append(d)
    if not Xs:
        return np.zeros((0, LOOKBACK, feat_arr.shape[1])), np.zeros(0), np.array([])
    return np.stack(Xs), np.array(ys, dtype=float), np.array(ds)


def _train_one(model, Xtr, ytr, Xval, yval, seed, is_binary):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.MSELoss()
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    Xval_t = torch.tensor(Xval, dtype=torch.float32)
    yval_t = torch.tensor(yval, dtype=torch.float32)
    stopper = EarlyStopper(patience=PATIENCE)
    best_state, best_loss = None, float("inf")
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xval_t) if len(Xval_t) else torch.zeros(0)
            vloss = float(loss_fn(vpred, yval_t)) if len(Xval_t) else float(loss)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if stopper.should_stop(vloss):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


_DF_CACHE = None


def _get_df():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = wf.load_features()
    return _DF_CACHE


def run_deep_grid(model_names: list, targets: list, horizons: list, folds: list, seeds: list,
                   results_jsonl: Path):
    df = _get_df()
    fcols = wf.feature_cols(df)
    dates_all = df["date"].to_numpy()
    feat_full = df[fcols].to_numpy(dtype=float)
    # global median impute (per-fold median imputation is applied below using train-only stats)
    results = []
    done = set()
    if results_jsonl.exists():
        with open(results_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r.get("model"), r.get("target"), r.get("horizon"),
                               r.get("fold"), r.get("seed")))
                except Exception:
                    pass

    with open(results_jsonl, "a") as out:
        for model_name in model_names:
            for target in targets:
                ycol_prefix = wf.TARGET_PREFIX[target]
                for horizon in horizons:
                    ycol = f"{ycol_prefix}{horizon}"
                    target_by_date_full = dict(zip(dates_all, df[ycol].to_numpy(dtype=float)))
                    for fold in folds:
                        train_end = pd.Timestamp(fold["embargo"][f"h{horizon}"]["train_end"])
                        train_start = pd.Timestamp(fold["train_start"])
                        oos_start = pd.Timestamp(fold["oos_start"])
                        oos_end = pd.Timestamp(fold["oos_end"])

                        train_row_mask = (dates_all >= np.datetime64(train_start)) & (dates_all <= np.datetime64(train_end))
                        # median impute using train-fold rows only, applied to the whole
                        # history array (train-only statistic, no leakage)
                        train_idx = np.where(train_row_mask)[0]
                        med = np.nanmedian(feat_full[train_idx], axis=0)
                        med = np.where(np.isnan(med), 0.0, med)
                        feat_imputed = np.where(np.isnan(feat_full), med[None, :], feat_full)

                        train_dates_set = set(dates_all[train_idx])
                        Xtr_all, ytr_all, dtr_all = _build_windows(feat_imputed, dates_all,
                                                                     {d: v for d, v in target_by_date_full.items() if d in train_dates_set})
                        if len(Xtr_all) < 100:
                            continue
                        # time-ordered val split: last 15% of training windows
                        n_val = max(1, int(0.15 * len(Xtr_all)))
                        Xtr, ytr = Xtr_all[:-n_val], ytr_all[:-n_val]
                        Xval, yval = Xtr_all[-n_val:], ytr_all[-n_val:]

                        test_row_mask = (dates_all >= np.datetime64(oos_start)) & (dates_all <= np.datetime64(oos_end))
                        test_idx = np.where(test_row_mask)[0]
                        test_dates_set = set(dates_all[test_idx])
                        Xte, yte, dte = _build_windows(feat_imputed, dates_all,
                                                         {d: v for d, v in target_by_date_full.items() if d in test_dates_set})
                        if len(Xte) == 0:
                            continue

                        for seed in seeds:
                            key = (model_name, target, horizon, fold["fold"], seed)
                            if key in done:
                                continue
                            t0 = time.time()
                            is_binary = (target == "t3")
                            net_cls = LSTMNet if model_name == "lstm" else TransformerNet
                            model = net_cls(len(fcols))
                            model = _train_one(model, Xtr, ytr, Xval, yval, seed, is_binary)
                            model.eval()
                            with torch.no_grad():
                                raw = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
                            dt = time.time() - t0
                            if is_binary:
                                proba = 1.0 / (1.0 + np.exp(-raw))
                                metrics = wf.compute_metrics_t3(yte, proba)
                                arr = proba
                            else:
                                metrics = wf.compute_metrics_t1(yte, raw)
                                arr = raw
                            wf.PRED_DIR.mkdir(parents=True, exist_ok=True)
                            fname = wf.PRED_DIR / f"{model_name}_{target}_h{horizon}_fold{fold['fold']}_seed{seed}.npy"
                            np.save(fname, arr)
                            metrics.update({
                                "model": model_name, "target": target, "horizon": horizon,
                                "fold": fold["fold"], "oos_year": fold["oos_year"], "seed": seed,
                                "train_n": int(len(ytr)), "test_n": int(len(yte)), "wall_s": dt,
                            })
                            out.write(json.dumps(metrics, default=float) + "\n")
                            out.flush()
                            results.append(metrics)
                            print(f"[deep] {model_name} {target} h{horizon} fold{fold['fold']} seed{seed} "
                                  f"done in {dt:.1f}s", flush=True)
    return results
