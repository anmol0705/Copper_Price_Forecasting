"""Baseline models for copper price forecasting."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Base classes
# ============================================================================

class TorchBaseline(nn.Module):
    """Base class for PyTorch-based baselines with common training loop."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = config.get("device", "cpu")
        self.horizons = config.get("horizons", [1, 5, 10, 22])

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.get("lr", 1e-3),
                                     weight_decay=self.config.get("weight_decay", 1e-5))
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 20)
        best_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = self._forward_flat(x)  # (B, num_horizons)
                loss = F.mse_loss(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            if val_loader:
                val_loss = self._evaluate(val_loader)
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state:
            self.load_state_dict(best_state)
        self.to(self.device)

    def _evaluate(self, loader: DataLoader) -> float:
        self.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self._forward_flat(x)
                total += F.mse_loss(pred, y, reduction="sum").item()
                n += y.shape[0]
        return total / max(n, 1)

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                pred = self._forward_flat(x)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================================
# Deep Learning baselines
# ============================================================================

class LSTMBaseline(TorchBaseline):
    """Standard LSTM on raw multivariate prices."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)
        nl = config.get("num_layers", 2)
        dp = config.get("dropout", 0.1)
        self.lstm = nn.LSTM(nv, hd, nl, batch_first=True,
                            dropout=dp if nl > 1 else 0)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])

    def _forward_flat(self, x):
        # x: (B, T, num_vars)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return torch.cat([head(h) for head in self.heads], dim=-1)


class GRUBaseline(TorchBaseline):
    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)
        nl = config.get("num_layers", 2)
        dp = config.get("dropout", 0.1)
        self.gru = nn.GRU(nv, hd, nl, batch_first=True,
                          dropout=dp if nl > 1 else 0)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])

    def _forward_flat(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return torch.cat([head(h) for head in self.heads], dim=-1)


class CNNLSTMBaseline(TorchBaseline):
    """1D CNN feature extractor + LSTM."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)
        self.cnn = nn.Sequential(
            nn.Conv1d(nv, hd, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hd, hd, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hd, hd, 2, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])

    def _forward_flat(self, x):
        # x: (B, T, N) -> CNN expects (B, N, T)
        c = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, hd)
        out, _ = self.lstm(c)
        h = out[:, -1, :]
        return torch.cat([head(h) for head in self.heads], dim=-1)


class TransformerBaseline(TorchBaseline):
    """Vanilla Transformer encoder."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)
        nhead = config.get("num_heads", 4)
        nl = config.get("num_layers", 2)
        self.input_proj = nn.Linear(nv, hd)
        self.pos_enc = nn.Parameter(torch.randn(1, config.get("lookback", 60), hd) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(hd, nhead, hd * 4,
                                                    batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, nl)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])

    def _forward_flat(self, x):
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        h = self.encoder(h)
        h = h.mean(dim=1)  # mean pooling
        return torch.cat([head(h) for head in self.heads], dim=-1)


# ============================================================================
# VMD + DL baselines
# ============================================================================

class VMDLSTMBaseline(TorchBaseline):
    """VMD-LSTM: separate LSTM per mode, predictions summed (Liu et al. 2019)."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        K = config.get("num_modes", 5)
        hd = config.get("hidden_dim", 64)
        self.K = K
        self.lstms = nn.ModuleList([
            nn.LSTM(nv, hd, 2, batch_first=True) for _ in range(K)
        ])
        self.heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])
            for _ in range(K)
        ])

    def _forward_flat(self, x):
        # x: (B, T, K, N)
        B, T, K, N = x.shape
        preds = torch.zeros(B, len(self.horizons), device=x.device)
        for k in range(self.K):
            mode_x = x[:, :, k, :]  # (B, T, N)
            out, _ = self.lstms[k](mode_x)
            h = out[:, -1, :]
            mode_pred = torch.cat([self.heads[k][i](h) for i in range(len(self.horizons))], dim=-1)
            preds = preds + mode_pred
        return preds


class VMDTransformerBaseline(TorchBaseline):
    """VMD + Transformer per mode."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        K = config.get("num_modes", 5)
        hd = config.get("hidden_dim", 64)
        self.K = K
        self.projs = nn.ModuleList([nn.Linear(nv, hd) for _ in range(K)])
        encoder_layer = nn.TransformerEncoderLayer(hd, 4, hd * 4,
                                                    batch_first=True, dropout=0.1)
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, 2) for _ in range(K)
        ])
        self.heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])
            for _ in range(K)
        ])

    def _forward_flat(self, x):
        B, T, K, N = x.shape
        preds = torch.zeros(B, len(self.horizons), device=x.device)
        for k in range(self.K):
            h = self.projs[k](x[:, :, k, :])
            h = self.encoders[k](h).mean(dim=1)
            mode_pred = torch.cat([self.heads[k][i](h) for i in range(len(self.horizons))], dim=-1)
            preds = preds + mode_pred
        return preds


class VMDAttentionLSTM(TorchBaseline):
    """VMD + LSTM + attention fusion (no graph). Tests fusion without graph."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        K = config.get("num_modes", 5)
        hd = config.get("hidden_dim", 64)
        self.K = K
        self.lstms = nn.ModuleList([
            nn.LSTM(nv, hd, 2, batch_first=True) for _ in range(K)
        ])
        self.attn_query = nn.Parameter(torch.randn(hd))
        self.attn_key = nn.Linear(hd, hd)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])

    def _forward_flat(self, x):
        B, T, K, N = x.shape
        mode_outs = []
        for k in range(self.K):
            out, _ = self.lstms[k](x[:, :, k, :])
            mode_outs.append(out[:, -1, :])  # (B, hd)
        stacked = torch.stack(mode_outs, dim=1)  # (B, K, hd)
        keys = self.attn_key(stacked)
        attn = torch.einsum("bkh,h->bk", keys, self.attn_query)
        attn = F.softmax(attn, dim=-1)
        fused = torch.einsum("bk,bkh->bh", attn, stacked)
        return torch.cat([head(fused) for head in self.heads], dim=-1)


# ============================================================================
# GNN baselines
# ============================================================================

class SimpleMTGNN(TorchBaseline):
    """Simplified MTGNN: learned graph + GCN + temporal conv."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)
        lookback = config.get("lookback", 60)

        # Learned adjacency
        self.node_emb1 = nn.Embedding(nv, 16)
        self.node_emb2 = nn.Embedding(nv, 16)

        # Spatial: simple message passing
        self.spatial = nn.Linear(hd, hd)
        self.input_proj = nn.Linear(1, hd)

        # Temporal: dilated causal conv
        self.tcn = nn.Sequential(
            nn.Conv1d(hd * nv, hd * nv, kernel_size=3, padding=2, dilation=1, groups=nv),
            nn.ReLU(),
            nn.Conv1d(hd * nv, hd * nv, kernel_size=3, padding=4, dilation=2, groups=nv),
            nn.ReLU(),
        )
        self.output_proj = nn.Linear(hd, len(self.horizons))
        self.nv = nv
        self.hd = hd

    def _forward_flat(self, x):
        # x: (B, T, N)
        B, T, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))  # (B, T, N, hd)

        # Learned adjacency
        idx = torch.arange(N, device=x.device)
        adj = torch.softmax(F.relu(self.node_emb1(idx) @ self.node_emb2(idx).T), dim=-1)

        # Graph conv at each timestep
        for t in range(T):
            ht = h[:, t]  # (B, N, hd)
            msg = torch.einsum("ij,bjd->bid", adj, ht)
            h[:, t] = ht + F.relu(self.spatial(msg))

        # Temporal conv on copper node
        # Reshape for TCN: (B, N*hd, T)
        h_flat = h.permute(0, 2, 3, 1).reshape(B, N * self.hd, T)
        h_flat = self.tcn(h_flat)[:, :, :T]
        h_flat = h_flat.reshape(B, N, self.hd, T)
        copper_h = h_flat[:, 0, :, -1]  # (B, hd)
        return self.output_proj(copper_h)


class GNNTransformer(TorchBaseline):
    """GNN + Transformer (no decomposition). Tests graph without VMD."""

    def __init__(self, config: dict):
        super().__init__(config)
        nv = config["num_vars"]
        hd = config.get("hidden_dim", 64)

        self.node_emb1 = nn.Embedding(nv, 16)
        self.node_emb2 = nn.Embedding(nv, 16)
        self.input_proj = nn.Linear(1, hd)
        self.spatial = nn.Linear(hd, hd)
        encoder_layer = nn.TransformerEncoderLayer(hd, 4, hd * 4,
                                                    batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.heads = nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])
        self.nv = nv

    def _forward_flat(self, x):
        B, T, N = x.shape
        h = self.input_proj(x.unsqueeze(-1))  # (B, T, N, hd)
        idx = torch.arange(N, device=x.device)
        adj = torch.softmax(F.relu(self.node_emb1(idx) @ self.node_emb2(idx).T), dim=-1)

        for t in range(T):
            ht = h[:, t]
            msg = torch.einsum("ij,bjd->bid", adj, ht)
            h[:, t] = ht + F.relu(self.spatial(msg))

        copper_seq = h[:, :, 0, :]  # (B, T, hd)
        enc = self.encoder(copper_seq).mean(dim=1)
        return torch.cat([head(enc) for head in self.heads], dim=-1)


# ============================================================================
# Traditional ML baselines
# ============================================================================

class XGBoostBaseline:
    """XGBoost with flattened features. One model per horizon."""

    def __init__(self, config: dict):
        self.config = config
        self.horizons = config.get("horizons", [1, 5, 10, 22])
        self.models = {}

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        import xgboost as xgb
        X_train, Y_train = self._loader_to_numpy(train_loader)
        X_val, Y_val = self._loader_to_numpy(val_loader) if val_loader else (None, None)

        for i, h in enumerate(self.horizons):
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=20 if X_val is not None else None,
                verbosity=0
            )
            eval_set = [(X_val, Y_val[:, i])] if X_val is not None else None
            model.fit(X_train, Y_train[:, i], eval_set=eval_set, verbose=False)
            self.models[h] = model

    def predict(self, loader: DataLoader) -> np.ndarray:
        X, _ = self._loader_to_numpy(loader)
        preds = np.column_stack([self.models[h].predict(X) for h in self.horizons])
        return preds

    def _loader_to_numpy(self, loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy().reshape(x.size(0), -1))
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)


class ARIMABaseline:
    """ARIMA baseline on the raw copper price channel (feature index 0).

    Design choice (a) from the task brief, simplified: statsmodels has no
    notion of "fit on a DataLoader of overlapping lookback windows" the way
    the DL/GBM baselines do, so ARIMA is adapted as follows:

      * `fit()` runs a small (p, d, q) grid search (AIC-minimizing, statsmodels
        `ARIMA`, not `pmdarima` -- `pmdarima` is not a repo dependency and this
        avoids adding a new heavy one) on a subsample of training windows, and
        keeps the order with the best mean AIC. This happens once.
      * `predict()` walk-forwards: for every lookback window in the loader it
        re-fits ARIMA(order) on that window's own price history (this is the
        standard rolling/expanding ARIMA forecasting protocol -- a single
        historical fit would go stale over a multi-year test set) and calls
        `get_forecast(steps=max(horizons))` once per window, reading off every
        locked horizon (1/5/10/22) from the same forecast path rather than
        fitting a separate model per horizon.
      * The loader's price channel is z-score normalized upstream (see
        `RawPriceDataset`), using the *training* split's per-variable mean/std
        (index 0 = copper). `predict()` de-normalizes the ARIMA forecast and
        the window's last observed value back to real price units using those
        stats (recovered from the loader's underlying dataset -- see
        `_get_norm_stats`), then computes a proper log-return prediction
        `log(p_hat / p_last)` to match the pipeline's actual target
        `y = log(p_{t+h} / p_t)`. (An earlier version of this baseline used
        `forecast[h-1] - last_val` -- a normalized-price-level *difference* --
        as an additive proxy for a log-*return* target; those two quantities
        differ by roughly `price_level / std_train`, which for copper over
        the 2010-2019 training window is ~5-6x, making ARIMA's errors look
        catastrophically worse than its actual forecast skill. Fixed here.)

      * Indexing note: the lookback window supplied by `RawPriceDataset` is
        `values_norm[t - lookback : t]`, i.e. it ends at `t - 1`, not `t`.
        `get_forecast(steps=...).predicted_mean` is 0-indexed relative to the
        *last observed point*, so a forecast `k` steps ahead of the window's
        last value (at `t - 1`) lands on day `(t - 1) + k`. To land exactly on
        day `t + h` (matching horizon `h` of the `t`-anchored target), we need
        `k = h + 1`, i.e. `forecast[h]` (0-indexed) from a `steps=max_h + 1`
        call -- not `forecast[h - 1]` from `steps=max_h`, which lands one day
        short (on `t + h - 1`). Fixed here as well.
    """

    def __init__(self, config: dict):
        self.config = config
        self.horizons = config.get("horizons", [1, 5, 10, 22])
        self.order = None  # selected by fit() via AIC grid search
        self._max_p = config.get("arima_max_p", 3)
        self._max_d = config.get("arima_max_d", 2)
        self._max_q = config.get("arima_max_q", 3)
        self._n_order_samples = config.get("arima_order_search_samples", 20)

    @property
    def name(self) -> str:
        return "ARIMA"

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        X_train, _ = self._loader_to_series(train_loader)
        if len(X_train) == 0:
            self.order = (1, 1, 0)
            return

        step = max(1, len(X_train) // self._n_order_samples)
        sample = X_train[::step][: self._n_order_samples]

        best_aic = np.inf
        best_order = (1, 1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self._max_p + 1):
                for d in range(self._max_d + 1):
                    for q in range(self._max_q + 1):
                        if p == 0 and q == 0:
                            continue
                        aics = []
                        for series in sample:
                            try:
                                res = ARIMA(series, order=(p, d, q)).fit()
                                if np.isfinite(res.aic):
                                    aics.append(res.aic)
                            except Exception:
                                continue
                        if aics:
                            mean_aic = float(np.mean(aics))
                            if mean_aic < best_aic:
                                best_aic = mean_aic
                                best_order = (p, d, q)

        self.order = best_order
        logger.info(f"ARIMA selected order {self.order} (mean AIC={best_aic:.4f})")

    def predict(self, loader: DataLoader) -> np.ndarray:
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        if self.order is None:
            self.order = (1, 1, 0)

        X, _ = self._loader_to_series(loader)
        mu0, sigma0 = self._get_norm_stats(loader)
        max_h = max(self.horizons)
        # steps=max_h+1: forecast[0] is the window's next point (day t, one
        # past the window's last observed point at t-1); forecast[h] is day
        # t+h, matching horizon h's target base at t (see class docstring).
        forecast_steps = max_h + 1
        preds = np.zeros((len(X), len(self.horizons)))
        eps = 1e-8

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, series in enumerate(X):
                last_val = series[-1]
                try:
                    res = ARIMA(series, order=self.order).fit()
                    forecast = np.asarray(
                        res.get_forecast(steps=forecast_steps).predicted_mean)
                    p_last = max(last_val * sigma0 + mu0, eps)
                    for j, h in enumerate(self.horizons):
                        p_hat = max(forecast[h] * sigma0 + mu0, eps)
                        preds[i, j] = np.log(p_hat / p_last)
                except Exception:
                    preds[i, :] = 0.0

        return preds

    def _get_norm_stats(self, loader) -> Tuple[float, float]:
        """Recover the copper price channel's (index 0) training mean/std
        from the loader's underlying dataset, so the z-scored series this
        baseline operates on can be de-normalized back to real price units.

        `RawPriceDataset` (the dataset actually backing ARIMA's loaders in
        the active pipeline) stores per-variable `.mean`/`.std` arrays fit on
        the training split; index 0 is copper. Falls back to (0.0, 1.0) --
        i.e. treats the series as already being in price units -- if no such
        attributes are found, which only happens for datasets that don't
        expose them (e.g. ad hoc synthetic TensorDatasets used in tests).
        """
        dataset = getattr(loader, "dataset", None)
        mean = getattr(dataset, "mean", None)
        std = getattr(dataset, "std", None)
        if mean is not None and std is not None:
            mean = np.asarray(mean).reshape(-1)
            std = np.asarray(std).reshape(-1)
            if mean.size >= 1 and std.size >= 1:
                return float(mean[0]), float(std[0])
        logger.warning("ARIMABaseline: no per-variable mean/std found on the "
                       "loader's dataset; treating the ARIMA input series as "
                       "already unnormalized (mean=0, std=1) for the "
                       "log-return conversion.")
        return 0.0, 1.0

    def _loader_to_series(self, loader):
        """Extract the copper price channel (index 0) as (N, lookback) arrays."""
        xs, ys = [], []
        for x, y in loader:
            x_np = x.numpy()
            if x_np.ndim == 4:
                # VMD-mode loader (B, T, K, N): reconstruct raw series by
                # summing modes for the copper variable (index 0).
                series = x_np[:, :, :, 0].sum(axis=2)  # (B, T)
            else:
                # Raw loader (B, T, N)
                series = x_np[:, :, 0]  # (B, T)
            xs.append(series)
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)


class LightGBMBaseline:
    """LightGBM with flattened features."""

    def __init__(self, config: dict):
        self.config = config
        self.horizons = config.get("horizons", [1, 5, 10, 22])
        self.models = {}

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        import lightgbm as lgb
        X_train, Y_train = self._loader_to_numpy(train_loader)
        X_val, Y_val = self._loader_to_numpy(val_loader) if val_loader else (None, None)

        for i, h in enumerate(self.horizons):
            callbacks = [lgb.early_stopping(20)] if X_val is not None else None
            model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbosity=-1
            )
            eval_set = [(X_val, Y_val[:, i])] if X_val is not None else None
            model.fit(X_train, Y_train[:, i], eval_set=eval_set, callbacks=callbacks)
            self.models[h] = model

    def predict(self, loader: DataLoader) -> np.ndarray:
        X, _ = self._loader_to_numpy(loader)
        return np.column_stack([self.models[h].predict(X) for h in self.horizons])

    def _loader_to_numpy(self, loader):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.numpy().reshape(x.size(0), -1))
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)
