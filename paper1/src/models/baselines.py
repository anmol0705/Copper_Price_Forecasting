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
        min_epochs = self.config.get("min_epochs", 0)
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
                # min_epochs floor mirrors VMDMFGNNTrainer.fit (trainer.py):
                # best-checkpoint selection AND early-stopping patience are both
                # gated on the same floor, so a baseline can't win on an
                # under-trained epoch-0-ish checkpoint the way VMD-MFGNN's fix
                # already guards against (see trainer.py comment above
                # best_eligible_val_mse). Without this, the 7 baseline models
                # were the only main-table entries not honoring the floor.
                if (epoch + 1) >= min_epochs and val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                elif (epoch + 1) >= min_epochs:
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
    """VMD-LSTM (Liu, Wang, Guo et al. 2020, "Non-ferrous metals price
    forecasting based on variational mode decomposition and LSTM network",
    *Resources Policy*): one LSTM per VMD mode, per-mode predictions summed
    back into the forecast.

    **Univariate by design.** Liu et al.'s method decomposes the *target*
    price series alone and fits an independent LSTM to each resulting mode;
    there are no exogenous variables in the loop. An earlier version of this
    class fed all `N` variables jointly into each per-mode LSTM, which made it
    a multivariate model the cited paper does not describe -- a
    mischaracterisation, not merely a simplification. Corrected here: each
    per-mode LSTM consumes the copper channel of its own mode only
    (`input_size=1`). Copper is variable index 0 of the mode tensor, the same
    convention `ARIMABaseline._loader_to_series` relies on.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        K = config.get("num_modes", 5)
        hd = config.get("hidden_dim", 64)
        self.K = K
        self.target_idx = int(config.get("target_var_idx", 0))
        self.lstms = nn.ModuleList([
            nn.LSTM(1, hd, 2, batch_first=True) for _ in range(K)
        ])
        self.heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(hd, 1) for _ in self.horizons])
            for _ in range(K)
        ])

    def _forward_flat(self, x):
        # x: (B, T, K, N) -- only the target variable's own modes are used.
        B, T, K, N = x.shape
        ti = self.target_idx
        preds = torch.zeros(B, len(self.horizons), device=x.device)
        for k in range(self.K):
            mode_x = x[:, :, k, ti:ti + 1]  # (B, T, 1)
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


# ---------------------------------------------------------------------------
# Faithful MTGNN reproduction (Wu et al., KDD 2020)
#
# The five private classes below (`_MTGNNNConv` ... `_MTGNNLayerNorm`) and the
# `_MTGNNNet` backbone are direct ports of the official implementation at
# https://github.com/nnzhan/MTGNN (`layer.py` and `net.py`, `master` branch,
# maintained by the paper's lead author). They are kept byte-for-byte
# equivalent to the originals in every computation; the only edits are
# mechanical Python-style renames (snake_case classes -> CamelCase private
# names) and two device-handling changes that are behaviour-preserving:
#   * `graph_constructor` took a `device` argument and allocated its top-k mask
#     on it; here the mask is allocated on the adjacency's own device, so the
#     module moves correctly under `TorchBaseline.fit()`'s `self.to(device)`.
#   * `gtnet.idx` was a plain tensor attribute; here it is a registered buffer
#     for the same reason.
# ---------------------------------------------------------------------------

class _MTGNNNConv(nn.Module):
    """MTGNN `nconv`: one-hop graph propagation over the node axis."""

    def forward(self, x, A):
        return torch.einsum("ncwl,vw->ncvl", (x, A)).contiguous()


class _MTGNNLinear(nn.Module):
    """MTGNN `linear`: 1x1 conv acting as a channel-wise MLP."""

    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0),
                             stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class _MTGNNMixProp(nn.Module):
    """MTGNN `mixprop`: mix-hop propagation.

    Runs `gdep` information-propagation steps with the retain ratio `alpha`
    (`h <- alpha * x + (1 - alpha) * A_norm h`), *keeps every intermediate hop*,
    concatenates all `gdep + 1` of them along the channel axis, and mixes them
    with a single 1x1 conv (the "information selection" step). This is MTGNN's
    key GCN innovation and is NOT equivalent to stacking `gdep` plain GCN
    layers, which would discard the lower-order hops.
    """

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = _MTGNNNConv()
        self.mlp = _MTGNNLinear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        return self.mlp(ho)


class _MTGNNDilatedInception(nn.Module):
    """MTGNN `dilated_inception`: four parallel dilated 1-D convolutions with
    kernel sizes {2, 3, 6, 7}, each producing `cout / 4` channels, truncated to
    the shortest output length and concatenated. Distinct from (and strictly
    richer than) a stack of single-kernel dilated causal convs.
    """

    kernel_set = [2, 3, 6, 7]

    def __init__(self, cin, cout, dilation_factor=2):
        super().__init__()
        assert cout % len(self.kernel_set) == 0, (
            f"MTGNN dilated_inception requires out-channels divisible by "
            f"{len(self.kernel_set)}, got {cout}")
        self.tconv = nn.ModuleList()
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = [conv(input) for conv in self.tconv]
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        return torch.cat(x, dim=1)


class _MTGNNGraphConstructor(nn.Module):
    """MTGNN `graph_constructor`: the uni-directional saturated top-k graph.

    Two node-embedding tables are passed through their own linear maps and a
    *saturating* `tanh(alpha * .)` nonlinearity, combined ANTI-SYMMETRICALLY
    (`M1 M2^T - M2 M1^T`) so that at most one direction of each pair survives,
    saturated again by `relu(tanh(alpha * a))`, and finally sparsified by a
    hard top-k mask per row. The `alpha` (`tanhalpha`, default 3) controls the
    saturation rate, which is what makes this "saturated top-k" rather than
    this project's plain `softmax(relu(E1 E2^T))` top-k.

    Note (faithfulness, deliberate): the official code adds uniform jitter
    `adj + rand_like(adj) * 0.01` *inside* the top-k selection, and does so in
    both train and eval mode. This is ported unchanged, so predictions from
    this baseline are mildly non-deterministic at inference. It is kept rather
    than "fixed" because the point of this class is fidelity to the published
    implementation.
    """

    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super().__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def _nodevecs(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))
        return nodevec1, nodevec2

    def forward(self, idx):
        nodevec1, nodevec2 = self._nodevecs(idx)
        a = (torch.mm(nodevec1, nodevec2.transpose(1, 0))
             - torch.mm(nodevec2, nodevec1.transpose(1, 0)))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0), device=adj.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        return adj * mask

    def full_a(self, idx):
        """Dense (un-sparsified) adjacency, MTGNN's `fullA` -- for diagnostics."""
        nodevec1, nodevec2 = self._nodevecs(idx)
        a = (torch.mm(nodevec1, nodevec2.transpose(1, 0))
             - torch.mm(nodevec2, nodevec1.transpose(1, 0)))
        return F.relu(torch.tanh(self.alpha * a))


class _MTGNNLayerNorm(nn.Module):
    """MTGNN's node-indexed LayerNorm (`layer.py::LayerNorm`)."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]),
                                self.weight[:, idx, :], self.bias[:, idx, :],
                                self.eps)
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight,
                            self.bias, self.eps)


class _MTGNNNet(nn.Module):
    """Port of `net.py::gtnet` -- the full MTGNN backbone.

    Defaults are exactly the official repo's `net.py` signature defaults plus
    `train_multi_step.py`'s argparse default for `gcn_depth` (=2, which is
    positional in `gtnet.__init__` and so has no signature default).
    """

    def __init__(self, gcn_true=True, build_a_true=True, gcn_depth=2,
                 num_nodes=16, predefined_A=None, static_feat=None,
                 dropout=0.3, subgraph_size=20, node_dim=40,
                 dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128,
                 seq_length=12, in_dim=2, out_dim=12, layers=3,
                 propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super().__init__()
        self.gcn_true = gcn_true
        self.buildA_true = build_a_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = _MTGNNGraphConstructor(num_nodes, subgraph_size, node_dim,
                                         alpha=tanhalpha,
                                         static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1)
                                * (dilation_exponential ** layers - 1)
                                / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1)
                                    * (dilation_exponential ** j - 1)
                                    / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(_MTGNNDilatedInception(
                    residual_channels, conv_channels,
                    dilation_factor=new_dilation))
                self.gate_convs.append(_MTGNNDilatedInception(
                    residual_channels, conv_channels,
                    dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=residual_channels, kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=conv_channels, out_channels=skip_channels,
                        kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=conv_channels, out_channels=skip_channels,
                        kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(_MTGNNMixProp(
                        conv_channels, residual_channels, gcn_depth, dropout,
                        propalpha))
                    self.gconv2.append(_MTGNNMixProp(
                        conv_channels, residual_channels, gcn_depth, dropout,
                        propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(_MTGNNLayerNorm(
                        (residual_channels, num_nodes,
                         self.seq_length - rf_size_j + 1),
                        elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(_MTGNNLayerNorm(
                        (residual_channels, num_nodes,
                         self.receptive_field - rf_size_j + 1),
                        elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1), bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim,
                                   out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(
                in_channels=residual_channels, out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim,
                                   out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels,
                                   out_channels=skip_channels,
                                   kernel_size=(1, 1), bias=True)

        self.register_buffer("idx", torch.arange(self.num_nodes))

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, \
            "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx if idx is None else idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout,
                                    training=self.training))
        for i in range(self.layers):
            residual = x
            filter_ = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter_ * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.skip_convs[i](x)
            skip = s + skip
            if self.gcn_true:
                x = (self.gconv1[i](x, adp)
                     + self.gconv2[i](x, adp.transpose(1, 0)))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx if idx is None else idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class MTGNNBaseline(TorchBaseline):
    """Faithful reproduction of MTGNN (Wu, Pan, Long, Jiang, Chang & Zhang,
    "Connecting the Dots: Multivariate Time Series Forecasting with Graph
    Neural Networks", KDD 2020), ported from the authors' official
    implementation at https://github.com/nnzhan/MTGNN (`layer.py`, `net.py`).

    Unlike `SimpleMTGNN` (kept alongside this class as a deliberately
    lower-capacity graph-learning baseline), this class implements all four of
    MTGNN's actual components:

      * **Saturated top-k graph learning** (`_MTGNNGraphConstructor`):
        anti-symmetric bilinear score over two linearly-mapped node-embedding
        tables, saturated twice by `tanh(alpha * .)` with `alpha = tanhalpha`,
        then hard row-wise top-k sparsification -- not `SimpleMTGNN`'s dense
        `softmax(relu(E1 E2^T))`.
      * **Mix-hop propagation** (`_MTGNNMixProp`): `gcn_depth` propagation
        steps with retain ratio `propalpha`, concatenating every intermediate
        hop and mixing them with a 1x1 conv; applied twice per layer, once on
        the adjacency and once on its transpose (in/out-flow).
      * **Dilated-inception temporal convolution**
        (`_MTGNNDilatedInception`): four parallel dilated convs with kernel
        sizes {2, 3, 6, 7} concatenated, used in a gated filter/gate pair --
        not `SimpleMTGNN`'s two plain single-kernel dilated convs.
      * **Output skip connections**: a `skip0` projection of the raw input
        plus a per-layer `skip_conv`, summed with a final `skipE`, feeding the
        two end convolutions; plus per-layer residual connections and MTGNN's
        node-indexed LayerNorm.

    Adaptations for this project (everything else is the official default):

      * `in_dim=1` (this project supplies a single price channel per variable;
        MTGNN's traffic setup uses 2: value + time-of-day).
      * `out_dim=len(horizons)` and the copper node (variable index 0) is read
        off the output, so one model emits all 4 horizons, matching the
        `TorchBaseline` multi-horizon convention used by every other baseline
        here. MTGNN's own multi-step setting is structurally identical
        (`out_dim = seq_out_len`).
      * `seq_length=lookback` (60 here vs. the repo's 12). Receptive field is
        `layers * 6 + 1 = 19 < 60`, so the `seq_length > receptive_field`
        branch of the official code is the one exercised, as intended.
      * `subgraph_size` (top-k) is clamped to `min(20, num_vars - 1)` because
        the official default of 20 exceeds this project's node count and
        `topk` would raise. Fidelity caveat: with N=16 and k=15, top-k
        sparsification is close to vacuous here regardless of the clamp --
        MTGNN's sparsification matters at its original N=207/N=137 scale.

    Architectural hyperparameters are NOT taken from this project's config
    (`hidden_dim`, `dropout`, ...) precisely so the reproduction claim holds:
    `gcn_depth=2`, `node_dim=40`, `dropout=0.3`, `conv_channels=32`,
    `residual_channels=32`, `skip_channels=64`, `end_channels=128`,
    `layers=3`, `dilation_exponential=1`, `propalpha=0.05`, `tanhalpha=3` are
    the repo's own values (`net.py` signature defaults; `gcn_depth` from
    `train_multi_step.py`'s argparse default). Only `num_vars`, `lookback`,
    `horizons` and `device` come from config. Override any of them explicitly
    via config keys prefixed `mtgnn_` if an ablation needs to.

    Scope of the reproduction claim: this is an *architecturally* faithful
    reproduction, trained under this project's common baseline protocol
    (`TorchBaseline.fit`: Adam, lr 1e-3, weight decay 1e-5, gradient clipping
    at 1.0, shared early stopping) rather than MTGNN's own training script,
    which additionally uses curriculum learning over the output horizon and
    node subsampling (`num_split`). The shared loop is deliberate -- every
    baseline in this paper must be trained and evaluated identically -- but
    end-to-end fidelity is not claimed, only architectural fidelity.

    Known non-determinism (ported deliberately, not a bug): the official
    graph constructor adds uniform jitter inside its top-k selection in both
    train and eval mode, so `predict()` is mildly stochastic.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        nv = int(config["num_vars"])
        lookback = int(config.get("lookback", 60))
        g = config.get  # official defaults below; `mtgnn_`-prefixed overrides

        self.num_vars = nv
        self.net = _MTGNNNet(
            gcn_true=True,
            build_a_true=True,
            gcn_depth=int(g("mtgnn_gcn_depth", 2)),
            num_nodes=nv,
            dropout=float(g("mtgnn_dropout", 0.3)),
            subgraph_size=min(int(g("mtgnn_subgraph_size", 20)),
                              max(nv - 1, 1)),
            node_dim=int(g("mtgnn_node_dim", 40)),
            dilation_exponential=int(g("mtgnn_dilation_exponential", 1)),
            conv_channels=int(g("mtgnn_conv_channels", 32)),
            residual_channels=int(g("mtgnn_residual_channels", 32)),
            skip_channels=int(g("mtgnn_skip_channels", 64)),
            end_channels=int(g("mtgnn_end_channels", 128)),
            seq_length=lookback,
            in_dim=1,
            out_dim=len(self.horizons),
            layers=int(g("mtgnn_layers", 3)),
            propalpha=float(g("mtgnn_propalpha", 0.05)),
            tanhalpha=float(g("mtgnn_tanhalpha", 3)),
        )

    def _forward_flat(self, x):
        # x: (B, T, N) -> MTGNN expects (B, in_dim, N, T)
        inp = x.permute(0, 2, 1).unsqueeze(1)          # (B, 1, N, T)
        out = self.net(inp)                            # (B, H, N, 1)
        return out[:, :, 0, 0]                         # copper node -> (B, H)


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
