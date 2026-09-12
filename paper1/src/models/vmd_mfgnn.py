"""VMD-MFGNN: Multi-Frequency Graph Neural Network with Variational Mode Decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import dense_to_sparse
from typing import Dict, List, Optional, Tuple


class FrequencyGraphConstructor(nn.Module):
    """Constructs per-frequency-band adjacency matrices."""

    def __init__(self, num_vars: int, embed_dim: int = 16,
                 graph_type: str = "learned", topk: Optional[int] = None,
                 normalize_embeddings: bool = False,
                 use_temperature: bool = False):
        super().__init__()
        self.num_vars = num_vars
        self.graph_type = graph_type
        # For num_vars=10 (locked protocol), keep each node's top-5 outgoing
        # edges (~half connectivity) so the learned graph is a meaningfully
        # sparse structure instead of the fully-connected softmax output.
        self.topk = topk if topk is not None else max(1, num_vars // 2)
        # OFF by default (unchanged behavior). When True, L2-normalizes
        # emb1/emb2 to unit norm before the bilinear dot product, turning
        # the adjacency logits into cosine similarities. This is one of two
        # fixes (see STATUS.md "Adjacency Collapse Diagnosed") for the
        # isotropic norm collapse of these embeddings under Adam's coupled
        # weight decay: with unit-norm embeddings the softmax spread can no
        # longer be washed out just by the optimizer shrinking ||emb||.
        self.normalize_embeddings = normalize_embeddings
        # OFF by default (unchanged behavior). When True, adds ONE learnable
        # scalar log-temperature per graph constructor (i.e. one per VMD band
        # in VMDMFGNN, one total in PooledGraphMFGNN) and divides the
        # adjacency logits by tau = exp(log_temperature) before the softmax.
        #
        # WHY log-parameterized (this is not a stylistic choice): the fix this
        # implements targets the same weight-decay collapse that already
        # destroys emb1/emb2. A RAW `tau` nn.Parameter under coupled L2 decay
        # is pulled multiplicatively toward 0 (simulated this session: ~2e-4
        # after 3000 steps), and tau -> 0 blows the logits up / re-uniformizes
        # the graph once combined with top-k. Parameterizing `s` with
        # tau = exp(s) means decay pulls s -> 0, i.e. tau -> 1.0, which is
        # exactly the neutral (current) behavior: the failure mode degrades
        # gracefully instead of catastrophically. `s` should ALSO be placed in
        # the zero-weight-decay parameter group (see trainer.py's
        # `no_decay_graph_embeddings`), which matches on the parameter name
        # "log_temperature"; the exp() parameterization is the safety net for
        # when it is not.
        #
        # Initialized at 0.0 => tau == 1.0 => softmax(logits / 1.0), i.e.
        # numerically identical to the pre-existing path at step 0.
        self.use_temperature = use_temperature

        if graph_type == "learned":
            self.emb1 = nn.Embedding(num_vars, embed_dim)
            self.emb2 = nn.Embedding(num_vars, embed_dim)
            nn.init.xavier_uniform_(self.emb1.weight)
            nn.init.xavier_uniform_(self.emb2.weight)
            if use_temperature:
                self.log_temperature = nn.Parameter(torch.zeros(()))

    def _adjacency_logits_to_adj(self, logits: torch.Tensor) -> torch.Tensor:
        """softmax(logits / tau) with tau = exp(log_temperature) when the
        temperature fix is enabled, and plain softmax(logits) otherwise."""
        if self.use_temperature and hasattr(self, "log_temperature"):
            logits = logits / torch.exp(self.log_temperature)
        return torch.softmax(logits, dim=-1)

    def _topk_sparsify(self, adj: torch.Tensor) -> torch.Tensor:
        """Zero out all but each row's top-k highest-weight entries.

        Applied to the dense adjacency BEFORE dense_to_sparse so that only
        the surviving (top-k) entries appear in the sparse edge list —
        entries zeroed here never reach dense_to_sparse (which drops exact
        zeros), so gradients only flow through the kept edges. This keeps
        the fix differentiable-compatible since masking is just element-wise
        zeroing of the dense tensor, not an in-place/non-differentiable op
        on the surviving values.
        """
        k = min(self.topk, adj.size(-1))
        topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_idx, 1.0)
        return adj * mask

    def forward(self, precomputed_adj: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (edge_index, edge_weight) for PyG."""
        if self.graph_type == "correlation" and precomputed_adj is not None:
            adj = precomputed_adj
        else:
            idx = torch.arange(self.num_vars, device=self.emb1.weight.device)
            e1 = self.emb1(idx)  # (N, d)
            e2 = self.emb2(idx)  # (N, d)
            if self.normalize_embeddings:
                e1 = F.normalize(e1, dim=-1)
                e2 = F.normalize(e2, dim=-1)
            adj = self._adjacency_logits_to_adj(F.relu(e1 @ e2.T))  # (N, N)
            # Scope sparsification to the learned-graph path only — the
            # correlation-graph path is already sparse via its own
            # thresholding in compute_correlation_adjacency().
            adj = self._topk_sparsify(adj)

        edge_index, edge_weight = dense_to_sparse(adj)
        return edge_index, edge_weight

    def get_adjacency(self) -> torch.Tensor:
        """Returns the top-k sparsified adjacency actually used in forward()."""
        with torch.no_grad():
            idx = torch.arange(self.num_vars, device=self.emb1.weight.device)
            e1 = self.emb1(idx)
            e2 = self.emb2(idx)
            if self.normalize_embeddings:
                e1 = F.normalize(e1, dim=-1)
                e2 = F.normalize(e2, dim=-1)
            adj = self._adjacency_logits_to_adj(F.relu(e1 @ e2.T))
            return self._topk_sparsify(adj)

    @staticmethod
    def compute_correlation_adjacency(band_signal: torch.Tensor,
                                       threshold: float = 0.3) -> torch.Tensor:
        """Build a correlation-graph adjacency matrix for one frequency band.

        band_signal: (T, num_vars) — a single band's mode signal for every
            variable over a training window (rolling/expanding window slice).
        threshold: edges are kept only where |Pearson r| > threshold; all
            other entries are zeroed out. 0.3 is used as a conventional
            "weak-to-moderate correlation" cutoff for financial time series
            graphs (consistent with the correlation-graph ablation in the
            locked protocol) so the resulting graph is sparse rather than
            fully connected.

        Returns: (num_vars, num_vars) adjacency matrix of thresholded
            Pearson correlations (self-loops on the diagonal are kept at 1).
        """
        if band_signal.dim() != 2:
            raise ValueError(
                f"band_signal must be (T, num_vars), got shape {tuple(band_signal.shape)}"
            )
        # torch.corrcoef expects variables as rows, so transpose to (N, T)
        corr = torch.corrcoef(band_signal.T)  # (N, N)
        corr = torch.nan_to_num(corr, nan=0.0)
        mask = corr.abs() > threshold
        adj = corr * mask
        return adj


class BandGraphEncoder(nn.Module):
    """Shared (across forecast horizons) per-band graph feature extractor:
    input projection + GAT over the learned adjacency, applied at every
    timestep. Produces per-timestep, per-variable node representations and
    stops BEFORE any temporal (LSTM) modelling.

    Split out of the original `FrequencyBandModule` for the multi-horizon
    architecture change (PLAN.md §3c): the graph/GAT stage stays FULLY SHARED
    across all forecast horizons — so the learned adjacency keeps receiving
    the same combined 4-horizon gradient it did before, rather than having
    that (already weak) gradient split four ways — while the temporal stage
    downstream of it is instantiated once per horizon (see `HorizonBranch`).

    `FrequencyBandModule` below subclasses this and re-adds the LSTM, so the
    original combined GAT+LSTM module (used by `PooledGraphMFGNN`) keeps its
    exact previous behavior AND its exact previous parameter names/state-dict
    keys (input_proj / gat_layers / gat_norms / lstm / temporal_norm).
    """

    def __init__(self, num_vars: int, input_dim: int, hidden_dim: int,
                 num_heads: int = 4, num_gnn_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers. GATv2 (Brody, Alon & Yahav, ICLR 2022) replaces the
        # original GAT here: it is a same-cost drop-in with strictly more
        # expressive (dynamic rather than static) attention. Constructor
        # signature verified identical to GATConv's for the installed
        # torch_geometric (2.8.0.post1) for every argument used below.
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                          dropout=dropout, concat=True, edge_dim=1)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, num_vars) — one band's data for all variables
        Returns: (batch, lookback, num_vars, hidden_dim) — per-timestep GAT
            node representations (no temporal pooling applied yet).
        """
        B, T, N = x.shape
        # Project each variable's scalar value to hidden_dim
        x = x.unsqueeze(-1)  # (B, T, N, 1)
        x = self.input_proj(x)  # (B, T, N, hidden_dim)

        # Apply GAT at each timestep
        gat_out = []
        for t in range(T):
            h = x[:, t]  # (B, N, hidden_dim)
            # Process each batch item through GAT
            batch_h = h.reshape(B * N, -1)  # treat as B separate graphs
            # Create batched edge_index
            batch_edge_index = self._batch_edge_index(edge_index, B, N)
            batch_edge_weight = edge_weight.repeat(B)

            for gat, norm in zip(self.gat_layers, self.gat_norms):
                residual = batch_h
                batch_h = gat(batch_h, batch_edge_index,
                              edge_attr=batch_edge_weight.unsqueeze(-1))
                batch_h = norm(batch_h + residual)
                batch_h = self.dropout(F.elu(batch_h))

            gat_out.append(batch_h.reshape(B, N, -1))

        # Stack temporal: (B, T, N, hidden_dim)
        return torch.stack(gat_out, dim=1)

    def _batch_edge_index(self, edge_index: torch.Tensor, batch_size: int,
                          num_nodes: int) -> torch.Tensor:
        """Create batched edge_index for B independent graphs."""
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        offsets = offsets.unsqueeze(1).expand(-1, edge_index.size(1))  # (B, E)
        batch_ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, E)
        batch_ei = batch_ei + offsets.unsqueeze(1)  # (B, 2, E)
        return batch_ei.reshape(2, -1)  # (2, B*E)


class FrequencyBandModule(BandGraphEncoder):
    """Processes one frequency band: GAT for cross-variable + LSTM for temporal.

    Unchanged in behavior and in parameter naming from the pre-§3c version
    (apart from the GATConv -> GATv2Conv swap inherited from
    `BandGraphEncoder`). `VMDMFGNN` no longer uses this class — it now uses
    `BandGraphEncoder` + per-horizon `HorizonBranch`es — but
    `PooledGraphMFGNN` (a different, pooled-graph architecture) still does,
    so it is kept intact rather than deleted.
    """

    def __init__(self, num_vars: int, input_dim: int, hidden_dim: int,
                 num_heads: int = 4, num_gnn_layers: int = 2,
                 temporal_layers: int = 2, dropout: float = 0.1):
        super().__init__(num_vars, input_dim, hidden_dim, num_heads,
                         num_gnn_layers, dropout)
        # Temporal LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=temporal_layers,
                            batch_first=True, dropout=dropout if temporal_layers > 1 else 0)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, num_vars) — one band's data for all variables
        Returns: (batch, num_vars, hidden_dim)
        """
        gat_out = super().forward(x, edge_index, edge_weight)  # (B, T, N, H)
        N = gat_out.size(2)

        # LSTM per variable
        out = []
        for v in range(N):
            var_seq = gat_out[:, :, v, :]  # (B, T, hidden_dim)
            lstm_out, _ = self.lstm(var_seq)
            out.append(lstm_out[:, -1, :])  # (B, hidden_dim)
        out = torch.stack(out, dim=1)  # (B, N, hidden_dim)
        return self.temporal_norm(out)


class HorizonBranch(nn.Module):
    """Everything downstream of the shared graph/GAT stage, for ONE horizon.

    Holds one LSTM (+ LayerNorm) per VMD band, one `AttentionFusion` over
    those K band representations, and one prediction head. Instantiated once
    per forecast horizon by `VMDMFGNN`, so K bands x H horizons LSTMs exist
    in total (5 x 4 = 20 at the project's default settings) instead of the
    previous K shared ones — each individually the same size as before.

    Rationale (PLAN.md §3c): one temporal representation previously had to
    serve both the 1-day (noise-dominated) and 22-day (trend-dominated)
    horizon simultaneously; the graph stage stays shared precisely so its
    gradient signal is NOT diluted by this split.
    """

    def __init__(self, num_modes: int, hidden_dim: int,
                 temporal_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_modes = num_modes
        self.lstms = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, num_layers=temporal_layers,
                    batch_first=True,
                    dropout=dropout if temporal_layers > 1 else 0)
            for _ in range(num_modes)
        ])
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_modes)
        ])
        self.fusion = AttentionFusion(hidden_dim, num_modes)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, band_node_seqs: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        band_node_seqs: list of K tensors, each (batch, lookback, hidden_dim) —
            the shared GAT stage's output for the TARGET node (copper), one
            entry per band.
        Returns: (prediction (batch,), attention weights (batch, K))
        """
        band_outputs = []
        for k, seq in enumerate(band_node_seqs):
            lstm_out, _ = self.lstms[k](seq)  # (B, T, H)
            band_outputs.append(self.temporal_norms[k](lstm_out[:, -1, :]))
        fused, attn_weights = self.fusion(band_outputs)
        return self.head(fused).squeeze(-1), attn_weights


class AttentionFusion(nn.Module):
    """Attention-weighted fusion across K frequency bands."""

    def __init__(self, hidden_dim: int, num_modes: int):
        super().__init__()
        # Small init (std=0.02) so attention logits at hidden_dim=64 start
        # near-uniform across bands instead of saturating on one random
        # band before any training (logits scale with std * sqrt(hidden_dim)).
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_modes = num_modes

    def forward(self, band_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        band_outputs: list of K tensors, each (batch, hidden_dim)
        Returns: (batch, hidden_dim), (batch, K) attention weights
        """
        keys = torch.stack([self.key_proj(h) for h in band_outputs], dim=1)  # (B, K, H)
        attn = torch.einsum("bkh,h->bk", keys, self.query)  # (B, K)
        attn_weights = F.softmax(attn, dim=-1)  # (B, K)
        stacked = torch.stack(band_outputs, dim=1)  # (B, K, H)
        fused = torch.einsum("bk,bkh->bh", attn_weights, stacked)  # (B, H)
        return fused, attn_weights


class VMDMFGNN(nn.Module):
    """Multi-Frequency Graph Neural Network with VMD decomposition."""

    def __init__(self, num_vars: int, num_modes: int = 5, hidden_dim: int = 64,
                 num_heads: int = 4, num_gnn_layers: int = 2,
                 temporal_layers: int = 2, dropout: float = 0.1,
                 horizons: List[int] = [1, 5, 10, 22],
                 graph_type: str = "learned",
                 normalize_graph_embeddings: bool = False,
                 use_graph_temperature: bool = False):
        super().__init__()
        self.num_vars = num_vars
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.horizons = horizons

        # Per-band graph constructors
        self.graph_constructors = nn.ModuleList([
            FrequencyGraphConstructor(num_vars, embed_dim=16, graph_type=graph_type,
                                       normalize_embeddings=normalize_graph_embeddings,
                                       use_temperature=use_graph_temperature)
            for _ in range(num_modes)
        ])

        # Per-band graph+GAT feature extractors, SHARED across all horizons
        # (PLAN.md §3c): the adjacency keeps receiving the combined gradient
        # of every horizon's loss, exactly as before this change.
        self.band_encoders = nn.ModuleList([
            BandGraphEncoder(num_vars, 1, hidden_dim, num_heads,
                             num_gnn_layers, dropout)
            for _ in range(num_modes)
        ])

        # Per-horizon temporal (LSTM) + band fusion + prediction head. Each
        # horizon gets its own K LSTMs, its own AttentionFusion, and its own
        # head, all consuming the SAME shared GAT output.
        self.horizon_branches = nn.ModuleDict({
            str(h): HorizonBranch(num_modes, hidden_dim, temporal_layers, dropout)
            for h in horizons
        })

        self._attention_weights = None
        self._attention_weights_per_horizon = None

    def forward(self, x: torch.Tensor,
                precomputed_adjs: Optional[List[torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        """
        x: (batch, lookback, K, num_vars) — VMD mode features
        Returns: dict with predictions per horizon + attention weights
        """
        B, T, K, N = x.shape

        # ---- Shared stage: per-band graph + GAT (identical for every horizon)
        band_node_seqs = []
        for k in range(self.num_modes):
            adj = precomputed_adjs[k] if precomputed_adjs else None
            edge_index, edge_weight = self.graph_constructors[k](adj)
            band_x = x[:, :, k, :]  # (B, T, N)
            gat_out = self.band_encoders[k](band_x, edge_index, edge_weight)
            # Extract copper node (index 0) time series: (B, T, hidden_dim).
            # Only the target node's sequence is carried into the temporal
            # stage — the pre-§3c code also ran its LSTM over all N nodes but
            # then used node 0 alone, and the LSTM is applied independently
            # per node, so this is mathematically equivalent and strictly
            # cheaper (N-fold fewer LSTM sequence evaluations). Not bit-
            # identical under a fixed seed in train mode, since the old code
            # drew N dropout masks per band where this draws one.
            band_node_seqs.append(gat_out[:, :, 0, :])

        # ---- Per-horizon stage: own LSTMs + own fusion + own head
        predictions = {}
        attn_per_horizon = {}
        for h in self.horizons:
            pred, attn_weights = self.horizon_branches[str(h)](band_node_seqs)
            predictions[str(h)] = pred
            attn_per_horizon[str(h)] = attn_weights.detach()

        self._attention_weights_per_horizon = attn_per_horizon
        # Backward-compatible single (B, K) view for existing consumers
        # (trainer.py's interpretability dump, visualize.plot_attention_weights):
        # the mean band-attention across horizons.
        self._attention_weights = torch.stack(
            list(attn_per_horizon.values()), dim=0).mean(dim=0)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Band attention averaged over horizons, shape (batch, K).

        Kept at this shape for backward compatibility with existing
        consumers; use `get_attention_weights_per_horizon()` for the new
        per-horizon breakdown introduced with the §3c architecture.
        """
        return self._attention_weights

    def get_attention_weights_per_horizon(self) -> Optional[Dict[str, torch.Tensor]]:
        """Per-horizon band attention: {str(horizon): (batch, K)}."""
        return self._attention_weights_per_horizon

    def get_learned_graphs(self) -> List[torch.Tensor]:
        graphs = []
        for gc in self.graph_constructors:
            if gc.graph_type == "learned":
                graphs.append(gc.get_adjacency())
        return graphs
