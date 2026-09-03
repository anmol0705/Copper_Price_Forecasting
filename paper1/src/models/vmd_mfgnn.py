"""VMD-MFGNN: Multi-Frequency Graph Neural Network with Variational Mode Decomposition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from typing import Dict, List, Optional, Tuple


class FrequencyGraphConstructor(nn.Module):
    """Constructs per-frequency-band adjacency matrices."""

    def __init__(self, num_vars: int, embed_dim: int = 16,
                 graph_type: str = "learned", topk: Optional[int] = None,
                 normalize_embeddings: bool = False):
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

        if graph_type == "learned":
            self.emb1 = nn.Embedding(num_vars, embed_dim)
            self.emb2 = nn.Embedding(num_vars, embed_dim)
            nn.init.xavier_uniform_(self.emb1.weight)
            nn.init.xavier_uniform_(self.emb2.weight)

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
            adj = torch.softmax(F.relu(e1 @ e2.T), dim=-1)  # (N, N)
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
            adj = torch.softmax(F.relu(e1 @ e2.T), dim=-1)
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


class FrequencyBandModule(nn.Module):
    """Processes one frequency band: GAT for cross-variable + LSTM for temporal."""

    def __init__(self, num_vars: int, input_dim: int, hidden_dim: int,
                 num_heads: int = 4, num_gnn_layers: int = 2,
                 temporal_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, concat=True, edge_dim=1)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        # Temporal LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=temporal_layers,
                            batch_first=True, dropout=dropout if temporal_layers > 1 else 0)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, num_vars) — one band's data for all variables
        Returns: (batch, num_vars, hidden_dim)
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
        gat_out = torch.stack(gat_out, dim=1)

        # LSTM per variable
        out = []
        for v in range(N):
            var_seq = gat_out[:, :, v, :]  # (B, T, hidden_dim)
            lstm_out, _ = self.lstm(var_seq)
            out.append(lstm_out[:, -1, :])  # (B, hidden_dim)
        out = torch.stack(out, dim=1)  # (B, N, hidden_dim)
        return self.temporal_norm(out)

    def _batch_edge_index(self, edge_index: torch.Tensor, batch_size: int,
                          num_nodes: int) -> torch.Tensor:
        """Create batched edge_index for B independent graphs."""
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        offsets = offsets.unsqueeze(1).expand(-1, edge_index.size(1))  # (B, E)
        batch_ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, E)
        batch_ei = batch_ei + offsets.unsqueeze(1)  # (B, 2, E)
        return batch_ei.reshape(2, -1)  # (2, B*E)


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
                 normalize_graph_embeddings: bool = False):
        super().__init__()
        self.num_vars = num_vars
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.horizons = horizons

        # Per-band graph constructors
        self.graph_constructors = nn.ModuleList([
            FrequencyGraphConstructor(num_vars, embed_dim=16, graph_type=graph_type,
                                       normalize_embeddings=normalize_graph_embeddings)
            for _ in range(num_modes)
        ])

        # Per-band processing modules
        self.band_modules = nn.ModuleList([
            FrequencyBandModule(num_vars, 1, hidden_dim, num_heads,
                                num_gnn_layers, temporal_layers, dropout)
            for _ in range(num_modes)
        ])

        # Attention fusion
        self.fusion = AttentionFusion(hidden_dim, num_modes)

        # Prediction heads (one per horizon)
        self.pred_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for h in horizons
        })

        self._attention_weights = None

    def forward(self, x: torch.Tensor,
                precomputed_adjs: Optional[List[torch.Tensor]] = None
                ) -> Dict[str, torch.Tensor]:
        """
        x: (batch, lookback, K, num_vars) — VMD mode features
        Returns: dict with predictions per horizon + attention weights
        """
        B, T, K, N = x.shape

        band_outputs = []
        for k in range(self.num_modes):
            adj = precomputed_adjs[k] if precomputed_adjs else None
            edge_index, edge_weight = self.graph_constructors[k](adj)
            band_x = x[:, :, k, :]  # (B, T, N)
            band_out = self.band_modules[k](band_x, edge_index, edge_weight)
            # Extract copper node (index 0)
            copper_repr = band_out[:, 0, :]  # (B, hidden_dim)
            band_outputs.append(copper_repr)

        # Fuse across frequency bands
        fused, attn_weights = self.fusion(band_outputs)
        self._attention_weights = attn_weights.detach()

        # Predict at each horizon
        predictions = {}
        for h in self.horizons:
            predictions[str(h)] = self.pred_heads[str(h)](fused).squeeze(-1)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._attention_weights

    def get_learned_graphs(self) -> List[torch.Tensor]:
        graphs = []
        for gc in self.graph_constructors:
            if gc.graph_type == "learned":
                graphs.append(gc.get_adjacency())
        return graphs
