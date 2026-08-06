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
                 graph_type: str = "learned"):
        super().__init__()
        self.num_vars = num_vars
        self.graph_type = graph_type

        if graph_type == "learned":
            self.emb1 = nn.Embedding(num_vars, embed_dim)
            self.emb2 = nn.Embedding(num_vars, embed_dim)
            nn.init.xavier_uniform_(self.emb1.weight)
            nn.init.xavier_uniform_(self.emb2.weight)

    def forward(self, precomputed_adj: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (edge_index, edge_weight) for PyG."""
        if self.graph_type == "correlation" and precomputed_adj is not None:
            adj = precomputed_adj
        else:
            idx = torch.arange(self.num_vars, device=self.emb1.weight.device)
            e1 = self.emb1(idx)  # (N, d)
            e2 = self.emb2(idx)  # (N, d)
            adj = torch.softmax(F.relu(e1 @ e2.T), dim=-1)  # (N, N)

        edge_index, edge_weight = dense_to_sparse(adj)
        return edge_index, edge_weight

    def get_adjacency(self) -> torch.Tensor:
        with torch.no_grad():
            idx = torch.arange(self.num_vars, device=self.emb1.weight.device)
            e1 = self.emb1(idx)
            e2 = self.emb2(idx)
            return torch.softmax(F.relu(e1 @ e2.T), dim=-1)


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
                        dropout=dropout, concat=True)
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
                batch_h = gat(batch_h, batch_edge_index, batch_edge_weight)
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
        self.query = nn.Parameter(torch.randn(hidden_dim))
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
                 graph_type: str = "learned"):
        super().__init__()
        self.num_vars = num_vars
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.horizons = horizons

        # Per-band graph constructors
        self.graph_constructors = nn.ModuleList([
            FrequencyGraphConstructor(num_vars, embed_dim=16, graph_type=graph_type)
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
