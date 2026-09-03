"""PooledGraphMFGNN: ablation variant of VMD-MFGNN that pools the K VMD
frequency bands into a single unified per-variable representation and
builds ONE graph over it, instead of K separate per-band graphs.

This is the "Per-Band Graphs vs. Pooled Single Graph" ablation
(highest-priority ablation per vmd-mfgnn-protocol/SKILL.md) — it is the
empirical backbone of the paper's novelty claim vs. prior art (MBTI-Net).

Design choice: to isolate the "K graphs vs. 1 graph" variable and avoid
confounding the comparison with unrelated architecture changes, this file
reuses the exact same `FrequencyGraphConstructor` (learned softmax
adjacency) and `FrequencyBandModule` (GAT + LSTM) classes from
`vmd_mfgnn.py` unmodified. The only new component is a small learnable
pooling projection (`nn.Linear(num_modes, 1)`) that concatenates the K
band values for each (timestep, variable) pair and projects them down to
the single scalar signal that `FrequencyBandModule` already expects
(input_dim=1) — mirroring how the per-band model feeds one scalar per
variable per band into the same module. Everything downstream (GAT
layers, LSTM, prediction heads) is structurally identical to
`VMDMFGNN`, so a parameter-count-aware comparison (see
`count_parameters`) can attribute any accuracy delta to the pooled-vs
per-band graph design rather than incidental size differences.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

try:
    from .vmd_mfgnn import FrequencyGraphConstructor, FrequencyBandModule
except ImportError:  # pragma: no cover - fallback for direct script execution
    from vmd_mfgnn import FrequencyGraphConstructor, FrequencyBandModule


class PooledGraphMFGNN(nn.Module):
    """VMD-MFGNN variant with a single pooled graph instead of K per-band graphs."""

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
        self.graph_type = graph_type

        # Pools the K band values for each (timestep, variable) into one
        # scalar representation. This is the "pool K band representations
        # into ONE representation" step, done BEFORE any graph is built.
        self.band_pool = nn.Linear(num_modes, 1)

        # Single graph constructor (built once over the pooled representation,
        # not once per band). Reuses the same learned softmax adjacency
        # mechanism as the per-band model's graph_type="learned" path.
        self.graph_constructor = FrequencyGraphConstructor(
            num_vars, embed_dim=16, graph_type=graph_type,
            normalize_embeddings=normalize_graph_embeddings)

        # Single GAT+LSTM processing module (structurally identical to one
        # of VMDMFGNN's per-band FrequencyBandModule instances).
        self.band_module = FrequencyBandModule(
            num_vars, 1, hidden_dim, num_heads,
            num_gnn_layers, temporal_layers, dropout)

        # Prediction heads (one per horizon) — identical structure to VMDMFGNN.
        self.pred_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for h in horizons
        })

    def forward(self, x: torch.Tensor,
                precomputed_adj: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        x: (batch, lookback, K, num_vars) — VMD mode features, same input
           shape as VMDMFGNN.forward.
        precomputed_adj: optional single (num_vars, num_vars) adjacency,
           used when graph_type=="correlation" (analogous to VMDMFGNN's
           precomputed_adjs, but only one matrix since there is only one
           graph here).
        Returns: dict with predictions per horizon.
        """
        B, T, K, N = x.shape
        assert K == self.num_modes, f"expected {self.num_modes} VMD modes, got {K}"

        # Pool the K bands per (batch, timestep, variable): (B, T, K, N) ->
        # (B, T, N, K) -> Linear(K, 1) -> (B, T, N)
        pooled = x.permute(0, 1, 3, 2)  # (B, T, N, K)
        pooled = self.band_pool(pooled).squeeze(-1)  # (B, T, N)

        # Build ONE graph over the pooled representation.
        edge_index, edge_weight = self.graph_constructor(precomputed_adj)

        # Single GAT + LSTM pass (no K-fold repetition, no cross-band fusion needed).
        out = self.band_module(pooled, edge_index, edge_weight)  # (B, N, hidden_dim)
        copper_repr = out[:, 0, :]  # (B, hidden_dim)

        predictions = {}
        for h in self.horizons:
            predictions[str(h)] = self.pred_heads[str(h)](copper_repr).squeeze(-1)

        return predictions

    def get_learned_graph(self) -> Optional[torch.Tensor]:
        if self.graph_constructor.graph_type == "learned":
            return self.graph_constructor.get_adjacency()
        return None


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in `model`.

    Used to confirm that any accuracy delta between VMDMFGNN (per-band
    graphs) and PooledGraphMFGNN (single pooled graph) reflects the graph
    design choice and not an incidental difference in model capacity.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
