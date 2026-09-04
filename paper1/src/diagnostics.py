"""Reusable graph-collapse diagnostics.

Extracted and generalized from the methodology already used to diagnose and
test-fix the "Adjacency Collapse" bug (see STATUS.md "Adjacency Collapse
Diagnosed", 2026-08-09, and the graph-fix experiment notebook,
`notebooks/archive_paper1_vmd_mfgnn/vmd_mfgnn_v2_colab_graphfix.ipynb`, STEP 5
diagnostic cell). This module exists so the K-sweep (item 1), decomposition
comparison (item 2), and loss-function comparison (item 3) experiments can
all run the SAME diagnostic instead of three divergent reimplementations,
directly answering: does the collapse finding depend on K / decomposition
method / loss function, or is it invariant?

Three complementary checks, matching the task brief's item 1 exactly:
  1. `adjacency_state_diagnostic` -- per-band RMS embedding magnitude vs.
     Xavier-init reference, softmax spread, and fraction of kept edges
     within a tight tolerance of exactly 1/N (the original diagnosis's
     smoking-gun signature of collapse). Operates on a trained (or freshly
     initialized) model snapshot -- no gradients needed.
  2. `gradient_magnitude_diagnostic` -- runs a handful of real
     forward/backward passes on live data and records the actual gradient
     norm reaching emb1/emb2 (and, for comparison, the rest of the model),
     answering "do these parameters receive non-trivial gradient at all"
     directly, rather than only inferring it from where the weights ended up.
  3. `adjacency_movement_from_init` -- compares a trained model's learned
     adjacency against a freshly-initialized model at the SAME seed/config,
     via both L2 distance and cosine similarity of the raw (pre-softmax)
     embedding directions -- this is the "did the embedding direction ever
     rotate, or did it only shrink in place" check from the original
     diagnosis (which found cosine similarity 0.993-0.996 between two
     independently trained runs, i.e. no rotation at all).
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Reference values from STATUS.md's "Adjacency Collapse Diagnosed" finding,
# measured on the original (K=5, VMD, MSE-loss) unfixed checkpoints. Kept
# here as fixed constants so every later run (different K/decomposition/loss)
# is compared against the SAME original yardstick, not a shifting baseline.
XAVIER_INIT_RMS_REFERENCE = 0.2885
COLLAPSED_RMS_REFERENCE = 0.0089
UNIFORM_TOL = 1e-3


def adjacency_state_diagnostic(model, num_vars: int) -> Dict:
    """Per-band collapse diagnostic on a model's CURRENT weights (no
    gradients touched). Works for both VMDMFGNN (model.graph_constructors,
    a ModuleList) and PooledGraphMFGNN (model.graph_constructor, a single
    module) -- normalizes both into a list internally.

    Returns a dict: {"bands": [per-band dict, ...], "n_ok": int,
    "n_total": int, "verdict": str}. Each per-band dict has keys
    emb1_rms/emb2_rms/rms_ratio/softmax_std/frac_near_uniform/collapsed.
    """
    if hasattr(model, "graph_constructors"):
        constructors = list(model.graph_constructors)
    elif hasattr(model, "graph_constructor"):
        constructors = [model.graph_constructor]
    else:
        raise AttributeError(
            "Model has neither graph_constructors (VMDMFGNN) nor "
            "graph_constructor (PooledGraphMFGNN) -- cannot run the "
            "adjacency diagnostic on this model type."
        )

    N = num_vars
    bands = []
    for k, gc in enumerate(constructors):
        if gc.graph_type != "learned":
            continue
        e1 = gc.emb1.weight.detach()
        e2 = gc.emb2.weight.detach()
        emb1_rms = e1.pow(2).mean().sqrt().item()
        emb2_rms = e2.pow(2).mean().sqrt().item()
        rms_avg = (emb1_rms + emb2_rms) / 2
        rms_ratio = rms_avg / XAVIER_INIT_RMS_REFERENCE

        e1n, e2n = e1, e2
        if getattr(gc, "normalize_embeddings", False):
            e1n, e2n = F.normalize(e1, dim=-1), F.normalize(e2, dim=-1)
        full_adj = torch.softmax(F.relu(e1n @ e2n.T), dim=-1)
        softmax_std = full_adj.std().item()

        kept_adj = gc.get_adjacency()
        nonzero = kept_adj[kept_adj > 0]
        frac_near_uniform = (
            nonzero.sub(1.0 / N).abs() < UNIFORM_TOL
        ).float().mean().item() if len(nonzero) else float("nan")

        collapsed = (rms_ratio < 0.5) or (frac_near_uniform > 0.8)
        bands.append({
            "band": k, "emb1_rms": emb1_rms, "emb2_rms": emb2_rms,
            "rms_ratio": rms_ratio, "softmax_std": softmax_std,
            "frac_near_uniform": frac_near_uniform, "collapsed": collapsed,
        })

    n_total = len(bands)
    n_ok = sum(1 for b in bands if not b["collapsed"])
    if n_total == 0:
        verdict = "N/A (no learned-graph-type bands found)"
    elif n_ok == n_total:
        verdict = "NOT COLLAPSED -- no band shows the diagnosed collapse-to-uniform pattern"
    elif n_ok == 0:
        verdict = "COLLAPSED -- all bands show the diagnosed pattern"
    else:
        verdict = f"PARTIAL -- {n_ok}/{n_total} bands avoided collapse"

    return {"bands": bands, "n_ok": n_ok, "n_total": n_total, "verdict": verdict}


def gradient_magnitude_diagnostic(model, loader, device, loss_fn=F.mse_loss,
                                   num_batches: int = 5) -> Dict:
    """Runs `num_batches` real forward/backward passes (using the SAME
    trainer-style summed-over-horizons loss) and records the actual gradient
    norm reaching emb1/emb2 vs. the rest of the model's trainable parameters.

    This directly answers item 1's "checking whether emb1/emb2 ... parameters
    receive non-trivial gradient" -- rather than only inferring it after the
    fact from where the weights ended up (which is what
    `adjacency_state_diagnostic` / `adjacency_movement_from_init` do).

    Returns: {"emb_grad_norm_mean": float, "emb_grad_norm_per_batch": [...],
              "other_grad_norm_mean": float, "n_batches_used": int}.
    Model is left in eval() mode with gradients zeroed after this call (does
    not corrupt a training run if called mid-training on the live model --
    though the caller should typically use a fresh checkpoint copy instead).
    """
    model.train()
    emb_norms, other_norms = [], []
    n_used = 0
    for x, y in loader:
        if n_used >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        preds = model(x)
        loss = sum(loss_fn(preds[str(h)], y[:, i])
                   for i, h in enumerate(model.horizons))
        loss.backward()

        emb_sq, other_sq = 0.0, 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g_sq = p.grad.detach().pow(2).sum().item()
            if ".emb1." in name or ".emb2." in name:
                emb_sq += g_sq
            else:
                other_sq += g_sq
        emb_norms.append(emb_sq ** 0.5)
        other_norms.append(other_sq ** 0.5)
        n_used += 1

    model.zero_grad(set_to_none=True)
    model.eval()

    if n_used == 0:
        return {"emb_grad_norm_mean": float("nan"),
                "emb_grad_norm_per_batch": [], "other_grad_norm_mean": float("nan"),
                "n_batches_used": 0}

    return {
        "emb_grad_norm_mean": sum(emb_norms) / n_used,
        "emb_grad_norm_per_batch": emb_norms,
        "other_grad_norm_mean": sum(other_norms) / n_used,
        "n_batches_used": n_used,
    }


def adjacency_movement_from_init(trained_model, fresh_model) -> Dict:
    """Compares `trained_model`'s learned per-band embeddings against
    `fresh_model` (same architecture/config, freshly initialized, e.g. same
    seed -- NOT trained). Reports, per band:
      - l2_distance: ||E_trained - E_init||_2 (raw embedding, pre-softmax)
      - cosine_similarity: how much the embedding DIRECTION rotated (the
        original diagnosis's key finding was cosine ~0.993-0.996 between
        independently trained runs, i.e. essentially zero rotation -- values
        near 1.0 here reproduce that "shrank in place, never rotated"
        pattern; values well below 1.0 indicate genuine directional
        learning).
      - rms_ratio_to_init: trained embedding's RMS magnitude / init's RMS
        magnitude (near 0 = isotropic collapse toward the origin).

    Both models must have the same graph-embedding structure (same
    num_vars/embed_dim/num_modes) -- shapes are checked and a ValueError is
    raised on mismatch to avoid a silently-meaningless comparison.
    """
    def get_constructors(m):
        if hasattr(m, "graph_constructors"):
            return list(m.graph_constructors)
        if hasattr(m, "graph_constructor"):
            return [m.graph_constructor]
        raise AttributeError("Model has no graph_constructor(s) attribute.")

    trained_gcs = get_constructors(trained_model)
    fresh_gcs = get_constructors(fresh_model)
    if len(trained_gcs) != len(fresh_gcs):
        raise ValueError(
            f"Band-count mismatch: trained model has {len(trained_gcs)} "
            f"graph constructors, fresh model has {len(fresh_gcs)}."
        )

    bands = []
    for k, (tgc, fgc) in enumerate(zip(trained_gcs, fresh_gcs)):
        if tgc.graph_type != "learned" or fgc.graph_type != "learned":
            continue
        e1_t = tgc.emb1.weight.detach().flatten()
        e2_t = tgc.emb2.weight.detach().flatten()
        e1_f = fgc.emb1.weight.detach().flatten()
        e2_f = fgc.emb2.weight.detach().flatten()
        if e1_t.shape != e1_f.shape:
            raise ValueError(
                f"emb1 shape mismatch at band {k}: trained {tuple(e1_t.shape)} "
                f"vs fresh {tuple(e1_f.shape)} -- not a valid init-vs-trained comparison."
            )

        e_t = torch.cat([e1_t, e2_t])
        e_f = torch.cat([e1_f, e2_f])
        l2_distance = (e_t - e_f).norm().item()
        cosine_similarity = F.cosine_similarity(e_t.unsqueeze(0), e_f.unsqueeze(0)).item()
        rms_init = e_f.pow(2).mean().sqrt().item()
        rms_trained = e_t.pow(2).mean().sqrt().item()
        rms_ratio_to_init = rms_trained / rms_init if rms_init > 0 else float("nan")

        bands.append({
            "band": k, "l2_distance": l2_distance,
            "cosine_similarity": cosine_similarity,
            "rms_ratio_to_init": rms_ratio_to_init,
        })

    if bands:
        mean_cos = sum(b["cosine_similarity"] for b in bands) / len(bands)
    else:
        mean_cos = float("nan")

    return {
        "bands": bands,
        "mean_cosine_similarity": mean_cos,
        "note": ("cosine_similarity near 1.0 across bands reproduces the "
                 "original diagnosis's 'shrank in place, never rotated' "
                 "pattern (measured 0.993-0.996 between independent runs); "
                 "well below 1.0 indicates genuine directional learning."),
    }


def run_full_diagnostic(trained_model, fresh_model, loader, device,
                         num_vars: int, loss_fn=F.mse_loss,
                         num_grad_batches: int = 5) -> Dict:
    """Convenience wrapper running all three checks and packaging them for
    a per-(K/method/loss/seed) results row. This is the single function each
    experiment driver in `experiments.py` calls so every experiment uses
    an identical diagnostic methodology.
    """
    return {
        "adjacency_state": adjacency_state_diagnostic(trained_model, num_vars),
        "gradient_magnitude": gradient_magnitude_diagnostic(
            trained_model, loader, device, loss_fn=loss_fn, num_batches=num_grad_batches),
        "adjacency_movement": adjacency_movement_from_init(trained_model, fresh_model),
    }
