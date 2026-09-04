"""Drivers for the 5 robustness/gap-closing experiments (see task brief):
  1. VMD band-count (K) sweep
  2. Alternative decomposition method comparison (VMD vs EMD vs CEEMDAN)
  3. Alternative loss function comparison (MSE vs MAE vs Huber)
  4. Multi-seed robustness check (full_model / pooled_graph_matched_dim, 5 seeds)
  5. Wider HPO pass (thin wrapper around hpo.run_hpo/build_data_for_k_values)

Design: every experiment (1-4) is a resumable grid over "cells" (e.g. one K
value, one decomposition method, one loss function, one (variant, seed)
pair). Each cell's result is appended as one line to a `.jsonl` file
IMMEDIATELY after it completes (mirroring the pattern proven in
paper2/notebooks/cubench_colab.ipynb's grid_results.jsonl, and the
resumability fix in trainer.py's fit()/checkpoint_path machinery). On
restart, a cell already present in the jsonl AND whose checkpoint file
genuinely exists on disk is skipped -- requiring the actual artifact file
(not just a log line) matches this project's documented fix for the
equivalent bug in CuBench (see git log "Fix resume logic to require actual
prediction files, not just log lines").

Every cell also runs the SAME gradient/adjacency diagnostic
(`diagnostics.run_full_diagnostic`), so all 4 experiments answer the same
question with the same methodology: does the graph-collapse finding depend
on this axis, or is it invariant?

This module intentionally contains real, importable, syntactically-checked
logic (not notebook-only cells) so it can be unit-exercised locally at tiny
scale (see tests/test_experiments_resume.py) even without a GPU or
torch_geometric installed -- the resumability mechanics are tested
independently of the (GPU-only) model training itself.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _checkpoint_is_complete(checkpoint_path) -> bool:
    """True only if `checkpoint_path` exists AND its saved dict has
    completed=True (see trainer.py's fit() docstring: a checkpoint can exist
    on disk mid-training, with completed=False, if a crash happened after a
    best-val-loss save but before the loop actually finished). Used by every
    experiment's artifact_check_fn below instead of a bare existence check,
    so a partially-synced/interrupted checkpoint is correctly treated as
    incomplete on resume, not mistaken for a finished run.
    """
    import torch
    p = Path(checkpoint_path)
    if not p.exists():
        return False
    try:
        loaded = torch.load(p, map_location="cpu")
    except Exception:
        return False
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return bool(loaded.get("completed", False))
    return True  # legacy bare-state-dict checkpoint, treated as complete (matches trainer.py)


# ---------------------------------------------------------------------------
# Generic resumable-grid machinery (no model/torch dependency -- reusable and
# independently testable).
# ---------------------------------------------------------------------------

def load_completed_cells(jsonl_path: str, artifact_check_fn=None) -> Dict[Tuple, dict]:
    """Reads `jsonl_path` (one JSON object per line, each with a "cell_key"
    field -- a list, since JSON has no tuple type) and returns {tuple(key):
    row} for every row that is genuinely complete.

    A row counts as complete only if:
      (a) it parses as valid JSON with a "cell_key" field, AND
      (b) `artifact_check_fn(row)` returns True, if `artifact_check_fn` is
          given (e.g. "does this row's checkpoint file still exist on
          disk?") -- this is the "require actual files, not just log lines"
          fix mirrored from CuBench.
    Malformed lines (e.g. a truncated final line from a killed process) are
    skipped with a warning, not raised.
    """
    path = Path(jsonl_path)
    completed = {}
    if not path.exists():
        return completed
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{jsonl_path}:{line_no}: malformed JSON line, skipping")
                continue
            if "cell_key" not in row:
                continue
            if artifact_check_fn is not None and not artifact_check_fn(row):
                logger.warning(
                    f"{jsonl_path}:{line_no}: cell_key={row['cell_key']} has a "
                    f"jsonl row but its required artifact is missing on disk -- "
                    f"treating as INCOMPLETE and will re-run."
                )
                continue
            completed[tuple(row["cell_key"])] = row
    return completed


def append_cell_result(jsonl_path: str, cell_key: list, result: dict) -> None:
    """Appends one cell's result as a single JSON line, flushed immediately
    (so a kill -9 right after this call still leaves the line durable).
    """
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"cell_key": cell_key, **result}

    def convert(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON-serializable: {type(obj)}")

    # Defensive: if a prior process crashed mid-write and left the file
    # without a trailing newline (a truncated final line, no \n), a plain
    # "a" append here would silently concatenate this new row onto the end
    # of that broken line instead of starting a fresh one -- permanently
    # corrupting both. Guard against it by ensuring the file ends with a
    # newline before appending.
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_byte = f.read(1)
        if last_byte != b"\n":
            with open(path, "a") as f:
                f.write("\n")

    with open(path, "a") as f:
        f.write(json.dumps(row, default=convert) + "\n")
        f.flush()


def run_resumable_grid(cells: List[dict], key_fn, run_fn, jsonl_path: str,
                        artifact_check_fn=None) -> List[dict]:
    """Drives a resumable grid: for each cell in `cells`, computes its key
    via `key_fn(cell)`, skips it if already completed (per
    `load_completed_cells`), otherwise runs `run_fn(cell)` -> dict and
    appends the result immediately. Returns the full list of results (both
    freshly-run and previously-completed) in `cells` order.
    """
    completed = load_completed_cells(jsonl_path, artifact_check_fn=artifact_check_fn)
    results = []
    for cell in cells:
        key = list(key_fn(cell))
        key_t = tuple(key)
        if key_t in completed:
            logger.info(f"[resume] cell {key} already completed, skipping")
            results.append(completed[key_t])
            continue
        logger.info(f"Running cell {key}...")
        result = run_fn(cell)
        append_cell_result(jsonl_path, key, result)
        results.append({"cell_key": key, **result})
    return results


# ---------------------------------------------------------------------------
# Shared helpers for the model-training experiments (1-4)
# ---------------------------------------------------------------------------

def _load_tuned_hyperparams(hpo_best_params_path: str, fallback_model_cfg: dict,
                             fallback_lr: float) -> dict:
    """Loads the already-tuned hyperparameters (item 1-3 use these, NOT a
    fresh HPO run each -- see task brief: "using the already-tuned other
    hyperparameters"). Falls back to configs/default.yaml's untuned model
    settings (with a loud warning) if the expected file is absent, so a
    local dry run without results/archive_paper1/hpo_best_params.json
    doesn't hard-crash.
    """
    p = Path(hpo_best_params_path)
    if p.exists():
        with open(p) as f:
            best = json.load(f)
        logger.info(f"Loaded tuned hyperparameters from {p}: {best}")
        return best
    logger.warning(
        f"{p} not found -- falling back to configs/default.yaml's UNTUNED "
        f"model settings. Results from this run will not be directly "
        f"comparable to the tuned main-pipeline numbers; re-run with the "
        f"real hpo_best_params.json present for a genuine comparison."
    )
    return {
        "hidden_dim": fallback_model_cfg["hidden_dim"],
        "num_heads": fallback_model_cfg["num_heads"],
        "dropout": fallback_model_cfg["dropout"],
        "num_gnn_layers": fallback_model_cfg["num_gnn_layers"],
        "learning_rate": fallback_lr,
    }


def _build_tuned_config(base_config: dict, best_params: dict) -> dict:
    cfg = copy.deepcopy(base_config)
    mc = cfg["model"]
    mc["hidden_dim"] = best_params["hidden_dim"]
    mc["num_heads"] = best_params["num_heads"]
    mc["dropout"] = best_params["dropout"]
    mc["num_gnn_layers"] = best_params["num_gnn_layers"]
    cfg["training"]["learning_rate"] = best_params["learning_rate"]
    return cfg


def _train_and_diagnose(model_ctor, fresh_model_ctor, config: dict, data: Dict,
                         checkpoint_path: str, seed: int,
                         pred_dir: Optional[str] = None, pred_prefix: str = "model"):
    """Shared train-one-model-then-diagnose routine used by every experiment
    below. model_ctor()/fresh_model_ctor() are zero-arg callables returning a
    freshly-constructed (untrained) model -- fresh_model_ctor is called
    AFTER training (with the same seed re-applied) purely to get an
    init-only reference for `adjacency_movement_from_init`, never trained.

    pred_dir: if given, saves raw test-set prediction + ground-truth .npy
    arrays (one pair per horizon, named f"{pred_prefix}_{h}.npy" /
    f"{pred_prefix}_{h}_true.npy") to this directory. This matters because
    this project's standing practice is to re-verify every reported number
    from raw .npy files, not trust aggregated JSON/jsonl metrics alone (see
    the CuBench predictions incident, which this mirrors the fix for) --
    without this, the jsonl's test_metrics would be the ONLY record of each
    cell's result, with no way to independently recompute or audit it later.

    Returns (test_metrics: dict, diagnostic: dict, trainer).
    """
    import numpy as np
    from .trainer import VMDMFGNNTrainer
    from .utils import set_seed, get_device
    from .diagnostics import run_full_diagnostic

    set_seed(seed)
    model = model_ctor()
    trainer = VMDMFGNNTrainer(model, config)
    trainer.fit(data["train_loader"], data["val_loader"],
                checkpoint_path=Path(checkpoint_path))
    test_metrics = trainer.evaluate(data["test_loader"])

    if pred_dir is not None:
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        preds = trainer.predict(data["test_loader"])
        test_ys = [y.numpy() if hasattr(y, "numpy") else y for _, y in data["test_loader"]]
        test_true = np.concatenate(test_ys, axis=0)
        for i, h in enumerate(data["horizons"]):
            np.save(Path(pred_dir) / f"{pred_prefix}_{h}.npy", preds[str(h)])
            np.save(Path(pred_dir) / f"{pred_prefix}_{h}_true.npy", test_true[:, i])

    set_seed(seed)  # same seed -> same init as the model that was just trained
    fresh_model = fresh_model_ctor()
    device = get_device()
    fresh_model.to(device)

    diag = run_full_diagnostic(
        trainer.model, fresh_model, data["val_loader"], device,
        num_vars=data["num_vars"], loss_fn=trainer._loss_fn,
    )
    return test_metrics, diag, trainer


# ---------------------------------------------------------------------------
# Item 1: VMD band-count (K) sweep
# ---------------------------------------------------------------------------

def run_k_sweep(base_config: dict, k_values: List[int] = (3, 5, 7, 9),
                 hpo_best_params_path: str = "results/archive_paper1/hpo_best_params.json",
                 results_dir: str = "results/robustness/k_sweep") -> List[dict]:
    """Retrains full VMD-MFGNN at each K in `k_values`, using the existing
    tuned hyperparameters (item 1). Each K gets its own VMD-modes cache
    (data_pipeline.create_datasets(modes_cache_path=...)) so K values never
    invalidate each other's decomposition cache, and its own checkpoint, so
    a Colab disconnect resumes at the next incomplete K rather than
    retraining completed ones.
    """
    from .data_pipeline import create_datasets
    from .models.vmd_mfgnn import VMDMFGNN

    best_params = _load_tuned_hyperparams(
        hpo_best_params_path, base_config["model"], base_config["training"]["learning_rate"])
    seed = base_config["training"]["seed"]
    jsonl_path = f"{results_dir}/k_sweep_results.jsonl"

    def artifact_check(row):
        return _checkpoint_is_complete(row.get("checkpoint_path", ""))

    def run_cell(cell: dict) -> dict:
        k = cell["K"]
        k_config = copy.deepcopy(base_config)
        k_config["vmd"]["K"] = k
        tuned_config = _build_tuned_config(k_config, best_params)

        # If this K matches the main pipeline's already-configured K, point
        # at its EXISTING cache (data/vmd_modes.npy) instead of a fresh
        # per-K path -- lets this sweep reuse the main pipeline's already-
        # computed decomposition instead of needlessly recomputing an
        # identical K from scratch (this is the case the Section 3 wall-
        # clock estimate assumes: "only 3 NEW decompositions needed").
        if k == base_config["vmd"]["K"]:
            cache_path = "data/vmd_modes.npy"
        else:
            cache_path = f"data/vmd_modes_K{k}.npy"
        data = create_datasets(tuned_config, modes_cache_path=cache_path)
        ckpt_path = f"{results_dir}/checkpoints/full_model_K{k}.pt"

        def ctor():
            return VMDMFGNN(
                num_vars=data["num_vars"], num_modes=data["num_modes"],
                hidden_dim=tuned_config["model"]["hidden_dim"],
                num_heads=tuned_config["model"]["num_heads"],
                num_gnn_layers=tuned_config["model"]["num_gnn_layers"],
                temporal_layers=tuned_config["model"]["temporal_layers"],
                dropout=tuned_config["model"]["dropout"],
                horizons=data["horizons"], graph_type="learned",
            )

        test_metrics, diag, _ = _train_and_diagnose(
            ctor, ctor, tuned_config, data, ckpt_path, seed,
            pred_dir=f"{results_dir}/predictions", pred_prefix=f"full_model_K{k}")
        return {"K": k, "test_metrics": test_metrics, "diagnostic": diag,
                "checkpoint_path": ckpt_path}

    cells = [{"K": k} for k in k_values]
    return run_resumable_grid(cells, key_fn=lambda c: (c["K"],),
                               run_fn=run_cell, jsonl_path=jsonl_path,
                               artifact_check_fn=artifact_check)


# ---------------------------------------------------------------------------
# Item 2: Alternative decomposition method comparison
# ---------------------------------------------------------------------------

def run_decomposition_comparison(base_config: dict,
                                  methods: List[str] = ("vmd", "emd"),
                                  hpo_best_params_path: str = "results/archive_paper1/hpo_best_params.json",
                                  results_dir: str = "results/robustness/decomposition") -> List[dict]:
    """Retrains full model with each decomposition `method` (default: VMD
    vs EMD; pass methods=("vmd","emd","ceemdan") to include the optional,
    compute-expensive CEEMDAN variant -- see CEEMDANDecomposer's docstring
    for the disclosed cost tradeoff). Uses the SAME K as the main pipeline's
    config (`base_config["vmd"]["K"]`) for every method, per the task brief's
    "same K if EMD naturally produces a similar number of components"
    instruction -- EMDDecomposer's docstring documents exactly how K
    interacts with EMD's data-driven natural IMF count (truncate-and-sum if
    EMD produces more, zero-pad if fewer).
    """
    from .data_pipeline import create_datasets
    from .models.vmd_mfgnn import VMDMFGNN

    best_params = _load_tuned_hyperparams(
        hpo_best_params_path, base_config["model"], base_config["training"]["learning_rate"])
    seed = base_config["training"]["seed"]
    jsonl_path = f"{results_dir}/decomposition_results.jsonl"
    K = base_config["vmd"]["K"]

    def artifact_check(row):
        return _checkpoint_is_complete(row.get("checkpoint_path", ""))

    def run_cell(cell: dict) -> dict:
        method = cell["method"]
        tuned_config = _build_tuned_config(base_config, best_params)
        data = create_datasets(tuned_config, decomposition_method=method,
                                modes_cache_path=f"data/{method}_modes_K{K}.npy")
        ckpt_path = f"{results_dir}/checkpoints/full_model_{method}.pt"

        def ctor():
            return VMDMFGNN(
                num_vars=data["num_vars"], num_modes=data["num_modes"],
                hidden_dim=tuned_config["model"]["hidden_dim"],
                num_heads=tuned_config["model"]["num_heads"],
                num_gnn_layers=tuned_config["model"]["num_gnn_layers"],
                temporal_layers=tuned_config["model"]["temporal_layers"],
                dropout=tuned_config["model"]["dropout"],
                horizons=data["horizons"], graph_type="learned",
            )

        test_metrics, diag, _ = _train_and_diagnose(
            ctor, ctor, tuned_config, data, ckpt_path, seed,
            pred_dir=f"{results_dir}/predictions", pred_prefix=f"full_model_{method}")
        return {"method": method, "effective_K": data["num_modes"],
                "test_metrics": test_metrics, "diagnostic": diag,
                "checkpoint_path": ckpt_path}

    cells = [{"method": m} for m in methods]
    return run_resumable_grid(cells, key_fn=lambda c: (c["method"],),
                               run_fn=run_cell, jsonl_path=jsonl_path,
                               artifact_check_fn=artifact_check)


# ---------------------------------------------------------------------------
# Item 3: Alternative loss function comparison
# ---------------------------------------------------------------------------

def run_loss_comparison(base_config: dict,
                         loss_fns: List[str] = ("mse", "mae", "huber"),
                         hpo_best_params_path: str = "results/archive_paper1/hpo_best_params.json",
                         results_dir: str = "results/robustness/loss_comparison") -> List[dict]:
    """Retrains full model under each `training.loss_fn` choice. Reuses ONE
    data build (decomposition doesn't depend on the loss function) across all
    loss choices -- only the trainer's loss changes.
    """
    from .data_pipeline import create_datasets
    from .models.vmd_mfgnn import VMDMFGNN

    best_params = _load_tuned_hyperparams(
        hpo_best_params_path, base_config["model"], base_config["training"]["learning_rate"])
    seed = base_config["training"]["seed"]
    jsonl_path = f"{results_dir}/loss_comparison_results.jsonl"

    tuned_config_base = _build_tuned_config(base_config, best_params)
    data = create_datasets(tuned_config_base)

    def artifact_check(row):
        return _checkpoint_is_complete(row.get("checkpoint_path", ""))

    def run_cell(cell: dict) -> dict:
        loss_fn = cell["loss_fn"]
        cfg = copy.deepcopy(tuned_config_base)
        cfg["training"]["loss_fn"] = loss_fn
        ckpt_path = f"{results_dir}/checkpoints/full_model_{loss_fn}.pt"

        def ctor():
            return VMDMFGNN(
                num_vars=data["num_vars"], num_modes=data["num_modes"],
                hidden_dim=cfg["model"]["hidden_dim"],
                num_heads=cfg["model"]["num_heads"],
                num_gnn_layers=cfg["model"]["num_gnn_layers"],
                temporal_layers=cfg["model"]["temporal_layers"],
                dropout=cfg["model"]["dropout"],
                horizons=data["horizons"], graph_type="learned",
            )

        test_metrics, diag, _ = _train_and_diagnose(
            ctor, ctor, cfg, data, ckpt_path, seed,
            pred_dir=f"{results_dir}/predictions", pred_prefix=f"full_model_{loss_fn}")
        return {"loss_fn": loss_fn, "test_metrics": test_metrics,
                "diagnostic": diag, "checkpoint_path": ckpt_path}

    cells = [{"loss_fn": lf} for lf in loss_fns]
    return run_resumable_grid(cells, key_fn=lambda c: (c["loss_fn"],),
                               run_fn=run_cell, jsonl_path=jsonl_path,
                               artifact_check_fn=artifact_check)


# ---------------------------------------------------------------------------
# Item 4: Multi-seed robustness check
# ---------------------------------------------------------------------------

def run_multiseed(base_config: dict,
                   seeds: List[int] = (42, 43, 44, 45, 46),
                   variants: List[str] = ("full_model", "pooled_graph_matched_dim"),
                   hpo_best_params_path: str = "results/archive_paper1/hpo_best_params.json",
                   results_dir: str = "results/robustness/multiseed") -> Dict:
    """Re-runs ONLY full_model and pooled_graph_matched_dim (the two ablation
    rows carrying RQ1's central claim, per STATUS.md's already-recommended
    "middle path") across `seeds`, reporting mean +/- std per horizon plus a
    paired significance test (item 4). Uses the exact same model-construction
    logic as trainer.run_ablation_studies' full_model/pooled_graph_matched_dim
    variants (make_vmd_mfgnn/make_pooled), just parameterized over seed.

    Returns {"cells": [...], "aggregate": {variant: {h: {"mean":...,
    "std":...}}}, "significance": {h: paired_seed_significance_test(...)}}.

    DISCLOSURE (read before comparing against results/ablation_results.json):
    this function trains both variants at the TUNED hyperparameters
    (`_build_tuned_config`, same `hpo_best_params.json` as items 1-3), NOT
    the untuned hidden_dim=64 config that `trainer.run_ablation_studies`
    used for the paper's existing ablation table. This is a deliberate
    choice (arguably the more informative comparison, and consistent with
    the paper's already-published tuned confound-check in Section 5.3.1),
    but it means these seeds' full_model/pooled_graph_matched_dim numbers
    will NOT line up directly against results/ablation_results.json's rows
    -- report/paper text using this section's output must say so explicitly
    rather than silently cross-referencing the two tables as if comparable.
    """
    from .data_pipeline import create_datasets
    from .models.vmd_mfgnn import VMDMFGNN
    from .models.pooled_graph_mfgnn import PooledGraphMFGNN
    from .utils import paired_seed_significance_test

    best_params = _load_tuned_hyperparams(
        hpo_best_params_path, base_config["model"], base_config["training"]["learning_rate"])
    tuned_config = _build_tuned_config(base_config, best_params)
    jsonl_path = f"{results_dir}/multiseed_results.jsonl"
    data = create_datasets(tuned_config)
    mc = tuned_config["model"]

    def make_full_model():
        return VMDMFGNN(
            num_vars=data["num_vars"], num_modes=data["num_modes"],
            hidden_dim=mc["hidden_dim"], num_heads=mc["num_heads"],
            num_gnn_layers=mc["num_gnn_layers"], temporal_layers=mc["temporal_layers"],
            dropout=mc["dropout"], horizons=data["horizons"], graph_type="learned",
        )

    def make_pooled_matched_dim():
        return PooledGraphMFGNN(
            num_vars=data["num_vars"], num_modes=data["num_modes"],
            hidden_dim=mc["hidden_dim"], num_heads=mc["num_heads"],
            num_gnn_layers=mc["num_gnn_layers"], temporal_layers=mc["temporal_layers"],
            dropout=mc["dropout"], horizons=data["horizons"], graph_type="learned",
        )

    ctors = {"full_model": make_full_model, "pooled_graph_matched_dim": make_pooled_matched_dim}

    def artifact_check(row):
        return _checkpoint_is_complete(row.get("checkpoint_path", ""))

    def run_cell(cell: dict) -> dict:
        variant, seed = cell["variant"], cell["seed"]
        ckpt_path = f"{results_dir}/checkpoints/{variant}_seed{seed}.pt"
        ctor = ctors[variant]
        test_metrics, diag, _ = _train_and_diagnose(
            ctor, ctor, tuned_config, data, ckpt_path, seed,
            pred_dir=f"{results_dir}/predictions", pred_prefix=f"{variant}_seed{seed}")
        return {"variant": variant, "seed": seed, "test_metrics": test_metrics,
                "diagnostic": diag, "checkpoint_path": ckpt_path}

    cells = [{"variant": v, "seed": s} for v in variants for s in seeds]
    results = run_resumable_grid(
        cells, key_fn=lambda c: (c["variant"], c["seed"]),
        run_fn=run_cell, jsonl_path=jsonl_path, artifact_check_fn=artifact_check)

    # ---- Aggregate mean/std per variant/horizon ----
    horizons = data["horizons"]
    aggregate = {v: {} for v in variants}
    per_variant_per_horizon = {v: {h: [] for h in horizons} for v in variants}
    for row in results:
        v, tm = row["variant"], row["test_metrics"]
        for h in horizons:
            per_variant_per_horizon[v][h].append(tm[f"h{h}"]["rmse"])
    for v in variants:
        for h in horizons:
            vals = per_variant_per_horizon[v][h]
            import numpy as np
            aggregate[v][f"h{h}"] = {
                "rmse_mean": float(np.mean(vals)), "rmse_std": float(np.std(vals)),
                "n_seeds": len(vals),
            }

    significance = {}
    if len(variants) == 2 and all(len(per_variant_per_horizon[v][horizons[0]]) == len(seeds)
                                   for v in variants):
        v1, v2 = variants
        for h in horizons:
            significance[f"h{h}"] = paired_seed_significance_test(
                per_variant_per_horizon[v1][h], per_variant_per_horizon[v2][h])

    return {"cells": results, "aggregate": aggregate, "significance": significance,
            "variants_compared": list(variants)}


# ---------------------------------------------------------------------------
# Item 5: Wider HPO pass (thin orchestration wrapper)
# ---------------------------------------------------------------------------

def run_wider_hpo(base_config: dict, k_values: Optional[List[int]] = (3, 5, 7, 9),
                   n_trials: int = 30, trial_epochs: int = 25,
                   results_dir: str = "results/robustness/wider_hpo") -> dict:
    """Runs the wider Optuna search (K + weight_decay + batch_size added to
    the base 5-parameter space), writing to `results_dir` (NOT
    results/hpo_best_params.json, so the original tuned-hyperparameter run
    used by items 1-4 above is never overwritten).

    n_trials=30 (vs. the original 10): chosen as a realistic Colab CPU/GPU
    time budget compromise -- see the notebook's Section 7 wall-clock
    estimate for the reasoning (roughly 3x the original trial count, still
    a small fraction of a full day even with K/decomposition rebuilding
    dominating cost, since VMD/EMD decomposition is precomputed ONCE per K
    via build_data_for_k_values, not per trial).
    """
    from .hpo import run_hpo, build_data_for_k_values
    from .data_pipeline import create_datasets

    if k_values is not None:
        data_by_k = build_data_for_k_values(base_config, list(k_values))
        base_data = data_by_k[base_config["vmd"]["K"]] if base_config["vmd"]["K"] in data_by_k \
            else next(iter(data_by_k.values()))
    else:
        data_by_k = None
        base_data = create_datasets(base_config)

    return run_hpo(
        base_config, base_data, n_trials=n_trials, trial_epochs=trial_epochs,
        search_weight_decay=True, search_batch_size=True, data_by_k=data_by_k,
        results_dir=results_dir, best_params_filename="hpo_wider_best_params.json",
        trials_csv_filename="hpo_wider_trials.csv",
    )
