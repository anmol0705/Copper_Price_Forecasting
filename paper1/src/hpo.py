"""Optuna-based hyperparameter optimization for VMD-MFGNN ONLY.

Scope note (see task brief / orchestrator decision 2026-08-07): the 6 locked
baselines (ARIMA/XGBoost/LSTM/Transformer/VMD-LSTM/SimpleMTGNN) and the 4
ablation variants intentionally do NOT go through HPO -- they keep using
config["model"] / their hardcoded settings unmodified. This asymmetry is by
design and will be disclosed in the paper. This module is opt-in: it is only
invoked from run_all_experiments() when config["hpo"]["enabled"] is True.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_hpo(config: dict, data: Dict, n_trials: int = 15, trial_epochs: int = 25,
            search_weight_decay: bool = False, search_batch_size: bool = False,
            data_by_k: Optional[Dict[int, Dict]] = None,
            results_dir: str = "results",
            best_params_filename: str = "hpo_best_params.json",
            trials_csv_filename: str = "hpo_trials.csv") -> dict:
    """Run an Optuna study to tune VMD-MFGNN hyperparameters.

    Base search space (always on, unchanged from the original 4-trial-run
    implementation): hidden_dim, learning_rate, dropout, num_heads,
    num_gnn_layers. num_vars/num_modes/horizons/graph_type are taken from the
    real config/data and are NOT searched (unless `data_by_k` widens K, see
    below).

    Wider-search additions (item 5, all opt-in via explicit kwargs so the
    default call -- as used by run_all_experiments() with the checked-in
    configs/default.yaml -- is completely unchanged):
      - search_weight_decay=True: also searches training.weight_decay,
        log-uniform in [1e-6, 1e-3].
      - search_batch_size=True: also searches training.batch_size,
        categorical {16, 32, 64}. Requires `data`'s VMD train/val Datasets
        (data["train_ds"]/data["val_ds"]) to exist so a fresh DataLoader can
        be built per trial at the sampled batch size (the pre-built
        data["train_loader"]/data["val_loader"] are fixed at whatever batch
        size create_datasets() used and are only used as a fallback for
        trials that don't sample this parameter).
      - data_by_k={K: create_datasets(...)-style dict, ...}: also searches
        VMD/EMD band count K, categorical over data_by_k's keys. Each trial
        that samples a given K uses that K's pre-built train/val loaders
        (VMD decomposition itself is NOT re-run per trial -- it must already
        exist in data_by_k, e.g. via `build_data_for_k_values`). If K is not
        being searched (data_by_k is None, the default), behavior is
        unchanged: every trial uses `data`'s single fixed K.

    Each trial trains a fresh VMD-MFGNN for `trial_epochs` epochs (a reduced
    budget vs. the full config["training"]["epochs"]) and reports per-epoch
    validation avg_mse to Optuna so the median pruner can cut off unpromising
    trials early.

    Returns the best trial's hyperparameters as a dict (always includes the
    base 5 keys; includes "weight_decay"/"batch_size"/"K" only if that
    search dimension was enabled).

    Also writes, for reproducibility/reporting in the paper:
        {results_dir}/{best_params_filename}  -- the dict above
        {results_dir}/{trials_csv_filename}   -- full Optuna trial history
    `results_dir`/filenames are overridable so the wider-search pass (item 5)
    can write to a separate location (e.g. results/hpo_wider/) without
    overwriting the original run's results/hpo_best_params.json.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from torch.utils.data import DataLoader

    from .models.vmd_mfgnn import VMDMFGNN
    from .trainer import VMDMFGNNTrainer

    horizons = data["horizons"]
    num_vars = data["num_vars"]
    num_modes = data["num_modes"]
    graph_type = config["model"]["graph_type"]
    temporal_layers = config["model"]["temporal_layers"]
    seed = config.get("training", {}).get("seed", 42)
    base_weight_decay = config["training"]["weight_decay"]
    base_batch_size = config["training"]["batch_size"]

    hidden_dim_choices = [32, 64, 128]
    num_heads_choices = [2, 4, 8]
    num_gnn_layers_choices = [1, 2, 3]
    weight_decay_lo, weight_decay_hi = 1e-6, 1e-3
    batch_size_choices = [16, 32, 64]

    def objective(trial: "optuna.Trial") -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_choices)
        num_heads = trial.suggest_categorical("num_heads", num_heads_choices)
        # VMDMFGNN's GAT layer does hidden_dim // num_heads per head; keep the
        # search space to combinations that divide evenly so the effective
        # output dim always equals hidden_dim exactly.
        if hidden_dim % num_heads != 0:
            raise optuna.TrialPruned(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        num_gnn_layers = trial.suggest_categorical("num_gnn_layers", num_gnn_layers_choices)

        trial_config = copy.deepcopy(config)
        trial_config["training"]["epochs"] = trial_epochs
        trial_config["training"]["learning_rate"] = learning_rate

        if search_weight_decay:
            trial_config["training"]["weight_decay"] = trial.suggest_float(
                "weight_decay", weight_decay_lo, weight_decay_hi, log=True)
        else:
            trial_config["training"]["weight_decay"] = base_weight_decay

        # Select this trial's data (possibly K-specific).
        trial_data = data
        if data_by_k is not None:
            k_choice = trial.suggest_categorical("K", sorted(data_by_k.keys()))
            trial_data = data_by_k[k_choice]
            trial_num_vars = trial_data["num_vars"]
            trial_num_modes = trial_data["num_modes"]
        else:
            trial_num_vars, trial_num_modes = num_vars, num_modes

        if search_batch_size:
            bs = trial.suggest_categorical("batch_size", batch_size_choices)
            train_loader = DataLoader(trial_data["train_ds"], batch_size=bs, shuffle=True)
            val_loader = DataLoader(trial_data["val_ds"], batch_size=bs)
        else:
            train_loader = trial_data["train_loader"]
            val_loader = trial_data["val_loader"]

        model = VMDMFGNN(
            num_vars=trial_num_vars, num_modes=trial_num_modes,
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_gnn_layers=num_gnn_layers,
            temporal_layers=temporal_layers,
            dropout=dropout, horizons=horizons,
            graph_type=graph_type,
        )
        trainer = VMDMFGNNTrainer(model, trial_config)

        best_val_mse = float("inf")
        for epoch in range(trial_epochs):
            trainer.train_epoch(train_loader)
            val_results = trainer.evaluate(val_loader)
            val_mse = val_results["avg_mse"]
            best_val_mse = min(best_val_mse, val_mse)

            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_mse

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    # Fill in any dimension not searched with the base config's value, so
    # downstream consumers always get a complete, self-describing dict
    # regardless of which optional dimensions were enabled.
    best_params.setdefault("weight_decay", base_weight_decay)
    best_params.setdefault("batch_size", base_batch_size)
    logger.info(f"HPO complete. Best params: {best_params} "
                f"(val_mse={study.best_trial.value:.6f})")

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / best_params_filename, "w") as f:
        json.dump(best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / trials_csv_filename, index=False)

    logger.info(f"HPO artifacts saved to {out_dir / best_params_filename} and "
                f"{out_dir / trials_csv_filename}")
    return best_params


def build_data_for_k_values(config: dict, k_values: List[int]) -> Dict[int, Dict]:
    """Build one `create_datasets()`-style data dict per K value in
    `k_values`, each with its own decomposed-modes cache path (so sibling K
    values never invalidate each other's cache -- see
    `create_datasets(modes_cache_path=...)`). Used by the wider-HPO pass
    (item 5) when K is included in the search space, and reusable by the
    K-sweep experiment (item 1) for the same reason.

    Does NOT mutate `config` -- each K gets its own deep copy with
    config["vmd"]["K"] overridden.

    The K matching `config["vmd"]["K"]` (the main pipeline's already-
    configured K) is pointed at the EXISTING "data/vmd_modes.npy" cache
    instead of a fresh per-K path -- same reuse-the-main-cache logic as
    `experiments.run_k_sweep`, so the two experiments genuinely share that
    cache instead of each independently recomputing an identical K.
    """
    from .data_pipeline import create_datasets

    data_by_k = {}
    for k in k_values:
        k_config = copy.deepcopy(config)
        k_config["vmd"]["K"] = k
        logger.info(f"Building datasets for K={k}...")
        cache_path = "data/vmd_modes.npy" if k == config["vmd"]["K"] \
            else f"data/vmd_modes_K{k}.npy"
        data_by_k[k] = create_datasets(k_config, modes_cache_path=cache_path)
    return data_by_k
