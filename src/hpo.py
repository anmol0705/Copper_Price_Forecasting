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
from typing import Dict

logger = logging.getLogger(__name__)


def run_hpo(config: dict, data: Dict, n_trials: int = 15, trial_epochs: int = 25) -> dict:
    """Run a lightweight Optuna study to tune VMD-MFGNN hyperparameters.

    Searches: hidden_dim, learning_rate, dropout, num_heads, num_gnn_layers.
    num_vars/num_modes/horizons/graph_type are taken from the real config and
    data and are NOT searched. Each trial trains a fresh VMD-MFGNN for
    `trial_epochs` epochs (a reduced budget vs. the full
    config["training"]["epochs"]) and reports per-epoch validation avg_mse to
    Optuna so the median pruner can cut off unpromising trials early.

    Returns the best trial's hyperparameters as a dict, e.g.:
        {"hidden_dim": ..., "learning_rate": ..., "dropout": ...,
         "num_heads": ..., "num_gnn_layers": ...}

    Also writes, for reproducibility/reporting in the paper:
        results/hpo_best_params.json  -- the dict above
        results/hpo_trials.csv        -- full Optuna trial history
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    from .models.vmd_mfgnn import VMDMFGNN
    from .trainer import VMDMFGNNTrainer

    horizons = data["horizons"]
    num_vars = data["num_vars"]
    num_modes = data["num_modes"]
    graph_type = config["model"]["graph_type"]
    temporal_layers = config["model"]["temporal_layers"]
    seed = config.get("training", {}).get("seed", 42)

    hidden_dim_choices = [32, 64, 128]
    num_heads_choices = [2, 4, 8]
    num_gnn_layers_choices = [1, 2, 3]

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

        model = VMDMFGNN(
            num_vars=num_vars, num_modes=num_modes,
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_gnn_layers=num_gnn_layers,
            temporal_layers=temporal_layers,
            dropout=dropout, horizons=horizons,
            graph_type=graph_type,
        )
        trainer = VMDMFGNNTrainer(model, trial_config)

        best_val_mse = float("inf")
        for epoch in range(trial_epochs):
            trainer.train_epoch(data["train_loader"])
            val_results = trainer.evaluate(data["val_loader"])
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
    logger.info(f"HPO complete. Best params: {best_params} "
                f"(val_mse={study.best_trial.value:.6f})")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "hpo_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_dir / "hpo_trials.csv", index=False)

    logger.info("HPO artifacts saved to results/hpo_best_params.json and "
                "results/hpo_trials.csv")
    return best_params
