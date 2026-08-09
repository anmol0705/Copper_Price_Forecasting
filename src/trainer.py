"""Training and evaluation pipeline for VMD-MFGNN and all baselines."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (EarlyStopper, compute_metrics, diebold_mariano_test,
                    get_device, save_results, set_seed)

logger = logging.getLogger(__name__)


def _get_graph_type(model) -> str:
    """Detect a model's graph_type. VMDMFGNN itself doesn't store a
    top-level `graph_type` attribute (only its per-band FrequencyGraphConstructor
    submodules, in `model.graph_constructors`, do); PooledGraphMFGNN and
    `_UnsqueezeModeWrapper` do store one directly. Checked in that order.
    """
    if hasattr(model, "graph_type") and model.graph_type is not None:
        return model.graph_type
    graph_constructors = getattr(model, "graph_constructors", None)
    if graph_constructors is not None and len(graph_constructors) > 0:
        return graph_constructors[0].graph_type
    return None


class VMDMFGNNTrainer:
    """Trainer for the VMD-MFGNN model."""

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        tc = config["training"]
        if tc.get("no_decay_graph_embeddings", False):
            # OFF by default. When enabled, excludes the graph embedding
            # parameters (FrequencyGraphConstructor's emb1/emb2, for every
            # band, in both VMDMFGNN and PooledGraphMFGNN) from weight
            # decay. This directly targets the "Adjacency Collapse
            # Diagnosed" root cause in STATUS.md: Adam's coupled L2 decay
            # exerts a constant pull toward zero on these embeddings while
            # the task gradient reaching them is weak, causing an
            # isotropic norm collapse that makes the learned adjacency
            # degenerate to uniform (1/N). Parameter names are matched via
            # the ".emb1." / ".emb2." substring, which is present in both
            # models' dotted parameter paths (e.g.
            # "graph_constructors.0.emb1.weight" for VMDMFGNN's per-band
            # ModuleList, "graph_constructor.emb1.weight" for
            # PooledGraphMFGNN's single constructor).
            no_decay_params, decay_params = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if ".emb1." in name or ".emb2." in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            self.optimizer = torch.optim.Adam(
                [
                    {"params": decay_params, "weight_decay": tc["weight_decay"]},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=tc["learning_rate"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=tc["learning_rate"],
                weight_decay=tc["weight_decay"]
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc["epochs"]
        )
        self.stopper = EarlyStopper(patience=tc["patience"])

    def _forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass that threads precomputed correlation adjacency
        matrices through the model when graph_type=="correlation".

        For "correlation" graphs, the adjacency matrices used are the FROZEN
        per-band matrices computed once from the training split at the start
        of `fit()` (see `_freeze_correlation_adjs`), stored in
        `self._frozen_correlation_adjs`. This keeps the correlation graph a
        genuinely fixed/precomputed structure -- identical across
        train_epoch/evaluate/predict and never influenced by validation or
        test batches (avoids the small look-ahead leak that recomputing
        per-batch on the unshuffled test loader used to cause). If, for some
        reason, `_forward` is called before `_freeze_correlation_adjs` has
        run (e.g. direct use outside `fit()`), it falls back to computing
        the adjacency from the current batch, same as before. For "learned"
        graphs (or any model without a graph_type attribute, e.g. baselines
        routed through this trainer), behavior is unchanged.
        """
        graph_type = _get_graph_type(self.model)
        if graph_type == "correlation" and x.dim() == 4:
            frozen = getattr(self, "_frozen_correlation_adjs", None)
            if frozen is not None:
                precomputed_adjs = frozen
            else:
                precomputed_adjs = self._compute_precomputed_adjs(x)
            return self.model(x, precomputed_adjs=precomputed_adjs)
        return self.model(x)

    def _compute_precomputed_adjs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute one correlation adjacency matrix per VMD band from the
        current input batch. x: (B, T, K, N) -> list of K (N, N) matrices.

        Since compute_correlation_adjacency expects a single (T, N) band
        signal (not batched), the batch dimension is collapsed by
        concatenating all samples' timesteps along the time axis, giving a
        longer single "series" per band to correlate over.

        This is now only used as a fallback (see `_forward`'s docstring) --
        the normal path uses `_freeze_correlation_adjs`'s once-computed,
        train-split-only matrices instead.
        """
        from .models.vmd_mfgnn import FrequencyGraphConstructor
        B, T, K, N = x.shape
        precomputed_adjs = []
        for k in range(K):
            band = x[:, :, k, :].reshape(B * T, N)  # (B*T, N)
            adj = FrequencyGraphConstructor.compute_correlation_adjacency(band)
            precomputed_adjs.append(adj)
        return precomputed_adjs

    def _freeze_correlation_adjs(self, train_loader: DataLoader) -> None:
        """Compute the K per-band correlation adjacency matrices ONCE, using
        only the TRAINING split's data, and cache them on this trainer
        instance as `self._frozen_correlation_adjs`.

        Called once at the start of `fit()`, only when the model's
        graph_type == "correlation" (skipped entirely for "learned" graphs
        to avoid wasted work). All of train_epoch/evaluate/predict then read
        from this same frozen cache via `_forward`, instead of each batch
        (including validation/test batches) recomputing its own adjacency --
        which previously let the unshuffled test loader leak later
        chronological windows into earlier test samples' adjacency.
        """
        from .models.vmd_mfgnn import FrequencyGraphConstructor
        all_x = []
        for x, _ in train_loader:
            all_x.append(x)
        full_x = torch.cat(all_x, dim=0)  # (N_train_samples, T, K, N_vars)
        _, T, K, N = full_x.shape
        frozen_adjs = []
        for k in range(K):
            band = full_x[:, :, k, :].reshape(-1, N)  # (N_train_samples*T, N)
            adj = FrequencyGraphConstructor.compute_correlation_adjacency(band)
            frozen_adjs.append(adj.to(self.device))
        self._frozen_correlation_adjs = frozen_adjs
        logger.info(f"Froze {K} correlation adjacency matrices from the "
                    f"training split only ({full_x.size(0)} samples, "
                    f"computed once for this fit() call).")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self._forward(x)
            loss = sum(F.mse_loss(preds[str(h)], y[:, i])
                       for i, h in enumerate(self.model.horizons))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / max(n, 1)

    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds = {str(h): [] for h in self.model.horizons}
        all_targets = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self._forward(x)
                for i, h in enumerate(self.model.horizons):
                    all_preds[str(h)].append(preds[str(h)].cpu().numpy())
                all_targets.append(y.cpu().numpy())

        targets = np.concatenate(all_targets, axis=0)
        results = {}
        total_loss = 0.0
        for i, h in enumerate(self.model.horizons):
            pred = np.concatenate(all_preds[str(h)])
            true = targets[:, i]
            metrics = compute_metrics(true, pred)
            results[f"h{h}"] = metrics
            total_loss += metrics["rmse"] ** 2
        results["avg_mse"] = total_loss / len(self.model.horizons)
        return results

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = None, checkpoint_path: Optional[Path] = None) -> Dict:
        """Train the model, with optional disk-based checkpointing.

        Checkpoint file format: a dict `{"state_dict": ..., "epoch": epoch,
        "completed": bool}`. `"completed"` is True only when the training
        loop actually finished (either early-stopping triggered or the
        final configured epoch was reached) -- NOT after every mid-training
        best-val-loss improvement. This distinguishes a genuinely-finished
        run's checkpoint from one left behind by an interrupted run (e.g. a
        Colab disconnect at epoch 3 of 200), which would otherwise look
        identical on resume. Legacy checkpoints saved before this flag
        existed (a bare state dict, no wrapping dict) are treated as
        `completed=True` for backward compatibility.

        Resumability semantics (read before wiring a checkpoint_path into a
        new call site):
          - If `checkpoint_path` is given, the file already exists on disk,
            AND its `"completed"` flag is True, training is SKIPPED
            ENTIRELY: the saved state dict is loaded into `self.model` and a
            minimal history dict is returned. This is intentional
            resumability for a genuinely finished run -- callers that need a
            guaranteed fresh run must delete any stale checkpoint file
            before calling fit().
          - If `checkpoint_path` is given, the file exists, but its
            `"completed"` flag is False (an incomplete checkpoint from a
            prior interrupted run), a WARNING is logged and training does
            NOT skip: the saved weights are loaded into `self.model` as a
            starting point, and a normal fresh `epochs`-length training loop
            runs from there -- genuine resume-from-partial-progress, rather
            than either silently skipping training or discarding all prior
            progress.
          - Whenever training runs (fresh, or resumed from an incomplete
            checkpoint) and `checkpoint_path` is given, every time a new
            best validation loss is found, the best-so-far state dict is
            written to `checkpoint_path` with `completed=False` (in addition
            to being kept in memory as `best_state`), so a crash mid-training
            still leaves the best-so-far weights recoverable. When the loop
            actually exits (early-stopping or final epoch reached), the
            checkpoint is re-written once more with `completed=True`.
          - If `checkpoint_path` is None (the default), behavior is
            byte-for-byte identical to the pre-checkpointing implementation:
            no disk I/O, in-memory best_state only. This keeps existing
            callers that don't pass checkpoint_path fully backward
            compatible.
        """
        epochs = epochs or self.config["training"]["epochs"]

        if _get_graph_type(self.model) == "correlation":
            self._freeze_correlation_adjs(train_loader)

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                loaded = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    state = loaded["state_dict"]
                    completed = bool(loaded.get("completed", False))
                else:
                    # Legacy bare-state-dict checkpoint predating the
                    # completion flag -- treat as a finished run.
                    state = loaded
                    completed = True

                try:
                    self.model.load_state_dict(state)
                    self.model.to(self.device)
                except RuntimeError as e:
                    # The on-disk checkpoint's tensor shapes don't match this
                    # model's architecture -- most commonly because a
                    # hyperparameter that determines parameter shapes (e.g.
                    # hidden_dim) changed since the checkpoint was saved, such
                    # as HPO picking a different hidden_dim on a later run
                    # than whatever produced the stale checkpoint at this
                    # same path. Rather than crash, discard the incompatible
                    # checkpoint and fall through to a completely fresh
                    # training run, exactly as if checkpoint_path didn't
                    # exist. New checkpoints saved below (as training
                    # progresses) will overwrite this stale file with
                    # correctly-shaped weights, so future resumes at this
                    # model's current architecture will work normally again.
                    logger.warning(
                        f"Checkpoint at {checkpoint_path} is incompatible "
                        f"with the current model's architecture (likely "
                        f"because a shape-determining hyperparameter such as "
                        f"hidden_dim changed since it was saved -- e.g. HPO "
                        f"selected a different hidden_dim). Ignoring the "
                        f"stale checkpoint and training FRESH instead of "
                        f"resuming. Original error: {e}"
                    )
                else:
                    if completed:
                        logger.info(f"Resuming from checkpoint {checkpoint_path}: "
                                    f"training previously completed, skipping "
                                    f"training entirely.")
                        return {"train_loss": [], "val_metrics": [],
                                "resumed_from_checkpoint": str(checkpoint_path)}
                    else:
                        logger.warning(
                            f"Incomplete checkpoint found at {checkpoint_path} "
                            f"(a prior run was interrupted before completion). "
                            f"Loaded its weights as a starting point and "
                            f"continuing with a fresh {epochs}-epoch training "
                            f"run rather than skipping training."
                        )

        best_state = None
        history = {"train_loss": [], "val_metrics": []}
        last_epoch = -1

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_results = self.evaluate(val_loader)
            self.scheduler.step()
            dt = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_metrics"].append(val_results)

            val_mse = val_results["avg_mse"]
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.6f} "
                            f"| val_mse={val_mse:.6f} | {dt:.1f}s")

            last_epoch = epoch
            if val_mse < self.stopper.best_loss:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": best_state, "epoch": epoch,
                                "completed": False}, checkpoint_path)

            if self.stopper.should_stop(val_mse):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            if checkpoint_path is not None:
                # Final save: the loop has now actually exited (either via
                # early-stopping or by reaching the last configured epoch),
                # so this is the one save marked completed=True.
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best_state, "epoch": last_epoch,
                            "completed": True}, checkpoint_path)
        return history

    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_preds = {str(h): [] for h in self.model.horizons}
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                preds = self._forward(x)
                for h in self.model.horizons:
                    all_preds[str(h)].append(preds[str(h)].cpu().numpy())
        return {k: np.concatenate(v) for k, v in all_preds.items()}


def run_baseline(model, train_loader, val_loader, test_loader,
                 model_name: str = None) -> Dict:
    """Run a single baseline model and return test metrics."""
    name = model_name or getattr(model, "name", model.__class__.__name__)
    logger.info(f"Training {name}...")
    t0 = time.time()

    try:
        model.fit(train_loader, val_loader)
        preds = model.predict(test_loader)

        # Collect true values
        ys = []
        for _, y in test_loader:
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else y)
        true = np.concatenate(ys, axis=0)

        horizons = model.horizons if hasattr(model, "horizons") else model.config.get("horizons", [1, 5, 10, 22])
        results = {"name": name, "time": time.time() - t0}
        pred_dir = Path("results/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, h in enumerate(horizons):
            pred_h = preds[:, i] if preds.ndim == 2 else preds
            true_h = true[:, i] if true.ndim == 2 else true
            results[f"h{h}"] = compute_metrics(true_h, pred_h)
            np.save(pred_dir / f"{name}_{h}.npy", pred_h)
            np.save(pred_dir / f"{name}_{h}_true.npy", true_h)
        logger.info(f"  {name} done in {results['time']:.1f}s")
        return results

    except Exception as e:
        logger.error(f"  {name} failed: {e}")
        return {"name": name, "error": str(e)}


def run_all_experiments(config: dict, data: Dict):
    """Run VMD-MFGNN and all baselines, compute comparative results.

    Returns (all_results, trainer) so callers (e.g. figure generation) can
    access the trained VMD-MFGNN model/trainer after this function returns.
    """
    from .models.vmd_mfgnn import VMDMFGNN
    from .models.baselines import (
        LSTMBaseline, TransformerBaseline, VMDLSTMBaseline,
        SimpleMTGNN, XGBoostBaseline, ARIMABaseline,
    )

    seed = config["training"]["seed"]
    device = str(get_device())
    horizons = data["horizons"]
    num_vars = data["num_vars"]
    num_modes = data["num_modes"]

    base_cfg = {
        "num_vars": num_vars, "num_modes": num_modes,
        "hidden_dim": config["model"]["hidden_dim"],
        "num_layers": config["model"]["temporal_layers"],
        "num_heads": config["model"]["num_heads"],
        "dropout": config["model"]["dropout"],
        "lookback": config["data"]["lookback"],
        "horizons": horizons,
        "epochs": config["training"]["epochs"],
        "lr": config["training"]["learning_rate"],
        "weight_decay": 1e-5,
        "patience": config["training"]["patience"],
        "device": device,
    }

    all_results = {}

    # ---- VMD-MFGNN (our model) ----
    logger.info("=" * 60)
    logger.info("Training VMD-MFGNN (proposed model)")
    logger.info("=" * 60)
    mc = config["model"]
    vmd_mfgnn_config = config

    # HPO is opt-in and applies ONLY to VMD-MFGNN, never to baselines or
    # ablation variants (see vmd-mfgnn-protocol/SKILL.md / task brief). When
    # disabled (the default), behavior below is unchanged from the
    # already-verified pipeline: mc stays == config["model"].
    if config.get("hpo", {}).get("enabled", False):
        import copy
        from .hpo import run_hpo
        n_trials = config["hpo"].get("n_trials", 15)
        trial_epochs = config["hpo"].get("trial_epochs", 25)
        logger.info(f"HPO enabled: running Optuna search ({n_trials} trials, "
                    f"{trial_epochs} epochs/trial) for VMD-MFGNN only")
        best_params = run_hpo(config, data, n_trials=n_trials, trial_epochs=trial_epochs)
        logger.info(f"HPO winner hyperparameters: {best_params}")
        mc = dict(mc)
        mc["hidden_dim"] = best_params["hidden_dim"]
        mc["num_heads"] = best_params["num_heads"]
        mc["dropout"] = best_params["dropout"]
        mc["num_gnn_layers"] = best_params["num_gnn_layers"]
        vmd_mfgnn_config = copy.deepcopy(config)
        vmd_mfgnn_config["model"] = mc
        vmd_mfgnn_config["training"]["learning_rate"] = best_params["learning_rate"]

    set_seed(seed)
    model = VMDMFGNN(
        num_vars=num_vars, num_modes=num_modes,
        hidden_dim=mc["hidden_dim"], num_heads=mc["num_heads"],
        num_gnn_layers=mc["num_gnn_layers"],
        temporal_layers=mc["temporal_layers"],
        dropout=mc["dropout"], horizons=horizons,
        graph_type=mc["graph_type"]
    )
    trainer = VMDMFGNNTrainer(model, vmd_mfgnn_config)
    trainer.fit(data["train_loader"], data["val_loader"],
                checkpoint_path=Path("results/checkpoints/vmd_mfgnn.pt"))
    test_results = trainer.evaluate(data["test_loader"])
    test_results["name"] = "VMD-MFGNN"
    all_results["VMD-MFGNN"] = test_results
    logger.info(f"VMD-MFGNN results: {test_results}")

    # ---- Save VMD-MFGNN predictions + interpretability artifacts ----
    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    interp_dir = Path("results/interpretability")
    interp_dir.mkdir(parents=True, exist_ok=True)

    vmd_preds = trainer.predict(data["test_loader"])
    test_ys = []
    for _, y in data["test_loader"]:
        test_ys.append(y.numpy() if isinstance(y, torch.Tensor) else y)
    test_true = np.concatenate(test_ys, axis=0)
    for i, h in enumerate(horizons):
        np.save(pred_dir / f"VMD-MFGNN_{h}.npy", vmd_preds[str(h)])
        np.save(pred_dir / f"VMD-MFGNN_{h}_true.npy", test_true[:, i])

    # get_learned_graphs()/get_attention_weights() reflect whatever the last
    # forward pass populated; trainer.predict() above just ran a full pass
    # over the test set, so attention weights correspond to test-time
    # activity and learned graphs are the (input-independent) final learned
    # adjacencies.
    learned_graphs = model.get_learned_graphs()
    attention_weights = model.get_attention_weights()
    if learned_graphs:
        torch.save(learned_graphs, interp_dir / "learned_graphs.pt")
    if attention_weights is not None:
        torch.save(attention_weights, interp_dir / "attention_weights.pt")

    # ---- Raw price baselines ----
    # NOTE: constructors are deferred (lambdas), not pre-built instances, so
    # that set_seed() can run immediately before each model's construction
    # (per-model seeding, see docstring note below / SKILL.md Batch 2 fix) --
    # this keeps every model's init+shuffle RNG stream independent of
    # whatever ran before it (order-independent, checkpoint-resume-safe,
    # HPO-toggle-safe).
    raw_baseline_ctors = [
        ("ARIMA", lambda: ARIMABaseline(base_cfg)),
        ("LSTM", lambda: LSTMBaseline(base_cfg)),
        ("Transformer", lambda: TransformerBaseline(base_cfg)),
        ("SimpleMTGNN", lambda: SimpleMTGNN(base_cfg)),
        ("XGBoost", lambda: XGBoostBaseline(base_cfg)),
    ]
    for name, ctor in raw_baseline_ctors:
        set_seed(seed)
        bl = ctor()
        res = run_baseline(bl, data["raw_train_loader"],
                           data["raw_val_loader"], data["raw_test_loader"], name)
        all_results[name] = res

    # ---- VMD baselines (need mode data) ----
    vmd_baseline_ctors = [
        ("VMD-LSTM", lambda: VMDLSTMBaseline(base_cfg)),
    ]
    for name, ctor in vmd_baseline_ctors:
        set_seed(seed)
        bl = ctor()
        res = run_baseline(bl, data["train_loader"],
                           data["val_loader"], data["test_loader"], name)
        all_results[name] = res

    # ---- Save results ----
    save_results(all_results, "results/all_results.json")
    logger.info(f"All results saved to results/all_results.json")

    # ---- Print summary table ----
    print_results_table(all_results, horizons)

    # ---- Diebold-Mariano significance tests vs. each baseline ----
    baseline_names = [name for name, _ in raw_baseline_ctors] + [name for name, _ in vmd_baseline_ctors]
    run_significance_tests("VMD-MFGNN", baseline_names, horizons)

    return all_results, trainer


def print_results_table(results: Dict, horizons: List[int]):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<25}", end="")
    for h in horizons:
        print(f"| h={h:<3} RMSE  MAE    DA   ", end="")
    print("\n" + "-" * 90)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<25} ERROR: {res['error']}")
            continue
        print(f"{name:<25}", end="")
        for h in horizons:
            key = f"h{h}"
            if key in res:
                m = res[key]
                print(f"| {m['rmse']:.4f} {m['mae']:.4f} {m['da']:.1f}% ", end="")
            else:
                print(f"| {'N/A':>22} ", end="")
        print()
    print("=" * 90)


def run_significance_tests(vmd_mfgnn_name: str, baseline_names: List[str],
                            horizons: List[int]) -> Dict:
    """Diebold-Mariano test of VMD-MFGNN vs. each baseline, per horizon.

    Loads the prediction/true arrays saved by run_baseline()/run_all_experiments()
    under results/predictions/, computes each model's raw error array
    (pred - true), and runs diebold_mariano_test(vmd_errors, baseline_errors,
    horizon=h). Saves a flat table to results/significance_table.json.
    """
    pred_dir = Path("results/predictions")
    table = {}
    for h in horizons:
        vmd_pred_path = pred_dir / f"{vmd_mfgnn_name}_{h}.npy"
        vmd_true_path = pred_dir / f"{vmd_mfgnn_name}_{h}_true.npy"
        if not (vmd_pred_path.exists() and vmd_true_path.exists()):
            logger.warning(f"Missing VMD-MFGNN prediction arrays for h={h}, skipping")
            continue
        vmd_errors = np.load(vmd_pred_path) - np.load(vmd_true_path)

        for name in baseline_names:
            pred_path = pred_dir / f"{name}_{h}.npy"
            true_path = pred_dir / f"{name}_{h}_true.npy"
            if not (pred_path.exists() and true_path.exists()):
                logger.warning(f"Missing prediction arrays for {name} h={h}, skipping")
                continue
            baseline_errors = np.load(pred_path) - np.load(true_path)
            dm = diebold_mariano_test(vmd_errors, baseline_errors, horizon=h)
            table[f"{name}_h{h}"] = {
                "baseline": name, "horizon": h,
                "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
            }

    save_results(table, "results/significance_table.json")
    logger.info("Significance table saved to results/significance_table.json")
    return table


class _UnsqueezeModeWrapper(torch.nn.Module):
    """Wraps a VMDMFGNN(num_modes=1, ...) instance so it can be driven by a
    RawPriceDataset loader, which yields 3D (B, T, num_vars) batches. Adds a
    singleton mode dimension so the underlying model sees its expected 4D
    (B, T, 1, num_vars) input. Used only for the "no-VMD" ablation, where the
    data must be genuine raw prices (never VMD-decomposed), not a
    `num_modes=1` CopperDataset workaround.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.horizons = inner.horizons
        self.graph_type = _get_graph_type(inner)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (B, T, N) -> (B, T, 1, N)
        return self.inner(x, **kwargs)

    def get_attention_weights(self):
        return self.inner.get_attention_weights()

    def get_learned_graphs(self):
        return self.inner.get_learned_graphs()


def _find_matched_hidden_dim(build_fn, target_params: int, num_heads: int,
                              lo: int = 32, hi: int = 512) -> "tuple[int, int]":
    """Search candidate `hidden_dim` values (multiples of `num_heads`,
    respecting GATConv's `hidden_dim % num_heads == 0` requirement) for the
    one whose `build_fn(hidden_dim)` model has a parameter count
    (via `pooled_graph_mfgnn.count_parameters`) closest to `target_params`.

    build_fn: callable taking a single `hidden_dim` int and returning a
        freshly-constructed nn.Module (CPU, untrained -- only used to count
        parameters, never trained itself).
    Returns: (best_hidden_dim, best_param_count).
    """
    from .models.pooled_graph_mfgnn import count_parameters

    candidates = [h for h in range(lo, hi + 1) if h % num_heads == 0]
    best_h, best_count, best_diff = None, None, None
    for h in candidates:
        m = build_fn(h)
        c = count_parameters(m)
        diff = abs(c - target_params)
        if best_diff is None or diff < best_diff:
            best_h, best_count, best_diff = h, c, diff
        del m
    return best_h, best_count


def run_ablation_studies(config: dict, data: Dict) -> Dict:
    """Run the 3 locked ablation studies (see vmd-mfgnn-protocol/SKILL.md):
    (1) VMD vs. no-VMD, (2) per-band vs. pooled graph, (3) learned vs.
    correlation graph. 6 model variants total: full_model, two
    capacity-unmatched-vs-matched pairs for the pooled-graph and no-VMD
    ablations (see below), and correlation_graph.

    Capacity-matched ablations (fix for the pooled-graph / no-VMD parameter
    confound flagged by the independent expert review, see STATUS.md
    "CONSOLIDATED VERDICT" #3/#4): both PooledGraphMFGNN and the
    num_modes=1 no-VMD VMDMFGNN have substantially fewer parameters than
    full_model at the config's shared hidden_dim, which confounds "per-band
    beats pooled" / "VMD beats no-VMD" with "bigger model beats smaller
    model." Each of these two ablations is now run TWICE:
      - "*_matched_dim": the original equal-hidden_dim setting (kept
        unchanged for continuity), capacity-UNMATCHED vs. full_model.
      - "*_matched_params": a separately-searched hidden_dim (via
        `_find_matched_hidden_dim`) that brings the variant's parameter
        count close to full_model's, capacity-MATCHED.
    This gives two honest comparisons instead of one confounded one.
    """
    from .models.vmd_mfgnn import VMDMFGNN
    from .models.pooled_graph_mfgnn import PooledGraphMFGNN, count_parameters

    seed = config["training"]["seed"]
    mc = config["model"]
    horizons = data["horizons"]
    num_vars = data["num_vars"]
    num_modes = data["num_modes"]
    ablation_results = {}

    def make_vmd_mfgnn(num_modes_, graph_type_, hidden_dim_=None):
        return VMDMFGNN(
            num_vars=num_vars, num_modes=num_modes_,
            hidden_dim=hidden_dim_ if hidden_dim_ is not None else mc["hidden_dim"],
            num_heads=mc["num_heads"],
            num_gnn_layers=mc["num_gnn_layers"],
            temporal_layers=mc["temporal_layers"],
            dropout=mc["dropout"], horizons=horizons,
            graph_type=graph_type_,
        )

    def make_pooled(hidden_dim_=None):
        return PooledGraphMFGNN(
            num_vars=num_vars, num_modes=num_modes,
            hidden_dim=hidden_dim_ if hidden_dim_ is not None else mc["hidden_dim"],
            num_heads=mc["num_heads"],
            num_gnn_layers=mc["num_gnn_layers"],
            temporal_layers=mc["temporal_layers"],
            dropout=mc["dropout"], horizons=horizons,
            graph_type="learned",
        )

    def run_variant(key, model, train_loader, val_loader, test_loader):
        # NOTE: set_seed(seed) is called by the caller immediately before
        # `model` was constructed (per-model seeding fix), so each variant's
        # init+shuffle RNG stream is independent of whichever variant ran
        # before it, regardless of checkpoint-resume/HPO state.
        trainer = VMDMFGNNTrainer(model, config)
        trainer.fit(train_loader, val_loader,
                    checkpoint_path=Path(f"results/checkpoints/{key}.pt"))
        results = trainer.evaluate(test_loader)
        results["name"] = key
        ablation_results[key] = results
        return trainer

    # (a) full_model: standard VMD-MFGNN, learned per-band graphs, real VMD data.
    logger.info("Ablation: full_model")
    set_seed(seed)
    full_model = make_vmd_mfgnn(num_modes, "learned")
    full_params = count_parameters(full_model)
    logger.info(f"full_model param count: {full_params:,} "
                f"(hidden_dim={mc['hidden_dim']})")
    run_variant("full_model", full_model, data["train_loader"],
                data["val_loader"], data["test_loader"])

    # (b1) no_vmd_raw_price_matched_dim: genuine no-VMD baseline at the
    # config's shared hidden_dim (capacity-UNMATCHED vs. full_model, kept
    # for continuity with the pre-fix ablation). Trained/evaluated on
    # RawPriceDataset-derived loaders (raw_train/val/test_loader) — data
    # that has never touched VMD decomposition — via a num_modes=1 VMDMFGNN
    # wrapped to unsqueeze a singleton mode dim onto the 3D raw batches.
    logger.info("Ablation: no_vmd_raw_price_matched_dim")
    set_seed(seed)
    no_vmd_dim_inner = make_vmd_mfgnn(1, "learned")
    no_vmd_dim_params = count_parameters(no_vmd_dim_inner)
    logger.info(f"no_vmd_raw_price_matched_dim param count: {no_vmd_dim_params:,} "
                f"vs full_model's {full_params:,} "
                f"({(no_vmd_dim_params - full_params) / full_params * 100:+.1f}%)")
    no_vmd_dim_model = _UnsqueezeModeWrapper(no_vmd_dim_inner)
    run_variant("no_vmd_raw_price_matched_dim", no_vmd_dim_model,
                data["raw_train_loader"], data["raw_val_loader"], data["raw_test_loader"])

    # (b2) no_vmd_raw_price_matched_params: same no-VMD setup, but at a
    # hidden_dim searched to bring parameter count close to full_model's
    # (capacity-MATCHED).
    def build_no_vmd(h):
        return make_vmd_mfgnn(1, "learned", hidden_dim_=h)

    matched_h, matched_c = _find_matched_hidden_dim(build_no_vmd, full_params, mc["num_heads"])
    logger.info(f"matched no_vmd_raw_price to hidden_dim={matched_h}, "
                f"{matched_c:,} params vs full_model's {full_params:,} "
                f"({(matched_c - full_params) / full_params * 100:+.1f}%)")
    logger.info("Ablation: no_vmd_raw_price_matched_params")
    set_seed(seed)
    no_vmd_params_inner = build_no_vmd(matched_h)
    no_vmd_params_model = _UnsqueezeModeWrapper(no_vmd_params_inner)
    run_variant("no_vmd_raw_price_matched_params", no_vmd_params_model,
                data["raw_train_loader"], data["raw_val_loader"], data["raw_test_loader"])

    # (c1) pooled_graph_matched_dim: PooledGraphMFGNN — ONE pooled graph
    # instead of K per-band graphs, at the config's shared hidden_dim
    # (capacity-UNMATCHED vs. full_model, kept for continuity). Trained/
    # evaluated on the same VMD-mode loaders as full_model. Highest-priority
    # ablation per locked protocol.
    logger.info("Ablation: pooled_graph_matched_dim")
    set_seed(seed)
    pooled_dim_model = make_pooled()
    pooled_dim_params = count_parameters(pooled_dim_model)
    logger.info(f"pooled_graph_matched_dim param count: {pooled_dim_params:,} "
                f"vs full_model's {full_params:,} "
                f"({(pooled_dim_params - full_params) / full_params * 100:+.1f}%)")
    run_variant("pooled_graph_matched_dim", pooled_dim_model,
                data["train_loader"], data["val_loader"], data["test_loader"])

    # (c2) pooled_graph_matched_params: same pooled-graph setup, but at a
    # hidden_dim searched to bring parameter count close to full_model's
    # (capacity-MATCHED) — this is the honest version of the paper's
    # central "per-band beats pooled" comparison.
    matched_h, matched_c = _find_matched_hidden_dim(make_pooled, full_params, mc["num_heads"])
    logger.info(f"matched pooled_graph to hidden_dim={matched_h}, "
                f"{matched_c:,} params vs full_model's {full_params:,} "
                f"({(matched_c - full_params) / full_params * 100:+.1f}%)")
    logger.info("Ablation: pooled_graph_matched_params")
    set_seed(seed)
    pooled_params_model = make_pooled(matched_h)
    run_variant("pooled_graph_matched_params", pooled_params_model,
                data["train_loader"], data["val_loader"], data["test_loader"])

    # (d) correlation_graph: VMDMFGNN with graph_type="correlation", per-band
    # frozen (train-split-only, computed once) correlation adjacency
    # threaded through via the trainer's _forward/_freeze_correlation_adjs
    # helpers. Same VMD-mode loaders.
    logger.info("Ablation: correlation_graph")
    set_seed(seed)
    corr_model = make_vmd_mfgnn(num_modes, "correlation")
    run_variant("correlation_graph", corr_model,
                data["train_loader"], data["val_loader"], data["test_loader"])

    save_results(ablation_results, "results/ablation_results.json")
    return ablation_results
