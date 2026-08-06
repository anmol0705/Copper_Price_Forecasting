"""Training and evaluation pipeline for VMD-MFGNN and all baselines."""

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (EarlyStopper, compute_metrics, diebold_mariano_test,
                    get_device, save_results, set_seed)

logger = logging.getLogger(__name__)


class VMDMFGNNTrainer:
    """Trainer for the VMD-MFGNN model."""

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        tc = config["training"]
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"]
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc["epochs"]
        )
        self.stopper = EarlyStopper(patience=tc["patience"])

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x)
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
                preds = self.model(x)
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
            epochs: int = None) -> Dict:
        epochs = epochs or self.config["training"]["epochs"]
        best_state = None
        history = {"train_loss": [], "val_metrics": []}

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

            if val_mse < self.stopper.best_loss:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if self.stopper.should_stop(val_mse):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return history

    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_preds = {str(h): [] for h in self.model.horizons}
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                preds = self.model(x)
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
        for i, h in enumerate(horizons):
            pred_h = preds[:, i] if preds.ndim == 2 else preds
            true_h = true[:, i] if true.ndim == 2 else true
            results[f"h{h}"] = compute_metrics(true_h, pred_h)
        logger.info(f"  {name} done in {results['time']:.1f}s")
        return results

    except Exception as e:
        logger.error(f"  {name} failed: {e}")
        return {"name": name, "error": str(e)}


def run_all_experiments(config: dict, data: Dict) -> Dict:
    """Run VMD-MFGNN and all baselines, compute comparative results."""
    from .models.vmd_mfgnn import VMDMFGNN
    from .models.baselines import (
        LSTMBaseline, GRUBaseline, CNNLSTMBaseline, TransformerBaseline,
        VMDLSTMBaseline, VMDTransformerBaseline, VMDAttentionLSTM,
        SimpleMTGNN, GNNTransformer, XGBoostBaseline, LightGBMBaseline,
    )

    set_seed(config["training"]["seed"])
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
    model = VMDMFGNN(
        num_vars=num_vars, num_modes=num_modes,
        hidden_dim=mc["hidden_dim"], num_heads=mc["num_heads"],
        num_gnn_layers=mc["num_gnn_layers"],
        temporal_layers=mc["temporal_layers"],
        dropout=mc["dropout"], horizons=horizons,
        graph_type=mc["graph_type"]
    )
    trainer = VMDMFGNNTrainer(model, config)
    trainer.fit(data["train_loader"], data["val_loader"])
    test_results = trainer.evaluate(data["test_loader"])
    test_results["name"] = "VMD-MFGNN"
    all_results["VMD-MFGNN"] = test_results
    logger.info(f"VMD-MFGNN results: {test_results}")

    # ---- Raw price baselines ----
    raw_baselines = [
        ("LSTM", LSTMBaseline(base_cfg)),
        ("GRU", GRUBaseline(base_cfg)),
        ("CNN-LSTM", CNNLSTMBaseline(base_cfg)),
        ("Transformer", TransformerBaseline(base_cfg)),
        ("SimpleMTGNN", SimpleMTGNN(base_cfg)),
        ("GNN-Transformer", GNNTransformer(base_cfg)),
        ("XGBoost", XGBoostBaseline(base_cfg)),
        ("LightGBM", LightGBMBaseline(base_cfg)),
    ]
    for name, bl in raw_baselines:
        res = run_baseline(bl, data["raw_train_loader"],
                           data["raw_val_loader"], data["raw_test_loader"], name)
        all_results[name] = res

    # ---- VMD baselines (need mode data) ----
    vmd_baselines = [
        ("VMD-LSTM", VMDLSTMBaseline(base_cfg)),
        ("VMD-Transformer", VMDTransformerBaseline(base_cfg)),
        ("VMD-Attention-LSTM", VMDAttentionLSTM(base_cfg)),
    ]
    for name, bl in vmd_baselines:
        res = run_baseline(bl, data["train_loader"],
                           data["val_loader"], data["test_loader"], name)
        all_results[name] = res

    # ---- Save results ----
    save_results(all_results, "results/all_results.json")
    logger.info(f"All results saved to results/all_results.json")

    # ---- Print summary table ----
    print_results_table(all_results, horizons)

    return all_results


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


def run_ablation_studies(config: dict, data: Dict) -> Dict:
    """Run ablation experiments to isolate component contributions."""
    from .models.vmd_mfgnn import VMDMFGNN

    set_seed(config["training"]["seed"])
    mc = config["model"]
    horizons = data["horizons"]
    num_vars = data["num_vars"]
    num_modes = data["num_modes"]
    ablation_results = {}

    ablations = {
        "full_model": {"num_modes": num_modes, "graph_type": "learned"},
        "no_vmd_single_graph": {"num_modes": 1, "graph_type": "learned"},
        "correlation_graph": {"num_modes": num_modes, "graph_type": "correlation"},
    }

    for abl_name, overrides in ablations.items():
        logger.info(f"Ablation: {abl_name}")
        model = VMDMFGNN(
            num_vars=num_vars,
            num_modes=overrides.get("num_modes", num_modes),
            hidden_dim=mc["hidden_dim"], num_heads=mc["num_heads"],
            num_gnn_layers=mc["num_gnn_layers"],
            temporal_layers=mc["temporal_layers"],
            dropout=mc["dropout"], horizons=horizons,
            graph_type=overrides.get("graph_type", "learned")
        )
        trainer = VMDMFGNNTrainer(model, config)
        trainer.fit(data["train_loader"], data["val_loader"])
        results = trainer.evaluate(data["test_loader"])
        results["name"] = abl_name
        ablation_results[abl_name] = results

    save_results(ablation_results, "results/ablation_results.json")
    return ablation_results
