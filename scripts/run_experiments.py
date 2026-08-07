#!/usr/bin/env python3
"""Main entry point: download data, run VMD-MFGNN + all baselines, generate results."""

import sys
import logging
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure output directories exist BEFORE setting up logging
(project_root / "results").mkdir(exist_ok=True)
(project_root / "results" / "figures").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "results" / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    from src.utils import load_config, set_seed
    from src.data_pipeline import create_datasets
    from src.trainer import run_all_experiments, run_ablation_studies
    from src.visualize import generate_all_figures

    config = load_config(str(project_root / "configs" / "default.yaml"))
    set_seed(config["training"]["seed"])

    logger.info("=" * 60)
    logger.info("STEP 1: Data Download & VMD Decomposition")
    logger.info("=" * 60)
    data = create_datasets(config)
    logger.info(f"  Variables: {data['variable_names']}")
    logger.info(f"  Train/Val/Test samples: "
                f"{len(data['train_ds'])}/{len(data['val_ds'])}/{len(data['test_ds'])}")

    logger.info("=" * 60)
    logger.info("STEP 2: Run All Models")
    logger.info("=" * 60)
    results, vmd_mfgnn_trainer = run_all_experiments(config, data)

    logger.info("=" * 60)
    logger.info("STEP 3: Ablation Studies")
    logger.info("=" * 60)
    ablation_results = run_ablation_studies(config, data)

    logger.info("=" * 60)
    logger.info("STEP 4: Generate Figures")
    logger.info("=" * 60)
    # Get attention weights from the trained model and convert to numpy if needed
    attn_weights = vmd_mfgnn_trainer.model.get_attention_weights()
    if hasattr(attn_weights, 'cpu'):  # torch tensor
        attn_weights = attn_weights.cpu().numpy()
    elif not isinstance(attn_weights, np.ndarray):
        import numpy as np
        attn_weights = np.array(attn_weights)

    generate_all_figures(config, data, results, ablation_results,
                         output_dir=str(project_root / "results" / "figures"),
                         model=vmd_mfgnn_trainer.model,
                         attn_weights=attn_weights)

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
