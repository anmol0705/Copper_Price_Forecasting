"""Week A2 deliverable: build the full CuBench feature matrix (Blocks A/B'/C/D + T1/T2/T3
targets) from the cached raw panel in data/cubench/raw/, and write it to
data/cubench/features.parquet.

Run: .venv_corr/Scripts/python.exe scripts/cubench_build_features.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.cubench.features import build_feature_matrix, build_raw_panel
from src.cubench.targets import build_all_targets
from src.cubench.folds import make_folds, save_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_PATH = Path("data/cubench/features.parquet")


def main():
    panel = build_raw_panel()
    feat, meta = build_feature_matrix(panel=panel)
    targets = build_all_targets(panel)

    # feat and targets are both built directly from `panel` and share its exact row
    # order (positional RangeIndex 0..n-1) -- concat by position, NOT by a label-based
    # .join(), which would silently produce all-NaN targets if `feat`'s index has been
    # changed (e.g. to `date`) while `targets`'s index remains the default RangeIndex.
    assert len(feat) == len(targets), "feat/targets row-count mismatch -- not positionally alignable"
    full = pd.concat([feat.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)

    logger.info(f"Wrote {full.shape} to {OUT_PATH}")
    logger.info(f"Date range: {full['date'].min().date()} -> {full['date'].max().date()}")
    logger.info(f"GARCH fit meta: {meta['garch']}")

    folds = make_folds(panel["date"])
    save_folds(folds)

    # --- NaN summary report ---
    nan_frac = full.drop(columns=["date"]).isna().mean().sort_values(ascending=False)
    logger.info("Top 15 columns by NaN fraction:")
    for col, frac in nan_frac.head(15).items():
        logger.info(f"  {col}: {frac:.4f}")

    return full, meta, folds


if __name__ == "__main__":
    main()
