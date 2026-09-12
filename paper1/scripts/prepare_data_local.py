"""Local (non-Colab) data-prep step: real yfinance + FRED download and the
leakage-safe expanding-window VMD decomposition, run once on a machine with
real internet access and a CPU (VMD has no GPU path -- it gains nothing from
a Colab GPU runtime). Produces the exact same data/raw_prices.csv,
data/vmd_modes.npy, data/vmd_modes_meta.json that create_datasets() would
build inside Colab; upload that data/ directory to Google Drive and Colab's
restore_data_cache_from_drive() (see the notebook) will pick it up via the
same sha256-of-price-array + params cache-hit check, skipping the download
and the ~30-45 min VMD step entirely.

Usage: python scripts/prepare_data_local.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.utils import load_config, set_seed  # noqa: E402
from src.data_pipeline import create_datasets  # noqa: E402


def main():
    config = load_config("configs/default.yaml")
    set_seed(config["training"]["seed"])

    print("Date range:", config["data"]["start_date"], "->", config["data"]["end_date"])
    print("VMD K:", config["vmd"]["K"])

    # debug_fast=False is mandatory here: True selects VMDDecomposerFast, the
    # leaky full-series batch decomposer, never valid for a real/reported run.
    data = create_datasets(config, debug_fast=False)

    print()
    print("=== Verification ===")
    print("Variables:", data["variable_names"])
    print("num_vars:", data["num_vars"], "num_modes:", data["num_modes"])
    print("Train/Val/Test samples:", len(data["train_ds"]), len(data["val_ds"]), len(data["test_ds"]))

    assert data["num_vars"] == 15, (
        f"expected 15 variables (N=8->15 expansion, aluminum removed -- see "
        f"PLAN.md Section 1), got {data['num_vars']}: {data['variable_names']}"
    )
    assert "aluminum" not in data["variable_names"], (
        "aluminum should have been removed from TICKERS -- see PLAN.md Section 4"
    )
    assert data["num_modes"] == config["vmd"]["K"]

    x0, y0 = data["train_ds"][0]
    print("Sample X shape:", tuple(x0.shape), "Y shape:", tuple(y0.shape))
    assert x0.shape == (config["data"]["lookback"], config["vmd"]["K"], data["num_vars"])

    import pandas as pd
    prices = pd.read_csv("data/raw_prices.csv", index_col=0, parse_dates=True)
    print()
    print("Panel date range:", prices.index.min(), "->", prices.index.max())
    print("Panel row count:", len(prices))
    assert prices.index.min() <= pd.Timestamp("2010-01-05"), (
        f"panel starts at {prices.index.min()}, expected ~2010-01-04 -- a late "
        f"start here would mean a variable is silently truncating the panel "
        f"again (the exact ALI=F bug this expansion already fixed once)."
    )
    print("No truncation: panel starts at the expected ~2010-01-04 date.")

    print()
    print("=== All checks passed. data/ is ready to upload to Google Drive. ===")
    print("Files to upload (as Copper_Paper1.2/data_cache/ in Drive):")
    for f in sorted(Path("data").iterdir()):
        print(" ", f, f"({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
