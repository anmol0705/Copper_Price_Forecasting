import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

TICKERS = {
    "copper": "HG=F",
    "aluminum": "ALI=F",
    "zinc": "ZNC=F",
    "nickel": "NI=F",
    "gold": "GC=F",
    "oil": "CL=F",
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "us10y": "^TNX",
}

VARIABLE_NAMES = list(TICKERS.keys())


class DataDownloader:
    def __init__(self, start: str = "2010-01-01", end: str = "2025-12-31",
                 cache_path: str = "data/raw_prices.csv"):
        self.start = start
        self.end = end
        self.cache_path = Path(cache_path)

    def download(self) -> pd.DataFrame:
        if self.cache_path.exists():
            logger.info(f"Loading cached data from {self.cache_path}")
            df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            if len(df) > 100:
                return df

        import yfinance as yf
        logger.info("Downloading price data from yfinance...")
        frames = {}
        for name, ticker in TICKERS.items():
            try:
                data = yf.download(ticker, start=self.start, end=self.end,
                                   progress=False, auto_adjust=True)
                if len(data) > 0:
                    frames[name] = data["Close"]
                    logger.info(f"  {name} ({ticker}): {len(data)} rows")
                else:
                    logger.warning(f"  {name} ({ticker}): no data returned")
            except Exception as e:
                logger.warning(f"  {name} ({ticker}): failed - {e}")

        df = pd.DataFrame(frames)
        df = df.ffill(limit=5).dropna()
        df.index.name = "date"

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path)
        logger.info(f"Saved {len(df)} rows to {self.cache_path}")
        return df


class VMDDecomposer:
    """Rolling-window VMD to prevent temporal leakage."""

    def __init__(self, K: int = 5, alpha: int = 2000, tau: float = 0.0,
                 tol: float = 1e-7, max_iter: int = 500, rolling_window: int = 252):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.rolling_window = rolling_window

    def decompose_series(self, signal: np.ndarray) -> np.ndarray:
        """Decompose a single series using rolling-window VMD.
        Returns: (K, T) array of modes.
        """
        from vmdpy import VMD

        T = len(signal)
        modes = np.zeros((self.K, T))
        win = self.rolling_window

        # For the first window, decompose the available data
        for t in range(T):
            start = max(0, t - win + 1)
            segment = signal[start:t + 1]
            if len(segment) < 2 * self.K:
                # Too short for VMD, use simple approach
                modes[0, t] = segment[-1]
                continue
            try:
                u, _, _ = VMD(segment, self.alpha, self.tau, self.K, 0, 1, self.tol)
                # Take the last value of each mode
                for k in range(self.K):
                    modes[k, t] = u[k, -1]
            except Exception:
                # Fallback: assign to first mode
                modes[0, t] = segment[-1]

        return modes

    def decompose_all(self, df: pd.DataFrame,
                      cache_path: Optional[str] = None) -> np.ndarray:
        """Decompose all variables. Returns: (num_vars, K, T) array."""
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached VMD modes from {cache_path}")
            return np.load(cache_path)

        num_vars = df.shape[1]
        T = df.shape[0]
        all_modes = np.zeros((num_vars, self.K, T))

        for i, col in enumerate(df.columns):
            logger.info(f"  VMD decomposing {col} ({i+1}/{num_vars})...")
            signal = df[col].values.astype(np.float64)
            all_modes[i] = self.decompose_series(signal)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_modes)
            logger.info(f"Saved VMD modes to {cache_path}")

        return all_modes


class VMDDecomposerFast:
    """Batch VMD: decompose full series once (non-rolling). Faster but has leakage.
    Use for quick experiments; VMDDecomposer for final results."""

    def __init__(self, K: int = 5, alpha: int = 2000, tau: float = 0.0,
                 tol: float = 1e-7):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol

    def decompose_all(self, df: pd.DataFrame,
                      cache_path: Optional[str] = None) -> np.ndarray:
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached VMD modes from {cache_path}")
            return np.load(cache_path)

        from vmdpy import VMD

        num_vars = df.shape[1]
        T = df.shape[0]
        all_modes = np.zeros((num_vars, self.K, T))

        for i, col in enumerate(df.columns):
            logger.info(f"  VMD decomposing {col} ({i+1}/{num_vars})...")
            signal = df[col].values.astype(np.float64)
            try:
                u, _, _ = VMD(signal, self.alpha, self.tau, self.K, 0, 1, self.tol)
                all_modes[i] = u
            except Exception as e:
                logger.warning(f"  VMD failed for {col}: {e}")
                all_modes[i, 0] = signal

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_modes)

        return all_modes


def compute_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Compute forward log returns: log(p_{t+h}/p_t)."""
    returns = np.full(len(prices), np.nan)
    returns[:len(prices) - horizon] = np.log(prices[horizon:] / prices[:len(prices) - horizon])
    return returns


def compute_mode_correlation_graph(modes: np.ndarray, band_idx: int,
                                   window: int = 60) -> np.ndarray:
    """Compute correlation matrix for a specific VMD band across variables.
    modes: (num_vars, K, T). Returns: (num_vars, num_vars) correlation matrix.
    """
    band_data = modes[:, band_idx, -window:]  # (num_vars, window)
    corr = np.corrcoef(band_data)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    return np.abs(corr)


class CopperDataset(Dataset):
    """Dataset for VMD-MFGNN: sliding window of decomposed modes."""

    def __init__(self, modes: np.ndarray, prices: pd.DataFrame,
                 lookback: int = 60, horizons: List[int] = [1, 5, 10, 22],
                 start_idx: int = 0, end_idx: Optional[int] = None,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        """
        modes: (num_vars, K, T)
        prices: DataFrame with copper as first column
        """
        self.lookback = lookback
        self.horizons = horizons
        self.num_vars, self.K, self.T = modes.shape
        end_idx = end_idx or self.T

        # Compute targets: forward returns of copper (index 0)
        copper_prices = prices.iloc[:, 0].values
        self.targets = {}
        for h in horizons:
            self.targets[h] = compute_returns(copper_prices, h)

        # Determine valid indices (have full lookback + max horizon)
        max_h = max(horizons)
        self.indices = []
        for t in range(max(lookback, start_idx), min(end_idx, self.T - max_h)):
            if not any(np.isnan(self.targets[h][t]) for h in horizons):
                self.indices.append(t)

        # Normalize modes (fit on provided stats or compute)
        modes_2d = modes.reshape(-1, self.T)  # (num_vars*K, T)
        if mean is None:
            train_slice = modes_2d[:, start_idx:end_idx]
            self.mean = train_slice.mean(axis=1, keepdims=True)
            self.std = train_slice.std(axis=1, keepdims=True) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.modes_norm = (modes_2d - self.mean) / self.std
        self.modes_norm = self.modes_norm.reshape(self.num_vars, self.K, self.T)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]
        # X: (lookback, K, num_vars)
        x = self.modes_norm[:, :, t - self.lookback:t]  # (num_vars, K, lookback)
        x = x.transpose(2, 1, 0)  # (lookback, K, num_vars)

        # Y: (num_horizons,)
        y = np.array([self.targets[h][t] for h in self.horizons])

        return torch.FloatTensor(x), torch.FloatTensor(y)


class RawPriceDataset(Dataset):
    """Dataset for baselines that use raw prices (no VMD)."""

    def __init__(self, prices: pd.DataFrame, lookback: int = 60,
                 horizons: List[int] = [1, 5, 10, 22],
                 start_idx: int = 0, end_idx: Optional[int] = None,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        self.lookback = lookback
        self.horizons = horizons
        values = prices.values  # (T, num_vars)
        T, self.num_vars = values.shape
        end_idx = end_idx or T

        copper_prices = values[:, 0]
        self.targets = {}
        for h in horizons:
            self.targets[h] = compute_returns(copper_prices, h)

        max_h = max(horizons)
        self.indices = []
        for t in range(max(lookback, start_idx), min(end_idx, T - max_h)):
            if not any(np.isnan(self.targets[h][t]) for h in horizons):
                self.indices.append(t)

        if mean is None:
            train_slice = values[start_idx:end_idx]
            self.mean = train_slice.mean(axis=0)
            self.std = train_slice.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.values_norm = (values - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]
        x = self.values_norm[t - self.lookback:t]  # (lookback, num_vars)
        y = np.array([self.targets[h][t] for h in self.horizons])
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_datasets(config: dict) -> Dict:
    """Orchestrate data download, VMD decomposition, and dataset creation."""
    dc = config["data"]
    vc = config["vmd"]

    # Download
    downloader = DataDownloader(dc["start_date"], dc["end_date"])
    prices = downloader.download()

    # Ensure column order matches VARIABLE_NAMES
    available = [v for v in VARIABLE_NAMES if v in prices.columns]
    prices = prices[available]
    logger.info(f"Using {len(available)} variables: {available}")

    # Date-based split indices
    dates = prices.index
    train_end = pd.Timestamp(dc["train_end"])
    val_end = pd.Timestamp(dc["val_end"])
    train_idx = int((dates <= train_end).sum())
    val_idx = int((dates <= val_end).sum())
    logger.info(f"Split: train={train_idx}, val={val_idx-train_idx}, test={len(dates)-val_idx}")

    # VMD decomposition (use fast version for initial experiments)
    decomposer = VMDDecomposerFast(K=vc["K"], alpha=vc["alpha"],
                                    tau=vc["tau"], tol=vc["tol"])
    modes = decomposer.decompose_all(prices, cache_path="data/vmd_modes.npy")

    lookback = dc["lookback"]
    horizons = dc["horizons"]

    # VMD datasets
    train_ds = CopperDataset(modes, prices, lookback, horizons,
                             start_idx=0, end_idx=train_idx)
    val_ds = CopperDataset(modes, prices, lookback, horizons,
                           start_idx=train_idx, end_idx=val_idx,
                           mean=train_ds.mean, std=train_ds.std)
    test_ds = CopperDataset(modes, prices, lookback, horizons,
                            start_idx=val_idx,
                            mean=train_ds.mean, std=train_ds.std)

    # Raw price datasets (for baselines)
    raw_train = RawPriceDataset(prices, lookback, horizons,
                                start_idx=0, end_idx=train_idx)
    raw_val = RawPriceDataset(prices, lookback, horizons,
                              start_idx=train_idx, end_idx=val_idx,
                              mean=raw_train.mean, std=raw_train.std)
    raw_test = RawPriceDataset(prices, lookback, horizons,
                               start_idx=val_idx,
                               mean=raw_train.mean, std=raw_train.std)

    bs = config["training"]["batch_size"]
    return {
        "prices": prices,
        "modes": modes,
        "variable_names": available,
        "num_vars": len(available),
        "num_modes": vc["K"],
        "horizons": horizons,
        # VMD dataloaders
        "train_loader": DataLoader(train_ds, batch_size=bs, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=bs),
        "test_loader": DataLoader(test_ds, batch_size=bs),
        # Raw dataloaders
        "raw_train_loader": DataLoader(raw_train, batch_size=bs, shuffle=True),
        "raw_val_loader": DataLoader(raw_val, batch_size=bs),
        "raw_test_loader": DataLoader(raw_test, batch_size=bs),
        # Datasets (for direct access)
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "raw_train_ds": raw_train,
        "split_indices": {"train": train_idx, "val": val_idx},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils import load_config
    config = load_config("configs/default.yaml")
    data = create_datasets(config)
    print(f"Train samples: {len(data['train_ds'])}")
    print(f"Val samples: {len(data['val_ds'])}")
    print(f"Test samples: {len(data['test_ds'])}")
    x, y = data["train_ds"][0]
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
