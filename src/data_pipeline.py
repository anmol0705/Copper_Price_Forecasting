import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Locked 8-variable scope (see vmd-mfgnn-protocol/SKILL.md), all sourced via
# Yahoo Finance only. FRED and BDI are explicitly out of scope.
#
# Note on zinc/nickel (dropped 2026-08-07): there is no standalone
# COMEX/NYMEX-style "=F" futures contract for zinc or nickel on Yahoo
# Finance (they trade on the LME, which Yahoo does not mirror as a tradable
# future). "ZNC=F" / "NI=F" used previously do not resolve to real
# instruments. As a substitute, the NASDAQ Commodity sub-indices
# "^NQCIZNER" (zinc) and "^NQCINIER" (nickel) were used as the closest
# available proxy. On a live production data-download run on 2026-08-07
# (covering 2010-2025), both of those sub-index tickers were confirmed
# dead/delisted on Yahoo Finance -- yfinance raises
# `YFPricesMissingError('possibly delisted; no price data found')` for
# both. With no further viable Yahoo Finance substitute for zinc/nickel,
# they have been dropped entirely from the locked scope. This is a
# deliberate, disclosed reduction from the original 10-variable locked
# scope to an 8-variable scope: copper, aluminum, gold, oil, dxy, sp500,
# vix, us10y. See vmd-mfgnn-protocol/SKILL.md for the authoritative,
# up-to-date scope table.
TICKERS = {
    "copper": "HG=F",
    "aluminum": "ALI=F",
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
                    # yfinance can return MultiIndex columns (even for a
                    # single ticker, depending on version/call shape), in
                    # which case data["Close"] comes back as a 1-column
                    # DataFrame instead of a Series. Flatten defensively so
                    # `frames` always holds genuine Series -- an un-flattened
                    # DataFrame value here is what causes pd.DataFrame(frames)
                    # below to raise a cryptic "If using all scalar values,
                    # you must pass an index" error.
                    close = data["Close"]
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    frames[name] = close
                    valid = close.dropna()
                    first_date = valid.index.min() if len(valid) > 0 else None
                    last_date = valid.index.max() if len(valid) > 0 else None
                    logger.info(f"  {name} ({ticker}): {len(data)} rows, "
                                f"valid range {first_date} to {last_date}")
                else:
                    logger.warning(f"  {name} ({ticker}): no data returned")
            except Exception as e:
                logger.warning(f"  {name} ({ticker}): failed - {e}")

        missing = [f"{name} ({TICKERS[name]})" for name in TICKERS if name not in frames]
        if missing:
            raise ValueError(
                f"DataDownloader.download() failed to retrieve usable data for "
                f"{len(missing)}/{len(TICKERS)} ticker(s): {', '.join(missing)}. "
                f"Refusing to build a partial/malformed price DataFrame -- fix "
                f"or remove the failing ticker(s) from TICKERS and re-run."
            )

        df = pd.DataFrame(frames)
        df = df.ffill(limit=5).dropna()
        df.index.name = "date"

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path)
        logger.info(f"Saved {len(df)} rows to {self.cache_path}")
        return df


class VMDDecomposer:
    """True daily-refit rolling-window VMD -- leakage-safe by construction and
    genuinely daily-varying (not a piecewise-constant staircase like
    `VMDDecomposerExpanding`).

    At every trading day t, VMD is refit from scratch on the window
    signal[max(0, t-window+1) : t+1] -- i.e. strictly data up to and
    including day t, never anything after it -- and only the LAST point of
    each resulting mode is kept as that day's feature value. Because this
    happens at every single t (not every `refit_interval` days), each mode's
    assigned series is one genuine, independently-computed value per day,
    matching the daily resolution of the raw price data instead of holding a
    stale value constant across a ~21-day block.

    Mode ordering caveat: VMD's ADMM optimization is not guaranteed to
    converge to the same mode ordering across independent fits (mode k on
    day t and mode k on day t+1 could correspond to different frequency
    bands). To keep mode identity consistent over time, each day's modes are
    re-sorted by ascending center frequency (`omega`, returned by
    `vmdpy.VMD` alongside the modes) before being stored, so mode 0 is
    always the lowest-frequency band and mode K-1 the highest, every day.
    """

    def __init__(self, K: int = 5, alpha: int = 2000, tau: float = 0.0,
                 tol: float = 1e-7, max_iter: int = 500, rolling_window: int = 252):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.rolling_window = rolling_window

    def decompose_series(self, signal: np.ndarray) -> np.ndarray:
        """Decompose a single series using rolling-window VMD, refit at
        EVERY trading day. Returns: (K, T) array of modes.
        """
        from vmdpy import VMD

        T = len(signal)
        modes = np.zeros((self.K, T))
        win = self.rolling_window

        for t in range(T):
            start = max(0, t - win + 1)
            segment = signal[start:t + 1]  # strictly data[start..t], no future
            if len(segment) < 2 * self.K:
                # Too short for VMD, use simple approach
                modes[0, t] = segment[-1]
                continue
            try:
                u, _, omega = VMD(segment, self.alpha, self.tau, self.K, 0, 1, self.tol)
                # omega has shape (n_iters_used, K); the final row holds the
                # converged center frequencies for this fit. Sort ascending
                # so mode 0 is always lowest-frequency, mode K-1 highest,
                # consistently across independent daily fits.
                final_omega = omega[-1, :]
                order = np.argsort(final_omega)
                # Take the last value of each mode, applying the frequency-
                # sorted order so mode identity is stable day to day.
                for k in range(self.K):
                    modes[k, t] = u[order[k], -1]
            except Exception as e:
                logger.warning(f"VMD failed at t={t}: {e}")
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


class VMDDecomposerExpanding:
    """Leakage-safe expanding-window VMD with periodic refit.

    NO LONGER THE DEFAULT (see `VMDDecomposer` above, which now is). This
    class is kept on disk for reference/comparison, but `build_vmd_modes`'s
    default (non-debug) path no longer calls it. Reason: refitting only
    every `refit_interval` days and holding the last-point mode values
    constant across the whole block produces piecewise-constant "staircase"
    features -- a 60-day lookback window built from this decomposer contains
    only ~3-4 distinct values per (variable, mode), not real daily-varying
    data. A benchmark showed true daily-refit VMD (`VMDDecomposer`, rolling
    252-day window) costs only ~30-40 minutes one-time CPU for the full
    ~4,000-day x 10-variable dataset (cached afterward), so the
    "computationally infeasible" rationale below no longer holds and this
    class is superseded. (Note: the benchmark's "~4,000-day x 10-variable"
    figure predates the 2026-08-07 zinc/nickel scope reduction to 8
    variables; the per-variable cost is unaffected.)

    True daily-refit VMD (see `VMDDecomposer` above) is leakage-free -- each
    day's modes come from a window that ends exactly at that day.

    This class instead refits only every `refit_interval` trading days
    (locked at 21, i.e. ~1 trading month per vmd-mfgnn-protocol/SKILL.md). At
    each refit day R we run VMD once on the EXPANDING window signal[0:R+1]
    -- all history available *as of* day R, and nothing after it. We take
    the last-point mode values of that decomposition and hold them
    (carry-forward) as the assigned modes for every day from R up to (but
    not including) the next refit day R + refit_interval.

    This guarantees zero future leakage: the modes assigned to day t were
    computed from a decomposition window signal[0:R+1] where R <= t, so no
    index > t is ever used. The cost is T/refit_interval VMD calls instead of
    T, making it tractable, at the price of the assigned modes being
    piecewise-constant within each 21-day block instead of updated daily.
    """

    def __init__(self, K: int = 5, alpha: int = 2000, tau: float = 0.0,
                 tol: float = 1e-7, max_iter: int = 500, refit_interval: int = 21):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.refit_interval = refit_interval

    def decompose_series(self, signal: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Decompose a single series using expanding-window VMD, refit every
        `refit_interval` days. Returns: ((K, T) modes array, list of refit day indices).
        """
        from vmdpy import VMD

        T = len(signal)
        modes = np.zeros((self.K, T))
        refit_days = list(range(0, T, self.refit_interval))

        last_values = np.zeros(self.K)
        for R in refit_days:
            segment = signal[:R + 1]  # expanding window: strictly data[0..R], no future
            if len(segment) < 2 * self.K:
                last_values = np.zeros(self.K)
                last_values[0] = segment[-1]
            else:
                try:
                    u, _, _ = VMD(segment, self.alpha, self.tau, self.K, 0, 1, self.tol)
                    last_values = u[:, -1]
                except Exception as e:
                    logger.warning(f"VMD failed for window at refit day R={R}: {e}")
                    last_values = np.zeros(self.K)
                    last_values[0] = segment[-1]

            block_end = min(R + self.refit_interval, T)
            modes[:, R:block_end] = last_values[:, None]

        return modes, refit_days

    def decompose_all(self, df: pd.DataFrame) -> np.ndarray:
        """Decompose all variables. Returns: (num_vars, K, T) array.
        Caching is handled by the caller (`build_vmd_modes`), since the cache
        needs a data hash + metadata sidecar, not just a plain .npy dump.
        """
        num_vars = df.shape[1]
        T = df.shape[0]
        all_modes = np.zeros((num_vars, self.K, T))

        for i, col in enumerate(df.columns):
            logger.info(f"  Expanding-window VMD decomposing {col} ({i + 1}/{num_vars})...")
            signal = df[col].values.astype(np.float64)
            all_modes[i], _ = self.decompose_series(signal)

        return all_modes


class VMDDecomposerFast:
    """DEBUG-ONLY: batch VMD, decomposes the full series once (non-rolling).
    Fast, but the decomposition of the training period is computed jointly
    with validation/test-period data, i.e. it LEAKS future (including
    test-set) data into training-period samples. Must never be the default
    path for real experiments/reported results -- only reachable via the
    explicit `debug_fast=True` argument to `build_vmd_modes` /
    `create_datasets`. Use `VMDDecomposer` (the default, true daily-refit
    rolling-window decomposer) for anything that touches reported metrics."""

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


def _hash_price_array(values: np.ndarray) -> str:
    """SHA-256 hash of the raw price array bytes, used to invalidate a stale
    VMD cache if the underlying data pull changes."""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_vmd_modes(prices: pd.DataFrame, vc: dict, dc: dict,
                     cache_path: str = "data/vmd_modes.npy",
                     meta_path: Optional[str] = None,
                     debug_fast: bool = False) -> np.ndarray:
    """Build (or load from cache) VMD-decomposed modes for all variables.

    debug_fast: if True, uses `VMDDecomposerFast` (full-series batch VMD).
    This is fast but LEAKS future/test-period data into training-period
    decompositions -- it exists only for quick, throwaway debugging/iteration
    and must NEVER be the default used for real experiments or reported
    metrics. The default (debug_fast=False) uses the leakage-safe, true
    daily-refit `VMDDecomposer` (rolling window of `vc["rolling_window"]`
    (default 252) trading days, refit at every single day -- see its
    docstring for why this replaced the old `VMDDecomposerExpanding`
    carry-forward approach, which produced piecewise-constant "staircase"
    features).

    Caching: results are cached to `cache_path` (a .npy) plus a JSON metadata
    sidecar recording the split dates, VMD parameters, refit cadence, and a
    sha256 hash of the input price array. If any of those change (e.g. a
    fresh data pull with different prices, or different K/alpha), the cache
    is treated as stale and recomputed -- so a prior run's cache can never be
    silently reused against new/different data.
    """
    cache_path = Path(cache_path)
    meta_path = Path(meta_path) if meta_path is not None else \
        cache_path.with_name(cache_path.stem + "_meta.json")

    price_values = prices.values.astype(np.float64)
    data_hash = _hash_price_array(price_values)
    rolling_window = vc.get("rolling_window", 252)

    meta = {
        "K": vc["K"],
        "alpha": vc["alpha"],
        "tau": vc["tau"],
        "tol": vc["tol"],
        "rolling_window": rolling_window,
        "decomposer": "VMDDecomposerFast" if debug_fast else "VMDDecomposer",
        "debug_fast": debug_fast,
        "split_dates": {
            "start_date": dc.get("start_date"),
            "train_end": dc.get("train_end"),
            "val_end": dc.get("val_end"),
            "end_date": dc.get("end_date"),
        },
        "variables": list(prices.columns),
        "num_rows": int(len(prices)),
        "data_hash": data_hash,
    }

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            cached_meta = {}
        if cached_meta == meta:
            logger.info(f"Loading cached VMD modes from {cache_path} "
                        f"(data hash + params match, debug_fast={debug_fast})")
            return np.load(cache_path)
        else:
            logger.warning(f"VMD cache at {cache_path} is stale (data hash or "
                            f"params changed) -- recomputing")

    if debug_fast:
        logger.warning("debug_fast=True: using VMDDecomposerFast (full-series "
                        "batch VMD). This LEAKS future/test data into training "
                        "decompositions -- for quick debugging only, never for "
                        "reported results.")
        decomposer = VMDDecomposerFast(K=vc["K"], alpha=vc["alpha"],
                                        tau=vc["tau"], tol=vc["tol"])
        modes = decomposer.decompose_all(prices, cache_path=None)
    else:
        decomposer = VMDDecomposer(K=vc["K"], alpha=vc["alpha"],
                                    tau=vc["tau"], tol=vc["tol"],
                                    max_iter=vc.get("max_iter", 500),
                                    rolling_window=rolling_window)
        modes = decomposer.decompose_all(prices, cache_path=None)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, modes)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved VMD modes to {cache_path} and metadata to {meta_path}")

    return modes


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


def create_datasets(config: dict, debug_fast: bool = False) -> Dict:
    """Orchestrate data download, VMD decomposition, and dataset creation.

    debug_fast: passed through to `build_vmd_modes`. Leave False (default)
    for any real experiment/reported result -- it selects the leakage-safe,
    true daily-refit rolling-window VMD (`VMDDecomposer`). Only set True for
    quick, throwaway local debugging.
    """
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

    # VMD decomposition: leakage-safe, true daily-refit rolling-window VMD
    # by default (see VMDDecomposer / build_vmd_modes docstrings).
    # debug_fast=True opts into the leaky full-series batch decomposer for
    # quick iteration only -- never for reported results.
    modes = build_vmd_modes(prices, vc, dc, cache_path="data/vmd_modes.npy",
                             debug_fast=debug_fast)

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
