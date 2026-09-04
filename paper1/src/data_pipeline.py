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


class EMDDecomposer:
    """True daily-refit rolling-window EMD (Empirical Mode Decomposition) --
    a drop-in alternative decomposer to `VMDDecomposer`, mirroring its exact
    interface and leakage-safety discipline (see item 2 of the robustness
    experiments: alternative decomposition method comparison).

    At every trading day t, EMD is run from scratch (via `PyEMD.EMD`, the
    `EMD-signal` package's sifting-based implementation) on the window
    signal[max(0, t-window+1) : t+1] -- strictly data up to and including day
    t, never anything after it -- and only the LAST point of each resulting
    IMF (intrinsic mode function) is kept as that day's feature value. This
    is architecturally identical to `VMDDecomposer`'s daily-refit rolling
    scheme, just swapping the decomposition algorithm (EMD's data-driven
    recursive sifting instead of VMD's variational/ADMM optimization).

    Mode-count caveat (the key structural difference from VMD): EMD does not
    take a fixed number-of-modes parameter -- the sifting process determines
    how many IMFs a given window naturally decomposes into (typically
    ~log2(window_length), often 6-9 IMFs for a 252-day window), and this can
    vary slightly from window to window. To keep a fixed-width (K, T) output
    array (so it slots into `CopperDataset`/the rest of the pipeline
    unchanged), the K highest-frequency IMFs (the *first* K, since PyEMD
    returns IMFs ordered highest-frequency first) are kept in slots
    [0, K-1). If EMD produces MORE than K IMFs for a window (including its
    final residual/trend), everything beyond the first K-1 IMFs is SUMMED
    into slot K-1 (mirroring VMD's mode-K-1 acting as the coarsest/lowest-
    frequency band) so no information is silently dropped. If EMD produces
    FEWER than K IMFs, the unused trailing slots are zero-padded. This
    zero-padding/summing choice, and the resulting effective mode count, must
    be reported explicitly in any comparison against VMD -- it is NOT the
    same guarantee VMD gives (VMD is asked for exactly K bands and returns
    exactly K).

    Because IMF ordering from independent per-window sifts is not guaranteed
    to align in any particular frequency sense across days the way VMD's
    omega-sort re-establishes, no re-sorting is applied here beyond PyEMD's
    own convention (highest-frequency IMF first, which is stable in practice
    because sifting always peels off the fastest-oscillating component
    first) -- this mirrors VMD's mode-identity caveat but resolved by PyEMD's
    own algorithmic convention rather than an explicit post-hoc frequency
    sort.
    """

    def __init__(self, K: int = 5, rolling_window: int = 252,
                 max_imfs: Optional[int] = None):
        self.K = K
        self.rolling_window = rolling_window
        # PyEMD's EMD(max_imf=...) caps sifting early; None (PyEMD default:
        # unlimited, i.e. -1) lets it decompose naturally and we truncate/sum
        # down to K afterward as documented above.
        self.max_imfs = max_imfs

    def decompose_series(self, signal: np.ndarray) -> np.ndarray:
        """Decompose a single series using rolling-window EMD, refit at
        EVERY trading day. Returns: (K, T) array of modes.
        """
        from PyEMD import EMD

        T = len(signal)
        modes = np.zeros((self.K, T))
        win = self.rolling_window

        for t in range(T):
            start = max(0, t - win + 1)
            segment = signal[start:t + 1].astype(np.float64)  # strictly data[start..t]
            if len(segment) < 8:
                # Too short for a meaningful sift (EMD needs enough points to
                # find extrema); fall back to holding the raw value in slot 0,
                # same convention as VMDDecomposer's short-segment fallback.
                modes[0, t] = segment[-1]
                continue
            try:
                emd = EMD()
                if self.max_imfs is not None:
                    imfs = emd.emd(segment, max_imf=self.max_imfs)
                else:
                    imfs = emd.emd(segment)
                n_imfs = imfs.shape[0]
                if n_imfs == 0:
                    modes[0, t] = segment[-1]
                    continue
                n_keep = min(self.K, n_imfs)
                for k in range(n_keep - 1):
                    modes[k, t] = imfs[k, -1]
                # Last kept slot absorbs the k-th IMF plus everything beyond
                # (including EMD's implicit trend/residual) so no information
                # is silently dropped when n_imfs > K.
                modes[n_keep - 1, t] = imfs[n_keep - 1:, -1].sum()
                # If n_imfs < K, slots [n_imfs, K) stay at their zero-init.
            except Exception as e:
                logger.warning(f"EMD failed at t={t}: {e}")
                modes[0, t] = segment[-1]

        return modes

    def decompose_all(self, df: pd.DataFrame,
                      cache_path: Optional[str] = None) -> np.ndarray:
        """Decompose all variables. Returns: (num_vars, K, T) array."""
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached EMD modes from {cache_path}")
            return np.load(cache_path)

        num_vars = df.shape[1]
        T = df.shape[0]
        all_modes = np.zeros((num_vars, self.K, T))

        for i, col in enumerate(df.columns):
            logger.info(f"  EMD decomposing {col} ({i+1}/{num_vars})...")
            signal = df[col].values.astype(np.float64)
            all_modes[i] = self.decompose_series(signal)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_modes)
            logger.info(f"Saved EMD modes to {cache_path}")

        return all_modes


class CEEMDANDecomposer:
    """EXPERIMENTAL / OPTIONAL, NOT part of the primary decomposition
    comparison -- see item 2's "if also feasible" clause.

    CEEMDAN (Complete Ensemble EMD with Adaptive Noise) is implemented here
    for completeness (via `PyEMD.CEEMDAN`), with the SAME true-daily-refit
    rolling-window interface as `EMDDecomposer`/`VMDDecomposer`. It is
    deliberately NOT wired into the main comparison run by default because of
    a real, disclosed compute-cost constraint: CEEMDAN is an ensemble method
    that reruns EMD `trials` times per window (each with different injected
    noise) and averages -- even at a reduced `trials=20` (vs PyEMD's default
    100), that is a ~20x per-window cost multiplier over plain EMD. Applied
    at every trading day (true daily refit, matching this project's
    leakage-safety discipline -- no shortcuts to a cheaper expanding/refit-
    interval scheme are taken here, to keep the comparison apples-to-apples
    with VMDDecomposer/EMDDecomposer), this would push one-time CPU cost from
    EMD's/VMD's tens of minutes into many hours to ~1+ day for the full
    ~4,000-day x 8-variable dataset on a single Colab CPU core -- not a
    realistic addition to an already multi-section robustness notebook on a
    free/standard Colab time budget.

    ADDITIONAL, EMPIRICALLY-DISCOVERED cost factor (found during this task's
    local verification, not merely theorized): PyEMD's CEEMDAN defaults to
    `parallel=True`, spawning a fresh `multiprocessing` worker pool PER
    CALL. On Windows (spawn-based process creation, no fork), that per-call
    pool-spawn overhead dominates wall-clock time at small window sizes far
    more than the `trials` multiplier alone would predict -- an initial
    local leakage-test attempt at T=120/rolling_window=40/trials=5 did not
    finish within 6 minutes of wall-clock time before being killed, an order
    of magnitude slower than a naive trials-multiplier estimate would
    suggest. `parallel=False` (set below) avoids that per-window pool-spawn
    entirely, trading it for a straightforwardly serial (still `trials`-x
    slower than plain EMD, but no per-call multiprocessing overhead added on
    top) cost -- confirmed to actually finish locally (see this task's
    report for the real, timed local verification result with
    `parallel=False`). Any future large-compute-budget run of this class
    SHOULD use `parallel=False` for this reason, unless run on a
    fork-capable OS (Linux, which Colab's runtime is) where the per-call
    spawn cost is much lower -- even there, benchmark a handful of windows
    before committing to a full run.

    A SECOND empirically-discovered issue (see `__init__`'s docstring note):
    PyEMD's CEEMDAN does not seed its noise RNG by default, so naively
    calling it gives non-reproducible results run-to-run for the identical
    window -- fixed here via `noise_seed(random_seed_base + t)` per window.

    It is included in this file, tested for correctness/leakage-safety at
    small scale (see the local verification in this task's report), and
    left available for a future run with a larger compute budget, rather
    than silently omitted.
    """

    def __init__(self, K: int = 5, rolling_window: int = 252, trials: int = 20,
                 parallel: bool = False, random_seed_base: int = 0):
        self.K = K
        self.rolling_window = rolling_window
        self.trials = trials
        # False by default -- see the class docstring's empirical finding on
        # Windows' per-call multiprocessing-pool-spawn overhead. Colab runs
        # on Linux (fork-based), where True may be faster; benchmark first.
        self.parallel = parallel
        # SECOND empirically-discovered issue (also found during this task's
        # local verification): PyEMD's CEEMDAN does NOT seed its internal
        # noise RNG by default -- each `CEEMDAN().ceemdan(...)` call draws
        # fresh, unseeded ensemble noise, so decomposing the SAME window
        # twice gives DIFFERENT results, and a leakage test comparing two
        # runs would show "differences" at every t purely from this
        # nondeterminism, not from any actual future-data leak (this is
        # exactly what the first attempt at this test showed before the fix
        # below). Fixed by seeding `noise_seed(random_seed_base + t)`
        # per-day BEFORE each window's decomposition -- deterministic per
        # (t, random_seed_base), so the same day's window always produces
        # the same result, and re-running decompose_series is reproducible
        # (a real requirement for any published result using this
        # decomposer, not just for the leakage test).
        self.random_seed_base = random_seed_base

    def decompose_series(self, signal: np.ndarray) -> np.ndarray:
        from PyEMD import CEEMDAN

        T = len(signal)
        modes = np.zeros((self.K, T))
        win = self.rolling_window

        for t in range(T):
            start = max(0, t - win + 1)
            segment = signal[start:t + 1].astype(np.float64)
            if len(segment) < 8:
                modes[0, t] = segment[-1]
                continue
            try:
                ceemdan = CEEMDAN(trials=self.trials, parallel=self.parallel)
                ceemdan.noise_seed(self.random_seed_base + t)
                imfs = ceemdan.ceemdan(segment)
                n_imfs = imfs.shape[0]
                if n_imfs == 0:
                    modes[0, t] = segment[-1]
                    continue
                n_keep = min(self.K, n_imfs)
                for k in range(n_keep - 1):
                    modes[k, t] = imfs[k, -1]
                modes[n_keep - 1, t] = imfs[n_keep - 1:, -1].sum()
            except Exception as e:
                logger.warning(f"CEEMDAN failed at t={t}: {e}")
                modes[0, t] = segment[-1]

        return modes

    def decompose_all(self, df: pd.DataFrame,
                      cache_path: Optional[str] = None) -> np.ndarray:
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached CEEMDAN modes from {cache_path}")
            return np.load(cache_path)

        num_vars = df.shape[1]
        T = df.shape[0]
        all_modes = np.zeros((num_vars, self.K, T))
        for i, col in enumerate(df.columns):
            logger.info(f"  CEEMDAN decomposing {col} ({i+1}/{num_vars})...")
            signal = df[col].values.astype(np.float64)
            all_modes[i] = self.decompose_series(signal)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_modes)
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


def build_decomposed_modes(prices: pd.DataFrame, vc: dict, dc: dict,
                            cache_path: str = "data/vmd_modes.npy",
                            meta_path: Optional[str] = None,
                            debug_fast: bool = False,
                            method: str = "vmd") -> np.ndarray:
    """Build (or load from cache) decomposed modes for all variables, for
    whichever decomposition `method` is selected. This generalizes the
    original VMD-only `build_vmd_modes` (kept below as a thin backward-
    compatible wrapper calling this with method="vmd") to also support "emd"
    and "ceemdan" (item 2: alternative decomposition method comparison) --
    both mirror VMD's true-daily-refit rolling-window leakage-safety
    discipline exactly (see `EMDDecomposer`/`CEEMDANDecomposer` docstrings).

    method: "vmd" (default, unchanged behavior), "emd", or "ceemdan".
    debug_fast: only meaningful for method=="vmd" (routes to
        `VMDDecomposerFast`, the leaky full-series batch decomposer, for
        quick throwaway debugging only). Ignored for "emd"/"ceemdan" (which
        have no fast/leaky variant implemented -- always leakage-safe).

    Caching: results are cached to `cache_path` (a .npy) plus a JSON metadata
    sidecar recording the split dates, decomposition method + its parameters,
    and a sha256 hash of the input price array. If any of those change (e.g.
    a fresh data pull with different prices, a different K, or a different
    `method`), the cache is treated as stale and recomputed -- so a prior
    run's cache can never be silently reused against new/different
    data/method. Callers doing a method/K comparison should pass distinct
    `cache_path`s per (method, K) combination (see `experiments.py`) so each
    variant's cache persists independently rather than invalidating siblings.
    """
    cache_path = Path(cache_path)
    meta_path = Path(meta_path) if meta_path is not None else \
        cache_path.with_name(cache_path.stem + "_meta.json")

    price_values = prices.values.astype(np.float64)
    data_hash = _hash_price_array(price_values)
    rolling_window = vc.get("rolling_window", 252)

    meta = {
        "method": method,
        "K": vc["K"],
        "rolling_window": rolling_window,
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
    if method == "vmd":
        meta.update({
            "alpha": vc["alpha"], "tau": vc["tau"], "tol": vc["tol"],
            "decomposer": "VMDDecomposerFast" if debug_fast else "VMDDecomposer",
            "debug_fast": debug_fast,
        })
    elif method == "ceemdan":
        meta["trials"] = vc.get("ceemdan_trials", 20)

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                cached_meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            cached_meta = {}
        # Backward compatibility: a cache written by the OLD build_vmd_modes
        # (before this function existed) has no "method" key at all. Treat
        # such a cache as matching a method=="vmd" request as long as every
        # OTHER key matches, so pre-existing caches (including ones restored
        # from Drive from before this change) are not needlessly invalidated
        # -- avoids silently forcing an expensive recompute + falsifying any
        # "warm cache" wall-clock estimate that assumed the old cache still
        # hits.
        compare_meta = cached_meta
        if method == "vmd" and "method" not in cached_meta:
            compare_meta = dict(cached_meta)
            compare_meta["method"] = "vmd"
        if compare_meta == meta:
            logger.info(f"Loading cached {method.upper()} modes from {cache_path} "
                        f"(data hash + params match)")
            return np.load(cache_path)
        else:
            logger.warning(f"{method.upper()} cache at {cache_path} is stale "
                            f"(data hash or params changed) -- recomputing")

    if method == "vmd":
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
    elif method == "emd":
        decomposer = EMDDecomposer(K=vc["K"], rolling_window=rolling_window)
        modes = decomposer.decompose_all(prices, cache_path=None)
    elif method == "ceemdan":
        decomposer = CEEMDANDecomposer(K=vc["K"], rolling_window=rolling_window,
                                        trials=vc.get("ceemdan_trials", 20))
        modes = decomposer.decompose_all(prices, cache_path=None)
    else:
        raise ValueError(f"Unknown decomposition method: {method!r} "
                          f"(expected 'vmd', 'emd', or 'ceemdan')")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, modes)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved {method.upper()} modes to {cache_path} and metadata to {meta_path}")

    return modes


def build_vmd_modes(prices: pd.DataFrame, vc: dict, dc: dict,
                     cache_path: str = "data/vmd_modes.npy",
                     meta_path: Optional[str] = None,
                     debug_fast: bool = False) -> np.ndarray:
    """Backward-compatible VMD-only wrapper around `build_decomposed_modes`
    (method="vmd"). Every existing call site (`create_datasets`, the archived
    Colab notebooks, etc.) keeps working byte-for-byte unchanged. See
    `build_decomposed_modes` for the generalized (VMD/EMD/CEEMDAN) version
    used by the new decomposition-comparison experiment.
    """
    return build_decomposed_modes(prices, vc, dc, cache_path=cache_path,
                                   meta_path=meta_path, debug_fast=debug_fast,
                                   method="vmd")


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


def create_datasets(config: dict, debug_fast: bool = False,
                     decomposition_method: str = "vmd",
                     modes_cache_path: Optional[str] = None) -> Dict:
    """Orchestrate data download, decomposition, and dataset creation.

    debug_fast: passed through to `build_decomposed_modes`. Leave False
    (default) for any real experiment/reported result -- it selects the
    leakage-safe, true daily-refit rolling-window VMD (`VMDDecomposer`).
    Only set True for quick, throwaway local debugging. Ignored unless
    decomposition_method=="vmd".

    decomposition_method: "vmd" (default, unchanged behavior), "emd", or
    "ceemdan" -- selects which decomposer `build_decomposed_modes` uses (see
    item 2: alternative decomposition method comparison). config["vmd"]["K"]
    is still the requested band/mode count for whichever method is chosen
    (see EMDDecomposer/CEEMDANDecomposer docstrings for how K interacts with
    a data-driven method's natural mode count).

    modes_cache_path: override for the decomposed-modes cache path. Defaults
    to "data/vmd_modes.npy" (unchanged) when None AND decomposition_method
    == "vmd" (byte-for-byte backward compatible); for any other method, or
    when explicitly given (e.g. by a K-sweep/decomposition-comparison driver
    that needs one cache file per (method, K) combination so variants don't
    invalidate each other's cache), this path is used instead.
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

    # Decomposition: leakage-safe, true daily-refit rolling-window VMD by
    # default (see VMDDecomposer / build_decomposed_modes docstrings).
    # debug_fast=True opts into the leaky full-series batch VMD decomposer
    # for quick iteration only -- never for reported results.
    if modes_cache_path is None:
        modes_cache_path = "data/vmd_modes.npy" if decomposition_method == "vmd" \
            else f"data/{decomposition_method}_modes_K{vc['K']}.npy"
    modes = build_decomposed_modes(prices, vc, dc, cache_path=modes_cache_path,
                                    debug_fast=debug_fast,
                                    method=decomposition_method)

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
