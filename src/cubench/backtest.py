"""Transaction-cost-adjusted backtest — vol-scaled strategy (sigma_target = 15% annualized),
cost curve at cost_bps in {0,1,2,5,10,20,50}, DSR + PBO (CSCV S=16) via pypbo with a
hand-derived DSR cross-check.

Full spec: docs/cubench_implementation_plan.md Section 4.4, 4.3 (DSR/PBO).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as _stats

SIGMA_TARGET_ANNUAL = 0.15
COST_BPS_GRID = [0, 1, 2, 5, 10, 20, 50]
GROSS_SHARPE_BUG_THRESHOLD = 1.5
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Strategy construction
# ---------------------------------------------------------------------------

def vol_scaled_weights(median_return_signal: np.ndarray, vol_forecast_annualized: np.ndarray,
                        sigma_target: float = SIGMA_TARGET_ANNUAL) -> np.ndarray:
    """w_t = sign(qhat_0.5) * min(1, sigma_target / sigmahat_t)."""
    sig = np.asarray(median_return_signal, dtype=float)
    vol = np.asarray(vol_forecast_annualized, dtype=float)
    vol = np.clip(vol, 1e-6, None)
    scale = np.minimum(1.0, sigma_target / vol)
    return np.sign(sig) * scale


def strategy_returns(weights: np.ndarray, realized_returns: np.ndarray,
                      cost_bps: float) -> tuple:
    """Returns (net_returns, turnover_series) for a single cost level.
    Cost charged as cost_bps/1e4 * |w_t - w_{t-1}| per rebalance, applied to
    the day's realized return."""
    w = np.asarray(weights, dtype=float)
    r = np.asarray(realized_returns, dtype=float)
    gross = w * r
    turnover = np.abs(np.diff(w, prepend=w[0] if len(w) else 0.0))
    cost = (cost_bps / 1e4) * turnover
    net = gross - cost
    return net, turnover


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(np.min(dd))


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.mean(r > 0))


def payoff_asymmetry(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(np.mean(wins) / abs(np.mean(losses)))


def cost_curve(weights: np.ndarray, realized_returns: np.ndarray,
               cost_bps_grid: list = None) -> dict:
    """Sharpe / turnover / drawdown / hit-rate / payoff-asymmetry at each cost
    level. Applies the pre-committed calibration check: gross Sharpe > 1.5
    is flagged as a suspected bug."""
    cost_bps_grid = cost_bps_grid or COST_BPS_GRID
    curve = {}
    gross_sharpe = None
    for c in cost_bps_grid:
        net, turnover = strategy_returns(weights, realized_returns, c)
        sr = sharpe_ratio(net)
        row = {
            "sharpe": sr,
            "mean_daily_turnover": float(np.mean(turnover)) if len(turnover) else float("nan"),
            "max_drawdown": max_drawdown(net),
            "hit_rate": hit_rate(net),
            "payoff_asymmetry": payoff_asymmetry(net),
            "n_obs": int(len(net)),
        }
        curve[str(c)] = row
        if c == 0:
            gross_sharpe = sr
    flagged = bool(gross_sharpe is not None and np.isfinite(gross_sharpe)
                   and gross_sharpe > GROSS_SHARPE_BUG_THRESHOLD)
    return {
        "cost_curve": curve,
        "gross_sharpe": gross_sharpe,
        "gross_sharpe_suspected_bug": flagged,
        "gross_sharpe_bug_threshold": GROSS_SHARPE_BUG_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# DSR: hand-derived closed form + pypbo cross-check
# ---------------------------------------------------------------------------

def _psr_hand(sharpe: float, T: int, skew: float, kurtosis: float, target_sharpe: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio, hand-derived closed form (Bailey & Lopez de
    Prado 2012). Matches pypbo.psr's formula, implemented independently."""
    denom = math.sqrt(max(1.0 - skew * sharpe + (kurtosis - 1) / 4.0 * sharpe ** 2, 1e-12))
    value = (sharpe - target_sharpe) * math.sqrt(T - 1) / denom
    return float(_stats.norm.cdf(value))


def _expected_max_hand(N: int) -> float:
    """Expected max of N iid standard normals (Bailey & Lopez de Prado 2014
    closed-form approximation), hand-derived, matches pypbo.expected_max."""
    euler_gamma = 0.5772156649015329
    return float((1 - euler_gamma) * _stats.norm.ppf(1 - 1.0 / N)
                 + euler_gamma * _stats.norm.ppf(1 - math.exp(-1) / N))


def dsr_hand(test_sharpe: float, sharpe_std: float, N: int, T: int,
             skew: float, kurtosis: float) -> float:
    """Deflated Sharpe Ratio, hand-implemented closed form."""
    target_sharpe = sharpe_std * _expected_max_hand(N)
    return _psr_hand(test_sharpe, T, skew, kurtosis, target_sharpe)


def validate_dsr_against_pypbo(test_sharpe: float = 0.6, sharpe_std: float = 0.3,
                                N: int = 33, T: int = 750, skew: float = -0.2,
                                kurtosis: float = 4.5) -> dict:
    """Runs both the hand implementation and pypbo's, asserts agreement to
    1e-6, per the plan's 'do not trust unvalidated third-party numerics'
    standing practice."""
    hand = dsr_hand(test_sharpe, sharpe_std, N, T, skew, kurtosis)
    try:
        import pypbo
        lib = float(pypbo.dsr(test_sharpe, sharpe_std, N, T, skew, kurtosis))
        agree = bool(abs(hand - lib) < 1e-6)
        return {"hand": hand, "pypbo": lib, "abs_diff": abs(hand - lib),
                "agree_to_1e-6": agree, "pypbo_available": True}
    except Exception as e:
        return {"hand": hand, "pypbo": None, "abs_diff": None,
                "agree_to_1e-6": None, "pypbo_available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# PBO via CSCV (S=16 partitions), pypbo if available else hand-implemented
# ---------------------------------------------------------------------------

def cscv_pbo_hand(returns_matrix: np.ndarray, S: int = 16) -> dict:
    """Hand-implemented Combinatorially Symmetric Cross-Validation PBO.

    returns_matrix: (T, N) array — N candidate "trials" (e.g. N seeds or N
    ablation rungs of a strategy), T time-ordered return observations.
    Splits T into S contiguous blocks; for every way of choosing S/2 blocks
    as the in-sample (IS) set and the complementary S/2 as OOS, ranks
    trials by IS Sharpe, finds the OOS rank of the IS-best trial, converts
    to a logit; PBO = fraction of logits <= 0 (OOS underperforms median)."""
    R = np.asarray(returns_matrix, dtype=float)
    T, N = R.shape
    if S % 2 != 0:
        raise ValueError("S must be even")
    block_size = T // S
    if block_size < 2:
        raise ValueError("Too few observations for S blocks")
    blocks = [R[i * block_size:(i + 1) * block_size] for i in range(S)]
    half = S // 2
    logits = []
    for is_idx in itertools.combinations(range(S), half):
        oos_idx = [i for i in range(S) if i not in is_idx]
        is_data = np.concatenate([blocks[i] for i in is_idx], axis=0)
        oos_data = np.concatenate([blocks[i] for i in oos_idx], axis=0)
        is_sharpe = np.array([sharpe_ratio(is_data[:, j]) for j in range(N)])
        oos_sharpe = np.array([sharpe_ratio(oos_data[:, j]) for j in range(N)])
        if not np.any(np.isfinite(is_sharpe)):
            continue
        best_j = int(np.nanargmax(is_sharpe))
        rank_oos = _stats.rankdata(oos_sharpe)[best_j]
        omega = rank_oos / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logit = math.log(omega / (1 - omega))
        logits.append(logit)
    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0)) if len(logits) else float("nan")
    return {"pbo": pbo, "n_combinations": len(logits), "S": S, "N_trials": N,
            "logits": logits.tolist(), "likely_overfit": bool(np.isfinite(pbo) and pbo > 0.5)}


def cscv_pbo(returns_matrix: np.ndarray, S: int = 16, prefer_pypbo: bool = True) -> dict:
    """Wrapper: use pypbo.pbo if importable, else fall back to the hand
    implementation. Both use S=16 combinatorially-symmetric partitions."""
    if prefer_pypbo:
        try:
            import pandas as pd
            import pypbo as _pypbo
            R = np.asarray(returns_matrix, dtype=float)
            df = pd.DataFrame(R)

            def sharpe_metric(x):
                return sharpe_ratio(x.values if hasattr(x, "values") else x)

            result = _pypbo.pbo(df, S=S, metric_func=sharpe_metric, threshold=0.0)
            if hasattr(result, "pbo"):
                pbo_val = float(result.pbo)
            elif isinstance(result, (tuple, list)):
                pbo_val = float(result[0])
            else:
                pbo_val = float(result)
            return {"pbo": pbo_val, "S": S, "N_trials": R.shape[1],
                    "likely_overfit": bool(np.isfinite(pbo_val) and pbo_val > 0.5),
                    "source": "pypbo"}
        except Exception as e:
            fallback = cscv_pbo_hand(returns_matrix, S=S)
            fallback["source"] = "hand_fallback"
            fallback["pypbo_error"] = str(e)
            return fallback
    return {**cscv_pbo_hand(returns_matrix, S=S), "source": "hand_fallback"}
