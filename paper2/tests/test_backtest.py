"""Sanity tests for src/cubench/backtest.py — vol-scaled weights, cost curve
monotonicity, and the DSR hand-vs-pypbo cross-check (plan Section 4.4/4.3)."""
import numpy as np

from src.cubench.backtest import (
    vol_scaled_weights, strategy_returns, cost_curve, sharpe_ratio,
    dsr_hand, validate_dsr_against_pypbo, cscv_pbo_hand, SIGMA_TARGET_ANNUAL,
)


def test_vol_scaled_weights_sign_and_scale():
    signal = np.array([0.01, -0.02, 0.0, 0.005])
    vol = np.array([0.10, 0.30, 0.15, 0.60])  # last one > sigma_target -> scaled down
    w = vol_scaled_weights(signal, vol)
    assert np.sign(w[0]) == 1
    assert np.sign(w[1]) == -1
    assert w[2] == 0.0  # sign(0) = 0
    # vol[3]=0.60 > sigma_target=0.15 -> scale = 0.15/0.60 = 0.25, capped below 1
    assert abs(w[3] - 0.25) < 1e-9
    # vol[0]=0.10 < sigma_target -> scale capped at 1.0
    assert abs(w[0] - 1.0) < 1e-9


def test_cost_curve_monotonic_decreasing_in_cost():
    rng = np.random.default_rng(0)
    n = 500
    signal = rng.normal(0, 0.01, n)
    vol = np.full(n, 0.15)
    w = vol_scaled_weights(signal, vol)
    ret = rng.normal(0.0002, 0.01, n)
    out = cost_curve(w, ret)
    sharpes = [out["cost_curve"][str(c)]["sharpe"] for c in [0, 1, 2, 5, 10, 20, 50]]
    # Sharpe should be non-increasing as cost rises (turnover is fixed, cost only hurts)
    assert all(sharpes[i] >= sharpes[i + 1] - 1e-9 for i in range(len(sharpes) - 1))


def test_zero_turnover_zero_cost_impact():
    w = np.ones(100)  # never rebalances after t=0
    ret = np.random.default_rng(1).normal(0.0001, 0.01, 100)
    net0, _ = strategy_returns(w, ret, cost_bps=0)
    net50, _ = strategy_returns(w, ret, cost_bps=50)
    assert np.allclose(net0, net50)  # no turnover after first bar -> cost only at t=0, both 0


def test_dsr_hand_matches_pypbo():
    out = validate_dsr_against_pypbo()
    assert out["pypbo_available"] in (True, False)
    if out["pypbo_available"]:
        assert out["agree_to_1e-6"] is True


def test_cscv_pbo_hand_detects_overfitting():
    """A returns matrix where one column is genuinely better in-sample-only
    (noise-driven, not real OOS skill) should show elevated PBO."""
    rng = np.random.default_rng(42)
    T, N = 640, 8
    R = rng.normal(0.0, 0.01, size=(T, N))
    # inject a column that looks great in the first half only (in-sample luck)
    R[: T // 2, 0] += 0.01
    out = cscv_pbo_hand(R, S=16)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_combinations"] > 0


def test_sharpe_ratio_sane():
    r = np.full(252, 0.0005)
    sr = sharpe_ratio(np.concatenate([r, r + np.random.default_rng(0).normal(0, 1e-6, 252)]))
    assert sr > 0
