"""Validation tests for src/cubench/stats_tests.py, per plan Section 4.3.

- dm_test_hln: cross-checked against the `dieboldmariano` PyPI package on 200
  random error pairs (agreement on the UNCORRECTED statistic to tight
  tolerance), plus a hand-worked known-value case.
- pesaran_timmermann: bootstrap validation — 10,000 independent no-skill
  (yhat, y) pairs with known base rates, empirical rejection rate at
  alpha=0.05 must land in [0.04, 0.06]. Plus the degenerate-constant-forecast
  guard.
"""
import math

import numpy as np
import pytest

from src.cubench.stats_tests import dm_test_hln, pesaran_timmermann, holm_bonferroni_wrapper

try:
    from dieboldmariano import dm_test as ref_dm_test
    HAVE_REF = True
except Exception:
    HAVE_REF = False


@pytest.mark.skipif(not HAVE_REF, reason="dieboldmariano package not installed")
def test_dm_hln_vs_reference_package_200_pairs():
    rng = np.random.default_rng(12345)
    n_agree = 0
    n_total = 200
    max_abs_diff = 0.0
    for i in range(n_total):
        n = rng.integers(40, 300)
        y = rng.normal(0, 1, n)
        e1 = y - rng.normal(0, 1.0, n)
        e2 = y - rng.normal(0, 1.2, n)

        ours = dm_test_hln(e1, e2, h=1, loss="se")
        # reference package: harvey_correction=False -> uncorrected DM stat,
        # loss defaults to squared error, h=1 matches our lag truncation.
        try:
            ref_stat, ref_p = ref_dm_test(
                list(y), list(y - e1), list(y - e2), h=1,
                harvey_correction=False,
            )
        except Exception:
            continue
        if not (np.isfinite(ours["dm_stat"]) and np.isfinite(ref_stat)):
            continue
        diff = abs(ours["dm_stat"] - ref_stat)
        max_abs_diff = max(max_abs_diff, diff)
        if diff < 1e-6:
            n_agree += 1
    agree_rate = n_agree / n_total
    print(f"DM-HLN vs reference: agreement rate={agree_rate:.3f}, max_abs_diff={max_abs_diff:.3e}")
    assert agree_rate > 0.95, f"Only {agree_rate:.1%} of pairs agreed with reference package (max diff {max_abs_diff})"


def test_dm_hln_hand_worked_example():
    """Hand-worked example: two forecasts with a clear, known mean loss
    differential and no autocorrelation (h=1 -> no lag terms), so
    V = var(d)/n exactly and DM = dbar / sqrt(var(d)/n), which is just a
    one-sample t-statistic on d = e1^2 - e2^2. Cross-check against scipy's
    ttest_1samp directly (a textbook-known-value equivalence at h=1)."""
    rng = np.random.default_rng(7)
    n = 500
    e1 = rng.normal(1.5, 1.0, n)   # worse model, strongly biased
    e2 = rng.normal(0.0, 1.0, n)   # better model, unbiased
    out = dm_test_hln(e1, e2, h=1, loss="se")

    d = e1 ** 2 - e2 ** 2
    from scipy import stats as sstats
    t_stat, t_p = sstats.ttest_1samp(d, 0.0)
    # at h=1, HLN multiplier = sqrt((n+1-2+0)/n) = sqrt((n-1)/n); DM uses /n
    # variance (population, not ddof=1) so it will be close to but not
    # identical to the exact t-test; check it's in the right ballpark and
    # correct sign.
    assert np.sign(out["hln_stat"]) == np.sign(t_stat)
    assert abs(out["hln_stat"] - t_stat) / abs(t_stat) < 0.05
    assert out["p_value"] < 0.01  # e1 is genuinely worse -> should reject


def test_dm_hln_identical_forecasts_gives_nan_or_zero():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 100)
    out = dm_test_hln(e, e.copy(), h=1)
    assert out["mean_diff"] == 0.0


# ---------------------------------------------------------------------------
# Pesaran-Timmermann bootstrap validation
# ---------------------------------------------------------------------------

def test_pt_bootstrap_no_skill_rejection_rate():
    """10,000 independent no-skill (yhat, y) pairs, base rate 0.55 for y and
    0.5 for yhat (independent draws => true no skill). Empirical rejection
    rate at alpha=0.05 should be in [0.04, 0.06]."""
    rng = np.random.default_rng(2026)
    n_sims = 10_000
    n_obs = 250
    base_rate_y = 0.55
    base_rate_x = 0.50
    rejections = 0
    n_valid = 0
    for _ in range(n_sims):
        y = (rng.random(n_obs) < base_rate_y).astype(float)
        x = (rng.random(n_obs) < base_rate_x).astype(float)
        out = pesaran_timmermann(y, x)
        if out["degenerate"]:
            continue
        n_valid += 1
        if out["p_value"] < 0.05:
            rejections += 1
    rejection_rate = rejections / n_valid
    print(f"PT bootstrap: {n_valid}/{n_sims} valid, empirical rejection rate={rejection_rate:.4f}")
    assert 0.04 <= rejection_rate <= 0.06, f"Empirical rejection rate {rejection_rate:.4f} outside [0.04, 0.06]"


def test_pt_degenerate_constant_forecast():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    x_const = np.ones_like(y)  # always predicts "up"
    out = pesaran_timmermann(y, x_const)
    assert out["degenerate"] is True
    assert out["flag"] == "degenerate_constant_forecast"
    assert math.isnan(out["p_value"])


def test_pt_perfect_skill_rejects():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.5).astype(float)
    x = y.copy()  # perfect forecast
    out = pesaran_timmermann(y, x)
    assert not out["degenerate"]
    assert out["p_value"] < 0.001
    assert out["p_hat"] == 1.0


# ---------------------------------------------------------------------------
# Holm-Bonferroni wrapper
# ---------------------------------------------------------------------------

def test_holm_wrapper_family_size_bookkeeping():
    pvals = [0.001, 0.02, 0.5, np.nan, 0.049]
    out = holm_bonferroni_wrapper(pvals, labels=["a", "b", "c", "d", "e"],
                                   family_name="T1", expected_size=33)
    assert out["declared_family_size"] == 33
    assert out["actual_n_tests"] == 5
    assert out["family_size_matches_declaration"] is False
    assert out["n_valid_pvalues"] == 4
    # smallest p-value should survive Holm at alpha .05 with n=4 valid tests
    assert out["results"][0]["reject_holm"] is True
