"""Local (no-GPU, CPU-only) leakage-safety verification for EMDDecomposer.

Mirrors the causal-perturbation discipline already used elsewhere in this
project (see STATUS.md's VMD leakage spot-checks): decompose the same base
signal twice, once unperturbed and once with a large perturbation injected
strictly AFTER a cutoff day, and confirm the two runs' per-day mode values
are bit-identical for every day <= cutoff. If EMDDecomposer were leaking
future data, perturbing day cutoff+1..end would change the modes assigned to
days <= cutoff too, and this test would fail.

Run: python tests/test_emd_leakage.py   (from paper1/, with PyEMD installed)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_pipeline import EMDDecomposer, CEEMDANDecomposer  # noqa: E402


def check_leakage_safety(dec, label: str, T: int, cutoff: int) -> bool:
    rng = np.random.default_rng(42)
    t = np.arange(T)
    signal = (100 + 0.05 * t
              + 3 * np.sin(2 * np.pi * t / 20)
              + 1.5 * np.sin(2 * np.pi * t / 60)
              + rng.normal(0, 0.3, T))

    perturbed = signal.copy()
    perturbed[cutoff + 1:] += 1000.0  # large, impossible-to-miss perturbation

    print(f"\n--- {label} (T={T}, cutoff={cutoff}) ---")
    print("Decomposing original signal...")
    modes_orig = dec.decompose_series(signal)
    print("Decomposing perturbed signal...")
    modes_pert = dec.decompose_series(perturbed)

    # Days <= cutoff must be BIT-IDENTICAL between the two runs -- their
    # decomposition windows never touch the perturbed region (window is at
    # most `rolling_window` long and ends at day t <= cutoff, so it only
    # ever spans [t-window+1, t], entirely within the unperturbed prefix for
    # t <= cutoff).
    prefix_orig = modes_orig[:, :cutoff + 1]
    prefix_pert = modes_pert[:, :cutoff + 1]

    max_abs_diff = np.max(np.abs(prefix_orig - prefix_pert))
    identical = np.array_equal(prefix_orig, prefix_pert)

    print(f"Prefix (days 0..{cutoff}) max abs diff between "
          f"original/perturbed runs: {max_abs_diff:.10e}")
    print(f"Prefix bit-identical: {identical}")

    # Sanity check the perturbation actually reaches the tail (proves the
    # perturbation itself is real and the decomposer is sensitive to it,
    # i.e. this isn't a vacuous pass from a broken decomposer that ignores
    # its input).
    tail_orig = modes_orig[:, cutoff + 5:cutoff + 15]
    tail_pert = modes_pert[:, cutoff + 5:cutoff + 15]
    tail_differs = not np.allclose(tail_orig, tail_pert, atol=1e-6)
    print(f"Tail (days > cutoff) differs between runs (expected True): {tail_differs}")

    ok = identical and tail_differs
    print(f"=== {label}:", "PASS -- leakage-safe "
          "(prefix unaffected by future perturbation, tail is affected)"
          if ok else "FAIL", "===")
    return ok


def main():
    K = 5
    ok_emd = check_leakage_safety(
        EMDDecomposer(K=K, rolling_window=60), "EMDDecomposer", T=400, cutoff=250)

    # CEEMDAN at a small scale (trials=5, tiny window/T) so this local check
    # stays cheap (~seconds) -- CEEMDAN's ensemble cost is the whole reason
    # it's not run at full scale in the notebook (see CEEMDANDecomposer's
    # docstring). This still exercises the identical leakage-safety code
    # path (only `trials` and CEEMDAN's own algorithm differ from EMD).
    ok_ceemdan = check_leakage_safety(
        CEEMDANDecomposer(K=K, rolling_window=40, trials=5), "CEEMDANDecomposer",
        T=120, cutoff=70)

    ok = ok_emd and ok_ceemdan
    print("\n=== OVERALL RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
