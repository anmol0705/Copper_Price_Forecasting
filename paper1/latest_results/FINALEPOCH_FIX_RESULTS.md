# Final-Epoch Re-Diagnosis: Results and Verdict

**Status:** definitive. Closes the last open item in `ROBUSTNESS_ANALYSIS_FINAL.md` §9 ("If any
compute is spent at all, the highest-value target is ... the three artefact cells").

**Data analysed:**
- `paper1/latest_results/decomposition_finalepoch_fix/` — 1 cell (EMD), `decomposition_results.jsonl`, 1 checkpoint, 8 `.npy`
- `paper1/latest_results/multiseed_finalepoch_fix/` — 4 cells, `multiseed_results.jsonl`, 4 checkpoints, 32 `.npy`

**Method:** every number below was recomputed from the raw artefacts. Test metrics were
independently recomputed from the `.npy` prediction/ground-truth pairs using this repo's own
`src/utils.compute_metrics` formulae (§6). Epoch counts were read from the checkpoint `epoch`
field and from the new `final_epoch` diagnostic field. Cross-experiment statistics were
recomputed from all four original `.jsonl` files plus their 19 checkpoints.

**Epoch convention.** Two conventions are in play and they differ by one. `final_epoch` and the
checkpoint `epoch` field are **0-based loop indices** (`i`). `main.tex` and
`ROBUSTNESS_ANALYSIS_FINAL.md` prose speak of **epochs of training** (`E = i + 1`), because
`fit()` completes a full training pass before each evaluation. Both are given in every table
below. All draft LaTeX in §8 uses `E`.

---

## 1. Executive summary

| # | Question | Answer |
|---|---|---|
| 1 | Did `diagnose_final_epoch` fire for all three originally-broken cells? | **Yes, all three.** `diagnosed_final_epoch: true`, `final_epoch` = 20, 21, 21 (i.e. 21, 22, 22 epochs of training), all far above `min_epochs_for_valid_diagnosis = 5`. The fallback path did **not** fire anywhere. |
| 2 | Did it fire for the two bonus re-runs? | **Yes.** `final_epoch` = 44 and 67 (45 and 68 epochs). |
| 3 | **Does EMD still avoid the collapse once genuinely trained?** | **No. EMD collapses, unambiguously and harder than VMD.** 0 of 5 bands survive; `frac_near_uniform = 1.0` in every band; gradient reaching the embeddings is **exactly 0.0**; embedding RMS is $1.1$–$1.3\times10^{-3}$ of the Xavier reference. The earlier "EMD is NOT COLLAPSED" verdict is confirmed as a pure early-stopping artefact. |
| 4 | Does the headline cross-experiment finding survive? | **Yes, unchanged in rank, sharper in one respect.** On the corrected 19-cell set (artefact cells *replaced*, not added): Spearman $\rho = -0.990$ (cosine) and $-0.993$ (RMS ratio) — **bit-identical to the published values** — with the low-epoch anchors removed and the epoch range narrowed from 1–37 to 5–37. |
| 5 | Is 19-of-19 collapse now supportable? | **Yes, with one qualifier that must travel with it: 19 / 19 runs COLLAPSED when diagnosed at $\ge 5$ epochs of training (range 5–37).** The previous "16 of 16 (three unusable)" formulation is retired. The claim is *not* "collapse at every epoch count" — the same three runs read NOT COLLAPSED at their 1–2-epoch snapshots. See §5.2. |
| 6 | Does anything about the multi-seed RQ1 comparison change? | **No.** Both re-run cells reproduced their originals' accuracy to $\sim10^{-9}$; the paired tests are literally unchanged. Only the two collapse verdicts change (NOT COLLAPSED → COLLAPSED). |
| 7 | Recompute-from-raw verification | **Passes.** All 5 cells × 4 horizons × 5 metrics reproduce the jsonl; max relative deviation $2.2\times10^{-5}$ (in MAPE), max relative deviation in RMSE/MAE/DA/R² $\le 1.2\times10^{-5}$. |

---

## 2. Did the fix fire? (Task 1)

`src/experiments.py:324` swaps `trainer.final_state` in for the diagnostic only when
`diagnose_final_epoch and final_epoch is not None and final_epoch >= min_epochs_for_valid_diagnosis`
(default 5), and records the outcome in `diagnostic.diagnosed_final_epoch` /
`diagnostic.final_epoch`. `test_metrics` are computed *before* the swap and `best_state` is
restored *after* it (`experiments.py:295, 343-347`), so accuracy numbers are untouched.

| cell | originally broken? | `diagnosed_final_epoch` | `final_epoch` (i) | epochs trained (E) | ≥ threshold 5? | fallback fired? |
|---|---|---|---:|---:|---|---|
| `emd` | **yes** | **true** | **20** | **21** | yes | no |
| `full_model` seed 44 | **yes** | **true** | **21** | **22** | yes | no |
| `pooled_graph_matched_dim` seed 45 | **yes** | **true** | **21** | **22** | yes | no |
| `full_model` seed 45 | no (bonus) | true | 44 | 45 | yes | no |
| `pooled_graph_matched_dim` seed 44 | no (bonus) | true | 67 | 68 | yes | no |

**All three originally-broken cells were genuinely diagnosed from final-epoch weights.** No cell
fell back to `best_state`.

### 2.1 The three target cells reproduced their original runs; the two bonus cells did not

This matters: it establishes that the new final-epoch diagnostic describes the *same training
trajectory* whose best-val snapshot produced the original artefact, and is not a different run.

| cell | original `last_epoch` | new `final_epoch` | original test avg_MSE | new test avg_MSE | relative diff |
|---|---:|---:|---:|---:|---:|
| `emd` | 20 | **20** | 0.002725858736 | 0.002725858954 | $8.0\times10^{-8}$ |
| `full_model` s44 | 21 | **21** | 0.002695500902 | 0.002695497744 | $1.2\times10^{-6}$ |
| `pooled` s45 | 21 | **21** | 0.002748647363 | 0.002748647035 | $1.2\times10^{-7}$ |
| `full_model` s45 | 24 | 44 | 0.002687279870 | 0.002772547727 | 3.17 % |
| `pooled` s44 | 36 | 67 | 0.002931958596 | 0.002792653031 | 4.99 % |

For the EMD cell, the re-run's saved `best_state` embedding RMS is **0.230800262093544**, identical
to the original run's to all 15 printed digits. The prediction arrays are not *bit*-identical
(GPU non-determinism at the $10^{-7}$ level), but the three target cells retraced the same
early-stopping trajectory to the same epoch. The two bonus cells diverged and trained far
longer — see §5.2, where that divergence is put to use.

Ground-truth arrays across all five new cells are **bit-identical** to each other and to the
original run's (`np.array_equal` True, shape (984,) at every horizon), so all comparisons here
are on the same held-out data.

---

## 3. The real collapse-diagnostic numbers for all 5 new cells (Task 2)

`rms_ratio` is against the Xavier reference RMS 0.2885 (the `adjacency_state` diagnostic);
`rms/init` is against the run's own same-seed random initialisation (the `adjacency_movement`
diagnostic). `emb_grad` is the mean gradient norm reaching `emb1`/`emb2` over 5 real
forward/backward passes.

| cell | E | verdict | n_ok / n_total | `frac_near_uniform` (all bands) | `rms_ratio` (vs Xavier) | `softmax_std` | **emb_grad** | other_grad | **cos to init** | **rms/init** |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| **`emd`** | 21 | **COLLAPSED** | **0 / 5** | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.09e−3 … 1.34e−3 | 1.9e−8 … 3.2e−8 | **0.0 (exactly)** | 0.1795 | **0.61348** | **1.219e−3** |
| **`full_model` s44** | 22 | **COLLAPSED** | **0 / 5** | 1.0, 1.0, 1.0, 1.0, 1.0 | 7.17e−4 … 8.53e−4 | 8.8e−9 … 1.7e−8 | **0.0 (exactly)** | 0.1699 | **0.59727** | **7.706e−4** |
| **`pooled` s45** | 22 | **COLLAPSED** | **0 / 1** | 1.0 | 6.19e−4 | 7.8e−9 | 2.68e−22 | 0.2222 | **0.55505** | **6.540e−4** |
| `full_model` s45 | 45 | COLLAPSED | 0 / 5 | 1.0, 1.0, 1.0, 1.0, 1.0 | 8.66e−11 … 1.53e−10 | **0.0 (exactly)** | **0.0 (exactly)** | 0.2106 | 1.914e−2 | 1.358e−10 |
| `pooled` s44 | 68 | COLLAPSED | 0 / 1 | 1.0 | 2.25e−22 | **0.0 (exactly)** | **0.0 (exactly)** | 0.1923 | 2.254e−14 | 3.151e−22 |

Per-band cosine similarity to initialisation, for the record:

- `emd`: 0.6223, 0.6156, 0.6413, 0.5729, 0.6153 (mean 0.61348)
- `full_model` s44: 0.6340, 0.5824, 0.5922, 0.6094, 0.5684 (mean 0.59727)
- `pooled` s45: 0.5550 (single band)
- `full_model` s45: 0.01859, 0.01801, 0.03031, 0.01562, 0.01318 (mean 0.019141)
- `pooled` s44: 2.254e−14 (single band)

**Every one of the five is COLLAPSED, on both independent criteria** — `rms_ratio < 0.5` *and*
`frac_near_uniform > 0.8` (`src/diagnostics.py:97`). The uniformity criterion is the stronger
statement here: **100 % of surviving adjacency edges sit within tolerance of $1/N$ in every band
of every cell.** `other_grad_norm_mean` is 0.17–0.22 throughout, so the rest of the network is
receiving an ordinary gradient at the same moment the graph branch receives none: this is not a
dead model, it is a dead graph branch inside a live model.

### 3.1 A caution on reading `cosine_similarity` here

`src/diagnostics.py:247` attaches a note to `adjacency_movement`: *"cosine\_similarity near 1.0
… reproduces the 'shrank in place, never rotated' pattern; well below 1.0 indicates genuine
directional learning."* **That note must not be applied at these magnitudes.** Cosine is
computed on the raw embedding vector, and once that vector has shrunk to $10^{-3}$ (or
$10^{-22}$) of its initialisation, its residual direction is dominated by accumulated
floating-point residue, not by learning. The `pooled` s44 cell makes this unmistakable: its
cosine is $2.25\times10^{-14}$ — that is not "fully rotated", it is a numerically zero vector,
and its own gradient is exactly 0.0, which is the definition of *no* learning signal. Cosine
decline is therefore **confounded with shrinkage** and is a monotone *correlate* of collapse
progression, not independent evidence of learning.

**The load-bearing evidence for the collapse verdicts is `frac_near_uniform = 1.0`,
`rms_ratio` $\ll$ 0.5, and `emb_grad = 0`, not the cosine.** The cosine's value in this document
is as the smooth, well-behaved regression variable in §5 (where it must be restricted to
pre-underflow cells for the same reason).

---

## 4. The decisive EMD question (Task 3)

### 4.1 Same run, two snapshots

| EMD (`effective_K = 5`, matched to VMD) | best-validation weights, **E = 1** (as originally reported) | final-epoch weights, **E = 21** (this re-run) |
|---|---|---|
| verdict | **NOT COLLAPSED** — no band shows the pattern | **COLLAPSED** — all bands show the pattern |
| n_ok / n_total | 5 / 5 | **0 / 5** |
| `frac_near_uniform` per band | 0.156, 0.000, 0.156, 0.063, 0.063 | **1.0, 1.0, 1.0, 1.0, 1.0** |
| `rms_ratio` vs Xavier | 0.818, 0.779, 0.802, 0.794, 0.809 | **1.34e−3, 1.24e−3, 1.15e−3, 1.12e−3, 1.09e−3** |
| `softmax_std` | 1.16e−2 … 1.89e−2 | **1.9e−8 … 3.2e−8** |
| `emb_grad_norm_mean` | 2.396e−8 | **0.0 (exactly)** |
| `other_grad_norm_mean` | — | 0.1795 |
| mean cosine to init | 0.99434 | 0.61348 |
| mean rms / init | 0.8158 | 1.219e−3 |

The embedding norm falls by a factor of **669** between the two snapshots of the *same run*, the
adjacency goes from measurably non-uniform to bit-level uniform in every band, and the gradient
reaching the embeddings goes to exactly zero.

### 4.2 EMD versus VMD, both at genuinely-trained weights

| | VMD (E = 11, best-val, unchanged) | EMD (E = 21, final-epoch) |
|---|---|---|
| verdict | COLLAPSED (0/5) | COLLAPSED (0/5) |
| `frac_near_uniform` | 1.0 in all 5 bands | 1.0 in all 5 bands |
| `rms_ratio` vs Xavier | 5.66e−2 … 6.37e−2 | 1.09e−3 … 1.34e−3 |
| `softmax_std` | 7.0e−5 … 8.3e−5 | 1.9e−8 … 3.2e−8 |
| `emb_grad_norm_mean` | 3.741e−14 | 0.0 |
| cos to init | 0.79397 | 0.61348 |
| test avg_MSE | 0.002773 | 0.002726 |

**EMD does not merely also collapse; it is further along the identical trajectory, exactly as
far as its extra 10 epochs of training predict.** Quantitatively: EMD's cosine of 0.61348 sits
**−0.0297** from the value predicted by the previously published 19-cell regression
(`cos = 0.97483 − 0.016583 × i`, predicted 0.6432) — *inside* that fit's own largest residual of
0.0425. The corresponding residuals for `full_model` s44 and `pooled` s45 are **−0.0293** and
**−0.0715**.

### 4.3 Verdict, stated plainly

**The corrected diagnostic confirms the paper's narrative. EMD was falsely flagged as NOT
COLLAPSED purely because its checkpoint had trained for one epoch. Once diagnosed on genuinely
trained weights from the same run, EMD collapses, with every band uniform and exactly zero
gradient reaching the graph embeddings. Nothing genuinely different shows up. There is no
defensible reading in which EMD protects against the collapse.** The decomposition axis, which
previously carried *no* usable evidence, now carries a clean 2-of-2.

### 4.4 The one honest caveat a referee will find

The re-diagnosis is asymmetric by design: `test_metrics` come from `best_state`, the diagnostic
from `final_state`. For these three cells those are different weight sets, so the *deployed*
(best-validation) model is the near-initialisation one, not the collapsed one. This does not
create an escape route, and the paper should say why:

- At **best-validation** weights the graph embeddings are statistically indistinguishable from
  random initialisation (cosine 0.994 to their own init, 82 % of init norm, gradient $2.4\times10^{-8}$).
- At **final-epoch** weights they are uniform with exactly zero gradient.

**Neither state contains learned relational structure.** The early-stopped checkpoint is not a
model whose graph learned something and then stopped; it is a model whose graph never started.
The choice between the two snapshots is a choice between *random* and *uniform*, and the paper's
claim — that the per-band mechanism demonstrates no learned relational structure — holds under
both. We did not measure the final-epoch weights' test accuracy, and do not claim it.

---

## 5. Cross-experiment headline finding (Task 4)

### 5.1 The correct comparison is *replace*, not *add*

The three re-diagnoses are new snapshots of runs already in the 19-cell suite, not new cells.
Pooling all 24 rows would double-count those three runs and, worse, would retain precisely the
E = 1/2/2 anchors the correction exists to remove. The corrected set is therefore **19 cells
with the three artefact rows swapped for their final-epoch re-diagnoses**. The two bonus
re-runs (`full_model` s45, `pooled` s44) are re-runs of configurations already present and are
reported separately in §5.2, not counted as cells.

### 5.2 Corrected 19-cell set

| axis | cell | i | **E** | cos to init | rms / init | emb_grad | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| k_sweep | K=7 | 4 | 5 | 0.92661 | 3.306e−1 | 2.92e−10 | COLLAPSED |
| multiseed | full_model s45 | 4 | 5 | 0.92501 | 3.254e−1 | 2.68e−10 | COLLAPSED |
| loss | mae | 8 | 9 | 0.83735 | 1.133e−1 | 1.17e−11 | COLLAPSED |
| multiseed | full_model s43 | 8 | 9 | 0.83524 | 1.112e−1 | 6.25e−13 | COLLAPSED |
| multiseed | full_model s46 | 8 | 9 | 0.84795 | 1.180e−1 | 8.74e−13 | COLLAPSED |
| multiseed | pooled s42 | 8 | 9 | 0.84407 | 1.185e−1 | 8.89e−15 | COLLAPSED |
| k_sweep | K=9 | 9 | 10 | 0.81359 | 8.375e−2 | 8.55e−14 | COLLAPSED |
| decomposition | vmd | 10 | 11 | 0.79397 | 6.130e−2 | 3.74e−14 | COLLAPSED |
| loss | mse | 10 | 11 | 0.79397 | 6.130e−2 | 3.51e−14 | COLLAPSED |
| multiseed | pooled s43 | 10 | 11 | 0.80579 | 6.183e−2 | 3.75e−16 | COLLAPSED |
| multiseed | full_model s42 | 12 | 13 | 0.75269 | 3.139e−2 | 2.55e−15 | COLLAPSED |
| k_sweep | K=5 | 13 | 14 | 0.73301 | 2.200e−2 | 7.12e−16 | COLLAPSED |
| k_sweep | K=3 | 14 | 15 | 0.71824 | 1.576e−2 | 3.88e−18 | COLLAPSED |
| multiseed | pooled s46 | 14 | 15 | 0.71562 | 1.550e−2 | 1.01e−17 | COLLAPSED |
| multiseed | pooled s44 | 16 | 17 | 0.71565 | 6.864e−3 | 1.93e−20 | COLLAPSED |
| **decomposition** | **emd (final-ep)** | **20** | **21** | **0.61348** | **1.219e−3** | **0.0** | **COLLAPSED** |
| **multiseed** | **full_model s44 (final-ep)** | **21** | **22** | **0.59727** | **7.706e−4** | **0.0** | **COLLAPSED** |
| **multiseed** | **pooled s45 (final-ep)** | **21** | **22** | **0.55505** | **6.540e−4** | **2.68e−22** | **COLLAPSED** |
| loss | huber | 36 | 37 | 0.42030 | 9.385e−8 | 0.0 | COLLAPSED |

**19 / 19 runs COLLAPSED when diagnosed at $\ge 5$ epochs of training (range 5–37).**

State it as *runs*, not as independent observations, and always with the epoch qualifier. The
three re-diagnosed cells are the same three runs whose 1–2-epoch snapshots read NOT COLLAPSED
(§5.5), so the suite does **not** establish "collapse at every epoch count"; it establishes
collapse across 5–37 epochs, and separately establishes — on those same runs — that the
diagnostic reads NOT COLLAPSED at 1–2 epochs because the decay has not yet crossed its
threshold. Both facts are ours and both must travel together. The claim the paper is entitled to
is: *every run in this suite collapses once it has trained for as few as five epochs, and no
configuration axis delays or prevents it.*

### 5.3 Correlations: published 19 vs corrected 19

| statistic | published 19 (artefact cells at best-val) | **corrected 19 (artefact cells at final epoch)** |
|---|---:|---:|
| Spearman $\rho$(epoch, cosine to init) | −0.9899 ($p = 7.5\times10^{-16}$) | **−0.9899 ($p = 7.5\times10^{-16}$)** |
| Spearman $\rho$(epoch, RMS ratio) | −0.9930 ($p = 3.4\times10^{-17}$) | **−0.9930 ($p = 3.4\times10^{-17}$)** |
| Spearman $\rho$(epoch, log₁₀ RMS ratio) | −0.9930 | **−0.9930** |
| Spearman $\rho$(epoch, log₁₀ emb grad) | −0.9607 ($p = 6.8\times10^{-11}$) | **−0.9571 ($p = 1.4\times10^{-10}$)** |
| Pearson $r$(epoch, cosine) | −0.9884 | −0.9823 |
| Linear cosine fit | $0.97483 - 0.016583\,i$, R² = 0.9768, max\|resid\| 0.0425 | $0.96699 - 0.016780\,i$, **R² = 0.9649**, max\|resid\| **0.0596** |
| — same fit in epochs $E$ | $0.99142 - 0.016583\,E$ | $\mathbf{0.98377 - 0.016780\,E}$ |
| log₁₀(RMS ratio) fit | $0.4522 - 0.1852\,i$, R² = 0.9477 | $0.7117 - 0.1957\,i$, **R² = 0.9684** |
| residual-by-axis ANOVA | $F = 0.543$, $p = 0.660$ | $F = 0.791$, $p = 0.518$ |
| residual-by-axis Kruskal–Wallis | $H = 1.694$, $p = 0.638$ | $H = 2.143$, $p = 0.543$ |
| epoch range ($E$) | 1 – 37 | **5 – 37** |
| cosine range | 0.4203 – 0.9943 | **0.4203 – 0.9266** |
| RMS-ratio range | 8.158e−1 – 9.385e−8 | **3.306e−1 – 9.385e−8** |

**The headline finding holds.** The two rank correlations the paper actually quotes are
*numerically identical* to the published values. This is not a coincidence and not an accidental
recomputation of the same set — it is exact by construction, and verified directly. In the
published set the three artefact cells held epoch ranks $\{1, 2.5, 2.5\}$ and cosine ranks
$\{19, 18, 17\}$; in the corrected set they hold epoch ranks $\{16, 17.5, 17.5\}$ and cosine
ranks $\{4, 3, 2\}$. In both cases they form a contiguous block at one extreme of the epoch
ordering paired with the mirror-image contiguous block at the opposite extreme of the cosine
ordering, with no inversions against the other 16 cells and with the within-pair order
(`full_model` s44 above `pooled` s45) preserved on both variables. A rank statistic cannot see
the difference. **That is a stronger result than a numerical coincidence: it means the finding
never depended on the three low-epoch anchors at all** — they were monotone-consistent
before the correction and monotone-consistent after it, at the opposite end of the range.
(The gradient-norm correlation does move slightly, $-0.9607 \to -0.9571$, because three
gradients became exactly 0.0 and introduced ties.) The linear cosine fit's R² softens slightly (0.977 → 0.965) and
its largest residual grows (0.043 → 0.060) because the corrected set is compressed into a
narrower epoch range; the log₁₀(RMS) fit *improves* (0.948 → 0.968). No residual axis effect is
detectable either way.

**Are the three new points outliers?** No. Their residuals against the *original, unmodified*
fit are −0.030, −0.029 and −0.072; the original fit's own largest residual was 0.043. Two of the
three are inside it and the third is 1.7× it, on a fit whose response range is 0.52. They land
where the trend says they should.

### 5.4 One correlation that must NOT be recomputed on the corrected set

$\rho$(epoch, test avg\_MSE) was **+0.456, $p = 0.050$** on the published 19 and computes to
**+0.048, $p = 0.846$** on the corrected 19. **Do not use the corrected value.** For the three
replaced cells, `test_metrics` describe the `best_state` weights while `final_epoch` describes a
different snapshot, so the corrected pairing mixes two weight sets and the statistic is not
well-defined. The published +0.456 is internally consistent (best-val metrics against best-val
epoch) and should stand. This affects the HPO paragraph of `main.tex`, which quotes
$\rho = 0.456$, $p = 0.05$; **that number needs no change**, but a note in the analysis record is
warranted so a later pass does not "helpfully" recompute it.

### 5.5 Two free corroborations from the bonus cells

1. **Two additional same-configuration, same-seed replicate pairs.** `full_model` s45 ran to
   E = 25 then E = 45 on the two occasions, with test avg\_MSE 0.002687 vs 0.002773 (**3.17 %**
   spread); `pooled` s44 ran to E = 37 then E = 68, avg\_MSE 0.002932 vs 0.002793 (**4.99 %**).
   Both sit inside the 7.54 % four-replicate noise floor already reported, and both confirm the
   §3.2 observation that run-to-run nondeterminism perturbs the *epoch count itself* — here by
   20 and 31 epochs.
2. **A within-run, two-snapshot confirmation of the epoch trend** — strictly stronger than the
   across-run replicate check, because configuration, seed *and* trajectory are all held fixed:

   | cell | best-val snapshot | final-epoch snapshot |
   |---|---|---|
   | `emd` | E = 1: cos 0.9943, rms/init 0.8158, grad 2.4e−8, NOT COLLAPSED | E = 21: cos 0.6135, rms/init 1.22e−3, grad 0.0, COLLAPSED |
   | `full_model` s44 | E = 2: cos 0.9828, rms/init 0.6633, grad 1.3e−8, NOT COLLAPSED | E = 22: cos 0.5973, rms/init 7.71e−4, grad 0.0, COLLAPSED |
   | `pooled` s45 | E = 2: cos 0.9811, rms/init 0.6517, grad 3.9e−8, NOT COLLAPSED | E = 22: cos 0.5551, rms/init 6.54e−4, grad 2.7e−22, COLLAPSED |

   Three runs, each observed twice, each moving down the same curve by the amount its extra
   20 epochs predict. The diagnostic verdict is a function of *when you look*, and nothing else.
   The bonus cells extend this to E = 45 (rms/init 1.4e−10) and E = 68 (rms/init 3.2e−22).

---

## 6. Multi-seed / RQ1 comparison (Task 5)

**The paper's existing disclaimer — "their accuracy metrics and the paired tests above are
unaffected" — is verified literally true.** Both cells re-ran to the same final epoch (21 and 21)
and reproduced their original test metrics to $\sim10^{-9}$ absolute:

| cell | h=1 RMSE original / new | h=22 RMSE original / new |
|---|---|---|
| `full_model` s44 | 0.018134132 / 0.018134133 | 0.075818790 / 0.075818790 |
| `pooled` s45 | 0.018355569 / 0.018355569 | 0.077374640 / 0.077374640 |

So the published paired $t$-tests ($p = 0.96, 0.13, 0.88, 0.50$ at $h = 1, 5, 10, 22$) and
Wilcoxon results stand exactly as reported. **The only thing that changes is the two collapse
verdicts**, both NOT COLLAPSED → COLLAPSED, bringing per-seed collapse coverage to 5/5 seeds in
both variants.

Are the two corrected cells consistent with the rest of the multi-seed cells? Yes. All ten
multi-seed cells are now COLLAPSED with 0 bands surviving; the two corrected ones sit at
E = 22 with cosine 0.597 / 0.555 and rms/init 7.7e−4 / 6.5e−4, continuous with the other eight
(E = 5–17, cosine 0.925 → 0.716) and residuals of −0.029 / −0.072 against the published fit.
Nothing about the multi-seed axis distinguishes them beyond training length.

**Robustness check only (not a proposed replacement).** Substituting the two bonus re-runs'
accuracy numbers as well (i.e. all four re-run cells) and re-running the paired tests gives
$p = 0.77, 0.22, 0.75, 0.92$ — still non-significant at every horizon, though the sign of the
difference flips at $h = 10$ and $h = 22$. That instability is the expected consequence of
re-rolling two of five paired observations at a 3–5 % noise level and is further evidence that
this comparison resolves nothing at $n = 5$. **The paper should keep its published $p$-values**
(computed from the original, internally consistent five-seed set) and should not substitute
these.

The capacity caveat is untouched: direct parameter counting on the new checkpoints again gives
`full_model` 1,461,124 and `pooled_graph_matched_dim` 315,530, the same 78.4 % deficit.

---

## 7. Recompute-from-raw verification (Task 6)

Performed on **all five** new cells (not the required minimum of two), at all four horizons, for
all five metrics, using `src/utils.compute_metrics`' exact formulae against the raw `.npy`
prediction/ground-truth pairs (984 test samples per horizon).

| cell | max abs diff RMSE | max abs diff MAE | max abs diff DA | max abs diff R² | avg\_mse diff |
|---|---|---|---|---|---|
| `emd` | 8.1e−10 | 9.3e−10 | **0.0** | 1.2e−7 | 7.3e−12 |
| `full_model` s44 | 4.2e−9 | 0.0 | **0.0** | 1.2e−7 | 1.2e−10 |
| `full_model` s45 | 1.5e−9 | 0.0 | **0.0** | 1.2e−7 | 2.9e−11 |
| `pooled` s44 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| `pooled` s45 | 0.0 | 3.7e−9 | **0.0** | 0.0 | 0.0 |

**Maximum relative deviation across all 5 cells × 4 horizons × 5 metrics: $2.2\times10^{-5}$**
(occurring in MAPE, which is stored at float32 precision in the jsonl). Directional accuracy
matches exactly in all 20 cell-horizon combinations. Every reported test metric in both new
`.jsonl` files is confirmed against the raw arrays. Ground-truth arrays are bit-identical across
all five cells and to the original run's.

Reproduce with the scripts used for this document (kept alongside as
`verify_fix.py` / `replaced19.py` in the session scratchpad; core logic is 40 lines and is
restated in §9).

---

## 8. Draft replacement text for the five data-dependent `\todohl` markers in `main.tex`

`main.tex` was **not modified**. There are 8 `\todohl` markers; lines 59, 754 (venue, repo URL)
and the email placeholder are out of scope here. The five data-dependent ones are at lines
**477, 526, 561, 605, 730**. Each block below gives the exact replacement.

---

### TODO 1 — line 477: the blanket "provisional numbers" banner

**Action: DELETE the entire `\noindent\todohl{...}` block.** It exists only to flag the pending
re-diagnosis, which has now landed; every value in the subsection is final. Nothing replaces it.
The paragraph that follows (`Section~\ref{sec:collapse}'s adjacency-collapse diagnosis...`)
becomes the subsection's opening paragraph.

One consequential edit accompanies the deletion. Later in that same paragraph, replace:

```latex
--- applying the \emph{same} three-part gradient/adjacency diagnostic (embedding
RMS relative to initialization, gradient norm reaching
$\mathbf{E}^{(1)}_k/\mathbf{E}^{(2)}_k$, and cosine similarity against the run's
own random initialization) to every one of the 19 trained cells. All 19 cells
were trained at the tuned configuration of Section~\ref{sec:hpo} with weight
decay $10^{-5}$, and all 19 completed. Two of the five experiments carry defects
we disclose in full below rather than repair, for reasons we state; a third
defect, affecting three individual cells, is the subject of the pending
re-diagnosis flagged above.
```

with:

```latex
--- applying the \emph{same} three-part gradient/adjacency diagnostic (embedding
RMS relative to initialization, gradient norm reaching
$\mathbf{E}^{(1)}_k/\mathbf{E}^{(2)}_k$, and cosine similarity against the run's
own random initialization) to every one of the 19 trained cells. All 19 cells
were trained at the tuned configuration of Section~\ref{sec:hpo} with weight
decay $10^{-5}$, and all 19 completed. Three of the cells were initially
diagnosed from best-validation checkpoints that early stopping had frozen after
one to two epochs of training; we identified this, added a code path that
captures the final-epoch weights alongside the best-validation ones, and re-ran
those three cells, so every diagnostic value reported below is measured on a
checkpoint embodying at least five epochs of training. Two of the five
experiments carry further defects we disclose in full below rather than repair,
for reasons we state.
```

---

### TODO 2 — line 526: the cross-experiment correlation statistics

**Replace the `\todohl{[TODO: these four statistics ...]}` block with nothing (delete it), and
update the preceding sentences' numbers as follows.** Replace, in the paragraph beginning
`\textbf{The collapse is invariant to every axis we varied...}`:

```latex
Across all 19 cells, the
trained graph embeddings' cosine similarity to their own random initialization is
predicted by the number of epochs of training embodied in the saved checkpoint,
with Spearman $\rho = -0.990$ ($p = 7.5\times10^{-16}$); the embeddings' RMS
magnitude relative to initialization gives $\rho = -0.993$ ($p =
3.4\times10^{-17}$), and the gradient norm reaching them gives $\rho = -0.961$
($p = 6.8\times10^{-11}$). A linear fit in training epochs explains 97.7\% of the
variance in cosine-to-initialization, and the residuals from that fit show no
detectable dependence on which experiment produced the cell (one-way ANOVA across
the four axes: $F = 0.543$, $p = 0.660$; Kruskal--Wallis $p = 0.638$; largest
absolute residual 0.043 against a full observed range of 0.420--0.994). Relative
embedding magnitude falls from 82\% of its initialization value in the
least-trained cell (one epoch) to $9\times10^{-8}$ of it in the most-trained
(37 epochs; an absolute embedding RMS of $1.5\times10^{-8}$ against a Xavier
reference of 0.2885).
```

with:

```latex
Across all 19 cells, each diagnosed on a checkpoint embodying between 5 and 37
epochs of training, the trained graph embeddings' cosine similarity to their own
random initialization is predicted by that epoch count with Spearman
$\rho = -0.990$ ($p = 7.5\times10^{-16}$); the embeddings' RMS magnitude relative
to initialization gives $\rho = -0.993$ ($p = 3.4\times10^{-17}$), and the
gradient norm reaching them gives $\rho = -0.957$ ($p = 1.4\times10^{-10}$). A
linear fit in training epochs explains 96.5\% of the variance in
cosine-to-initialization ($\mathrm{cos} = 0.984 - 0.0168\,E$), a fit in
$\log_{10}$ of the relative RMS explains 96.8\%, and the residuals from the
cosine fit show no detectable dependence on which experiment produced the cell
(one-way ANOVA across the four axes: $F = 0.791$, $p = 0.518$; Kruskal--Wallis
$p = 0.543$; largest absolute residual 0.060 against a full observed range of
0.420--0.927). Relative embedding magnitude falls from 33\% of its
initialization value in the least-trained cell (five epochs) to
$9\times10^{-8}$ of it in the most-trained (37 epochs; an absolute embedding RMS
of $1.5\times10^{-8}$ against a Xavier reference of 0.2885), passing through
$1.2\times10^{-3}$ at 21 epochs.
```

*Note for the integrator:* the two headline rank correlations ($-0.990$, $-0.993$) are unchanged
because the correction preserved the rank ordering — worth one sentence if space allows, but not
required. The HPO paragraph's separate $\rho = 0.456$, $p = 0.05$ figure for epoch-count versus
test average MSE **must be left alone**; see §5.4 above for why it cannot be recomputed on the
corrected set.

---

### TODO 3 — line 561: the coverage claim and the EMD caveat

**Replace the whole "early-stopping artifact" paragraph plus the "Restricting attention to the 16
cells" paragraph and the trailing `\todohl{...}` block** (i.e. everything from
`\textbf{An early-stopping artifact, and why we report it...}` through
`...must be reported as one.]}`) with:

```latex
\textbf{An early-stopping artifact we detected, corrected, and report because it
is instructive.} On a first pass, three of the 19 cells --- the EMD arm, one seed
of the per-band variant, and one seed of the pooled variant --- were flagged
\texttt{NOT COLLAPSED} by the automated diagnostic. They were not
counterexamples. Our trainer saves the best-validation weights but stamps them
with the loop's final epoch index; because the best-validation save condition and
the early stopper's improvement condition are the same predicate (a strict
improvement on the running best, with zero tolerance), and every run early-stopped
well inside the 200-epoch budget, the weights actually in a checkpoint are exactly
those from epoch $(\text{recorded epoch} - 20)$ under the patience of 20 used
throughout. For these three cells the recorded indices were 0, 1 and 1, so their
saved weights embodied one to two epochs of training against five to thirty-seven
in every other cell, and their verdicts measured training duration rather than any
property of the axis being varied. We therefore added a diagnostic path that
retains the final-epoch weights alongside the best-validation ones and re-ran the
three cells. All three reproduced their original early-stopping trajectory to the
same final epoch and their original test metrics to within $2\times10^{-6}$
relative, confirming that the re-diagnosis describes the same run.

\textbf{All three collapse once diagnosed on genuinely trained weights.} At 21,
22 and 22 epochs of training respectively, every band of all three cells is
flagged collapsed on both of the diagnostic's independent criteria: the fraction
of surviving adjacency edges within tolerance of $1/N$ is \textbf{1.00 in every
band of all three}, and embedding RMS has fallen to $6\times10^{-4}$ to
$1.3\times10^{-3}$ of the Xavier reference. The gradient norm reaching the
embeddings is \emph{exactly} zero for the EMD and per-band seed-44 cells and
$2.7\times10^{-22}$ for the pooled seed-45 cell, while the gradient reaching the
rest of the network is 0.17--0.22 in the same measurement --- a dead graph branch
inside a live model. The three cells thus provide a within-run confirmation of the
epoch relationship above that is stronger than any across-run comparison, because
configuration, seed and training trajectory are all held fixed and only the
snapshot moves: the same three runs read 0.994, 0.983 and 0.981
cosine-to-initialization at one to two epochs and 0.613, 0.597 and 0.555 at
twenty-one to twenty-two. Two further cells re-run as a consistency check, which
happened to train for 45 and 68 epochs, extend the same trajectory to relative
embedding magnitudes of $1.4\times10^{-10}$ and $3.2\times10^{-22}$.

We note for completeness that for these three cells the deployed
(best-validation) checkpoint and the diagnosed (final-epoch) checkpoint are
different weight sets. This does not soften the finding, because neither state
contains learned relational structure: at the best-validation snapshot the
embeddings are statistically indistinguishable from their random initialization
(cosine 0.98--0.99, 65--82\% of initialization norm), and at the final-epoch
snapshot they are uniform with no gradient. The choice between the two is a choice
between random and uniform.

With the correction in place, the collapse is observed in \textbf{19 of 19}
runs, each diagnosed at a checkpoint embodying at least five epochs of training:
at all four band counts, in both decomposition methods, in all three loss arms,
and at all five seeds in each of the two graph variants. We state the epoch
qualifier because it is load-bearing: the three re-diagnosed cells are the same
runs whose one-to-two-epoch snapshots read \texttt{NOT COLLAPSED}, so what this
suite establishes is not that the collapse is present at every epoch count, but
that it is present in every run once as few as five epochs have elapsed, and that
no axis we varied delays or prevents it. The
decomposition-axis caveat we previously stated is withdrawn --- EMD does not
resist the collapse, and in fact, having trained ten epochs longer than the VMD
cell, sits further along the identical trajectory (relative embedding magnitude
$1.2\times10^{-3}$ against VMD's $6.1\times10^{-2}$, and adjacency uniform to
within $3\times10^{-8}$ against VMD's $8\times10^{-5}$).
```

---

### TODO 4 — line 605: the multi-seed cells' pending verdicts

**Replace the `\todohl{[TODO: two of the ten multi-seed cells ...]}` block with:**

```latex
Two of the ten multi-seed cells (per-band seed 44, pooled seed 45) are among the
three re-diagnosed from final-epoch weights described above. Both re-ran to the
same final epoch as originally recorded and reproduced their test metrics to
within $10^{-8}$ absolute, so the paired tests reported here are unchanged; only
their collapse verdicts change, from \texttt{NOT COLLAPSED} to
\texttt{COLLAPSED}, which is what brings the per-seed coverage to all five seeds
in both variants.
```

*(If the integrator prefers, this can be folded into the preceding sentence rather than kept as
its own sentence; the substance is the two clauses "paired tests unchanged" and "verdicts
changed".)*

---

### TODO 5 — line 730: the limitations-list item

**Replace the entire limitation item** (from `\item \textbf{Three robustness cells are diagnosed
from barely-trained checkpoints...}` through the closing `...once it completes.]}`) with:

```latex
    \item \textbf{Three robustness cells were initially diagnosed from barely-trained checkpoints; the diagnosis has been corrected and we report the correction rather than suppress it.} Three of Section~\ref{sec:robustness}'s 19 diagnostic cells (the EMD arm, the per-band seed-44 run, and the pooled seed-45 run) early-stopped such that their saved best-validation weights embodied only one to two epochs of training, so their automated ``not collapsed'' verdicts measured training duration rather than any property of the axis being varied. We added a diagnostic path that retains each run's final-epoch weights alongside its best-validation weights and re-ran those three cells; all three reproduced their original trajectories to the same final epoch and their original test metrics to within $2\times10^{-6}$ relative, and all three are collapsed when diagnosed at 21--22 epochs of training, with every band's surviving adjacency edges uniform to within tolerance of $1/N$ and the gradient reaching the graph embeddings exactly zero or of order $10^{-22}$. The collapse claim in that section is therefore stated over 19 of 19 runs, each diagnosed at a checkpoint embodying at least five epochs of training, rather than over 16 of 16 with three cells set aside; the qualifier matters, since the three re-diagnosed runs are the same ones whose one-to-two-epoch snapshots read ``not collapsed'', so what the suite establishes is that the collapse appears in every run within a few epochs and that no axis we varied delays it, not that it is present at every epoch count. The decomposition axis now carries usable evidence in both directions. Two residual caveats remain. First, for these three cells the deployed best-validation checkpoint and the diagnosed final-epoch checkpoint are different weight sets; we did not measure the final-epoch weights' test accuracy, and we rest the claim on the observation that neither snapshot contains learned relational structure --- the earlier one is indistinguishable from random initialization (cosine 0.98--0.99 to its own init) and the later one is uniform with no gradient. Second, the decomposition axis still rests on a single cell per method, so it establishes that EMD does not \emph{avoid} the collapse but does not support any accuracy comparison between the two decompositions.
```

---

## 9. Reproducing this document

```
python verify_fix.py    # raw-.npy recomputation of all 5 cells; ground-truth identity;
                        # checkpoint contents; multi-seed paired tests
python replaced19.py    # replaced-19 correlations, fits, ANOVA; original-vs-new run
                        # comparison; within-run two-snapshot table
```

Every number traces to: the two new `.jsonl` files, the 5 new `.pt` checkpoints, the 40 new
`.npy` arrays, and — for the comparison set — the four original `.jsonl` files and their 19
checkpoints under `latest_results/{k_sweep,decomposition,loss_comparison,multiseed}/`.

## 10. Files

- This document: `paper1/latest_results/FINALEPOCH_FIX_RESULTS.md`
- Supersedes, for the three artefact cells only, the verdicts in
  `paper1/latest_results/ROBUSTNESS_ANALYSIS_FINAL.md` §3.3 and rows 1--3 of its §2 table. All
  other findings in that document (Bug A, Bug B, the wider-HPO analysis, the replicate noise
  floor) are unaffected.
- `paper1/latest_results/collapse_invariance_summary.csv` is now stale for those three rows and
  should not be cited without this document beside it.
