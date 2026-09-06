# Paper 1 (VMD-MFGNN) — Robustness Experiments: Final Analysis

**Status:** definitive. Supersedes the earlier partial-run analysis (`Copper_Paper1/ROBUSTNESS_ANALYSIS.md`).

**Note on the superseded document:** that file is *not present anywhere in this repository or in either results zip* — it was checked for by name, by glob, and across all git history and both archive zips, and does not exist on disk. This document is therefore written to be fully self-contained; nothing below requires the reader to consult it.

**Data analysed:** `paper1/latest_results/` — a new, independent, **complete** run of all five robustness experiments (the earlier run had 4 of 5 finished and the wider-HPO study at 19/30 trials). No code changes were made between the two runs.

**Method:** every number below was recomputed independently from the raw artefacts — the five `.jsonl` result files, the 19 `.pt` checkpoints, the 32 `.npy` prediction/ground-truth arrays, and the Optuna SQLite study. The notebook's own auto-generated `collapse_invariance_summary.csv` was used only as a cross-check (it agrees exactly; see §7).

---

## 1. Executive summary

| # | Claim | Verdict |
|---|---|---|
| 1 | The headline finding replicates: graph-embedding state is explained by **how many epochs the saved checkpoint had trained**, not by K / decomposition method / loss / seed / variant | **Replicates, more strongly than before.** ρ = −0.990 (cosine), ρ = −0.993 (RMS ratio), n = 19, p < 1e−15. Linear fit R² = 0.977. No residual axis effect (ANOVA p = 0.66) |
| 2 | Bug A — multi-seed used the wrong RQ1 comparison variant | **STILL PRESENT.** `pooled_graph_matched_dim`, 315,530 params vs 1,461,124 — a **78.4 % capacity deficit**, confirmed by direct parameter counting from checkpoints, not by trusting the variant string |
| 3 | Bug B — Huber setting is degenerate | **STILL PRESENT.** β = 1.0 with max abs residual 0.284 (28.4 % of β) ⇒ `smooth_l1` is in its quadratic branch for 100 % of samples ⇒ mathematically **exactly 0.5 × MSE**. The "3-way loss comparison" tests 2 distinct losses |
| 4 | **NEW (this pass): a third defect** — the "NOT COLLAPSED" verdicts are an early-stopping artefact | **CONFIRMED.** All three runs the notebook labels NOT COLLAPSED are runs whose *saved* weights had trained for **≤ 1 epoch** |
| 5 | **NEW: wider HPO (30 trials, 8 params incl. K/weight_decay/batch_size)** finds a better config | **Marginally.** Best val MSE 0.0020145 vs the existing config's 0.0020509 — a **1.78 % improvement**, smaller than the 3.1 % seed spread and far smaller than the 7.5 % same-config replicate spread. Not a meaningful gain |
| 6 | Recommendation | **Do not re-run either broken experiment.** Disclose both as limitations. Reasoning in §9 |

---

## 2. The five experiments — complete summary table

All 19 training cells completed (`completed=True` in every checkpoint). Config for every cell: tuned hyperparameters from `results/archive_paper1/hpo_best_params.json` (hidden_dim 128, heads 2, lr 0.001530, dropout 0.0616, gnn_layers 1), weight_decay 1e−5, batch 32, Adam with **coupled** L2 decay, patience 20, max 200 epochs.

`best_ep` = the epoch whose weights are actually in the checkpoint (see §3 — this is *not* the `epoch` field). `cos` = mean cosine similarity of the trained graph embeddings against their own random initialisation. `rms_ratio` = trained embedding RMS ÷ init RMS. `emb_grad` = mean gradient norm reaching emb1/emb2.

| Experiment | Cell | last_ep | **best_ep** | cos | rms_ratio | emb_grad | verdict | test avg_MSE |
|---|---|---:|---:|---:|---:|---:|---|---:|
| decomposition | **emd** | 20 | **0** | 0.9943 | 0.8158 | 2.4e−08 | *NOT COLLAPSED* † | 0.002726 |
| multiseed | **full_model seed44** | 21 | **1** | 0.9828 | 0.6633 | 1.3e−08 | *NOT COLLAPSED* † | 0.002696 |
| multiseed | **pooled seed45** | 21 | **1** | 0.9811 | 0.6517 | 3.9e−08 | *NOT COLLAPSED* † | 0.002749 |
| k_sweep | K=7 | 24 | 4 | 0.9266 | 0.3306 | 2.9e−10 | COLLAPSED (0/7) | 0.002769 |
| multiseed | full_model seed45 | 24 | 4 | 0.9250 | 0.3254 | 2.7e−10 | COLLAPSED (0/5) | 0.002687 |
| loss | mae | 28 | 8 | 0.8373 | 0.1133 | 1.2e−11 | COLLAPSED (0/5) | 0.002691 |
| multiseed | full_model seed43 | 28 | 8 | 0.8352 | 0.1112 | 6.2e−13 | COLLAPSED (0/5) | 0.002746 |
| multiseed | full_model seed46 | 28 | 8 | 0.8479 | 0.1181 | 8.7e−13 | COLLAPSED (0/5) | 0.002742 |
| multiseed | pooled seed42 | 28 | 8 | 0.8441 | 0.1185 | 8.9e−15 | COLLAPSED (0/1) | 0.002758 |
| k_sweep | K=9 | 29 | 9 | 0.8136 | 0.0838 | 8.6e−14 | COLLAPSED (0/9) | 0.002828 |
| decomposition | vmd | 30 | 10 | 0.7940 | 0.0613 | 3.7e−14 | COLLAPSED (0/5) | 0.002773 |
| loss | mse | 30 | 10 | 0.7940 | 0.0613 | 3.5e−14 | COLLAPSED (0/5) | 0.002943 |
| multiseed | pooled seed43 | 30 | 10 | 0.8058 | 0.0618 | 3.8e−16 | COLLAPSED (0/1) | 0.002697 |
| multiseed | full_model seed42 | 32 | 12 | 0.7527 | 0.0314 | 2.5e−15 | COLLAPSED (0/5) | 0.002772 |
| k_sweep | K=5 | 33 | 13 | 0.7330 | 0.0220 | 7.1e−16 | COLLAPSED (0/5) | 0.002981 |
| k_sweep | K=3 | 34 | 14 | 0.7182 | 0.0158 | 3.9e−18 | COLLAPSED (0/3) | 0.002757 |
| multiseed | pooled seed46 | 34 | 14 | 0.7156 | 0.0155 | 1.0e−17 | COLLAPSED (0/1) | 0.002763 |
| multiseed | pooled seed44 | 36 | 16 | 0.7156 | 0.0069 | 1.9e−20 | COLLAPSED (0/1) | 0.002932 |
| loss | **huber** | 56 | **36** | 0.4203 | 5e−08 | **0.0** | COLLAPSED (0/5) | 0.002712 |

† These three verdicts are artefacts — see §3. They should not be reported as "EMD avoids collapse" or "seed 44 avoids collapse".

The 5th experiment (wider HPO) produced no checkpoints; it is analysed in §6.

**Note the table is sorted by `best_ep` and every diagnostic column is monotone in it.** That is the whole finding.

---

## 3. The headline finding: replicates — with one correction that sharpens it

### 3.1 What the correlation is *against*

`trainer.fit()` writes `{"state_dict": best_state, "epoch": last_epoch, "completed": True}`. The state dict is the **best-validation** weights; the `epoch` field is the **last** epoch the loop ran. Because the best-val save condition and `EarlyStopper.should_stop`'s improvement condition are the same predicate, and every one of the 19 runs early-stopped (max last_ep = 56, budget = 200 epochs, patience = 20):

```
best_epoch = checkpoint["epoch"] − 20      (exactly, for all 19 runs)
```

The correct explanatory variable is therefore **epochs of training embodied in the saved checkpoint**, ranging **0 → 36**, not 20 → 56. The Spearman coefficients are unaffected by the constant shift, but the *interpretation* changes materially (§3.3).

### 3.2 The numbers (n = 19, pooled across all four experiment axes)

| Relationship | Spearman ρ | p | Pearson r | p |
|---|---:|---:|---:|---:|
| best_epoch vs mean cosine-to-init | **−0.9899** | 7.5e−16 | −0.9884 | 2.4e−15 |
| best_epoch vs mean RMS-ratio-to-init | **−0.9930** | 3.4e−17 | −0.674 | 1.6e−03 |
| best_epoch vs log₁₀(RMS ratio) | −0.9930 | 3.4e−17 | **−0.9735** | 2.5e−12 |
| best_epoch vs log₁₀(emb grad norm) | −0.9607 | 6.8e−11 | −0.9844 | 2.9e−14 |
| best_epoch vs **test avg_MSE** | +0.456 | 0.050 | +0.169 | 0.49 |

Regression fits:

```
cosine_to_init      = 0.9748 − 0.01658 × best_epoch          R² = 0.977
log10(rms_ratio)    = 0.4522 − 0.1852  × best_epoch          R² = 0.948
```

**Do not quote a decay rate from that second fit.** Its intercept implies rms_ratio = 2.83 at best_ep 0 and 1.85 at best_ep 1, against observed values of 0.82 and 0.66 — it overshoots the low end by 3–4× because it is dominated by the mid/high-epoch mass, and the log-scale R² hides that. What is robust is the **rank** relationship (ρ = −0.993 over all 19 cells) plus the observed endpoints: embedding norm falls monotonically from **0.82 of initialisation at 1 epoch of training to 1.5e−08 at 37 epochs**. The cosine fit is well-behaved by contrast (intercept 0.975 vs the least-trained cell's observed 0.994) and is safe to quote.

**Does the axis add anything beyond epoch count?** Residuals from the cosine fit, grouped by which experiment produced the cell:

| axis | n | mean residual | sd |
|---|---:|---:|---:|
| k_sweep (K = 3/5/7/9) | 4 | −0.0111 | 0.0178 |
| decomposition (VMD/EMD) | 2 | +0.0022 | 0.0173 |
| loss (MSE/MAE/Huber) | 3 | +0.0075 | 0.0250 |
| multiseed (2 variants × 5 seeds) | 10 | +0.0017 | 0.0166 |

ANOVA F = 0.543, p = 0.660. Kruskal–Wallis H = 1.694, p = 0.638. Max absolute residual 0.043, against a full cosine range of 0.420 → 0.994.

**A within-replicate confirmation.** The four accidental identical-config replicates (§6.3 — same tuned params, K=5, VMD, MSE, *and* seed 42, reached through four different experiment drivers) landed at best_epoch 10, 10, 12, 13 with cosine 0.794, 0.794, 0.753, 0.733 — sitting on the fitted line. So the relationship tracks training duration even with the axis *and* the seed held fixed. It also shows that run-to-run nondeterminism perturbs the epoch count itself, which is precisely why the axis and seed labels carry no residual information once epoch count is known.

**Conclusion: the headline finding replicates, and does so on a fully-complete, independent 19-run dataset.** Once you know how many epochs a checkpoint trained for, knowing whether it was K=3 or K=9, VMD or EMD, MSE or MAE or Huber, seed 42 or 46, per-band or pooled, tells you essentially nothing further about its graph-embedding state.

### 3.3 The correction: the three "NOT COLLAPSED" verdicts are an early-stopping artefact

Sorting by `best_epoch` (§2) makes it immediate: the three cells the diagnostic labels NOT COLLAPSED are the three least-trained cells in the entire dataset. Their saved weights come from epoch **indices 0, 1 and 1** — and since `fit()` trains a full pass before evaluating, that is **one to two epochs of training**, against 5 to 37 in every other cell. They retain 65–82 % of their initialisation norm, where every other cell retains 0.7–33 %.

These are not counterexamples; they are cells caught before the decay process crossed the diagnostic's `rms_ratio < 0.5` threshold. (They are not literally untrained either — at 0.65–0.82 of init norm they have already shed 18–35 %, and they moved in the same direction as everything else, just less far.) Their verdict is a function of training duration alone, which is the point.

This is the identical artefact `main.tex` already documents for `full_model_graphfix` ("early-stopped at epoch 20 under a patience of 20 … effectively the epoch-0 initialization state"). It has recurred here, in three cells, undetected by the notebook's own summary.

**Consequences.** (a) Any statement of the form "EMD avoids the collapse" or "1 of 5 seeds avoided collapse" that is drawn from `collapse_invariance_summary.csv` is wrong and must not enter the paper. (b) The correct statement is that collapse is **universal across every cell that actually trained**: all 16 runs with best_epoch ≥ 4 are COLLAPSED, 16/16, and the three exceptions trained for one to two epochs each. (c) The `n_ok`/`verdict` column of the auto-generated summary is unsafe to cite without the best_epoch column beside it.

### 3.4 Mechanism corroboration

`trainer.py` uses `torch.optim.Adam(..., weight_decay=…)` — **coupled** L2, i.e. `wd × p` is added into the gradient, so its magnitude does not scale with the loss. That predicts: shrink the task loss and the decay pull, relative to the task gradient, gets stronger. The degenerate Huber arm (§5) is an unintended but clean test of exactly that, and it behaved as predicted: it is the most-trained (best_ep 36) and by far the most collapsed run in the set, terminating in literal numerical underflow (§5.2).

---

## 4. Bug A — multi-seed uses the wrong RQ1 variant: **STILL PRESENT**

`multiseed_results.jsonl` variant strings, all 10 rows:

```
full_model                 : seeds [42, 43, 44, 45, 46]
pooled_graph_matched_dim   : seeds [42, 43, 44, 45, 46]
```

`pooled_graph_matched_params` does not appear. Direct parameter counts from the checkpoint state dictionaries (not from the variant string):

| variant | parameters | ratio |
|---|---:|---:|
| `full_model` (VMDMFGNN) | **1,461,124** | 1.000 |
| `pooled_graph_matched_dim` (PooledGraphMFGNN) | **315,530** | 0.2160 |

⇒ **78.4 % parameter deficit** — closely consistent with the 78.5 % figure `main.tex` reports for this variant, though not the same measurement: the manuscript's ablation table is at the **untuned** config (hidden_dim 64) while these cells are at the tuned one (hidden_dim 128), so these are two different model pairs that happen to land 0.1 pp apart. This is a *dimension*-matched, not capacity-matched, comparison; it is not the fair-capacity contrast that carries RQ1.

The comparison the experiment produced, for the record (paired by seed, n = 5, `paired_seed_significance_test` reproduced independently):

| horizon | full_model RMSE (mean ± sd) | pooled_matched_dim RMSE | Δ | paired t (p) | Wilcoxon p |
|---|---|---|---:|---|---:|
| h=1  | 0.018238 ± 0.000163 | 0.018241 ± 0.000207 | −0.000003 | −0.051 (0.962) | 1.000 |
| h=5  | 0.040005 ± 0.000413 | 0.040634 ± 0.000335 | −0.000630 | −1.885 (0.133) | 0.063 |
| h=10 | 0.055572 ± 0.000395 | 0.055620 ± 0.000398 | −0.000048 | −0.165 (0.877) | 1.000 |
| h=22 | 0.076761 ± 0.000829 | 0.077697 ± 0.001896 | −0.000936 | −0.738 (0.502) | 0.625 |

No horizon reaches significance at α = 0.05 under either test.

**What this does and does not license.** It does *not* license "per-band and capacity-matched pooling are equivalent" — the pooled arm here has 78 % fewer parameters. It *does* license the weaker, still-useful statement that a per-band model with 4.6× the parameters cannot beat a much smaller pooled model at any horizon across five seeds — which is directionally consistent with, and only strengthens, the paper's already-negative RQ1 answer.

---

## 5. Bug B — the Huber configuration is degenerate: **STILL PRESENT**

### 5.1 The setting

`src/trainer.py`:
```python
"huber": lambda pred, target: F.smooth_l1_loss(pred, target, beta=1.0),
```

`smooth_l1_loss` with `beta = 1.0` is `0.5·x²/β = 0.5·x²` for `|x| < β`, and `|x| − 0.5β` otherwise. So the arm is a genuine Huber loss only for residuals exceeding 1.0.

### 5.2 The residual distribution says it never gets there

Recomputed from the raw `.npy` prediction/ground-truth pairs (all three loss arms × four horizons, 984 test samples each):

- **Global max |residual| across every loss arm and horizon: 0.28421** — i.e. **28.4 % of β**.
- Target scale for context: h=22 true returns span [−0.302, +0.180], sd 0.0758.
- Per-arm maxima: MSE 0.2778, MAE 0.2842, Huber 0.2822. Per-horizon 99th percentiles never exceed 0.233.

⇒ `smooth_l1` operates in its quadratic branch for **100 % of samples, at every horizon, in every arm**. The Huber arm is **mathematically identical to 0.5 × MSE**. The "MSE vs MAE vs Huber" comparison tested **two** distinct loss functions, not three.

### 5.3 …but the degenerate arm is an accidental, informative natural experiment

If Huber ≡ 0.5·MSE, why does that arm behave differently from the MSE arm at all (best_ep 36 vs 10; cos 0.42 vs 0.79)? Because Adam's `weight_decay` is **coupled**: halving the task loss halves the task gradient while leaving the `wd × p` term untouched, doubling the decay's influence relative to the learning signal. Direct inspection of `loss_comparison/checkpoints/full_model_huber.pt` confirms the endpoint:

| arm | emb1/emb2 RMS range | max |w| range |
|---|---|---|
| MAE | 2.90e−02 … 3.68e−02 | 8.9e−02 … 9.7e−02 |
| MSE | 1.56e−02 … 2.03e−02 | 5.2e−02 … 5.8e−02 |
| **Huber (=0.5·MSE)** | **1.52e−08 … 4.07e−08** | 1.2e−07 … 2.3e−07 |

(Xavier reference RMS = 0.2885.) At band 0, `relu(E₁E₂ᵀ)` has max 4.5e−15, and its softmax has **standard deviation exactly 0.0** — a bit-exact uniform 1/N adjacency. Hence the reported `emb_grad_norm_mean = 0.0`: not a rounding artefact but the terminal absorbing state of the decay process, where the graph branch contributes an exactly constant output and receives exactly zero gradient.

So the arm should be reported twice: as a **real defect** for the loss-comparison claim, and as **independent corroboration** of the weight-decay mechanism the paper's Section~\ref{sec:collapse}/\ref{sec:graphfix} already argues for.

---

## 6. NEW: the wider HPO pass (30 trials, 8-parameter space)

Loaded from `latest_results/wider_hpo/hpo_wider_study.db` (Optuna 4.9.0, study `vmd_mfgnn_wider_hpo`). Objective: **best validation `avg_mse` over 25 epochs**, minimised. Search space adds K ∈ {3,5,7,9}, weight_decay (log-uniform 1e−6…1e−3) and batch_size ∈ {16,32,64} to the original five parameters.

### 6.1 Completion

**30 trials: 12 COMPLETE, 18 PRUNED.** Every one of the 18 pruned trials was pruned **at epoch 0** — `MedianPruner()` is constructed with default `n_warmup_steps=0`, so a trial is killed after its very first epoch if it is below the running median. The search therefore evaluated 12 configurations at the intended 25-epoch budget and screened the other 18 on a single epoch of validation loss.

### 6.2 Best trial vs. the config in production

| | existing tuned config (10-trial study) | **wider-HPO best (trial #26)** |
|---|---|---|
| hidden_dim | 128 | 128 |
| num_heads | 2 | 8 |
| learning_rate | 0.0015305 | 0.0027729 |
| dropout | 0.06161 | 0.09037 |
| num_gnn_layers | 1 | 2 |
| weight_decay | 1e−5 (fixed, not searched) | **1.999e−4** (20× larger) |
| batch_size | 32 (fixed, not searched) | **16** |
| K | 5 (fixed, not searched) | **9** |
| **best val avg_MSE** | **0.0020509** | **0.0020145** |

**Improvement: 1.78 %.** Only **2 of the 12** completed wider trials beat the existing config's value at all; the second-best (0.0020506) ties it to four significant figures. Spread across the 12 completed trials is 1.16× best-to-worst (0.002014 → 0.002337).

### 6.3 Is 1.78 % meaningful? No.

Three independent noise estimates from this same dataset, all larger than the gain:

1. **Seed noise.** `full_model` test avg_MSE across 5 seeds: mean 0.0027287, sd 0.0000361, **CV 1.32 %**, best-to-worst **3.12 %**. (Pooled variant: CV 3.21 %, range 8.46 %.)
2. **Same-config replicate noise (the strongest estimate — seed held fixed too).** Four cells in this dataset are the *identical* configuration — tuned params, K = 5, VMD, MSE, seed 42: `k_sweep K=5`, `decomposition vmd`, `loss mse`, `multiseed full_model_seed42`. Their test avg_MSE: 0.002772 / 0.002773 / 0.002943 / 0.002981 — a **7.54 % spread**. Their ground-truth `.npy` arrays are bit-identical across all four arms (verified: `np.array_equal` True, shape (984,)), so this is genuine run-to-run nondeterminism on identical data at a fixed seed, not a test-set mismatch. Their predictions nonetheless differ (pairwise correlation 0.65–0.68, max abs difference up to 0.111).
3. **HPO's own step-count confound.** The winning trial uses batch_size 16 against the original's 32, i.e. roughly 2× the optimiser steps inside the same 25-epoch budget. Part of the 1.78 % is training-step count, not configuration quality.

**Verdict: the wider search confirms the existing 10-trial / 5-parameter configuration was already effectively optimal.** A 1.78 % val-MSE gain that sits inside a 7.5 % same-config replicate band is not a finding. Nothing here warrants re-running the main results at the new configuration.

### 6.4 Marginals are confounded — do not table them as effects

TPE concentrated hard: 8 of the 12 completed trials are K = 9, and **all three batch_size = 32 trials are also the three K = 3 trials**. K and batch_size are therefore aliased in the completed set, and the K=9-looks-best / bs=16-looks-best marginals are artefacts of sampling concentration plus epoch-0 pruning (which structurally favours high learning rate and small batch — exactly the region the survivors occupy). Optuna's fANOVA importances (K 0.377, learning_rate 0.346, dropout 0.098, num_heads 0.081, weight_decay 0.040, batch_size 0.038, num_gnn_layers 0.018, hidden_dim 0.003) inherit the same confound and should be reported, if at all, only with that caveat.

Independent corroboration that the K marginal is not real: the K-sweep experiment (§2) trained each K to convergence on the **test** set and produced avg_MSE K=3 0.002757, K=5 0.002981, K=7 0.002769, K=9 0.002828 — an ordering that does not match the HPO validation ordering, and whose entire range (8.1 %) sits inside the 7.5 % replicate noise floor.

### 6.5 Does the collapse persist at the HPO optimum?

**Not directly checkable — stated as a limitation.** `run_hpo` saves no per-trial checkpoints and runs no diagnostics; `wider_hpo/` contains only `hpo_wider_study.db`, `hpo_wider_best_params.json` and `hpo_wider_trials.csv`. No embedding or gradient state exists for any of the 30 trials.

Two things *can* be established without it:

1. **The HPO objective is structurally blind to collapse.** The objective is validation `avg_mse` alone. In this dataset the three near-initialisation (i.e. definitionally not-yet-collapsed) checkpoints have test avg_MSE 0.002726 / 0.002696 / 0.002749 — squarely mid-range among the 16 fully-collapsed ones (0.002687 … 0.002981), and best_epoch correlates with test avg_MSE only at ρ = +0.456 (p = 0.050, Pearson r = 0.169, p = 0.49). Collapse costs essentially nothing on the tuned objective, so **no amount of HPO under this objective can select against it**, regardless of how wide the search space is.
2. **The optimum's own hyperparameters point the wrong way** — it selects `weight_decay = 2.0e−4`, **20× the 1e−5 used in every other experiment**, against a collapse mechanism the paper attributes to coupled L2 decay. The prediction is that collapse at the HPO optimum would be *more* severe, not less. **This is inference from the established mechanism, not a measurement**, and must be labelled as such wherever it is used.

If a future run wants this closed, the minimal fix is to have `run_hpo` call `diagnostics.adjacency_state_diagnostic` on the best trial's final model and record it alongside the objective value — cheap, and it would make the "HPO cannot see the collapse" point empirical rather than inferential.

---

## 7. Cross-check against the notebook's own auto-generated summary

`latest_results/collapse_invariance_summary.csv` (19 rows) and `figures/collapse_invariance_grad_norms.png` were produced by the notebook's own Section 8 aggregation. Compared row-by-row against the independent recomputation:

- **All 19 rows agree exactly** on `verdict`, `n_ok`, `n_total`, `emb_grad_norm_mean`, `other_grad_norm_mean` and `mean_cosine_similarity_vs_init`, to full printed precision (e.g. K=3: 3.87505152412166e−18 / 0.1556404135802266 / 0.7182441552480062 — identical). The aggregation is faithful to the `.jsonl` rows.
- **But it is materially incomplete in one respect:** it carries no epoch column, so its three NOT COLLAPSED verdicts (`emd`, `full_model_seed44`, `pooled_graph_matched_dim_seed45`) are presented without the fact that all three checkpoints trained for ≤ 1 epoch (§3.3). Read on its own it invites exactly the wrong conclusion. **Do not cite this CSV without the best_epoch column beside it.**

---

## 8. Full list of issues found in this dataset

| # | Issue | Severity | Present in this run? |
|---|---|---|---|
| A | Multi-seed compares against `pooled_graph_matched_dim` (78.4 % parameter deficit), not `pooled_graph_matched_params` | Medium — weakens but does not invert the RQ1 claim | **Yes** |
| B | Huber β = 1.0 ⇒ exactly 0.5 × MSE for this data; only 2 distinct losses tested | Medium — a stated 3-way comparison is really 2-way | **Yes** |
| C | **New:** three cells report NOT COLLAPSED but their saved weights trained ≤ 1 epoch; the auto-summary does not disclose this | **High** — reading it at face value produces a false "EMD/seed-44 avoids collapse" finding | **Yes** |
| D | **New:** all 18 pruned HPO trials pruned at epoch 0 (`MedianPruner` default `n_warmup_steps=0`); the wider search screened 60 % of its budget on one epoch | Medium — weakens the "wider search" claim; biases survivors toward high-lr/small-batch | **Yes** |
| E | **New:** HPO trials capture no diagnostics, so collapse at the optimum is not measurable | Low — mitigated by the structural argument in §6.5 | **Yes** |
| F | **New:** four accidental identical-config replicates disagree by 7.5 % on test avg_MSE at a fixed seed (ground-truth arrays verified bit-identical), i.e. run-to-run nondeterminism exceeds every axis effect measured | Informative, not a defect | **Yes** |

---

## 9. Overall verdict and recommendation

**The scientific content of the robustness suite is intact and, on this complete run, stronger than before.** The single claim these five experiments were built to test — *is the adjacency-collapse finding an artefact of the specific K / decomposition / loss / seed we happened to choose?* — is answered decisively in the negative, on 19 independent runs, with a cross-experiment correlation of ρ = −0.99 and no detectable residual axis effect (p = 0.66). Collapse occurs in **16 of 16 cells that trained for more than one epoch**, across every value of every axis tested. The two known defects do not touch that claim.

### Recommendation: do not re-run either broken experiment. Disclose both.

**Multi-seed with `pooled_graph_matched_params`.** Three reasons, in order of weight:
1. `main.tex` **already retracts** the `pooled_graph_matched_params_tuned` comparison, having traced its apparent per-band advantage to a constant-sign forecasting artefact (Section~\ref{sec:constantforecast}). Re-running a variant whose prior comparison the paper explicitly withdrew does not restore a claim; there is no claim left to restore.
2. The result the run *did* produce is already the conservative one. A per-band model with **4.6× the parameters** fails to beat the smaller pooled model at any of four horizons across five seeds (p = 0.96 / 0.13 / 0.88 / 0.50). Restoring capacity parity can only move the pooled arm's performance up or leave it flat — either way strengthening an already-negative RQ1. There is no plausible outcome of the corrected run that changes the paper's answer.
3. Cost is ~5 further multi-hour GPU cells for a result whose direction is already determined.

**Loss comparison with a genuine third loss.** The axis it would test lands inside the noise floor. Test avg_MSE across the three loss arms spans 0.002691 → 0.002943 (9.4 %), while four *identical-configuration* replicates span 7.5 % (§6.3). A third loss would not be resolvable above run-to-run nondeterminism at n = 1 per arm. Meanwhile the defective arm is already earning its keep as an unintended loss-scale ablation that corroborates the weight-decay mechanism (§5.3) — arguably more informative than the well-formed Huber run would have been.

**What must change instead — all documentation, no compute:**
- Report the loss axis as **"MSE vs MAE"**, two losses, never three. State the β=1.0 / max-residual-0.284 arithmetic explicitly so a reader can verify the degeneracy.
- Report the multi-seed comparison as **"per-band vs a 78.4 %-smaller pooled variant"**, never as capacity-matched, and cite `main.tex`'s existing retraction of the matched-params comparison.
- **Never report the three NOT COLLAPSED verdicts without their ≤1-epoch training length.** This is the highest-risk item in the dataset, because the auto-generated CSV presents them without that context and they look like a genuine finding.
- Report the wider HPO as a **negative/confirmatory** result: a 3× wider, 8-parameter search finds 1.78 %, inside the noise floor; the existing configuration stands.
- Disclose the epoch-0 pruning of 18/30 HPO trials.

**If any compute is spent at all**, the highest-value target is not either broken experiment: it is the three artefact cells (EMD, `full_model` seed 44, `pooled` seed 45), so the collapse claim becomes 19/19 rather than 16/16-plus-3-unusable. Note that **raising `patience` would not on its own fix this**: the diagnostic runs on `trainer.model`, which `fit()` loads with `best_state`, so if validation never improves past epoch 1 a longer run still diagnoses the epoch-1 weights. The correct fix is a code change in `_train_and_diagnose` — run `diagnostics.run_full_diagnostic` on the **final-epoch** model as well as the best-validation one. And because the final-epoch weights of these three cells were never saved, they are **unrecoverable from the existing artefacts**; re-training is the only route. That is 3 cells plus a small code change, and it removes the one issue in this dataset that could produce a wrong published claim.

---

## 10. Draft paper text

Drafted in `main.tex`'s voice and hedging register, for insertion as a new subsection in Section~\ref{sec:results} (suggested placement: after Section~\ref{sec:graphfix}, before Section~\ref{sec:constantforecast}). **`main.tex` has not been modified.** Cross-reference labels assume the existing ones in `main.tex`.

---

```latex
\subsection{Robustness Checks: Is the Collapse an Artifact of Our Configuration Choices?}
\label{sec:robustness}

Section~\ref{sec:collapse}'s adjacency-collapse diagnosis and Section~\ref{sec:graphfix}'s
failed remedy were both established at a single configuration: $K=5$ bands, VMD
decomposition, MSE training loss, seed 42, and one set of tuned hyperparameters. A
reader is entitled to ask whether the finding is a property of the architecture or an
artifact of that particular corner of the configuration space. We therefore ran five
robustness experiments---a band-count sweep ($K \in \{3,5,7,9\}$), a decomposition-method
comparison (VMD versus EMD at matched $K$), a training-loss comparison, a five-seed
repetition of the two ablation rows carrying RQ1, and a substantially wider
hyperparameter search---applying the \emph{same} three-part gradient/adjacency
diagnostic (embedding RMS versus initialization, gradient norm reaching
$\mathbf{E}^{(1)}_k/\mathbf{E}^{(2)}_k$, and cosine similarity against a same-seed
untrained reference) to every one of the 19 trained cells. Two of the five carry
defects we disclose in full below rather than repair, for reasons we state.

\textbf{The collapse is invariant to every axis we varied, and is explained almost
entirely by one variable we did not intend to vary.} Across all 19 cells, the trained
graph embeddings' cosine similarity to their own random initialization is predicted by
the number of epochs the saved checkpoint had trained for, with Spearman
$\rho = -0.990$ ($p = 7.5\times10^{-16}$); the embeddings' RMS magnitude relative to
initialization gives $\rho = -0.993$ ($p = 3.4\times10^{-17}$), and the gradient norm
reaching them gives $\rho = -0.961$. A linear fit in training epochs explains 97.7\% of
the variance in cosine-to-initialization, and the residuals from that fit show no
detectable dependence on which experiment produced the cell (one-way ANOVA across the
four axes: $F = 0.543$, $p = 0.660$; Kruskal--Wallis $p = 0.638$; largest absolute
residual 0.043 against a full range of 0.420--0.994). Embedding magnitude falls
monotonically from 82\% of its initialization value after one epoch of training to
$1.5\times10^{-8}$ after 37. Put plainly: once one knows
how long a checkpoint trained, knowing whether it used three bands or nine, VMD or EMD,
MSE or MAE, or which of five seeds, adds essentially nothing. This is the signature of a
process driven by the optimizer rather than by the task---consistent with
Section~\ref{sec:graphfix}'s finding that removing weight decay's pull does not
substitute a learning signal for it.

\textbf{An early-stopping artifact, and why we report it rather than the verdict it
produces.} Three of the 19 cells---the EMD arm, and one seed each of the per-band and
pooled variants---are flagged \texttt{NOT COLLAPSED} by the automated diagnostic. They
are not counterexamples. Our trainer saves the best-validation weights but stamps them
with the loop's final epoch index; under a patience of 20 with early stopping, the
weights actually in the checkpoint are those from epoch $(\text{recorded epoch} - 20)$.
For these three cells the recorded indices are 0, 1 and 1, and since a full training pass
precedes each evaluation, their saved weights reflect one to two epochs of training
against five to thirty-seven in every other cell. They retain 65--82\% of their
initialization norm, where every other cell retains 0.7--33\%---they have moved, in the
same direction as everything else, simply not yet far enough to cross the diagnostic's
threshold. This is the same artifact we already document for \texttt{full\_model\_graphfix}
in Section~\ref{sec:graphfix}, recurring here in three further cells; we flag it because a
summary table reporting these verdicts without training length would invite the false
conclusion that EMD, or a particular seed, protects against the collapse. Restricting
attention to cells that trained beyond this margin, the collapse is observed in \textbf{16 of 16},
at every band count, both decomposition methods, both distinct losses, both graph
variants and all five seeds.

\textbf{Accuracy differences across these axes are smaller than run-to-run noise.} Four
of the 19 cells are, by construction, the identical configuration
($K=5$, VMD, MSE, seed 42) reached through four different experiment drivers, and we
verified their held-out ground-truth arrays are bit-identical. Their test average MSE
nonetheless spans 7.54\% (0.002772--0.002981), which we take as this study's honest
run-to-run noise floor at fixed seed. Every axis effect we measured is smaller: the
band-count sweep spans 8.1\%, the loss comparison 9.4\%, and the five-seed repetition
has a coefficient of variation of 1.32\%. We therefore report no accuracy ranking over
$K$, decomposition method or loss; none is resolvable at $n=1$ per cell. These four cells
also serve as a within-replicate check on the paragraph above: they retain 10, 10, 12 and
13 epochs of training and record cosine-to-initialization of 0.794, 0.794, 0.753 and
0.733, tracking training duration with configuration \emph{and} seed both held fixed, and
showing that run-to-run nondeterminism perturbs the epoch count itself---which is why the
configuration labels carry no residual information once training length is known.

\textbf{The five-seed RQ1 check, and its capacity caveat.} Repeating the two ablation
rows that carry RQ1 across seeds 42--46 at tuned hyperparameters, the per-band model's
test RMSE is statistically indistinguishable from the pooled variant's at every horizon
(paired $t$-test over seeds: $p = 0.96$, $0.13$, $0.88$, $0.50$ at $h = 1, 5, 10, 22$;
Wilcoxon signed-rank likewise non-significant throughout). \emph{This comparison is not
capacity-matched.} It re-ran \texttt{pooled\_graph\_matched\_dim}, which we confirm by
direct parameter counting from the saved checkpoints carries 315{,}530 parameters
against the per-band model's 1{,}461{,}124---a 78.4\% deficit---rather than the
parameter-matched variant. We report the comparison anyway, in the conservative
direction it happens to run: a per-band model with $4.6\times$ the parameters does not
beat a much smaller pooled model at any horizon across five seeds. This strengthens
rather than weakens Section~\ref{sec:discussion}'s negative answer to RQ1, and we note
that the parameter-matched comparison is in any case one this paper has already
retracted on independent grounds (Section~\ref{sec:constantforecast}).

\textbf{The loss comparison tested two losses, not three.} We configured MSE, MAE and
Huber arms. The Huber arm used PyTorch's \texttt{smooth\_l1\_loss} at its default
$\beta = 1.0$, which is quadratic---exactly $\tfrac{1}{2}x^2$---for residuals below 1.0.
The largest absolute residual observed anywhere in this experiment, across all three
arms and all four horizons, is 0.284, i.e. 28.4\% of $\beta$; the loss therefore never
once entered its linear branch, and the Huber arm is mathematically identical to
$\tfrac{1}{2}\times$MSE. We report it as a two-loss comparison. The arm is not
uninformative, however: because Adam's coupled $L_2$ decay does not scale with the loss,
halving the task gradient while leaving the decay pull intact doubles the decay's
relative influence, and this arm consequently trained longest (36 epochs of retained
weights, against 10 for MSE) and collapsed hardest, terminating with embedding RMS of
$1.5\times10^{-8}$ against a Xavier reference of 0.2885, a bit-exactly uniform adjacency,
and a gradient reaching the embeddings of \emph{exactly} zero. An unintended loss-scale
ablation thus independently corroborates the weight-decay mechanism argued for in
Sections~\ref{sec:collapse} and~\ref{sec:graphfix}.

\textbf{A wider hyperparameter search does not rescue the model, and confirms the
original search was adequate.} The study reported in Section~\ref{sec:hpo} searched five
parameters over 10 trials at fixed $K=5$, batch size 32 and weight decay $10^{-5}$. We
re-ran a study three times larger and with three additional dimensions---$K \in
\{3,5,7,9\}$, weight decay (log-uniform, $10^{-6}$ to $10^{-3}$) and batch size $\in
\{16,32,64\}$---for 30 trials under an identical objective (best validation average MSE
over 25 epochs). Twelve trials completed and 18 were pruned. The best configuration found
(hidden dimension 128, 8 heads, learning rate $\approx 0.00277$, dropout $\approx 0.0904$,
2 GNN layers, weight decay $\approx 2.0\times10^{-4}$, batch size 16, $K=9$) attains
validation MSE 0.0020145 against the deployed configuration's 0.0020509---an improvement
of \textbf{1.78\%}, with only 2 of the 12 completed trials beating the deployed value at
all. That margin is smaller than this study's 3.12\% across-seed spread and far smaller
than its 7.54\% same-configuration replicate spread, and part of it is attributable to
optimizer step count rather than configuration quality, since the winning trial halves
the batch size within a fixed epoch budget. We therefore report the wider search as a
confirmatory negative: the original five-parameter, ten-trial configuration was already
effectively optimal for this objective, and we do not re-run the main results at the new
one. Three caveats attach. First, all 18 pruned trials were pruned after a single epoch,
the median pruner having been constructed without warmup steps, so 60\% of the search
budget was screened on one epoch of validation loss---which structurally favors the
high-learning-rate, small-batch region the survivors occupy. Second, the sampler
concentrated on $K = 9$ (8 of 12 completed trials), and batch size is fully aliased with
$K$ in the completed set, so the per-parameter marginals and importances are confounded
and we do not interpret them; independently, the band-count sweep's converged test
results do not reproduce the search's implied $K$ ordering, and span less than the noise
floor. Third, and most consequential for this paper's argument, \textbf{the search
objective is structurally blind to the collapse}: it scores only validation MSE, and in
our 19 diagnosed cells the near-initialization checkpoints score squarely mid-range among
the collapsed ones (training length correlates with test average MSE at only
$\rho = 0.456$, $p = 0.05$). Collapse is free under this objective, so no widening of the
search space could select against it. We did not capture embedding diagnostics for
individual search trials and therefore cannot state directly whether the collapse
persists at the wider optimum; we note only, as inference from the mechanism rather than
as measurement, that the selected configuration's weight decay is 20 times the value used
elsewhere in this paper, which points toward more severe collapse rather than less.

Taken together, these checks close the most obvious escape route from
Sections~\ref{sec:collapse} and~\ref{sec:graphfix}'s negative findings. The collapse is
not a property of $K=5$, of VMD specifically, of the squared-error loss, of a single
seed, or of an insufficiently searched hyperparameter space. It is a property of what
happens to these embeddings under the optimizer once training proceeds past its first
epoch, and it does not depend on any modeling choice we were able to vary.
```

---

## 11. Reproducing this analysis

```
python analyze.py    # 19-cell cross-experiment table, correlations, multiseed test, residual check
python hpo.py        # Optuna study load, trial listing, marginals, importances
python verify2.py    # best_epoch correction, replicate identity, Huber checkpoint state
```
(scripts saved alongside this document in `paper1/latest_results/`; every number above traces to `latest_results/` artefacts —
`.jsonl` rows, `.pt` checkpoint state dicts, `.npy` prediction arrays, and `hpo_wider_study.db`.)
