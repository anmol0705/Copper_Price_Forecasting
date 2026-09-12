# VMD-MFGNN Finalization Plan — FINALIZED

Scope for this pass: **dataset expansion (N=8→15, revised down from 16 — see §1
addendum) + GATv2 swap + mechanism-diagnostic experiments (§3, §3b) + multi-horizon
architecture change (§3c, new) + faithful benchmark reproduction (§7, new) + text
fixes.** Star-graph work is explicitly excluded from this pass per user instruction
(tracked separately in `docs/todo.txt` item 9/graph-topology research for later).

---

## TL;DR — what's implemented, what's remaining (updated 2026-09-12)

**All CODE for this pass is now implemented and independently re-verified** (real
Python execution, not just agent self-report) for every section below except the
FRED-download re-run. **Nothing has been retrained yet** — every number currently in
`main.tex` still reflects the old 8-variable, GAT, single-shared-LSTM architecture,
and is now known to be additionally confounded by a graph-batching bug (see below).
**No text in `main.tex` beyond a few disclosure sentences has been updated** — the
architecture description, all result tables, the baselines section, and the collapse
narrative are all still describing the pre-this-pass system, because there are no new
numbers to put in them yet.

### Implemented and committed
| Item | Commit(s) |
|---|---|
| GATv2 swap (`vmd_mfgnn.py`) | `5e1db90` |
| Temperature-parameterized graph-fix (`tau=exp(s)`, no-decay group) | `5e1db90` |
| Multi-horizon architecture: shared graph, 20 per-horizon LSTMs | `5e1db90` |
| Multi-horizon loss reweighting | `5e1db90` |
| Faithful MTGNN reproduction (ported from `nnzhan/MTGNN`) | `e2a9e74`, `a7e5495` |
| VMD-LSTM corrected to genuinely univariate-per-mode | `e2a9e74` |
| N=8→15 dataset expansion code (tickers, FRED wiring) | `6d94bd0` |
| ALI=F removed entirely (2014-start bug found + fixed) | `6d94bd0`, `76c5285` |
| CLP=X bad-tick fix (2016-12-22, interpolated) | `6d94bd0` |
| Gradient-diagnostic empty-match-vs-zero fix | `94e36c4` |
| Null-control experiment (real/shuffle/gaussian arms) | `94e36c4` |
| **CRITICAL: `_batch_edge_index` cross-sample wiring bug, fixed** | `debce76` |
| `min_epochs` floor threaded into all 7 baselines (+ a self-caught bug in that fix) | `4ff1b0e`, `c9d6fde` |

### Remaining — blocking, must happen before any retraining is trusted
- [ ] **Nothing else code-wise blocks retraining.** The `_batch_edge_index` fix is the
      one item that had to land before any run would be meaningful, and it's done.
- [ ] Full real FRED+yfinance download has **not** been re-run against the final,
      fixed N=15 pipeline. The only real full-pull test that happened ran against the
      pre-fix N=16-with-ALI=F version. §1.1 below.
- [ ] Delete/rebuild `data/raw_prices.csv` and any cached VMD-mode files so the N=15
      pipeline is actually exercised, not silently cache-hit against stale N=8 data.
- [ ] Move aside (do not delete) `results/checkpoints/` before the next training run —
      needed as a before/after reference for how much the batching-bug fix changed
      results, which the paper's own transparency section may want to cite.

### Remaining — not blocking, but real and tracked
- [ ] `main.tex` text updates for everything above (architecture section — GATv2,
      temperature fix, 20-LSTM structure + param-count disclosure; baselines section —
      MTGNN now faithful, VMD-LSTM fix; Limitations — two-degenerate-regimes framing,
      decay-coefficient-independence note). All deliberately deferred until real
      numbers exist to put next to them — see §5.
- [ ] `main.tex` Section V-E "gradient exactly 0.0" claim — now known to be described
      by float32 underflow (7.9e-85 in float64), not a structural zero, per the fixed
      diagnostic. Needs re-measurement against the *actual* archived graph-fix
      checkpoints with the patched diagnostic (only synthetic/quick checkpoints have
      been checked so far) before rewriting the sentence.
- [ ] Bibitem for GATv2 (Brody/Alon/Yahav ICLR 2022) and for the MTGNN GitHub repo —
      not yet added, low-cost, do alongside the text updates above.
- [ ] Confirm no `num_vars: 8` hardcode remains in `configs/default.yaml` or elsewhere
      (checked clean as of when this plan was written; re-check after real N=15 pull).
- [ ] `vmd-mfgnn-protocol/SKILL.md` "locked 8-variable scope" line — stale, text-only.

### Remaining — requires actual training (the big one)
- [ ] One consolidated training run (Colab or equivalent): rebuild N=15 dataset for
      real, HPO against the new §3c architecture, main results (7 baselines + MTGNN),
      ablation, robustness suite, null-control experiment (now meaningful — previously
      produced bitwise-identical results across all three arms, which is what led to
      finding the batching bug), epoch-count archiving, temperature-fix ablation
      variant.
- [ ] Whether the paper's central "isotropic collapse" / "no graph advantage"
      narrative survives now that the graph is correctly wired at `batch_size=32` is
      **unknown** and cannot be determined without this run. Section V may need a
      substantial rewrite depending on the outcome — this is the single biggest open
      question in the whole paper right now.
- [ ] Full `main.tex` rewrite of every results-bearing section once real numbers exist.
- [ ] Re-run the adversarial review pass on the rewritten sections before calling this
      done.

---

## ADDENDUM (post-agent-review): N=16 revised to N=15, two real data bugs fixed

The dataset-expansion agent's real, full-history download surfaced two serious,
previously-unknown data-quality issues, both now fixed in `src/data_pipeline.py`
(commit `6d94bd0`):

1. **`ALI=F` (aluminum) actually starts 2014-05-06 on Yahoo Finance, not 2010** —
   despite every other ticker in this study (original 7 + all 4 newly-added) starting
   2010-01-04. Because the pipeline drops any row with a NaN in any column, this was
   **silently truncating the entire panel to 2014–2025 for the ENTIRE life of this
   project, including the original N=8 published results** — a ~25% row-count loss
   hidden behind `main.tex`'s stated "January 2010" / "~4,000 trading days" claims.
   paper2 independently found and flagged the same issue on 2026-08-19
   (`cubench_data_availability.md`) but resolved it differently there (tree models,
   unlike this project's VMD+neural pipeline, tolerate missing values).
   **Decision: `ALI=F` removed entirely** rather than shifting this whole paper's
   2010–2019/2020–2021/2022–2025 splits (referenced throughout all 33 pages) to
   2014+. This drops the target variable count from the originally-planned N=16 to
   **N=15**. §4's original "keep ALI=F, disclose liquidity caveat" decision is
   superseded — the Limitations item has been rewritten to describe the removal and
   its reason (commit `76c5285`).
2. **`CLP=X` (Chilean peso) has one verified bad tick** on 2016-12-22 (value 5.0
   against a ~670 neighborhood on every adjacent day), inside the training split.
   Fixed via linear interpolation from the immediate neighboring trading days —
   applied narrowly to this one disclosed, verified point, not a general outlier
   scrubber.

**Every mention of "N=16" elsewhere in this plan below should be read as N=15.** The
variable list in §1's table is otherwise unchanged (13 new-scope rows remain: 7
original + silver/clp/fcx/fxi + baa10y/dfii10/ppiaco/indpro; aluminum's row is struck).

Status legend: `[ ]` not started · `[~]` in progress / partially done · `[x]` done

---

## 0. Already done (before this plan was written, no action needed)

- [x] `src/fred_utils.py` created — ported and adapted from `paper2/src/cubench/data.py`
      + `pit.py` (curl-based FRED fetch, coverage-regression hard-assert, PIT join for
      monthly series).
- [x] `src/data_pipeline.py` `TICKERS` dict edited — added `silver` (`SI=F`), `clp`
      (`CLP=X`), `fcx` (`FCX`), `fxi` (`FXI`); aluminum later removed entirely (see
      addendum). `FRED_DAILY_SERIES` / `FRED_MONTHLY_SERIES` name lists added as module
      constants, now wired in (§1.1).

---

## 1. Dataset expansion: N=8 → N=15

Final variable list (7 surviving original + 8 new; aluminum removed, see addendum),
all verified against real data this session (either directly, or reused from
`paper2/docs/correlation_feasibility_findings.md` + `paper2/docs/cubench_data_availability.md`,
both already-run and already-verified):

| # | Name | Source | Ticker/Series ID | Status |
|---|---|---|---|---|
| 1 | copper (target) | yfinance | `HG=F` | existing |
| 2 | ~~aluminum~~ | ~~yfinance~~ | ~~`ALI=F`~~ | **removed — 2014 start-date bug, see addendum** |
| 3 | gold | yfinance | `GC=F` | existing |
| 4 | oil | yfinance | `CL=F` | existing |
| 5 | DXY | yfinance | `DX-Y.NYB` | existing |
| 6 | S&P 500 | yfinance | `^GSPC` | existing |
| 7 | VIX | yfinance | `^VIX` | existing |
| 8 | US10Y (nominal) | yfinance | `^TNX` | existing |
| 9 | silver | yfinance | `SI=F` | new — r=0.44, most sign-stable metal (paper2-verified) |
| 10 | Chilean peso | yfinance | `CLP=X` | new — producer-currency signal, bad tick fixed (addendum) |
| 11 | Freeport-McMoRan | yfinance | `FCX` | new — copper-supply equity signal, 4,198 rows, zero truncation |
| 12 | iShares China Large-Cap | yfinance | `FXI` | new — China-demand proxy, 4,198 rows, zero truncation |
| 13 | Baa credit spread | FRED (curl) | `BAA10Y` | new — daily, replaces truncated `BAMLH0A0HYM2` |
| 14 | 10Y TIPS real yield | FRED (curl) | `DFII10` | new — daily, Granger-significant at all 3 lags |
| 15 | PPI, all commodities | FRED (curl), monthly, PIT-joined +30d lag | `PPIACO` | new |
| 16 | Industrial production | FRED (curl), monthly, PIT-joined +22d lag | `INDPRO` | new |

(Table numbering kept at 16 rows for traceability; row 2 is struck, so the live
variable count is **15**.)

**Explicitly rejected** (researched, not included, reasons logged in `QNA_LOG.md`):
China PMI/`BSCICP03CNM665S` (r=0.01, not significant); `BAMLH0A0HYM2` (FRED-truncated
to 754 obs); `CPER` (near-tautological with the target); `COPX`/`MCHI` (shorter
history than `FCX`/`FXI`, would truncate the panel).

### 1.1 Code changes

- [x] `src/data_pipeline.py`: `TICKERS` dict expanded, aluminum removed.
- [x] `src/data_pipeline.py`: `DataDownloader.download()` now calls a new
      `_download_fred_extensions()` method — pulls `BAA10Y`/`DFII10` via
      `merge_asof(direction="backward")`, `PPIACO`/`INDPRO` via
      `fred_utils.pit_join_monthly()`, with fail-loud validation (row count must be
      exactly unchanged post-merge, all 4 columns present, none all-NaN). Committed
      `6d94bd0`.
- [x] CLP=X bad-tick fix and ALI=F removal both applied inside `download()`, fail-loud
      cache-staleness guard updated to check for both missing and unexpected columns.
- [ ] **Full real download has not been re-run against this final N=15 pipeline.** The
      dataset agent's own full-pull test (verified (3014, 16) rows, 2014-05-06→
      2025-12-30) ran *before* the ALI=F fix, i.e. against the buggy N=16-with-aluminum
      version — that run is exactly what revealed the bug, and its output is now
      stale by construction. Needs a fresh real pull post-fix.
- [ ] Confirm the final merged N=15 panel's date range doesn't shrink below 2010-01-04
      once actually pulled (expected not to, since `DFII10`/`BAA10Y` both start
      ≤2010-01-05 per paper2's verification, but not yet empirically confirmed against
      the fixed pipeline).
- [x] `configs/default.yaml` — no `num_vars: 8` hardcode found (checked when this plan
      was written); tickers block extended by the dataset agent in parallel with the
      core-architecture agent's own edits, both sets of edits coexist without conflict.
- [ ] `vmd-mfgnn-protocol/SKILL.md` "locked 8-variable scope" line — still stale,
      text-only fix, not yet done.

### 1.2 What this invalidates (full re-run required, not incremental)

Every result in the paper is tied to the old 8-variable dataset AND the old (buggy)
graph-batching behavior. Once the real N=15 pull happens, **all of the following must
be regenerated from scratch**, not patched:
- Main results table (Table IV) and all model comparisons (now 8 models incl. faithful
  MTGNN)
- HPO (search space assumes the old architecture scale; needs re-running against the
  new 20-LSTM multi-horizon structure)
- Ablation study (Table VIII)
- Robustness suite (19-cell collapse-invariance measurements)
- Collapse diagnosis numbers throughout Section V — **now additionally uncertain
  because of the graph-batching fix, not just the dataset change**
- Graph-fix experiment (re-run with the new temperature-parameterized variant too)
- Economic significance / DM tests
- Null-control experiment (code-ready, not yet run for real)

This is the largest single remaining item and dominates the total compute/time
budget. Everything else in this plan is already code-complete and should be batched
into the **same** full re-run.

---

## 2. GATv2 swap — DONE

- [x] `src/models/vmd_mfgnn.py`: `GATv2Conv` imported alongside `GATConv`, swapped
      into the GAT layer construction site with identical constructor arguments
      (confirmed true drop-in). Committed `5e1db90`.
- [x] `src/models/pooled_graph_mfgnn.py`: inherits the swap automatically (reuses
      `FrequencyBandModule`/`BandGraphEncoder` unmodified) — independently verified,
      not just assumed.
- [ ] `main.tex`: architecture description (Section III) still says "GAT", not
      GATv2 — deliberately deferred, since Section III will need a full rewrite once
      real GATv2 numbers exist anyway (§1.2). Do this as part of that rewrite, not
      before.
- [ ] Bibitem for GATv2 (Brody/Alon/Yahav, ICLR 2022) — not yet added.

---

## 3. Mechanism-diagnostic experiments — code done, none run for real yet

- [x] **Null-control experiment**: `src/experiments.py` — `run_null_control_experiment`,
      `build_null_control_data`, `_make_null_inputs`, `_materialize_loader` added.
      Independently verified: real/shuffle/gaussian arms behave exactly as documented
      (target channel preserved in both null arms, non-target channels genuinely
      altered, shuffle preserves each channel's value set while reordering samples).
      Committed `94e36c4`. **Not yet run for real** — its original smoke test produced
      bitwise-identical results across all three arms, which is what led to finding the
      `_batch_edge_index` bug; now that the bug is fixed, this experiment is safe and
      meaningful to run.
- [x] **Close the self-disclosed epoch-count gap** — code-side prerequisite done: the
      `min_epochs` floor (already implemented for VMD-MFGNN) is now also threaded
      through all 7 baseline models' training loop (`base_cfg`, `TorchBaseline.fit()`),
      closing a gap where baselines were the only main-table models not honoring the
      floor. Committed `4ff1b0e`, corrected `c9d6fde` (see that commit message for a
      self-caught bug in the first version of this fix). **Actually re-running the 7
      main-results models and archiving per-model effective-epoch counts has not
      happened yet** — needs the same consolidated training run as everything else.
- [x] **Multi-horizon loss reweighting**: implemented as part of the §3c architecture
      change (`_NAIVE_RMSE_REFERENCE`, `_HORIZON_WEIGHTINGS`, `_normalize_weights()`,
      `_weights_from_naive_rmse()`, new `training.horizon_loss_weighting` config,
      validated against `{"none","target_variance","naive_rmse"}`). Committed `5e1db90`.
      Currently defaults to `"none"` in `configs/default.yaml` — needs to be set/tuned
      during the HPO pass, not decided yet.

---

## 3c. Multi-horizon architecture — DONE: shared graph, per-horizon LSTM

Resolved the "combined vs. one-horizon-at-a-time" question. Neither extreme: kept VMD
decomposition and the per-band graph+GAT construction **fully shared** across all 4
horizons (so the graph still receives the same combined 4-horizon gradient it always
did — no additional collapse risk from this change specifically), but gave each
horizon its **own** LSTM, fusion, and prediction head downstream of that shared graph
output (20 LSTMs — 5 bands × 4 horizons — instead of 5).

- [x] `src/models/vmd_mfgnn.py`: `FrequencyBandModule` split into a shared
      `BandGraphEncoder` (graph+GAT, unchanged gradient path) and a new
      `HorizonBranch` class (per-horizon LSTM×5 + LayerNorm×5 + own `AttentionFusion`
      + own head), instantiated once per horizon inside a `horizon_branches:
      ModuleDict`. `FrequencyBandModule` now subclasses `BandGraphEncoder` (kept
      deliberately, to preserve flat parameter names for `PooledGraphMFGNN`
      compatibility).
- [x] `VMDMFGNN.forward()` updated: builds the shared graph+GAT once, extracts
      copper's sequence, routes into all 4 horizon branches. Return contract
      unchanged (`{str(h): (batch,)}`).
- [x] Independently re-verified: forward/backward pass clean (0/213 params without
      gradient at a test config), `PooledGraphMFGNN` confirmed to still work and
      inherit the same code path (0/37 params without gradient at B=32).
      Committed `5e1db90`.
- [ ] **Parameter-count/complexity disclosure**: real impact measured —
      394,756 → 1,448,708 params (+267%), mostly from the 20-LSTM structure. This
      **must** be written into `main.tex`'s complexity-analysis section prominently,
      not as a footnote — not yet done, deferred to the same text-rewrite pass as
      everything else in §1.2.
- [ ] Needs its own full HPO/main-results/ablation re-run — folds into the single
      consolidated training run, not a separate one.

**Also found and fixed during this work, not originally in scope:** the
`_batch_edge_index` critical bug (see the new top-of-plan item and below).

---

## 3b. Collapse-mechanism deep-dive — code done

### 3b.1 Diagnostic bug — DONE

- [x] `src/diagnostics.py`'s `gradient_magnitude_diagnostic` now also returns
      `matched_param_count`, `matched_param_names`, `matched_grad_none_count`, so a
      `0.0` return can be told apart from "no parameters matched." No existing key
      changed. Committed `94e36c4`.
- [x] Independently re-verified: on a fresh model, `matched_param_count` is always 10
      (full model)/2 (pooled), never 0, in every checkpoint tested — the original
      "empty-match" hypothesis for the "gradient exactly 0.0" claim is **wrong**. The
      real explanation, found instead: one checkpoint's exact 0.0 is float32 underflow
      of a true value of 7.9e-85 measured in float64.
- [ ] `main.tex` Section V-E still says "gradient reaches exactly 0.0" — **not yet
      corrected**. Needs re-measurement against the actual archived graph-fix
      checkpoints (only synthetic/quick checkpoints have been checked so far), then a
      rewrite to state the real float32-underflow explanation.

### 3b.2 Temperature-parameterized normalized embeddings — DONE

- [x] `src/models/vmd_mfgnn.py`'s `FrequencyGraphConstructor`: new
      `use_graph_temperature` flag, `self.log_temperature = nn.Parameter(torch.zeros(()))`
      per band, `tau = exp(log_temperature)` (never a raw `nn.Parameter`, which would
      self-destruct via the same collapse mechanism — confirmed by simulation this
      session). New `_adjacency_logits_to_adj()` helper used by both `forward()` and
      `get_adjacency()`.
- [x] `log_temperature` added to the same no-decay optimizer group as `emb1`/`emb2`
      in `src/trainer.py` (`no_decay_graph_embeddings` matcher extended). Committed
      `5e1db90`.
- [x] Independently re-verified: bitwise-identical adjacency at init (temperature
      starts at 1.0, no behavior change until trained), correct no-decay group
      membership confirmed.
- [ ] `configs/default.yaml`: `model.use_graph_temperature: false` by default — not
      yet flipped on for a real experiment run.
- [ ] New ablation/experiment variant (`full_model_graphfix_temp` /
      `pooled_graph_graphfix_temp`) alongside the existing graph-fix variants — **not
      yet added** as a distinct entry in the experiment/ablation runner. The
      underlying flag exists; wiring it into the ablation-study variant list is a
      small remaining step before the consolidated run.
- [ ] Multi-seed validation (this session's research found single-seed embedding
      norms vary ~4000x across seeds of the same config) — not yet run, needs the
      same training pass.

### 3b.3 Ruled out this session, don't implement — reference only, no action

AdamW (mathematically equivalent to the existing no-decay fix at this project's
weight-decay magnitude), BatchNorm/LayerNorm directly on the embedding table (category
error — no batch axis), scaled-init-alone (real but partial, delay not cure),
correlation-based pretraining alone (bigger head start toward the same collapse, not a
fix). Full reasoning preserved for the paper's own text; no code changes intended.

---

## 4. Data-quality fix: `ALI=F` — DECIDED (revised): removed entirely

**Original decision (superseded):** keep `ALI=F`, disclose a liquidity caveat
(63% zero-volume days, high>low on only 15.4% of days) — this was based on incomplete
information.

**Revised decision, made after the dataset-expansion agent's real full-history pull
found the 2014-05-06 true-start-date bug (see addendum):** `ALI=F` is **removed from
the input scope entirely**, not kept-with-caveat, because the liquidity issue turned
out to be the smaller of two problems — the real one was that its true start date was
silently truncating the *entire panel* to 2014–2025 via `dropna()`, contradicting the
paper's own stated "January 2010" / "~4,000 trading days" claims for every result ever
produced, including the already-published N=8 numbers.

- [x] `ALI=F` removed from `TICKERS` in `src/data_pipeline.py`, with an extensive
      code comment explaining the bug, its discovery, and the decision rationale.
      Committed `6d94bd0`.
- [x] `main.tex` Limitations item rewritten to describe the removal (not a liquidity
      disclosure) and clarify it affects only the expanded-scope (N=15) results, not
      retroactively the already-reported 2010–2025 window for the original N=8 study.
      Committed `76c5285`. Recompiled clean (pdflatex twice), citation integrity
      re-verified.

---

## 5. Text-only fixes

- [x] MBTI-Net bibliography title fix — done.
- [x] Static-vs-dynamic-graph limitation clause — done.
- [x] VMD boundary-effect Limitations item — done.
- [x] Feature-set-expansion future-work item — done.
- [x] ALI=F Limitations item (revised version) — done, see §4.
- [ ] "Two degenerate regimes" finding (collapsed-uniform under decay vs.
      frozen-random without it — neither is a learned graph) into Section V — **not
      yet written**, deliberately deferred: this claim needs to be re-checked against
      the graph-batching fix before being restated as-is, since the original finding
      was measured under the broken batching behavior.
- [ ] Weight-decay-coefficient-independence refinement — same status, same reason:
      not yet written, needs re-verification post-batching-fix before being restated.
- [ ] GATv2 architecture description + citation (§2), MTGNN baseline disclosure
      update (§7), parameter-count disclosure (§3c), Section V-E gradient claim
      (§3b.1) — all deferred to the single post-retraining rewrite pass, tracked
      above, not duplicated here.

---

## 6. Explicitly NOT in this pass

- Star-graph topology variant — user instruction: not now, tracked separately in
  `docs/todo.txt` item 9.
- GATv2-vs-edge-weight-redundancy ablation — folded into the GATv2 swap (§2) as a
  straight swap, not a separate with/without-edge_attr ablation variant.
- Distributional/pinball loss (`docs/todo.txt` item 10) — deferred.
- MODWT robustness arm — deferred.
- Full MBTI-Net reproduction — already assessed NO-GO this session (VMD likely leaks;
  behavioral-signal data requires paywalled Baidu Index + Wind terminal access, neither
  accessible; this project's existing `pooled_graph_mfgnn.py` ablation is already the
  "minimal honest version" a from-scratch reproduction would have produced).

---

## 7. Faithful benchmark reproduction — DONE: MTGNN

Ported from the real public repo `nnzhan/MTGNN` (Wu et al., KDD 2020) rather than
reimplemented from the paper description: saturated top-k graph learning,
anti-symmetric adjacency (M1M2ᵀ − M2M1ᵀ), mix-hop propagation, dilated-inception TCN
(kernel set {2,3,6,7}), gated filter/gate, skip connections, node-indexed LayerNorm.

- [x] `src/models/baselines.py`: new classes `_MTGNNNConv`, `_MTGNNLinear`,
      `_MTGNNMixProp`, `_MTGNNDilatedInception`, `_MTGNNGraphConstructor`,
      `_MTGNNLayerNorm`, `_MTGNNNet`, and public `MTGNNBaseline(TorchBaseline)`.
      Deliberate, disclosed deviations from the repo defaults: `in_dim=1` (vs 2),
      `out_dim=len(horizons)` reading node 0 (copper), `seq_length=lookback` (60 vs
      12), `subgraph_size` clamped to `min(20, num_vars−1)`. Committed `e2a9e74`.
- [x] `SimpleMTGNN` kept as a separate row (not replaced) — a genuine second,
      lower-capacity graph-learning baseline now that the faithful version exists
      alongside it.
- [x] `VMDLSTMBaseline` corrected to genuinely univariate-per-mode (was feeding all
      N variables jointly into each per-mode LSTM): `nn.ModuleList` of per-mode LSTMs,
      each consuming only the target (copper) channel. Docstring's cited year/journal
      corrected. Committed `e2a9e74`.
- [x] `src/models/__init__.py` exports `MTGNNBaseline`.
- [x] `src/trainer.py`: `MTGNNBaseline` imported and added to `raw_baseline_ctors`
      alongside (not replacing) the existing `SimpleMTGNN` entry. Committed `a7e5495`.
- [x] Independently re-verified: forward `(4,4)` output shape at N=15, finite;
      backward reaches 88/94 params (the 6 gradient-free `residual_convs` params
      confirmed correct — only active in the `gcn_true=False` fallback path, matching
      the real upstream architecture's own conditional structure, not a porting bug).
      VMD-LSTM: forward correct at 4D `(B,T,K,N)` input, 0/80 params without gradient,
      explicit univariate-invariance test (perturbing non-target variables by +1000.0
      changed output by only 5.96e-08).
- [ ] `main.tex` Section IV-B (Baselines) disclosure paragraph — still describes the
      old "six baselines, SimpleMTGNN/VMD-LSTM both simplified" framing. Needs
      updating to name MTGNN as a faithful reproduction and describe the VMD-LSTM fix
      — deferred to the post-retraining text-rewrite pass, since it needs real numbers
      for all 8 models anyway.
- [ ] Bibitem for the MTGNN GitHub repo (code provenance) alongside the existing
      `Wu2020mtgnn` citation — not yet added.

---

## NEW, discovered during implementation — not in original scope, now resolved but consequential

**`_batch_edge_index` never wired within-sample graph edges at `batch_size>1`.** Found
by the diagnostics agent (the null-control experiment's three arms were producing
bitwise-identical results — the trigger for investigating), independently reproduced
from scratch with concrete tensor values before trusting it (at B=2, N=4, edges
{0→1,1→2,2→3}, the batched result had zero real within-sample edges — every edge
pointed across samples instead), then fixed (`.permute(1,0,2)` before the reshape) and
re-verified at B=1,2,3,32 — 0 cross-sample edges at every batch size, including the
project's real configured `batch_size=32`. `PooledGraphMFGNN` confirmed to inherit the
fix automatically. Committed `debce76`.

**Implication, stated plainly:** every previously archived experimental result in this
project's history — main results, ablation table, robustness suite, and the original
graph-fix/collapse experiment — used a graph mechanism that never propagated real
cross-variable structure at `batch_size>1`. A reviewer who found this independently
would call those results void, not merely confounded. Whether the paper's central
"isotropic collapse" / "no graph advantage in ablation" narrative survives with the
graph correctly wired is **unknown** and can only be resolved by real retraining —
this is now the single biggest open question in the paper, ahead of everything else in
this plan.

---

## Execution order from here

**Everything code-side is done.** What's left:

1. Real N=15 FRED+yfinance download against the fixed pipeline (§1.1) — confirm date
   range, confirm no silent truncation.
2. Delete/rebuild `data/raw_prices.csv` and cached VMD-mode files.
3. Move aside (don't delete) `results/checkpoints/` for a before/after reference.
4. Set `model.use_graph_temperature: true` and add the `*_graphfix_temp` ablation
   variant before the run, or run it as a follow-up pass — decide before starting.
5. One consolidated training session: N=15 dataset, HPO against the §3c architecture,
   main results (8 models incl. faithful MTGNN), ablation, robustness suite,
   null-control experiment, epoch-count archiving, temperature-fix variant.
6. Pull results back, rewrite every affected section of `main.tex` (architecture,
   main results, ablation, robustness, collapse diagnosis incl. the corrected
   "gradient exactly 0.0" claim, baselines, complexity/param-count disclosure,
   GATv2 + MTGNN-repo bibitems).
7. Re-run the adversarial review pass (statistical/architecture/narrative) on the
   rewritten sections before calling this done.

**Time-boxing note**: the training run is realistically hours, not minutes — 15
variables and the 20-LSTM architecture both add real cost over the old 8-variable,
5-LSTM setup, and every experiment in the pipeline (HPO, main, ablation, robustness×19
cells, null-control×3 arms) needs to run against the new data and architecture. All
code is ready to fire in one session; this plan does not compress the wall-clock time
itself.
