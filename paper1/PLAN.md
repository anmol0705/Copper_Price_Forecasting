# VMD-MFGNN Finalization Plan — FINALIZED

Scope for this pass: **dataset expansion (N=8→15, revised down from 16 — see §1
addendum) + GATv2 swap + mechanism-diagnostic experiments (§3, §3b) + multi-horizon
architecture change (§3c, new) + faithful benchmark reproduction (§7, new) + text
fixes.** Star-graph work is explicitly excluded from this pass per user instruction
(tracked separately in `docs/todo.txt` item 9/graph-topology research for later).

**IMPLEMENTATION STATUS (updated live as agents land and are independently
re-verified, not just trusted on their own report):**
- [x] §2 GATv2 swap — done, independently re-verified, committed (`5e1db90`).
- [x] §3b.2 Temperature-parameterized graph-fix — done, independently re-verified
      (bitwise-identical adjacency at init, correct no-decay group membership,
      mechanism confirmed to uncap edge-weight spread), committed (`5e1db90`).
- [x] §3c Multi-horizon architecture (shared graph, per-horizon LSTM) — done,
      independently re-verified (forward/backward pass, PooledGraphMFGNN unaffected,
      real GATv2 confirmed in use), committed (`5e1db90`). **Parameter count impact:
      394,756 → 1,448,708 (+267%), mostly from the 20-LSTM architecture — needs
      prominent disclosure in the paper's complexity section, not a footnote.**
- [x] §3 Multi-horizon loss reweighting — done as part of the above, committed (`5e1db90`).
- [x] §7 Faithful MTGNN reproduction + VMD-LSTM univariate fix — done, independently
      re-verified at N=15 (forward/backward pass, univariate-invariance test showing
      6e-8 max diff when non-target variables are perturbed by +1000), trainer.py
      wiring completed, committed (`e2a9e74`, `a7e5495`).
- [x] §4 ALI=F — **decision revised mid-implementation, see addendum below.**
- [ ] §1 Dataset expansion — **done but revised down to N=15, see addendum below.**
      Code committed (`6d94bd0`); full real download not yet re-run against the final
      15-variable + CLP-fix pipeline (the agent's own full-pull test ran against the
      pre-fix N=16-with-ALI=F version).
- [x] §3b.1 Diagnostic bug fix — done, independently re-verified (`matched_param_count`
      always populated, never silently 0; confirmed the "gradient exactly 0.0" claim's
      real cause is float32 underflow on at least one checkpoint — 7.9e-85 in float64,
      NOT the empty-match ambiguity originally suspected), committed (`94e36c4`).
- [x] Null-control experiment (real/shuffle/gaussian arms) — code done, independently
      re-verified (`_make_null_inputs` behaves exactly as documented for all three
      modes), committed (`94e36c4`). **Not yet run for real** — needs the graph-batching
      fix below to be meaningful (previously produced bitwise-identical results across
      all three arms, which is what led to discovering that bug).

**NEW, DISCOVERED DURING IMPLEMENTATION — NOT IN ORIGINAL SCOPE, NOW TOP PRIORITY:**
- [x] **CRITICAL FIX: `_batch_edge_index` never wired within-sample graph edges at
      batch_size>1.** Found by the diagnostics agent (null-control's three arms were
      bitwise-identical — the trigger for investigating), independently reproduced from
      scratch with concrete tensor values before trusting it, then fixed
      (`.permute(1,0,2)` before the reshape) and re-verified at B=1,2,3,32 (0
      cross-sample edges at every batch size, including the project's real
      `batch_size=32`). `PooledGraphMFGNN` confirmed to inherit the fix automatically
      (shares the same code path via inheritance). Committed (`debce76`).
      **Implication: every previously archived experimental result in this project's
      history — main results, ablation table, robustness suite, the original
      graph-fix/collapse experiment — used a graph mechanism that never propagated real
      cross-variable structure at batch_size>1. This directly confounds (does not
      necessarily invalidate, but confounds) the paper's central "isotropic collapse"
      and "no graph advantage in ablation" narrative. Whether collapse still happens
      with correctly-wired graphs can only be determined by real retraining — cannot be
      resolved from CPU smoke tests. This must happen before any other retraining, and
      the paper's Section V (collapse narrative) is likely to need a substantial rewrite
      depending on the outcome.**
- [ ] `min_epochs` floor is threaded for `VMDMFGNNTrainer`/ablations but NOT for the
      7 baseline models (`base_cfg` omits it; `TorchBaseline.fit()` has no such concept).
      Flagged by the diagnostics agent, not yet fixed. Affects completeness of the
      epoch-count-confound closure (currently only 1 of 8 main-table models honors the
      floor).

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
variable list in §1's table is otherwise unchanged (13 rows remain: 7 original +
silver/clp/fcx/fxi + baa10y/dfii10/ppiaco/indpro; aluminum's row is struck).

Status legend: `[ ]` not started · `[~]` in progress / partially done · `[x]` done

---

## 0. Already done (before this plan was written, no action needed)

- [x] `src/fred_utils.py` created — ported and adapted from `paper2/src/cubench/data.py`
      + `pit.py` (curl-based FRED fetch, coverage-regression hard-assert, PIT join for
      monthly series). Not yet imported/used anywhere.
- [x] `src/data_pipeline.py` `TICKERS` dict edited — added `silver` (`SI=F`), `clp`
      (`CLP=X`), `fcx` (`FCX`), `fxi` (`FXI`). `FRED_DAILY_SERIES` /
      `FRED_MONTHLY_SERIES` name lists added as module constants. **Not yet wired into
      `DataDownloader.download()` — the FRED series are declared but nothing pulls
      them yet.** This is the first pending code item below.

---

## 1. Dataset expansion: N=8 → N=16

Final variable list (8 existing + 8 new), all verified against real data this session
(either directly, or reused from `paper2/docs/correlation_feasibility_findings.md` +
`paper2/docs/cubench_data_availability.md`, both already-run and already-verified):

| # | Name | Source | Ticker/Series ID | Status |
|---|---|---|---|---|
| 1 | copper (target) | yfinance | `HG=F` | existing |
| 2 | aluminum | yfinance | `ALI=F` | existing — **liquidity caveat, see §4** |
| 3 | gold | yfinance | `GC=F` | existing |
| 4 | oil | yfinance | `CL=F` | existing |
| 5 | DXY | yfinance | `DX-Y.NYB` | existing |
| 6 | S&P 500 | yfinance | `^GSPC` | existing |
| 7 | VIX | yfinance | `^VIX` | existing |
| 8 | US10Y (nominal) | yfinance | `^TNX` | existing |
| 9 | silver | yfinance | `SI=F` | **new** — r=0.44, most sign-stable metal (paper2-verified) |
| 10 | Chilean peso | yfinance | `CLP=X` | **new** — producer-currency signal, 4,164 obs verified |
| 11 | Freeport-McMoRan | yfinance | `FCX` | **new** — copper-supply equity signal, 4,198 rows, zero truncation (verified this session) |
| 12 | iShares China Large-Cap | yfinance | `FXI` | **new** — China-demand proxy, 4,198 rows, zero truncation (verified this session) |
| 13 | Baa credit spread | FRED (curl) | `BAA10Y` | **new** — daily, full history from 2009, replaces the truncated `BAMLH0A0HYM2` (paper2-verified) |
| 14 | 10Y TIPS real yield | FRED (curl) | `DFII10` | **new** — daily, weak linear corr but Granger-significant at all 3 lags (paper2-verified) |
| 15 | PPI, all commodities | FRED (curl), monthly, PIT-joined +30d lag | `PPIACO` | **new** (paper2-verified) |
| 16 | Industrial production | FRED (curl), monthly, PIT-joined +22d lag | `INDPRO` | **new** (paper2-verified) |

**Explicitly rejected** (researched, not included, reasons logged in `QNA_LOG.md`):
China PMI/`BSCICP03CNM665S` (r=0.01, not significant, paper2's own recommendation is
ablation-only negative result, not a headline feature); `BAMLH0A0HYM2` (FRED-truncated
to 754 obs); `CPER` (copper futures ETF — near-tautological with the target, not a
genuine new signal); `COPX`/`MCHI` (shorter history than `FCX`/`FXI`, would truncate the
panel, largely redundant with them).

### 1.1 Code changes needed

- [x] `src/data_pipeline.py`: `TICKERS` dict expanded (done above).
- [ ] `src/data_pipeline.py`: `DataDownloader.download()` — after the existing
      yfinance loop builds `df`, add a new step that:
  1. Pulls `BAA10Y` and `DFII10` via `fred_utils.fetch_fred_series_curl`, calls
     `fred_utils.verify_fred_coverage()` on each (hard-fails on regression), merges
     each directly onto `df` by date (`pd.merge_asof` or a plain date-indexed join —
     no PIT lag needed, these are daily series published same-day).
  2. Pulls `PPIACO` and `INDPRO` via the same fetch function, then merges each onto
     `df` via `fred_utils.pit_join_monthly()` — **not** `reindex().ffill()` — to avoid
     leaking a not-yet-published print into an earlier trading day.
  3. Keeps the existing fail-loudly convention: if any FRED series can't be pulled or
     fails its coverage check, raise rather than silently proceeding with 8 or 12
     variables instead of 16.
- [ ] Confirm the final merged `df`'s date range doesn't shrink below the current
      panel's start date because of a FRED series with a later first-obs (checked:
      `DFII10`/`BAA10Y` both start ≤2010-01-05 per paper2's verification, so this
      should be a non-issue, but verify empirically once actually pulled).
- [ ] Update `configs/default.yaml` if it hardcodes `num_vars: 8` anywhere (grep found
      none as of this plan being written, but re-check after the above changes).
- [ ] Update `vmd-mfgnn-protocol/SKILL.md` / any other doc that states the "locked
      8-variable scope" as current, since it will no longer be accurate. (Text-only.)

### 1.2 What this invalidates (full re-run required, not incremental)

Every result in the paper is tied to the current 8-variable dataset. Once N changes,
**all of the following must be regenerated from scratch**, not patched:
- Main results table (Table IV) and all 7 model comparisons
- HPO (search space assumes current architecture scale; may need re-running)
- Ablation study (Table VIII)
- Robustness suite (19-cell collapse-invariance measurements)
- Collapse diagnosis numbers throughout Section V
- Graph-fix experiment
- Economic significance / DM tests

This is the largest single change in this plan and dominates the total compute/time
budget. Everything else below (GATv2, null control, etc.) should be batched into the
**same** full re-run rather than run separately, since a full re-run is happening
regardless.

---

## 2. GATv2 swap

- [ ] `src/models/vmd_mfgnn.py`: change `from torch_geometric.nn import GATConv` to
      also import `GATv2Conv`; swap the `GATConv(...)` construction in
      `FrequencyBandModule.__init__` (currently line ~137) to `GATv2Conv(...)` with the
      same arguments (`in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout,
      concat=True, edge_dim=1`). Confirm `GATv2Conv`'s constructor signature matches
      (PyG's `GATv2Conv` is designed as a drop-in replacement, but verify no argument
      renames between versions installed in this project's environment).
- [ ] `src/models/pooled_graph_mfgnn.py`: same swap if it independently imports/uses
      `GATConv` (uses `FrequencyBandModule` from `vmd_mfgnn.py`, so likely inherits the
      swap automatically — verify, don't assume).
- [ ] `main.tex`: update the architecture description (Section III, wherever GAT is
      named) to say GATv2, add one sentence citing Brody/Alon/Yahav ICLR 2022 and
      stating why (static-attention limitation, same-cost strict improvement) —
      pre-empting the reviewer question rather than waiting for it.
- [ ] Add `Brody2022` or similar bibitem for the GATv2 paper.

---

## 3. Mechanism-diagnostic experiments (cheap, high-value, from this session's research)

- [ ] **Null-control experiment**: feed the per-band learned-weight mechanism
      shuffled/randomized channels instead of real VMD bands (new small experiment
      function, likely in `src/experiments.py` or a dedicated script). Decisive, cheap
      test of whether the collapse is downstream of decomposition choice at all.
- [ ] **Close the self-disclosed epoch-count gap**: re-run the 7 main-results models
      with the existing `min_epochs` floor (already implemented, currently defaults to
      0 for Table IV/Table VI per `main.tex` line 374) and archive per-model effective-
      epoch counts. Expected result: no change to the numbers, but converts an admitted
      limitation into a closed check.
- [ ] **Multi-horizon loss reweighting**: normalize each horizon's loss term by that
      horizon's target variance (or similar) in `src/trainer.py`'s loss computation
      (currently an unweighted sum across horizons, `trainer.py:197-198`), since h=22
      currently dominates the gradient ~18x by scale alone. Complementary to, not a
      substitute for, the architecture change in §3c below — the combined loss still
      backpropagates through the shared graph layer with this same scale imbalance
      regardless of what happens downstream.

---

## 3c. Multi-horizon architecture — DECIDED: shared graph, per-horizon LSTM (not full separation)

Resolves the "combined vs. one-horizon-at-a-time" question. **Neither extreme**:

- NOT full separation into 4 independent models (own graph+GAT+LSTM each) — this would
  make each graph receive gradient from only one horizon's loss instead of four
  combined, and the paper's own diagnosis is that this gradient is already too weak;
  splitting it four ways plausibly worsens collapse in every copy for no forecasting
  benefit, since the graph is already established as inert regardless.
- NOT the current fully-shared design either — one LSTM's output currently feeds all
  4 horizon heads, forcing one temporal representation to serve both 1-day
  (noise-dominated) and 22-day (trend-dominated) patterns, which the paper's own
  research (LSTM carries most of the real predictive signal, graph carries ~none)
  suggests is the wrong place to force sharing.

**Decision: keep VMD decomposition and the per-band graph+GAT construction FULLY
SHARED across all 4 horizons (unchanged gradient signal reaching the graph — no
additional risk to the collapse diagnosis), but give each horizon its own LSTM, fusion,
and prediction head downstream of that shared graph output.**

- [ ] `src/models/vmd_mfgnn.py`: split `FrequencyBandModule` into two pieces:
  1. A shared per-band graph+GAT feature extractor (unchanged: builds the learned
     adjacency, runs GAT, produces per-timestep per-band node representations) — this
     part stays IDENTICAL across all 4 horizons, so the graph still receives the same
     combined 4-horizon gradient signal it does today.
  2. A per-horizon LSTM + `AttentionFusion` + prediction head, instantiated 4 times
     (once per horizon in `horizons`), each consuming the shared per-band GAT output
     independently. I.e. 5 bands × 4 horizons = 20 LSTMs instead of the current 5,
     each small; graph/GAT parameter count unchanged.
- [ ] Update `VMDMFGNN.forward()` to route the shared GAT output into 4 separate
      per-horizon temporal+fusion+head pipelines instead of one shared pipeline
      feeding 4 heads.
- [ ] Parameter-count and complexity-analysis sections of `main.tex` (Section III-complexity)
      need updating to reflect the new per-horizon LSTM cost — report this explicitly,
      don't let it go undisclosed.
- [ ] This changes the model architecture significantly enough that it needs its own
      full HPO/main-results/ablation re-run alongside everything else in §1.2 — folds
      into the same consolidated Colab session, not a separate one.

---

## 3b. Collapse-mechanism deep-dive (from dedicated Opus research, this session) — HIGH PRIORITY

A focused research pass on "can normalization/init/pretraining/AdamW eliminate the
collapse" came back with one urgent bug-fix and one genuinely new, best-available fix.
Full reasoning in `QNA_LOG.md`.

### 3b.1 Diagnostic bug — fix before trusting or re-stating the "gradient exactly 0.0" claim

`src/diagnostics.py`'s `gradient_magnitude_diagnostic` does `if p.grad is None: continue`
when summing gradient norms by matching parameter name (`.emb1.`/`.emb2.`). **An empty
match set (zero parameters found) and a genuine zero gradient both return exactly
`0.0`** — the function cannot currently tell these apart. Isolated testing of the graph-
construction computation found a robustly nonzero gradient at realistic embedding scale
(norm 0.73), and the project's own checkpoints show embeddings moved during training
(cosine-to-init 0.993-0.996, not exactly 1.0) -- both facts are inconsistent with a
truly zero gradient. **The "gradient reaches exactly 0.0" sentence already written into
`main.tex`'s graph-fix section this session is very likely describing a diagnostic
artifact, not a real physical zero.**

- [ ] Patch `gradient_magnitude_diagnostic` (`src/diagnostics.py`) to also return a
      matched-parameter count, so `0.0` and "no match" are distinguishable in future runs.
- [ ] Re-run the diagnostic on the existing graph-fix checkpoints with the patched
      version before deciding what to write in the paper.
- [ ] Update `main.tex`'s Section V-E (graph-fix) to either (a) confirm a genuinely
      small-but-nonzero gradient and correct "exactly 0.0" to the real measured value, or
      (b) if the patched diagnostic somehow still returns a genuine zero, keep the claim
      but note it was verified against the parameter-count bug specifically.

### 3b.2 New experiment: temperature-parameterized normalized embeddings (best available fix)

The existing graph-fix (unit-normalize embeddings + exclude from weight decay) avoids
collapse but is capped at only ~2-2.7x ratio between the strongest and weakest edge
weight (quantified this session: the un-normalized frozen-random graph already sits at
~2.08x, so normalization alone buys almost nothing -- the missing ingredient is a
learnable softmax temperature). A naively-added raw temperature parameter would destroy
itself via the identical collapse mechanism (simulated: collapses to 0.00022 after 3000
steps of decay). The fix, worked out and simulated this session:

- [ ] `src/models/vmd_mfgnn.py`'s `FrequencyGraphConstructor`: add a learnable log-
      temperature parameter **parameterized as `tau = exp(s)`, never as a raw
      `nn.Parameter` directly** (simulated: `s` itself decays gracefully to near-zero
      under decay, i.e. `tau` stays near 1.0 and degrades slowly, versus a raw `tau`
      parameter which collapses to ~0.0002 and re-uniformizes the graph). Apply as
      `softmax(ReLU(E1 @ E2.T) / tau, dim=-1)`.
  - [ ] Add `s` to the same no-decay parameter group as `emb1`/`emb2` in
        `src/trainer.py`'s optimizer setup (the existing `no_decay_graph_embeddings`
        mechanism already does this for the embeddings -- extend the name-matching
        pattern to also catch the new temperature parameter).
  - [ ] Keep the existing `normalize_embeddings=True` behavior (unit-normalize E1/E2
        before the bilinear product) -- temperature is additive to this, not a
        replacement for it. Note for the paper: under normalization, the task
        gradient w.r.t. the embeddings is measured as exactly orthogonal to the
        embedding direction, meaning the task loss cannot defend embedding norm at
        all -- normalization keeps the *adjacency* differentiated even as norm itself
        continues to (harmlessly) decay. Worth one sentence in the mechanism writeup.
- [ ] New ablation/experiment variant (alongside the existing `full_model_graphfix` /
      `pooled_graph_graphfix`): same architecture with the temperature fix added,
      call it e.g. `full_model_graphfix_temp` / `pooled_graph_graphfix_temp`.
- [ ] Success metric is NOT "did it avoid collapse" (the existing graph-fix already
      avoids collapse and still learns nothing) -- it's (a) cosine-to-init well below
      1.0 (genuine directional movement, not just scale preservation) and (b) adjacency
      max/min ratio meaningfully above the ~2.08 a frozen-random graph already gets for
      free. Both are new checks to add to `src/diagnostics.py`'s reporting.
- [ ] Run multi-seed (this session's research found single-seed embedding norms vary
      ~4000x across seeds of the same config -- do not trust a single-seed result here).

### 3b.3 Ruled out this session, don't implement (with reasons, for the paper's own text)

- **AdamW (decoupled weight decay)**: mathematically and empirically equivalent to the
  already-implemented no-decay graph-fix at this project's weight-decay magnitude
  (simulated: identical trajectories to 4 decimal places). Not a new experiment --
  would just reproduce an existing result under a different name. Do not run as a
  separate arm; do not present as a distinct fix in the paper.
- **BatchNorm/LayerNorm directly on the embedding table**: category error -- the
  embedding table has no batch axis (it's a fixed parameter table read identically
  every forward pass, not a per-sample activation). LayerNorm on the derived adjacency
  logits is coherent but redundant with, and slightly worse-engineered than, the
  existing normalization fix. Confirmed via literature search this specific
  application is essentially absent from the 2023-2026 GNN literature.
- **Scaled initialization alone** (e.g. 2x Xavier): a real, quantified partial
  mitigation (buys ~20 extra epochs before collapse at this project's step count) but
  a delay, not a cure. Could mention as a robustness note in Limitations, not worth a
  dedicated experiment given 3b.2 is available.
- **Correlation-based pretraining/warm-start alone**: confirmed to be "a bigger head
  start toward the same collapse," not a real fix, unless combined with removing decay
  (in which case decay removal, not pretraining, is doing the work). Has a legitimate
  but different justification -- "inject structure the task gradient demonstrably can't
  find" -- which is a weaker, more honest claim than "we learned the graph." Not
  implementing this pass; flag as future work with the honest framing if mentioned.

---

## 4. Data-quality fix: `ALI=F` liquidity issue — DECIDED: Option A (disclose only)

Found this session: `ALI=F` (aluminum) is confirmed illiquid on Yahoo Finance — 63%
zero-volume days, high>low on only 15.4% of days. This affects the EXISTING, already-
published aluminum node, independent of the N=16 expansion.

**Decision: keep `ALI=F`, disclose the liquidity issue in Limitations.** Rationale: the
r=0.41 correlation is real and paper2-verified (a genuine literature-consistent signal,
not noise), the liquidity caveat is a one-sentence disclosure rather than a correctness
bug, and swapping introduces a new, unverified ticker into an already-large batch of
changes for no clear benefit — a reviewer is far more likely to respect "we checked,
here's the caveat" than to penalize a disclosed, real, if imperfect, signal.

- [x] Keep `ALI=F` in `TICKERS` (no code change needed here).
- [x] Add one sentence to `main.tex`'s Limitations — DONE, committed (`9ccffb8`).
      Verified: recompiled clean (33 pages, no errors), 41/41 citation integrity.

---

## 5. Text-only fixes (no retraining needed, can happen in parallel with the above)

- [ ] Write the "two degenerate regimes" finding into Section V (collapsed-uniform
      under weight decay vs. frozen-random without it — neither is a learned graph).
      Uses data already in the repo, no new experiment.
- [ ] Add the weight-decay-coefficient-independence mechanism refinement (collapse
      rate empirically independent of the decay coefficient's magnitude once the task
      gradient falls below it) — also from data already in the repo.
- [x] MBTI-Net bibliography title fix — done.
- [x] Static-vs-dynamic-graph limitation clause — done.

---

## 6. Explicitly NOT in this pass

- Star-graph topology variant (Variant A/B from the graph-topology research) — user
  instruction: not now, tracked separately.
- GATv2-vs-edge-weight-redundancy ablation (`docs/todo.txt` item 9b) — folded into the
  GATv2 swap above (§2) as a straight swap, not built as a separate with/without-
  edge_attr ablation variant, to keep this pass's scope bounded.
- Distributional/pinball loss (`docs/todo.txt` item 10) — deferred, doesn't touch the
  collapse mechanism, lower priority than the above.
- MODWT robustness arm — deferred, nice-to-have, not required for this pass.
- Full MBTI-Net reproduction — already assessed NO-GO this session (their VMD likely
  leaks; behavioral-signal data requires Baidu Index + a Wind terminal subscription,
  neither accessible; this project's existing `pooled_graph_mfgnn.py` ablation is
  already the "minimal honest version" a from-scratch reproduction would have
  produced). Not revisited. §7 below is a *different* decision — see there.

---

## 7. Faithful benchmark reproduction — DECIDED: reproduce MTGNN (official code exists)

**The gap this closes**: the benchmarking-methodology review earlier this session found
that two of the paper's six baselines (`SimpleMTGNN`, `VMD-LSTM`) are disclosed
*simplified* reimplementations, not faithful reproductions of the original published
architectures — a real, if honestly-disclosed, weakness. MBTI-Net (§6 above) can't be
fixed this way because it has no public code and needs paywalled data. **MTGNN is
different: its official code is public** (`github.com/nnzhan/MTGNN`, maintained by the
paper's lead author, Wu et al., KDD 2020) — verified this session, not assumed. This
makes a genuinely faithful reproduction feasible, unlike MBTI-Net.

MTGNN is also the single most relevant baseline to get right: this paper's own graph
construction explicitly follows MTGNN's adjacency parameterization (cited in Section
III), so a faithful MTGNN baseline is a stronger, more defensible comparison point than
the current simplified version, and directly answers "why isn't this the real MTGNN?"
before a reviewer asks it.

- [ ] Add a new baseline class in `src/models/baselines.py`, e.g. `MTGNNBaseline`,
      implementing the real architecture from the official repo: graph-learning layer
      with the saturated top-k mechanism (not this project's simpler top-k
      sparsification), mix-hop propagation layers, dilated-inception temporal
      convolution (not `SimpleMTGNN`'s two plain dilated causal convs), and output skip
      connections. Port/adapt from `nnzhan/MTGNN`'s public code rather than
      reimplementing from the paper description alone, to minimize the risk of a subtly
      incorrect reproduction.
  - [ ] Adapt only what's necessary for this project's data shape (N=16 variables,
        4 forecast horizons, this project's own train/val/test split and lookback
        window) — do not change MTGNN's own architectural hyperparameters/defaults
        without a documented reason.
  - [ ] Wire into the existing baseline-training pipeline (`run_baseline` / whichever
        function in `src/trainer.py` currently trains the other 6 baselines) so it's
        evaluated identically (same metrics, same DM significance testing, same
        sign-concentration check) as everything else.
- [ ] Keep `SimpleMTGNN` as a separate row (do not remove it) — it becomes a genuine
      second, lower-capacity graph-learning baseline once the real MTGNN exists
      alongside it, which is more informative than replacing one with the other.
- [ ] `src/models/baselines.py`'s `VMDLSTMBaseline`: smaller, cheaper correction (not a
      full new-paper reproduction) — the benchmarking review found it currently feeds
      all N variables jointly into each per-mode LSTM, whereas Liu et al. 2020's actual
      VMD-LSTM design is univariate per mode (one LSTM per decomposed variable). Fix
      this to match the cited paper's actual design, since it's currently mischaracterized
      in the docstring/citation, not just simplified.
- [ ] `main.tex` Section IV-B (Baselines): update the disclosure paragraph — MTGNN is
      now a faithful reproduction (name it as such, cite the official repo), SimpleMTGNN
      remains disclosed as a simplified variant kept for comparison, VMD-LSTM's fix is
      noted (now matches Liu et al. 2020's per-mode-univariate design).
- [ ] Add a bibitem/citation for the MTGNN GitHub repo alongside the existing
      `Wu2020mtgnn` paper citation, since code provenance matters for a reproduction claim.

---

## Execution order once you give the go-ahead

**Code changes (can all happen in parallel, none blocks another):**
1. Finish wiring FRED pulls into `DataDownloader.download()` (§1.1).
2. GATv2 swap (§2).
3. Diagnostic bug fix (§3b.1) — do this FIRST among the §3/§3b items, since it gates
   whether the "gradient exactly 0.0" claim needs correcting in the paper at all.
4. Null-control experiment, epoch-gap closure, multi-horizon loss reweighting (§3),
   the temperature-parameterized graph-fix (§3b.2), and the shared-graph/per-horizon-
   LSTM architecture change (§3c).
5. MTGNN faithful reproduction + VMD-LSTM per-mode-univariate fix (§7).
6. `ALI=F` Limitations disclosure sentence (§4) — text-only, no code, can happen anytime.

**Then, sequentially (each depends on the code above being done):**
7. Delete cached `data/raw_prices.csv` and any cached VMD-mode files so the expanded
   16-variable dataset is actually rebuilt, not silently skipped by the existing
   cache-hit logic.
8. One consolidated Colab session that: rebuilds the 16-variable dataset, re-runs HPO
   (now against the §3c architecture), main results (now including the faithful MTGNN
   baseline), ablation, robustness suite, and the new null-control + epoch-archiving +
   temperature-fix experiments.
9. Pull results back, rewrite every affected section of `main.tex` with real numbers
   (this is a substantial rewrite — architecture description, main results, ablation,
   robustness, collapse diagnosis, and baselines sections all get new numbers).
10. Re-run the adversarial review pass (statistical/architecture/narrative) on the
    rewritten sections before calling this done.

**This plan is final.** Every open decision point above has been resolved (dataset
list, ALI=F, the §3/§3b mechanism experiments, MTGNN as the benchmark-reproduction
target). Give the go-ahead and execution starts at step 1.

**Time-boxing note**: step 5 (the actual training) is realistically hours, not
minutes — 16 variables means VMD decomposition cost roughly doubles, and every
experiment in the existing pipeline (HPO, main, ablation, robustness×19 cells) needs
to run against the new data. This plan gets everything code-ready to fire in one
Colab session; it does not compress the training wall-clock time itself.
