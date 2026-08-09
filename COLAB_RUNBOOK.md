# Colab Runbook — VMD-MFGNN v2 Full-Scale Run

Companion doc for `notebooks/vmd_mfgnn_v2_colab.ipynb`. The notebook contains
this same content in its "RUNBOOK" markdown cell; this file is a copy for
offline reading. If the two ever diverge, the notebook cell is authoritative
since that's what a human actually runs.

## Expected runtime (best-effort estimates — see assumptions)

Assumptions: Colab T4 (16GB) or better GPU, `hidden_dim=64`,
`num_gnn_layers=2`, `num_heads=4`, `temporal_layers=2`, dataset is ~2,500
train-day windows (2010-2019), `batch_size=32` (~80 batches/epoch),
`epochs=200` with `patience=20` early stopping. These are architecturally
small models (hidden_dim 64) over a modest number of samples, so per-epoch
cost is low; the dominant cost is epoch *count*, gated by early stopping,
which is data-dependent and cannot be predicted exactly.

| Stage | Estimate | Notes |
|---|---|---|
| Data download (10 tickers, yfinance) | 1-3 min | Network-bound, small CSVs |
| VMD decomposition (expanding window, refit every 21 days, ~10 vars x ~4000 days -> ~190 refits/var) | 15-45 min | CPU-bound (`vmdpy`), no GPU benefit; the single biggest *fixed* cost since it is not parallelized across variables |
| Hyperparameter optimization (optional, `hpo.enabled: true`) | +~20-55 min | Only runs if opted in via `configs/default.yaml`; see "Optional HPO stage" note below |
| VMD-MFGNN training | 10-30 min | Small model; bounded mostly by early-stopping epoch count |
| LSTM / Transformer / SimpleMTGNN / VMD-LSTM (each) | 5-20 min each | Similar order of magnitude to VMD-MFGNN |
| XGBoost | 2-8 min | One `XGBRegressor` per horizon (4 total), CPU-bound |
| ARIMA | 5-20 min | Walk-forward: refits a small ARIMA(p,d,q) per test-window (~750+ windows), CPU-bound; possibly the slowest baseline |
| Ablations (6 variants -- 3 studies, 2 run capacity-matched and -unmatched) | 45-135 min total | Same order as individual training runs, x1.5 variant count vs. the original 4-variant design |
| Significance tests + figure generation | 1-5 min | Cheap, CPU-bound |
| **Total, rough (HPO off)** | **~2-5 hours** | Dominated by early-stopping epoch counts and VMD's fixed CPU cost, neither precisely predictable ahead of time. Budget for at least one Colab disconnect. |
| **Total, rough (HPO on)** | **~2.5-6 hours** | Adds the HPO stage above on top of the HPO-off total. |

These are architecture-based estimates, not measurements (built without GPU
or network access in the packaging environment). Watch the actual per-epoch
timing `src/trainer.py` logs every 10 epochs once running, and re-budget
from there.

#### Optional HPO stage — runtime estimate reasoning

`configs/default.yaml`'s `hpo:` block defaults to `enabled: false` (no extra
cost). If set to `true` with the shipped defaults (`n_trials=15`,
`trial_epochs=25`), `run_hpo()` trains up to `15 * 25 = 375` "trial epochs"
of VMD-MFGNN (Optuna's median pruner will cut some trials short, so this is
a ceiling, not a guarantee). The main VMD-MFGNN training run above budgets
for up to `epochs=200`, and its estimated 10-30 min already covers that full
budget in the worst case. Scaling proportionally: `375 / 200 ≈ 1.875`, so
the HPO stage adds roughly `1.875 * (10-30 min) ≈ 19-56 min`, rounded here
to **~20-55 min**. In practice this is likely an overestimate since (a)
pruning stops unpromising trials early and (b) some trial configs (e.g.
`hidden_dim=32`) are cheaper per epoch than the default `hidden_dim=64`.

### Checkpointing (trainer-native)

VMD-MFGNN's and each ablation variant's training is checkpointed via
`VMDMFGNNTrainer.fit(..., checkpoint_path=...)` (added to `src/trainer.py`
after this notebook was first packaged): `fit()` itself checks whether the
given `.pt` path already exists, loads and skips training if so, and
otherwise saves the best-so-far state dict there as validation loss
improves. This replaced the notebook's original hand-rolled "check if a
`.pt` file exists, manually `torch.load`/`torch.save`" logic in the
VMD-MFGNN and ablation cells — that logic is now redundant and has been
removed, leaving less notebook-side code to maintain. The notebook's own
responsibility is now only to (a) point `checkpoint_path` at a location
under `results/checkpoints/` and (b) call `sync_to_drive(('results',))`
afterward, since the trainer only writes locally and has no notion of
Google Drive. The 6 baselines still use their own manual "predictions
already on disk" resume check (below), since baseline `.fit()` methods have
no `checkpoint_path` parameter.

## What to do if Colab disconnects

- **During data/VMD stage:** Re-run the "STEP 1" data cell. It checks Google
  Drive first (via `restore_from_drive()`) for a previously-saved
  `vmd_modes.npy` + `vmd_modes_meta.json`; if found and the metadata matches
  (same date range/params/data hash — this check lives in
  `src/data_pipeline.py`'s `build_vmd_modes()`), it loads from Drive instead
  of recomputing. If the raw price CSV was already downloaded, that's reused
  too. Worst case (nothing on Drive yet), you only lose this stage's
  progress.
- **During the optional HPO stage:** `run_hpo()` itself has no internal
  trial-by-trial checkpointing, but the STEP 1.5 HPO cell is resume-aware at
  the whole-search level: it checks whether `results/hpo_best_params.json`
  already exists on disk (e.g. restored from Drive by `restore_from_drive()`
  earlier in the notebook) before calling `run_hpo()`. If found, it loads the
  previously-completed winner directly (logged as `[resume] Found existing
  results/hpo_best_params.json ...`) and skips the search entirely — a
  disconnect *after* HPO finished and synced to Drive costs nothing on
  re-run. If interrupted mid-search (no `hpo_best_params.json` on disk yet),
  re-running the HPO cell restarts the whole search from trial 0 — Optuna's
  own trial state isn't persisted, so there is no mid-search resume. If
  you'd rather not lose partial progress, disable HPO for the retry
  (`hpo.enabled: false` in `configs/default.yaml`) and use its default
  `model:`/`training:` values instead.
- **During training stage:** Re-run the STEP 2 training cell. VMD-MFGNN is
  resumed via `VMDMFGNNTrainer.fit(checkpoint_path=...)` (see "Checkpointing"
  above): if `results/checkpoints/vmd_mfgnn.pt` already exists (e.g.
  restored from Drive), training is skipped and those weights are loaded
  directly. Each of the 6 baselines is checked individually against Drive
  via its own predictions-exist check: if a baseline's prediction arrays
  already exist, training is skipped for that model and its test metrics
  are recomputed from the saved predictions instead. Only models that
  hadn't finished when the disconnect happened will retrain. **Caveat:**
  `src/trainer.py`'s own `run_all_experiments()` only writes
  `results/all_results.json` once, at the very end, after every model
  finishes, and has no built-in resume; this notebook does not call that
  function directly for the main training stage — it drives the same
  underlying building blocks (`VMDMFGNNTrainer`, `run_baseline`, same model
  classes and `base_cfg`) in a resumable loop instead, without modifying
  `src/trainer.py`.
- **During ablation stage:** Same idea. Each of the 6 ablation variants
  (`full_model`, `no_vmd_raw_price_matched_dim`,
  `no_vmd_raw_price_matched_params`, `pooled_graph_matched_dim`,
  `pooled_graph_matched_params`, `correlation_graph` — the pooled-graph and
  no-VMD ablations each run twice, at a capacity-unmatched and a
  capacity-matched `hidden_dim`, per the post-review capacity-matching fix)
  is checkpointed via `VMDMFGNNTrainer.fit(checkpoint_path=...)` pointed at
  `results/checkpoints/{name}.pt` (the same paths `run_ablation_studies()`
  itself uses), then `results/` is synced to Drive right after each variant
  finishes. `run_ablation_studies()` in `src/trainer.py` has no per-variant
  checkpointing of its own (only one `save_results()` call at the very end),
  so the notebook reimplements its 6 variants inline — importing the exact
  same helpers it uses (`VMDMFGNN`, `PooledGraphMFGNN`, `count_parameters`,
  `_UnsqueezeModeWrapper`, `_find_matched_hidden_dim`, `VMDMFGNNTrainer`).
- **During figures/significance stage:** Cheap to just re-run; nothing here
  is individually checkpointed since the whole stage should take well under
  5 minutes once the model and predictions already exist.

## Known risks to watch for

1. **Zinc/nickel ticker substitution.** `src/data_pipeline.py` uses
   `^NQCIZNER` (zinc) and `^NQCINIER` (nickel) — NASDAQ Commodity
   sub-indices — because no standalone Yahoo Finance futures ticker exists
   for LME-only zinc/nickel (`ZNC=F`/`NI=F` do not resolve to real
   instruments). **Verify, once real network access is available, that
   these two tickers actually return non-empty daily data for the full
   2010-2025 range** — the STEP 1 data cell logs each ticker's returned row
   count and explicitly flags if `zinc`/`nickel` ends up missing from
   `data['variable_names']`. If either comes back empty/short, decide before
   re-running between: (a) dropping that variable from the 10-var set and
   documenting a 9-variable scope deviation, or (b) finding and substituting
   another genuine Yahoo Finance-listed proxy. Do not silently reintroduce
   `ZNC=F`/`NI=F` — they don't resolve.
2. **NaNs in the significance table.** At tiny smoke-test scale, prior
   verification runs saw 2/24 NaN entries in
   `results/significance_table.json` (likely `diebold_mariano_test`'s
   `var_d <= 0` degeneracy guard in `src/utils.py` tripping on a very
   small/degenerate sample). This may or may not reproduce at full scale —
   full scale has far more test-set samples per horizon, which should make
   the HAC variance estimate less degenerate. The notebook's STEP 2
   significance cell already checks for and prints any NaN entries; **after
   the real run, also manually open `results/significance_table.json` and
   confirm before treating the paper's significance claims as final.**
3. This notebook does not modify `src/*.py`, `scripts/*.py`, or
   `configs/*.yaml` — it imports and orchestrates them as-is. To change
   hyperparameters, edit `configs/default.yaml` in the repo (or the
   uploaded/cloned copy) before running, not in the notebook. To opt into
   HPO, set `hpo.enabled: true` in that same file before running.
4. **HPO tunes VMD-MFGNN only** (see the notebook's STEP 1.5 markdown cell
   and `src/hpo.py`'s module docstring) — the 6 locked baselines and all 6
   ablation variants are never tuned by HPO. This is a deliberate, disclosed
   scope limitation, not an oversight. HPO trial results (for reporting in
   the paper) are saved to `results/hpo_best_params.json` (winning
   hyperparameters) and `results/hpo_trials.csv` (full trial history).

## Verification performed when this notebook was packaged

Packaged and verified in a sandboxed environment with **no GPU and no
network access to Yahoo Finance**, so the following could NOT be run and are
not claimed to be verified end-to-end:
- Actual GPU training of any model.
- Actual Yahoo Finance downloads (including whether the zinc/nickel proxy
  tickers return valid data).
- Actual wall-clock timing of any stage.

What WAS verified:
- The `.ipynb` file is well-formed JSON (`json.load` succeeds) with valid
  nbformat 4.5 structure.
- Every code cell's source parses as valid Python (`ast.parse`, after
  stripping `!`/`%` Colab-magic lines).
- Every function/class call the notebook makes against the repo
  (`load_config`, `create_datasets`, `VMDMFGNNTrainer`, `run_baseline`,
  `run_significance_tests`, `print_results_table`, `VMDMFGNN`,
  `PooledGraphMFGNN`, `_UnsqueezeModeWrapper`, `generate_all_figures`, etc.)
  was checked against its actual signature by reading `src/utils.py`,
  `src/data_pipeline.py`, `src/trainer.py`, `src/models/vmd_mfgnn.py`,
  `src/models/pooled_graph_mfgnn.py`, `src/models/baselines.py`, and
  `src/visualize.py` directly — not guessed from memory.
