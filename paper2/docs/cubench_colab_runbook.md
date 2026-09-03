# CuBench Colab Runbook

Companion reference for `paper2/notebooks/cubench_colab.ipynb` — the priority Phase 6
deliverable that runs the full CuBench pipeline to 100% grid completion. Local compute
ran out of throughput/time budget partway through the full experimental grid (see
`STATUS.md`, search "LOCAL GRID TRAINING STOPPED"); this notebook is the real path to
finishing it, not a formality.

**Repo layout note (post-restructure):** all CuBench code, data, and results now live
under `paper2/` in the repo (`paper1/` holds the separate VMD-MFGNN paper). The Colab
notebook `%cd`s into `paper2/` right after cloning, so every relative path below
(`src/cubench/...`, `scripts/cubench_*.py`, `results/cubench/...`, `data/cubench/...`,
`tests/test_leakage.py`) is relative to `paper2/` as the working directory — both in
the notebook and for local terminal commands run from `D:\copper\paper2`.

This doc covers: cell order, expected runtime per section, what to do if a section
fails, how resume works, and how to get final results back into `D:\copper\paper2`. It does
not duplicate the notebook's own inline markdown — read both, but this is the
one-page version to keep open while the run is in progress.

---

## 1. Before you start

- Use a **standard (CPU) Colab runtime**. Nothing in this pipeline uses a GPU (plan
  decision D12) — do not pay for/request a GPU runtime, it buys nothing here.
- Have a Google account with Drive access ready; the notebook mounts Drive in the
  first cell and uses it for incremental backup.
- Know your repo's branch. As of this writing the CuBench code lives on
  `graph-fix-experiment` (not necessarily `main`) — the notebook's clone cell has a
  `BRANCH` variable at the top; check it matches wherever the code actually is before
  running.

## 2. Cell order and expected runtime

Run cells top to bottom in one pass on a fresh session. Sections are independently
re-runnable (each reads its own real inputs and either skips already-done work or
regenerates cleanly from a fixed function of its inputs), so if a session dies, just
reconnect and re-run from wherever the notebook left off — you do not need to restart
from cell 1 except to re-mount Drive and re-clone.

| Section | What it does | Estimated wall-clock | Resumable? |
|---|---|---|---|
| 1. Setup | Mount Drive, clone repo, install deps (incl. the pypbo manual-copy workaround) | 5–10 min | Yes — re-run freely, idempotent |
| 2. Data acquisition | Real yfinance + curl-FRED pulls, FRED coverage assertions | 2–5 min | Yes — re-pulls fresh each time (cheap) |
| 3. Feature engineering | `cubench_build_features.py` + known-good shape assertion (4023×81) | 1–3 min | Yes |
| 4. Leakage tests | `pytest tests/test_leakage.py`, hard gate | ~6–10 min (382s observed locally) | Yes, but see below |
| 5. Full grid training | nulls → econometric → linear → trees → deep, incremental save + Drive sync | **6–14 hours total** (see breakdown below) | **Yes — this is the point of the whole design** |
| 6. Statistical evaluation | `cubench_evaluate.py` (DM/PT/Holm/backtest/base-rate) | 10 min – a few hours depending on grid coverage and whether `--ablation-full` completes within budget | **Yes — every layer, including ablations, resumes from disk by default** |
| 7. Full ablations (A0–A7) | `cubench_evaluate.py --ablation-full`, no time cap beyond what you set | **Up to several hours** | **Yes — incremental per-cell jsonl resume, same design as Section 5** |
| 8. Figures | `cubench_make_figures.py` | 1–5 min | Yes |
| 9. Package/download | zip `results/cubench/`, download + Drive copy | <1 min | Yes |

**Section 5 breakdown** (based on locally-observed per-cell throughput before the
local machine's throughput ceiling was hit):
- nulls + econometric + linear: 10–30 min combined — these were already 100% complete
  locally and should reproduce here almost instantly, serving as an end-to-end sanity
  check that the Colab environment behaves like the dev machine.
- trees (lgbm, xgboost, catboost, randomforest — run **sequentially**, one model at a
  time, deliberately, to avoid the CPU contention that was diagnosed locally as the
  actual throughput bottleneck): **4–10 hours** for the full 3×3×11×5 = 495-cell grid
  per model family. This is the dominant cost of the whole notebook.
- deep (lstm, transformer): **1–3 hours**, but note this only covers a
  **restricted subgrid** (targets {t1,t3}, horizon h=1 only, 66 cells/model) — see the
  discrepancy note below, this is not the full grid the plan's model roster table
  describes, and this notebook cannot widen it without a code change outside Phase 6's
  scope.

Colab's free tier typically enforces a ~12-hour (often less, and variable) session
cap, so **Section 5 will very likely span 2+ sessions**. This is expected, not a
failure — that is exactly why the incremental-jsonl + Drive-sync + resume design
exists (see Section 4 below).

## 3. If a section fails

- **Section 2 (data)**: a curl/yfinance failure usually means a transient network
  issue or a genuine upstream outage. Re-run the cell. If FRED coverage assertions
  fail (a series now serves less history than expected), this is a **real finding**,
  not a bug to route around — it happened once already with `BAMLH0A0HYM2` (see plan
  D4/R2). Do not proceed past a coverage regression; investigate which series
  regressed and whether the downstream feature/model set needs to change (a decision
  for the user, not something to silently patch in the notebook).
- **Section 4 (leakage tests)**: **do not proceed to Section 5 on a failure.** A
  failure here means either a genuine environment-dependent bug (different
  pandas/numpy point-release behavior between the dev machine and Colab's installed
  versions) or a real leakage bug the local partial run never exercised because it
  never got far enough. Read the pytest output, identify which of the 7 gates failed,
  and treat it exactly as seriously as the equivalent local failure would be treated —
  this project's whole discipline exists to prevent silently training on a matrix that
  peeks at the future.
- **Section 5 (grid training)**: a disconnect or interruption here loses **at most the
  one in-flight cell** — everything already written to
  `results/cubench/grid_results.jsonl` is safe (flushed to disk immediately after each
  cell, per `src/cubench/walkforward.py::run_grid`) and was additionally synced to
  Drive after each family finished. Reconnect, re-mount Drive, re-clone/pull the repo,
  re-run the "restore from Drive" cell, and re-run whichever family cell you were on —
  it will skip everything already done and continue from there. See Section 4 below
  for exactly how this was verified to work.
- **Section 6/7 (evaluation/ablations)**: `cubench_evaluate.py` is now resumable by
  default at the level of every individual layer, not just "safe to re-run wholesale."
  Each statistical/backtest/regime layer checks whether its own output file already
  exists on disk and, if so, loads it back into memory instead of recomputing — a
  disconnect between layers loses at most whatever layer was mid-computation when the
  session died. The **ablation layer** (`run_ablations` in `cubench_evaluate.py`) uses
  the same incremental-jsonl-with-resume pattern as Section 5's main grid: every
  `(rung, target, horizon, fold, seed)` cell is appended to
  `results/cubench/ablations/ablation_results.jsonl` and flushed immediately after it's
  fit, and any cell already present in that file on restart is skipped rather than
  retrained. The backward-compatible `ablation_results.json` is rebuilt by aggregating
  the jsonl on every run, so it always reflects the true on-disk state even mid-run.
  Just reconnect and re-run the same evaluation cell — it resumes automatically; no
  special handling needed. Pass `--force` only if you deliberately want to discard all
  cached layer outputs (including the ablation jsonl) and recompute everything from
  scratch.
- **Section 8 (figures)**: `cubench_make_figures.py` reads only existing Phase 4
  outputs + `features.parquet`, so it's safe to re-run any time, including on partial
  grid coverage (it labels sparse figures explicitly rather than pretending they're
  complete — see the docstring in the script itself).

## 4. How resume actually works (and how it was verified)

The mechanism is entirely inside `src/cubench/walkforward.py::run_grid`, unmodified
by this phase:

1. At start-up, it reads `results/cubench/grid_results.jsonl` line by line and builds
   a `done` set of `(model, target, horizon, fold, seed)` keys.
2. For every cell in the full grid it's asked to run, if the key is already in `done`,
   it's skipped — no retraining, no wasted compute.
3. Every completed cell is written as one JSON line, appended and **flushed
   immediately** (`out.flush()`), not buffered until the process exits.

**This was tested for real, not just read and trusted**, against the local partial
grid (`results/cubench/grid_results.jsonl`, 1543 lines at the time of the test):

- Ran `python scripts/cubench_run_grid.py --family trees --model xgboost --targets t1
  --horizons 22` (a genuinely incomplete cell: 40 of 55 expected xgboost/t1/h=22 cells
  present) under a 25-second timeout to force an interruption mid-run. It produced 5
  new lines (40→45) before being killed — proving incremental flush-per-cell works,
  not just flush-at-exit.
- Re-ran the identical command. It correctly skipped the 45 already-done cells and
  continued from there, adding only new ones — proving the dedup/skip logic works on
  a genuinely interrupted, resumed run, not just a clean restart.
- Checked for duplicate `(model,target,horizon,fold,seed)` keys across the whole file
  before and after: **16 pre-existing duplicate keys** (unrelated to this test — from
  an earlier point in local history, most likely two concurrent local processes
  racing to append) were present both before and after; the resume test itself added
  **zero new duplicates**. `scripts/cubench_aggregate.py` dedups by key (last line
  wins) when building `all_results.json`, so pre-existing duplicates are harmless to
  final numbers — but **this notebook deliberately runs tree-model families
  sequentially, one Colab cell at a time**, specifically to avoid recreating that race
  condition (which only occurs when two processes append to the same file
  concurrently — a single sequential process, which is all this notebook ever runs,
  cannot race against itself).

Net: resume is real, tested with an actual kill-and-restart, and safe against
duplication under the sequential-execution pattern this notebook uses.

## 5. Getting results back into `D:\copper`

1. Section 9's download cell produces `/content/cubench_results.zip` (a zip of the
   entire `results/cubench/` tree — predictions, significance tables, backtest,
   ablations, regimes, figures, the aggregated `all_results.json`, and the raw
   `grid_results.jsonl`) and also copies it to
   `MyDrive/cubench_backup/cubench_results.zip` as a fallback if the browser download
   doesn't trigger.
2. Locally:
   ```
   powershell -Command "Expand-Archive -Path cubench_results.zip -DestinationPath D:\copper\paper2\results\cubench -Force"
   ```
3. Optionally re-run `scripts/cubench_make_figures.py` locally (from `D:\copper\paper2`
   as the working directory) against the downloaded data to confirm it reproduces the
   same figures from the same code — this is the project's standing "recompute from
   raw" discipline, and it's a cheap (1–5 min) check.
4. If you want the executed notebook itself in the repo for the record (useful for
   showing real output, not just code): File → Download → `.ipynb` in Colab, replace
   `paper2/notebooks/cubench_colab.ipynb` locally, and commit both the notebook and the new
   `results/cubench/` contents together.

## 6. Known discrepancies found while packaging (not silently fixed)

Per the Phase 6 brief, anything that looked like a code issue while packaging was
reported here rather than patched, since a fix at this stage should be a deliberate,
reviewed decision:

- **Deep-learning grid is a documented subset, not the full 3×3 grid.**
  `src/cubench/models/deep.py` and `scripts/cubench_run_grid.py` restrict lstm/
  transformer to targets `{t1, t3}` and horizon `h=1` only (T2 and h∈{5,22} are
  skipped), with an in-code rationale (full grid = 594 NN trainings, judged infeasible
  in one local session). This is hardcoded, not a notebook-level choice — even with
  Colab's larger time budget, this notebook's deep-learning section will only ever
  cover 66 cells/model (11 folds × 3 seeds × 2 targets × 1 horizon), not the full grid
  the plan's Section 3 model roster table implies for a 12-model-family × 3-target ×
  3-horizon benchmark.
- **Deep-learning training budget was also cut** from the plan's `epochs=100,
  patience=20` to `MAX_EPOCHS=40, PATIENCE=8` in `deep.py`, documented in-code as a
  response to local CPU contention from concurrently-running tree families. Colab
  running tree families sequentially (as this notebook does) removes that specific
  contention, but the epoch/patience values are still hardcoded lower than the plan
  specifies — full restoration to 100/20 would be a deliberate code change outside
  this phase's scope.
- **Small pre-existing duplicate-key count in `grid_results.jsonl`** (16 duplicate
  `(model,target,horizon,fold,seed)` keys found in the local partial-grid file,
  harmless because aggregation dedups by key) — most likely caused by two local
  processes appending concurrently at some point in local history. Not a bug in the
  resume logic itself (verified above), just a reason the notebook insists on running
  tree families sequentially rather than in parallel Colab cells.
- **`cubench_evaluate.py`'s ablation layer previously had no incremental resume**
  (flagged during Phase 6 packaging) — this has since been fixed: it now uses the same
  per-cell jsonl resume discipline as the main grid (see Section 3 above and
  `results/cubench/ablations/ablation_results.jsonl`). Verified with a real
  force-killed-mid-run test: partial cells survived, the resumed run skipped them with
  zero duplicate keys, and the aggregated `ablation_results.json` reflected the
  combined state correctly.

None of the above required modifying `src/cubench/*.py`, `configs/*.yaml`, or
`requirements*.txt` — the notebook works around none of them silently; it documents
them inline (in both the notebook markdown and here) and lets the actual grid/ablation
runs reflect the shipped code's real, current behavior.

## 7. Verification performed when this notebook was packaged

- `notebooks/cubench_colab.ipynb` validated as well-formed via `nbformat.validate()`
  (53 cells: 30 code, 23 markdown).
- Locally executed, against the real repo, the exact logic of: the required-files
  check, the import smoke test (lightgbm/xgboost/catboost/arch/shap/pypbo/
  dieboldmariano — all import cleanly in `.venv_corr`), the FRED first-observation-date
  coverage assertions (against the real cached `data/cubench/raw/*.csv`, all pass),
  the feature-matrix shape/date-range assertion (real `data/cubench/features.parquet`,
  confirmed `(4023, 81)`, `2010-01-04` → `2025-12-30`), the Drive-sync helper and the
  progress-checker helper (against the real `results/cubench/grid_results.jsonl`).
- Ran the real `tests/test_leakage.py` suite locally in `.venv_corr`:
  **10 passed, 34 warnings in 382.17s** — the exact command Section 4 of the notebook
  runs.
- Ran real (not simulated) resume/interruption testing on
  `scripts/cubench_run_grid.py` as described in Section 4 above.
