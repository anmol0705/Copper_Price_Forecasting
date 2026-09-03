# Incident: predictions lost after a fully-completed Colab grid run (2026-08-25)

## What happened

The full CuBench grid (2,838 cells: model x target x horizon x fold x seed) was run
to completion on Google Colab via `notebooks/cubench_colab.ipynb`. The run reported
success and `results/cubench/grid_results.jsonl` downloaded with all 2,838 unique
cells present (verified by deduping on `(model, target, horizon, fold, seed)`). But
`results/cubench/predictions/` — the directory `src/cubench/walkforward.py::run_one`
writes one `.npy` array to per cell — downloaded **completely empty**: zero files.

This matters specifically for this project: CuBench has a standing, repeatedly-used
practice of independently re-verifying every reported number from the raw `.npy`
prediction arrays rather than trusting `grid_results.jsonl`'s aggregated metrics
blindly (this has caught real bugs at multiple prior phases — see `STATUS.md`).
Without the `.npy` files, that verification practice cannot be applied to this run's
results at all, even though the summary metrics survived.

## Why

The notebook's Drive-sync helpers (`sync_to_drive()` / `restore_from_drive_if_present()`,
in the "Drive sync helpers" cell of Section 5) only ever copied
`results/cubench/grid_results.jsonl` to `MyDrive/cubench_backup/`. They never touched
`results/cubench/predictions/`. `grid_results.jsonl` is a small append-only text file,
cheap to sync after every model family — so it reliably made it to Drive. The `.npy`
prediction arrays only ever existed on the Colab VM's local (ephemeral) disk. A
runtime restart/disconnect sometime between "training genuinely finished" and "the
final `results/cubench/` zip was built and downloaded" would explain exactly this
signature: summary metrics intact (already text-synced to Drive), raw arrays gone
(never persisted anywhere durable).

## Root-cause problem this created

`src/cubench/walkforward.py`'s resume/dedup logic (used by every
`scripts/cubench_run_grid.py --family ...` invocation) considered a cell "already
done" and skipped retraining it purely because a line for that
`(model, target, horizon, fold, seed)` key already existed in `grid_results.jsonl` —
it never checked whether the corresponding `.npy` file actually existed on disk.

This meant simply re-running the Colab notebook against the user's already-downloaded
`grid_results.jsonl` (or a re-clone that restores it from Drive) would have skipped
regenerating **all 2,838 cells**, since the resume logic would see 2,838 "done" jsonl
lines and never notice the `.npy` files backing them were entirely absent — permanently
locking in the missing-predictions problem on every future run.

## What was fixed

### 1. `src/cubench/walkforward.py` (the actual root-cause fix)

Added `prediction_file_path()` and `prediction_file_ok()`, and changed `run_grid()`'s
dedup check so a cell is only treated as "already done" if **both**:
- its line exists in `grid_results.jsonl`, **and**
- its `predictions/{model}_{target}_h{h}_fold{k}_seed{s}.npy` file genuinely exists on
  disk, is non-empty, and loads successfully with `np.load`.

A jsonl line with no matching (or corrupt/truncated) `.npy` is now treated as **not
done**, and the cell is retrained. This makes "done" mean "the raw array this
project's verification practice depends on is really there," not just "a summary line
exists somewhere." Existing correctly-completed local runs (where the `.npy` genuinely
exists) are unaffected — only cells with missing/corrupt predictions are now retrained.

### 2. `notebooks/cubench_colab.ipynb` (the Drive-sync gap)

The "Drive sync helpers" cell in Section 5 now also backs up
`results/cubench/predictions/`:
- `sync_predictions_to_drive(note)` copies every local `.npy` to
  `MyDrive/cubench_backup/predictions/`, called from inside `sync_to_drive()` — so
  every existing "sync after nulls" / "after econometric" / "after lgbm" / etc. call
  already scattered through Section 5 now backs up predictions too, at the same
  per-family cadence (not per-cell, to avoid slow/bandwidth-heavy Drive I/O over
  ~2,838 small files).
- `restore_predictions_from_drive_if_present()` restores any `.npy` present on Drive
  but missing locally, called from `restore_from_drive_if_present()` on notebook
  start. Like the existing jsonl restore, this is purely presence-based (never
  compares file size/mtime to decide "freshness") — the project has a real, documented
  history of a size/mtime-based staleness bug (`docs/todo.txt` item 5) and this
  deliberately avoids that class of bug. Safe here because each `.npy` is written
  exactly once and never edited in place, so "local already has it" always means
  "local's copy is current."

### 3. This document

## Verification performed (real execution, not just review)

1. **Real downloaded jsonl, no predictions on disk**: ran the fixed dedup logic
   against the actual `D:\copper\cubench_results\grid_results.jsonl` (2,838 unique
   cells) with `PRED_DIR` pointed at an empty directory. Result: 0 cells reported
   "done," all 2,838 correctly flagged as needing regeneration.
2. **Positive case (no regression)**: ran a real grid slice (`null_persist`, target
   `t1`, all 3 horizons, 11 folds) via `walkforward.run_grid` against real feature
   data. First run produced 33 real `.npy` files; an immediate re-run correctly
   skipped all 33 (both jsonl line and `.npy` genuinely present).
3. **Corruption case**: deleted one specific cell's `.npy`
   (`null_persist_t1_h5_fold0_seed0.npy`) while leaving its jsonl line intact, then
   re-ran. Exactly that one cell was regenerated; the other 32 were correctly left
   alone (file count unchanged at 33 after the rerun).
4. **Drive-sync/restore simulation**: used a local directory as a stand-in for Drive.
   Simulated a full "session 1 completes + syncs, VM disk wiped, session 2 restores"
   cycle — both `grid_results.jsonl` and all 5 simulated `.npy` files were correctly
   restored in the fresh session, with byte-identical array content. Also simulated a
   same-session merge (new local progress synced, then restore called again) — jsonl
   lines merge per the pre-existing append behavior (duplicates tolerated, dedup'd at
   aggregation time, unchanged from before this fix), and predictions correctly did
   NOT duplicate (presence-based restore is idempotent).

All 4 checks used the real project code (`src/cubench/walkforward.py`) and, where
applicable, real local feature/fold data — not simulated logic.

## What to expect on the next Colab run

**The full grid will be retrained from scratch.** With the fix in place, the resume
logic correctly determines that none of the 2,838 cells backed by the user's
already-downloaded `grid_results.jsonl` have a valid `.npy` on disk (because none do —
the predictions directory is genuinely empty). So the existing `grid_results.jsonl`
does not save any retraining time this run, even after being restored from Drive or a
fresh clone.

This is the honest, expected consequence of the root-cause fix, not a shortcoming of
it: the fix's entire point is to stop the resume logic from pretending cells are done
when their predictions don't exist. The alternative (leaving the old behavior in
place) would have "saved time" only by permanently and silently discarding this
project's ability to verify its own results from raw predictions.

Practically: expect the next Colab run's Section 5 (grid training) to take similar
wall-clock time to the first successful run (the notebook's own estimate: roughly
6-14 hours of CPU time, likely spanning more than one Colab session on the free tier —
see the Section 5 intro cell). This is not a new risk — Colab's throughput already
proved capable of finishing the full 2,838-cell grid once — and this time, with the
Drive-sync extension in place, the raw `.npy` predictions will actually survive a
runtime restart and make it into the final download, which they did not before.
