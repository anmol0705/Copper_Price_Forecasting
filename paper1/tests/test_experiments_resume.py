"""Local (no-GPU, no-torch_geometric) test of experiments.py's resumable-grid
mechanics -- the same jsonl-append/dedup pattern used throughout the new
Colab notebook's Sections 3-6. Does NOT touch any model/training code
(that requires torch_geometric, unavailable in this environment) -- it
verifies the resume/dedup/artifact-check logic in isolation, using a fake
`run_fn` that writes a "checkpoint" file and a fake `artifact_check_fn` that
requires it to exist, exactly like the real experiments do with real
model checkpoints.

Simulates a Colab-disconnect-style partial run: run a 4-cell grid, "kill" it
after 2 cells by truncating the jsonl mid-line (as a crash would), then
re-run and confirm cell 1 (fully done, checkpoint present) is skipped, the
truncated cell is correctly detected as incomplete and re-run, and cells 3-4
run fresh.

Run: python tests/test_experiments_resume.py   (from paper1/)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments import (  # noqa: E402
    append_cell_result, load_completed_cells, run_resumable_grid,
    _checkpoint_is_complete)

WORKDIR = Path(__file__).resolve().parent / "_scratch_resume_test"


def cleanup():
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)


def fake_artifact_check(row):
    return Path(row["checkpoint_path"]).exists()


def make_run_fn(call_log):
    def run_fn(cell):
        call_log.append(cell["k"])
        ckpt = WORKDIR / f"ckpt_{cell['k']}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("fake checkpoint bytes")
        return {"k": cell["k"], "metric": cell["k"] * 1.5, "checkpoint_path": str(ckpt)}
    return run_fn


def test_fresh_run_executes_all_cells():
    cleanup()
    jsonl = WORKDIR / "grid.jsonl"
    cells = [{"k": k} for k in [3, 5, 7, 9]]
    call_log = []
    results = run_resumable_grid(
        cells, key_fn=lambda c: (c["k"],), run_fn=make_run_fn(call_log),
        jsonl_path=str(jsonl), artifact_check_fn=fake_artifact_check)
    ok = call_log == [3, 5, 7, 9] and len(results) == 4
    print(f"{'PASS' if ok else 'FAIL'}: fresh run executed all 4 cells in order "
          f"(call_log={call_log})")
    return ok


def test_resume_skips_completed_and_reruns_incomplete():
    cleanup()
    jsonl = WORKDIR / "grid.jsonl"

    # Simulate: cells k=3 and k=5 completed genuinely (checkpoint + full jsonl
    # line). Then simulate a crash mid-write of k=7's line (truncated JSON,
    # as os-level buffering + a kill -9 could produce) -- no checkpoint for
    # k=7 either, since the crash happened before run_fn's "training" (in
    # this fake, file-write) completed.
    for k in [3, 5]:
        ckpt = WORKDIR / f"ckpt_{k}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("fake checkpoint bytes")
        append_cell_result(str(jsonl), [k], {"k": k, "metric": k * 1.5,
                                              "checkpoint_path": str(ckpt)})
    with open(jsonl, "a") as f:
        f.write('{"cell_key": [7], "k": 7, "metric": 10.5, "checkpoint_p')  # truncated, no trailing newline flush of a complete object

    # Pre-check: load_completed_cells must report only k=3,5 as done (2 cells),
    # NOT k=7 (malformed line) -- proving the malformed-line-skip path works.
    completed = load_completed_cells(str(jsonl), artifact_check_fn=fake_artifact_check)
    precheck_ok = set(completed.keys()) == {(3,), (5,)}
    print(f"{'PASS' if precheck_ok else 'FAIL'}: pre-resume state shows exactly "
          f"{{3,5}} completed (got {set(completed.keys())})")

    # Now resume: re-run the full 4-cell grid. Expect k=3,5 skipped (no
    # run_fn call), k=7 and k=9 to actually execute.
    cells = [{"k": k} for k in [3, 5, 7, 9]]
    call_log = []
    results = run_resumable_grid(
        cells, key_fn=lambda c: (c["k"],), run_fn=make_run_fn(call_log),
        jsonl_path=str(jsonl), artifact_check_fn=fake_artifact_check)

    resume_ok = call_log == [7, 9]
    print(f"{'PASS' if resume_ok else 'FAIL'}: resume only re-ran incomplete "
          f"cells [7, 9] (got call_log={call_log})")

    # Final jsonl should now have 4 valid completed cells when reloaded
    # (the truncated k=7 line's malformed entry is superseded/ignored, and a
    # fresh valid k=7 line was appended after it).
    final_completed = load_completed_cells(str(jsonl), artifact_check_fn=fake_artifact_check)
    final_ok = set(final_completed.keys()) == {(3,), (5,), (7,), (9,)}
    print(f"{'PASS' if final_ok else 'FAIL'}: final state shows all 4 cells "
          f"completed (got {set(final_completed.keys())})")

    ok = precheck_ok and resume_ok and final_ok and len(results) == 4
    return ok


def test_artifact_missing_forces_rerun_even_with_jsonl_row():
    """Reproduces the exact bug class this project already fixed once in
    CuBench: a jsonl line exists but its underlying artifact file was lost
    (e.g. an ephemeral Colab VM disk wipe between 'training finished' and
    'zip+download', per docs/cubench_colab_resume_incident.md). Confirms
    load_completed_cells treats this as INCOMPLETE, not done.
    """
    cleanup()
    jsonl = WORKDIR / "grid.jsonl"
    # A row that LOOKS complete (valid JSON, full cell_key) but whose
    # checkpoint_path was never actually created on disk.
    append_cell_result(str(jsonl), [3], {"k": 3, "metric": 4.5,
                                          "checkpoint_path": str(WORKDIR / "ghost.pt")})
    completed = load_completed_cells(str(jsonl), artifact_check_fn=fake_artifact_check)
    ok = completed == {}
    print(f"{'PASS' if ok else 'FAIL'}: a jsonl row whose checkpoint file is "
          f"missing on disk is correctly treated as incomplete (got {completed})")
    return ok


def test_checkpoint_completed_flag_is_respected():
    """`_checkpoint_is_complete` (used by every real experiment's
    artifact_check_fn, unlike this file's other tests which use a bare
    Path.exists() fake) must distinguish a genuinely-finished checkpoint
    (trainer.py's fit() writes completed=True only when the training loop
    actually exits) from a mid-training best-so-far save (completed=False,
    e.g. left behind by a crash between a best-val-loss improvement and the
    loop's natural end)."""
    import torch
    cleanup()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    complete_path = WORKDIR / "complete.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)}, "epoch": 10, "completed": True},
               complete_path)
    incomplete_path = WORKDIR / "incomplete.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)}, "epoch": 3, "completed": False},
               incomplete_path)
    legacy_path = WORKDIR / "legacy.pt"
    torch.save({"w": torch.zeros(2)}, legacy_path)  # bare state dict, pre-flag era
    missing_path = WORKDIR / "does_not_exist.pt"

    checks = {
        "completed=True checkpoint -> True": _checkpoint_is_complete(complete_path) is True,
        "completed=False checkpoint -> False": _checkpoint_is_complete(incomplete_path) is False,
        "legacy bare-state-dict -> True": _checkpoint_is_complete(legacy_path) is True,
        "missing file -> False": _checkpoint_is_complete(missing_path) is False,
    }
    for label, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {label}")
    return all(checks.values())


def test_one_failing_cell_does_not_abort_the_grid():
    """A single cell raising (OOM / NaN loss / transient Colab hiccup) must
    NOT abort the whole experiment -- in a Colab "Run All" that would also
    halt every subsequent notebook cell, ending a multi-hour unattended run
    with nothing saved for anything after it. The failing cell must get NO
    jsonl row (so it is naturally retried next run), and the remaining cells
    must still execute and be recorded.
    """
    cleanup()
    jsonl = WORKDIR / "grid.jsonl"
    call_log = []
    good = make_run_fn(call_log)

    def flaky_run_fn(cell):
        if cell["k"] == 7:
            call_log.append(cell["k"])
            raise RuntimeError("CUDA out of memory (simulated)")
        return good(cell)

    cells = [{"k": k} for k in [3, 5, 7, 9]]
    results = run_resumable_grid(
        cells, key_fn=lambda c: (c["k"],), run_fn=flaky_run_fn,
        jsonl_path=str(jsonl), artifact_check_fn=fake_artifact_check)

    ran_all = call_log == [3, 5, 7, 9]
    print(f"{'PASS' if ran_all else 'FAIL'}: grid continued past the failing cell "
          f"(call_log={call_log})")
    completed = load_completed_cells(str(jsonl), artifact_check_fn=fake_artifact_check)
    recorded_ok = set(completed.keys()) == {(3,), (5,), (9,)}
    print(f"{'PASS' if recorded_ok else 'FAIL'}: failed cell k=7 has no jsonl row; "
          f"3/5/9 recorded (got {set(completed.keys())})")
    returned_ok = len(results) == 3 and all("metric" in r for r in results)
    print(f"{'PASS' if returned_ok else 'FAIL'}: returned rows exclude the failed "
          f"cell entirely, so downstream summary code never sees a malformed row "
          f"(got {len(results)} rows)")

    # Re-run: the previously-failing cell (now healthy) must be the ONLY one retried.
    call_log2 = []
    run_resumable_grid(
        cells, key_fn=lambda c: (c["k"],), run_fn=make_run_fn(call_log2),
        jsonl_path=str(jsonl), artifact_check_fn=fake_artifact_check)
    retry_ok = call_log2 == [7]
    print(f"{'PASS' if retry_ok else 'FAIL'}: next run retried ONLY the previously "
          f"failed cell (got call_log={call_log2})")
    return ran_all and recorded_ok and returned_ok and retry_ok


def test_on_cell_done_fires_after_the_jsonl_line_is_durable():
    """The notebook passes its Drive-sync helper as `on_cell_done` so each
    finished cell of a multi-hour grid is backed up IMMEDIATELY rather than
    only after the whole grid returns (the difference between losing one
    in-flight cell and losing 9 completed ones when Colab disconnects).
    Verify the callback fires once per FRESHLY-completed cell, and that the
    cell's jsonl line is already on disk by the time it runs -- otherwise a
    sync triggered by the callback would back up a jsonl missing the very row
    it was called for.
    """
    cleanup()
    jsonl = WORKDIR / "grid.jsonl"
    seen = []

    def on_done(key, result):
        # Line count read from DISK, not memory: proves durability ordering.
        n_lines = sum(1 for line in open(jsonl) if line.strip())
        seen.append((tuple(key), n_lines))

    cells = [{"k": k} for k in [3, 5, 7]]
    run_resumable_grid(cells, key_fn=lambda c: (c["k"],),
                       run_fn=make_run_fn([]), jsonl_path=str(jsonl),
                       artifact_check_fn=fake_artifact_check, on_cell_done=on_done)
    fired_ok = [k for k, _ in seen] == [(3,), (5,), (7,)]
    durable_ok = [n for _, n in seen] == [1, 2, 3]
    print(f"{'PASS' if fired_ok else 'FAIL'}: on_cell_done fired once per cell in "
          f"order (got {[k for k, _ in seen]})")
    print(f"{'PASS' if durable_ok else 'FAIL'}: each callback saw its own row already "
          f"flushed to disk (line counts {[n for _, n in seen]})")

    # Second run: everything already complete -> callback must NOT re-fire
    # (a re-run must not trigger N redundant full Drive copies).
    seen.clear()
    run_resumable_grid(cells, key_fn=lambda c: (c["k"],),
                       run_fn=make_run_fn([]), jsonl_path=str(jsonl),
                       artifact_check_fn=fake_artifact_check, on_cell_done=on_done)
    noop_ok = seen == []
    print(f"{'PASS' if noop_ok else 'FAIL'}: on_cell_done did NOT fire for "
          f"already-completed cells on resume (got {seen})")

    # A raising callback (e.g. a Drive I/O hiccup) must not kill the grid.
    cleanup()
    calls = []
    def bad_on_done(key, result):
        raise OSError("simulated Drive I/O error")
    run_resumable_grid(cells, key_fn=lambda c: (c["k"],),
                       run_fn=make_run_fn(calls), jsonl_path=str(jsonl),
                       artifact_check_fn=fake_artifact_check, on_cell_done=bad_on_done)
    survive_ok = calls == [3, 5, 7]
    print(f"{'PASS' if survive_ok else 'FAIL'}: a raising on_cell_done callback is "
          f"swallowed and the grid completes (got {calls})")
    return fired_ok and durable_ok and noop_ok and survive_ok


def main():
    results = [
        test_fresh_run_executes_all_cells(),
        test_one_failing_cell_does_not_abort_the_grid(),
        test_on_cell_done_fires_after_the_jsonl_line_is_durable(),
        test_resume_skips_completed_and_reruns_incomplete(),
        test_artifact_missing_forces_rerun_even_with_jsonl_row(),
        test_checkpoint_completed_flag_is_respected(),
    ]
    cleanup()
    ok = all(results)
    print("\n=== OVERALL:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
