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


def main():
    results = [
        test_fresh_run_executes_all_cells(),
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
