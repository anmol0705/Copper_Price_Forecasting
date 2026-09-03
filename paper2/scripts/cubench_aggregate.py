"""Aggregate results/cubench/grid_results.jsonl (+ pilot) into
results/cubench/all_results.json via src/utils.py::save_results (Paper 1's numpy-safe
JSON serializer, reused not reimplemented)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import save_results

JSONL = Path("results/cubench/grid_results.jsonl")
OUT = Path("results/cubench/all_results.json")


def main():
    cells = {}
    n_lines = 0
    n_errors = 0
    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            r = json.loads(line)
            key = (r.get("model"), r.get("target"), r.get("horizon"), r.get("fold"), r.get("seed"))
            if "error" in r:
                n_errors += 1
            cells[key] = r  # last write wins (handles reruns/dupes)

    by_model = defaultdict(list)
    for r in cells.values():
        by_model[r.get("model")].append(r)

    summary = {}
    for model, rows in by_model.items():
        ok_rows = [r for r in rows if "error" not in r]
        summary[model] = {
            "n_cells": len(rows),
            "n_errors": len(rows) - len(ok_rows),
        }
        for target in ["t1", "t2", "t3"]:
            trows = [r for r in ok_rows if r.get("target") == target]
            if not trows:
                continue
            if target == "t1":
                summary[model]["t1_rmse_mean"] = sum(r["rmse"] for r in trows) / len(trows)
                summary[model]["t1_qlike_mean"] = sum(r["qlike"] for r in trows) / len(trows)
            elif target == "t2":
                summary[model]["t2_pinball_avg_mean"] = sum(r["pinball_avg"] for r in trows) / len(trows)
                cr = [r["crossing_rate"] for r in trows if "crossing_rate" in r]
                if cr:
                    summary[model]["t2_crossing_rate_mean"] = sum(cr) / len(cr)
            elif target == "t3":
                summary[model]["t3_accuracy_mean"] = sum(r["accuracy"] for r in trows) / len(trows)

    out = {
        "cells": list(cells.values()),
        "summary_by_model": summary,
        "n_unique_cells": len(cells),
        "n_lines_read": n_lines,
        "n_errors": n_errors,
    }
    save_results(out, str(OUT))
    print(f"Wrote {OUT} with {len(cells)} unique cells ({n_errors} errors) from {n_lines} jsonl lines")
    for m, s in sorted(summary.items()):
        print(" ", m, s)


if __name__ == "__main__":
    main()
