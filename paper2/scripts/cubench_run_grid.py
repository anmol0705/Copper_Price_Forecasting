"""CuBench full-grid runner. Invoked per model-family so each call stays short and
progress is persisted incrementally (results/cubench/grid_results.jsonl, one line per
model x target x horizon x fold x seed cell, flushed immediately) -- safe to interrupt
and resume; already-completed cells are skipped on rerun.

Usage:
    python scripts/cubench_run_grid.py --family nulls
    python scripts/cubench_run_grid.py --family econometric
    python scripts/cubench_run_grid.py --family linear
    python scripts/cubench_run_grid.py --family trees --model lgbm
    python scripts/cubench_run_grid.py --family deep --model lstm
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.cubench import walkforward as wf
from src.cubench.models import nulls, econometric, linear, trees, deep

RESULTS_JSONL = Path("results/cubench/grid_results.json").with_suffix(".jsonl")
ALL_HORIZONS = [1, 5, 22]
ALL_TARGETS = ["t1", "t2", "t3"]


def build_null_models():
    return {
        "null_persist": (nulls.null_persist_factory, [0]),
        "null_rollmean": (nulls.null_rollmean_factory, [0]),
        "null_zero": (nulls.null_zero_factory, [0]),
        "null_majority": (nulls.null_majority_factory, [0]),
        "null_always_down": (nulls.null_always_down_factory, [0]),
    }


NULL_TARGET_MAP = {
    "null_persist": ["t1", "t2", "t3"],
    "null_rollmean": ["t1"],
    "null_zero": ["t2", "t3"],
    "null_majority": ["t3"],
    "null_always_down": ["t3"],
}


def build_econometric_models():
    return {
        "har_rv": (econometric.har_rv_factory, [0]),
        "har_rv_q": (econometric.har_rv_q_factory, [0]),
        "garch11": (econometric.garch11_factory, [0]),
        "gjr_garch": (econometric.gjr_garch_factory, [0]),
        "arima": (econometric.arima_factory, [0]),
    }


ECON_TARGET_MAP = {
    "har_rv": ["t1", "t2"],
    "har_rv_q": ["t1", "t2"],
    "garch11": ["t1", "t2"],
    "gjr_garch": ["t1", "t2"],
    "arima": ["t1", "t2", "t3"],
}


def build_linear_models():
    return {"elasticnet": (None, [0])}  # target-specific factory built per-target below


def build_tree_models(cfg):
    return {
        "lgbm": trees.lgbm_factory,
        "xgboost": trees.xgboost_factory,
        "catboost": trees.catboost_factory,
        "randomforest": trees.randomforest_factory,
    }


def run_family_for_target(model_id, factory, target, horizons, folds, seeds, df):
    for h in horizons:
        for fold in folds:
            for seed in seeds:
                # dedupe handled by run_grid; call run_grid per-model/target/horizon
                pass
    models = {model_id: (factory, seeds)}
    return wf.run_grid(models, [target], horizons, folds, df, RESULTS_JSONL.with_suffix(""), append=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                     choices=["nulls", "econometric", "linear", "trees", "deep"])
    ap.add_argument("--model", default=None, help="restrict to one model id within the family")
    ap.add_argument("--horizons", default=None, help="comma list, default all")
    ap.add_argument("--targets", default=None, help="comma list, default all")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")] if args.horizons else ALL_HORIZONS
    targets = args.targets.split(",") if args.targets else ALL_TARGETS

    df = wf.load_features()
    folds = wf.load_folds()["folds"]
    cfg = yaml.safe_load(open("configs/cubench.yaml"))

    t0 = time.time()
    n_cells = 0

    if args.family == "nulls":
        models = build_null_models()
        for model_id, (factory, seeds) in models.items():
            if args.model and model_id != args.model:
                continue
            tmap = [t for t in NULL_TARGET_MAP[model_id] if t in targets]
            for target in tmap:
                res = run_family_for_target(model_id, factory, target, horizons, folds, seeds, df)
                n_cells += len(res)
                print(f"[nulls] {model_id} {target}: {len(res)} cells, total {time.time()-t0:.1f}s", flush=True)

    elif args.family == "econometric":
        models = build_econometric_models()
        for model_id, (factory, seeds) in models.items():
            if args.model and model_id != args.model:
                continue
            tmap = [t for t in ECON_TARGET_MAP[model_id] if t in targets]
            for target in tmap:
                res = run_family_for_target(model_id, factory, target, horizons, folds, seeds, df)
                n_cells += len(res)
                print(f"[econometric] {model_id} {target}: {len(res)} cells, total {time.time()-t0:.1f}s", flush=True)

    elif args.family == "linear":
        for target in targets:
            factory = linear.elasticnet_factory(target)
            res = run_family_for_target("elasticnet", factory, target, horizons, folds, [0], df)
            n_cells += len(res)
            print(f"[linear] elasticnet {target}: {len(res)} cells, total {time.time()-t0:.1f}s", flush=True)

    elif args.family == "trees":
        tree_factories = build_tree_models(cfg)
        seeds = cfg["lgbm_hyperparameters"]["seeds"]
        model_ids = [args.model] if args.model else list(tree_factories.keys())
        cfg_key = {"lgbm": "lgbm_hyperparameters", "xgboost": "xgboost_hyperparameters",
                   "catboost": "catboost_hyperparameters", "randomforest": "randomforest_hyperparameters"}
        for model_id in model_ids:
            factory_fn = tree_factories[model_id]
            model_cfg = cfg[cfg_key[model_id]]
            model_seeds = model_cfg.get("seeds", seeds)
            for target in targets:
                factory = factory_fn(target, model_cfg)
                res = run_family_for_target(model_id, factory, target, horizons, folds, model_seeds, df)
                n_cells += len(res)
                print(f"[trees] {model_id} {target}: {len(res)} cells, total {time.time()-t0:.1f}s", flush=True)

    elif args.family == "deep":
        model_names = [args.model] if args.model else ["lstm", "transformer"]
        deep_targets = [t for t in ["t1", "t3"] if t in targets]  # T2 deep skipped, see deep.py docstring
        deep_horizons = [h for h in [1] if h in horizons]  # h=1 only, see deep.py docstring
        deep_seeds = cfg["deep_model_placeholder"]["seeds"]
        res = deep.run_deep_grid(model_names, deep_targets, deep_horizons, folds, deep_seeds, RESULTS_JSONL)
        n_cells += len(res)
        print(f"[deep] {model_names}: {len(res)} cells, total {time.time()-t0:.1f}s", flush=True)

    print(f"DONE family={args.family} model={args.model} cells={n_cells} wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
