"""Single Optuna study (50 trials, TPE) to sanity-check/refine the plan's fixed LightGBM
hyperparameters, run ONCE on the 2010-2014 initial training block only (train
2010-01-04..2012-12-31, internal validation 2013-01-01..2014-12-31), BEFORE any OOS year
is touched. Per docs/cubench_implementation_plan.md Section 3.4 HPO policy.

Tuned on T1 (regression, log RV), horizon=1, as the representative cell (T1/h1 is the
primary headline target; the plan specifies a single study, not one per target/horizon).
Writes results/cubench/hpo_result.json with best params vs. plan defaults.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb

from src.cubench import walkforward as wf

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULTS = {
    "n_estimators": 600, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 5,
    "min_child_samples": 60, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.7, "reg_lambda": 1.0, "reg_alpha": 0.1,
}


def main():
    df = wf.load_features()
    fcols = wf.feature_cols(df)
    train_start, train_end = "2010-01-04", "2012-12-31"
    val_start, val_end = "2013-01-01", "2014-12-31"
    ycol = "y1_h1"

    tr_mask = (df["date"] >= train_start) & (df["date"] <= train_end) & df[ycol].notna()
    va_mask = (df["date"] >= val_start) & (df["date"] <= val_end) & df[ycol].notna()
    Xtr, ytr = df.loc[tr_mask, fcols], df.loc[tr_mask, ycol].to_numpy()
    Xva, yva = df.loc[va_mask, fcols], df.loc[va_mask, ycol].to_numpy()
    print(f"HPO train n={len(Xtr)}, val n={len(Xva)}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "random_state": 42, "n_jobs": 1, "verbosity": -1,
        }
        m = lgb.LGBMRegressor(objective="regression", **params)
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    wall = time.time() - t0

    # score the plan's fixed defaults on the same split for direct comparison
    m_def = lgb.LGBMRegressor(objective="regression", random_state=42, n_jobs=1,
                               verbosity=-1, **DEFAULTS)
    m_def.fit(Xtr, ytr)
    rmse_default = float(np.sqrt(np.mean((yva - m_def.predict(Xva)) ** 2)))

    result = {
        "n_trials": 50, "wall_s": wall,
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "default_params": DEFAULTS,
        "default_rmse": rmse_default,
        "improvement_pct": 100.0 * (rmse_default - study.best_value) / rmse_default,
        "train_window": [train_start, train_end],
        "val_window": [val_start, val_end],
        "target": "y1_h1",
    }
    out = Path(__file__).resolve().parents[1] / "results" / "cubench" / "hpo_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
