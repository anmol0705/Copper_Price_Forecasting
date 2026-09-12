# VMD-MFGNN v2 Locked Scope Protocol

Reference this file instead of re-deriving scope.

---

## Locked Input Variables (8 total)

> **SUPERSEDED (2026-09-12)**: this 8-variable, Yahoo-Finance-only scope (including
> "Explicitly out of scope: FRED") has been replaced by the N=16 expansion — 12 Yahoo
> Finance tickers + 4 FRED series — described in `paper1/PLAN.md` Section 1. The table
> and notes below are retained as accurate *history* of the v2 locked scope, not as the
> current scope; `paper1/src/data_pipeline.py`'s `TICKERS` / `FRED_*_SERIES` are the
> source of truth.

All data sourced via **Yahoo Finance only**. Explicitly out of scope: FRED, TC/RC, COT, BDI.

| Variable | Ticker | Source |
|----------|--------|--------|
| Copper | HG=F | Yahoo Finance |
| Aluminum | ALI=F | Yahoo Finance |
| Gold | GC=F | Yahoo Finance |
| Oil | CL=F | Yahoo Finance |
| DXY (US Dollar Index) | DX-Y.NYB | Yahoo Finance |
| S&P 500 | ^GSPC | Yahoo Finance |
| VIX (Volatility Index) | ^VIX | Yahoo Finance |
| US 10Y Yield | ^TNX | Yahoo Finance |

**Scope reduction (2026-08-07)**: Zinc and Nickel have been dropped from the
locked scope. They were originally sourced via NASDAQ Commodity sub-index
proxies (`^NQCIZNER` for zinc, `^NQCINIER` for nickel) since no standalone
COMEX/NYMEX-style Yahoo Finance futures contract exists for either metal (they
trade on the LME, which Yahoo does not mirror). On a live production data
download run on 2026-08-07 (the real 2010-2025 pull), both proxy tickers were
confirmed dead/delisted -- yfinance raises
`YFPricesMissingError('possibly delisted; no price data found')` for both.
With no further viable Yahoo Finance substitute, zinc and nickel are removed
from the locked scope entirely. This is a deliberate, disclosed reduction
from the original 10-variable scope to the current 8-variable scope (copper,
aluminum, gold, oil, dxy, sp500, vix, us10y). `src/data_pipeline.py`'s
`TICKERS` dict and `configs/default.yaml`'s `tickers:` section were updated
to match this table exactly, zero drift. Any paper text/limitations section
referencing "10 variables" must be updated to "8 variables" as part of the
Phase 4 LaTeX rewrite.

---

## Prediction Horizons

- 1 trading day
- 5 trading days
- 10 trading days
- 22 trading days

---

## Variational Mode Decomposition (VMD) Specification

- **K (number of modes)**: 5
- **Window type**: Expanding-window (leakage-safe, not full-series batch)
- **Refit interval**: Every 21 trading days
- **Caching**: Metadata stored with split dates, parameters, and data hash

---

## Graph Types

### In Scope
- **Learned graph** (primary): Used by the main VMD-MFGNN model; learned end-to-end during training
- **Correlation graph** (ablation only): Precomputed adjacency matrix threaded through forward()

### Out of Scope
- Granger graph (explicitly excluded, even if old config comments reference it)

---

## Baseline Models

### Locked In Scope (6 models)
1. ARIMA
2. XGBoost
3. LSTM
4. Transformer
5. VMD-LSTM
6. SimpleMTGNN

### Out of Scope — Remove from Active Pipeline (6 models)
Must be removed from `run_all_experiments` and `run_ablation_studies`:
1. GRU
2. CNN-LSTM
3. VMD-Transformer
4. VMD-Attention-LSTM
5. GNN-Transformer
6. LightGBM

**Note**: Code files may remain on disk; they must not be invoked during active experiment runs.

---

## Locked Ablation Studies (3 total)

### (1) VMD vs. No-VMD
- **No-VMD side constraint**: Must use a genuine raw-price dataloader/model, not a `num_modes=1` workaround
- Ensures fair comparison of VMD's decomposition benefit

### (2) Per-Band Graphs vs. Pooled Single Graph ← **HIGHEST PRIORITY**
- **Per-band side**: K separate graphs (one per VMD mode), standard VMD-MFGNN
- **Pooled side**: Separate model variant (`pooled_graph_mfgnn`) that:
  - Pools K band representations into a single unified representation
  - Builds **one graph only** (not K graphs)
  - Contrasts against per-band architecture
- **Significance**: This ablation is the empirical backbone of the paper's novelty claim vs. prior art (MBTI-Net); treat as non-negotiable and highest priority

### (3) Learned Graph vs. Correlation Graph
- **Learned side**: Standard end-to-end graph learning
- **Correlation side**: Uses precomputed adjacency matrix via `precomputed_adj` parameter

### Out of Scope — Do Not Implement
- K-sensitivity sweep ablation (varying number of modes)

---

## Data Splits

| Split | Date Range |
|-------|------------|
| Train | 2010–2019 |
| Validation | 2020–2021 |
| Test | 2022–2025 |

---

## Active Pipeline Constraint

Anything in the repo outside this scope (extra baselines, extra graph types, extra ablations) gets removed from the ACTIVE pipeline (`run_all_experiments`, `run_ablation_studies`), not necessarily deleted from disk, unless a subagent's specific brief says otherwise.

---

## Hyperparameter Optimization (added 2026-08-07)

- The proposed model, VMD-MFGNN, gets real hyperparameter optimization: a lightweight Optuna study (TPE sampler + median pruner), searching `hidden_dim` ∈ {32,64,128}, `num_heads` ∈ {2,4,8} (constrained so `hidden_dim % num_heads == 0`), `learning_rate` (log-uniform 1e-4 to 1e-2), `dropout` (uniform 0.05 to 0.3), `num_gnn_layers` ∈ {1,2,3}. Implemented in `src/hpo.py`'s `run_hpo(config, data, n_trials=15, trial_epochs=25)`, wired into `run_all_experiments()` behind a config flag `hpo.enabled` (default `false` — opt-in only). Results saved to `results/hpo_best_params.json` and `results/hpo_trials.csv` for reproducibility reporting.
- The 6 baselines (ARIMA, XGBoost, LSTM, Transformer, VMD-LSTM, SimpleMTGNN) do NOT get HPO — they run at their current standard/hardcoded settings. This asymmetry is deliberate, not an oversight: full HPO across all 7 models would multiply compute cost roughly by the trial count, risking the 14-day deadline, and tuning-only-your-own-model-vs-standard-baselines is standard, defensible practice in comparison papers under time constraints, PROVIDED it's disclosed. This must be stated explicitly in the paper's Limitations section during the Phase 4 LaTeX rewrite — do not let it read as if all models were tuned equally.
- Model checkpointing: `VMDMFGNNTrainer.fit()` now supports an optional `checkpoint_path` parameter that saves the best-validation-loss model weights to disk (not just in-memory) and can resume from a saved checkpoint (skips retraining if the checkpoint file already exists at that path). This is now a core library feature (`src/trainer.py`), not just something the Colab notebook does. Both `run_all_experiments()` (VMD-MFGNN) and `run_ablation_studies()` (all 4 variants) use default checkpoint paths under `results/checkpoints/`. Anyone re-running the pipeline should be aware: a stale checkpoint file from a prior run will cause the next run to skip training and just reload old weights — this is intentional resumability, not a bug, but delete `results/checkpoints/` if you want a guaranteed fresh run.

---

*Last updated: 2026-08-07*
