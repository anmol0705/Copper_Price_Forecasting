# Orchestrator Report: Current State of the Copper VMD-MFGNN Codebase

Generated from local repository inspection on 2026-08-06. This report only states what is present in the workspace and flags unverifiable claims explicitly.

## 1. Repository Inventory

Root path: `D:\copper`

The repository is not currently a Git repository: `git status --short` fails with `fatal: not a git repository`.

Top-level assets:

- `requirements.txt`: Python dependency list for the experimental pipeline.
- `configs/default.yaml`: single experiment configuration.
- `scripts/run_experiments.py`: main orchestration script.
- `src/`: Python package containing data pipeline, model, baselines, training, utilities, and visualization.
- `paper/main.tex`: Elsevier/Resources Policy-style manuscript draft.
- `paper/main.pdf` and LaTeX build artifacts: compiled manuscript output exists.
- Research notes:
  - `RESEARCH_BLUEPRINT.md`
  - `literature_review_copper_price_forecasting.md`
  - `literature_gap_analysis.md`
  - `gnn_literature_review.md`
  - `vmd_research.md`
  - `copper_fundamentals.md`
- Directories `data/`, `notebooks/`, and `results/` exist but had no files in the recursive listing performed during inspection.
- Python cache directories exist under `src/__pycache__` and `src/models/__pycache__`.

No tests, CI configuration, README, license, setup metadata, or checked-in experiment result JSON files were found.

## 2. Intended Project According to Docs and Manuscript

The project targets copper price forecasting using a proposed architecture called VMD-MFGNN: Variational Mode Decomposition plus frequency-specific graph neural networks.

The manuscript claims the method:

- Uses daily data from 2010 to 2025.
- Forecasts copper returns at horizons `[1, 5, 10, 22]` trading days.
- Uses 10 input variables: copper, aluminum, zinc, nickel, gold, WTI oil, DXY, S&P 500, VIX, and US 10-year Treasury yield.
- Decomposes each variable into VMD modes.
- Builds one graph per frequency band.
- Applies GAT plus LSTM per band.
- Fuses frequency-band representations using learned attention.
- Compares against 15 baselines, including econometric models.
- Runs ablation studies and interpretability analysis.

Important status distinction: much of the manuscript is aspirational. The results tables in `paper/main.tex` are still placeholders with `--` values, and the text explicitly says to populate tables after running experiments.

## 3. Implemented Configuration

File: `configs/default.yaml`

Configured data settings:

- Date range: `2010-01-01` to `2025-12-31`.
- Target: `copper`.
- Train split ends: `2019-12-31`.
- Validation split ends: `2021-12-31`.
- Test split is implied as dates after validation.
- Lookback window: `60`.
- Forecast horizons: `[1, 5, 10, 22]`.

Configured tickers:

- `copper: HG=F`
- `aluminum: ALI=F`
- `zinc: ZNC=F`
- `nickel: NI=F`
- `gold: GC=F`
- `oil: CL=F`
- `dxy: DX-Y.NYB`
- `sp500: ^GSPC`
- `vix: ^VIX`

Configured FRED series:

- `us10y: DGS10`
- `bdi: DBAIDTLNM`

Configured VMD:

- `K: 5`
- `alpha: 2000`
- `tau: 0.0`
- `tol: 1e-7`
- `max_iter: 500`
- `rolling_window: 252`

Configured model:

- Hidden size: `64`
- GNN heads: `4`
- GNN layers: `2`
- Temporal module config says `lstm` or `transformer`, but only LSTM is implemented in the proposed model.
- Temporal layers: `2`
- Dropout: `0.1`
- Graph type: `learned`; comments list `learned`, `correlation`, and `granger`.

Configured training:

- Batch size: `32`
- Epochs: `200`
- Learning rate: `0.001`
- Weight decay: `1e-5`
- Patience: `20`
- Scheduler: `cosine`
- Seed: `42`

## 4. Main Execution Flow

File: `scripts/run_experiments.py`

The main script performs four high-level steps:

1. Loads `configs/default.yaml`.
2. Sets random seeds.
3. Creates `results/` and `results/figures/`.
4. Calls:
   - `create_datasets(config)`
   - `run_all_experiments(config, data)`
   - `run_ablation_studies(config, data)`
   - `generate_all_figures(config, data, results, ablation_results, output_dir=...)`

The script sets up logging to both console and `results/experiment.log`.

Risk: the `FileHandler` is constructed before the script creates `results/`. In the current workspace `results/` exists, so this should not fail now. In a fresh clone without that directory, logging setup can fail before directory creation.

## 5. Data Pipeline Implementation

File: `src/data_pipeline.py`

### Data Download

Implemented class: `DataDownloader`

Behavior:

- Checks for a CSV cache at `data/raw_prices.csv`.
- If the cache exists and has more than 100 rows, it is loaded and returned.
- Otherwise downloads price data from Yahoo Finance using `yfinance`.
- For each hardcoded ticker, it takes the adjusted `Close` series.
- It forward-fills up to 5 missing values, drops remaining missing rows, names the index `date`, and saves `data/raw_prices.csv`.

Important mismatch:

- `DataDownloader` uses the hardcoded `TICKERS` dictionary in `src/data_pipeline.py`, not `configs/default.yaml`.
- The implemented `TICKERS` includes `us10y: ^TNX`, while the config declares `fred_series.us10y: DGS10`.
- The implemented code does not use `fredapi` at all.
- The implemented code does not fetch the configured Baltic Dry Index (`bdi`).
- Therefore, manuscript statements that data comes from Yahoo Finance and FRED are not true for the current code path. Current code uses Yahoo Finance only.

### VMD Decomposition

Implemented classes:

- `VMDDecomposer`: rolling-window VMD intended to reduce temporal leakage.
- `VMDDecomposerFast`: full-series batch VMD, explicitly documented as faster but leaky.

Actual pipeline behavior:

- `create_datasets()` uses `VMDDecomposerFast`, not `VMDDecomposer`.
- It writes or reads `data/vmd_modes.npy`.
- The full available price panel is decomposed before train/validation/test dataset construction.

Major methodological risk:

- Full-series VMD uses future validation/test information when generating modes for earlier dates. This is a temporal leakage issue for final publishable experiments.
- The config has `rolling_window: 252`, but the active pipeline ignores it.

### Dataset Construction

Implemented datasets:

- `CopperDataset`: consumes VMD modes shaped `(num_vars, K, T)` and returns tensors shaped `(lookback, K, num_vars)`.
- `RawPriceDataset`: consumes raw prices and returns tensors shaped `(lookback, num_vars)`.

Targets:

- Forward log returns are computed as `log(p[t+h] / p[t])` for copper.
- Each sample returns one vector of returns for all configured horizons.

Splitting:

- Split indices are computed from dates:
  - `train_idx = count(dates <= train_end)`
  - `val_idx = count(dates <= val_end)`
- Training uses indices before `train_idx`.
- Validation starts at `train_idx`.
- Test starts at `val_idx`.

Normalization:

- Training datasets compute mean/std.
- Validation and test datasets reuse training mean/std.
- This part is correctly structured to avoid validation/test normalization leakage.

Assumption:

- Both datasets assume copper is the first column. `create_datasets()` enforces hardcoded `VARIABLE_NAMES` order for available variables, so this holds if copper is downloaded.

## 6. Proposed Model Implementation

File: `src/models/vmd_mfgnn.py`

Main class: `VMDMFGNN`

Implemented components:

- `FrequencyGraphConstructor`
  - For `graph_type="learned"`, creates two trainable node embedding tables per frequency band.
  - Builds adjacency as `softmax(relu(E1 @ E2.T), dim=-1)`.
  - Converts dense adjacency to PyTorch Geometric sparse edge format.
- `FrequencyBandModule`
  - Projects scalar per-variable mode values into hidden dimension.
  - Applies one or more `GATConv` layers at each timestep.
  - Applies an LSTM per variable over the lookback window.
  - Returns hidden representations per variable.
- `AttentionFusion`
  - Stacks per-mode copper representations.
  - Applies one learned query over projected mode representations.
  - Produces a single fused representation.
- Prediction heads
  - One MLP head per forecast horizon.
  - Each head maps the same fused representation to one return forecast.

Actual forward contract:

- Input: `(batch, lookback, K, num_vars)`.
- Output: dict keyed by horizon string, e.g. `"1"`, `"5"`, `"10"`, `"22"`.
- Each output tensor shape is `(batch,)`.

Interpretability hooks:

- `get_attention_weights()` returns the last batch's mode attention weights.
- `get_learned_graphs()` returns learned adjacency matrices only for learned graph constructors.

Important limitations:

- Attention fusion is not horizon-specific. The manuscript discusses mode importance across horizons, but the implementation has one shared fusion vector used by all horizon heads.
- `graph_type="correlation"` is not functional through the training pipeline. The constructor only avoids learned embeddings when `graph_type=="correlation"`, but if no `precomputed_adj` is passed, it tries to access `self.emb1`, which does not exist.
- `graph_type="granger"` is listed in config comments but is not implemented.
- There is no implemented dynamic graph, regime graph, causal graph, or shared-graph variant.
- Learned adjacency includes all nonzero softmax edges. No sparsification or top-k pruning is implemented.
- The GAT is run inside a Python loop over timesteps and creates batched edge indices repeatedly, which is likely slow for full experiments.

## 7. Baselines Implementation

File: `src/models/baselines.py`

Implemented baseline families:

Raw price deep learning:

- `LSTMBaseline`
- `GRUBaseline`
- `CNNLSTMBaseline`
- `TransformerBaseline`

VMD deep learning:

- `VMDLSTMBaseline`
- `VMDTransformerBaseline`
- `VMDAttentionLSTM`

Graph-style raw price baselines:

- `SimpleMTGNN`
- `GNNTransformer`

Tree baselines:

- `XGBoostBaseline`
- `LightGBMBaseline`

Shared training behavior for PyTorch baselines:

- Adam optimizer.
- MSE loss over all horizons.
- Gradient clipping.
- Early stopping on validation loss.
- No scheduler in baseline trainer.
- No history or checkpoints saved.

Important mismatch with manuscript:

- The manuscript claims 15 baselines including ARIMA, VAR, and StemGNN-style.
- The code implements 11 baselines plus the proposed model.
- ARIMA, VAR, and StemGNN-style are not implemented.

Potential baseline concerns:

- `SimpleMTGNN` is a simplified learned-adjacency/temporal-convolution model, not a faithful full MTGNN implementation.
- `GNNTransformer` is a custom simplified graph-plus-transformer baseline.
- Traditional tree baselines flatten the windowed input and train one model per horizon.

## 8. Training and Evaluation Implementation

File: `src/trainer.py`

### Proposed Model Trainer

Class: `VMDMFGNNTrainer`

Implemented behavior:

- Detects device via `get_device()`.
- Uses Adam optimizer.
- Uses cosine annealing scheduler.
- Sums MSE over horizon-specific outputs.
- Clips gradient norm to `1.0`.
- Evaluates RMSE, MAE, MAPE, R2, and directional accuracy.
- Saves best in-memory state according to validation average MSE.
- Uses early stopping.

No implemented behavior:

- No checkpoint files.
- No prediction CSV/NumPy output.
- No training-history JSON output.
- No per-epoch validation table except logs every 10 epochs.

### Experiment Runner

Function: `run_all_experiments(config, data)`

Runs:

- Proposed `VMD-MFGNN`.
- Raw baselines: LSTM, GRU, CNN-LSTM, Transformer, SimpleMTGNN, GNN-Transformer, XGBoost, LightGBM.
- VMD baselines: VMD-LSTM, VMD-Transformer, VMD-Attention-LSTM.

Saves:

- `results/all_results.json`.

Prints:

- A simple metrics table to stdout.

### Ablation Runner

Function: `run_ablation_studies(config, data)`

Implemented ablations:

- `full_model`: `num_modes=num_modes`, `graph_type="learned"`.
- `no_vmd_single_graph`: `num_modes=1`, `graph_type="learned"`.
- `correlation_graph`: `num_modes=num_modes`, `graph_type="correlation"`.

Major issues:

- `no_vmd_single_graph` still consumes the VMD-mode dataloader. It only uses mode index 0 because the model is instantiated with one mode. This is not a true "no VMD" ablation on raw prices.
- `correlation_graph` will fail unless precomputed adjacency matrices are passed into `VMDMFGNN.forward()`. The trainer never passes them.
- The manuscript lists additional ablations not implemented: single shared graph, no attention fusion, and mode sensitivity over `K in {3,5,7,9}`.

### Statistical Testing

Utility function exists:

- `diebold_mariano_test()` in `src/utils.py`.

But:

- It is not called in `run_all_experiments()`.
- No significance table is produced.
- No model error arrays are saved to support later statistical testing.

## 9. Visualization Implementation

File: `src/visualize.py`

Implemented plotting functions:

- `plot_vmd_decomposition()`: saves Fig. 1 decomposition PNG/PDF.
- `plot_frequency_graphs()`: saves learned graph heatmaps PNG/PDF if passed a trained model.
- `plot_attention_weights()`: saves mode attention PNG/PDF if passed attention weights.
- `plot_results_comparison()`: saves model comparison PNG/PDF.
- `plot_ablation_results()`: saves ablation PNG/PDF.
- `generate_all_figures()`: creates output directory and calls selected plotting functions.

Important gap:

- `generate_all_figures()` only calls:
  - VMD decomposition
  - results comparison
  - ablation results
- It does not call `plot_frequency_graphs()` or `plot_attention_weights()`.
- `run_all_experiments()` does not return the trained proposed model or saved attention weights.
- Therefore, the interpretability figures claimed in the paper are not generated by the current end-to-end script.

## 10. Utilities Implementation

File: `src/utils.py`

Implemented:

- YAML config loading.
- Random seed setup for Python, NumPy, PyTorch, and CUDA.
- Device selection with CUDA, Apple MPS, then CPU fallback.
- Early stopping helper.
- Metrics: RMSE, MAE, MAPE, R2, and directional accuracy.
- Diebold-Mariano test using squared error differential and Newey-West style horizon bandwidth.
- JSON result saving with NumPy scalar/array conversion.

Potential metric concern:

- Directional accuracy is computed as `mean(sign(y_true) == sign(y_pred))`. Since the target is a return, this measures predicted return direction, which is appropriate. It does not compare price-level direction directly.

## 11. Manuscript Status

File: `paper/main.tex`

Implemented manuscript sections:

- Abstract
- Introduction
- Related Work
- Methodology
- Experimental Setup
- Results and Discussion
- Conclusion
- Data Availability
- CRediT statement
- Embedded bibliography

Current manuscript gaps:

- Results table is unpopulated.
- Ablation table is unpopulated.
- Interpretability analysis describes expected patterns rather than observed generated results.
- Author names, emails, affiliation, and repository URL are placeholders.
- Claims about outperforming baselines are not supported by checked-in results.
- Claims about FRED data are inconsistent with the current code path.
- Claims about 15 baselines are inconsistent with implemented baselines.

The compiled `paper/main.pdf` exists, but its presence does not imply experiments were run.

## 12. Research Notes Status

The Markdown research files provide literature/background planning, not executable implementation.

Observed contents at heading level:

- `RESEARCH_BLUEPRINT.md`: opportunity, architecture proposal, differentiation, experimental design, paper structure, publication strategy, implementation roadmap, risk mitigation, future extensions, key references.
- `vmd_research.md`: VMD theory, VMD in forecasting, decomposition/deep learning combinations, VMD+GNN gap, advanced decomposition methods, technical challenges including temporal leakage.
- `literature_review_copper_price_forecasting.md`: traditional/econometric, ML, hybrid/decomposition, GNN approaches, datasets/features, gaps.
- `literature_gap_analysis.md`: VMD+GNN/copper novelty analysis, proposed framework options, publication strategy, required baselines/ablations, risk assessment.
- `gnn_literature_review.md`: GNN architectures for time series, finance, commodity markets, graph construction, dynamic graphs, interpretability, implementation considerations.
- `copper_fundamentals.md`: copper market structure, supply/demand, macro relationships, data sources, underexplored features, recommended data stack.

These files include many literature and market claims. I did not independently verify external citations or market facts against the internet during this codebase inspection. Treat them as internal notes until separately validated.

## 13. Verification Performed

Commands/results:

- `rg --files`: listed all tracked-looking files in the directory tree.
- `Get-ChildItem -Force`: confirmed top-level directories and files.
- `Get-ChildItem -Recurse -Force data,notebooks,results`: returned no files, indicating those directories are empty in this workspace.
- `python -m compileall -q src scripts`: passed, so Python syntax compilation succeeds for source and script files.

Attempted smoke test:

- A lightweight model/dataset smoke test failed immediately because `numpy` is not installed in the current Python environment:
  - `ModuleNotFoundError: No module named 'numpy'`

Therefore:

- I verified syntax but did not verify runtime execution, data download, VMD decomposition, model forward pass, training, or figure generation.
- Dependencies from `requirements.txt` need to be installed before runtime validation.

## 14. High-Priority Blockers

1. Install and validate dependencies.
   - Current environment lacks at least `numpy`.
   - Without dependencies, no runtime path can be trusted.

2. Remove temporal leakage from final experiments.
   - Current pipeline uses `VMDDecomposerFast` on the full series before splitting.
   - For publishable results, use rolling or expanding-window decomposition that does not use future data.

3. Fix data source/config mismatch.
   - Either make code consume `configs/default.yaml` tickers/FRED series or update config/manuscript to match actual Yahoo-only behavior.
   - Implement FRED ingestion if the paper keeps FRED claims.
   - Decide whether `bdi` is in scope.

4. Fix non-learned graph modes.
   - `correlation` graph currently cannot run through the trainer.
   - `granger` is listed but not implemented.
   - Need a clear API for precomputed per-band adjacency matrices.

5. Fix ablation definitions.
   - `no_vmd_single_graph` is not currently no-VMD.
   - `correlation_graph` likely fails.
   - Manuscript-listed ablations are not all implemented.

6. Align baseline claims.
   - Implement ARIMA, VAR, and StemGNN-style baselines or remove them from claims.
   - Decide whether the count is 11, 15, 18, or another final number.

7. Produce and save experiment artifacts.
   - Run experiments after dependency and leakage fixes.
   - Save all metrics, predictions, errors, trained-model metadata, and generated figures.

8. Generate interpretability artifacts.
   - Return or save the trained proposed model.
   - Save learned adjacency matrices.
   - Save attention weights over the test set.
   - Call graph and attention plotting functions in the main script.

## 15. Recommended Next Engineering Plan

Phase 1: Runtime environment and smoke tests

- Create a reproducible environment file or lockfile.
- Install `requirements.txt`.
- Add a minimal synthetic-data smoke test for:
  - `CopperDataset`
  - `RawPriceDataset`
  - `VMDMFGNN.forward`
  - each baseline forward/training loop on tiny data
- Confirm PyTorch Geometric and `GATConv` work in the environment.

Phase 2: Data pipeline correctness

- Make `DataDownloader` use config-driven tickers.
- Add FRED ingestion or remove FRED from config/manuscript.
- Store raw source metadata.
- Add checks for missing variables and fail clearly if the target is absent.
- Decide whether Yahoo Finance proxies are acceptable for the paper.

Phase 3: Leakage-safe VMD

- Replace `VMDDecomposerFast` for final experiments.
- Cache leakage-safe modes with metadata containing split dates, VMD parameters, and source data hash.
- Keep fast decomposition behind an explicit debug flag only.

Phase 4: Graph and ablation repair

- Implement correlation adjacency generation per VMD band and split/window.
- Implement or remove Granger graph mode.
- Add a true raw-price no-VMD graph model for the no-VMD ablation.
- Add single shared graph and no-attention variants if the manuscript keeps those claims.
- Add mode sensitivity runs for `K`.

Phase 5: Results and paper integration

- Save:
  - `results/all_results.json`
  - `results/ablation_results.json`
  - per-model predictions
  - per-model target arrays
  - model runtime and seed metadata
  - learned graphs
  - attention weights
- Run Diebold-Mariano tests and save a significance table.
- Populate manuscript tables and replace expectation language with observed findings.

## 16. Bottom Line for the Main Orchestrator

The repository currently contains a coherent prototype for VMD-MFGNN copper return forecasting, including data download, VMD decomposition, datasets, a proposed GAT+LSTM multi-frequency model, 11 implemented baselines, training/evaluation utilities, plotting functions, and a manuscript draft.

It is not yet a complete, publication-ready experimental codebase. The largest issues are temporal leakage from full-series VMD, unimplemented or broken graph modes/ablations, mismatch between manuscript claims and implemented baselines/data sources, absent experiment artifacts, absent tests, and an unprepared runtime environment.

The next orchestrator should treat the current code as a promising prototype and focus first on reproducibility and methodological correctness before running expensive experiments or filling manuscript results.
