# CuBench Phase 5 figure manifest

Generated from real data present at run time. 79/83 (model, target, horizon) cells currently have all 11 OOS folds; the rest are partial and are labeled as such in every figure that uses them.

Re-run with `.venv_corr/Scripts/python scripts/cubench_make_figures.py` any time after more grid cells land (e.g. after the Phase 6 Colab run) -- no code changes needed, figures just get fuller.

| Figure | Shows | Data completeness now | What changes once the grid is full |
|---|---|---|---|
| fig1_model_comparison | QLIKE (T1) and pinball loss (T2) by model, faceted by horizon; hatched bars = partial grid; har_rv shown as a reference line | Nulls, econometric (har_rv/har_rv_q/garch11/gjr_garch/arima) and elasticnet fully complete (11/11 folds). lgbm essentially complete for T1/T2. catboost/xgboost/randomforest complete for T1 only (no T2/T3 cells yet). lstm/transformer effectively absent (1 or 0 cells). | Tree-model bars stop being hatched; lstm/transformer bars appear for the first time; T2/T3 bars appear for catboost/xgboost/randomforest. |
| fig2_dm_significance_heatmap | Pairwise DM test (T1, vs har_rv and vs null_persist), Holm-corrected significance | Only models/horizons with pooled T1 predictions across at least 10 common folds are shown -- this currently includes trees (lgbm/catboost/xgboost/randomforest have complete or near-complete T1) but not lstm/transformer | More rows (deep models) once their T1 grid lands; no other structural change. |
| fig3_base_rate_diagnostics | Per-(model,horizon,fold) T3 base-rate diagnostic verdict (ok / suspect_low_dispersion / constant_forecast_artifact), plus T1/T2 dispersion-ratio checks | T3 diagnostics currently cover only models with T3 predictions: nulls (constant by design), arima (collapses to one class in every fold, all 33 cells flagged constant_forecast_artifact), elasticnet (mostly ok, some suspect_low_dispersion). Tree/deep models have NO T3 cells yet, so they are absent from this figure -- absent is not the same as 'passed'. | Tree/deep rows appear; the real open question (does lgbm's T3 pass this check?) gets answered for the first time. |
| fig4_regime_performance | QLIKE by 5 calendar regimes and 3 volatility terciles, T1 h=1, for every model with regime data | All T1 h=1 models present (nulls, econometric, elasticnet, lgbm, catboost, xgboost, randomforest all have complete or near-complete h=1 T1 folds). Real finding visible: lgbm QLIKE is markedly worse in R4 (2021-22) and R5 (2023-25) than R1/R2 (2015-19). | Mostly stable -- this figure is already close to final for h=1 T1; h=5/h=22 regime panels could be added once more targets are computed by regimes.py for other horizons. |
| fig5_cost_curve | Net Sharpe vs. round-trip cost (bps), by model, h=1, with the pre-registered realistic 0.4-0.8 gross-Sharpe band shaded; LightGBM's DSR/PBO likely_overfit finding annotated directly on its curve | 8 models with both T1+T2 h=1 predictions: nulls, econometric, elasticnet, lgbm. catboost/xgboost/randomforest lack T2 h=1, so are absent. LightGBM: gross Sharpe 0.43 (looks realistic) but PBO=0.73 -> likely_overfit=True. | catboost/xgboost/randomforest curves appear once T2 h=1 exists for them; more seeds -> tighter DSR/PBO estimates for all models, not just lgbm. |
| fig6_realized_vol_timeseries | Copper realized volatility (annualized, implied from y1_h1), 2010-2025, with the 5 calendar regimes shaded | Fully complete -- computed directly from data/cubench/features.parquet, no model dependency at all. | None -- this figure is already final. |
| fig7_ablation_waterfall | A0->A5 block-wise QLIKE waterfall, LightGBM, T1 h=1 only, smoke test (3 folds, seed=42) | Explicitly a smoke test, not the full A0-A7 x 3 targets x 3 horizons x 5 seeds grid -- labeled as such directly in the figure title. Only reached A0->A5 before the evaluation script's time budget stopped it (A6/A7 comparisons missing). | Full 3x3x5-seed ablation grid across all 8 rungs once run with `--ablation-full`; figure code needs no change, just re-run. |
| fig8_quantile_calibration | Empirical vs nominal coverage (Kupiec test) for T2 quantile forecasts, per horizon, per model | Same model coverage as fig1's T2 panel -- nulls, econometric, elasticnet, lgbm (near-complete). catboost/xgboost/randomforest absent (no T2 yet). | catboost/xgboost/randomforest lines appear; lstm/transformer lines appear once their T2 grid exists. |

## Coverage snapshot at generation time

| model | target | horizon | folds present | seeds present |
|---|---|---|---|---|
| arima | t1 | 1 | 11/11 | 1 |
| arima | t1 | 5 | 11/11 | 1 |
| arima | t1 | 22 | 11/11 | 1 |
| arima | t2 | 1 | 11/11 | 1 |
| arima | t2 | 5 | 11/11 | 1 |
| arima | t2 | 22 | 11/11 | 1 |
| arima | t3 | 1 | 11/11 | 1 |
| arima | t3 | 5 | 11/11 | 1 |
| arima | t3 | 22 | 11/11 | 1 |
| catboost | t1 | 1 | 11/11 | 5 |
| catboost | t1 | 5 | 11/11 | 5 |
| catboost | t1 | 22 | 11/11 | 5 |
| catboost | t2 | 1 | 2/11 | 5 |
| elasticnet | t1 | 1 | 11/11 | 1 |
| elasticnet | t1 | 5 | 11/11 | 1 |
| elasticnet | t1 | 22 | 11/11 | 1 |
| elasticnet | t2 | 1 | 11/11 | 1 |
| elasticnet | t2 | 5 | 11/11 | 1 |
| elasticnet | t2 | 22 | 11/11 | 1 |
| elasticnet | t3 | 1 | 11/11 | 1 |
| elasticnet | t3 | 5 | 11/11 | 1 |
| elasticnet | t3 | 22 | 11/11 | 1 |
| garch11 | t1 | 1 | 11/11 | 1 |
| garch11 | t1 | 5 | 11/11 | 1 |
| garch11 | t1 | 22 | 11/11 | 1 |
| garch11 | t2 | 1 | 11/11 | 1 |
| garch11 | t2 | 5 | 11/11 | 1 |
| garch11 | t2 | 22 | 11/11 | 1 |
| gjr_garch | t1 | 1 | 11/11 | 1 |
| gjr_garch | t1 | 5 | 11/11 | 1 |
| gjr_garch | t1 | 22 | 11/11 | 1 |
| gjr_garch | t2 | 1 | 11/11 | 1 |
| gjr_garch | t2 | 5 | 11/11 | 1 |
| gjr_garch | t2 | 22 | 11/11 | 1 |
| har_rv | t1 | 1 | 11/11 | 1 |
| har_rv | t1 | 5 | 11/11 | 1 |
| har_rv | t1 | 22 | 11/11 | 1 |
| har_rv | t2 | 1 | 11/11 | 1 |
| har_rv | t2 | 5 | 11/11 | 1 |
| har_rv | t2 | 22 | 11/11 | 1 |
| har_rv_q | t1 | 1 | 11/11 | 1 |
| har_rv_q | t1 | 5 | 11/11 | 1 |
| har_rv_q | t1 | 22 | 11/11 | 1 |
| har_rv_q | t2 | 1 | 11/11 | 1 |
| har_rv_q | t2 | 5 | 11/11 | 1 |
| har_rv_q | t2 | 22 | 11/11 | 1 |
| lgbm | t1 | 1 | 11/11 | 5 |
| lgbm | t1 | 5 | 11/11 | 5 |
| lgbm | t1 | 22 | 11/11 | 5 |
| lgbm | t2 | 1 | 11/11 | 5 |
| lgbm | t2 | 5 | 11/11 | 5 |
| lgbm | t2 | 22 | 11/11 | 5 |
| lstm | t1 | 1 | 1/11 | 1 |
| null_always_down | t3 | 1 | 11/11 | 1 |
| null_always_down | t3 | 5 | 11/11 | 1 |
| null_always_down | t3 | 22 | 11/11 | 1 |
| null_majority | t3 | 1 | 11/11 | 1 |
| null_majority | t3 | 5 | 11/11 | 1 |
| null_majority | t3 | 22 | 11/11 | 1 |
| null_persist | t1 | 1 | 11/11 | 1 |
| null_persist | t1 | 5 | 11/11 | 1 |
| null_persist | t1 | 22 | 11/11 | 1 |
| null_persist | t2 | 1 | 11/11 | 1 |
| null_persist | t2 | 5 | 11/11 | 1 |
| null_persist | t2 | 22 | 11/11 | 1 |
| null_persist | t3 | 1 | 11/11 | 1 |
| null_persist | t3 | 5 | 11/11 | 1 |
| null_persist | t3 | 22 | 11/11 | 1 |
| null_rollmean | t1 | 1 | 11/11 | 1 |
| null_rollmean | t1 | 5 | 11/11 | 1 |
| null_rollmean | t1 | 22 | 11/11 | 1 |
| null_zero | t2 | 1 | 11/11 | 1 |
| null_zero | t2 | 5 | 11/11 | 1 |
| null_zero | t2 | 22 | 11/11 | 1 |
| null_zero | t3 | 1 | 11/11 | 1 |
| null_zero | t3 | 5 | 11/11 | 1 |
| null_zero | t3 | 22 | 11/11 | 1 |
| randomforest | t1 | 1 | 11/11 | 5 |
| randomforest | t1 | 5 | 11/11 | 5 |
| randomforest | t1 | 22 | 9/11 | 5 |
| xgboost | t1 | 1 | 11/11 | 5 |
| xgboost | t1 | 5 | 11/11 | 5 |
| xgboost | t1 | 22 | 8/11 | 5 |