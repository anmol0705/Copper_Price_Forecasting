# results/archive_paper1 — file provenance notes

**`ablation_results.json` is superseded and does not match the paper.**
It was produced before the `min_epochs` checkpoint-selection fix (see `src/utils.py`'s
`EarlyStopper` and `src/trainer.py`'s `run_ablation_studies`) and reports different
numbers than Table VIII in `manuscript/main.tex` for 4 of its 6 variants. Use
**`ablation_results_minepochs_fix.json`** instead — this is the corrected re-run
(`min_epochs=10`) that the paper actually reports, pulled from the Colab run's Google
Drive sync (`Copper_Paper1/ablation_minepochs_fix/ablation_results_minepochs_fix.json`).

**`graphfix_minepochs_fix_metrics.json`** is the corresponding corrected graph-fix
experiment re-run (`full_model_graphfix` trained to a real 57 epochs, `pooled_graph_graphfix`
to 27 epochs — both previously affected by the same checkpoint-selection bug and reporting
near-epoch-0, effectively untrained metrics before this fix).

Per-sample predictions for both corrected runs (`results/ablation_predictions/`,
`results/graphfix_minepochs_fix/predictions/`) and the corrected checkpoints
(`results/graphfix_minepochs_fix/checkpoints/`) also need to be pulled down from the
same Google Drive folder before this repository is fully self-contained; they are not
yet included in this checkout.
