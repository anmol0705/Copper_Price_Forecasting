import optuna, json, numpy as np
from pathlib import Path
optuna.logging.set_verbosity(optuna.logging.WARNING)
R = Path(r"D:\copper\paper1\latest_results\wider_hpo")
s = optuna.load_study(study_name="vmd_mfgnn_wider_hpo", storage=f"sqlite:///{R/'hpo_wider_study.db'}")
print("n_trials", len(s.trials))
from collections import Counter
print(Counter(t.state.name for t in s.trials))
print("\nbest trial #", s.best_trial.number, "value", s.best_trial.value)
print("best params", json.dumps(s.best_trial.params, indent=2))
print("\nsaved hpo_wider_best_params.json:", open(R/'hpo_wider_best_params.json').read())
print("existing tuned:", open(r"D:\copper\paper1\results\archive_paper1\hpo_best_params.json").read())

print("\n--- all COMPLETE trials (sorted by value) ---")
comp=[t for t in s.trials if t.state.name=="COMPLETE"]
for t in sorted(comp,key=lambda t:t.value):
    print(f" #{t.number:>2} val_mse={t.value:.7f}  {t.params}")
vals=np.array([t.value for t in comp])
print(f"\nCOMPLETE n={len(comp)} best={vals.min():.7f} median={np.median(vals):.7f} worst={vals.max():.7f} spread={vals.max()/vals.min():.3f}x")
print(f"best vs 2nd best: {sorted(vals)[0]:.7f} vs {sorted(vals)[1]:.7f}  ({100*(sorted(vals)[1]-sorted(vals)[0])/sorted(vals)[1]:.2f}% better)")

print("\n--- pruned trials ---")
for t in s.trials:
    if t.state.name=="PRUNED":
        iv=t.intermediate_values
        last=max(iv) if iv else None
        print(f" #{t.number:>2} pruned@epoch {last}  best_reported={min(iv.values()) if iv else None}  {t.params}")

print("\n--- param importance (COMPLETE trials) ---")
try:
    imp=optuna.importance.get_param_importances(s)
    for k,v in imp.items(): print(f"  {k:<16}{v:.4f}")
except Exception as e: print("  failed:",e)

print("\n--- marginal: best value per K ---")
for k in sorted({t.params.get('K') for t in comp if 'K' in t.params}):
    sub=[t.value for t in comp if t.params.get('K')==k]
    print(f"  K={k}: n={len(sub)} best={min(sub):.7f} mean={np.mean(sub):.7f}")
print("--- best per hidden_dim / num_gnn_layers / batch_size ---")
for key in ["hidden_dim","num_heads","num_gnn_layers","batch_size"]:
    for v in sorted({t.params.get(key) for t in comp if key in t.params}):
        sub=[t.value for t in comp if t.params.get(key)==v]
        print(f"  {key}={v}: n={len(sub)} best={min(sub):.7f} mean={np.mean(sub):.7f}")

import pandas as pd
df=pd.read_csv(R/'hpo_wider_trials.csv')
print("\ntrials csv cols:", list(df.columns))
