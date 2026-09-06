import json, glob, os, math
from pathlib import Path
import numpy as np, torch
from scipy import stats

R = Path(r"D:\copper\paper1\latest_results")

def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]

specs = [
    ("k_sweep", R/"k_sweep/k_sweep_results.jsonl", lambda r: f"K={r['K']}", "K", lambda r: r['K']),
    ("decomposition", R/"decomposition/decomposition_results.jsonl", lambda r: r['method'], "decomposition_method", lambda r: r['method']),
    ("loss", R/"loss_comparison/loss_comparison_results.jsonl", lambda r: r['loss_fn'], "loss_fn", lambda r: r['loss_fn']),
    ("multiseed", R/"multiseed/multiseed_results.jsonl", lambda r: f"{r['variant']}_seed{r['seed']}", "seed_x_variant", lambda r: (r['variant'], r['seed'])),
]

recs = []
for exp, path, label, axis, val in specs:
    for r in rows(path):
        ck = R / Path(r["checkpoint_path"]).relative_to("results/robustness")
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        nparam = sum(p.numel() for p in meta["state_dict"].values())
        d = r["diagnostic"]
        am = d["adjacency_movement"]
        rms = [b["rms_ratio_to_init"] for b in am["bands"]]
        recs.append(dict(
            exp=exp, label=label(r), axis=axis,
            epoch_last=meta["epoch"], epochs_run=meta["epoch"]+1,
            completed=meta["completed"], nparam=nparam,
            n_bands=len(am["bands"]),
            cos=am["mean_cosine_similarity"],
            rms_ratio=float(np.mean(rms)),
            emb_grad=d["gradient_magnitude"]["emb_grad_norm_mean"],
            other_grad=d["gradient_magnitude"]["other_grad_norm_mean"],
            n_ok=d["adjacency_state"]["n_ok"], n_total=d["adjacency_state"]["n_total"],
            verdict=d["adjacency_state"]["verdict"].split(" --")[0],
            frac_uniform=float(np.mean([b["frac_near_uniform"] for b in d["adjacency_state"]["bands"]])),
            rmse1=r["test_metrics"]["h1"]["rmse"], rmse22=r["test_metrics"]["h22"]["rmse"],
            avg_mse=r["test_metrics"]["avg_mse"],
            variant=r.get("variant"), seed=r.get("seed"), ckpt=str(ck),
        ))

print(f"{'exp':<13}{'label':<34}{'ep':>4}{'nparam':>9}{'bands':>6}{'cos':>8}{'rms_r':>9}{'embgrad':>11}{'othgrad':>9}{'nok':>5}{'avg_mse':>10}  verdict")
for x in sorted(recs, key=lambda z: (z['exp'], z['label'])):
    print(f"{x['exp']:<13}{x['label']:<34}{x['epochs_run']:>4}{x['nparam']:>9}{x['n_bands']:>6}{x['cos']:>8.4f}{x['rms_ratio']:>9.4f}{x['emb_grad']:>11.2e}{x['other_grad']:>9.4f}{x['n_ok']:>3}/{x['n_total']:<2}{x['avg_mse']:>10.6f}  {x['verdict']}")

print("\nN runs =", len(recs))
ep = np.array([x['epochs_run'] for x in recs], float)
cos = np.array([x['cos'] for x in recs], float)
rr = np.array([x['rms_ratio'] for x in recs], float)
eg = np.array([x['emb_grad'] for x in recs], float)

def rep(name, a, b):
    sr, sp = stats.spearmanr(a, b); pr, pp = stats.pearsonr(a, b)
    print(f"{name:<46} spearman rho={sr:+.4f} (p={sp:.2e})   pearson r={pr:+.4f} (p={pp:.2e})")

print("\n--- ALL RUNS pooled across experiments ---")
rep("epochs vs mean_cosine_similarity", ep, cos)
rep("epochs vs mean rms_ratio_to_init", ep, rr)
rep("epochs vs log10(rms_ratio_to_init)", ep, np.log10(rr))
rep("epochs vs log10(emb_grad_norm)", ep, np.log10(eg + 1e-30))
rep("cosine vs rms_ratio", cos, rr)

# partial: does axis add anything beyond epochs? Use residual approach per axis
print("\n--- Within-axis: is there any axis effect at fixed-ish epochs? ---")
for exp in ["k_sweep","decomposition","loss","multiseed"]:
    sub=[x for x in recs if x['exp']==exp]
    e=np.array([x['epochs_run'] for x in sub],float); c=np.array([x['cos'] for x in sub],float)
    if len(sub)>2:
        sr,sp=stats.spearmanr(e,c)
        print(f"  {exp:<13} n={len(sub)}  epochs {e.astype(int).tolist()}  cos {[round(v,3) for v in c]}  rho={sr:+.3f} p={sp:.3f}")
    else:
        print(f"  {exp:<13} n={len(sub)}  epochs {e.astype(int).tolist()}  cos {[round(v,3) for v in c]}")

# multiseed detail
print("\n--- MULTISEED ---")
ms = rows(R/"multiseed/multiseed_results.jsonl")
print("variants present:", sorted({r['variant'] for r in ms}))
import collections
for v in sorted({r['variant'] for r in ms}):
    sub=[r for r in ms if r['variant']==v]
    print(f"  {v}: seeds={sorted(r['seed'] for r in sub)}")
for h in ["h1","h5","h10","h22"]:
    a=[r["test_metrics"][h]["rmse"] for r in sorted([r for r in ms if r['variant']=='full_model'],key=lambda r:r['seed'])]
    b=[r["test_metrics"][h]["rmse"] for r in sorted([r for r in ms if r['variant']!='full_model'],key=lambda r:r['seed'])]
    t=stats.ttest_rel(a,b); w=stats.wilcoxon(a,b)
    print(f"  {h}: full {np.mean(a):.6f}+-{np.std(a):.6f}  pooled {np.mean(b):.6f}+-{np.std(b):.6f}  "
          f"diff={np.mean(a)-np.mean(b):+.6f}  t={t.statistic:+.3f} p={t.pvalue:.4f}  wilcoxon p={w.pvalue:.4f}")

# param counts
print("\n  param counts:")
for v in sorted({r['variant'] for r in ms}):
    r=[x for x in recs if x['variant']==v][0]
    print(f"    {v}: {r['nparam']:,}")
fm=[x for x in recs if x['variant']=='full_model'][0]['nparam']
pg=[x for x in recs if x['variant'] and x['variant']!='full_model'][0]['nparam']
print(f"    pooled/full = {pg/fm:.4f}  -> deficit {100*(1-pg/fm):.1f}%")

# residuals for huber
print("\n--- HUBER / residual check (loss_comparison predictions) ---")
for lf in ["mse","mae","huber"]:
    mx=[]
    for h in [1,5,10,22]:
        p=np.load(R/f"loss_comparison/predictions/full_model_{lf}_{h}.npy").ravel()
        t=np.load(R/f"loss_comparison/predictions/full_model_{lf}_{h}_true.npy").ravel()
        r_=np.abs(p-t); mx.append((h,r_.max(),r_.mean(),np.percentile(r_,99)))
    print(f"  {lf}: " + "  ".join(f"h{h}: max={m:.4f} mean={me:.4f} p99={p99:.4f}" for h,m,me,p99 in mx))
    print(f"     -> global max |residual| = {max(m for _,m,_,_ in mx):.5f}  (Huber beta=1.0)")

# target scale
print("\n  target scale (h22 true):", end=" ")
t=np.load(R/"loss_comparison/predictions/full_model_mse_22_true.npy").ravel()
print(f"min={t.min():.4f} max={t.max():.4f} std={t.std():.4f}")
