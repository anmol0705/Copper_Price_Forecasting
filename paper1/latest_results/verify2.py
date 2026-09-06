import json, numpy as np, torch
from pathlib import Path
from scipy import stats
R = Path(r"D:\copper\paper1\latest_results")

print("=== 1. best_epoch = checkpoint['epoch'] - patience(20) ===")
specs=[("k_sweep",R/"k_sweep/k_sweep_results.jsonl",lambda r:f"K={r['K']}"),
       ("decomp",R/"decomposition/decomposition_results.jsonl",lambda r:r['method']),
       ("loss",R/"loss_comparison/loss_comparison_results.jsonl",lambda r:r['loss_fn']),
       ("multiseed",R/"multiseed/multiseed_results.jsonl",lambda r:f"{r['variant']}_s{r['seed']}")]
recs=[]
for exp,p,lab in specs:
    for r in [json.loads(l) for l in open(p) if l.strip()]:
        ck=R/Path(r["checkpoint_path"]).relative_to("results/robustness")
        m=torch.load(ck,map_location="cpu",weights_only=False)
        am=r["diagnostic"]["adjacency_movement"]
        recs.append(dict(exp=exp,label=lab(r),last=m["epoch"],best=m["epoch"]-20,
            cos=am["mean_cosine_similarity"],
            rms=float(np.mean([b["rms_ratio_to_init"] for b in am["bands"]])),
            eg=r["diagnostic"]["gradient_magnitude"]["emb_grad_norm_mean"],
            nok=r["diagnostic"]["adjacency_state"]["n_ok"],
            nt=r["diagnostic"]["adjacency_state"]["n_total"],
            amse=r["test_metrics"]["avg_mse"], ck=ck))
recs.sort(key=lambda x:x["best"])
print(f"{'exp':<10}{'label':<32}{'last':>5}{'best_ep':>8}{'cos':>8}{'rms_ratio':>11}{'embgrad':>11}{'nok':>7}{'avg_mse':>10}")
for x in recs:
    print(f"{x['exp']:<10}{x['label']:<32}{x['last']:>5}{x['best']:>8}{x['cos']:>8.4f}{x['rms']:>11.5f}{x['eg']:>11.2e}{x['nok']:>4}/{x['nt']:<2}{x['amse']:>10.6f}"
          + ("   <-- NOT COLLAPSED" if x['nok']==x['nt'] else ""))

be=np.array([x['best'] for x in recs],float); cos=np.array([x['cos'] for x in recs]); rms=np.array([x['rms'] for x in recs]); eg=np.array([x['eg'] for x in recs])
for n,a,b in [("best_epoch vs cosine",be,cos),("best_epoch vs rms_ratio",be,rms),
              ("best_epoch vs log10 rms_ratio",be,np.log10(rms+1e-30)),
              ("best_epoch vs log10 emb_grad",be,np.log10(eg+1e-30)),
              ("best_epoch vs test avg_mse",be,np.array([x['amse'] for x in recs]))]:
    sr,sp=stats.spearmanr(a,b); pr,pp=stats.pearsonr(a,b)
    print(f"  {n:<34} spearman={sr:+.4f} (p={sp:.2e})  pearson={pr:+.4f} (p={pp:.2e})")

print("\n=== 2. Accidental replicates: same config (K=5,VMD,MSE,seed42) ===")
reps=[("k_sweep K=5",R/"k_sweep/predictions/full_model_K5"),("decomp vmd",R/"decomposition/predictions/full_model_vmd"),
      ("loss mse",R/"loss_comparison/predictions/full_model_mse"),("multiseed full_model_seed42",R/"multiseed/predictions/full_model_seed42")]
am=[x['amse'] for x in recs if x['label'] in ('K=5','vmd','mse','full_model_s42')]
print("  test avg_mse:",[f"{v:.6f}" for v in sorted(am)], f" spread={100*(max(am)-min(am))/min(am):.2f}%")
print("  ground-truth array identity across the 4 arms (h=22):")
base=np.load(str(reps[0][1])+"_22_true.npy")
for n,pfx in reps:
    a=np.load(str(pfx)+"_22_true.npy")
    print(f"    {n:<28} shape={a.shape} identical_to_k_sweep={np.array_equal(a,base)} maxdiff={np.abs(a-base).max() if a.shape==base.shape else 'shape-mismatch'}")
print("  prediction array identity (h=22):")
basep=np.load(str(reps[0][1])+"_22.npy")
for n,pfx in reps:
    a=np.load(str(pfx)+"_22.npy")
    print(f"    {n:<28} identical={np.array_equal(a,basep)} maxdiff={np.abs(a-basep).max():.6g} corr={np.corrcoef(a.ravel(),basep.ravel())[0,1]:.4f}")

print("\n=== 3. Huber checkpoint embedding state ===")
for name,f in [("huber",R/"loss_comparison/checkpoints/full_model_huber.pt"),
               ("mse",R/"loss_comparison/checkpoints/full_model_mse.pt"),
               ("mae",R/"loss_comparison/checkpoints/full_model_mae.pt")]:
    sd=torch.load(f,map_location="cpu",weights_only=False)["state_dict"]
    e=[(k,v) for k,v in sd.items() if ".emb1." in k or ".emb2." in k]
    rms=[float(v.pow(2).mean().sqrt()) for _,v in e]; mx=[float(v.abs().max()) for _,v in e]
    print(f"  {name:<6} n_emb_tensors={len(e)} rms range=[{min(rms):.3e},{max(rms):.3e}] max|w| range=[{min(mx):.3e},{max(mx):.3e}]  (Xavier ref 0.2885)")
    if name=="huber":
        import torch.nn.functional as F
        e1=sd["graph_constructors.0.emb1.weight"]; e2=sd["graph_constructors.0.emb2.weight"]
        logits=F.relu(e1@e2.T); print(f"    band0 relu(e1@e2^T): max={float(logits.max()):.3e}  n_nonzero={int((logits>0).sum())}/{logits.numel()}")
        print(f"    softmax of that: std={float(torch.softmax(logits,-1).std()):.3e}  (uniform 1/8={1/8})")

print("\n=== 4. Huber = 0.5*MSE check ===")
maxres=0
for lf in ["mse","mae","huber"]:
    for h in [1,5,10,22]:
        p=np.load(R/f"loss_comparison/predictions/full_model_{lf}_{h}.npy").ravel()
        t=np.load(R/f"loss_comparison/predictions/full_model_{lf}_{h}_true.npy").ravel()
        maxres=max(maxres,float(np.abs(p-t).max()))
print(f"  global max |residual| over all loss arms/horizons = {maxres:.5f}  vs Huber beta = 1.0")
print(f"  -> {100*maxres/1.0:.1f}% of beta; smooth_l1 in quadratic branch for 100% of samples => 0.5*x^2 = 0.5*MSE exactly")
