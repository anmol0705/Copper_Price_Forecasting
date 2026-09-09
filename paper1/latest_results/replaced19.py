import json, os, numpy as np, torch
from scipy import stats
from collections import defaultdict
B = r"D:\copper\paper1\latest_results"
L = lambda d, f: [json.loads(l) for l in open(os.path.join(B, d, f))]

def row(axis, key, ep_idx, dg, mse):
    return dict(axis=axis, key=key, i=ep_idx, E=ep_idx + 1,
                cos=dg["adjacency_movement"]["mean_cosine_similarity"],
                rms=float(np.mean([b["rms_ratio_to_init"] for b in dg["adjacency_movement"]["bands"]])),
                grad=dg["gradient_magnitude"]["emb_grad_norm_mean"],
                verdict=dg["adjacency_state"]["verdict"].split(" --")[0], mse=mse)

cells = {}
for d, f in [("k_sweep","k_sweep_results.jsonl"),("decomposition","decomposition_results.jsonl"),
             ("loss_comparison","loss_comparison_results.jsonl"),("multiseed","multiseed_results.jsonl")]:
    for r in L(d, f):
        ck = os.path.join(B, d, "checkpoints", os.path.basename(r["checkpoint_path"]))
        ep = torch.load(ck, map_location="cpu", weights_only=False)["epoch"]
        cells[str(r["cell_key"])] = row(d, str(r["cell_key"]), ep - 20, r["diagnostic"], r["test_metrics"]["avg_mse"])

ORIG = {k: dict(v) for k, v in cells.items()}
# replace the 3 artefact cells with their final-epoch re-diagnoses
nd = L("decomposition_finalepoch_fix","decomposition_results.jsonl")[0]
cells["['emd']"] = row("decomposition","['emd'] (final-ep)", nd["diagnostic"]["final_epoch"], nd["diagnostic"], nd["test_metrics"]["avg_mse"])
MS = L("multiseed_finalepoch_fix","multiseed_results.jsonl")
def ms(v,s): return [r for r in MS if r["variant"]==v and r["seed"]==s][0]
for k,(v,s) in [("['full_model', 44]",("full_model",44)),("['pooled_graph_matched_dim', 45]",("pooled_graph_matched_dim",45))]:
    r=ms(*(v,s)); cells[k]=row("multiseed",k+" (final-ep)", r["diagnostic"]["final_epoch"], r["diagnostic"], r["test_metrics"]["avg_mse"])
R = list(cells.values())
assert len(R)==19

def block(cs,label,use=None):
    e=[c["i"] for c in cs]
    out={}
    for m,arr in [("cos",[c["cos"] for c in cs]),("rms",[c["rms"] for c in cs]),
                  ("log10rms",list(np.log10(np.maximum([c["rms"] for c in cs],1e-30)))),
                  ("log10grad",list(np.log10(np.maximum([c["grad"] for c in cs],1e-40)))),
                  ("mse",[c["mse"] for c in cs])]:
        s=stats.spearmanr(e,arr); p=stats.pearsonr(e,arr)
        print(f"  {label} n={len(cs)} Spearman(ep,{m:9s})={s.statistic:+.4f} p={s.pvalue:.3e} | Pearson={p.statistic:+.4f} p={p.pvalue:.3e}")
    sl,ic,rv,_,_=stats.linregress(e,[c["cos"] for c in cs])
    mr=max(abs(c["cos"]-(ic+sl*c["i"])) for c in cs)
    print(f"  {label} cosine fit (epoch INDEX): cos = {ic:.5f} {sl:+.6f}*i   R2={rv**2:.4f}  max|resid|={mr:.4f}")
    print(f"  {label} cosine fit (EPOCHS=i+1) : cos = {ic-sl:.5f} {sl:+.6f}*E   R2={rv**2:.4f}")
    slr,icr,rvr,_,_=stats.linregress(e,np.log10(np.maximum([c["rms"] for c in cs],1e-30)))
    print(f"  {label} log10(rms) fit: {icr:.4f} {slr:+.4f}*i  R2={rvr**2:.4f}")
    res=defaultdict(list)
    for c in cs: res[c["axis"]].append(c["cos"]-(ic+sl*c["i"]))
    for k,v in sorted(res.items()): print(f"     {k:16s} n={len(v):2d} mean={np.mean(v):+.4f} sd={(np.std(v,ddof=1) if len(v)>1 else float('nan')):.4f}")
    g=[v for v in res.values() if len(v)>1]
    print(f"     ANOVA F={stats.f_oneway(*g).statistic:.4f} p={stats.f_oneway(*g).pvalue:.4f} | Kruskal H={stats.kruskal(*g).statistic:.4f} p={stats.kruskal(*g).pvalue:.4f}")
    return ic,sl,rv**2,mr

print("="*90); print("ORIGINAL 19 (as published in ROBUSTNESS_ANALYSIS_FINAL.md)"); print("="*90)
oic,osl,or2,omr = block(list(ORIG.values()),"ORIG")
print()
print("="*90); print("REPLACED 19 (3 artefact cells swapped for their final-epoch re-diagnoses)"); print("="*90)
print(f"  {'axis':16s} {'cell':44s} {'i':>3s} {'E':>3s} {'cos':>9s} {'rms/init':>11s} {'emb_grad':>10s} verdict")
for c in sorted(R,key=lambda c:c["i"]):
    print(f"  {c['axis']:16s} {c['key']:44s} {c['i']:3d} {c['E']:3d} {c['cos']:9.5f} {c['rms']:11.3e} {c['grad']:10.2e} {c['verdict']}")
nic,nsl,nr2,nmr = block(R,"REPL")
print(f"\n  COLLAPSED count: {sum(1 for c in R if c['verdict']=='COLLAPSED')}/19")
print(f"  epoch index range {min(c['i'] for c in R)}-{max(c['i'] for c in R)}  => EPOCHS OF TRAINING {min(c['E'] for c in R)}-{max(c['E'] for c in R)}")
print(f"  cosine range {min(c['cos'] for c in R):.4f}-{max(c['cos'] for c in R):.4f}")
print(f"  rms/init range {min(c['rms'] for c in R):.3e} ({[c['key'] for c in R if c['rms']==min(x['rms'] for x in R)]}) to {max(c['rms'] for c in R):.3e} ({[c['key'] for c in R if c['rms']==max(x['rms'] for x in R)]})")
print(f"  test avg_mse range {min(c['mse'] for c in R):.6f}-{max(c['mse'] for c in R):.6f}  spread {100*(max(c['mse'] for c in R)/min(c['mse'] for c in R)-1):.2f}%")

print("\n  EMD residual against the ORIGINAL-19 fit: %.4f (orig max|resid| %.4f)" % (cells["['emd']"]["cos"]-(oic+osl*cells["['emd']"]["i"]), omr))
for k in ["['full_model', 44]","['pooled_graph_matched_dim', 45]"]:
    print(f"  {k} residual vs ORIGINAL-19 fit: {cells[k]['cos']-(oic+osl*cells[k]['i']):+.4f}")

print()
print("="*90); print("BONUS RE-RUNS AS SAME-CONFIG REPLICATE PAIRS (not counted in the 19)"); print("="*90)
old_ms=L("multiseed","multiseed_results.jsonl")
for v,s in [("full_model",45),("pooled_graph_matched_dim",44)]:
    o=[r for r in old_ms if r["variant"]==v and r["seed"]==s][0]; n=ms(v,s)
    oi=torch.load(os.path.join(B,"multiseed","checkpoints",f"{v}_seed{s}.pt"),map_location="cpu",weights_only=False)["epoch"]
    ni=n["diagnostic"]["final_epoch"]
    a,b=o["test_metrics"]["avg_mse"],n["test_metrics"]["avg_mse"]
    print(f"  {v} seed{s}: last_epoch {oi} -> {ni} (E {oi+1} -> {ni+1}); avg_mse {a:.6f} vs {b:.6f}  spread {100*(max(a,b)/min(a,b)-1):.2f}%")
    print(f"     old best_state diag: i={oi-20} cos={o['diagnostic']['adjacency_movement']['mean_cosine_similarity']:.5f}")
    print(f"     new final-ep  diag : i={ni} cos={n['diagnostic']['adjacency_movement']['mean_cosine_similarity']:.4e} rms={np.mean([x['rms_ratio_to_init'] for x in n['diagnostic']['adjacency_movement']['bands']]):.3e} verdict={n['diagnostic']['adjacency_state']['verdict'].split(' --')[0]}")
    print(f"     new-cell residual vs REPLACED-19 fit: {n['diagnostic']['adjacency_movement']['mean_cosine_similarity']-(nic+nsl*ni):+.4f}")

print()
print("="*90); print("WITHIN-RUN TWO-SNAPSHOT CHECK (same run, best-val vs final-epoch weights)"); print("="*90)
for k,(od,of,sel) in [("emd",("decomposition","decomposition_results.jsonl",lambda r:r.get("method")=="emd")),
                      ("full_model s44",("multiseed","multiseed_results.jsonl",lambda r:r["variant"]=="full_model" and r["seed"]==44)),
                      ("pooled s45",("multiseed","multiseed_results.jsonl",lambda r:r["variant"]=="pooled_graph_matched_dim" and r["seed"]==45))]:
    o=[r for r in L(od,of) if sel(r)][0]; do=o["diagnostic"]
    n=cells["['emd']"] if k=="emd" else (cells["['full_model', 44]"] if "s44" in k else cells["['pooled_graph_matched_dim', 45]"])
    oi=torch.load(os.path.join(B,od,"checkpoints",os.path.basename(o["checkpoint_path"])),map_location="cpu",weights_only=False)["epoch"]-20
    print(f"  {k:16s} best-val i={oi:2d} (E={oi+1}): cos={do['adjacency_movement']['mean_cosine_similarity']:.5f} rms={np.mean([b['rms_ratio_to_init'] for b in do['adjacency_movement']['bands']]):.4f} grad={do['gradient_magnitude']['emb_grad_norm_mean']:.2e} {do['adjacency_state']['verdict'].split(' --')[0]}")
    print(f"  {'':16s} final    i={n['i']:2d} (E={n['E']}): cos={n['cos']:.5f} rms={n['rms']:.3e} grad={n['grad']:.2e} {n['verdict']}")
