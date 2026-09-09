import json, glob, os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

BASE = r"D:\copper\paper1\latest_results"

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten(); y_pred = np.asarray(y_pred).flatten()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nz = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100
    r2 = r2_score(y_true, y_pred)
    da = np.mean(np.sign(y_true) == np.sign(y_pred))*100
    return dict(rmse=rmse, mae=mae, mape=mape, r2=r2, da=da)

print("="*78); print("PART 1: RECOMPUTE-FROM-RAW VERIFICATION (all 5 new cells)"); print("="*78)

cells = []
for line in open(os.path.join(BASE,"decomposition_finalepoch_fix","decomposition_results.jsonl")):
    r = json.loads(line); cells.append((r, os.path.join(BASE,"decomposition_finalepoch_fix","predictions"), "full_model_emd", "emd"))
for line in open(os.path.join(BASE,"multiseed_finalepoch_fix","multiseed_results.jsonl")):
    r = json.loads(line); pfx = f"{r['variant']}_seed{r['seed']}"
    cells.append((r, os.path.join(BASE,"multiseed_finalepoch_fix","predictions"), pfx, pfx))

maxdiff_overall = 0.0
for r, pdir, pfx, name in cells:
    print(f"\n--- {name} ---  n_test = ", end="")
    mses = []
    first = True
    for h in [1,5,10,22]:
        p = np.load(os.path.join(pdir, f"{pfx}_{h}.npy"))
        t = np.load(os.path.join(pdir, f"{pfx}_{h}_true.npy"))
        if first: print(len(t)); first=False
        m = compute_metrics(t,p)
        j = r["test_metrics"][f"h{h}"]
        mses.append(mean_squared_error(t,p))
        diffs = {k: abs(m[k]-j[k]) for k in ["rmse","mae","mape","r2","da"]}
        maxdiff_overall = max(maxdiff_overall, max(diffs[k]/max(abs(j[k]),1e-9) for k in diffs))
        print(f"  h={h:2d} recomputed rmse={m['rmse']:.12f} mae={m['mae']:.12f} da={m['da']:.10f} r2={m['r2']:.10f}")
        print(f"        jsonl      rmse={j['rmse']:.12f} mae={j['mae']:.12f} da={j['da']:.10f} r2={j['r2']:.10f}")
        print(f"        absdiff    rmse={diffs['rmse']:.3e} mae={diffs['mae']:.3e} da={diffs['da']:.3e} mape={diffs['mape']:.3e} r2={diffs['r2']:.3e}")
    avg = float(np.mean(mses))
    print(f"  avg_mse recomputed={avg:.15f}  jsonl={r['test_metrics']['avg_mse']:.15f}  diff={abs(avg-r['test_metrics']['avg_mse']):.3e}")
print(f"\nMAX RELATIVE DIFFERENCE ACROSS ALL 5 CELLS x 4 HORIZONS x 5 METRICS: {maxdiff_overall:.3e}")

print()
print("="*78); print("PART 2: GROUND-TRUTH IDENTITY CHECK ACROSS CELLS"); print("="*78)
ref = {h: np.load(os.path.join(BASE,"decomposition_finalepoch_fix","predictions",f"full_model_emd_{h}_true.npy")) for h in [1,5,10,22]}
for r, pdir, pfx, name in cells[1:]:
    same = all(np.array_equal(ref[h], np.load(os.path.join(pdir,f"{pfx}_{h}_true.npy"))) for h in [1,5,10,22])
    print(f"  {name:35s} ground truth bit-identical to emd cell: {same}")
# also vs ORIGINAL run's ground truth
try:
    o = np.load(os.path.join(BASE,"decomposition","predictions","full_model_emd_1_true.npy"))
    print(f"  original decomposition/emd h1 ground truth identical to new run: {np.array_equal(o, ref[1])}")
except Exception as e:
    print("  original gt check:", e)

print()
print("="*78); print("PART 3: NEW CELLS vs ORIGINAL 19-CELL EPOCH<->MOVEMENT TREND"); print("="*78)

# --- load ORIGINAL 19 cells from their jsonl + checkpoint epochs
import torch
orig = []
srcs = [("k_sweep","k_sweep_results.jsonl"), ("decomposition","decomposition_results.jsonl"),
        ("loss_comparison","loss_results.jsonl"), ("multiseed","multiseed_results.jsonl")]
for d, f in srcs:
    p = os.path.join(BASE, d, f)
    if not os.path.exists(p):
        print("MISSING", p); continue
    for line in open(p):
        r = json.loads(line)
        ck = r.get("checkpoint_path","")
        local = os.path.join(BASE, d, "checkpoints", os.path.basename(ck))
        ep = None
        if os.path.exists(local):
            ep = torch.load(local, map_location="cpu", weights_only=False)["epoch"]
        dg = r["diagnostic"]
        orig.append(dict(axis=d, key=str(r.get("cell_key")), last_ep=ep, best_ep=(ep-20 if ep is not None else None),
                         cos=dg["adjacency_movement"]["mean_cosine_similarity"],
                         rms=float(np.mean([b["rms_ratio_to_init"] for b in dg["adjacency_movement"]["bands"]])),
                         grad=dg["gradient_magnitude"]["emb_grad_norm_mean"],
                         verdict=dg["adjacency_state"]["verdict"].split(" --")[0],
                         mse=r["test_metrics"]["avg_mse"]))
print(f"loaded {len(orig)} original cells")

new = []
for r, pdir, pfx, name in cells:
    dg = r["diagnostic"]
    new.append(dict(axis="NEW", key=name, last_ep=dg["final_epoch"], best_ep=dg["final_epoch"],
                    cos=dg["adjacency_movement"]["mean_cosine_similarity"],
                    rms=float(np.mean([b["rms_ratio_to_init"] for b in dg["adjacency_movement"]["bands"]])),
                    grad=dg["gradient_magnitude"]["emb_grad_norm_mean"],
                    verdict=dg["adjacency_state"]["verdict"].split(" --")[0],
                    mse=r["test_metrics"]["avg_mse"]))

def spear(xs, ys):
    return stats.spearmanr(xs, ys)

O = [c for c in orig if c["best_ep"] is not None]
print("\nORIGINAL 19 (recomputed):")
for c in sorted(O, key=lambda c:c["best_ep"]):
    print(f"  {c['axis']:14s} {c['key']:34s} last={c['last_ep']:3d} best_ep={c['best_ep']:3d} cos={c['cos']:.4f} rms={c['rms']:.4e} grad={c['grad']:.2e} {c['verdict']}")
r1 = spear([c["best_ep"] for c in O], [c["cos"] for c in O])
r2_ = spear([c["best_ep"] for c in O], [c["rms"] for c in O])
print(f"  n={len(O)} Spearman(epoch,cos) = {r1.statistic:.4f} p={r1.pvalue:.2e}")
print(f"       Spearman(epoch,rms) = {r2_.statistic:.4f} p={r2_.pvalue:.2e}")
# linear fit cosine
sl, ic, rv, pv, se = stats.linregress([c["best_ep"] for c in O],[c["cos"] for c in O])
print(f"  cosine fit: cos = {ic:.5f} + {sl:.6f}*epoch   R2={rv**2:.4f}")

print("\nNEW 5 (diagnosed at final epoch):")
for c in new:
    pred = ic + sl*c["best_ep"]
    print(f"  {c['key']:34s} final_ep={c['last_ep']:3d} cos={c['cos']:.5f} (fit pred {pred:+.4f}, resid {c['cos']-pred:+.4f}) rms={c['rms']:.4e} grad={c['grad']:.2e} {c['verdict']}")
print(f"  original fit max |residual| = {max(abs(c['cos']-(ic+sl*c['best_ep'])) for c in O):.4f}")

ALL = O + new
r3 = spear([c["best_ep"] for c in ALL],[c["cos"] for c in ALL])
r4 = spear([c["best_ep"] for c in ALL],[c["rms"] for c in ALL])
print(f"\nPOOLED n={len(ALL)}: Spearman(epoch,cos) = {r3.statistic:.4f} p={r3.pvalue:.3e}")
print(f"POOLED n={len(ALL)}: Spearman(epoch,rms) = {r4.statistic:.4f} p={r4.pvalue:.3e}")
lr = np.log10(np.maximum([c["rms"] for c in ALL],1e-30))
r5 = stats.spearmanr([c["best_ep"] for c in ALL], lr)
pe = stats.pearsonr([c["best_ep"] for c in ALL], lr)
print(f"POOLED n={len(ALL)}: Spearman(epoch,log10 rms) = {r5.statistic:.4f} p={r5.pvalue:.3e}; Pearson r={pe.statistic:.4f} p={pe.pvalue:.3e}")

# restricted: exclude numerically-underflowed cells (rms < 1e-7) for the LINEAR cosine fit
SUB = [c for c in ALL if c["rms"] > 1e-7]
sl2, ic2, rv2, pv2, se2 = stats.linregress([c["best_ep"] for c in SUB],[c["cos"] for c in SUB])
print(f"\nLinear cosine fit on n={len(SUB)} non-underflowed cells: cos = {ic2:.5f} + {sl2:.6f}*ep  R2={rv2**2:.4f}")
sl3, ic3, rv3, pv3, _ = stats.linregress([c["best_ep"] for c in ALL],[c["cos"] for c in ALL])
print(f"Linear cosine fit on all n={len(ALL)}:                 cos = {ic3:.5f} + {sl3:.6f}*ep  R2={rv3**2:.4f}")

# residual-by-axis ANOVA on the pooled set using the pooled fit
from collections import defaultdict
res = defaultdict(list)
for c in ALL:
    res[c["axis"]].append(c["cos"] - (ic3 + sl3*c["best_ep"]))
print("\nResiduals from pooled linear fit, by axis:")
for k,v in res.items():
    print(f"  {k:14s} n={len(v):2d} mean={np.mean(v):+.4f} sd={np.std(v,ddof=1) if len(v)>1 else float('nan'):.4f}")
F = stats.f_oneway(*[v for v in res.values() if len(v)>1])
print(f"  ANOVA F={F.statistic:.4f} p={F.pvalue:.4f}")
H = stats.kruskal(*[v for v in res.values() if len(v)>1])
print(f"  Kruskal H={H.statistic:.4f} p={H.pvalue:.4f}")

print()
print("="*78); print("PART 4: COLLAPSE COUNT"); print("="*78)
nc_old = [c for c in O if c["verdict"].startswith("NOT")]
print("original NOT COLLAPSED cells:", [(c['key'],c['best_ep']) for c in nc_old])
print("new cells verdicts:", [(c['key'],c['last_ep'],c['verdict']) for c in new])
trained_old = [c for c in O if c["best_ep"] >= 5]
print(f"original cells with best_ep>=5: {len(trained_old)}, all COLLAPSED: {all(c['verdict']=='COLLAPSED' for c in trained_old)}")

print()
print("="*78); print("PART 5: MULTISEED COMPARISON, ORIGINAL vs NEW CELLS"); print("="*78)
ms = [json.loads(l) for l in open(os.path.join(BASE,"multiseed","multiseed_results.jsonl"))]
for h in [1,5,10,22]:
    for v in ["full_model","pooled_graph_matched_dim"]:
        rows = sorted([(r["seed"], r["test_metrics"][f"h{h}"]["rmse"]) for r in ms if r["variant"]==v])
        print(f"  h={h:2d} {v:26s} " + " ".join(f"s{s}={x:.6f}" for s,x in rows))
print("\nNEW re-run RMSE for the 4 re-run cells (vs their original counterparts):")
for r,_,_,name in cells[1:]:
    orig_row = [x for x in ms if x["variant"]==r["variant"] and x["seed"]==r["seed"]][0]
    print(f"  {name:32s} " + " ".join(f"h{h}: new={r['test_metrics'][f'h{h}']['rmse']:.6f} old={orig_row['test_metrics'][f'h{h}']['rmse']:.6f}" for h in [1,22]))

print("\nPaired test on ORIGINAL 5 seeds (reproduce), then with new-run values substituted for seed44 full / seed45 pooled:")
for h in [1,5,10,22]:
    seeds=[42,43,44,45,46]
    a=[[x for x in ms if x["variant"]=="full_model" and x["seed"]==s][0]["test_metrics"][f"h{h}"]["rmse"] for s in seeds]
    b=[[x for x in ms if x["variant"]=="pooled_graph_matched_dim" and x["seed"]==s][0]["test_metrics"][f"h{h}"]["rmse"] for s in seeds]
    t=stats.ttest_rel(a,b); w=stats.wilcoxon(a,b)
    a2=list(a); b2=list(b)
    newmap={(r["variant"],r["seed"]):r for r,_,_,_ in cells[1:]}
    a2[2]=newmap[("full_model",44)]["test_metrics"][f"h{h}"]["rmse"]
    a2[3]=newmap[("full_model",45)]["test_metrics"][f"h{h}"]["rmse"]
    b2[2]=newmap[("pooled_graph_matched_dim",44)]["test_metrics"][f"h{h}"]["rmse"]
    b2[3]=newmap[("pooled_graph_matched_dim",45)]["test_metrics"][f"h{h}"]["rmse"]
    t2=stats.ttest_rel(a2,b2); w2=stats.wilcoxon(a2,b2)
    print(f"  h={h:2d} ORIG t={t.statistic:+.3f} p={t.pvalue:.3f} wil_p={w.pvalue:.3f} | mean_full={np.mean(a):.6f} mean_pool={np.mean(b):.6f}")
    print(f"        SUBS t={t2.statistic:+.3f} p={t2.pvalue:.3f} wil_p={w2.pvalue:.3f} | mean_full={np.mean(a2):.6f} mean_pool={np.mean(b2):.6f}")

print()
print("="*78); print("PART 6: CHECKPOINT CONTENTS OF NEW RUNS (does .pt hold best or final state?)"); print("="*78)
for d, f in [("decomposition_finalepoch_fix","full_model_emd.pt")] + [("multiseed_finalepoch_fix",x) for x in ["full_model_seed44.pt","full_model_seed45.pt","pooled_graph_matched_dim_seed44.pt","pooled_graph_matched_dim_seed45.pt"]]:
    p = os.path.join(BASE,d,"checkpoints",f)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    embk = [k for k in sd if ".emb1." in k or ".emb2." in k]
    rms = float(np.mean([sd[k].float().pow(2).mean().sqrt().item() for k in embk]))
    nparam = sum(v.numel() for v in sd.values())
    print(f"  {f:38s} epoch={ck['epoch']:3d} completed={ck['completed']} params={nparam:,} emb_rms_in_ckpt={rms:.4e}")
