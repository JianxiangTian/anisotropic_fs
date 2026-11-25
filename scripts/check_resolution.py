"""Read-only check of the published planar GB resolution comparison."""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import csv
import numpy as np

from analyse import OUT, DATA, EDGES as edges, blocks, REFS, TESTS
refs=[blocks(f"gb_edh_{seed}").mean(axis=0) for seed in REFS]
rows = []
for seed in TESTS:
    arrays = [blocks(f"gb_{method}_{seed}") for method in ("edh","efs")]
    for step in (1, 2):
        e = edges[::step]
        w = np.diff(e**2)
        def rebin(a):
            b = a.reshape(*a.shape[:-3], len(w), step, 12, 12)
            return np.einsum("...hsij,hs->...hij", b, np.diff(edges**2).reshape(-1,step)/w[:,None])
        r1, r2 = [rebin(r) for r in refs]
        r = (r1+r2)/2
        shell_scores = []
        for a in arrays:
            rebinned = rebin(a)
            values = [rebinned[i:i+10].mean(axis=0) for i in range(len(a)-9)]
            means = np.stack(values)
            shell_scores.append([np.mean((means-ref)**2, axis=(0,2,3)) for ref in (r,r1,r2)])
        masks = {"full": np.ones(len(w),dtype=bool), "core": e[1:]<=.9+1e-9,
                 "interior": (e[:-1]>=.9-1e-9)&(e[1:]<=5.5+1e-9),
                 "outer": e[:-1]>=5.5-1e-9}
        assert np.all(sum(masks[k].astype(int) for k in ("core","interior","outer"))==1)
        for name,mask in masks.items():
            m=np.array([[np.average(s[mask],weights=w[mask]) for s in method] for method in shell_scores])
            ratios=np.sqrt(m[1]/m[0])
            rows.append(dict(seed=seed,width=.025*step,region=name,EDH_RMS=np.sqrt(m[0,0]),
                             EFS_RMS=np.sqrt(m[1,0]),ratio=ratios[0],ref1_ratio=ratios[1],ref2_ratio=ratios[2],
                             volume_fraction=w[mask].sum()/w.sum(),full_weighted_MSE_difference=(m[1,0]-m[0,0])*w[mask].sum()/w.sum()))
with (OUT/"shell_check.csv").open("w",newline="",encoding="utf8") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
orig=list(csv.DictReader((OUT/"joint_errors.csv").open()))
for r in rows:
    if r["region"] in ("full","interior"):
        a=next(x for x in orig if x["seed"]==str(r["seed"]) and x["M"]=="1000" and x["bins"]=="12" and x["bootstrap_block"]=="500" and float(x["dr"])==r["width"] and x["quantity"]==r["region"])
        np.testing.assert_allclose(r["ratio"],float(a["ratio"]),rtol=1e-12)
    print(r["seed"],r["width"],r["region"],*[round(r[k],7) for k in ("ratio","ref1_ratio","ref2_ratio","volume_fraction","full_weighted_MSE_difference")])

