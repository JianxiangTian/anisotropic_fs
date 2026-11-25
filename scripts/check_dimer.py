"""Independent model check of the same M=1000 resolution comparison."""
from pathlib import Path
import sys
import csv
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from analyse import projected_error, windows, blocks, OUT, DATA
edges=np.load(DATA/"dimer_edges.npy")
weights=np.diff(edges**2)
refs=[blocks(f"dimer_edh_{seed}") for seed in (410061,410062)]
rows=[]
for seed in (310053,410063):
    means=np.stack([blocks(f"dimer_{method}_{seed}") for method in ("edh","efs")],axis=1)
    for step in (1,2):
        e=edges[::step]; w=np.diff(e**2)
        for bins in (1,3,6,12):
            def group(a):
                g=a.reshape(*a.shape[:-3],len(w),step,bins,12//bins,bins,12//bins).mean(axis=(-1,-3))
                return np.einsum("...hsij,hs->...hij",g,weights.reshape(-1,step)/w[:,None])
            t=group(means); rs=[group(r) for r in refs]
            for name,mask in [("full",np.ones(len(w),bool)),("interior",(e[:-1]>=.9-1e-9)&(e[1:]<=3.5+1e-9))]:
                v=t[:,:,mask]; ref=[r[:,mask] for r in rs]
                W=w[mask,None,None]*np.ones((bins,bins))
                for b in (5,10):
                    p,bs,ref95,gap=projected_error(v,ref,W,10,b,102518+seed+b)
                    ratio=p[1]/p[0]; lo,hi=np.quantile(bs[:,1]/bs[:,0],[.025,.975])
                    err=(windows(v,10)[::10]-np.mean([a.mean(0) for a in ref],axis=0))**2
                    non=(err*W).sum(axis=(2,3,4)).mean(0)
                    rows.append(dict(seed=seed,M=1000,dr=round(.025*step,3),bins_per_axis=bins,domain=name,block=100*b,ratio=ratio,lo=lo,hi=hi,nonoverlap_ratio=np.sqrt(non[1]/non[0])))
                    if bins==12 and b==5: print(rows[-1],flush=True)
with (OUT/"dimer_check.csv").open("w",newline="",encoding="utf8") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

