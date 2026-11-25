"""Paired window bootstrap conditional on the measured EDH reference."""
from analyse import OUT, EDGES as edges, blocks, TESTS, REFS, np, csv
refs=[blocks(f"gb_edh_{seed}").mean(axis=0) for seed in REFS]
rows = []
for seed in TESTS:
    tracks = [blocks(f"gb_{method}_{seed}") for method in ("edh","efs")]
    for stride in (1,2):
        e=edges[::stride];w=np.diff(e**2)
        def reduce(a):
            return np.einsum("...hsij,hs->...hij",a.reshape(*a.shape[:-3],len(w),stride,12,12), np.diff(edges**2).reshape(-1,stride)/w[:,None])
        ref=reduce(np.mean(refs,axis=0));ts=[reduce(a) for a in tracks]
        for M in (1000,2000):
            m=M//100
            for domain,mask in (("full",np.ones(len(w),bool)),("interior",(e[:-1]>=.9-1e-9)&(e[1:]<=5.5+1e-9))):
                losses=np.array([[np.average(np.mean((t[i:i+m].mean(0)[mask]-ref[mask])**2,axis=(1,2)),weights=w[mask]) for i in range(len(t)-m+1)] for t in ts])
                ratio=np.sqrt(losses[1].mean()/losses[0].mean())
                non=np.sqrt(losses[1,::m].mean()/losses[0,::m].mean())
                for B in (500,1000):
                    n=losses.shape[1]; k=m+B//100-1
                    rng=np.random.default_rng(4839001+seed+M+B)
                    starts=rng.integers(n,size=(2000,int(np.ceil(n/k)),1))
                    ix=((starts+np.arange(k))%n).reshape(2000,-1)[:,:n]
                    p=losses[:,ix].mean(axis=-1);lo,hi=np.quantile(np.sqrt(p[1]/p[0]),[.025,.975])
                    rows.append(dict(seed=seed,width=.025*stride,M=M,domain=domain,block=B,ratio=ratio,conditional_lo=lo,conditional_hi=hi,nonoverlap=non))
with (OUT/"conditional_check.csv").open("w",newline="",encoding="utf8") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
for r in rows:
    if r['M']==1000 and r['block']==500 and r['width']==.05:print(r)

