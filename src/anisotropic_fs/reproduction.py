"""Recompute manuscript statistics from the bundled archived numerical inputs."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind
from .statistics import boot,grouped,holm,stats

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def metric(value,reference,weights):
    weights=weights.reshape((-1,)+(1,)*(reference.ndim-1))
    return float(np.sqrt(np.sum(weights*(value-reference)**2)/np.sum(weights*reference**2)))

def planar_summary(root,model):
    """Calculate reference-bootstrap uncertainties and q95 confidence intervals.

    Per-repeat input errors include the corresponding reference draw.
    """
    root=Path(root);saved=read_json(root/f"data/efficiency/{model}.json")
    base="gb2d" if model.startswith("gb2d") else model
    z=np.load(root/f"data/blocks/{base}_reference500.npz",allow_pickle=False)
    rng=np.random.default_rng(saved["resampling_seed"]);repeats=saved["repeats"];ms=saved["M"]
    means={q:z[q].mean((0,1)) for q in ["radial","joint"]};draws={q:[] for q in means}
    for _ in range(repeats):
        ix=[rng.integers(0,40,40) for _ in range(2)]
        for q in means:draws[q].append(np.mean([a[idx].mean(0) for a,idx in zip(z[q],ix)],axis=0))
    ref={}
    for q in ["radial","joint"]:
        aa=z[q];se=np.sqrt(sum(a.var(0,ddof=1)/len(a) for a in aa))/2
        ref[q]=dict(relative_rms_SE=metric(means[q]+se,means[q],z['weights']),
                    bootstrap_error_q95=float(np.quantile([metric(x,means[q],z['weights']) for x in draws[q]],.95)),
                    between_reference_run_nrmse=metric(aa[0].mean(0),aa[1].mean(0),z['weights']))
        merged=[a.reshape(20,2,*a.shape[1:]).mean(1) for a in aa]
        rr=[np.mean([a[rng.integers(0,20,20)].mean(0) for a in merged],axis=0) for _ in range(repeats)]
        ref[q]['bootstrap1000_error_q95']=float(np.quantile([metric(x,means[q],z['weights']) for x in rr],.95))
    # Original selections consume RNG state before the summary bootstrap.
    for test in ['test1','test2']:
        for mode in ['dispersed_without_replacement','contiguous']:
            for m in ms:
                for _ in range(repeats):
                    if mode=='contiguous':rng.integers(0,10000-m+1)
                    else:rng.choice(10000,m,replace=False)
    with (root/f'data/efficiency/{model}_errors.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    summaries=[]
    for test in ['test1','test2']:
        for mode in ['dispersed_without_replacement','contiguous']:
            for q in ['radial','joint']:
                for method in ['DH','FS']:
                    for m in ms:
                        v=np.array([float(x['nrmse_with_reference_uncertainty']) for x in rows if x['test']==test and x['protocol']==mode and x['quantity']==q and x['method']==method and int(x['M'])==m])
                        assert len(v)==repeats
                        vals=np.quantile(v[rng.integers(0,len(v),(2000,len(v)))],.95,axis=1)
                        summaries.append(dict(test=test,protocol=mode,quantity=q,method=method,M=m,error_q95=float(np.quantile(v,.95)),conditional_q95_CI95=np.quantile(vals,[.025,.975]).tolist()))
    return dict(reference_check=ref,summaries=summaries)

def reference_3d(root):
    z=np.load(Path(root)/'data/blocks/gb3d_reference500.npz',allow_pickle=False);x=z['blocks'];w=z['weights']
    rng=np.random.default_rng(810144);out={}
    for label in ['radial','angular']:
        a=x[:,:,:,128:129] if label=='radial' else x[:,:,:,:128]
        weights=np.repeat(w/a.shape[-1],a.shape[-1]);a=a.reshape(2,160,-1)
        ref=a.mean((0,1));den=np.sqrt(np.sum(weights*ref**2));diff=a[0].mean(0)-a[1].mean(0);obs=np.sqrt(np.sum(weights*diff**2))/den;out[label]={}
        for size in [500,1000,2000,4000]:
            blocks=a.reshape(2,-1,size//500,a.shape[-1]).mean(2);n=blocks.shape[1]
            centered=blocks-blocks.mean(1,keepdims=True);h=(centered*np.sqrt(weights)).reshape(2*n,-1)/np.sqrt(n*(n-1));gram=h@h.T;norms=[]
            for _ in range(20):
                zz=rng.standard_normal((100,2*n));norms.extend((np.sqrt(np.maximum(0,np.einsum('bi,ij,bj->b',zz,gram,zz)))/den).tolist())
            norms=np.array(norms);u95=float(np.quantile(norms/2,.95));target=.005 if label=='radial' else .02
            out[label][str(size)]=dict(reference_RMS_SE=float(np.sqrt(np.trace(gram))/2/den),reference_error_norm95=u95,target_over3=target/3,precision_pass=bool(u95<target/3),seed_difference_NRMSE=float(obs),compatibility_norm_p=float((1+np.sum(norms>=obs))/2001),blocks_per_seed=n)
    return out

def efficiency_3d(root):
    d=read_json(Path(root)/'data/efficiency/gb3d.json');rng=np.random.default_rng(810146);out={}
    for m in sorted(d['by_M'],key=int):
        out[m]={}
        for mode in ['contiguous','dispersed']:
            ix=rng.integers(0,200,(2000,200));out[m][mode]={}
            for q in ['radial','angular']:
                r=d['by_M'][m][mode][q];sq=np.array([np.square(r[k]['errors']) for k in ['EDH','EFS']]).T
                vals=np.sqrt(sq[ix].mean(1));out[m][mode][q]={}
                for j,k in enumerate(['EDH','EFS']):
                    ci=np.quantile(vals[:,j],[.025,.975]);upper=float(ci[1]+d['reference_u95_max'][q]);target=.005 if q=='radial' else .02
                    out[m][mode][q][k]=dict(RMS=float(np.sqrt(sq[:,j].mean())),conditional_RMS_CI95=ci.tolist(),criterion_upper=upper,criterion_pass=bool(upper<target))
                out[m][mode][q]['ratio']=float(np.sqrt(sq[:,1].mean()/sq[:,0].mean()))
                out[m][mode][q]['ratio_CI95']=np.quantile(vals[:,1]/vals[:,0],[.025,.975]).tolist()
    return out

def endpoints(root,all_blocks=True):
    x=np.load(Path(root)/'data/blocks/gb3d_targets100.npz',allow_pickle=False)['targets100'];out={}
    for label,a in [('original20k',x[:,:200]),('new60k',x[:,200:]),('combined80k',x)]:
        out[label]={str(size):boot(np.array([grouped(row,size//100) for row in a])) for size in ([100,200,500,1000,2000,4000] if all_blocks else [500])}
    return out

def secondary(root):
    z=np.load(Path(root)/'data/blocks/gb3d_coarse100.npz',allow_pickle=False);out={}
    for label,sl in [('original20k',slice(0,200)),('new60k',slice(200,None)),('combined80k',slice(None))]:
        out[label]={}
        for size in [100,200,500,1000,2000,4000]:
            result={}
            for name,key,paired in [('moments60','features',True),('torque24','weak',False),('angles1536','angles',True)]:
                ps=[]
                for row in z[key][:,sl]:
                    a=grouped(row,size//100);v=a[:,1]-a[:,0] if paired else a;ps.extend(stats(v)[2])
                adj=holm(ps);result[name]=dict(minimum_Holm_p=float(adj.min()),rejections=int((adj<.05).sum()))
            target=z['features'][:,sl,1,2]-z['features'][:,sl,0,2];rows=[grouped(row,size//100) for row in target]
            result['individual_primary_Holm3']=holm([float(stats(row)[2]) for row in rows]).tolist()
            result['cross_seed_primary_Holm3']=holm([ttest_ind(rows[i],rows[j],equal_var=False).pvalue for i,j in [(0,1),(0,2),(1,2)]]).tolist()
            out[label][str(size)]=result
    return out

def table_ii(root,planar=None,three=None):
    root=Path(root);rows=[]
    for model,title in [('dimer2d','2D dimers'),('gb2d','2D Gay-Berne')]:
        datasets=[read_json(root/f'data/efficiency/{model}.json')]
        if model=='gb2d':datasets.append(read_json(root/'data/efficiency/gb2d_extended.json'))
        summaries=sum([d['summaries'] for d in datasets],[])
        if planar is not None:summaries=sum([planar[k]['summaries'] for k in ([model] if model=='dimer2d' else [model,'gb2d_extended'])],[])
        for q,target in [('radial',.005),('joint',.02)]:
            for mode in ['contiguous','dispersed_without_replacement']:
                counts={}
                for method in ['DH','FS']:
                    sub=[s for s in summaries if s['quantity']==q and s['protocol']==mode and s['method']==method];ms=sorted(set(s['M'] for s in sub));hits=[m for m in ms if all(s['conditional_q95_CI95'][1]<=target for s in sub if s['M']==m)]
                    ref=datasets[0]['reference_check'][q];assert max(ref['bootstrap_error_q95'],ref['bootstrap1000_error_q95'])<target/3
                    counts[method]=min(hits) if hits else None
                rows.append(dict(system=title,observable=q,design=mode,target=target,EDH_M=counts['DH'],EFS_M=counts['FS'],ratio=counts['DH']/counts['FS']))
    d=three if three is not None else read_json(root/'data/efficiency/gb3d.json')['by_M'];ms=sorted(map(int,d))
    for q,target in [('radial',.005),('angular',.02)]:
        for mode in ['contiguous','dispersed']:
            counts={}
            for method in ['EDH','EFS']:
                hits=[m for i,m in enumerate(ms) if all(d[str(n)][mode][q][method]['criterion_pass'] for n in ms[i:])];counts[method]=min(hits) if hits else None
            rows.append(dict(system='3D Gay-Berne',observable='joint' if q=='angular' else q,design=mode,target=target,EDH_M=counts['EDH'],EFS_M=counts['EFS'],ratio=counts['EDH']/counts['EFS']))
    return rows
