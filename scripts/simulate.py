"""Run a Monte Carlo simulation using a model configuration."""
from _common import ROOT
import argparse,json
from pathlib import Path
from types import SimpleNamespace

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int);p.add_argument('--equil',type=int);args=p.parse_args()
    config=json.loads(args.config.read_text());model=config['model'];v=config['parameters'];keys=['N','rho','T','equil','frames','stride','seed','step','turn']
    if model=='dimer2d':
        from anisotropic_fs.legacy.dimer_mc import run
        keys+=['ell']
    elif model=='gb2d':
        from anisotropic_fs.legacy.gb_mc import run
        keys+=['kappa']
    elif model=='gb3d':from anisotropic_fs.legacy.gb3d_mc import run
    else:raise ValueError(model)
    kw={k:v[k] for k in keys};kw['out']=str(args.out.resolve())
    if args.frames is not None:kw['frames']=args.frames
    if args.equil is not None:kw['equil']=args.equil
    if kw['frames']<1 or kw['equil']<0:raise ValueError('frames >= 1 and equil >= 0 required')
    if args.out.exists():raise FileExistsError('Choose a new run directory; existing results are not overwritten.')
    print('Starting',model,kw,flush=True);run(SimpleNamespace(**kw))

if __name__=='__main__':main()
