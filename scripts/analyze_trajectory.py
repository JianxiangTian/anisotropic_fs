"""Block EDH/EFS averages from a simulation directory containing snapshots.npy."""
from _common import ROOT
import argparse,json
from pathlib import Path
import numpy as np
from anisotropic_fs.estimators import estimate_frame,radial_grid,MODELS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--block-size',type=int,default=100);args=p.parse_args()
params=json.loads((args.run/'parameters.json').read_text());frames=np.load(args.run/'snapshots.npy',mmap_mode='r',allow_pickle=False)
# This public estimator interface uses the manuscript potential parameters.
if args.model=='dimer2d' and not np.isclose(float(params.get('ell',float('nan'))),0.5):
    raise ValueError('The bundled dimer estimator requires ell=0.5.')
if args.model=='gb2d' and not np.isclose(float(params.get('kappa',float('nan'))),3.0):
    raise ValueError('The bundled planar GB estimator requires kappa=3.')
if len(frames)==0:
    raise ValueError('The trajectory contains no configurations.')
if args.block_size<1 or len(frames)%args.block_size:raise ValueError('Use a block size dividing the number of saved frames; no frames are silently discarded.')
if args.out.exists():raise FileExistsError('Choose a new output directory')
args.out.mkdir(parents=True);blocks=[];edges=radial_grid(args.model)
for start in range(0,len(frames),args.block_size):
    total=None
    for frame in frames[start:start+args.block_size]:
        x=estimate_frame(args.model,frame,float(params['L']),float(params['T']),edges);v=np.array([x['edh'],x['efs']]);total=v if total is None else total+v
    blocks.append(total/args.block_size);print(start+args.block_size,'/',len(frames),flush=True)
np.savez_compressed(args.out/'angular_blocks.npz',edges=edges,blocks=np.array(blocks),block_size=args.block_size,frames_used=len(frames))
