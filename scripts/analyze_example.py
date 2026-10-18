"""Evaluate EDH and EFS on a supplied example configuration."""
from _common import ROOT
import argparse,json
from pathlib import Path
import numpy as np
from anisotropic_fs.estimators import estimate_frame,MODELS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,default='gb3d');p.add_argument('--out',type=Path,default=ROOT/'outputs/examples');args=p.parse_args()
z=np.load(ROOT/f'data/examples/{args.model}.npz',allow_pickle=False)
result=estimate_frame(args.model,z['frame'],float(z['L']),float(z['T']));args.out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.out/f'{args.model}_estimators.npz',**result)
print(json.dumps({'model':args.model,'particles':int(z['N']),'original_seed':int(z['seed']),'original_frame':int(z['source_frame_index']),'output':str(args.out)}))
