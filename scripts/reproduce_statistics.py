"""Replay numerical summaries and assert agreement with archived manuscript results."""
from _common import ROOT
import argparse,csv,json
from pathlib import Path
import numpy as np
from anisotropic_fs.reproduction import read_json,planar_summary,reference_3d,efficiency_3d,endpoints,secondary,table_ii

def close(a,b,label):
    if not np.allclose(a,b,rtol=1e-9,atol=1e-11,equal_nan=True):
        raise AssertionError(f'{label}: recomputation differs from archived result')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'outputs/statistics');p.add_argument('--primary-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','comparisons':[]};planar={}
    ep=endpoints(ROOT,all_blocks=not args.primary_only);saved=read_json(ROOT/'data/reported/primary_endpoint.json')['datasets']
    for dataset,bs in ep.items():
        for size,row in bs.items():
            for key in ['mean','SE','p','CI95_bootstrap_t']:
                close(row[key],saved[dataset]['blocks'][size][key],f'{dataset}/{size}/{key}')
    report['endpoints']=ep;report['comparisons'].append('Primary bootstrap means, SEs, p values and confidence intervals')
    print('Primary endpoint reproduced.',flush=True)
    if not args.primary_only:
        for model in ['dimer2d','gb2d','gb2d_extended']:
            x=planar_summary(ROOT,model);saved=read_json(ROOT/f'data/efficiency/{model}.json')
            for q,row in x['reference_check'].items():
                for key,v in row.items():close(v,saved['reference_check'][q][key],f'{model}/reference/{q}/{key}')
            for a,b in zip(x['summaries'],saved['summaries']):
                for key in ['test','protocol','quantity','method','M']:assert a[key]==b[key]
                for key in ['error_q95','conditional_q95_CI95']:close(a[key],b[key],f'{model}/{key}/{a["M"]}')
            planar[model]=x;print(model,'reference and percentile summaries reproduced.',flush=True)
        report['planar']=planar
        ref=reference_3d(ROOT);saved=read_json(ROOT/'data/reported/reference_adequacy.json')['channels']
        for q,bs in ref.items():
            for size,row in bs.items():
                for key,v in row.items():close(v,saved[q][size][key],f'3D reference/{q}/{size}/{key}')
        report['reference_3d']=ref;print('3D reference errors reproduced.',flush=True)
        three=efficiency_3d(ROOT);saved=read_json(ROOT/'data/efficiency/gb3d.json')['by_M']
        for m,modes in three.items():
            for mode,qs in modes.items():
                for q,row in qs.items():
                    for method in ['EDH','EFS']:
                        for key,v in row[method].items():close(v,saved[m][mode][q][method][key],f'3D efficiency/{m}/{mode}/{q}/{method}/{key}')
                    close(row['ratio'],saved[m][mode][q]['EFS_over_EDH_RMS_ratio'],'RMS ratio')
                    close(row['ratio_CI95'],saved[m][mode][q]['paired_ratio_CI95'],'paired ratio CI')
        report['efficiency_3d']=three
        sec=secondary(ROOT);saved=read_json(ROOT/'data/reported/secondary_endpoint.json')['datasets']
        for dataset,bs in sec.items():
            for size,row in bs.items():
                for k,v in row.items():
                    if isinstance(v,dict):
                        for kk,vv in v.items():close(vv,saved[dataset][size][k][kk],f'{dataset}/{size}/{k}/{kk}')
                    else:close(v,saved[dataset][size][k],f'{dataset}/{size}/{k}')
        report['secondary']=sec;report['comparisons'].extend(['Planar independent-reference bootstrap and percentile confidence bounds','3D reference multiplier errors and compatibility','3D paired RMS confidence bounds','3D primary cross-seed and secondary Holm families'])
        rows=table_ii(ROOT,planar,three)
    else:rows=table_ii(ROOT)
    expected={('2D dimers','radial'):(500,200),('2D dimers','joint'):(2000,1000),('2D Gay-Berne','radial'):(1000,500),('2D Gay-Berne','joint'):(5000,2000),('3D Gay-Berne','radial'):(500,100),('3D Gay-Berne','joint'):(2000,1000)}
    for row in rows:assert (row['EDH_M'],row['EFS_M'])==expected[(row['system'],row['observable'])]
    with (args.out/'table_II.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['table_II']=rows;report['status']='passed';(args.out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('PASS. Results written to',args.out)

if __name__=='__main__':main()
