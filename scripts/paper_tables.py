"""Generate the manuscript table values from the distributed model and analysis files."""
from pathlib import Path
import csv
import json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
RESULT=ROOT/"results"
OUT=ROOT/"outputs"/"paper_tables"
OUT.mkdir(parents=True,exist_ok=True)

def read(name):
    return json.loads((DATA/(name+".json")).read_text(encoding="utf8"))

def records(name):
    with (RESULT/(name+".csv")).open(encoding="utf8") as stream:
        return list(csv.DictReader(stream))

def save(name,header,rows):
    with (OUT/(name+".csv")).open("w",newline="",encoding="utf8") as stream:
        writer=csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

eff=[]
for model,names in [("2D dimers",["dimer_eff"]),("2D GB",["gb_eff","gb_large"])]:
    entries=[r for name in names for r in read(name)["thresholds"]]
    for quantity,target in [("radial",.005),("joint",.02)]:
        counts=[]
        for method in ("DH","FS"):
            passes=[r for r in entries if r["quantity"]==quantity and r["method"]==method and r["tolerance"]==target and r["reference_precision_gate_passed"] and r["smallest_tested_M_passing_both_test_seeds"]]
            per_design=[min(r["smallest_tested_M_passing_both_test_seeds"] for r in passes if r["protocol"]==mode) for mode in ("contiguous","dispersed_without_replacement")]
            assert len(set(per_design))==1
            counts.append(per_design[0])
        eff.append([model,quantity,target,*counts,counts[0]/counts[1]])
by_m=read("gb3d_eff")["by_M"]
ms=sorted(map(int,by_m))
for quantity,target in [("radial",.005),("angular",.02)]:
    counts=[]
    for method in ("EDH","EFS"):
        designs=[next(m for m in ms if all(by_m[str(j)][mode][quantity][method]["criterion_pass"] for j in ms if j>=m)) for mode in ("contiguous","dispersed")]
        assert designs[0]==designs[1]
        counts.append(designs[0])
    eff.append(["3D GB","joint" if quantity=="angular" else "radial",target,*counts,counts[0]/counts[1]])
save("I",["model","observable","target","EDH","EFS","ratio"],eff)
save("S1",["model","seed","role","frames","equilibration_sweeps"],read("chains"))
ref=read("reference_precision")
save("S2",["model","quantity","block_frames","error_norm_percent"],[[model,q,b,100*v] for model,qs in ref.items() for q,bs in qs.items() for b,v in bs.items()])
joint=records("joint_errors")
save("S3",["seed","dr","angular_cells","ratio"],[[r["seed"],r["dr"],int(r["bins"])**2,r["ratio"]] for r in joint if r["quantity"]=="full" and r["M"]=="1000" and r["bootstrap_block"]=="500"])
save("S4",["seed","M","estimator","1000_RMS"],[[r["seed"],r["M"],r["method"],1000*float(r["RMS"])] for r in records("mode_errors") if r["bootstrap_block"]=="500"])
shell=records("shell_check")
save("S5",["seed","region","100000_weighted_MSE_difference"],[[r["seed"],r["region"],1e5*float(r["full_weighted_MSE_difference"])] for r in shell if float(r["width"])==.05])
cross=[["GB",r["seed"],r["width"],r["ratio"]] for r in shell if r["region"]=="full"]
cross.extend(["Dimers",r["seed"],r["dr"],r["ratio"]] for r in records("dimer_check") if r["domain"]=="full" and r["bins_per_axis"]=="12" and r["block"]=="500")
save("S6",["model","seed","dr","ratio"],cross)

