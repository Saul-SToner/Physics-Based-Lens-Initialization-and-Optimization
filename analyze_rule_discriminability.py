"""Paired descriptive analysis of three rules without changing success labels."""

import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import isclose
from pathlib import Path

import numpy as np


TOLERANCES={"f_number":1e-9,"field_deg":1e-9,"mtf":1e-6,"rms_um":1e-3}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    x=np.asarray(values,dtype=float)
    if len(x)==0:
        return {"n":0}
    return {"n":len(x),"min":float(x.min()),"p05":float(np.quantile(x,.05)),
            "q25":float(np.quantile(x,.25)),"median":float(np.median(x)),
            "q75":float(np.quantile(x,.75)),"p95":float(np.quantile(x,.95)),
            "max":float(x.max()),"mean":float(x.mean()),"sample_sd":float(x.std(ddof=1)) if len(x)>1 else None}


def ranks(values):
    x=np.asarray(values)
    order=np.argsort(x,kind="stable")
    result=np.empty(len(x),dtype=float)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and x[order[end]]==x[order[start]]:
            end+=1
        result[order[start:end]]=(start+end-1)/2
        start=end
    return result


def same_specs(a,b):
    return (isclose(a["f_number"],b["f_number"],rel_tol=0,abs_tol=TOLERANCES["f_number"])
            and isclose(a["field_deg"],b["field_deg"],rel_tol=0,abs_tol=TOLERANCES["field_deg"])
            and all(isclose(a[f"frequency_{i}_lpmm"],b[f"frequency_{i}_lpmm"],rel_tol=0,abs_tol=1e-9)
                    for i in (0,1,2)))


def numeric_pareto(rows):
    frontier=[]
    for a in rows:
        dominated=any(
            b["f_number"]<=a["f_number"]+TOLERANCES["f_number"]
            and b["field_deg"]>=a["field_deg"]-TOLERANCES["field_deg"]
            and (b["f_number"]<a["f_number"]-TOLERANCES["f_number"]
                 or b["field_deg"]>a["field_deg"]+TOLERANCES["field_deg"])
            for b in rows)
        if not dominated:
            frontier.append(a["rule"])
    return frontier


def comparison(rows_a,rows_b,metric,higher,tolerance):
    differences=[a[metric]-b[metric] for a,b in zip(rows_a,rows_b)]
    adjusted=[value if higher else -value for value in differences]
    return {"n":len(differences),"a_better":sum(v>tolerance for v in adjusted),
            "numerical_ties":sum(abs(v)<=tolerance for v in adjusted),
            "b_better":sum(v < -tolerance for v in adjusted),
            "exact_ties":sum(v==0 for v in adjusted),
            "a_minus_b":stats(differences),"tie_tolerance":tolerance}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    inputs=root/"feature_engineering_100_v1"
    source_paths=[inputs/name for name in ("outcomes.json","labels.json","features.json")]
    source_paths.append(root/"background_runs/batch_300_20261118/state.json")
    original={str(path):sha256(path.read_bytes()).hexdigest() for path in source_paths}
    outcomes=load(inputs/"outcomes.json")
    labels=load(inputs/"labels.json")
    state=load(source_paths[-1])
    prior_completed=set()
    for history in (source_paths[-1].parent/"history").glob("state_*.json"):
        source_paths.append(history)
        original[str(history)]=sha256(history.read_bytes()).hexdigest()
        prior_completed.update(key for key,value in load(history)["jobs"].items() if value["status"]=="completed")
    rows=[]
    failed_ids={row["case_id"] for row in labels if row["target_success_mask"]==0}
    retry_records=[]
    for outcome in outcomes:
        case_id,rule=outcome["case_id"],outcome["rule"]
        key=f"case_{case_id:03d}_rule_{rule}"
        info=state["jobs"][key]
        directory=source_paths[3].parent/info["attempt"]
        path=directory/"experiment/result.json"
        result=load(path)
        original[str(path)]=sha256(path.read_bytes()).hexdigest()
        if original[str(path)]!=info["acceptance"]["result_sha256"]:
            raise RuntimeError("Accepted result hash mismatch")
        if (result["case_id"],result["rule"])!=(case_id,rule):
            raise RuntimeError("Case/rule identity mismatch")
        rejected=[step for step in result["steps"] if not step["accepted"]]
        last_rejected=rejected[-1] if rejected else None
        cause=("none" if last_rejected is None else
               "geometry" if not last_rejected["geometry_valid"] else
               "analysis_error" if "analysis_error" in last_rejected else "mtf")
        f_number=outcome["final_working_f_number"]
        field=outcome["final_max_field_deg"]
        f_short=f_number>20
        field_short=field<5
        deficit=("both" if f_short and field_short else
                 "aperture_only" if f_short else "field_only" if field_short else "none")
        row={"case_id":case_id,"rule":rule,"target_reached":outcome["target_reached"],
             "all_rules_failed":case_id in failed_ids,"f_number":f_number,"field_deg":field,
             "f_number_excess_ratio":(f_number-20)/20,
             "aperture_progress_fraction":20/f_number,"field_progress_fraction":field/5,
             "field_shortfall_fraction":(5-field)/5,"deficit_type":deficit,
             "last_rejection_type":cause,
             "last_rejection_reasons":last_rejected.get("rejection_reasons",[]) if last_rejected else [],
             "last_rejection_geometry_violations":last_rejected.get("geometry_violations",[]) if last_rejected else [],
             "stop_reasons":outcome["stop_reasons"],
             "requested_cycles":outcome["cost"]["requested_dls_cycles"],
             "actual_completed_cycles":outcome["cost"]["completed_dls_cycles"],
             "dls_seconds":outcome["cost"]["dls_seconds"],"group_seconds":outcome["cost"]["wall_seconds"],
             "attempts":outcome["cost"]["attempts"],
             "seconds_per_requested_cycle":outcome["cost"]["dls_seconds"]/outcome["cost"]["requested_dls_cycles"],
             "completion_epoch":"earlier_session" if key in prior_completed else "final_session",
             "rejected_steps":len(rejected)}
        for i,field_result in enumerate(outcome["fields"]):
            row.update({
                f"mtf_min_{i}":min(field_result["mtf_tangential"],field_result["mtf_sagittal"]),
                f"mtf_tangential_{i}":field_result["mtf_tangential"],
                f"mtf_sagittal_{i}":field_result["mtf_sagittal"],
                f"rms_{i}_um":field_result["rms_spot_radius"],
                f"frequency_{i}_lpmm":field_result["mtf_frequency_lp_per_mm"],
            })
        row["minimum_mtf_threshold_margin"]=min(row[f"mtf_min_{i}"]-threshold for i,threshold in enumerate((.3,.4,.5)))
        rows.append(row)
        for candidate in directory.parent.glob("attempt_*/experiment/result.json"):
            if candidate==path:
                continue
            retry=load(candidate)
            original[str(candidate)]=sha256(candidate.read_bytes()).hexdigest()
            retry_records.append({
                "case_id":case_id,"rule":rule,"path":str(candidate),
                "observed_steps":len(retry["steps"]),"requested_cycles":retry["cost"]["requested_dls_cycles"],
                "dls_seconds":retry["cost"]["dls_seconds"],"group_seconds":retry["cost"]["wall_seconds"],
            })
    by_rule={rule:sorted((r for r in rows if r["rule"]==rule),key=lambda r:r["case_id"]) for rule in (1,2,3)}
    metrics=["f_number","field_deg",*[f"mtf_min_{i}" for i in (0,1,2)],
             *[f"rms_{i}_um" for i in (0,1,2)],"dls_seconds","group_seconds","requested_cycles",
             "seconds_per_requested_cycle"]
    distributions={str(rule):{metric:stats([row[metric] for row in values]) for metric in metrics}
                   for rule,values in by_rule.items()}
    failures={}
    for rule,values in by_rule.items():
        subset=[r for r in values if r["all_rules_failed"]]
        failures[str(rule)]={
            "n":len(subset),"deficit_types":dict(Counter(r["deficit_type"] for r in subset)),
            "last_rejection_types":dict(Counter(r["last_rejection_type"] for r in subset)),
            "last_geometry_violations":dict(Counter(reason for r in subset for reason in r["last_rejection_geometry_violations"])),
            "f_number_excess_ratio":stats([r["f_number_excess_ratio"] for r in subset]),
            "aperture_progress_fraction":stats([r["aperture_progress_fraction"] for r in subset]),
            "field_shortfall_fraction":stats([r["field_shortfall_fraction"] for r in subset]),
        }
    pairwise={}
    paired_rows=[]
    for a,b in combinations((1,2,3),2):
        left,right=by_rule[a],by_rule[b]
        matched=[(x,y) for x,y in zip(left,right) if same_specs(x,y)]
        ml,mr=([x for x,y in matched],[y for x,y in matched])
        values={
            "f_number":comparison(left,right,"f_number",False,TOLERANCES["f_number"]),
            "field_deg":comparison(left,right,"field_deg",True,TOLERANCES["field_deg"]),
            "requested_cycles":comparison(left,right,"requested_cycles",False,0),
            "dls_seconds_observed":comparison(left,right,"dls_seconds",False,0),
            "group_seconds_observed":comparison(left,right,"group_seconds",False,0),
        }
        for metric in [f"mtf_min_{i}" for i in (0,1,2)]+[f"rms_{i}_um" for i in (0,1,2)]:
            values[metric+"_matched_specs"]=comparison(ml,mr,metric,metric.startswith("mtf"),
                                                       TOLERANCES["mtf"] if metric.startswith("mtf") else TOLERANCES["rms_um"])
        relationships=Counter()
        for x,y in zip(left,right):
            tied_f=abs(x["f_number"]-y["f_number"])<=TOLERANCES["f_number"]
            tied_field=abs(x["field_deg"]-y["field_deg"])<=TOLERANCES["field_deg"]
            a_f=x["f_number"]<y["f_number"]-TOLERANCES["f_number"]
            b_f=y["f_number"]<x["f_number"]-TOLERANCES["f_number"]
            a_field=x["field_deg"]>y["field_deg"]+TOLERANCES["field_deg"]
            b_field=y["field_deg"]>x["field_deg"]+TOLERANCES["field_deg"]
            relation=("endpoint_tie" if tied_f and tied_field else
                      "a_dominates" if (a_f or tied_f) and (a_field or tied_field) else
                      "b_dominates" if (b_f or tied_f) and (b_field or tied_field) else "tradeoff")
            relationships[relation]+=1
            paired_rows.append({
                "case_id":x["case_id"],"a":a,"b":b,"same_specs":same_specs(x,y),"endpoint_relation":relation,
                "a_minus_b_f_number":x["f_number"]-y["f_number"],
                "a_minus_b_field_deg":x["field_deg"]-y["field_deg"],
                "a_minus_b_mtf_edge":x["mtf_min_2"]-y["mtf_min_2"],
                "a_minus_b_rms_edge_um":x["rms_2_um"]-y["rms_2_um"],
                "a_minus_b_cycles":x["requested_cycles"]-y["requested_cycles"],
                "dls_time_ratio_a_over_b":x["dls_seconds"]/y["dls_seconds"],
            })
        pairwise[f"{a}_vs_{b}"]={"matched_specs_n":len(matched),"endpoint_relations":dict(relationships),
                                "metrics":values}
    endpoint_sets=Counter()
    for case_id in range(1,101):
        subset=[r for r in rows if r["case_id"]==case_id]
        endpoint_sets[",".join(map(str,numeric_pareto(subset)))]+=1
    budget=np.array([r["requested_cycles"] for r in rows])
    seconds=np.array([r["dls_seconds"] for r in rows])
    by_budget={}
    for value in sorted(set(budget)):
        subset=[r for r in rows if r["requested_cycles"]==value]
        by_budget[str(value)]={"n":len(subset),"dls_seconds":stats([r["dls_seconds"] for r in subset]),
                               "ratio_max_over_min":max(r["dls_seconds"] for r in subset)/min(r["dls_seconds"] for r in subset)}
    equal_budget_pairs=[r for r in paired_rows if r["a_minus_b_cycles"]==0]
    cost={
        "pearson_requested_vs_dls_seconds":float(np.corrcoef(budget,seconds)[0,1]),
        "spearman_requested_vs_dls_seconds":float(np.corrcoef(ranks(budget),ranks(seconds))[0,1]),
        "seconds_per_requested_cycle":stats([r["seconds_per_requested_cycle"] for r in rows]),
        "by_requested_budget":by_budget,
        "same_budget_pair_count":len(equal_budget_pairs),
        "same_budget_time_ratio_a_over_b":stats([r["dls_time_ratio_a_over_b"] for r in equal_budget_pairs]),
        "completed_cycle_counter_missing":sum(r["actual_completed_cycles"] is None for r in rows),
        "completed_attempt_requested_cycles":int(budget.sum()),
        "completed_attempt_dls_seconds":float(seconds.sum()),
        "extra_attempts_with_observed_dls":retry_records,
        "all_observed_requested_cycles":int(budget.sum()+sum(r["requested_cycles"] for r in retry_records)),
        "all_observed_dls_seconds":float(seconds.sum()+sum(r["dls_seconds"] for r in retry_records)),
        "timing_epochs":{epoch:{"n":len(subset:= [r for r in rows if r["completion_epoch"]==epoch]),
                               "seconds_per_requested_cycle":stats([r["seconds_per_requested_cycle"] for r in subset])}
                         for epoch in ("earlier_session","final_session")},
        "measurement_scope":"DLS seconds covers local optimizer calls. Group seconds excludes API startup/queueing. Timing is single-run under changing concurrent load.",
    }
    tie_sets={
        "success_set_distribution":dict(Counter(",".join(map(str,row["successful_rules"])) or "none" for row in labels)),
        "exact_pareto_distribution":dict(Counter(",".join(map(str,row["pareto_endpoint_rules"])) for row in labels)),
        "numerical_pareto_distribution":dict(endpoint_sets),
        "least_cycles_successful_set_distribution":dict(Counter(",".join(map(str,row["least_requested_cycles_successful_rules"])) or "none" for row in labels)),
    }
    provenance={"sources_sha256":original,"analysis_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    if any(sha256(Path(path).read_bytes()).hexdigest()!=value for path,value in original.items()):
        raise RuntimeError("Source file changed")
    summary={"distributions":distributions,"failed_78":failures,"pairwise":pairwise,
             "sets":tie_sets,"cost":cost,"tie_tolerances_from_prior_numeric_regression":TOLERANCES,
             "source_files_verified":len(original)}
    for name,data in (("analysis.json",summary),("group_metrics.json",rows),("paired_metrics.json",paired_rows),
                      ("failed_78_details.json",[r for r in rows if r["all_rules_failed"]]),
                      ("provenance.json",provenance)):
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"failures":failures,"pair_endpoints":{k:v["endpoint_relations"] for k,v in pairwise.items()},
                      "matched_n":{k:v["matched_specs_n"] for k,v in pairwise.items()},
                      "cost_pearson":cost["pearson_requested_vs_dls_seconds"],
                      "cost_spearman":cost["spearman_requested_vs_dls_seconds"],
                      "sets":tie_sets},indent=2))


if __name__=="__main__":
    main()
