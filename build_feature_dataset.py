"""Build initial-only features and separate rule outcomes for the first 100 cases."""

import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
from pathlib import Path

import numpy as np

from prescription_registry import INPUT_NAMES, sample_fingerprint


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def ratio(a,b):
    return a/b if b else None


def pareto_rules(outcomes):
    """Recorded endpoint dominance: smaller F/# and greater field are preferred."""
    endpoints = [row for row in outcomes if row["has_accepted_endpoint"]]
    frontier = []
    for row in endpoints:
        dominated = any(
            other["final_working_f_number"] <= row["final_working_f_number"]
            and other["final_max_field_deg"] >= row["final_max_field_deg"]
            and (other["final_working_f_number"] < row["final_working_f_number"]
                 or other["final_max_field_deg"] > row["final_max_field_deg"])
            for other in endpoints
        )
        if not dominated:
            frontier.append(row["rule"])
    return frontier


def build_initial_features(candidate, schema):
    def add(row,name,value,unit,formula,dependencies,group):
        if value is not None and not isfinite(value):
            raise ValueError(f"Nonfinite feature {name}")
        row[name] = value
        definition = {"unit":unit,"definition":formula,"dependencies":dependencies,
                      "group":group,"available_before_rule_selection":True}
        if name in schema and schema[name]!=definition:
            raise RuntimeError("Inconsistent feature schema")
        schema[name] = definition

    row = {}
    for index,(name,value) in enumerate(zip(INPUT_NAMES,candidate["inputs"])):
        add(row,name,value,"dimensionless" if name=="magnification" else "mm",
            "original sampled signed parameter",[f"candidate.inputs[{index}]"],"sampled")
    thin = candidate["theory"]["solution"]
    for name,key,unit in (
        ("thin_spacing_mm","d12","mm"),("thin_power_1_per_mm","k1","1/mm"),
        ("thin_power_2_per_mm","k2","1/mm"),("thin_efl_mm","efl","mm"),
    ):
        add(row,name,thin[key],unit,f"candidate.theory.solution.{key}",list(INPUT_NAMES),"paper_derived")
    entity = candidate["entity"]
    for index,value in enumerate(entity["radii_mm"],1):
        add(row,f"initial_r{index}_mm",value,"mm",f"entity.radii_mm[{index-1}]",
            list(INPUT_NAMES),"entity")
    for index,value in enumerate(entity["center_thicknesses_mm"],1):
        add(row,f"initial_t{index}_mm",value,"mm",f"entity.center_thicknesses_mm[{index-1}]",
            ["fixed construction"],"construction_constant")
    for name in ("air_gap_mm","object_to_first_vertex_mm","last_vertex_to_image_mm",
                 "stop_to_first_vertex_mm","last_vertex_to_exit_pupil_mm"):
        add(row,f"initial_{name}",entity[name],"mm",f"entity.{name}",list(INPUT_NAMES),"entity")
    add(row,"initial_efl_mm",entity["verification"]["efl_mm"],"mm",
        "independent restored thick-lens EFL",list(INPUT_NAMES),"entity")
    add(row,"index_nd",entity["index"],"dimensionless","fixed model glass index",
        ["fixed construction"],"construction_constant")
    for index,value in enumerate(entity["semi_apertures_mm"],1):
        add(row,f"initial_semi_aperture_{index}_mm",value,"mm",
            f"entity.semi_apertures_mm[{index-1}]",["fixed construction"],"construction_constant")
    dimensions = (
        ("thin_spacing_over_abs_efl",thin["d12"],abs(thin["efl"]),
         "thin_spacing_mm / abs(thin_efl_mm)",["thin_spacing_mm","thin_efl_mm"]),
        ("thin_power_ratio",thin["k1"],thin["k2"],
         "thin_power_1_per_mm / thin_power_2_per_mm",["thin_power_1_per_mm","thin_power_2_per_mm"]),
        ("thin_power_balance",thin["k1"]-thin["k2"],abs(thin["k1"])+abs(thin["k2"]),
         "(K1-K2)/(abs(K1)+abs(K2))",["thin_power_1_per_mm","thin_power_2_per_mm"]),
        ("initial_gap_over_abs_efl",entity["air_gap_mm"],abs(entity["verification"]["efl_mm"]),
         "initial_air_gap_mm / abs(initial_efl_mm)",["initial_air_gap_mm","initial_efl_mm"]),
        ("object_over_abs_efl",abs(candidate["inputs"][0]),abs(entity["verification"]["efl_mm"]),
         "abs(object_l1_mm)/abs(initial_efl_mm)",["object_l1_mm","initial_efl_mm"]),
        ("image_over_abs_efl",abs(candidate["inputs"][1]),abs(entity["verification"]["efl_mm"]),
         "abs(image_l2_mm)/abs(initial_efl_mm)",["image_l2_mm","initial_efl_mm"]),
    )
    for name,numerator,denominator,formula,dependencies in dimensions:
        add(row,name,ratio(numerator,denominator),"dimensionless",formula,dependencies,"dimensionless")
    for index in (1,2):
        a,b=entity["radii_mm"][(index-1)*2:index*2]
        c1,c2=1/a,1/b
        add(row,f"initial_bending_factor_{index}",ratio(c1+c2,c1-c2),"dimensionless",
            "(c_front+c_back)/(c_front-c_back)",
            [f"initial_r{2*index-1}_mm",f"initial_r{2*index}_mm"],"dimensionless")
        value=candidate["restored_geometry"]["edge_thickness_mm"][index-1]-0.001
        add(row,f"initial_edge_margin_{index}_mm",value,"mm",
            f"restored_geometry.edge_thickness_mm[{index-1}] - 0.001 mm",
            [f"initial_r{2*index-1}_mm",f"initial_r{2*index}_mm",f"initial_t{index}_mm"],"initial_geometry")
    add(row,"initial_air_gap_margin_mm",candidate["restored_geometry"]["minimum_air_gap_mm"]-0.001,"mm",
        "restored_geometry.minimum_air_gap_mm - 0.001 mm",
        ["initial_air_gap_mm","initial_r2_mm","initial_r3_mm"],"initial_geometry")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    source = root/"random_candidates_100/candidates.json"
    state_path = root/"background_runs/batch_300_20261118/state.json"
    dedup = root/"dedup_audit_20261118_v2/report.json"
    candidates = sorted(load(source)["candidates"],key=lambda row:row["id"])
    state = load(state_path)
    audit = load(dedup)
    if state["status"]!="completed" or state["completed"]!=300 or audit["usable_independent_cases"]!=100:
        raise RuntimeError("Expected 100 validated independent cases and 300 completed groups")
    provenance = {str(path):digest(path) for path in (source,state_path,dedup)}
    features, outcomes, labels, schema = [], [], [], {}
    for candidate in candidates:
        case_id = candidate["id"]
        fingerprint,_ = sample_fingerprint(candidate)
        row = build_initial_features(candidate, schema)
        features.append({"case_id":case_id,"prescription_sha256":fingerprint,"features":row})
        case_outcomes = []
        for rule in (1,2,3):
            info = state["jobs"][f"case_{case_id:03d}_rule_{rule}"]
            path = state_path.parent/info["attempt"]/"experiment/result.json"
            result = load(path)
            provenance[str(path)] = digest(path)
            if provenance[str(path)]!=info["acceptance"]["result_sha256"]:
                raise RuntimeError("Optimization result hash changed")
            if result["inputs"]!=candidate["inputs"] or (result["case_id"],result["rule"])!=(case_id,rule):
                raise RuntimeError("Feature-target identity mismatch")
            final = result["final_accepted"]
            target = {
                "case_id":case_id,"prescription_sha256":fingerprint,"rule":rule,
                "has_accepted_endpoint":final is not None,"target_reached":result["target_reached"],
                "final_working_f_number":final["working_f_number"] if final else None,
                "final_max_field_deg":final["max_field_deg"] if final else None,
                "fields":final["fields"] if final else None,
                "spot_radius_units":final["spot_radius_units"] if final else None,
                "final_conjugate_residuals":{
                    key:final[key] for key in (
                        "object_image_b_mm","magnification_error_at_image","exit_pupil_b_mm")
                } if final else None,
                "final_geometry_clearances_mm":final["clearances_mm"] if final else None,
                "cost":result["cost"],"stop_reasons":result["stop_reasons"],
            }
            if final:
                expected_success=(final["working_f_number"]==20 and final["max_field_deg"]==5
                                  and final["geometry_valid"] and all(f["meets_threshold"] for f in final["fields"]))
                if result["target_reached"]!=expected_success:
                    raise RuntimeError("Inconsistent target success label")
            outcomes.append(target)
            case_outcomes.append(target)
        successful = [r for r in case_outcomes if r["target_reached"]]
        minimum_cycles = min((r["cost"]["requested_dls_cycles"] for r in successful),default=None)
        labels.append({
            "case_id":case_id,"prescription_sha256":fingerprint,
            "target_success_mask":sum(1 << (r["rule"]-1) for r in successful),
            "successful_rules":[r["rule"] for r in successful],
            "pareto_endpoint_rules":pareto_rules(case_outcomes),
            "least_requested_cycles_successful_rules":[r["rule"] for r in successful
                if r["cost"]["requested_dls_cycles"]==minimum_cycles],
        })
    columns = list(schema)
    missing = {name:sum(row["features"][name] is None for row in features) for name in columns}
    constant = [name for name in columns if len({r["features"][name] for r in features})==1]
    identical, opposite, correlations = [], [], []
    arrays={name:np.array([row["features"][name] for row in features],dtype=float)
            for name in columns if not missing[name] and name not in constant}
    for a,b in combinations(arrays,2):
        if np.array_equal(arrays[a],arrays[b]):
            identical.append([a,b])
        if np.array_equal(arrays[a],-arrays[b]):
            opposite.append([a,b])
        correlations.append({"feature_a":a,"feature_b":b,"pearson_r":float(np.corrcoef(arrays[a],arrays[b])[0,1])})
    outliers=[]
    for name,values in arrays.items():
        q1,q3 = np.quantile(values,[0.25,0.75])
        low,high=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        for record,value in zip(features,values):
            if value<low or value>high:
                outliers.append({"case_id":record["case_id"],"feature":name,"value":float(value),
                                 "lower_fence":float(low),"upper_fence":float(high)})
    statistics = {}
    for name in columns:
        values=np.array([r["features"][name] for r in features if r["features"][name] is not None])
        statistics[name]={"missing":missing[name],"unique_values":len(set(values)),
                          "min":float(min(values)),"median":float(np.median(values)),
                          "max":float(max(values))} if len(values) else {"missing":100}
    for name in schema:
        schema[name]["constant_in_this_batch"]=name in constant
        schema[name]["suggested_use"]="retain as construction metadata" if name in constant else "initial feature candidate"
    paper9=list(INPUT_NAMES)+["thin_spacing_mm","thin_power_1_per_mm","thin_power_2_per_mm","thin_efl_mm"]
    quality={
        "case_count":len(features),"outcome_count":len(outcomes),"feature_count":len(columns),
        "missing_features":{k:v for k,v in missing.items() if v},
        "constant_features":constant,"exact_equal_feature_pairs":identical,
        "exact_opposite_feature_pairs":opposite,
        "largest_absolute_pearson_correlations":sorted(correlations,key=lambda row:abs(row["pearson_r"]),reverse=True)[:20],
        "feature_statistics":statistics,
        "iqr_outlier_flags":outliers,
        "iqr_policy":"1.5*IQR exploratory flags, all observations retained",
        "success_mask_distribution":dict(sorted(Counter(row["target_success_mask"] for row in labels).items())),
        "success_count_by_rule":{str(rule):sum(row["rule"]==rule and row["target_reached"] for row in outcomes) for rule in (1,2,3)},
        "endpoint_missing_by_rule":{str(rule):sum(row["rule"]==rule and not row["has_accepted_endpoint"] for row in outcomes) for rule in (1,2,3)},
        "pareto_set_distribution":dict(Counter(",".join(map(str,row["pareto_endpoint_rules"])) or "none" for row in labels)),
        "successful_cost_tie_distribution":dict(Counter(len(row["least_requested_cycles_successful_rules"]) for row in labels)),
        "all_three_reach_target":sum(row["target_success_mask"]==7 for row in labels),
        "no_rule_reaches_target":sum(row["target_success_mask"]==0 for row in labels),
        "unique_least_cycles_successful_cases":sum(len(row["least_requested_cycles_successful_rules"])==1 for row in labels),
        "tied_least_cycles_successful_cases":sum(len(row["least_requested_cycles_successful_rules"])>1 for row in labels),
        "duplicate_prescription_fingerprints":len(features)-len({row["prescription_sha256"] for row in features}),
    }
    dictionary={
        "metadata_columns":["case_id","prescription_sha256"],"features":schema,
        "feature_sets":{"raw_sampled_5":list(INPUT_NAMES),"paper_9":paper9,
                        "all_initial_nonconstant":[name for name in columns if name not in constant]},
        "outcomes_available_after_optimization":True,
        "outcome_units":{"fields.rms_spot_radius":"um","fields.mtf_tangential":"dimensionless",
                         "fields.mtf_sagittal":"dimensionless","fields.mtf_frequency_lp_per_mm":"lp/mm",
                         "final_geometry_clearances_mm":"mm","cost.dls_seconds":"s","cost.wall_seconds":"s"},
        "label_definitions":{
            "target_success_mask":"bit 0=Rule1, bit 1=Rule2, bit 2=Rule3 reaches F/20 and 5 degrees with accepted MTF/geometry",
            "pareto_endpoint_rules":"recorded endpoint nondominance over minimizing F/# and maximizing field; exact recorded values, ties retained",
            "least_requested_cycles_successful_rules":"minimum requested DLS cycles among successful rules; ties retained; empty if none succeeds",
        },
        "preprocessing":"raw values retained; no train/test split, scaling, imputation or supervised feature selection",
        "generation_degrees_of_freedom":"five random inputs; other feature columns are deterministic constructions",
    }
    changed=[name for name,value in provenance.items() if digest(Path(name))!=value]
    if changed:
        raise RuntimeError("Input files changed during feature extraction")
    for name,value in (("features.json",features),("outcomes.json",outcomes),("labels.json",labels),
                       ("dictionary.json",dictionary),("quality.json",quality)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"provenance.json").write_text(json.dumps({
        "source_sha256":provenance,"builder_sha256":digest(Path(__file__)),
        "output_schema":"initial-features-v1","source_files_unchanged":len(provenance),
    },indent=2)+"\n")
    lines=["# 前 100 套处方特征工程","",
           f'形成 {len(features)} 行初始特征、{len(outcomes)} 行规则结果和 {len(labels)} 行集合标签。'
           f'共 {len(columns)} 个特征字段，其中 {len(constant)} 个在本批恒定。', "",
           "| 规则 | 达到共同目标的处方数 | 无接受终点 |","|---|---:|---:|"]
    for rule in (1,2,3):
        lines.append(f'| Rule {rule} | {quality["success_count_by_rule"][str(rule)]} | {quality["endpoint_missing_by_rule"][str(rule)]} |')
    lines+=["","共同目标为 F/# 20、最大视场 5°，并通过原有几何及 MTF 验收。",
            "",f'三条规则均达标的处方 {quality["all_three_reach_target"]} 套，全部未达共同目标的处方 {quality["no_rule_reaches_target"]} 套。',
            f'成功规则中按请求循环数最少计，唯一最优 {quality["unique_least_cycles_successful_cases"]} 套，并列最优 {quality["tied_least_cycles_successful_cases"]} 套。',
            "","## 数据检查","",
            f'- 重复处方指纹 {quality["duplicate_prescription_fingerprints"]}，缺失初始特征字段 {len(quality["missing_features"])}。',
            f'- 常量字段：{", ".join(constant)}。',
            f'- 精确相反特征对：{opposite}。',
            f'- IQR 异常标记 {len(outliers)} 条，涉及 {len(set(row["case_id"] for row in outliers))} 套处方，全部保留。',
            "- 五个采样输入是生成过程的自由参数，其余初始特征由它们构造。额外列改善表达方式，独立处方数仍为 100。",
            "- 相关系数按绝对值排序展示前 20 对，不据此自动删列。",
            "","## 标签分布","",
            "| 成功集合掩码 | 处方数 |","|---|---:|"]
    for mask,count in quality["success_mask_distribution"].items():
        lines.append(f"| {mask} | {count} |")
    lines+=["","掩码 bit0、bit1、bit2 分别对应 Rule1、2、3。0 表示三条均未达共同目标，7 表示三条均达标。",
            "","## 使用边界","",
            "features.json 只含规则选择前可用数据。outcomes.json 单独保存优化终点、MTF、RMS、耗时、最终余量与残差。"
            "labels.json 保留成功集合、孔径/视场 Pareto 集合和成功规则中请求循环最少的集合。",
            "",
            "不同终点的孔径、视场及 MTF 评估频率不同，目标表保留各自频率。未以跨规格 MTF 高低强行选唯一胜者。"
            "共轭和放大率沿用软约束，终点残差完整保留。",
            "",
            "本轮未拟合归一化、插补或特征筛选参数，未划分训练集和测试集，未训练模型。"
            "后续可先用 raw_sampled_5 和 paper_9 两套特征做基线对照；预处理参数只在训练折拟合。",
            "",
            f'{len(provenance)} 个源文件在抽取前后哈希一致。',""]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:quality[k] for k in (
        "case_count","outcome_count","feature_count","constant_features","missing_features",
        "success_count_by_rule","success_mask_distribution","all_three_reach_target","no_rule_reaches_target"
    )},indent=2))


if __name__=="__main__":
    main()
