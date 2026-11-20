"""Read-only audit of sampled and restored prescriptions; write a new registry."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

from prescription_registry import (
    Registry, fingerprint, full_parameters, geometry_parameters, geometry_vector,
    precision_overlap, relative_distance, rounding_interval, sample_fingerprint,
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def duplicate_groups(rows, identity):
    groups = defaultdict(list)
    for row in rows:
        groups[identity(row)].append(row["id"])
    return [ids for ids in groups.values() if len(ids)>1]


def seed_parameters(path):
    text = path.read_text(encoding="utf-16")
    parts = re.split(r"^SURF \d+\s*$",text,flags=re.M)[1:]
    if len(parts)!=7 or not re.search(r"^  STOP\s*$",parts[1],flags=re.M):
        raise ValueError("Unexpected native surface layout")
    def number(part,key):
        return float(re.search(rf"^  {key} (\S+)",part,re.M)[1])
    curvature = [number(part,"CURV") for part in parts]
    distances = [number(part,"DISZ") for part in parts]
    material = []
    for index in (2,4):
        tokens = re.search(r"^  GLAS (.+)$",parts[index],re.M)[1].split()
        material.append([float(tokens[j]) for j in (3,4,5)])
    return {
        "curvatures_per_mm":curvature[2:6],
        "thicknesses_mm":[distances[2],distances[4]],
        "air_gap_mm":distances[3],
        "object_to_first_vertex_mm":distances[0]+distances[1],
        "stop_to_first_vertex_mm":distances[1],
        "last_vertex_to_image_mm":distances[5],
        "material_nd_vd_dpgf":material,
        "semi_apertures_mm":[number(parts[i],"DIAM") for i in range(2,6)],
        "working_f_number":float(re.search(r"^FNUM (\S+)",text,re.M)[1]),
        "fields_deg":[float(v) for v in re.search(r"^YFLN (.+)$",text,re.M)[1].split()],
        "primary_wavelength_mm":float(re.search(r"^WAVM 1 (\S+)",text,re.M)[1])*0.001,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--registry",type=Path,default=Path("prescription_registry/global.sqlite"))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    args.registry.parent.mkdir(parents=True,exist_ok=True)
    sources=[root/name/"candidates.json" for name in (
        "random_candidates_10","random_candidates_100","random_candidates_200")]
    first_plan_path=root/"background_runs/batch_300_20261118/plan.json"
    first_state_path=first_plan_path.with_name("state.json")
    next_plan_path=root/"background_runs/batch_300_20261118_02/plan.json"
    first_plan, state = load(first_plan_path),load(first_state_path)
    if state["status"]!="completed" or state["completed"]!=300:
        raise RuntimeError("Expected completed first batch")
    first=sorted(load(sources[1])["candidates"],key=lambda row:row["id"])
    if len(first)!=100 or not all(row["passed"] for row in first):
        raise RuntimeError("Expected 100 validated sampled candidates")
    by_id={row["id"]:row for row in first}
    checked={str(path):digest(path) for path in [*sources,first_plan_path,first_state_path,next_plan_path]}
    registry=Registry(args.registry)
    try:
        with registry.connection:
            for source in sources:
                data=load(source)
                for collection in ("candidates","rejected"):
                    for index,row in enumerate(data.get(collection,[])):
                        registry.register(row,str(source),checked[str(source)],f"{collection}:{index}")
            for plan_path in (first_plan_path,next_plan_path):
                plan=load(plan_path)
                for candidate in plan["cases"]:
                    fp,_=sample_fingerprint(candidate)
                    if not registry.lookup(candidate)["registered"]:
                        raise RuntimeError("Planned case absent from source registry")
                    existing=registry.connection.execute(
                        "SELECT sample_sha256 FROM batch_cases WHERE batch_path=? AND case_id=?",
                        (str(plan_path.parent),candidate["id"])).fetchone()
                    if existing and existing[0]!=fp:
                        raise RuntimeError("Conflicting global batch case identity")
                    registry.connection.execute("INSERT OR IGNORE INTO batch_cases VALUES (?,?,?)",
                                                (str(plan_path.parent),candidate["id"],fp))
            identity_issues=[]
            rules_by_case=defaultdict(list)
            native_by_case=defaultdict(list)
            native_records=[]
            plan_cases={row["id"]:row for row in first_plan["cases"]}
            for case in first:
                if case["inputs"]!=plan_cases[case["id"]]["inputs"]:
                    identity_issues.append({"case_id":case["id"],"reason":"plan_inputs_differ"})
            for key,job in state["jobs"].items():
                case_id,rule=job["case_id"],job["rule"]
                case=by_id.get(case_id)
                if case is None:
                    identity_issues.append({"job":key,"reason":"unexpected_case_id"})
                    continue
                rules_by_case[case_id].append(rule)
                directory=first_plan_path.parent/job["attempt"]
                result_path=directory/"experiment/result.json"
                seed_path=directory/"seed.zmx"
                result=load(result_path)
                checked[str(result_path)]=digest(result_path)
                checked[str(seed_path)]=digest(seed_path)
                problems=[]
                if key!=f"case_{case_id:03d}_rule_{rule}":
                    problems.append("job_key_mismatch")
                if (result["case_id"],result["rule"])!=(case_id,rule):
                    problems.append("result_identity_mismatch")
                if result["inputs"]!=case["inputs"]:
                    problems.append("result_inputs_mismatch")
                if result["seed_sha256"]!=checked[str(seed_path)]:
                    problems.append("seed_hash_mismatch")
                if job["acceptance"]["result_sha256"]!=checked[str(result_path)]:
                    problems.append("result_hash_mismatch")
                final=result["final_accepted"]
                if final:
                    final_path=directory/"experiment/final_accepted.zmx"
                    checked[str(final_path)]=digest(final_path)
                    if checked[str(final_path)]!=job["acceptance"]["lens_sha256"]:
                        problems.append("final_lens_hash_mismatch")
                native=seed_parameters(seed_path)
                native_vector = [1/value for value in native["curvatures_per_mm"]]
                native_vector += native["thicknesses_mm"]+[native["air_gap_mm"]]
                ranked_sources = sorted(
                    (relative_distance(native_vector,geometry_vector(source)),source["id"])
                    for source in first
                )
                if ranked_sources[0][1] != case_id or ranked_sources[0][0] == ranked_sources[1][0]:
                    identity_issues.append({"case_id":case_id,"rule":rule,
                                            "reason":"native_geometry_not_uniquely_closest_to_source_case"})
                native_hash,_=fingerprint(native,"opticstudio-initial-seed-v1")
                native_by_case[case_id].append(native_hash)
                native_records.append({"case_id":case_id,"rule":rule,
                                       "seed_sha256":native_hash,"parameters":native,
                                       "closest_source_case_id":ranked_sources[0][1],
                                       "source_native_relative_distance":ranked_sources[0][0]})
                for problem in problems:
                    identity_issues.append({"case_id":case_id,"rule":rule,"reason":problem})
                sample_hash,_=sample_fingerprint(case)
                registry.connection.execute("INSERT OR IGNORE INTO execution_refs VALUES (?,?,?,?,?,?)",(
                    str(first_plan_path.parent),case_id,rule,str(result_path),checked[str(result_path)],sample_hash))
            for case_id in by_id:
                if sorted(rules_by_case[case_id])!=[1,2,3]:
                    identity_issues.append({"case_id":case_id,"reason":"expected_rules_1_2_3"})
                if len(set(native_by_case[case_id]))!=1:
                    identity_issues.append({"case_id":case_id,"reason":"native_seed_differs_between_rules"})
            hashes=[]
            for row in first:
                sample,_=sample_fingerprint(row)
                geometry,_=fingerprint(geometry_parameters(row),"two-lens-geometry-v1")
                full,_=fingerprint(full_parameters(row),"two-lens-full-v1")
                hashes.append({"case_id":row["id"],"sampling_sha256":sample,
                               "geometry_sha256":geometry,"full_initial_sha256":full,
                               "normalized_parameters":full_parameters(row)})
            sample_groups=duplicate_groups(first,lambda row:sample_fingerprint(row)[0])
            geometry_groups=duplicate_groups(first,lambda row:fingerprint(
                geometry_parameters(row),"two-lens-geometry-v1")[0])
            full_groups=duplicate_groups(first,lambda row:fingerprint(full_parameters(row),"two-lens-full-v1")[0])
            input_intervals={r["id"]:[rounding_interval(v) for v in r["inputs"]] for r in first}
            geometry_intervals={r["id"]:[rounding_interval(v) for v in geometry_vector(r)] for r in first}
            pairs=[]
            for a,b in combinations(first,2):
                pairs.append({
                    "case_id_a":a["id"],"case_id_b":b["id"],
                    "sampling_storage_precision_overlap":precision_overlap(input_intervals[a["id"]],input_intervals[b["id"]]),
                    "geometry_storage_precision_overlap":precision_overlap(geometry_intervals[a["id"]],geometry_intervals[b["id"]]),
                    "structure_relative_linf_distance":relative_distance(geometry_vector(a),geometry_vector(b)),
                    "absolute_geometry_differences_mm":[abs(x-y) for x,y in zip(geometry_vector(a),geometry_vector(b))],
                })
            precision_pairs=[p for p in pairs if p["sampling_storage_precision_overlap"]]
            structure_precision_pairs=[p for p in pairs if p["geometry_storage_precision_overlap"]]
            closest=sorted(pairs,key=lambda p:p["structure_relative_linf_distance"])[:10]
            counts={name:registry.connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                    for name in ("prescriptions","geometry_variants","occurrences","batch_cases","execution_refs")}
            statuses=dict(registry.connection.execute(
                "SELECT validation_status,count(*) FROM prescriptions GROUP BY validation_status"))
            repeats=registry.connection.execute("""
                SELECT sample_sha256,count(*) FROM occurrences
                GROUP BY sample_sha256 HAVING count(*)>1
            """).fetchall()
            cross_duplicates=registry.connection.execute("""
                SELECT sample_sha256,count(DISTINCT batch_path) FROM batch_cases
                GROUP BY sample_sha256 HAVING count(DISTINCT batch_path)>1
            """).fetchall()
            next_cases=load(next_plan_path)["cases"]
            next_hashes={sample_fingerprint(row)[0] for row in next_cases}
            current_hashes={sample_fingerprint(row)[0] for row in first}
            summary={
                "audited_completed_cases":100,"audited_rule_results":len(state["jobs"]),
                "exact_sampling_duplicate_groups":sample_groups,
                "exact_full_prescription_duplicate_groups":full_groups,
                "exact_geometry_duplicate_groups":geometry_groups,
                "sampling_precision_overlap_pairs":precision_pairs,
                "geometry_precision_overlap_pairs":structure_precision_pairs,
                "identity_issues":identity_issues,
                "independent_exact_sampling_cases":len(current_hashes),
                "usable_independent_cases":len({sample_fingerprint(row)[0] for row in first
                    if not any(issue.get("case_id")==row["id"] for issue in identity_issues)}),
                "next_batch_overlap_with_first":len(next_hashes&current_hashes),
                "registry_counts":counts,"registry_validation_counts":statuses,
                "repeated_snapshot_fingerprint_count":len(repeats),
                "repeated_snapshot_excess_records":sum(count-1 for _,count in repeats),
                "cross_batch_duplicate_fingerprints":cross_duplicates,
                "engineering_similarity_tolerance":None,
                "geometry_vector_order":["r1_mm","r2_mm","r3_mm","r4_mm","t1_mm","t2_mm","air_gap_mm"],
                "storage_precision_rule":"closed exact-rational midpoint intervals between adjacent binary64 values; all coordinates must overlap",
                "precision_scope":"binary64 output representation; original sampler has no fixed decimal quantization",
                "similarity_ranking_rule":"max_i(abs(a_i-b_i)/max(abs(a_i),abs(b_i))); zero/zero gives zero",
            }
        changed=[name for name,value in checked.items() if digest(Path(name))!=value]
        if changed:
            raise RuntimeError(f"Audit source changed: {changed}")
        summary["source_files_verified_unchanged"]=len(checked)
        (output/"report.json").write_text(json.dumps(summary,indent=2)+"\n")
        (output/"fingerprints.json").write_text(json.dumps(hashes,indent=2)+"\n")
        (output/"native_seed_checks.json").write_text(json.dumps(native_records,indent=2)+"\n")
        (output/"suspected_duplicates.json").write_text(json.dumps({
            "storage_precision_candidates":precision_pairs,
            "geometry_storage_precision_candidates":structure_precision_pairs,
            "nearest_structure_pairs_for_review":closest,
            "review_scope":"Nearest pairs are ranked observations; no engineering similarity cutoff was supplied.",
        },indent=2)+"\n")
        with (output/"all_structure_pairs.jsonl").open("w") as stream:
            for row in pairs:
                stream.write(json.dumps(row)+"\n")
        (output/"provenance.json").write_text(json.dumps({
            "source_sha256":checked,"registry":str(args.registry.resolve()),
            "registry_schema":"prescription_registry.py",
        },indent=2)+"\n")
        lines=[
            "# 初始处方去重审计","",
            f'第一批 100 套处方、300 组规则结果已检查。可用独立处方为 {summary["usable_independent_cases"]} 套。',
            "",
            "| 检查 | 发现数量 |","|---|---:|",
            f'| 原始参数完全重复组 | {len(sample_groups)} |',
            f'| 完整初始处方完全重复组 | {len(full_groups)} |',
            f'| 半径、厚度、空气间隔结构完全重复组 | {len(geometry_groups)} |',
            f'| 原始参数存储精度区间重叠对 | {len(precision_pairs)} |',
            f'| 实体结构存储精度区间重叠对 | {len(structure_precision_pairs)} |',
            f'| ID / Rule / 处方对应问题 | {len(identity_issues)} |',
            f'| 后一批与第一批重复处方 | {len(next_hashes&current_hashes)} |',
            "",
            "长度统一为 mm，放大率和折射率为无量纲，波长由 µm 换算为 mm。"
            "原始数据来自 Python binary64，没有固定小数位量化。规范指纹采用 float.hex()、排序字段和版本化 JSON，"
            "保留全部二进制数值位，正负零统一为零，半径及共轭距离符号保留。",
            "",
            "原始指纹包含五个采样输入，以及折射率、材料模型、厚度、半口径、波长、初始孔径视场和参考面约定。"
            "另分别计算包含实体化参数的完整处方指纹和只比较实体结构的几何指纹。",
            "",
            "近似检查的区间由相邻 binary64 数的中点构造，使用 Fraction 精确计算端点，包含共享边界的情况。"
            "这只检验存储分辨率范围。随机源的 53 位随机数、对数映射及 exp 运算不构成固定小数网格，"
            "现有记录不足以给出工程容差意义的近似阈值。",
            "",
            "工程意义的高度相似结构尚无指定容差。以下为全部 4950 个结构对中的最近 10 对，"
            "用于人工审阅；完整距离和各尺寸差保存在 all_structure_pairs.jsonl。",
            "",
            "| Case A | Case B | 结构最大相对距离 |","|---|---|---:|",
        ]
        for pair in closest:
            lines.append(f'| {pair["case_id_a"]} | {pair["case_id_b"]} | {pair["structure_relative_linf_distance"]:.8f} |')
        lines += ["","## 全局登记","",
                  f'已登记 {statuses.get("valid",0)} 套有效随机处方和 {statuses.get("rejected",0)} 个剔除采样，'
                  f'共 {counts["occurrences"]} 条来源引用。',
                  "random_candidates_10、100、200 是同一随机序列的累积快照，前缀重复引用保存在 occurrences。"
                  "独立处方数量按参数指纹统计。两个正式批次计划各 100 套均已登记，第一批的 300 组结果已逐一校验。",
                  "",
                  "每个 ID 都检查 Rule 1、2、3 是否齐全、结果中的 ID/输入是否与源记录一致、"
                  "种子及结果哈希是否与验收记录一致、三条规则的实际种子光学参数是否一致。",
                  "",
                  "登记表位于 prescription_registry/global.sqlite。后续采样可调用 Registry.lookup(candidate)，"
                  "返回是否已登记及原验证状态。现有生成器和优化结果未调整，后续接入采样器时可在入队前调用。",
                  "",
                  f'{len(checked)} 个只读源文件哈希前后匹配。第二批运行状态和结果未修改。',""]
        (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
        print(json.dumps(summary,indent=2))
    finally:
        registry.close()


if __name__=="__main__":
    main()
