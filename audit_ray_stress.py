"""Compare the Ray stress run with the nine P1 serial reference results."""

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    report=json.loads((args.output/"stress_report.json").read_text())
    baseline=json.loads((root/"p1_random_3_v1/batch.json").read_text())
    comparisons=[]
    for reference in baseline["groups"]:
        case,rule=reference["case_id"],reference["rule"]
        directory=args.output/f"case_{case:03d}_rule_{rule}"
        path=directory/"experiment/result.json"
        comparison={"case_id":case,"rule":rule,"passed":False}
        if not path.exists():
            task=next((task for task in report["task_results"]
                       if Path(task["job"]).parent.name==directory.name), {})
            comparison["failure"]=(
                "not_started_admission_stopped" if task.get("skipped") else
                "terminated_by_low_memory_monitor" if any(
                    "128_MiB" in event["reason"] for event in report["stop_events"]) else
                "no_result"
            )
            comparison["endpoint_mtf_conjugate_comparison"]="unavailable_no_completed_optimization"
            comparisons.append(comparison)
            continue
        actual=json.loads(path.read_text())
        events=[json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
        complete=any(event["phase"]=="completed" for event in events)
        mismatches=[]
        a,b=actual["final_accepted"],reference["final_accepted"]
        comparison.update({"complete":complete,"reference_wall_seconds":reference["cost"]["wall_seconds"],
                           "parallel_wall_seconds":actual["cost"]["wall_seconds"],
                           "reference_steps":len(reference["steps"]),"parallel_steps":len(actual["steps"])})
        if a is None or b is None:
            mismatches.append("missing_accepted_endpoint")
        else:
            checks=[]
            for key,tolerance in (
                ("working_f_number",1e-9),("max_field_deg",1e-9),
                ("object_image_b_mm",1e-6),("magnification_error_at_image",1e-6),
                ("exit_pupil_b_mm",1e-6)):
                checks.append((key,a[key],b[key],tolerance))
            for i,(af,bf) in enumerate(zip(a["fields"],b["fields"])):
                for key,tolerance in (("mtf_tangential",1e-6),("mtf_sagittal",1e-6),
                                      ("rms_spot_radius",1e-3)):
                    checks.append((f"field_{i+1}_{key}",af[key],bf[key],tolerance))
            differences={name:{"parallel":av,"serial":bv,"absolute_difference":abs(av-bv),
                               "tolerance":tolerance} for name,av,bv,tolerance in checks}
            mismatches.extend(name for name,value in differences.items()
                              if value["absolute_difference"]>value["tolerance"])
            comparison["metric_comparison"]=differences
            comparison["geometry_valid"]=a["geometry_valid"]
            if not a["geometry_valid"]:
                mismatches.append("invalid_geometry")
        actual_steps=[(s["stage"],s["working_f_number"],s["max_field_deg"],s["accepted"])
                      for s in actual["steps"]]
        expected_steps=[(s["stage"],s["working_f_number"],s["max_field_deg"],s["accepted"])
                        for s in reference["steps"]]
        if actual_steps!=expected_steps:
            mismatches.append("step_sequence_changed")
        if actual["stop_reasons"]!=reference["stop_reasons"]:
            mismatches.append("stop_reasons_changed")
        comparison["mismatches"]=mismatches
        comparison["passed"]=complete and not mismatches
        comparisons.append(comparison)
    accepted=(report["released_eight"] and report["peak_active_sessions"]==8
              and report["peak_running_groups"]==8
              and len(report["task_results"])==9
              and all(c["passed"] for c in comparisons)
              and not report["stop_events"] and not report["historical_files_changed"])
    result={"accepted_for_batch_generation":accepted,
            "comparisons":comparisons,
            "observations":{key:report[key] for key in (
                "peak_active_sessions","peak_running_groups","peak_live_subprocesses",
                "minimum_available_memory_gib","mean_system_cpu_percent","peak_system_cpu_percent",
                "wall_seconds_including_ray_setup","workload_wall_seconds","stop_events",
        "historical_files_changed")}}
    samples=[json.loads(line) for line in (args.output/"telemetry.jsonl").read_text().splitlines()]
    at_eight=[sample for sample in samples if sample["running_groups"]==8]
    result["observations"]["samples_with_eight_running"]=len(at_eight)
    result["observations"]["mean_system_cpu_percent_at_eight_running"]=(
        sum(s["system_cpu_percent"] for s in at_eight)/len(at_eight) if at_eight else None)
    (args.output/"acceptance.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    lines=["# Ray 8 并行压力测试验收","",
           f"验收通过并可用于后续批量生成：{accepted}。","",
           "本轮实际启动 8 个任务并达到 8 会话并发峰值。可用物理内存跌破事先声明的 128 MiB 紧急阈值，"
           "监控主动终止这 8 组并停止准入，第 9 组跳过。未观测到完整优化终点，MTF 与共轭残差对照尚无法验收。"
           "本轮低内存主动中止，操作系统是否会发生 OOM 未继续探测。","",
           "Ray 声明 16 个 CPU 和 8 个 OpticStudio 槽位，每任务请求 2 CPU、1 槽位，"
           "每个独立子进程创建一个会话。DLS 保持 2 核、每步 50 循环。",
           "", "| Case | Rule | 完成 | 与 V1 对照通过 | 组耗时秒 | 差异/失败 |",
           "|---|---|---|---|---:|---|"]
    for row in comparisons:
        lines.append(f'| {row["case_id"]} | {row["rule"]} | {row.get("complete",False)} | '
                     f'{row["passed"]} | {row.get("parallel_wall_seconds",0):.2f} | '
            f'{row.get("mismatches",row.get("failure",""))} |')
    lines+=["","指标明细和数值容差见 acceptance.json，资源采样见 telemetry.jsonl。",
            "","系统 CPU 使用率以全机 0–100% 统计，包含其他应用；采样间隔约 0.5 秒。",
            "","若停止准入，排队任务会跳过，已启动任务在无紧急内存压力时继续保存结果。"
            "紧急内存不足会终止本次测试所属的子进程树。",
            "",json.dumps(result["observations"],indent=2),""]
    (args.output/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"accepted":accepted,**result["observations"]},indent=2))


if __name__=="__main__":
    main()
