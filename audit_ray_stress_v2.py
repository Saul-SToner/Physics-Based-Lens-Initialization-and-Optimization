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
    worker_audits=[json.loads(path.read_text()) for path in args.output.glob("case_*/ray_worker_audit.json")]
    model_audits=[json.loads(path.read_text()) for path in args.output.glob("case_*/model_import_audit.json")]
    dependency_checks=(len(worker_audits)==9 and
                       all(not row["model_modules_loaded"] for row in worker_audits))
    thread_checks=(len(model_audits)==9 and all(
        row["numpy_imported_after_limits"] and all(v=="1" for v in row["thread_environment"].values())
        and row["dls_cores"]==2 and row["dls_cycles_per_step"]==50
        for row in model_audits))
    accepted=accepted and dependency_checks and thread_checks
    result={"accepted_for_batch_generation":accepted,
            "scheduler_dependency_checks_passed":dependency_checks,
            "model_thread_checks_passed":thread_checks,
            "ray_task_worker_unique_pids":sorted(set(row["pid"] for row in worker_audits)),
            "model_import_private_bytes_range":[min(row["private_bytes_after_model_import"] for row in model_audits),max(row["private_bytes_after_model_import"] for row in model_audits)],

            "comparisons":comparisons,
            "observations":{key:report[key] for key in (
                "peak_active_sessions","peak_running_groups","peak_live_subprocesses",
                "minimum_available_memory_gib","mean_system_cpu_percent","peak_system_cpu_percent",
                "wall_seconds_including_ray_setup","workload_wall_seconds","stop_events",
        "historical_files_changed","peak_committed_gib","peak_commit_fraction")}}
    samples=[json.loads(line) for line in (args.output/"telemetry.jsonl").read_text().splitlines()]
    at_eight=[sample for sample in samples if sample["running_groups"]==8]
    result["observations"]["samples_with_eight_running"]=len(at_eight)
    result["observations"]["mean_system_cpu_percent_at_eight_running"]=(
        sum(s["system_cpu_percent"] for s in at_eight)/len(at_eight) if at_eight else None)
    (args.output/"acceptance.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    lines=["# Ray 8 并行压力测试验收","",
           f"验收通过并可用于后续批量生成：{accepted}。","",
           ("本轮在资源保护下完成 9 组完整 DLS 路径，未触发停止准入。"
            if not report["stop_events"] else
            "本轮触发资源保护，停止记录及未完成组见下表。"),"",
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
