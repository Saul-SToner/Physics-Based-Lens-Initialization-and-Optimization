"""Aggregate guarded Windows profiling without summing process peaks as RAM."""

from collections import defaultdict
from pathlib import Path
import json


ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/"resource_attribution"


def main():
    samples=[]
    for folder in ("resource_attribution","resource_attribution_detail"):
        samples.extend({"run":folder,**json.loads(line)} for line in
                       (ROOT/folder/"process_telemetry.jsonl").read_text().splitlines())
    processes={}
    for sample in samples:
        for process in sample["processes"]:
            identity=f'{process["pid"]}_{process["created"]}'
            row=processes.setdefault(identity,{
                "pid":process["pid"],"ppid":process["ppid"],"name":process["name"],
                "role":process["role"],"samples":0,"peak_private_bytes":0,
                "peak_working_set_bytes":0,"peak_cpu_one_core_percent":0,
                "cpu_one_core_percent_sum":0,"phases":[],
            })
            row["samples"]+=1
            row["peak_private_bytes"]=max(row["peak_private_bytes"],process["private_bytes"])
            row["peak_working_set_bytes"]=max(row["peak_working_set_bytes"],process["working_set_bytes"])
            row["peak_cpu_one_core_percent"]=max(row["peak_cpu_one_core_percent"],process["cpu_percent_one_core_basis"])
            row["cpu_one_core_percent_sum"]+=process["cpu_percent_one_core_basis"]
            if sample["phase"] not in row["phases"]:
                row["phases"].append(sample["phase"])
            if "program_kind" in process:
                row["program_kind"]=process["program_kind"]
                row["executable"]=process["executable"]
                row["peak_threads"]=max(row.get("peak_threads",0),process["threads"])
    for row in processes.values():
        row["mean_cpu_one_core_percent"]=row.pop("cpu_one_core_percent_sum")/row["samples"]
        row["peak_cpu_whole_machine_percent"]=row["peak_cpu_one_core_percent"]/32
    phases=[]
    groups=defaultdict(list)
    for sample in samples:
        groups[(sample["run"],sample["phase"])].append(sample)
    for (run,phase),rows in groups.items():
        roles={}
        for role in sorted(set(p["role"] for row in rows for p in row["processes"])):
            roles[role]={
                "max_process_count":max(sum(p["role"]==role for p in row["processes"]) for row in rows),
                "peak_simultaneous_private_bytes":max(sum(p["private_bytes"] for p in row["processes"] if p["role"]==role) for row in rows),
                "peak_simultaneous_working_set_bytes":max(sum(p["working_set_bytes"] for p in row["processes"] if p["role"]==role) for row in rows),
                "mean_cpu_one_core_percent":sum(sum(p["cpu_percent_one_core_basis"] for p in row["processes"] if p["role"]==role) for row in rows)/len(rows),
            }
        phases.append({
            "run":run,"phase":phase,"sample_count":len(rows),"roles":roles,
            "minimum_available_gib":min(row["available_physical_bytes"] for row in rows)/1024**3,
            "mean_system_cpu_percent":sum(row["system_cpu_percent"] for row in rows)/len(rows),
            "counter_peaks":{key:max(row["counters"][key] for row in rows if row["counters"][key] is not None)
                             for key in rows[0]["counters"]},
        })
    report={"processes":list(processes.values()),"phases":phases,
            "definitions":{
                "private_bytes":"Windows private committed virtual memory; includes nonresident private pages",
                "working_set_bytes":"Resident working set, includes shared pages; process sums can double-count sharing",
                "cpu_one_core_percent":"100 percent means one fully occupied logical CPU; divide by 32 for whole-machine share",
                "hard_fault_read_operations_per_sec":"System-wide Memory Page Reads/sec disk read operations",
                "hard_fault_pages_in_per_sec":"System-wide Memory Pages Input/sec pages brought in to resolve hard faults",
                "sampling":"Approximately 0.5 seconds plus collection time; brief analysis phases can be missed",
            }}
    (OUTPUT/"attribution.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=["# Ray / Python / OpticStudio 资源归因","",
           "原始 8 并行压力测试只保留了系统内存、系统 CPU 和 OpticStudio 总工作集，"
           "无法事后重建各进程私有内存和缺页率。本报告结合原始遥测与分离的短时补测，"
           "不把分离补测外推值视为原始 8 并行的逐进程实测。","",
           "补测包括 Ray 空载 8 任务、单组 DLS（50 循环、2 核）和 NumPy/CLR 导入分段，"
           "每段上限 120 秒，可用物理内存低于 1 GiB 即停止。全部补测均正常退出。","",
           "## 阶段资源","",
           "| 阶段 | 类别 | 私有提交峰值 MiB | 工作集峰值 MiB | CPU 均值（单核基准 %） |",
           "|---|---|---:|---:|---:|"]
    selected=("baseline","ray_startup","ray_eight_workers_idle","opticstudio_startup",
              "opticstudio_ready","dls_50_cycles_2_cores","opticstudio_after_analysis")
    for phase in phases:
        if phase["run"]!="resource_attribution" or phase["phase"] not in selected:
            continue
        for role,values in phase["roles"].items():
            lines.append(f'| {phase["phase"]} | {role} | '
                         f'{values["peak_simultaneous_private_bytes"]/1024**2:.1f} | '
                         f'{values["peak_simultaneous_working_set_bytes"]/1024**2:.1f} | '
                         f'{values["mean_cpu_one_core_percent"]:.1f} |')
    lines+=["","私有提交与工作集分别统计，同一行的两个峰值可能来自不同采样时刻。"
            "Ray/Python 启动器与实际解释器各自为独立 PID，原始数据保留二者。"
            "profile_python 类别含采样器、诊断驱动及对应子进程，不能全归给模型计算。","",
            "## 提交、缺页与磁盘","",
            "| 阶段 | 系统提交峰值 GiB | 页面调入峰值 页/s | 硬缺页读操作峰值 次/s | 磁盘吞吐峰值 MiB/s |",
            "|---|---:|---:|---:|---:|"]
    for phase in phases:
        if phase["run"]!="resource_attribution" or phase["phase"] not in selected:
            continue
        v=phase["counter_peaks"]
        lines.append(f'| {phase["phase"]} | {v["committed_bytes"]/1024**3:.3f} | '
                     f'{v["hard_fault_pages_in_per_sec"]:.1f} | '
                     f'{v["hard_fault_read_operations_per_sec"]:.1f} | '
                     f'{v["disk_bytes_per_sec"]/1024**2:.2f} |')
    lines+=["","页面调入可来自分页文件、可执行文件及映射文件。系统背景负载在基线时已存在，"
            "无法将系统总缺页或磁盘活动完全归因于测试进程。提交上限、磁盘传输率与队列长度在 JSON 中逐样本记录。","",
            "## 主要发现","",
            "- 原始压力测试在 8 个活跃会话时，OpticStudio 总工作集约 2.41 GiB，系统可用内存最低 78.5 MiB。",
            "- 分离补测中单个 OpticStudio 就绪工作集约 267 MiB，DLS 峰值约 311 MiB；DLS 额外增长约 44 MiB。",
            "- Ray 空载已有约 1.4 GiB 的 Python 私有提交，以及约 175 MiB 的 Ray 原生进程私有提交。"
            "详细身份补测确认 16 个实际 Ray 任务解释器，而只分配了 8 个任务。另有轻量启动器和监控代理。",
            "- 单个模型 Python 的私有提交在 NumPy 导入时就大幅增加。默认 OpenBLAS 编译上限为 24 线程，"
            "导入后进程共 27 个线程；独立导入对照将 OPENBLAS_NUM_THREADS=1 后变为 4 个线程。",
            "- NumPy 导入对照中私有提交从约 759 MiB 降至约 19 MiB，工作集约 31 MiB 降至 29 MiB。"
            "这主要节省提交空间，不能按每实例节省 740 MiB 物理 RAM 计算。",
            "- 单组 OpticStudio 的瞬时 CPU 峰值约占 1.6 个逻辑 CPU，与 2 核预算相符。"
            "启动阶段有导入/预热 CPU 和页面调入，后续磁盘及内存压力可能限制总 CPU 使用率。","",
            "## 可降低的开销","",
            "1. 在新版本执行环境中限制 NumPy/OpenBLAS 的辅助线程，保留 OpticStudio DLS 的 2 核和全部评价函数参数。"
            "只做了导入级对照，应用到正式生成前仍需单组数值回归。",
            "2. 评估按需启动 Ray worker，避免为 16 个逻辑 CPU 长期保留 16 个解释器。"
            "多出的约 8 个实际 worker 合计工作集估计约 0.55 GiB，实际可回收物理量需短测。"
            "保持 Ray CPU=16、每任务=2，不能直接把 Ray CPU 改为 8，否则最多只能接纳 4 组。",
            "3. 保持进程隔离，尽量让 Ray 调度 worker 只传路径和小型状态，不导入光学模型或搬运大数组。"
            "当前 object store 已限制为 80 MiB，进一步缩减的空间很小。",
            "4. 降低同一时刻的实例启动数量可减少启动瞬时压力，但不会消除 8 个活跃实例的稳态内存。",
            "5. 现有处方、JSON 和 MTF 数组规模很小。降低日志频率、采样或 DLS 循环不应作为本轮内存优化手段。","",
            "以上为资源配置建议，正式优化脚本及 V1 历史未调整。没有再执行 8 组完整 DLS 压力测试。","",
            "## 逐进程记录","",
            "以下为同一 PID/创建时间生命周期内的观测峰值，完整阶段归属见 attribution.json。","",
            "| PID | PPID | 类别 | 私有提交 MiB | 工作集 MiB | CPU 峰值（单核 %） |",
            "|---|---|---|---:|---:|---:|"]
    for row in sorted(processes.values(),key=lambda row:row["peak_private_bytes"],reverse=True):
        lines.append(f'| {row["pid"]} | {row["ppid"]} | {row.get("program_kind",row["role"])} | '
                     f'{row["peak_private_bytes"]/1024**2:.1f} | '
                     f'{row["peak_working_set_bytes"]/1024**2:.1f} | '
                     f'{row["peak_cpu_one_core_percent"]:.1f} |')
    lines+=["","## 资料口径","",
            "[微软内存与硬缺页计数说明](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)",
            "",
            "[Ray 内存管理](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html)",
            "",
            "[Ray worker 预启动实现](https://github.com/ray-project/ray/blob/master/src/ray/raylet/worker_pool.cc)",""]
    (OUTPUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Wrote",len(processes),"process lifetimes and",len(phases),"phase summaries")


if __name__=="__main__":
    main()
