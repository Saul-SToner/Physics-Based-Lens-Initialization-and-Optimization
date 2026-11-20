# Ray 8 并行压力测试验收

验收通过并可用于后续批量生成：True。

本轮在资源保护下完成 9 组完整 DLS 路径，未触发停止准入。

Ray 声明 16 个 CPU 和 8 个 OpticStudio 槽位，每任务请求 2 CPU、1 槽位，每个独立子进程创建一个会话。DLS 保持 2 核、每步 50 循环。

| Case | Rule | 完成 | 与 V1 对照通过 | 组耗时秒 | 差异/失败 |
|---|---|---|---|---:|---|
| 1 | 1 | True | True | 130.68 | [] |
| 1 | 2 | True | True | 232.70 | [] |
| 1 | 3 | True | True | 228.28 | [] |
| 2 | 1 | True | True | 168.61 | [] |
| 2 | 2 | True | True | 273.50 | [] |
| 2 | 3 | True | True | 250.34 | [] |
| 3 | 1 | True | True | 176.95 | [] |
| 3 | 2 | True | True | 227.33 | [] |
| 3 | 3 | True | True | 189.15 | [] |

指标明细和数值容差见 acceptance.json，资源采样见 telemetry.jsonl。

系统 CPU 使用率以全机 0–100% 统计，包含其他应用；采样间隔约 0.5 秒。

若停止准入，排队任务会跳过，已启动任务在无紧急内存压力时继续保存结果。紧急内存不足会终止本次测试所属的子进程树。

{
  "peak_active_sessions": 8,
  "peak_running_groups": 8,
  "peak_live_subprocesses": 8,
  "minimum_available_memory_gib": 1.0136604309082031,
  "mean_system_cpu_percent": 34.78566978193147,
  "peak_system_cpu_percent": 68.9,
  "wall_seconds_including_ray_setup": 344.9442402999998,
  "workload_wall_seconds": 342.05252070000006,
  "stop_events": [],
  "historical_files_changed": [],
  "peak_committed_gib": 23.221233367919922,
  "peak_commit_fraction": 0.7581836296145266,
  "samples_with_eight_running": 294,
  "mean_system_cpu_percent_at_eight_running": 45.814625850340136
}
