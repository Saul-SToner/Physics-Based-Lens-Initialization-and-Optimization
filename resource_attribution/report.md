# Ray / Python / OpticStudio 资源归因

原始 8 并行压力测试只保留了系统内存、系统 CPU 和 OpticStudio 总工作集，无法事后重建各进程私有内存和缺页率。本报告结合原始遥测与分离的短时补测，不把分离补测外推值视为原始 8 并行的逐进程实测。

补测包括 Ray 空载 8 任务、单组 DLS（50 循环、2 核）和 NumPy/CLR 导入分段，每段上限 120 秒，可用物理内存低于 1 GiB 即停止。全部补测均正常退出。

## 阶段资源

| 阶段 | 类别 | 私有提交峰值 MiB | 工作集峰值 MiB | CPU 均值（单核基准 %） |
|---|---|---:|---:|---:|
| baseline | profile_python | 17.9 | 31.8 | 5.6 |
| ray_startup | profile_python | 86.5 | 122.6 | 19.0 |
| ray_startup | ray_native | 172.5 | 187.2 | 0.6 |
| ray_startup | ray_python | 921.6 | 1294.0 | 95.5 |
| ray_eight_workers_idle | profile_python | 111.6 | 130.4 | 8.6 |
| ray_eight_workers_idle | ray_native | 175.5 | 190.9 | 3.9 |
| ray_eight_workers_idle | ray_python | 1420.7 | 1547.5 | 59.3 |
| opticstudio_startup | opticstudio | 242.3 | 260.6 | 14.2 |
| opticstudio_startup | profile_python | 852.6 | 157.6 | 8.4 |
| opticstudio_ready | opticstudio | 247.8 | 266.7 | 3.2 |
| opticstudio_ready | profile_python | 853.8 | 159.4 | 6.2 |
| dls_50_cycles_2_cores | opticstudio | 293.6 | 310.6 | 114.8 |
| dls_50_cycles_2_cores | profile_python | 855.3 | 161.5 | 5.1 |
| opticstudio_after_analysis | opticstudio | 252.9 | 274.8 | 17.8 |
| opticstudio_after_analysis | profile_python | 889.3 | 162.9 | 6.4 |

私有提交与工作集分别统计，同一行的两个峰值可能来自不同采样时刻。Ray/Python 启动器与实际解释器各自为独立 PID，原始数据保留二者。profile_python 类别含采样器、诊断驱动及对应子进程，不能全归给模型计算。

## 提交、缺页与磁盘

| 阶段 | 系统提交峰值 GiB | 页面调入峰值 页/s | 硬缺页读操作峰值 次/s | 磁盘吞吐峰值 MiB/s |
|---|---:|---:|---:|---:|
| baseline | 17.276 | 3314.1 | 561.5 | 12.74 |
| ray_startup | 18.703 | 5343.0 | 1230.5 | 20.29 |
| ray_eight_workers_idle | 19.206 | 159.9 | 66.2 | 2.36 |
| opticstudio_startup | 18.415 | 2995.3 | 713.2 | 11.85 |
| opticstudio_ready | 18.426 | 525.5 | 90.0 | 3.55 |
| dls_50_cycles_2_cores | 18.546 | 498.8 | 115.6 | 2.36 |
| opticstudio_after_analysis | 18.533 | 1142.4 | 193.6 | 4.87 |

页面调入可来自分页文件、可执行文件及映射文件。系统背景负载在基线时已存在，无法将系统总缺页或磁盘活动完全归因于测试进程。提交上限、磁盘传输率与队列长度在 JSON 中逐样本记录。

## 主要发现

- 原始压力测试在 8 个活跃会话时，OpticStudio 总工作集约 2.41 GiB，系统可用内存最低 78.5 MiB。
- 分离补测中单个 OpticStudio 就绪工作集约 267 MiB，DLS 峰值约 311 MiB；DLS 额外增长约 44 MiB。
- Ray 空载已有约 1.4 GiB 的 Python 私有提交，以及约 175 MiB 的 Ray 原生进程私有提交。详细身份补测确认 16 个实际 Ray 任务解释器，而只分配了 8 个任务。另有轻量启动器和监控代理。
- 单个模型 Python 的私有提交在 NumPy 导入时就大幅增加。默认 OpenBLAS 编译上限为 24 线程，导入后进程共 27 个线程；独立导入对照将 OPENBLAS_NUM_THREADS=1 后变为 4 个线程。
- NumPy 导入对照中私有提交从约 759 MiB 降至约 19 MiB，工作集约 31 MiB 降至 29 MiB。这主要节省提交空间，不能按每实例节省 740 MiB 物理 RAM 计算。
- 单组 OpticStudio 的瞬时 CPU 峰值约占 1.6 个逻辑 CPU，与 2 核预算相符。启动阶段有导入/预热 CPU 和页面调入，后续磁盘及内存压力可能限制总 CPU 使用率。

## 可降低的开销

1. 在新版本执行环境中限制 NumPy/OpenBLAS 的辅助线程，保留 OpticStudio DLS 的 2 核和全部评价函数参数。只做了导入级对照，应用到正式生成前仍需单组数值回归。
2. 评估按需启动 Ray worker，避免为 16 个逻辑 CPU 长期保留 16 个解释器。多出的约 8 个实际 worker 合计工作集估计约 0.55 GiB，实际可回收物理量需短测。保持 Ray CPU=16、每任务=2，不能直接把 Ray CPU 改为 8，否则最多只能接纳 4 组。
3. 保持进程隔离，尽量让 Ray 调度 worker 只传路径和小型状态，不导入光学模型或搬运大数组。当前 object store 已限制为 80 MiB，进一步缩减的空间很小。
4. 降低同一时刻的实例启动数量可减少启动瞬时压力，但不会消除 8 个活跃实例的稳态内存。
5. 现有处方、JSON 和 MTF 数组规模很小。降低日志频率、采样或 DLS 循环不应作为本轮内存优化手段。

以上为资源配置建议，正式优化脚本及 V1 历史未调整。没有再执行 8 组完整 DLS 压力测试。

## 逐进程记录

以下为同一 PID/创建时间生命周期内的观测峰值，完整阶段归属见 attribution.json。

| PID | PPID | 类别 | 私有提交 MiB | 工作集 MiB | CPU 峰值（单核 %） |
|---|---|---|---:|---:|---:|
| 43300 | 43584 | profile_python | 870.5 | 126.6 | 41.8 |
| 38292 | 45528 | resource_profiler | 786.1 | 65.5 | 35.5 |
| 38036 | 43300 | opticstudio | 314.9 | 310.6 | 159.9 |
| 36324 | 7768 | ray_native | 158.9 | 165.2 | 5.7 |
| 41876 | 30260 | native_or_other | 158.5 | 164.8 | 7.2 |
| 7768 | 44704 | profile_python | 92.9 | 94.2 | 38.6 |
| 30260 | 45032 | resource_profiler | 92.1 | 93.6 | 16.7 |
| 43332 | 41396 | ray_python | 72.4 | 71.1 | 64.3 |
| 42668 | 44660 | ray_task_worker | 72.3 | 71.0 | 2.3 |
| 46016 | 45628 | ray_python | 72.2 | 71.0 | 31.6 |
| 45488 | 44052 | ray_task_worker | 72.0 | 70.9 | 0.0 |
| 43252 | 43548 | ray_python | 71.9 | 70.8 | 77.0 |
| 27408 | 44096 | ray_python | 71.9 | 70.8 | 66.7 |
| 43516 | 42452 | ray_python | 71.9 | 70.9 | 70.6 |
| 40424 | 41040 | ray_task_worker | 71.8 | 70.7 | 2.4 |
| 37036 | 43532 | ray_python | 71.8 | 70.7 | 66.7 |
| 8956 | 20260 | ray_task_worker | 71.8 | 70.8 | 13.4 |
| 43184 | 40280 | ray_python | 71.8 | 70.7 | 62.3 |
| 37812 | 26944 | ray_task_worker | 71.7 | 70.6 | 4.2 |
| 45436 | 39864 | ray_task_worker | 71.7 | 70.6 | 0.0 |
| 43588 | 43608 | ray_task_worker | 71.7 | 69.9 | 1.9 |
| 37368 | 2228 | ray_task_worker | 71.6 | 70.6 | 0.0 |
| 34232 | 40144 | ray_task_worker | 71.6 | 69.9 | 0.0 |
| 21328 | 20272 | ray_task_worker | 71.6 | 70.6 | 11.6 |
| 42416 | 11956 | ray_python | 71.6 | 69.9 | 25.8 |
| 42680 | 41176 | ray_task_worker | 71.6 | 69.8 | 0.0 |
| 20332 | 39540 | ray_python | 71.4 | 69.8 | 27.9 |
| 24044 | 44884 | ray_task_worker | 71.4 | 69.8 | 7.4 |
| 45472 | 39892 | ray_python | 71.4 | 70.4 | 52.5 |
| 3832 | 44076 | ray_python | 71.4 | 69.7 | 29.6 |
| 13524 | 21004 | ray_python | 71.4 | 69.9 | 36.6 |
| 41764 | 9868 | ray_python | 71.3 | 69.7 | 27.9 |
| 17144 | 11300 | ray_task_worker | 71.3 | 69.7 | 12.4 |
| 42796 | 2104 | ray_python | 71.3 | 69.7 | 21.2 |
| 43468 | 41596 | ray_python | 71.3 | 69.7 | 35.2 |
| 40988 | 20424 | ray_task_worker | 71.3 | 69.7 | 0.0 |
| 44144 | 43600 | ray_task_worker | 71.2 | 69.4 | 0.0 |
| 23860 | 43360 | ray_task_worker | 71.1 | 69.5 | 11.8 |
| 29124 | 11560 | ray_python | 71.0 | 69.5 | 26.6 |
| 39132 | 38984 | ray_python | 54.3 | 69.8 | 25.5 |
| 8508 | 35964 | ray_python | 54.3 | 68.6 | 58.8 |
| 22388 | 41368 | ray_dashboard_agent | 54.1 | 68.6 | 21.0 |
| 44448 | 35492 | ray_runtime_env_agent | 53.9 | 70.3 | 16.9 |
| 3152 | 41288 | ray_python | 53.8 | 70.2 | 77.3 |
| 42764 | 18520 | ray_dashboard_agent | 53.8 | 68.6 | 66.7 |
| 42244 | 41548 | ray_python | 51.0 | 66.8 | 24.2 |
| 19576 | 27868 | ray_monitor | 50.9 | 66.8 | 45.5 |
| 43968 | 44648 | ray_python | 49.6 | 65.2 | 59.2 |
| 8688 | 33164 | ray_monitor | 49.1 | 64.8 | 12.0 |
| 41124 | 2468 | resource_profiler | 18.4 | 29.2 | 50.7 |
| 42632 | 37348 | profile_python | 17.2 | 28.1 | 37.2 |
| 22380 | 7768 | ray_native | 16.6 | 25.6 | 8.9 |
| 44772 | 30260 | ray_task_worker | 16.5 | 25.5 | 11.5 |
| 41288 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 41396 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 44096 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 2104 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 39540 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 35492 | 44772 | ray_runtime_env_agent | 0.8 | 4.1 | 0.0 |
| 20272 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 41040 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 43600 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 44660 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 41548 | 7768 | ray_python | 0.8 | 4.1 | 0.0 |
| 35964 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 44648 | 7768 | ray_python | 0.8 | 4.1 | 0.0 |
| 39892 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 9868 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 11956 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 33164 | 30260 | ray_monitor | 0.8 | 4.1 | 0.0 |
| 2228 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 20260 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 20424 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 40144 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 41176 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 44884 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 37348 | 15080 | profile_python | 0.8 | 4.1 | 0.0 |
| 42452 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 43548 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 45628 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 11560 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 21004 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 41596 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 43584 | 42632 | profile_python | 0.8 | 4.1 | 0.0 |
| 18520 | 30260 | ray_dashboard_agent | 0.8 | 4.1 | 0.0 |
| 41368 | 44772 | ray_dashboard_agent | 0.8 | 4.1 | 0.0 |
| 11300 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 26944 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 43360 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 38984 | 7768 | ray_python | 0.8 | 4.1 | 0.0 |
| 40280 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 43532 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 2468 | 29240 | resource_profiler | 0.8 | 4.1 | 0.0 |
| 27868 | 30260 | ray_monitor | 0.8 | 4.1 | 0.0 |
| 43608 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 44052 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 45528 | 41124 | resource_profiler | 0.8 | 4.1 | 0.0 |
| 44076 | 22380 | ray_python | 0.8 | 4.1 | 0.0 |
| 45032 | 41124 | resource_profiler | 0.8 | 4.1 | 0.0 |
| 39864 | 44772 | ray_task_worker | 0.8 | 4.1 | 0.0 |
| 44704 | 42632 | profile_python | 0.8 | 4.1 | 0.0 |

## 资料口径

[微软内存与硬缺页计数说明](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)

[Ray 内存管理](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html)

[Ray worker 预启动实现](https://github.com/ray-project/ray/blob/master/src/ray/raylet/worker_pool.cc)
