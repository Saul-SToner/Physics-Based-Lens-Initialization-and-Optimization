# 后台批处理与 PushPlus

Ray 配置为 16 个逻辑 CPU、每组请求 2 CPU 和 1 个槽位、最多 8 组。模型子进程在导入 NumPy 前设置辅助线程为 1。OpticStudio DLS 继续使用 2 核、每步 50 循环，评价函数与几何规则来自现有 P1 引擎。

## 使用

在项目目录打开 PowerShell，使用固定路径环境运行。启动命令返回后，隐藏的主进程和独立看门狗继续运行。

    .\.venv-background\Scripts\python.exe -B background_batch.py start background_runs\batch_001 --cases 1,2,3 --rules 1,2,3
    .\.venv-background\Scripts\python.exe -B background_batch.py status background_runs\batch_001
    .\.venv-background\Scripts\python.exe -B background_batch.py stop background_runs\batch_001
    .\.venv-background\Scripts\python.exe -B background_batch.py resume background_runs\batch_001

start 要求全新目录。候选来自 random_candidates_10/candidates.json，只接纳已验证的候选。当前不会自动扩大样本量。

恢复按完整实验组进行。已完成组重新校验结果和最终处方的哈希后跳过，中断组从原始种子重跑，创建 attempt_002 等新目录。旧尝试、处方和逐步日志保留。中断组的单步 DLS 状态暂不恢复。

科学代码、冻结流程和已验证运行配置保持校验。控制器补丁可在恢复时采用，每次执行将实际控制器哈希写入状态文件。

## 自动保护与清理

- 全项目文件锁限制为一个后台批次，避免同项目启动多个批次争用实例。
- 最多向 Ray 保留 8 个在途任务，完成后补位。实例逐个启动，就绪后再启动下一个，初始各实例就绪后释放并行 DLS。新实例启动前要求至少 1 GiB 可用物理内存。
- 状态读取使用 Windows 共享删除权限，状态写入使用进程间锁、独立临时文件和落盘后的原子替换，并对临时访问拒绝重试。恢复前保存旧状态快照。
- 每组独立模型进程和会话，退出码为零且结果通过身份、预算、几何与 MTF 校验后才标为完成。
- 可用物理内存持续 5 秒低于 256 MiB 时停止新任务准入。
- 可用物理内存低于 128 MiB，或系统提交超过上限 95% 时终止本次任务所属进程树。
- P1 参考组连续三步 DLS 耗时超过对应串行步骤 3 倍时停止准入，新候选通过超时保护。
- 默认单组总超时 1800 秒，启动屏障超时 240 秒。新批次可用 --group-timeout 显式指定管理超时。
- 人工 stop 会终止本批次的运行组并保留尝试目录，之后使用 resume。
- 看门狗每两秒检查主进程，主进程消失或心跳超过 180 秒时清理登记进程并记录失败。
- 清理前核对 PID 和创建时间，避免 PID 复用时误杀其他进程。

系统休眠或关机期间程序无法推进，重新开机后使用 resume。当前没有安装开机自动任务。

## 结果与通知

- plan.json 保存候选、规则、科学代码哈希及超时设置。
- state.json 保存当前状态，终态为 completed、stopped 或 failed。
- 每次执行产生独立的 telemetry_<event_id>.jsonl 和 summary_<event_id>.json。
- 每组每次尝试保留 job.json、worker.log、种子和 experiment/result.json。
- 主进程日志为 daemon.log，看门狗日志为 watchdog.log。

完成、停止或异常退出由独立看门狗发送 PushPlus。凭据优先读取环境变量 PUSHPLUS_TOKEN，否则读取用户指定旧项目 C:/ZemaxAuto/dji_zemax_checker/src/notify_pushplus.py 的常量，凭据不会复制进本项目。

HTTP 成功还需 JSON 响应 code=200 才标为接口受理。回执保存在 notification_<event_id>.json。已受理的同一事件跳过重复发送，未受理最多自动尝试三次，之后可手动重发。

    .\.venv-background\Scripts\python.exe -B background_batch.py notify background_runs\batch_001

code=200 表示接口受理。网络响应丢失时重试仍可能重复推送，消息包含事件 ID。推送失败不改变科学结果验收状态。

## 本次验证

8 项定向测试覆盖通知去重、接口错误、完成组校验、中断后新建尝试目录、PID 保护、超时清理和主进程异常。

独立后台实测运行 Case 3 / Rule 1，在低内存保护触发前保留了 9 次尝试，随后清理进程并得到停止推送 code=200。该组尚未完成完整光学路径。

固定环境位于 .venv-background，依赖版本见 requirements-background.txt。
