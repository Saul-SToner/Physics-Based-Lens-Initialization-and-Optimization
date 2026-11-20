"""Ray schedules eight two-CPU subprocesses for the nine P1 reference groups."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep
import traceback

import psutil
import ray


def write_json(path, value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


@ray.remote(num_cpus=2, resources={"opticstudio_slot":1}, max_retries=0)
def execute_group(installation, job, original_threads):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from time import monotonic, sleep
    import psutil

    directory=Path(job).parent
    stop=Path(json.loads(Path(job).read_text())["stop"])
    if stop.exists():
        return {"skipped":True,"reason":"admission_stopped","job":job}
    env=dict(os.environ)
    for name,value in original_threads.items():
        if value is None:
            env.pop(name,None)
        else:
            env[name]=value
    started=monotonic()
    with (directory/"worker.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(
            [sys.executable,"-u","-B",str(Path(__file__).with_name("ray_stress_worker.py")),
             installation,job],stdout=log,stderr=subprocess.STDOUT,env=env)
        write_json(directory/"launcher.json",{"ray_worker_pid":os.getpid(),"child_pid":process.pid,
                                              "started":started})
        emergency=directory.parents[0]/"EMERGENCY_STOP"
        while process.poll() is None:
            if emergency.exists():
                descendants=psutil.Process(process.pid).children(recursive=True)
                for child in reversed(descendants):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
                break
            sleep(0.25)
    return {"job":job,"returncode":process.returncode,"subprocess_wall_seconds":monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=args.output.resolve()
    output.mkdir(exist_ok=False,parents=True)
    baseline=json.loads((root/"p1_random_3_v1/batch.json").read_text())
    cases={c["id"]:c for c in json.loads((root/"random_candidates_10/candidates.json").read_text())["candidates"]}
    frozen=json.loads((root/"opticstudio_workflow_freeze_v1.json").read_text())["sha256"]
    originals={str(root/name):value for name,value in frozen.items()}
    originals.update({str(p):sha256(p.read_bytes()).hexdigest()
                      for p in (root/"p1_random_3_v1").rglob("*") if p.is_file()})
    threads={name:os.environ.get(name) for name in
             ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")}
    # Logical admission requires 16 CPUs for eight tasks requesting two each.
    init_start=monotonic()
    ray.init(num_cpus=16,resources={"opticstudio_slot":8},include_dashboard=False,
             object_store_memory=80*1024**2,log_to_driver=False)
    resources=ray.cluster_resources()
    capacity=min(int(resources.get("CPU",0)//2),int(resources.get("opticstudio_slot",0)))
    config={"ray_version":ray.__version__,"psutil_version":psutil.__version__,
            "logical_cpus_host":psutil.cpu_count(),"cluster_resources":resources,
            "task_cpu_request":2,"task_slot_request":1,"schedulable_groups":capacity,
            "dls_cores":2,"dls_cycles_per_step":50,"original_thread_environment":threads,
            "stop_policy":{"low_available_bytes":256*1024**2,"low_memory_duration_seconds":5,
                           "performance_slowdown_factor":3,"consecutive_slow_steps":3,
                           "emergency_available_bytes":128*1024**2,
                           "automatic_task_retries":0},
            "source_sha256":{name:sha256((root/name).read_bytes()).hexdigest() for name in
                             ("run_ray_stress.py","ray_stress_worker.py","run_p1.py","p1_engine.py")},
            "historical_sha256":originals}
    write_json(output/"configuration.json",config)
    if capacity!=8:
        ray.shutdown()
        raise RuntimeError(f"Ray schedules {capacity}, expected 8")
    refs={}
    for index,reference in enumerate(baseline["groups"]):
        case_id,rule=reference["case_id"],reference["rule"]
        directory=output/f"case_{case_id:03d}_rule_{rule}"
        directory.mkdir()
        shutil.copyfile(root/f"p1_random_3_v1/case_{case_id:03d}/seed.zmx",directory/"seed.zmx")
        job={"case_id":case_id,"rule":rule,"case":cases[case_id],
             "release":str(output/"RELEASE"),"stop":str(output/"STOP_ADMISSION")}
        write_json(directory/"job.json",job)
        ref=execute_group.remote(str(args.installation.resolve()),str(directory/"job.json"),threads)
        refs[ref]=(directory,reference)
    psutil.cpu_percent(interval=None)
    samples=[]
    task_results=[]
    stop_events=[]
    peak_sessions=peak_running=peak_tasks=0
    low_since=None
    released=False
    started=monotonic()

    def stop(reason, emergency=False):
        if not (output/"STOP_ADMISSION").exists():
            (output/"STOP_ADMISSION").write_text(reason)
            stop_events.append({"reason":reason,"seconds":monotonic()-started})
            print("STOP_ADMISSION",reason,flush=True)
        if emergency:
            (output/"EMERGENCY_STOP").write_text(reason)

    try:
        with (output/"telemetry.jsonl").open("w",encoding="utf-8") as telemetry:
            while refs:
                now=monotonic()
                available=psutil.virtual_memory().available
                statuses=[]
                for directory in output.glob("case_*"):
                    path=directory/"status.json"
                    if path.exists():
                        statuses.append(json.loads(path.read_text()))
                active=[s for s in statuses if s["phase"] in ("ready","running") and psutil.pid_exists(s["pid"])]
                running=[s for s in active if s["phase"]=="running"]
                live_children=0
                native_pids=[]
                native_rss=0
                for process in psutil.process_iter(["pid","name","memory_info"]):
                    if "opticstudio" in (process.info["name"] or "").lower():
                        native_pids.append(process.pid)
                        if process.info["memory_info"]:
                            native_rss+=process.info["memory_info"].rss
                for directory in output.glob("case_*"):
                    path=directory/"launcher.json"
                    if path.exists() and psutil.pid_exists(json.loads(path.read_text())["child_pid"]):
                        live_children+=1
                peak_sessions=max(peak_sessions,len(active))
                peak_running=max(peak_running,len(running))
                peak_tasks=max(peak_tasks,live_children)
                sample={"seconds":now-started,"available_memory_bytes":available,
                        "system_cpu_percent":psutil.cpu_percent(interval=None),
                        "active_sessions":len(active),"running_groups":len(running),
                        "live_subprocesses":live_children,"opticstudio_pids":native_pids,
                        "opticstudio_rss_bytes":native_rss}
                samples.append(sample)
                telemetry.write(json.dumps(sample)+"\n")
                telemetry.flush()
                if available<256*1024**2:
                    low_since=low_since or now
                    if now-low_since>=5:
                        stop("available_memory_below_256_MiB_for_5_seconds")
                else:
                    low_since=None
                if available<128*1024**2:
                    stop("available_memory_below_128_MiB",emergency=True)
                if not released and len(active)==8:
                    if not (output/"STOP_ADMISSION").exists():
                        (output/"RELEASE").write_text("eight sessions ready")
                        print("Released eight live sessions",flush=True)
                        released=True
                if not released and now-started>240:
                    stop("eight_session_startup_barrier_timeout")
                for _,(directory,reference) in refs.items():
                    path=directory/"experiment/result.json"
                    if not path.exists():
                        continue
                    progress=json.loads(path.read_text())
                    ratios=[a["dls_seconds"]/max(b["dls_seconds"],1e-6)
                            for a,b in zip(progress["steps"],reference["steps"])]
                    if len(ratios)>=3 and all(value>3 for value in ratios[-3:]):
                        stop(f"three_consecutive_steps_over_3x_baseline_in_{directory.name}")
                done,_=ray.wait(list(refs),num_returns=1,timeout=0)
                for ref in done:
                    directory,reference=refs.pop(ref)
                    try:
                        result=ray.get(ref)
                    except Exception:
                        result={"job":str(directory/"job.json"),"error":traceback.format_exc()}
                    task_results.append(result)
                    write_json(output/"task_results.json",task_results)
                    print(directory.name,result,flush=True)
                    if result.get("returncode",0)!=0 or "error" in result:
                        stop(f"abnormal_exit_{directory.name}")
                        launcher=directory/"launcher.json"
                        if launcher.exists():
                            pid=json.loads(launcher.read_text())["child_pid"]
                            if psutil.pid_exists(pid):
                                subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True)
                sleep(0.5)
    finally:
        ray.shutdown()
        report={"configuration":config,"released_eight":released,
                "peak_active_sessions":peak_sessions,"peak_running_groups":peak_running,
                "peak_live_subprocesses":peak_tasks,"task_results":task_results,
                "stop_events":stop_events,
                "wall_seconds_including_ray_setup":monotonic()-init_start,
                "workload_wall_seconds":monotonic()-started,
                "minimum_available_memory_gib":min(s["available_memory_bytes"] for s in samples)/1024**3 if samples else None,
                "mean_system_cpu_percent":sum(s["system_cpu_percent"] for s in samples)/len(samples) if samples else None,
                "peak_system_cpu_percent":max(s["system_cpu_percent"] for s in samples) if samples else None,
                "historical_files_changed":[path for path,value in originals.items()
                                             if sha256(Path(path).read_bytes()).hexdigest()!=value]}
        write_json(output/"stress_report.json",report)


if __name__=="__main__":
    main()
