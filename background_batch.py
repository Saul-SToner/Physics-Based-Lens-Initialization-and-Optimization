"""Detached, resumable Ray batch control with an independent notification watchdog."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep, time
import traceback
from uuid import uuid4

import psutil


ROOT = Path(__file__).resolve().parent
TERMINAL = {"completed", "stopped", "failed"}


def read(path):
    # Allow Windows writers to atomically replace a snapshot while it is read.
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except Exception:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        return json.load(stream)


@contextmanager
def write_lock(path):
    import msvcrt
    with Path(str(path)+".write.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def save(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)+"\n"
    with write_lock(path):
        temporary = path.with_suffix(f".{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(20):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == 19:
                        raise
                    sleep(0.05)
        finally:
            temporary.unlink(missing_ok=True)


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def alive(identity):
    try:
        return abs(psutil.Process(identity["pid"]).create_time()-identity["created"]) < 0.01
    except psutil.NoSuchProcess:
        return False


def terminate(identity):
    if alive(identity):
        subprocess.run(["taskkill", "/PID", str(identity["pid"]), "/T", "/F"], capture_output=True)


def validate_group(directory, case_id, rule):
    directory = Path(directory)
    result = read(directory/"experiment/result.json")
    if (result["case_id"], result["rule"]) != (case_id, rule):
        raise ValueError("Result identity mismatch")
    if not result["steps"] or result["cost"]["requested_dls_cycles"] != 50*len(result["steps"]):
        raise ValueError("Incomplete result or wrong cycle budget")
    if not result["audit"]["case_identity_verified"]:
        raise ValueError("Case verification missing")
    if result["final_accepted"] is not None:
        if not result["audit"].get("saved_final_verified"):
            raise ValueError("Final prescription reload verification missing")
        final = result["final_accepted"]
        if not final["geometry_valid"] or not all(f["meets_threshold"] for f in final["fields"]):
            raise ValueError("Final prescription failed acceptance")
        lens_hash = fingerprint(directory/"experiment/final_accepted.zmx")
    else:
        if not all(value == "initial_rejected" for value in result["stop_reasons"].values()):
            raise ValueError("Missing accepted endpoint without a valid rejection outcome")
        lens_hash = None
    return {"result_sha256": fingerprint(directory/"experiment/result.json"),
            "lens_sha256": lens_hash, "target_reached": result["target_reached"],
            "final_accepted": result["final_accepted"], "cost": result["cost"],
            "stop_reasons": result["stop_reasons"]}


def execute(job_path):
    """Ray scheduling worker. Optical dependencies are loaded only by its child."""
    directory = Path(job_path).parent
    job = read(job_path)
    run = Path(job["run_directory"])
    if (run/"STOP_ADMISSION").exists() or (run/"STOP_REQUEST").exists():
        return {"status": "skipped", "reason": "admission_stopped"}
    if any(name in sys.modules for name in ("p1_engine", "run_p1", "clr", "ZOSAPI")):
        raise RuntimeError("Unexpected model dependency in scheduling worker")
    if job.get("staggered_start"):
        deadline = time()+240
        while not (directory/"START_PERMIT").exists():
            if (run/"STOP_ADMISSION").exists() or (run/"STOP_REQUEST").exists():
                return {"status":"skipped", "reason":"admission_stopped"}
            if time() > deadline:
                return {"status":"skipped", "reason":"startup_admission_timeout"}
            sleep(0.2)
    started = time()
    reason = None
    with (directory/"worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-B", str(ROOT/"background_model_worker.py"), str(job_path)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        identity = {"pid": process.pid, "created": psutil.Process(process.pid).create_time()}
        save(directory/"launcher.json", identity)
        while process.poll() is None:
            if (run/"EMERGENCY_STOP").exists() or (run/"STOP_REQUEST").exists():
                reason = "resource_emergency_or_user_stop"
            elif time()-started > job["group_timeout_seconds"]:
                reason = "group_timeout"
            if reason:
                terminate(identity)
                process.wait(timeout=30)
                break
            sleep(0.5)
    return {"status": "returned", "returncode": process.returncode,
            "reason": reason, "wall_seconds": time()-started}


def set_terminal(directory, status, reason):
    state = read(directory/"state.json")
    state.update(status=status, reason=reason, heartbeat=time(),
                 finished_utc=datetime.now(timezone.utc).isoformat())
    save(directory/"state.json", state)
    telemetry = directory/f'telemetry_{state["event_id"]}.jsonl'
    metrics = {"samples":0, "minimum_available_bytes":None, "peak_committed_bytes":0}
    if telemetry.exists():
        with telemetry.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metrics["samples"] += 1
                available = sample["available_physical_bytes"]
                metrics["minimum_available_bytes"] = min(
                    metrics["minimum_available_bytes"] or available, available)
                metrics["peak_committed_bytes"] = max(
                    metrics["peak_committed_bytes"],
                    sample["system_memory_counters"].get("committed_bytes") or 0)
    save(directory/f'summary_{state["event_id"]}.json', {
        "status":status,"reason":reason,"completed":state.get("completed",0),
        "total":state.get("total",0),"jobs":state.get("jobs",{}),"resources":metrics,
    })
    return state


def prepare_attempt(directory, key, info, plan):
    base = directory/key
    base.mkdir(exist_ok=True)
    sequence = len(list(base.glob("attempt_*")))+1
    attempt_dir = base/f"attempt_{sequence:03d}"
    attempt_dir.mkdir()
    case = next(case for case in plan["cases"] if case["id"] == info["case_id"])
    save(attempt_dir/"job.json", {
        "case":case, "case_id":case["id"], "rule":info["rule"],
        "installation":plan["installation"], "run_directory":str(directory),
        "release_file":str(directory/"RELEASE"), "stop_file":str(directory/"STOP_ADMISSION"),
        "group_timeout_seconds":plan["group_timeout_seconds"], "startup_timeout_seconds":240,
        "staggered_start":True,
    })
    info.pop("error", None)
    info.pop("acceptance", None)
    info.update(status="queued", attempt=str(attempt_dir.relative_to(directory)))
    return attempt_dir


def admit_start(directory, pending_keys, state, free_bytes, stop_reason):
    """Grant at most one startup permit until that instance reports ready."""
    if stop_reason:
        return None
    waiting = []
    for key in pending_keys:
        attempt_dir = directory/state["jobs"][key]["attempt"]
        if (attempt_dir/"START_PERMIT").exists():
            status_path = attempt_dir/"model_status.json"
            if not status_path.exists():
                return None
            phase = read(status_path)["phase"]
            if phase == "failed":
                return "model_process_failed"
            if phase == "opening":
                return None
        else:
            waiting.append(attempt_dir)
    if not waiting:
        return None
    if free_bytes < 1024**3:
        return "startup_admission_requires_1_GiB_available"
    (waiting[0]/"START_PERMIT").write_text("granted")
    return None


def check_sources(plan):
    for name, expected in plan["source_sha256"].items():
        # Controller fixes may be adopted on resume. Scientific code and the
        # validated resource profile remain pinned, with actual controller
        # versions recorded for each execution.
        if name in ("background_batch.py", "background_model_worker.py"):
            continue
        if fingerprint(ROOT/name) != expected:
            raise RuntimeError(f"Source changed since preparation: {name}")
    workflow = read(ROOT/"opticstudio_workflow_freeze_v1.json")
    for name, expected in workflow["sha256"].items():
        if fingerprint(ROOT/name) != expected:
            raise RuntimeError(f"Frozen workflow changed: {name}")


def daemon(directory):
    import msvcrt

    lock_path = ROOT/"background_runs/runner.lock"
    lock_path.parent.mkdir(exist_ok=True)
    ray = None
    lock = lock_path.open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    locked = False
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        locked = True
        plan = read(directory/"plan.json")
        check_sources(plan)
        profile = read(ROOT/"ray_p1_stress_8_v2/validated_runtime_profile.json")
        state = read(directory/"state.json")
        state.update(status="running", heartbeat=time(), reason="")
        save(directory/"state.json", state)
        import ray as runtime
        ray = runtime
        if ray.__version__ != profile["ray_version"]:
            raise RuntimeError("Ray version differs from validated runtime")
        ray.init(num_cpus=16, resources={"opticstudio_slot": 8}, include_dashboard=False,
                 object_store_memory=80*1024**2, log_to_driver=False,
                 _system_config=profile["ray_system_config"])
        task = ray.remote(num_cpus=2, resources={"opticstudio_slot": 1}, max_retries=0)(execute)
        pending = {}
        waiting = []
        for job_key, info in state["jobs"].items():
            if info["status"] == "completed":
                verified = validate_group(directory/info["attempt"], info["case_id"], info["rule"])
                if verified["result_sha256"] != info["acceptance"]["result_sha256"]:
                    raise RuntimeError(f"Completed result changed: {job_key}")
                if verified["lens_sha256"] != info["acceptance"]["lens_sha256"]:
                    raise RuntimeError(f"Completed lens changed: {job_key}")
                continue
            info["status"] = "pending"
            waiting.append(job_key)
        waiting = iter(waiting)
        def fill_slots():
            while len(pending) < 8:
                key = next(waiting, None)
                if key is None:
                    break
                attempt_dir = prepare_attempt(directory, key, state["jobs"][key], plan)
                pending[task.remote(str(attempt_dir/"job.json"))] = key
        fill_slots()
        save(directory/"state.json", state)
        initial_barrier_count = min(8, len(pending))
        references = {}
        baseline = ROOT/"p1_random_3_v1/batch.json"
        if baseline.exists():
            references = {(row["case_id"], row["rule"]):row for row in read(baseline)["groups"]}
        from profile_resource_attribution import WindowsCounters
        counters = WindowsCounters()
        started = time()
        low_since = None
        stop_reason = None
        psutil.cpu_percent(None)
        def stop(reason, emergency=False):
            nonlocal stop_reason
            stop_reason = stop_reason or reason
            (directory/"STOP_ADMISSION").write_text(stop_reason)
            if emergency:
                (directory/"EMERGENCY_STOP").write_text(stop_reason)
        try:
            with (directory/f'telemetry_{state["event_id"]}.jsonl').open("w", encoding="utf-8") as stream:
                while pending:
                    free = psutil.virtual_memory().available
                    memory = counters.read()
                    if free < 256*1024**2:
                        low_since = low_since or time()
                        if time()-low_since >= 5:
                            stop("available_memory_below_256_MiB_for_5_seconds")
                    else:
                        low_since = None
                    if free < 128*1024**2:
                        stop("available_memory_below_128_MiB", True)
                    committed, limit = memory.get("committed_bytes"), memory.get("commit_limit_bytes")
                    if committed and limit and committed/limit > 0.95:
                        stop("system_commit_above_95_percent", True)
                    if (directory/"STOP_REQUEST").exists():
                        stop("user_stop")
                    admission_reason = admit_start(directory, list(pending.values()), state, free, stop_reason)
                    if admission_reason:
                        stop(admission_reason)
                    active = 0
                    owned = []
                    for key in pending.values():
                        attempt_dir = directory/state["jobs"][key]["attempt"]
                        path = attempt_dir/"model_status.json"
                        if path.exists() and read(path)["phase"] in ("ready", "running"):
                            active += 1
                            state["jobs"][key]["status"] = read(path)["phase"]
                        reference = references.get((state["jobs"][key]["case_id"], state["jobs"][key]["rule"]))
                        progress_path = attempt_dir/"experiment/result.json"
                        if reference and progress_path.exists():
                            progress = read(progress_path)
                            ratios = [a["dls_seconds"]/max(b["dls_seconds"], 1e-6)
                                      for a,b in zip(progress["steps"], reference["steps"])]
                            if len(ratios) >= 3 and all(ratio > 3 for ratio in ratios[-3:]):
                                stop(f"three_consecutive_steps_over_3x_baseline_{key}")
                        launcher = attempt_dir/"launcher.json"
                        if launcher.exists():
                            identity = read(launcher)
                            if alive(identity):
                                owned.append(identity)
                                try:
                                    owned.extend({"pid": child.pid, "created": child.create_time()}
                                                 for child in psutil.Process(identity["pid"]).children(recursive=True))
                                except psutil.NoSuchProcess:
                                    pass
                    save(directory/"owned_processes.json", owned)
                    if not (directory/"RELEASE").exists() and active >= initial_barrier_count and not stop_reason:
                        (directory/"RELEASE").write_text("ready")
                    if not (directory/"RELEASE").exists() and time()-started > 240:
                        stop("startup_timeout", True)
                    stream.write(json.dumps({
                        "utc": datetime.now(timezone.utc).isoformat(), "available_physical_bytes": free,
                        "system_memory_counters": memory, "system_cpu_percent": psutil.cpu_percent(None),
                        "active_groups": active,
                        "in_flight_tasks":len(pending),
                    })+"\n")
                    stream.flush()
                    finished, _ = ray.wait(list(pending), num_returns=1, timeout=0)
                    for ref in finished:
                        key = pending.pop(ref)
                        info = state["jobs"][key]
                        try:
                            outcome = ray.get(ref)
                            if outcome.get("returncode") != 0:
                                raise RuntimeError(str(outcome))
                            info["acceptance"] = validate_group(directory/info["attempt"], info["case_id"], info["rule"])
                            info["status"] = "completed"
                        except Exception as error:
                            info.update(status="interrupted", error=str(error))
                            stop(f"group_failed_{key}")
                            launcher = directory/info["attempt"]/"launcher.json"
                            if launcher.exists():
                                terminate(read(launcher))
                    if not stop_reason:
                        fill_slots()
                    state["completed"] = sum(info["status"] == "completed" for info in state["jobs"].values())
                    state["heartbeat"] = time()
                    save(directory/"state.json", state)
                    sleep(0.5)
        finally:
            counters.close()
        check_sources(plan)
        set_terminal(directory, "stopped" if stop_reason else "completed", stop_reason or "all_groups_validated")
    except Exception:
        set_terminal(directory, "failed", traceback.format_exc())
        for identity in read(directory/"owned_processes.json") if (directory/"owned_processes.json").exists() else []:
            terminate(identity)
    finally:
        if ray is not None:
            ray.shutdown()
        if locked:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


def watchdog(directory):
    from notify_pushplus import deliver_terminal

    identity = read(directory/"daemon.json")
    while True:
        state = read(directory/"state.json")
        if state["status"] in TERMINAL and not alive(identity):
            break
        if not alive(identity) or time()-state["heartbeat"] > 180:
            for owned in read(directory/"owned_processes.json") if (directory/"owned_processes.json").exists() else []:
                terminate(owned)
            terminate(identity)
            state = set_terminal(directory, "failed", "daemon_exited_or_heartbeat_stalled")
            break
        sleep(2)
    for attempt in range(3):
        result = deliver_terminal(directory, state)
        if result["accepted"]:
            break
        sleep(5*(attempt+1))


def detached(arguments, logfile):
    with Path(logfile).open("a", encoding="utf-8") as log:
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        process = subprocess.Popen(
            [sys.executable, "-u", "-B", str(Path(__file__).resolve()), *arguments],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            startupinfo=startup,
        )
    return {"pid": process.pid, "created": psutil.Process(process.pid).create_time()}


def launch(directory):
    state = read(directory/"state.json")
    if state["status"] == "completed":
        for info in state["jobs"].values():
            verified = validate_group(directory/info["attempt"], info["case_id"], info["rule"])
            if (verified["result_sha256"] != info["acceptance"]["result_sha256"]
                    or verified["lens_sha256"] != info["acceptance"]["lens_sha256"]):
                raise RuntimeError("Completed batch data changed")
        print("Already completed; validated groups will not be rerun")
        return
    if (directory/"daemon.json").exists() and alive(read(directory/"daemon.json")):
        raise RuntimeError("This batch is already running")
    check_sources(read(directory/"plan.json"))
    if state.get("event_id"):
        history = directory/"history"
        history.mkdir(exist_ok=True)
        snapshot = history/f'state_{state["event_id"]}.json'
        if not snapshot.exists():
            save(snapshot, state)
    for name in ("STOP_REQUEST", "STOP_ADMISSION", "EMERGENCY_STOP", "RELEASE"):
        (directory/name).unlink(missing_ok=True)
    state.update(status="prepared", event_id=uuid4().hex, heartbeat=time(), reason="",
                 runtime_policy={"max_in_flight":8,"serial_instance_startup":True,
                                 "minimum_available_before_instance_start_bytes":1024**3},
                 execution_sources={name:fingerprint(ROOT/name) for name in (
                     "background_batch.py", "background_model_worker.py", "notify_pushplus.py")})
    save(directory/"state.json", state)
    identity = detached(["_daemon", str(directory)], directory/"daemon.log")
    save(directory/"daemon.json", identity)
    watcher = detached(["_watch", str(directory)], directory/"watchdog.log")
    save(directory/"watchdog.json", watcher)
    print(json.dumps({"directory": str(directory), "daemon_pid": identity["pid"],
                      "watchdog_pid": watcher["pid"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "resume", "status", "stop", "notify", "_daemon", "_watch"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cases", default="1,2,3")
    parser.add_argument("--dataset", type=Path, default=ROOT/"random_candidates_10/candidates.json")
    parser.add_argument("--rules", default="1,2,3")
    parser.add_argument("--installation", default="C:/Program Files/Ansys Zemax OpticStudio 2024 R1.00")
    parser.add_argument("--group-timeout", type=int, default=1800)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "start":
        if directory.exists():
            parser.error("New batch directory already exists")
        candidates = read(args.dataset)["candidates"]
        cases = ([case["id"] for case in candidates if case["passed"]]
                 if args.cases == "all" else [int(value) for value in args.cases.split(",")])
        rules = [int(value) for value in args.rules.split(",")]
        if len(set(cases)) != len(cases) or not set(rules) <= {1,2,3} or len(set(rules)) != len(rules):
            parser.error("Duplicate case/rule or invalid rule")
        if args.group_timeout <= 240:
            parser.error("Group timeout must exceed startup timeout of 240 seconds")
        selected = [next(case for case in candidates if case["id"] == number and case["passed"]) for number in cases]
        validated = read(ROOT/"ray_p1_stress_8_v2/validated_runtime_profile.json")
        if not validated["validated"]:
            raise RuntimeError("Parallel runtime is not validated")
        source_names = ("run_p1.py", "p1_engine.py", "analyze_opticstudio.py", "lens_geometry.py",
                        "compare_opticstudio.py", "background_batch.py", "background_model_worker.py",
                        "profile_resource_attribution.py", "ray_p1_stress_8_v2/validated_runtime_profile.json")
        directory.mkdir(parents=True)
        save(directory/"plan.json", {
            "cases": selected, "rules": rules, "installation": args.installation,
            "dataset_path": str(args.dataset.resolve()), "dataset_sha256": fingerprint(args.dataset),
            "group_timeout_seconds": args.group_timeout,
            "resume_policy": "skip verified completed groups; restart interrupted groups from original seed in a new attempt directory",
            "source_sha256": {name: fingerprint(ROOT/name) for name in source_names},
        })
        jobs = {f'case_{case["id"]:03d}_rule_{rule}':
                {"case_id": case["id"], "rule": rule, "status": "pending"}
                for case in selected for rule in rules}
        save(directory/"state.json", {"status": "prepared", "jobs": jobs,
                                     "completed": 0, "total": len(jobs), "heartbeat": time()})
        launch(directory)
    elif args.command == "resume":
        launch(directory)
    elif args.command == "status":
        print(json.dumps(read(directory/"state.json"), indent=2))
    elif args.command == "stop":
        (directory/"STOP_REQUEST").write_text("user_stop")
        print("Stop requested")
    elif args.command == "notify":
        from notify_pushplus import deliver_terminal
        state = read(directory/"state.json")
        if state["status"] not in TERMINAL:
            raise RuntimeError("Batch has no terminal event")
        print(json.dumps(deliver_terminal(directory, state)))
    elif args.command == "_daemon":
        daemon(directory)
    elif args.command == "_watch":
        watchdog(directory)


if __name__ == "__main__":
    main()
