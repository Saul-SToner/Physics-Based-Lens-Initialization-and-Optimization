"""Short guarded Windows process profiling for Ray overhead and one DLS call."""

import argparse
import ctypes as ct
from ctypes import wintypes as wt
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import psutil


class CounterValue(ct.Structure):
    _fields_=[("status",wt.DWORD),("value",ct.c_double)]


class WindowsCounters:
    PATHS={
        "committed_bytes":r"\Memory\Committed Bytes",
        "commit_limit_bytes":r"\Memory\Commit Limit",
        "hard_fault_read_operations_per_sec":r"\Memory\Page Reads/sec",
        "hard_fault_pages_in_per_sec":r"\Memory\Pages Input/sec",
        "disk_bytes_per_sec":r"\PhysicalDisk(_Total)\Disk Bytes/sec",
        "disk_transfers_per_sec":r"\PhysicalDisk(_Total)\Disk Transfers/sec",
        "disk_queue_length":r"\PhysicalDisk(_Total)\Current Disk Queue Length",
    }

    def __init__(self):
        self.dll=ct.WinDLL("pdh")
        self.dll.PdhOpenQueryW.argtypes=[wt.LPCWSTR,ct.c_size_t,ct.POINTER(ct.c_void_p)]
        self.dll.PdhAddEnglishCounterW.argtypes=[ct.c_void_p,wt.LPCWSTR,ct.c_size_t,ct.POINTER(ct.c_void_p)]
        self.dll.PdhCollectQueryData.argtypes=[ct.c_void_p]
        self.dll.PdhGetFormattedCounterValue.argtypes=[ct.c_void_p,wt.DWORD,ct.c_void_p,ct.POINTER(CounterValue)]
        self.dll.PdhCloseQuery.argtypes=[ct.c_void_p]
        self.query=ct.c_void_p()
        if self.dll.PdhOpenQueryW(None,0,ct.byref(self.query)):
            raise OSError("PdhOpenQuery failed")
        self.handles={}
        self.errors={}
        for name,path in self.PATHS.items():
            handle=ct.c_void_p()
            status=self.dll.PdhAddEnglishCounterW(self.query,path,0,ct.byref(handle))
            if status:
                self.errors[name]=status
            else:
                self.handles[name]=handle
        self.dll.PdhCollectQueryData(self.query)

    def read(self):
        self.dll.PdhCollectQueryData(self.query)
        values={}
        for name,handle in self.handles.items():
            value=CounterValue()
            result=self.dll.PdhGetFormattedCounterValue(handle,0x200,None,ct.byref(value))
            values[name]=value.value if result==0 and value.status in (0,1) else None
        return values

    def close(self):
        self.dll.PdhCloseQuery(self.query)


def set_phase(output, phase):
    (output/"phase.txt").write_text(phase)


def ray_only(output):
    set_phase(output,"ray_startup")
    import ray
    ray.init(num_cpus=16,resources={"opticstudio_slot":8},include_dashboard=False,
             object_store_memory=80*1024**2,log_to_driver=False)
    @ray.remote(num_cpus=2,resources={"opticstudio_slot":1},max_retries=0)
    def hold(index, directory):
        from pathlib import Path
        from time import sleep
        p=Path(directory)
        (p/f"ray_ready_{index}").write_text(str(os.getpid()))
        for _ in range(200):
            if (p/"release_ray").exists():
                break
            sleep(0.1)
        return os.getpid()
    refs=[hold.remote(i,str(output)) for i in range(8)]
    deadline=monotonic()+60
    while len(list(output.glob("ray_ready_*")))<8 and monotonic()<deadline:
        sleep(0.2)
    set_phase(output,"ray_eight_workers_idle")
    sleep(5)
    (output/"release_ray").write_text("release")
    ray.get(refs)
    set_phase(output,"ray_shutdown")
    ray.shutdown()


def native(output, installation):
    from p1_engine import load_seed,configure_merit,inspect_prescription
    from analyze_opticstudio import measure
    root=Path(__file__).resolve().parent
    case=json.loads((root/"random_candidates_10/candidates.json").read_text())["candidates"][2]
    set_phase(output,"python_import_clr")
    import clr
    for name in ("ZOSAPI_Interfaces","ZOSAPI"):
        clr.AddReference(str(installation/(name+".dll")))
    import ZOSAPI as zos
    set_phase(output,"opticstudio_startup")
    connection=zos.ZOSAPI_Connection()
    app=connection.CreateNewApplication()
    if app is None:
        raise RuntimeError("No standalone application")
    try:
        if not app.IsValidLicenseForAPI:
            raise RuntimeError("No valid API license")
        system=app.PrimarySystem
        load_seed(system,root/"p1_random_3_v1/case_003/seed.zmx")
        set_phase(output,"opticstudio_ready")
        sleep(4)
        system.SystemData.Aperture.ApertureValue=20.0
        for number,angle in enumerate((0,3.535,5),1):
            system.SystemData.Fields.GetField(number).Y=angle
        set_phase(output,"merit_construction")
        configure_merit(zos,system,20.0,0.002,case["inputs"])
        tool=system.Tools.OpenLocalOptimization()
        try:
            tool.Algorithm=zos.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
            tool.Cycles=zos.Tools.Optimization.OptimizationCycles.Fixed_50_Cycles
            tool.NumberOfCores=2
            set_phase(output,"dls_50_cycles_2_cores")
            tool.RunAndWaitForCompletion()
        finally:
            tool.Close()
        set_phase(output,"fft_mtf_and_spot_analysis")
        result={**inspect_prescription(system,case["inputs"]),
                **measure(zos,system,20.0,(0,3.535,5))}
        (output/"native_result.json").write_text(json.dumps(result,indent=2))
        system.SaveAs(str(output/"diagnostic_result.zmx"))
        set_phase(output,"opticstudio_after_analysis")
        sleep(3)
    finally:
        set_phase(output,"opticstudio_shutdown")
        app.CloseApplication()


def imports_only(output, installation):
    for phase, action in (
        ("imports_before_numpy", lambda: None),
        ("imports_numpy", lambda: __import__("numpy")),
        ("imports_p1_engine", lambda: __import__("p1_engine")),
        ("imports_clr", lambda: __import__("clr")),
    ):
        action()
        set_phase(output,phase)
        sleep(3)
    import clr
    for name in ("ZOSAPI_Interfaces","ZOSAPI"):
        clr.AddReference(str(installation/(name+".dll")))
    set_phase(output,"imports_zosapi_assemblies")
    sleep(3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("installation",type=Path)
    parser.add_argument("--mode",choices=("ray","native","imports"))
    parser.add_argument("--profiles",nargs="+",choices=("ray","native","imports"),default=["ray","native"])
    args=parser.parse_args()
    output=args.output.resolve()
    if args.mode:
        if args.mode=="ray":
            ray_only(output)
        elif args.mode=="native":
            native(output,args.installation)
        else:
            imports_only(output,args.installation)
        return
    output.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parent
    prior={str(p):sha256(p.read_bytes()).hexdigest()
           for folder in ("p1_random_3_v1","ray_p1_stress_8")
           for p in (root/folder).rglob("*") if p.is_file()}
    frozen=json.loads((root/"opticstudio_workflow_freeze_v1.json").read_text())["sha256"]
    prior.update({str(root/p):value for p,value in frozen.items()})
    counters=WindowsCounters()
    processes={}
    profiles=[]
    start=monotonic()
    memory_floor=1024**3
    def sample(phase, stream):
        records=[]
        for process in psutil.process_iter(["pid","name","ppid"]):
            name=(process.info["name"] or "").lower()
            if not any(key in name for key in ("python","opticstudio","raylet","gcs_server")):
                continue
            try:
                identity=(process.pid,process.create_time())
                monitored=processes.setdefault(identity,process)
                cmd=" ".join(process.cmdline()).lower()
                program_kind=(
                    "ray_task_worker" if "default_worker.py" in cmd else
                    "ray_dashboard_agent" if "dashboard" in cmd else
                    "ray_runtime_env_agent" if "runtime_env" in cmd else
                    "ray_monitor" if "monitor.py" in cmd else
                    "ray_log_monitor" if "log_monitor" in cmd else
                    "resource_profiler" if "profile_resource_attribution.py" in cmd else
                    "native_or_other"
                )
                role=("opticstudio" if "opticstudio" in name else
                      "ray_native" if "raylet" in name or "gcs_server" in name else
                      "ray_python" if any(x in cmd for x in (
                          "ray::","default_worker.py","ray\\","ray/","dashboard","runtime_env")) else
                      "profile_python" if "profile_resource_attribution.py" in cmd else "other_python")
                memory=process.memory_info()
                records.append({"pid":process.pid,"ppid":process.ppid(),"name":process.name(),
                                "created":identity[1],"role":role,
                                "program_kind":program_kind,"threads":process.num_threads(),
                                "executable":process.exe(),
                                "private_bytes":memory.private,"working_set_bytes":memory.rss,
                                "cpu_percent_one_core_basis":monitored.cpu_percent(interval=None),
                                "read_bytes":process.io_counters().read_bytes,
                                "write_bytes":process.io_counters().write_bytes})
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        data={"seconds":monotonic()-start,"phase":phase,
              "available_physical_bytes":psutil.virtual_memory().available,
              "system_cpu_percent":psutil.cpu_percent(interval=None),
              "counters":counters.read(),"processes":records}
        stream.write(json.dumps(data)+"\n")
        stream.flush()
        return data
    try:
        psutil.cpu_percent(interval=None)
        with (output/"process_telemetry.jsonl").open("w",encoding="utf-8") as stream:
            for _ in range(5):
                sample("baseline",stream)
                sleep(0.5)
            for mode in args.profiles:
                directory=output/mode
                directory.mkdir()
                minimum=psutil.virtual_memory().available
                if minimum<1.5*1024**3:
                    profiles.append({"mode":mode,"skipped":"less_than_1.5_GiB_free_before_start"})
                    continue
                with (directory/"worker.log").open("w") as log:
                    process=subprocess.Popen(
                        [sys.executable,"-u","-B",__file__,str(directory),str(args.installation),"--mode",mode],
                        stdout=log,stderr=subprocess.STDOUT)
                    phase_path=directory/"phase.txt"
                    phase_start=monotonic()
                    reason=None
                    while process.poll() is None:
                        phase=phase_path.read_text() if phase_path.exists() else f"{mode}_python_startup"
                        data=sample(phase,stream)
                        minimum=min(minimum,data["available_physical_bytes"])
                        if minimum<memory_floor or monotonic()-phase_start>120:
                            reason="free_memory_below_1_GiB" if minimum<memory_floor else "120_second_limit"
                            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
                            process.wait()
                            break
                        sleep(0.5)
                    profiles.append({"mode":mode,"returncode":process.returncode,
                                     "guard_stop":reason,"minimum_free_gib":minimum/1024**3})
                for _ in range(6):
                    sample(f"{mode}_cleanup",stream)
                    sleep(0.5)
                print(profiles[-1],flush=True)
    finally:
        counters.close()
        changed=[p for p,value in prior.items() if sha256(Path(p).read_bytes()).hexdigest()!=value]
        (output/"run.json").write_text(json.dumps({
            "profiles":profiles,"counter_registration_errors":counters.errors,
            "history_changed":changed,"memory_floor_gib":1,"timeout_seconds_per_profile":120,
            "historical_files_verified":len(prior),"logical_cpu_count":psutil.cpu_count(),
        },indent=2)+"\n")


if __name__=="__main__":
    main()
