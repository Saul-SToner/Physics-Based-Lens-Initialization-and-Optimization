"""Barrier-synchronized DLS load test with separate native API processes."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
import traceback


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length",ctypes.c_ulong),("load",ctypes.c_ulong)] + [
        (name,ctypes.c_ulonglong) for name in
        ("total","available","page_total","page_available","virtual_total","virtual_available","extended")]


def available_memory():
    status=MemoryStatus()
    status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.available


def save(path, value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def worker(args):
    from p1_engine import attempt, load_seed
    root=Path(__file__).resolve().parent
    case=json.loads((root/"random_candidates_10/candidates.json").read_text())["candidates"][2]
    output=args.worker.resolve()
    app=None
    try:
        import clr
        for name in ("ZOSAPI_Interfaces","ZOSAPI"):
            clr.AddReference(str(args.installation.resolve()/(name+".dll")))
        import ZOSAPI as zos
        connection=zos.ZOSAPI_Connection()
        app=connection.CreateNewApplication()
        if app is None or not app.IsValidLicenseForAPI:
            raise RuntimeError("Licensed API session unavailable")
        system=app.PrimarySystem
        load_seed(system,root/"p1_random_3_v1/case_003/seed.zmx")
        output.with_suffix(".ready").write_text("ready")
        deadline=perf_counter()+240
        while not (output.parent/"go").exists():
            if perf_counter()>deadline:
                raise TimeoutError("Barrier timeout")
            sleep(0.1)
        result=attempt(zos,system,20.0,5.0,50,0.002,inputs=case["inputs"])
        system.SaveAs(str(output.with_suffix(".zmx")))
        save(output,{"success":True,"case_id":3,"result":result})
    except Exception:
        save(output,{"success":False,"error":traceback.format_exc()})
    finally:
        if app is not None:
            app.CloseApplication()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--worker",type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output=args.output.resolve()
    args.output.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parent
    frozen=json.loads((root/"opticstudio_workflow_freeze_v1.json").read_text())["sha256"]
    original={str(path):sha256(path.read_bytes()).hexdigest()
              for path in (root/"p1_random_3_v1").rglob("*") if path.is_file()}
    original.update({str(root/path):value for path,value in frozen.items()})
    waves=[]
    for count in (1,2,4,8):
        folder=args.output/f"concurrency_{count}"
        folder.mkdir()
        processes=[]
        logs=[]
        started=perf_counter()
        minimum_memory=available_memory()
        try:
            # Stagger native startups, then synchronize actual optimization.
            for index in range(count):
                path=folder/f"worker_{index+1}.json"
                log=(folder/f"worker_{index+1}.log").open("w")
                logs.append(log)
                process=subprocess.Popen(
                    [sys.executable,"-u","-B",__file__,str(args.installation),
                     str(args.output),"--worker",str(path)],stdout=log,stderr=subprocess.STDOUT)
                processes.append(process)
                deadline=perf_counter()+120
                while not path.with_suffix(".ready").exists():
                    minimum_memory=min(minimum_memory,available_memory())
                    if process.poll() is not None or perf_counter()>deadline:
                        raise RuntimeError(f"Worker {index+1} did not become ready")
                    sleep(0.2)
            barrier_time=perf_counter()
            (folder/"go").write_text("go")
            deadline=perf_counter()+600
            while any(process.poll() is None for process in processes):
                minimum_memory=min(minimum_memory,available_memory())
                if perf_counter()>deadline:
                    raise TimeoutError("Load test timeout")
                sleep(0.2)
            results=[json.loads((folder/f"worker_{i+1}.json").read_text()) for i in range(count)]
            wave={"concurrency":count,"wall_seconds":perf_counter()-started,
                  "barrier_to_all_closed_seconds":perf_counter()-barrier_time,
                  "minimum_available_memory_gib":minimum_memory/1024**3,"workers":results}
        except Exception:
            wave={"concurrency":count,"error":traceback.format_exc(),
                  "minimum_available_memory_gib":minimum_memory/1024**3}
        finally:
            for process in processes:
                if process.poll() is None:
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
            for log in logs:
                log.close()
        waves.append(wave)
        save(args.output/"benchmark.json",{"waves":waves})
        print(json.dumps({k:v for k,v in wave.items() if k!="workers"}),flush=True)
        if "error" in wave:
            break
    changed=[path for path,value in original.items() if sha256(Path(path).read_bytes()).hexdigest()!=value]
    save(args.output/"benchmark.json",{"waves":waves,"historical_files_changed":changed})
    if changed:
        raise RuntimeError(f"Historical files changed: {changed}")


if __name__=="__main__":
    main()
