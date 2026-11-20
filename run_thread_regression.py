"""Compare unchanged and single-thread auxiliary libraries on P1 case 2/rule 2."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep

import psutil


def main():
    root=Path(__file__).resolve().parent
    output=root/"ray_thread_regression"
    output.mkdir(exist_ok=False)
    installation="C:/Program Files/Ansys Zemax OpticStudio 2024 R1.00"
    case=json.loads((root/"random_candidates_10/candidates.json").read_text())["candidates"][1]
    reference=next(row for row in json.loads((root/"p1_random_3_v1/batch.json").read_text())["groups"]
                   if row["case_id"]==2 and row["rule"]==2)
    results={}
    for version,script in (("control","ray_stress_worker.py"),("limited","ray_stress_worker_v2.py")):
        directory=output/version
        directory.mkdir()
        shutil.copyfile(root/"p1_random_3_v1/case_002/seed.zmx",directory/"seed.zmx")
        (directory/"RELEASE").write_text("single group")
        job={"case_id":2,"rule":2,"case":case,"release":str(directory/"RELEASE"),
             "stop":str(directory/"STOP_ADMISSION")}
        (directory/"job.json").write_text(json.dumps(job))
        started=monotonic()
        samples=[]
        with (directory/"worker.log").open("w") as log:
            process=subprocess.Popen([sys.executable,"-u","-B",str(root/script),installation,
                                      str(directory/"job.json")],stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                free=psutil.virtual_memory().available
                samples.append(free)
                if free<1024**3:
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
                    raise RuntimeError("Single-group regression stopped at 1 GiB free memory")
                sleep(0.5)
        if process.returncode:
            raise RuntimeError(f"{version} failed, see worker.log")
        result=json.loads((directory/"experiment/result.json").read_text())
        results[version]={"subprocess_wall_seconds":monotonic()-started,
                          "minimum_available_gib":min(samples)/1024**3,"group":result}
        print(version,"completed",results[version]["subprocess_wall_seconds"],flush=True)
    comparisons={}
    for name,result in results.items():
        actual=result["group"]
        errors={}
        for key,tolerance in (("working_f_number",1e-9),("max_field_deg",1e-9),
                              ("object_image_b_mm",1e-6),("magnification_error_at_image",1e-6),
                              ("exit_pupil_b_mm",1e-6)):
            difference=abs(actual["final_accepted"][key]-reference["final_accepted"][key])
            errors[key]={"difference":difference,"tolerance":tolerance}
        for index,(field,expected) in enumerate(zip(actual["final_accepted"]["fields"],
                                                  reference["final_accepted"]["fields"]),1):
            for key,tolerance in (("mtf_tangential",1e-6),("mtf_sagittal",1e-6),
                                  ("rms_spot_radius",1e-3)):
                errors[f"field_{index}_{key}"]={"difference":abs(field[key]-expected[key]),
                                              "tolerance":tolerance}
        def steps(row):
            return [(step["stage"],step["working_f_number"],step["max_field_deg"],step["accepted"])
                    for step in row["steps"]]
        comparisons[name]={"numeric":errors,"same_step_sequence":steps(actual)==steps(reference),
                           "same_stop_reasons":actual["stop_reasons"]==reference["stop_reasons"]}
        comparisons[name]["passed"]=(all(e["difference"]<=e["tolerance"] for e in errors.values())
                                     and comparisons[name]["same_step_sequence"]
                                     and comparisons[name]["same_stop_reasons"])
    freeze=json.loads((root/"opticstudio_workflow_freeze_v1.json").read_text())["sha256"]
    changed=[name for name,value in freeze.items() if sha256((root/name).read_bytes()).hexdigest()!=value]
    payload={"passed":all(value["passed"] for value in comparisons.values()) and not changed,
             "case_id":2,"rule":2,"comparisons":comparisons,"results":results,"historical_changed":changed}
    (output/"regression.json").write_text(json.dumps(payload,indent=2)+"\n")
    print("regression passed",payload["passed"],flush=True)


if __name__=="__main__":
    main()
