"""Isolated model worker for resumable unattended groups."""

import os
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
if "numpy" in sys.modules:
    raise RuntimeError("NumPy loaded before numerical thread limits")

import argparse
import json
from pathlib import Path
from time import monotonic, sleep
import traceback

import run_p1
from run_p1 import assert_case, build_case, run_group
from background_batch import save as write_json

# Route this isolated process's checkpoint I/O through the Windows-safe writer.
# The frozen optical calculations, optimizer settings and rule logic are reused.
run_p1.write_json = write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    directory = args.job.parent
    app = None
    def status(phase, **values):
        write_json(directory/"model_status.json", {
            "phase": phase, "pid": os.getpid(), "monotonic": monotonic(), **values,
        })
    try:
        status("opening")
        import clr
        for name in ("ZOSAPI_Interfaces", "ZOSAPI"):
            clr.AddReference(str(Path(job["installation"])/(name+".dll")))
        import ZOSAPI as zos
        connection = zos.ZOSAPI_Connection()
        app = connection.CreateNewApplication()
        if app is None or not app.IsValidLicenseForAPI:
            raise RuntimeError("No licensed OpticStudio application")
        build_case(zos, app.PrimarySystem, job["case"])
        assert_case(zos, app.PrimarySystem, job["case"])
        seed = directory/"seed.zmx"
        app.PrimarySystem.SaveAs(str(seed))
        status("ready")
        deadline = monotonic()+job["startup_timeout_seconds"]
        while not Path(job["release_file"]).exists():
            if Path(job["stop_file"]).exists():
                raise RuntimeError("Stopped before release")
            if monotonic()>deadline:
                raise TimeoutError("Startup barrier timeout")
            sleep(0.2)
        status("running")
        result = run_group(zos, app.PrimarySystem, job["case"], job["rule"], seed,
                           directory/"experiment")
        status("completed", target_reached=result["target_reached"])
    except Exception:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.CloseApplication()


if __name__ == "__main__":
    main()
