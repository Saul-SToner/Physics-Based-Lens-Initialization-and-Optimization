"""One isolated P1 group for the Ray eight-session acceptance test."""

import argparse
import json
import os
from pathlib import Path
from time import monotonic, sleep
import traceback

from run_p1 import assert_case, run_group, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    directory = args.job.parent
    status = directory/"status.json"
    events = directory/"events.jsonl"

    def report(phase, **extra):
        event = {"case_id": job["case_id"], "rule": job["rule"],
                 "pid": os.getpid(), "phase": phase, "monotonic": monotonic(), **extra}
        write_json(status, event)
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event)+"\n")

    application = None
    try:
        report("opening")
        import clr
        for assembly in ("ZOSAPI_Interfaces", "ZOSAPI"):
            clr.AddReference(str(args.installation.resolve()/f"{assembly}.dll"))
        import ZOSAPI as zos

        connection = zos.ZOSAPI_Connection()
        application = connection.CreateNewApplication()
        if application is None or not application.IsValidLicenseForAPI:
            raise RuntimeError("No licensed independent OpticStudio API session")
        seed = directory/"seed.zmx"
        if not application.PrimarySystem.LoadFile(str(seed), False):
            raise RuntimeError("Cannot load independent seed copy")
        checks = assert_case(zos, application.PrimarySystem, job["case"])
        report("ready", identity_checks=checks)
        deadline = monotonic()+240
        release = Path(job["release"])
        stop = Path(job["stop"])
        while not release.exists():
            if stop.exists():
                raise RuntimeError("Admission stopped before workload release")
            if monotonic()>deadline:
                raise TimeoutError("Eight-worker barrier timed out")
            sleep(0.1)
        if stop.exists():
            raise RuntimeError("Admission stopped")
        report("running")
        result = run_group(zos, application.PrimarySystem, job["case"], job["rule"],
                           seed, directory/"experiment")
        report("completed", target_reached=result["target_reached"])
    except Exception:
        report("failed", error=traceback.format_exc())
        raise
    finally:
        if application is not None:
            application.CloseApplication()
        with events.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"phase":"session_closed","pid":os.getpid(),
                                     "monotonic":monotonic()})+"\n")


if __name__ == "__main__":
    main()
