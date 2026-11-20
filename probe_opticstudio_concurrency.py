"""Probe independent simultaneous OpticStudio API sessions and release them."""

import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import perf_counter


def worker(installation):
    import clr
    for name in ("ZOSAPI_Interfaces", "ZOSAPI"):
        clr.AddReference(str(Path(installation)/(name+".dll")))
    import ZOSAPI as zos
    connection = zos.ZOSAPI_Connection()
    application = None
    try:
        application = connection.CreateNewApplication()
        if application is None or not application.IsValidLicenseForAPI:
            raise RuntimeError("No licensed API application")
        print(json.dumps({"ready": True, "license": str(application.LicenseStatus),
                          "subscription": application.IsSubscriptionLicense}), flush=True)
        for line in sys.stdin:
            if line.strip() == "close":
                break
            if line.strip() == "ping":
                print(json.dumps({"alive": application.IsValidLicenseForAPI,
                                  "surfaces": application.PrimarySystem.LDE.NumberOfSurfaces}), flush=True)
    except Exception as error:
        print(json.dumps({"ready": False, "error": str(error)}), flush=True)
    finally:
        if application is not None:
            application.CloseApplication()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation")
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.installation)
        return
    processes = []
    report = {"attempts": [], "licensed_simultaneous_sessions": 0}
    def read_lines(process, mailbox):
        for line in process.stdout:
            try:
                mailbox.put(json.loads(line))
            except json.JSONDecodeError:
                pass
    try:
        for number in range(1, 17):
            start = perf_counter()
            process = subprocess.Popen(
                [sys.executable, "-u", "-B", __file__, args.installation, str(args.output), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            mailbox = queue.Queue()
            threading.Thread(target=read_lines, args=(process, mailbox), daemon=True).start()
            processes.append((process, mailbox))
            try:
                result = mailbox.get(timeout=90)
            except queue.Empty:
                result = {"ready": False, "error": "startup_timeout_90_seconds"}
            result.update({"instance_attempt": number, "startup_seconds": perf_counter()-start})
            report["attempts"].append(result)
            print(json.dumps(result), flush=True)
            if not result.get("ready"):
                break
            report["licensed_simultaneous_sessions"] = number
        report["held_session_checks"] = []
        for process,mailbox in processes[:report["licensed_simultaneous_sessions"]]:
            process.stdin.write("ping\n")
            process.stdin.flush()
            report["held_session_checks"].append(mailbox.get(timeout=30))
    finally:
        for process,_ in processes:
            if process.poll() is None:
                try:
                    process.stdin.write("close\n")
                    process.stdin.flush()
                    process.wait(timeout=30)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
        report["all_probe_processes_closed"] = all(p.poll() is not None for p,_ in processes)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
