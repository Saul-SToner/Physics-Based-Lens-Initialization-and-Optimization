"""Operational-only third-batch report; never reads optical validation metrics."""
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "background_runs/batch_1500_validation_20261118"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    state = read(RUN / "state.json")
    jobs = state["jobs"]
    strict_complete = []
    partial = []
    failed = []
    completed_wall = []
    for cid in range(201, 701):
        group = [jobs[f"case_{cid:03d}_rule_{rule}"] for rule in (1, 2, 3)]
        done = sum(x["status"] == "completed" for x in group)
        if done == 3:
            strict_complete.append(cid)
        elif done:
            partial.append({"case_id": cid, "completed_rules": done})
        if any(x["status"] in ("failed", "interrupted") for x in group):
            failed.append(cid)
        for item in group:
            if item["status"] == "completed" and item.get("acceptance"):
                completed_wall.append(float(item["acceptance"]["cost"]["wall_seconds"]))
    telemetry_files = sorted(RUN.glob("telemetry_*.jsonl"), key=lambda p: p.stat().st_mtime)
    resource = {"samples": 0, "latest_available_bytes": None, "minimum_available_bytes": None,
                "latest_cpu_percent": None, "peak_commit_fraction": None, "peak_active_groups": 0}
    if telemetry_files:
        with telemetry_files[-1].open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                available = sample["available_physical_bytes"]
                counters = sample["system_memory_counters"]
                committed, limit = counters.get("committed_bytes"), counters.get("commit_limit_bytes")
                resource["samples"] += 1
                resource["latest_available_bytes"] = available
                resource["minimum_available_bytes"] = min(resource["minimum_available_bytes"] or available, available)
                resource["latest_cpu_percent"] = sample["system_cpu_percent"]
                resource["peak_active_groups"] = max(resource["peak_active_groups"], sample["active_groups"])
                if committed and limit:
                    fraction = committed / limit
                    resource["peak_commit_fraction"] = max(resource["peak_commit_fraction"] or fraction, fraction)
    completed_groups = sum(x["status"] == "completed" for x in jobs.values())
    elapsed = None
    if telemetry_files:
        first = json.loads(telemetry_files[-1].read_text(encoding="utf-8").splitlines()[0])
        elapsed = time.time() - datetime.fromisoformat(first["utc"]).timestamp()
    group_rate = completed_groups / elapsed if elapsed and elapsed > 0 else None
    remaining_seconds = ((len(jobs) - completed_groups) / group_rate
                         if group_rate and group_rate > 0 else None)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": state["status"],
        "strict_completed_prescriptions": len(strict_complete),
        "definition": "A prescription counts only when Rules 1, 2, and 3 are all completed and accepted by the controller.",
        "completed_groups": completed_groups, "total_groups": len(jobs),
        "partially_completed_prescriptions": len(partial),
        "failed_or_interrupted_prescriptions": len(set(failed)),
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds_from_observed_group_throughput": remaining_seconds,
        "median_completed_group_wall_seconds": median(completed_wall) if completed_wall else None,
        "resources": resource,
        "validation_metrics_read": False,
        "optical_endpoints_read": False,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
