"""Read-only integrity audit for the paused third batch; emits no optical metrics."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from background_batch import validate_group

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "background_runs/batch_1500_validation_20261118"
FREEZE = ROOT / "third_batch_validation_freeze_v1"
OUT = RUN / "integrity_audit_paused_20261119.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    state, plan = read(RUN / "state.json"), read(RUN / "plan.json")
    freeze, receipt = read(FREEZE / "freeze_manifest.json"), read(FREEZE / "freeze_receipt.json")
    if state["status"] != "stopped" or state.get("reason") != "user_stop":
        raise RuntimeError("Audit expects a safely paused user-stop state")
    if plan["dataset_sha256"] != digest(Path(plan["dataset_path"])):
        raise RuntimeError("Candidate dataset changed")
    frozen_mismatches = []
    for relative, expected in freeze["artifact_sha256"].items():
        actual = digest(ROOT / relative)
        if actual != expected:
            frozen_mismatches.append({"path": relative, "expected": expected, "actual": actual})
    for name, expected in receipt["artifact_sha256"].items():
        actual = digest(FREEZE / name)
        if actual != expected:
            frozen_mismatches.append({"path": str((FREEZE / name).relative_to(ROOT)),
                                      "expected": expected, "actual": actual})

    completed_ok, completed_bad, attempt_paths = 0, [], set()
    identities = set()
    for key, info in state["jobs"].items():
        identity = (int(info["case_id"]), int(info["rule"]))
        if identity in identities:
            raise RuntimeError(f"Duplicate job identity: {identity}")
        identities.add(identity)
        if info["status"] != "completed":
            continue
        attempt = RUN / info["attempt"]
        if str(attempt.resolve()) in attempt_paths:
            raise RuntimeError("Two completed jobs point to the same attempt")
        attempt_paths.add(str(attempt.resolve()))
        try:
            verified = validate_group(attempt, identity[0], identity[1])
            acceptance = info["acceptance"]
            if verified["result_sha256"] != acceptance["result_sha256"]:
                raise RuntimeError("Result hash differs from accepted state")
            if verified["lens_sha256"] != acceptance["lens_sha256"]:
                raise RuntimeError("Lens hash differs from accepted state")
            completed_ok += 1
        except Exception as error:
            completed_bad.append({"job": key, "error": str(error)})

    strict, partial, untouched, interrupted = [], [], [], []
    for case_id in range(201, 701):
        rows = [state["jobs"][f"case_{case_id:03d}_rule_{rule}"] for rule in (1, 2, 3)]
        done = [int(row["rule"]) for row in rows if row["status"] == "completed"]
        if len(done) == 3:
            strict.append(case_id)
        elif done:
            partial.append({"case_id": case_id, "completed_rules": done,
                            "remaining_statuses": {str(row["rule"]): row["status"] for row in rows
                                                   if row["status"] != "completed"}})
        elif all(row["status"] == "pending" for row in rows):
            untouched.append(case_id)
        if any(row["status"] == "interrupted" for row in rows):
            interrupted.append(case_id)

    report = {
        "audit_type": "read-only integrity; validation outcomes embargoed",
        "run_status": state["status"], "stop_reason": state.get("reason"),
        "expected_jobs": 1500, "registered_jobs": len(state["jobs"]),
        "unique_case_rule_identities": len(identities),
        "job_status_counts": dict(Counter(row["status"] for row in state["jobs"].values())),
        "completed_groups_revalidated": completed_ok,
        "completed_group_integrity_failures": completed_bad,
        "strict_complete_prescriptions": len(strict),
        "strict_complete_case_range": [min(strict), max(strict)] if strict else None,
        "partial_prescriptions": partial,
        "untouched_prescriptions": len(untouched),
        "interrupted_by_user_stop_prescriptions": sorted(set(interrupted)),
        "frozen_artifact_mismatches": frozen_mismatches,
        "candidate_dataset_hash_match": True,
        "prediction_rows_frozen": receipt["prediction_rows"],
        "decision_rows_frozen": receipt["decision_rows"],
        "optical_metrics_reported": False,
        "model_validation_performed": False,
        "safe_to_resume": not completed_bad and not frozen_mismatches and len(identities) == 1500,
    }
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
