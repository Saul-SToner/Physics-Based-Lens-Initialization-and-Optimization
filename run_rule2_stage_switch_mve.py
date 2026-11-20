"""Run ten frozen Rule-2 stage-switch interventions from accepted historical endpoints."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from time import perf_counter
import traceback

from background_batch import save
from p1_engine import attempt, inspect_prescription, load_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rule2_stage_switch_mve_v1"
RUNS = OUT / "runs"
INSTALLATION = Path("C:/Program Files/Ansys Zemax OpticStudio 2024 R1.00")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def gap(f_number, field):
    return max((f_number - 20.0) / 20.0, 0.0) + max((5.0 - field) / 5.0, 0.0)


def historical(case_id, rule):
    batch = ROOT / ("background_runs/batch_300_20261118" if case_id <= 100 else
                    "background_runs/batch_300_20261118_02")
    state = read(batch / "state.json")
    info = state["jobs"][f"case_{case_id:03d}_rule_{rule}"]
    path = batch / info["attempt"] / "experiment/result.json"
    if digest(path) != info["acceptance"]["result_sha256"]:
        raise RuntimeError("Historical result hash mismatch")
    return read(path), path


def control_row(case_id, rule, result):
    final = result["final_accepted"]
    return {"case_id": case_id, "method": f"historical_rule_{rule}",
            "final_f_number": final["working_f_number"],
            "final_field_deg": final["max_field_deg"],
            "target_gap": gap(final["working_f_number"], final["max_field_deg"]),
            "target_reached": result["target_reached"],
            "requested_dls_cycles": result["cost"]["requested_dls_cycles"],
            "dls_calls": len(result["steps"]), "dls_seconds": result["cost"]["dls_seconds"],
            "wall_seconds": result["cost"]["wall_seconds"],
            "final_metrics": final}


def run_case(zos, app, case_id, audit_row, case_dir):
    original_r2, original_path = historical(case_id, 2)
    original_r3, _ = historical(case_id, 3)
    baseline = original_r2["final_accepted"]
    inputs = original_r2["inputs"]
    original_lens = ROOT / audit_row["final_lens_path"]
    if digest(original_lens) != read(OUT / "protocol_freeze.json")["historical_result_and_lens_sha256"][audit_row["final_lens_path"]]:
        raise RuntimeError("Historical Rule-2 lens changed")
    case_dir.mkdir(parents=True)
    system = app.PrimarySystem
    load_seed(system, original_lens)
    check = inspect_prescription(system, inputs)
    for key in ("object_image_b_mm", "magnification_error_at_image", "exit_pupil_b_mm"):
        if abs(check[key] - baseline[key]) > 1e-8:
            raise RuntimeError(f"Loaded baseline differs: {key}")
    accepted_path = case_dir / "final_accepted.zmx"
    system.SaveAs(str(accepted_path))
    baseline_copy_hash = digest(accepted_path)
    current_f = float(baseline["working_f_number"])
    field = float(baseline["max_field_deg"])
    factor, steps, stop_reason = 0.707, [], None
    started = perf_counter()

    def checkpoint():
        final = next((x for x in reversed(steps) if x["accepted"]), baseline)
        save(case_dir / "result.json", {
            "case_id": case_id, "intervention": "Rule2 field-limit then aperture continuation",
            "trigger_verified": original_r2["stop_reasons"] == {
                "aperture": "not_started_first_stage_limit", "field": "step_limit"},
            "historical_rule_2_result": str(original_path.relative_to(ROOT)),
            "historical_rule_2_result_sha256": digest(original_path),
            "baseline_lens_sha256": baseline_copy_hash,
            "settings": {"dls_cycles_per_step": 50, "cores": 2,
                         "glass_edge_target_mm": 0.002, "hard_clearance_mm": 0.001,
                         "field_fixed_deg": field},
            "stop_reason": stop_reason, "steps": steps, "final_accepted": final,
            "incremental_cost": {"dls_calls": len(steps),
                                 "requested_dls_cycles": 50 * len(steps),
                                 "dls_seconds": sum(x["dls_seconds"] for x in steps),
                                 "wall_seconds": perf_counter() - started},
        })

    try:
        while stop_reason is None:
            candidate_f = max(20.0, current_f * factor)
            row = attempt(zos, system, candidate_f, field, 50, 0.002, inputs=inputs)
            row.update({"case_id": case_id, "stage": "aperture_after_field_limit",
                        "step": len(steps) + 1, "candidate_f_number": candidate_f})
            row["rejection_reasons"] = (row["geometry_violations"] +
                ([row["analysis_error"]] if "analysis_error" in row else []) +
                [f"field_{i}_mtf" for i, value in enumerate(row.get("fields", []), 1)
                 if not value["meets_threshold"]])
            steps.append(row)
            system.SaveAs(str(case_dir / "last_attempt.zmx"))
            if row["accepted"]:
                current_f = candidate_f
                system.SaveAs(str(accepted_path))
                if current_f == 20.0:
                    stop_reason = "target_reached"
            else:
                load_seed(system, accepted_path)
                factor = sqrt(factor)
                if factor > 0.99:
                    stop_reason = "step_limit"
            checkpoint()
    except Exception:
        stop_reason = "exception"
        checkpoint()
        raise
    load_seed(system, accepted_path)
    verified = inspect_prescription(system, inputs)
    final_result = read(case_dir / "result.json")
    final = final_result["final_accepted"]
    for key in ("object_image_b_mm", "magnification_error_at_image", "exit_pupil_b_mm"):
        if abs(verified[key] - final[key]) > 1e-8:
            raise RuntimeError("Saved intervention endpoint verification failed")
    intervention = {"case_id": case_id, "method": "rule_2_stage_switch",
                    "final_f_number": final["working_f_number"],
                    "final_field_deg": final["max_field_deg"],
                    "target_gap": gap(final["working_f_number"], final["max_field_deg"]),
                    "target_reached": final["working_f_number"] == 20.0 and field == 5.0,
                    "requested_dls_cycles": original_r2["cost"]["requested_dls_cycles"] + 50 * len(steps),
                    "dls_calls": len(original_r2["steps"]) + len(steps),
                    "dls_seconds": original_r2["cost"]["dls_seconds"] + sum(x["dls_seconds"] for x in steps),
                    "wall_seconds": original_r2["cost"]["wall_seconds"] + final_result["incremental_cost"]["wall_seconds"],
                    "incremental_dls_calls": len(steps),
                    "incremental_wall_seconds": final_result["incremental_cost"]["wall_seconds"],
                    "stop_reason": stop_reason, "final_metrics": final}
    return [control_row(case_id, 2, original_r2), control_row(case_id, 3, original_r3), intervention]


def main():
    if RUNS.exists():
        raise RuntimeError("Run directory already exists")
    RUNS.mkdir()
    protocol = read(OUT / "protocol_freeze.json")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    selected = read(OUT / "selected_cases.json")
    audit = {x["case_id"]: x for x in read(OUT / "mechanism_audit.json")["cases"]}
    import clr
    for assembly in ("ZOSAPI_Interfaces", "ZOSAPI"):
        clr.AddReference(str(INSTALLATION / f"{assembly}.dll"))
    import ZOSAPI as zos
    all_rows, start = [], perf_counter()
    for number, selection in enumerate(selected, 1):
        case_id = selection["case_id"]
        connection = zos.ZOSAPI_Connection()
        app = connection.CreateNewApplication()
        if app is None or not app.IsValidLicenseForAPI:
            raise RuntimeError("Standalone OpticStudio unavailable")
        try:
            rows = run_case(zos, app, case_id, audit[case_id], RUNS / f"case_{case_id:03d}")
            all_rows.extend(rows)
        finally:
            app.CloseApplication()
        save(OUT / "progress.json", {"completed_cases": number, "total_cases": 10,
                                     "last_case_id": case_id, "wall_seconds": perf_counter() - start})
    save(OUT / "comparison_rows.json", all_rows)
    save(OUT / "run_receipt.json", {"completed_cases": 10, "new_opticstudio_runs": 10,
                                    "comparison_rows": 30, "wall_seconds": perf_counter() - start,
                                    "protocol_sha256": digest(OUT / "protocol_freeze.json"),
                                    "case_201_plus_used": False})
    print(json.dumps(read(OUT / "run_receipt.json"), indent=2))


if __name__ == "__main__":
    main()
