"""Run three randomized cases against three frozen continuation policies."""

import argparse
from hashlib import sha256
import json
from math import isclose, sqrt
from pathlib import Path
from time import perf_counter
import traceback

from analyze_opticstudio import WAVELENGTH_UM
from p1_engine import (
    START_F, START_FIELD, TARGET_F, TARGET_FIELD,
    attempt, configure_merit, inspect_prescription, load_seed,
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_case(zos, system, case):
    system.New(False)
    entity = case["entity"]
    data = system.SystemData
    data.Wavelengths.GetWavelength(1).Wavelength = WAVELENGTH_UM
    data.Aperture.ApertureType = zos.SystemData.ZemaxApertureType.ParaxialWorkingFNum
    data.Aperture.ApertureValue = START_F
    data.Fields.SetFieldType(zos.SystemData.FieldType.Angle)
    data.Fields.GetField(1).X = 0.0
    data.Fields.GetField(1).Y = 0.0
    for angle in (0.707, 1.0):
        data.Fields.AddField(0.0, angle, 1.0)
    lde = system.LDE
    for index in (2, 3, 4, 5):
        lde.InsertNewSurfaceAt(index)
    stop = lde.GetSurfaceAt(1)
    stop.IsStop = True
    stop.Comment = f"case_{case['id']:03d}_entrance_stop"
    entrance_gap = entity["stop_to_first_vertex_mm"]
    lde.GetSurfaceAt(0).Thickness = entity["object_to_first_vertex_mm"] - entrance_gap
    stop.Thickness = entrance_gap
    thicknesses = entity["center_thicknesses_mm"]
    distances = (thicknesses[0], entity["air_gap_mm"], thicknesses[1],
                 entity["last_vertex_to_image_mm"])
    for index, (radius, distance) in enumerate(zip(entity["radii_mm"], distances), 2):
        surface = lde.GetSurfaceAt(index)
        surface.Comment = f"case_{case['id']:03d}_surface_{index}"
        surface.Radius = radius
        surface.Thickness = distance
        surface.SemiDiameter = entity["semi_apertures_mm"][(index-2)//2]
        if index in (2, 4):
            solve = surface.MaterialCell.CreateSolveType(zos.Editors.SolveType.MaterialModel)
            model = solve._S_MaterialModel
            model.IndexNd = entity["index"]
            model.AbbeVd = 60.0
            model.dPgF = 0.0
            surface.MaterialCell.SetSolveData(solve)


def assert_case(zos, system, case):
    inputs = case["inputs"]
    check = inspect_prescription(system, inputs)
    if not check["geometry_valid"]:
        raise RuntimeError(f"Seed geometry invalid: {check}")
    if max(abs(check[name]) for name in (
        "object_image_b_mm", "magnification_error_at_image", "exit_pupil_b_mm",
    )) > 1e-8:
        raise RuntimeError(f"Seed conjugates differ from generator: {check}")
    if system.LDE.NumberOfSurfaces != 7 or system.SystemData.Fields.NumberOfFields != 3:
        raise RuntimeError("Unexpected surface or field count")
    if not system.LDE.GetSurfaceAt(1).IsStop:
        raise RuntimeError("Wrong stop surface")
    configure_merit(zos, system, START_F, 0.002, inputs)
    editor = system.MFE
    actual = {}
    for index in range(1, editor.NumberOfOperands + 1):
        operand = editor.GetOperandAt(index)
        if str(operand.Type) in ("PMAG", "EXPP"):
            actual[str(operand.Type)] = operand.Target
    expected = {"PMAG": inputs[4], "EXPP": inputs[3]-inputs[1]}
    if set(actual) != set(expected) or any(
        not isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12)
        for key, value in expected.items()
    ):
        raise RuntimeError(f"Wrong case targets: {actual}, expected {expected}")
    return {"seed_checks": check, "merit_targets": actual,
            "case_identity_verified": True}


def run_group(zos, system, case, rule, seed_path, directory):
    directory.mkdir()
    load_seed(system, seed_path)
    inputs = case["inputs"]
    audit = assert_case(zos, system, case)
    result_path = directory / "result.json"
    accepted_path = directory / "final_accepted.zmx"
    history = []
    started = perf_counter()
    current = {"aperture": START_F, "field": START_FIELD}
    factors = {"aperture": 0.707, "field": 1.414}
    targets = {"aperture": TARGET_F, "field": TARGET_FIELD}
    stopped = {}

    def persist():
        final = next((row for row in reversed(history) if row["accepted"]), None)
        result = {
            "case_id": case["id"], "rule": rule, "inputs": inputs,
            "seed_sha256": digest(seed_path), "audit": audit,
            "settings": {"dls_cycles_per_step": 50, "cores": 2,
                         "glass_edge_target_mm": 0.002, "hard_clearance_mm": 0.001},
            "target_reached": current == targets and final is not None,
            "stop_reasons": stopped, "final_accepted": final,
            "final_attempt": history[-1] if history else None,
            "cost": {
                "attempts": len(history), "requested_dls_cycles": sum(
                    row["dls_requested_cycles"] for row in history),
                "completed_dls_cycles": sum(row["dls_completed_cycles"] for row in history)
                    if history and all(row["dls_completed_cycles"] is not None for row in history) else None,
                "dls_seconds": sum(row["dls_seconds"] for row in history),
                "wall_seconds": perf_counter()-started,
            },
            "steps": history,
        }
        write_json(result_path, result)
        return result

    def try_step(candidate, stage):
        row = attempt(zos, system, candidate["aperture"], candidate["field"], 50,
                      0.002, inputs=inputs)
        row.update({"case_id": case["id"], "rule": rule, "step": len(history)+1, "stage": stage})
        row["rejection_reasons"] = (
            row["geometry_violations"] +
            ([row["analysis_error"]] if "analysis_error" in row else []) +
            [f"field_{index}_mtf" for index, field in enumerate(row.get("fields", []), 1)
             if not field["meets_threshold"]]
        )
        history.append(row)
        system.SaveAs(str(directory / "last_attempt.zmx"))
        if row["accepted"]:
            system.SaveAs(str(accepted_path))
        print(case["id"], rule, len(history), candidate, row["accepted"],
              row["rejection_reasons"], flush=True)
        persist()
        return row["accepted"]

    def advance(dimension):
        candidate = dict(current)
        proposed = current[dimension]*factors[dimension]
        candidate[dimension] = (max(targets[dimension], proposed) if dimension=="aperture"
                                else min(targets[dimension], proposed))
        if try_step(candidate, dimension):
            current.update(candidate)
            if current[dimension] == targets[dimension]:
                stopped[dimension] = "target_reached"
        else:
            load_seed(system, accepted_path)
            factors[dimension] = sqrt(factors[dimension])
            if ((dimension=="aperture" and factors[dimension]>0.99)
                    or (dimension=="field" and factors[dimension]<1.01)):
                stopped[dimension] = "step_limit"

    try:
        if not try_step(current, "initial"):
            stopped.update(dict.fromkeys(current, "initial_rejected"))
        elif rule == 3:
            while len(stopped)<2:
                for dimension in ("aperture", "field"):
                    if dimension not in stopped:
                        advance(dimension)
        else:
            order = ("aperture","field") if rule==1 else ("field","aperture")
            for dimension in order:
                while dimension not in stopped:
                    advance(dimension)
                if stopped[dimension] != "target_reached":
                    if dimension == order[0]:
                        stopped[order[1]] = "not_started_first_stage_limit"
                    break
    except Exception:
        stopped["exception"] = traceback.format_exc()
        persist()
        raise
    final = persist()
    if final["final_accepted"] is not None:
        load_seed(system, accepted_path)
        verified = inspect_prescription(system, inputs)
        for key in ("object_image_b_mm","magnification_error_at_image","exit_pupil_b_mm"):
            if abs(verified[key]-final["final_accepted"][key])>1e-8:
                raise RuntimeError("Saved prescription and final result disagree")
        final["audit"]["saved_final_verified"] = True
        write_json(result_path, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists")
    historical = list(root.glob("*.py")) + list(root.glob("opticstudio*freeze*.json"))
    for folder in root.glob("opticstudio*"):
        if folder.is_dir():
            historical.extend(path for path in folder.rglob("*") if path.is_file())
    historical.extend((root/"random_candidates_10").rglob("*"))
    original = {str(path.relative_to(root)): digest(path) for path in historical
                if path.is_file() and path.name not in ("p1_engine.py","run_p1.py")}
    source = json.loads((root/"random_candidates_10/candidates.json").read_text())
    cases = sorted((case for case in source["candidates"] if case["passed"]), key=lambda c:c["id"])[:3]
    if [case["id"] for case in cases] != [1,2,3]:
        raise RuntimeError("Expected validated cases 1, 2, 3")
    output.mkdir(parents=True)
    write_json(output/"run_manifest.json", {
        "case_id_mapping": "generator id is treated as case_id",
        "selected_case_ids": [case["id"] for case in cases],
        "sources": {name:digest(root/name) for name in ("run_p1.py","p1_engine.py","analyze_opticstudio.py")},
        "historical_sha256_before": original,
    })
    import clr

    for assembly in ("ZOSAPI_Interfaces","ZOSAPI"):
        clr.AddReference(str(args.installation.resolve()/f"{assembly}.dll"))
    import ZOSAPI as zos

    run_start = perf_counter()
    results = []
    for case in cases:
        for rule in (1,2,3):
            connection = zos.ZOSAPI_Connection()
            application = connection.CreateNewApplication()
            if application is None:
                raise RuntimeError("Standalone application unavailable")
            try:
                if not application.IsValidLicenseForAPI:
                    raise RuntimeError("Invalid API license")
                system = application.PrimarySystem
                case_dir = output/f"case_{case['id']:03d}"
                case_dir.mkdir(exist_ok=True)
                seed_path = case_dir/"seed.zmx"
                if not seed_path.exists():
                    build_case(zos, system, case)
                    assert_case(zos, system, case)
                    system.SaveAs(str(seed_path))
                results.append(run_group(zos, system, case, rule, seed_path, case_dir/f"rule_{rule}"))
            finally:
                application.CloseApplication()
    changed = [path for path,value in original.items() if digest(root/path)!=value]
    write_json(output/"batch.json", {
        "groups": results, "historical_files_changed": changed,
        "cost": {"groups":len(results), "wall_seconds_including_api_sessions":perf_counter()-run_start,
                 "requested_dls_cycles":sum(r["cost"]["requested_dls_cycles"] for r in results),
                 "dls_seconds":sum(r["cost"]["dls_seconds"] for r in results)},
    })
    if changed:
        raise RuntimeError(f"Historical files changed: {changed}")


if __name__ == "__main__":
    main()
