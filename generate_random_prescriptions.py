"""Generate ten auditable two-lens candidates without optical optimization."""

import argparse
from collections import Counter
from dataclasses import asdict, astuple
from hashlib import sha256
import importlib.metadata
import json
from math import exp, isfinite, log
from pathlib import Path
import platform
from random import Random

import numpy as np
from raytracing import Space, ThickLens

from compare_opticstudio import prescription_matrix, translate
from lens_geometry import check_lens_geometry
from two_lens import InvalidStructure, TwoLensDesign, design_two_lens
from validate_two_lens import validate_two_lens


RANGES = {
    "object_distance_magnitude_mm": (2, 8),
    "image_distance_mm": (1, 5),
    "entrance_pupil_fraction_of_object_distance": (0.35, 0.8),
    "exit_pupil_distance_mm": (20, 100),
    "magnification_magnitude": (1.5, 6),
}
THICKNESSES = (1.0, 1.0)
SEMI_APERTURES = (1.0, 1.0)
CLEARANCE_MM = 0.001


def restore(initial, inputs):
    l1, image, entrance, exit_pupil, magnification = inputs
    def residual(parameters):
        system = prescription_matrix(parameters, THICKNESSES)
        image_matrix = translate(image - 0.5) @ system @ translate(-l1 - 0.5)
        pupil_matrix = translate(exit_pupil - 0.5) @ system @ translate(-entrance - 0.5)
        return np.array([image_matrix[0, 1], image_matrix[0, 0] - magnification,
                         pupil_matrix[0, 1]])

    parameters = np.array(initial, dtype=float)
    for iteration in range(30):
        error = residual(parameters)
        if np.max(np.abs(error)) < 1e-10:
            return parameters, iteration
        steps = np.maximum(np.abs(parameters), 1.0) * 1e-6
        jacobian = np.column_stack([
            (residual(parameters + offset) - residual(parameters - offset)) / (2 * step)
            for offset, step in zip(np.diag(steps), steps)
        ])
        direction = np.linalg.solve(jacobian, error)
        for reduction in range(16):
            trial = parameters - direction / 2**reduction
            if trial[0] > 0 and np.linalg.norm(residual(trial)) < np.linalg.norm(error):
                parameters = trial
                break
        else:
            raise RuntimeError("conjugate_line_search_failed")
    raise RuntimeError("conjugate_iteration_limit")


def independent_entity_check(parameters, inputs):
    gap, c1, c2 = parameters
    l1, image, entrance, exit_pupil, magnification = inputs
    system = (ThickLens(n=1.5, R1=1/c2, R2=-1/c2, thickness=1)
              * Space(d=gap)
              * ThickLens(n=1.5, R1=1/c1, R2=-1/c1, thickness=1))
    im = Space(d=image - 0.5) * system * Space(d=-l1 - 0.5)
    pupil = Space(d=exit_pupil - 0.5) * system * Space(d=-entrance - 0.5)
    errors = {"object_image_b_mm": im.B, "magnification_error": im.A-magnification,
              "pupil_b_mm": pupil.B}
    limits = {"object_image_b_mm": 1e-8, "magnification_error": 1e-8, "pupil_b_mm": 1e-8}
    return {"residuals": errors, "tolerances": limits,
            "passed": all(isfinite(v) and abs(v) <= limits[k] for k,v in errors.items()),
            "efl_mm": system.effectiveFocalLengths().f2}


def evaluate(inputs):
    record = {"inputs": inputs, "passed": False, "failed_stage": None, "failure_reasons": []}
    stage = "theory"
    try:
        thin = design_two_lens(*inputs)
        check = validate_two_lens(inputs, astuple(thin))
        record["theory"] = {"solution": asdict(thin), "verification": asdict(check),
                            "passed": check.passed}
        if not check.passed:
            raise RuntimeError("independent_thin_lens_check_failed")
        stage = "initial_geometry"
        geometry = check_lens_geometry(thin, THICKNESSES, SEMI_APERTURES,
                                       minimum_clearance_mm=CLEARANCE_MM)
        record["initial_geometry"] = asdict(geometry)
        if not geometry.passed:
            record["failure_reasons"] = list(geometry.violations)
            raise RuntimeError("initial_geometry_failed")
        stage = "conjugate_restoration"
        parameters, iterations = restore((geometry.axial_air_gap_mm, thin.k1, thin.k2), inputs)
        gap,c1,c2 = map(float, parameters)
        stage = "restored_conjugates"
        entity_check = independent_entity_check(parameters, inputs)
        record["entity"] = {
            "radii_mm": [1/c1,-1/c1,1/c2,-1/c2], "center_thicknesses_mm": THICKNESSES,
            "air_gap_mm": gap, "object_to_first_vertex_mm": -inputs[0]-0.5,
            "last_vertex_to_image_mm": inputs[1]-0.5,
            "stop_to_first_vertex_mm": -inputs[2]-0.5,
            "last_vertex_to_exit_pupil_mm": inputs[3]-0.5,
            "semi_apertures_mm": SEMI_APERTURES, "index": 1.5,
            "restoration_iterations": iterations, "verification": entity_check,
        }
        if not entity_check["passed"]:
            raise RuntimeError("independent_thick_lens_check_failed")
        stage = "restored_geometry"
        restored = TwoLensDesign(gap+1, c1, c2, entity_check["efl_mm"])
        geometry = check_lens_geometry(restored, THICKNESSES, SEMI_APERTURES,
                                       minimum_clearance_mm=CLEARANCE_MM)
        record["restored_geometry"] = asdict(geometry)
        if not geometry.passed:
            record["failure_reasons"] = list(geometry.violations)
            raise RuntimeError("restored_geometry_failed")
        stage = "stop_placement"
        if not 0 < -inputs[2]-0.5 < -inputs[0]-0.5:
            raise RuntimeError("entrance_stop_outside_object_space")
        record["passed"] = True
    except (InvalidStructure, ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        record["failed_stage"] = stage
        record["failure_reasons"] = record["failure_reasons"] or [getattr(error, "reason", str(error))]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    manifest = Path(__file__).with_name("opticstudio_workflow_freeze_v1.json")
    frozen = json.loads(manifest.read_text())["sha256"]
    def verify():
        for path,digest in frozen.items():
            if sha256((manifest.parent/path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen file changed: {path}")
    verify()
    rng = Random(args.seed)
    def draw(name):
        low,high = RANGES[name]
        return exp(rng.uniform(log(low),log(high)))
    candidates = []
    for number in range(1,11):
        object_distance = draw("object_distance_magnitude_mm")
        image = draw("image_distance_mm")
        entrance = object_distance * draw("entrance_pupil_fraction_of_object_distance")
        exit_pupil = draw("exit_pupil_distance_mm")
        m = draw("magnification_magnitude")
        candidates.append({"id": number, **evaluate((-object_distance,image,-entrance,exit_pupil,-m))})
    success = sum(row["passed"] for row in candidates)
    sources = ("generate_random_prescriptions.py","two_lens.py","validate_two_lens.py",
               "lens_geometry.py","compare_opticstudio.py","analyze_opticstudio.py")
    report = {
        "seed": args.seed, "count": 10, "sampling": "independent log-uniform draws, no replacement of failures",
        "scope": "finite real object and image, negative magnification, entrance stop before lens 1",
        "sampling_ranges": RANGES,
        "assumptions": {"reference_planes": "lens vertex midpoints", "index": 1.5,
                        "center_thicknesses_mm": THICKNESSES, "semi_apertures_mm": SEMI_APERTURES,
                        "minimum_clearance_mm": CLEARANCE_MM},
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy","raytracing")}},
        "source_sha256": {name: sha256((manifest.parent/name).read_bytes()).hexdigest() for name in sources},
        "summary": {"success_count": success, "failure_count": 10-success, "success_rate": success/10,
                    "failures_by_stage": dict(Counter(row["failed_stage"] for row in candidates if not row["passed"])),
                    "failure_reasons": dict(Counter(reason for row in candidates for reason in row["failure_reasons"]))},
        "candidates": candidates,
    }
    verify()
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.mkdir(parents=True)
    (args.output/"candidates.json").write_text(payload+"\n",encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    main()
