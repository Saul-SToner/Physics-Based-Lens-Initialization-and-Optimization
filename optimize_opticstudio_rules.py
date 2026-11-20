"""Compare the paper's aperture-first and field-first continuation rules."""

import argparse
import json
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from analyze_opticstudio import INPUTS, MTF_REQUIREMENTS, WAVELENGTH_UM, measure
from compare_opticstudio import translate
from lens_geometry import _sag


START_F = 500.0
START_FIELD = 1.0
TARGET_F = 20.0
TARGET_FIELD = 5.0
SEMI_APERTURE_MM = 1.0
MIN_CLEARANCE_MM = 1e-3


def add_operand(zos, editor, name, target, weight, integers=(), doubles=()):
    operand = editor.AddOperand()
    operand.ChangeType(getattr(zos.Editors.MFE.MeritOperandType, name))
    operand.Target = target
    operand.Weight = weight
    for index, value in integers:
        column = getattr(zos.Editors.MFE.MeritColumn, f"Param{index}")
        operand.GetOperandCell(column).IntegerValue = value
    for index, value in doubles:
        column = getattr(zos.Editors.MFE.MeritColumn, f"Param{index}")
        operand.GetOperandCell(column).DoubleValue = value


def configure_merit(zos, system, f_number, glass_edge_target_mm):
    editor = system.MFE
    editor.DeleteAllRows()
    wizard = editor.SEQOptimizationWizard
    wizard.Data = 1
    wizard.Ring = 2
    wizard.OverallWeight = 1.0
    wizard.IsGlassUsed = True
    wizard.GlassMin = MIN_CLEARANCE_MM
    wizard.GlassMax = 1000.0
    wizard.GlassEdge = glass_edge_target_mm
    wizard.IsAirUsed = True
    wizard.AirMin = MIN_CLEARANCE_MM
    wizard.AirMax = 1000.0
    wizard.AirEdge = MIN_CLEARANCE_MM
    wizard.Apply()
    cutoff = 1000 / (WAVELENGTH_UM * f_number)
    # At each frequency use the unobscured diffraction ceiling as the goal.
    for field, ((fraction, _), goal) in enumerate(zip(MTF_REQUIREMENTS, (0.391, 0.470, 0.564)), 1):
        for name in ("MTFT", "MTFS"):
            add_operand(zos, editor, name, goal, 0.1,
                        ((1, 3), (2, 1), (3, field), (5, 1)),
                        ((4, cutoff * fraction),))
    add_operand(zos, editor, "PARY", 0.0, f_number**2, ((1, 6), (2, 1)), ((6, 1.0),))
    add_operand(zos, editor, "PMAG", INPUTS[4], 10.0, ((1, 1),))
    # EXPP is referenced to the image surface rather than the rear vertex.
    add_operand(zos, editor, "EXPP", INPUTS[3] - INPUTS[1], 0.0001)


def inspect_prescription(system):
    lde = system.LDE
    surfaces = [lde.GetSurfaceAt(i) for i in range(2, 6)]
    radii = [surface.Radius or float("inf") for surface in surfaces]
    t1, gap, t2, image_gap = (surface.Thickness for surface in surfaces)
    entrance_gap = lde.GetSurfaceAt(1).Thickness
    object_gap = lde.GetSurfaceAt(0).Thickness + entrance_gap
    pupil_gap = INPUTS[3] - t2 / 2
    matrix = np.eye(2)
    for n0, n1, radius, distance in zip((1, 1.5, 1, 1.5), (1.5, 1, 1.5, 1),
                                      radii, (t1, gap, t2, 0)):
        refraction = np.array([[1, 0], [-(n1 - n0) / radius, 1]])
        matrix = translate(distance, n1) @ refraction @ matrix
    image = translate(image_gap) @ matrix @ translate(object_gap)
    pupil = translate(pupil_gap) @ matrix @ translate(entrance_gap)
    violations = []
    sags = []
    for number, radius in enumerate(radii, 2):
        try:
            sags.append(_sag(radius, SEMI_APERTURE_MM))
        except ValueError:
            sags.append(None)
            violations.append(f"surface_{number}_radius_below_semi_aperture")
    clearances = {"lens_1_center": t1, "axial_air_gap": gap, "lens_2_center": t2}
    for name, distance, front, back in (
        ("lens_1_edge", t1, 0, 1),
        ("edge_air_gap", gap, 1, 2),
        ("lens_2_edge", t2, 2, 3),
    ):
        if sags[front] is not None and sags[back] is not None:
            clearances[name] = distance + sags[back] - sags[front]
    violations.extend(name for name, value in clearances.items()
                      if not isfinite(value) or value < MIN_CLEARANCE_MM)
    return {
        "geometry_valid": not violations,
        "geometry_violations": violations,
        "clearances_mm": clearances,
        "surface_radii_mm": [radius if isfinite(radius) else None for radius in radii],
        "first_vertex_to_image_mm": t1 + gap + t2 + image_gap,
        "object_image_b_mm": float(image[0, 1]),
        "magnification_error_at_image": float(image[0, 0] - INPUTS[4]),
        "exit_pupil_b_mm": float(pupil[0, 1]),
    }


def attempt(zos, system, f_number, field, cycles=5, glass_edge_target_mm=MIN_CLEARANCE_MM):
    system.SystemData.Aperture.ApertureValue = f_number
    angles = (0.0, 0.707 * field, field)
    for number, angle in enumerate(angles, 1):
        system.SystemData.Fields.GetField(number).Y = angle
    configure_merit(zos, system, f_number, glass_edge_target_mm)
    start = perf_counter()
    tool = system.Tools.OpenLocalOptimization()
    try:
        tool.Algorithm = zos.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
        tool.Cycles = getattr(zos.Tools.Optimization.OptimizationCycles, f"Fixed_{cycles}_Cycles")
        tool.NumberOfCores = 2
        tool.RunAndWaitForCompletion()
        merit = tool.CurrentMeritFunction
    finally:
        tool.Close()
    result = {
        "working_f_number": f_number,
        "max_field_deg": field,
        "dls_seconds": perf_counter() - start,
        "final_merit": merit if isfinite(merit) else None,
        **inspect_prescription(system),
    }
    if result["geometry_valid"]:
        try:
            result.update(measure(zos, system, f_number, angles))
        except RuntimeError as error:
            result["analysis_error"] = str(error)
    result["accepted"] = (
        result["geometry_valid"] and len(result.get("fields", [])) == 3
        and all(row["meets_threshold"] for row in result["fields"])
    )
    return result


def load_seed(system, start_file):
    if not system.LoadFile(str(start_file), False):
        raise RuntimeError(f"Could not load {start_file}")
    for index in (2, 3, 4, 5):
        system.LDE.GetSurfaceAt(index).RadiusCell.MakeSolveVariable()
    system.LDE.GetSurfaceAt(3).ThicknessCell.MakeSolveVariable()


def run_rule(zos, system, start_file, destination, order, cycles, glass_edge_target_mm):
    load_seed(system, start_file)
    history = []
    f_number, field = START_F, START_FIELD

    def try_step(candidate_f, candidate_field, stage):
        row = attempt(zos, system, candidate_f, candidate_field, cycles, glass_edge_target_mm)
        row["stage"] = stage
        history.append(row)
        print(destination.stem, stage, round(candidate_f, 4), round(candidate_field, 4),
              "accepted" if row["accepted"] else "rejected",
              ", ".join(row["geometry_violations"]), flush=True)
        if row["accepted"]:
            system.SaveAs(str(destination))
        return row["accepted"]

    if try_step(f_number, field, "initial"):
        for dimension in order:
            aperture = dimension == "aperture"
            factor = 0.707 if aperture else 1.414
            while factor <= 0.99 if aperture else factor >= 1.01:
                candidate_f = max(TARGET_F, f_number * factor) if aperture else f_number
                candidate_field = field if aperture else min(TARGET_FIELD, field * factor)
                if try_step(candidate_f, candidate_field, dimension):
                    f_number, field = candidate_f, candidate_field
                    if (f_number == TARGET_F if aperture else field == TARGET_FIELD):
                        break
                else:
                    system.LoadFile(str(destination), False)
                    factor = sqrt(factor)
            if (f_number != TARGET_F if aperture else field != TARGET_FIELD):
                break
    return {
        "dls_cycles_per_step": cycles,
        "glass_edge_target_mm": glass_edge_target_mm,
        "minimum_clearance_mm": MIN_CLEARANCE_MM,
        "target_reached": f_number == TARGET_F and field == TARGET_FIELD,
        "final_accepted": next((row for row in reversed(history) if row["accepted"]), None),
        "steps": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cycles", type=int, choices=(5, 10, 50), default=5)
    parser.add_argument("--rule", choices=("both", "rule_1", "rule_2"), default="both")
    parser.add_argument("--glass-edge-target-mm", type=float, default=MIN_CLEARANCE_MM)
    args = parser.parse_args()
    if not isfinite(args.glass_edge_target_mm) or args.glass_edge_target_mm < MIN_CLEARANCE_MM:
        parser.error("Glass edge target must be finite and at least the hard minimum clearance")
    import clr

    for assembly in ("ZOSAPI_Interfaces", "ZOSAPI"):
        clr.AddReference(str(args.installation.resolve() / f"{assembly}.dll"))
    import ZOSAPI as zos

    connection = zos.ZOSAPI_Connection()
    application = connection.CreateNewApplication()
    if application is None:
        raise RuntimeError("Could not start an OpticStudio standalone session")
    try:
        if not application.IsValidLicenseForAPI:
            raise RuntimeError(f"ZOS-API license unavailable: {application.LicenseStatus}")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        report = {}
        for name, order in (("rule_1", ("aperture", "field")), ("rule_2", ("field", "aperture"))):
            if args.rule in ("both", name):
                report[name] = run_rule(
                    zos, application.PrimarySystem, args.baseline.resolve(),
                    output / f"{name}.zmx", order, args.cycles, args.glass_edge_target_mm,
                )
        (output / "rules.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    finally:
        application.CloseApplication()


if __name__ == "__main__":
    main()
