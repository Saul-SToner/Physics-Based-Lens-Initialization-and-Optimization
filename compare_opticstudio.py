"""Compare fixed image plane, Quick Focus, and restored conjugates in OpticStudio."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_opticstudio import INPUTS, build_system, measure
from lens_geometry import check_lens_geometry
from two_lens import TwoLensDesign


def translate(distance, index=1.0):
    return np.array([[1.0, distance / index], [0.0, 1.0]])


def spherical_element(curvature, thickness):
    # Ray coordinates are (height, refractive index times paraxial angle).
    surface = np.array([[1.0, 0.0], [-0.5 * curvature, 1.0]])
    return surface @ translate(thickness, 1.5) @ surface


def prescription_matrix(parameters, thicknesses):
    gap, c1, c2 = parameters
    t1, t2 = thicknesses
    return spherical_element(c2, t2) @ translate(gap) @ spherical_element(c1, t1)


def restore_conjugates(initial, thicknesses, object_gap, image_gap, entrance_gap, exit_gap):
    def residual(parameters):
        matrix = prescription_matrix(parameters, thicknesses)
        image = translate(image_gap) @ matrix @ translate(object_gap)
        pupil = translate(exit_gap) @ matrix @ translate(entrance_gap)
        return np.array([image[0, 1], image[0, 0] - INPUTS[4], pupil[0, 1]])

    parameters = np.array(initial, dtype=float)
    for _ in range(30):
        error = residual(parameters)
        if np.max(np.abs(error)) < 1e-10:
            return parameters
        steps = np.maximum(np.abs(parameters), 1.0) * 1e-6
        offsets = np.diag(steps)
        jacobian = np.column_stack([
            (residual(parameters + offset) - residual(parameters - offset)) / (2 * step)
            for offset, step in zip(offsets, steps)
        ])
        direction = np.linalg.solve(jacobian, error)
        for reduction in range(16):
            trial = parameters - direction / (2 ** reduction)
            if trial[0] > 0 and np.linalg.norm(residual(trial)) < np.linalg.norm(error):
                parameters = trial
                break
        else:
            raise RuntimeError("Conjugate solve could not reduce the residual")
    raise RuntimeError("Conjugate solve did not converge")


def compare(zos, system, output):
    build_system(zos, system)
    lde = system.LDE
    original_object_gap = lde.GetSurfaceAt(0).Thickness
    first_thickness = lde.GetSurfaceAt(1).Thickness
    second_thickness = lde.GetSurfaceAt(3).Thickness
    thicknesses = (first_thickness, second_thickness)
    entrance_gap = -INPUTS[2] - first_thickness / 2
    exit_gap = INPUTS[3] - second_thickness / 2
    if not 0 < entrance_gap < original_object_gap:
        raise ValueError("The specified entrance pupil is outside the object-to-first-surface space")

    # A real stop at the specified entrance plane fixes the entrance pupil
    # identically in all groups. The exit pupil is determined by the lenses.
    stop = lde.InsertNewSurfaceAt(1)
    stop.Comment = "Stop at specified entrance pupil"
    stop.IsStop = True
    stop.Thickness = entrance_gap
    lde.GetSurfaceAt(0).Thickness = original_object_gap - entrance_gap
    first_front, first_back, second_front, last = (lde.GetSurfaceAt(i) for i in range(2, 6))
    image_gap = last.Thickness
    initial = (first_back.Thickness, 1 / first_front.Radius, 1 / second_front.Radius)

    def record(name, parameters):
        matrix = prescription_matrix(parameters, thicknesses)
        object_to_last = matrix @ translate(original_object_gap)
        pupil_to_last = matrix @ translate(entrance_gap)
        selected_image = translate(last.Thickness) @ object_to_last
        selected_exit = translate(exit_gap) @ pupil_to_last
        paraxial_focus = -object_to_last[0, 1] / object_to_last[1, 1]
        focus_matrix = translate(paraxial_focus) @ object_to_last
        operand = zos.Editors.MFE.MeritOperandType.EFFL
        result = {
            "effl_mm": system.MFE.GetOperandValue(operand, 0, 0, 0, 0, 0, 0, 0, 0),
            "vertex_to_image_mm": last.Thickness,
            "vertex_to_paraxial_focus_mm": paraxial_focus,
            "object_image_b_mm": selected_image[0, 1],
            "magnification_at_paraxial_focus": focus_matrix[0, 0],
            "exit_pupil_b_mm": selected_exit[0, 1],
            "vertex_to_exit_pupil_mm": -pupil_to_last[0, 1] / pupil_to_last[1, 1],
            "first_vertex_to_image_mm": sum(lde.GetSurfaceAt(i).Thickness for i in range(2, 6)),
            **measure(zos, system),
        }
        system.SaveAs(str((output / f"{name}.zmx").resolve()))
        return result

    groups = {"A_initial": record("A_initial", initial)}
    focus = system.Tools.OpenQuickFocus()
    try:
        focus.Criterion = zos.Tools.General.QuickFocusCriterion.SpotSizeRadial
        focus.UseCentroid = True
        focus.RunAndWaitForCompletion()
    finally:
        focus.Close()
    groups["B_quick_focus"] = record("B_quick_focus", initial)

    gap, c1, c2 = restore_conjugates(
        initial, thicknesses, original_object_gap, image_gap, entrance_gap, exit_gap,
    )
    geometry = check_lens_geometry(
        TwoLensDesign(gap + sum(thicknesses) / 2, c1, c2, 0), thicknesses, (1, 1),
    )
    if not geometry.passed:
        raise RuntimeError(f"Restored conjugates fail geometry checks: {geometry.violations}")
    first_front.Radius, first_back.Radius = 1 / c1, -1 / c1
    second_front.Radius, last.Radius = 1 / c2, -1 / c2
    first_back.Thickness, last.Thickness = gap, image_gap
    groups["C_restored_conjugates"] = record("C_restored_conjugates", (gap, c1, c2))
    return {
        "paper_magnification": INPUTS[4],
        "target_image_gap_mm": image_gap,
        "target_exit_pupil_gap_mm": exit_gap,
        "center_thicknesses_mm": thicknesses,
        "clear_semi_apertures_mm": [1, 1],
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
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
        args.output.mkdir(parents=True, exist_ok=True)
        report = compare(zos, application.PrimarySystem, args.output)
        payload = json.dumps(report, indent=2, allow_nan=False)
        (args.output / "comparison.json").write_text(payload + "\n", encoding="utf-8")
        print(payload)
    finally:
        application.CloseApplication()


if __name__ == "__main__":
    main()
