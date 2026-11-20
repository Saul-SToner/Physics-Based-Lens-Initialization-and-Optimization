"""Build and analyze the paper's two-lens example in OpticStudio."""

import argparse
import json
from pathlib import Path

from lens_geometry import check_lens_geometry
from two_lens import design_two_lens


INPUTS = (-2.704, 1.9055, -1.643, 105.436, -3.463)
WAVELENGTH_UM = 0.587
WORKING_F_NUMBER = 20
FIELDS_DEG = (0.0, 0.707, 1.0)
MTF_REQUIREMENTS = ((0.50, 0.30), (0.43, 0.40), (0.35, 0.50))


def build_system(zos, system):
    design = design_two_lens(*INPUTS)
    thicknesses = (1.0, 1.0)
    semi_apertures = (1.0, 1.0)
    geometry = check_lens_geometry(design, thicknesses, semi_apertures)
    if not geometry.passed:
        raise ValueError(geometry.violations)

    data = system.SystemData
    data.Wavelengths.GetWavelength(1).Wavelength = WAVELENGTH_UM
    data.Aperture.ApertureType = zos.SystemData.ZemaxApertureType.ParaxialWorkingFNum
    data.Aperture.ApertureValue = WORKING_F_NUMBER
    data.Fields.SetFieldType(zos.SystemData.FieldType.Angle)
    for angle in FIELDS_DEG[1:]:
        data.Fields.AddField(0.0, angle, 1.0)

    lde = system.LDE
    for index in (2, 3, 4):
        lde.InsertNewSurfaceAt(index)
    lde.GetSurfaceAt(0).Thickness = -INPUTS[0] - thicknesses[0] / 2
    following_distances = (
        thicknesses[0], geometry.axial_air_gap_mm,
        thicknesses[1], INPUTS[1] - thicknesses[1] / 2,
    )
    for index, (radius, distance) in enumerate(zip(design.radii_mm, following_distances), 1):
        surface = lde.GetSurfaceAt(index)
        surface.Comment = f"Lens {(index + 1) // 2} {'front' if index % 2 else 'back'}"
        surface.Radius = radius
        surface.Thickness = distance
        surface.SemiDiameter = semi_apertures[(index - 1) // 2]
        if index % 2:
            cell = surface.MaterialCell
            solve = cell.CreateSolveType(zos.Editors.SolveType.MaterialModel)
            model = solve._S_MaterialModel
            model.IndexNd = 1.5
            model.AbbeVd = 60.0
            model.dPgF = 0.0
            cell.SetSolveData(solve)
    lde.GetSurfaceAt(1).IsStop = True
    return design


def measure(zos, system, working_f_number=WORKING_F_NUMBER, fields_deg=FIELDS_DEG):
    import numpy as np

    spot = system.Analyses.New_StandardSpot()
    try:
        spot.ApplyAndWaitForCompletion()
        spot_data = spot.GetResults().SpotData
        radii = [spot_data.GetRMSSpotSizeFor(i, 1) for i in (1, 2, 3)]
    finally:
        spot.Close()

    cutoff = 1000 / (WAVELENGTH_UM * working_f_number)
    fft = system.Analyses.New_FftMtf()
    try:
        settings = zos.Analysis.Settings.Mtf.IAS_FftMtf(fft.GetSettings())
        settings.MaximumFrequency = cutoff
        settings.SampleSize = zos.Analysis.SampleSizes.S_256x256
        settings.Field.UseAllFields()
        fft.ApplyAndWaitForCompletion()
        results = fft.GetResults()
        if results.NumberOfDataSeries != len(fields_deg):
            raise RuntimeError(f"Expected three FFT MTF series, got {results.NumberOfDataSeries}")
        records = []
        for index, (angle, (fraction, threshold)) in enumerate(zip(fields_deg, MTF_REQUIREMENTS)):
            series = results.GetDataSeries(index)
            x = np.asarray(list(series.XData.Data))
            y = np.asarray(list(series.YData.Data)).reshape(len(x), 2)
            frequency = fraction * cutoff
            if not x[0] <= frequency <= x[-1]:
                raise RuntimeError(f"MTF frequency {frequency} outside sampled range {x[0]}–{x[-1]}")
            tangential, sagittal = (float(np.interp(frequency, x, y[:, col])) for col in (0, 1))
            records.append({
                "field_deg": angle, "rms_spot_radius": radii[index],
                "mtf_frequency_lp_per_mm": frequency, "mtf_tangential": tangential,
                "mtf_sagittal": sagittal, "mtf_threshold": threshold,
                "meets_threshold": min(tangential, sagittal) > threshold,
            })
    finally:
        fft.Close()
    return {"spot_radius_units": "um", "fields": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path, help="Directory containing ZOSAPI.dll")
    parser.add_argument("output", type=Path, help="Directory for the two prescriptions and JSON report")
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
        system = application.PrimarySystem
        design = build_system(zos, system)
        args.output.mkdir(parents=True, exist_ok=True)
        last_surface = system.LDE.GetSurfaceAt(4)
        operand = zos.Editors.MFE.MeritOperandType.EFFL
        report = {
            "wavelength_um": WAVELENGTH_UM,
            "nominal_working_f_number": WORKING_F_NUMBER,
            "thin_lens_efl_mm": design.efl,
            "opticstudio_efl_mm": system.MFE.GetOperandValue(operand, 0, 0, 0, 0, 0, 0, 0, 0),
            "initial_image_gap_mm": last_surface.Thickness,
            "initial": measure(zos, system),
        }
        system.SaveAs(str((args.output / "paper_example_initial.zmx").resolve()))
        focus = system.Tools.OpenQuickFocus()
        try:
            focus.Criterion = zos.Tools.General.QuickFocusCriterion.SpotSizeRadial
            focus.UseCentroid = True
            focus.RunAndWaitForCompletion()
        finally:
            focus.Close()
        report["refocused_image_gap_mm"] = last_surface.Thickness
        report["refocused"] = measure(zos, system)
        system.SaveAs(str((args.output / "paper_example_refocused.zmx").resolve()))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        application.CloseApplication()


if __name__ == "__main__":
    main()
