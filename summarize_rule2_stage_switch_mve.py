"""Summarize the frozen ten-case Rule-2 stage-switch MVE."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rule2_stage_switch_mve_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    rows = read(OUT / "comparison_rows.json")
    flat = []
    for row in rows:
        final = row.pop("final_metrics")
        fields = final["fields"]
        flat.append({**row,
                     "max_rms_spot_radius_um": max(x["rms_spot_radius"] for x in fields),
                     "minimum_mtf_margin": min(min(x["mtf_tangential"], x["mtf_sagittal"]) - x["mtf_threshold"] for x in fields),
                     "minimum_geometry_clearance_mm": min(final["clearances_mm"].values()),
                     "object_image_b_mm": final["object_image_b_mm"],
                     "magnification_error": final["magnification_error_at_image"],
                     "exit_pupil_b_mm": final["exit_pupil_b_mm"]})
    frame = pd.DataFrame(flat)
    aggregate = []
    for method, group in frame.groupby("method"):
        aggregate.append({"method": method, "n": int(len(group)),
                          "mean_final_f_number": float(group.final_f_number.mean()),
                          "median_final_f_number": float(group.final_f_number.median()),
                          "mean_final_field_deg": float(group.final_field_deg.mean()),
                          "mean_target_gap": float(group.target_gap.mean()),
                          "median_target_gap": float(group.target_gap.median()),
                          "mean_dls_calls": float(group.dls_calls.mean()),
                          "mean_requested_dls_cycles": float(group.requested_dls_cycles.mean()),
                          "mean_wall_seconds": float(group.wall_seconds.mean()),
                          "mean_max_rms_um": float(group.max_rms_spot_radius_um.mean()),
                          "minimum_observed_mtf_margin": float(group.minimum_mtf_margin.min()),
                          "minimum_observed_geometry_clearance_mm": float(group.minimum_geometry_clearance_mm.min())})
    wide = frame.pivot(index="case_id", columns="method")
    comparison = []
    for cid in sorted(frame.case_id.unique()):
        r2 = frame[(frame.case_id == cid) & (frame.method == "historical_rule_2")].iloc[0]
        r3 = frame[(frame.case_id == cid) & (frame.method == "historical_rule_3")].iloc[0]
        switch = frame[(frame.case_id == cid) & (frame.method == "rule_2_stage_switch")].iloc[0]
        comparison.append({"case_id": int(cid),
                           "rule2_f": float(r2.final_f_number), "switch_f": float(switch.final_f_number),
                           "rule3_f": float(r3.final_f_number),
                           "rule2_field": float(r2.final_field_deg), "switch_field": float(switch.final_field_deg),
                           "rule3_field": float(r3.final_field_deg),
                           "rule2_gap": float(r2.target_gap), "switch_gap": float(switch.target_gap),
                           "rule3_gap": float(r3.target_gap),
                           "gap_change_switch_minus_rule2": float(switch.target_gap - r2.target_gap),
                           "gap_change_switch_minus_rule3": float(switch.target_gap - r3.target_gap),
                           "incremental_dls_calls": int(switch.incremental_dls_calls),
                           "incremental_wall_seconds": float(switch.incremental_wall_seconds),
                           "switch_stop_reason": switch.stop_reason,
                           "switch_minimum_mtf_margin": float(switch.minimum_mtf_margin),
                           "switch_minimum_geometry_clearance_mm": float(switch.minimum_geometry_clearance_mm),
                           "switch_max_rms_um": float(switch.max_rms_spot_radius_um),
                           "switch_object_image_b_mm": float(switch.object_image_b_mm),
                           "switch_magnification_error": float(switch.magnification_error),
                           "switch_exit_pupil_b_mm": float(switch.exit_pupil_b_mm)})
    c = pd.DataFrame(comparison)
    summary = {"cases": list(map(int, sorted(frame.case_id.unique()))),
               "mechanism": {"f500_cases_in_case_1_200": 31,
                             "all_field_step_limit_aperture_not_started": True},
               "intervention": {"cases_run": 10,
                                "cases_with_lower_f_number": int(np.sum(c.switch_f < c.rule2_f)),
                                "cases_with_lower_gap_than_rule2": int(np.sum(c.switch_gap < c.rule2_gap - 1e-12)),
                                "cases_better_gap_than_rule3": int(np.sum(c.switch_gap < c.rule3_gap - 1e-12)),
                                "cases_equal_gap_to_rule3": int(np.sum(np.isclose(c.switch_gap, c.rule3_gap, atol=1e-12))),
                                "cases_worse_gap_than_rule3": int(np.sum(c.switch_gap > c.rule3_gap + 1e-12)),
                                "mean_gap_change_vs_rule2": float((c.switch_gap - c.rule2_gap).mean()),
                                "mean_gap_change_vs_rule3": float((c.switch_gap - c.rule3_gap).mean()),
                                "mean_incremental_dls_calls": float(c.incremental_dls_calls.mean()),
                                "range_incremental_dls_calls": [int(c.incremental_dls_calls.min()), int(c.incremental_dls_calls.max())],
                                "mean_incremental_wall_seconds": float(c.incremental_wall_seconds.mean()),
                                "all_final_endpoints_passed_original_acceptance": bool((c.switch_minimum_mtf_margin >= 0).all() and
                                                                                       (c.switch_minimum_geometry_clearance_mm >= .001).all())},
               "aggregate": aggregate}
    write(OUT / "comparison_table.json", comparison)
    write(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
