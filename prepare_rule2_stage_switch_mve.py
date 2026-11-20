"""Audit Rule-2 F/500 stops and freeze ten raw9-representative MVE cases."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from build_feature_dataset import build_initial_features
from run_rule_regression_v1 import RAW9

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rule2_stage_switch_mve_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def batch_for(case_id):
    return ROOT / ("background_runs/batch_300_20261118" if case_id <= 100 else
                   "background_runs/batch_300_20261118_02")


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    candidates = {int(x["id"]): x for x in read(ROOT / "random_candidates_200/candidates.json")["candidates"]}
    audit, hashes, schema = [], {}, {}
    for case_id in range(1, 201):
        batch = batch_for(case_id)
        state = read(batch / "state.json")
        info = state["jobs"][f"case_{case_id:03d}_rule_2"]
        path = batch / info["attempt"] / "experiment/result.json"
        result = read(path)
        final = result["final_accepted"]
        if final is None or not np.isclose(final["working_f_number"], 500.0, rtol=0, atol=1e-12):
            continue
        entered_aperture = any(step.get("stage") == "aperture" for step in result["steps"])
        feature = build_initial_features(candidates[case_id], schema)
        audit.append({"case_id": case_id, "rule": 2,
                      "final_f_number": float(final["working_f_number"]),
                      "final_field_deg": float(final["max_field_deg"]),
                      "entered_aperture": entered_aperture,
                      "stop_reasons": result["stop_reasons"],
                      "field_steps": sum(x.get("stage") == "field" for x in result["steps"]),
                      "accepted_field_steps": sum(x.get("stage") == "field" and x["accepted"] for x in result["steps"]),
                      "original_requested_dls_cycles": result["cost"]["requested_dls_cycles"],
                      "original_wall_seconds": result["cost"]["wall_seconds"],
                      "result_path": str(path.relative_to(ROOT)),
                      "final_lens_path": str((path.parent / "final_accepted.zmx").relative_to(ROOT)),
                      "raw9": {name: feature[name] for name in RAW9}})
        hashes[str(path.relative_to(ROOT))] = digest(path)
        hashes[str((path.parent / "final_accepted.zmx").relative_to(ROOT))] = digest(path.parent / "final_accepted.zmx")
    if len(audit) != 31:
        raise RuntimeError(f"Expected 31 Rule-2 F/500 cases, got {len(audit)}")
    expected_stop = {"aperture": "not_started_first_stage_limit", "field": "step_limit"}
    if any(x["entered_aperture"] or x["stop_reasons"] != expected_stop for x in audit):
        raise RuntimeError("F/500 mechanism is not homogeneous")

    matrix = np.asarray([[row["raw9"][name] for name in RAW9] for row in audit], dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    clustering = KMeans(n_clusters=10, random_state=20261119, n_init=50).fit(scaled)
    selected = []
    for cluster in range(10):
        members = np.where(clustering.labels_ == cluster)[0]
        distances = np.linalg.norm(scaled[members] - clustering.cluster_centers_[cluster], axis=1)
        index = int(members[np.argmin(distances)])
        selected.append({"cluster": cluster, "case_id": audit[index]["case_id"],
                         "cluster_size": int(len(members)),
                         "distance_to_centroid": float(distances.min())})
    selected.sort(key=lambda x: x["case_id"])
    if len({x["case_id"] for x in selected}) != 10:
        raise RuntimeError("Representative selection is not unique")

    protocol = {
        "status": "frozen_before_intervention",
        "training_scope_only": "Case 1-200",
        "f500_case_count": 31,
        "selected_case_ids": [x["case_id"] for x in selected],
        "selection": "10-cluster KMeans medoids on standardized raw9; seed 20261119; n_init 50",
        "intervention_trigger": "Rule 2 field=step_limit AND aperture=not_started_first_stage_limit",
        "intervention": "load original accepted Rule-2 endpoint and continue aperture only",
        "aperture_schedule": {"start_factor": 0.707, "rejection_factor_update": "sqrt(factor)",
                              "stop_threshold": 0.99, "target_f_number": 20.0},
        "dls_cycles_per_attempt": 50, "dls_cores": 2,
        "constraints_unchanged": ["MTF", "geometry", "conjugates", "magnification", "exit pupil"],
        "controls": ["historical original Rule 2", "historical Rule 3"],
        "new_runs": 10, "case_201_plus_used": False,
        "source_sha256": {name: digest(ROOT / name) for name in
                          ("p1_engine.py", "run_p1.py", "analyze_opticstudio.py",
                           "lens_geometry.py", "compare_opticstudio.py")},
        "historical_result_and_lens_sha256": hashes,
    }
    write(OUT / "mechanism_audit.json", {"count": len(audit),
                                         "common_mechanism_confirmed": True,
                                         "common_stop_reasons": expected_stop,
                                         "cases": audit})
    write(OUT / "selected_cases.json", selected)
    write(OUT / "protocol_freeze.json", protocol)
    print(json.dumps({"f500_count": len(audit), "selected": selected,
                      "common_stop_reasons": expected_stop}, indent=2))


if __name__ == "__main__":
    main()
