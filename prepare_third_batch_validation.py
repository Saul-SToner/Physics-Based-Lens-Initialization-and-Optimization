"""Freeze V2 artifacts and write all Case 201-700 predictions before Zemax starts."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_feature_dataset import build_initial_features
from prescription_registry import sample_fingerprint
from run_rule_regression_v1 import RAW9, conditioned_design

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "third_batch_validation_freeze_v1"
MODEL_DIR = ROOT / "ml_rule_regression_200_v2"
DATASET = ROOT / "random_candidates_700/candidates.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def gap(f_number, field):
    return np.maximum((np.asarray(f_number) - 20.0) / 20.0, 0.0) + np.maximum((5.0 - np.asarray(field)) / 5.0, 0.0)


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    source = read(DATASET)
    candidates = [x for x in source["candidates"] if 201 <= int(x["id"]) <= 700]
    if [int(x["id"]) for x in candidates] != list(range(201, 701)):
        raise RuntimeError("Expected contiguous Case 201-700")
    prior = [x for x in source["candidates"] if int(x["id"]) <= 200]
    prior_fp = {sample_fingerprint(x)[0] for x in prior}
    new_fp = [sample_fingerprint(x)[0] for x in candidates]
    if len(set(new_fp)) != 500 or prior_fp & set(new_fp):
        raise RuntimeError("Third-batch prescription overlap")

    schema, rows = {}, []
    for candidate, fingerprint in zip(candidates, new_fp):
        features = build_initial_features(candidate, schema)
        for rule in (1, 2, 3):
            rows.append({"prescription_id": int(candidate["id"]), "rule": rule,
                         "prescription_sha256": fingerprint, **features})
    frame = pd.DataFrame(rows).sort_values(["prescription_id", "rule"]).reset_index(drop=True)
    feature_dictionary = read(MODEL_DIR / "feature_dictionary.json")
    if RAW9 != feature_dictionary["raw9"]:
        raise RuntimeError("raw9 definition changed")
    x, design_columns = conditioned_design(frame, RAW9)
    if design_columns != feature_dictionary["design_columns"]:
        raise RuntimeError("Rule-conditioned design changed")

    f_model = joblib.load(MODEL_DIR / "model_final_f_number.joblib")
    field_model = joblib.load(MODEL_DIR / "model_final_field_deg.joblib")
    predicted_f = f_model.predict(x)
    predicted_field = field_model.predict(x)
    predicted_gap = gap(predicted_f, predicted_field)
    prediction_rows = []
    for i, row in frame.iterrows():
        prediction_rows.append({"prescription_id": int(row.prescription_id), "rule": int(row.rule),
                                "prescription_sha256": row.prescription_sha256,
                                "predicted_final_f_number": float(predicted_f[i]),
                                "predicted_final_field_deg": float(predicted_field[i]),
                                "predicted_target_gap": float(predicted_gap[i])})
    decisions = []
    pred = pd.DataFrame(prediction_rows)
    for cid, group in pred.groupby("prescription_id"):
        g = group.sort_values("rule")
        minimum = float(g.predicted_target_gap.min())
        tied = g[np.isclose(g.predicted_target_gap, minimum, rtol=0, atol=1e-12)]
        decisions.append({"prescription_id": int(cid), "selected_rule": int(tied.rule.min()),
                          "predicted_minimum_gap": minimum,
                          "tie_rules": list(map(int, tied.rule)),
                          "selection_available_before_optimization": True})

    frozen_paths = [MODEL_DIR / "model_final_f_number.joblib",
                    MODEL_DIR / "model_final_field_deg.joblib",
                    MODEL_DIR / "feature_dictionary.json", MODEL_DIR / "config.json",
                    ROOT / "ml_v2_oof_error_decision_analysis/decision_summary.json",
                    DATASET]
    freeze = {
        "status": "frozen_before_third_batch_optimization",
        "cases": [201, 700], "prescription_count": 500, "rule_conditioned_predictions": 1500,
        "models": "V2 RF/raw9 leaf=1 final models trained on Case 1-200",
        "raw9": RAW9, "design_columns": design_columns,
        "selection_policy": {
            "formula": "max((predicted_F_number-20)/20,0) + max((5-predicted_field_deg)/5,0)",
            "direction": "minimum", "tie_break": "smallest rule number",
            "primary_validation_contrast": "actual_gap(selected_rule) - actual_gap(fixed_Rule_3)",
            "secondary_endpoints": ["target_success_rate", "true_Pareto_membership", "harm_fraction"]
        },
        "artifact_sha256": {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths},
        "no_third_batch_outcome_read": True,
        "predictions_written_before_optimization": True,
        "validation_metrics_embargoed_until_user_requests_report": True
    }
    write(OUT / "predictions_preoptimization.json", prediction_rows)
    write(OUT / "rule_selection_preoptimization.json", decisions)
    write(OUT / "prescription_registry.json", [{"prescription_id": int(c["id"]),
                                                  "prescription_sha256": fp,
                                                  "inputs": c["inputs"]}
                                                 for c, fp in zip(candidates, new_fp)])
    write(OUT / "freeze_manifest.json", freeze)
    prediction_hashes = {name: digest(OUT / name) for name in
                         ("predictions_preoptimization.json", "rule_selection_preoptimization.json",
                          "prescription_registry.json", "freeze_manifest.json")}
    write(OUT / "freeze_receipt.json", {"artifact_sha256": prediction_hashes,
                                         "prediction_rows": 1500, "decision_rows": 500,
                                         "prior_overlap_count": 0})
    print(json.dumps({"output": str(OUT), "prediction_rows": len(prediction_rows),
                      "decision_rows": len(decisions), "choice_counts": pd.Series(
                          [x["selected_rule"] for x in decisions]).value_counts().sort_index().to_dict(),
                      "hashes": prediction_hashes}, indent=2))


if __name__ == "__main__":
    main()
