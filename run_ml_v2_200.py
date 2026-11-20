"""V2 rule-conditioned RF experiment on 200 prescriptions (600 rows)."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from build_feature_dataset import build_initial_features
from prescription_registry import sample_fingerprint
from run_rule_regression_v1 import RAW9, conditioned_design

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_rule_regression_200_v2"
SEED = 20261118
TARGETS = {"final_f_number": "working_f_number", "final_field_deg": "max_field_deg"}
RF_CONFIG = {"n_estimators": 500, "min_samples_leaf": 1, "max_depth": None,
             "max_features": "sqrt", "random_state": SEED, "n_jobs": 1}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, pred):
    constant = len(y) < 2 or np.all(y == y[0])
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "r2": None if constant else float(r2_score(y, pred))}


def build_dataset():
    candidates = read(ROOT / "random_candidates_200/candidates.json")
    if len(candidates["candidates"]) != 200:
        raise RuntimeError("Expected exactly 200 candidates")
    batches = [(ROOT / "background_runs/batch_300_20261118", range(1, 101)),
               (ROOT / "background_runs/batch_300_20261118_02", range(101, 201))]
    states = {cid: read(folder / "state.json") for folder, ids in batches for cid in ids}
    by_id = {int(x["id"]): x for x in candidates["candidates"]}
    schema, rows, result_hashes = {}, [], {}
    for case_id in range(1, 201):
        candidate = by_id[case_id]
        fp, _ = sample_fingerprint(candidate)
        feature = build_initial_features(candidate, schema)
        folder = batches[0][0] if case_id <= 100 else batches[1][0]
        state = states[case_id]
        for rule in (1, 2, 3):
            info = state["jobs"][f"case_{case_id:03d}_rule_{rule}"]
            path = folder / info["attempt"] / "experiment/result.json"
            actual_hash = digest(path)
            if actual_hash != info["acceptance"]["result_sha256"]:
                raise RuntimeError("Accepted result hash mismatch")
            result = read(path)
            if result["inputs"] != candidate["inputs"] or (int(result["case_id"]), int(result["rule"])) != (case_id, rule):
                raise RuntimeError("Feature/target identity mismatch")
            final = result["final_accepted"]
            if final is None:
                raise RuntimeError("Missing accepted endpoint")
            rows.append({"prescription_id": case_id, "rule": rule,
                         "prescription_sha256": fp, **feature,
                         "final_f_number": float(final["working_f_number"]),
                         "final_field_deg": float(final["max_field_deg"])})
            result_hashes[str(path.relative_to(ROOT))] = actual_hash
    frame = pd.DataFrame(rows).sort_values(["prescription_id", "rule"]).reset_index(drop=True)
    if len(frame) != 600 or frame.prescription_sha256.nunique() != 200:
        raise RuntimeError("Combined dataset cardinality failure")
    if any(set(frame.loc[frame.prescription_id == cid, "rule"]) != {1, 2, 3} for cid in range(1, 201)):
        raise RuntimeError("Incomplete rule rows")
    train_dict = read(ROOT / "feature_engineering_100_v1/dictionary.json")["features"]
    fields = ("unit", "definition", "dependencies", "group", "available_before_rule_selection")
    if set(schema) != set(train_dict) or any(any(schema[n][f] != train_dict[n][f] for f in fields) for n in schema):
        raise RuntimeError("Feature definition changed from V1")
    return frame, schema, result_hashes


def old_v1_metrics():
    pred = pd.DataFrame(read(ROOT / "ml_rule_regression_followup_100_v1/rf_sensitivity_predictions.json"))
    pred = pred[(pred.max_depth.isna()) & (pred.min_samples_leaf == 1)].copy()
    # Join the V1 F/# endpoint to define the pre-specified 500/non-500 grouping for both targets.
    f = pred[pred.target == "final_f_number"][["case_id", "rule", "actual"]].rename(columns={"actual": "f_actual"})
    pred = pred.merge(f, on=["case_id", "rule"], how="left", validate="many_to_one")
    rows = []
    for target in TARGETS:
        data = pred[pred.target == target]
        scopes = [("overall", np.ones(len(data), dtype=bool)),
                  ("f_number_500", np.isclose(data.f_actual, 500.0, rtol=0, atol=1e-12)),
                  ("non_f_number_500", ~np.isclose(data.f_actual, 500.0, rtol=0, atol=1e-12))]
        scopes += [(f"rule_{r}", data.rule.to_numpy() == r) for r in (1, 2, 3)]
        for scope, mask in scopes:
            rows.append({"version": "v1_grouped_cv_100", "target": target, "scope": scope,
                         **metrics(data.actual.to_numpy()[mask], data.predicted.to_numpy()[mask])})
    return rows


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    frame, schema, result_hashes = build_dataset()
    x, design_columns = conditioned_design(frame, RAW9)
    groups = frame.prescription_id.to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(x, groups=groups))
    fold_ids = np.zeros(len(frame), dtype=int)
    fold_manifest = []
    for fold, (train, valid) in enumerate(folds, 1):
        train_ids, valid_ids = set(groups[train]), set(groups[valid])
        if train_ids & valid_ids or len(valid_ids) != 40:
            raise RuntimeError("Grouped CV leakage or imbalance")
        fold_ids[valid] = fold
        fold_manifest.append({"fold": fold, "train_prescriptions": sorted(map(int, train_ids)),
                              "validation_prescriptions": sorted(map(int, valid_ids))})

    prediction_rows, metric_rows = [], []
    for target in TARGETS:
        y = frame[target].to_numpy(dtype=float)
        oof = np.full(len(frame), np.nan)
        for fold, (train, valid) in enumerate(folds, 1):
            model = RandomForestRegressor(**RF_CONFIG).fit(x[train], y[train])
            oof[valid] = model.predict(x[valid])
        if not np.isfinite(oof).all():
            raise RuntimeError("Incomplete OOF prediction")
        for i in range(len(frame)):
            prediction_rows.append({"prescription_id": int(groups[i]), "rule": int(frame.iloc[i].rule),
                                    "fold": int(fold_ids[i]), "target": target,
                                    "actual": float(y[i]), "predicted": float(oof[i]),
                                    "is_f500": bool(np.isclose(frame.iloc[i].final_f_number, 500.0, rtol=0, atol=1e-12))})
        f500 = np.isclose(frame.final_f_number.to_numpy(), 500.0, rtol=0, atol=1e-12)
        scopes = [("overall", np.ones(len(frame), dtype=bool)), ("f_number_500", f500),
                  ("non_f_number_500", ~f500)]
        scopes += [(f"rule_{r}", frame.rule.to_numpy() == r) for r in (1, 2, 3)]
        for scope, mask in scopes:
            metric_rows.append({"version": "v2_grouped_cv_200", "target": target, "scope": scope,
                                **metrics(y[mask], oof[mask])})

        final_model = RandomForestRegressor(**RF_CONFIG).fit(x, y)
        joblib.dump(final_model, OUT / f"model_{target}.joblib", compress=3)

    v1 = old_v1_metrics()
    v2_lookup = {(x["target"], x["scope"]): x for x in metric_rows}
    comparison = []
    for old in v1:
        new = v2_lookup[(old["target"], old["scope"])]
        comparison.append({"target": old["target"], "scope": old["scope"],
                           "v1_n": old["n"], "v2_n": new["n"],
                           **{f"v1_{m}": old[m] for m in ("mae", "rmse", "r2")},
                           **{f"v2_{m}": new[m] for m in ("mae", "rmse", "r2")},
                           **{f"delta_v2_minus_v1_{m}": None if old[m] is None or new[m] is None else new[m] - old[m]
                              for m in ("mae", "rmse", "r2")}})

    feature_dictionary = {"raw9": RAW9, "rule_encoding": "three fixed indicator blocks, each followed by raw9*indicator",
                          "design_columns": design_columns,
                          "source_feature_definitions": {name: schema[name] for name in RAW9},
                          "target_definitions": {"final_f_number": "accepted endpoint working_f_number",
                                                 "final_field_deg": "accepted endpoint max_field_deg"}}
    config = {"experiment": "ml_rule_regression_200_v2", "prescriptions": 200, "rows": 600,
              "cross_validation": {"class": "GroupKFold", "n_splits": 5, "shuffle": True,
                                   "random_state": SEED, "group": "prescription_id"},
              "model": {"class": "RandomForestRegressor", **RF_CONFIG},
              "targets": list(TARGETS), "feature_set": "raw9 + fixed rule conditioning",
              "historical_independent_test_preserved": True,
              "new_independent_test_available": False,
              "stage_aware_included": False,
              "second_batch_used_for_training": True}
    write(OUT / "predictions_oof.json", prediction_rows)
    write(OUT / "metrics.json", metric_rows)
    write(OUT / "v1_descriptive_comparison.json", comparison)
    write(OUT / "fold_assignments.json", fold_manifest)
    write(OUT / "feature_dictionary.json", feature_dictionary)
    write(OUT / "config.json", config)
    write(OUT / "dataset_manifest.json", {"prescription_ids": [1, 200], "prescription_count": 200,
                                           "rule_conditioned_rows": 600,
                                           "prescription_sha256": frame[["prescription_id", "prescription_sha256"]].drop_duplicates().to_dict("records")})
    model_hashes = {p.name: digest(p) for p in OUT.glob("model_*.joblib")}
    source_paths = [ROOT / "random_candidates_200/candidates.json",
                    ROOT / "background_runs/batch_300_20261118/state.json",
                    ROOT / "background_runs/batch_300_20261118_02/state.json",
                    ROOT / "ml_rule_regression_followup_100_v1/rf_sensitivity_predictions.json",
                    ROOT / "independent_f_number_test_batch2_v1/report.md"]
    write(OUT / "provenance.json", {"source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                                     "result_sha256": result_hashes, "model_sha256": model_hashes,
                                     "python": os.sys.version, "sklearn": sklearn.__version__,
                                     "old_outputs_overwritten": False,
                                     "new_independent_test_result": False})
    print(json.dumps({"metrics": metric_rows, "models": model_hashes}, indent=2))


if __name__ == "__main__":
    main()
