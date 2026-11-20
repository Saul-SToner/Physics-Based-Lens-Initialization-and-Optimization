"""Leakage-safe Stage-Aware F-number feasibility experiment on batch 1 only."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)

from run_rule_regression_v1 import RAW9, conditioned_design

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_stage_aware_f_number_100_v1"
SEED = 20261118


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def regression_metrics(y, pred):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None}


def classification_metrics(y, pred, prob):
    # Entered aperture is label 1; minority diagnostics explicitly treat not-entered as positive.
    out = {"n": int(len(y)), "entered_n": int(np.sum(y == 1)),
           "not_entered_n": int(np.sum(y == 0)),
           "accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "entered_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
           "entered_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
           "entered_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
           "not_entered_precision": float(precision_score(y, pred, pos_label=0, zero_division=0)),
           "not_entered_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
           "not_entered_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
           "brier": float(brier_score_loss(y, prob)),
           "log_loss": float(log_loss(y, np.column_stack([1 - prob, prob]), labels=[0, 1])),
           "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    out["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return out


def build_frame():
    feature_path = ROOT / "feature_engineering_100_v1/features.json"
    outcome_path = ROOT / "feature_engineering_100_v1/outcomes.json"
    state_path = ROOT / "background_runs/batch_300_20261118/state.json"
    features, outcomes, state = read(feature_path), read(outcome_path), read(state_path)
    if len(features) != 100 or len(outcomes) != 300:
        raise RuntimeError("Expected exactly batch-1 100 prescriptions and 300 rule outcomes")
    by_case = {int(x["case_id"]): x for x in features}
    rows, log_paths = [], []
    for outcome in outcomes:
        case_id, rule = int(outcome["case_id"]), int(outcome["rule"])
        job = state["jobs"][f"case_{case_id:03d}_rule_{rule}"]
        path = ROOT / "background_runs/batch_300_20261118" / job["attempt"] / "experiment/result.json"
        if digest(path) != job["acceptance"]["result_sha256"]:
            raise RuntimeError(f"Accepted result hash mismatch: {path}")
        result = read(path)
        entered = any(step.get("stage") == "aperture" for step in result.get("steps", []))
        f_number = float(outcome["final_working_f_number"])
        accepted_f = float(job["acceptance"]["final_accepted"]["working_f_number"])
        if not np.isclose(f_number, accepted_f, rtol=0, atol=1e-12):
            raise RuntimeError("Outcome/log endpoint mismatch")
        feature = by_case[case_id]
        if feature["prescription_sha256"] != outcome["prescription_sha256"]:
            raise RuntimeError("Feature/outcome prescription mismatch")
        rows.append({"case_id": case_id, "rule": rule, **feature["features"],
                     "entered_aperture": int(entered), "final_f_number": f_number,
                     "is_f500": bool(np.isclose(f_number, 500.0, rtol=0, atol=1e-12)),
                     "result_path": str(path.relative_to(ROOT)), "result_sha256": digest(path)})
        log_paths.append(path)
    frame = pd.DataFrame(rows).sort_values(["case_id", "rule"]).reset_index(drop=True)
    sources = [feature_path, outcome_path, state_path] + log_paths
    return frame, {str(p.relative_to(ROOT)): digest(p) for p in sources}


def folds_for(frame):
    assignments = {int(x["case_id"]): int(x["fold"])
                   for x in read(ROOT / "ml_rule_regression_100_v1/fold_assignments.json")}
    fold_ids = frame.case_id.map(assignments).to_numpy(dtype=int)
    folds = []
    for fold in range(1, 6):
        valid, train = np.where(fold_ids == fold)[0], np.where(fold_ids != fold)[0]
        if set(frame.iloc[train].case_id) & set(frame.iloc[valid].case_id):
            raise RuntimeError("Prescription leakage")
        folds.append((fold, train, valid))
    return fold_ids, folds


def existing_predictions(frame):
    leaf3 = pd.DataFrame(read(ROOT / "ml_rule_regression_100_v1/predictions.json"))
    leaf3 = leaf3[(leaf3.target == "final_f_number") & (leaf3.model == "random_forest") &
                  (leaf3.feature_set == "raw9")]
    leaf1 = pd.DataFrame(read(ROOT / "ml_rule_regression_followup_100_v1/rf_sensitivity_predictions.json"))
    leaf1 = leaf1[(leaf1.target == "final_f_number") & leaf1.max_depth.isna() &
                  (leaf1.min_samples_leaf == 1)]
    key = ["case_id", "rule"]
    base = frame[key].copy()
    for name, source in (("rf9_leaf3", leaf3), ("rf9_leaf1", leaf1)):
        source = source[key + ["actual", "predicted"]].rename(columns={"predicted": name})
        base = base.merge(source, on=key, how="left", validate="one_to_one")
    if base[["rf9_leaf3", "rf9_leaf1"]].isna().any().any():
        raise RuntimeError("Missing comparator OOF prediction")
    return base


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    frame, source_hashes = build_frame()
    fold_ids, folds = folds_for(frame)
    x, columns = conditioned_design(frame, RAW9)
    y_stage = frame.entered_aperture.to_numpy(dtype=int)
    y_f = frame.final_f_number.to_numpy(dtype=float)
    rules = frame.rule.to_numpy(dtype=int)

    stage_prob = np.full(len(frame), np.nan)
    stage_pred = np.full(len(frame), -1, dtype=int)
    rule_prob = np.full(len(frame), np.nan)
    hard = np.full(len(frame), np.nan)
    soft = np.full(len(frame), np.nan)
    fold_rows, leakage_audit = [], []
    for fold, train, valid in folds:
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=SEED + fold, n_jobs=1)
        clf.fit(x[train], y_stage[train])
        prob = clf.predict_proba(x[valid])[:, list(clf.classes_).index(1)]
        pred = (prob >= 0.5).astype(int)
        stage_prob[valid], stage_pred[valid] = prob, pred
        train_rule_means = {rule: float(np.mean(y_stage[train][rules[train] == rule]))
                            for rule in (1, 2, 3)}
        rule_prob[valid] = np.asarray([train_rule_means[r] for r in rules[valid]])
        entered_train = train[y_stage[train] == 1]
        reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=1,
                                    max_features="sqrt", random_state=SEED + fold, n_jobs=1)
        reg.fit(x[entered_train], y_f[entered_train])
        reg_pred = reg.predict(x[valid])
        hard[valid] = np.where(pred == 0, 500.0, reg_pred)
        soft[valid] = (1.0 - prob) * 500.0 + prob * reg_pred
        fold_rows.append({"fold": fold, "train_rows": int(len(train)), "valid_rows": int(len(valid)),
                          "train_not_entered": int(np.sum(y_stage[train] == 0)),
                          "valid_not_entered": int(np.sum(y_stage[valid] == 0)),
                          "regression_train_entered": int(len(entered_train)),
                          "classifier": classification_metrics(y_stage[valid], pred, prob)})
        leakage_audit.append({"fold": fold, "classifier_fit_rows": train.tolist(),
                              "regressor_fit_rows": entered_train.tolist(),
                              "validation_rows": valid.tolist(),
                              "validation_stage_used_for_prediction": False})

    if not (np.isfinite(stage_prob).all() and np.isfinite(hard).all() and np.isfinite(soft).all()):
        raise RuntimeError("Incomplete OOF prediction")
    rule_pred = (rule_prob >= 0.5).astype(int)
    comparator = existing_predictions(frame)
    predictions = frame[["case_id", "rule", "entered_aperture", "is_f500", "final_f_number"]].copy()
    predictions["fold"] = fold_ids
    predictions["stage_probability_entered"] = stage_prob
    predictions["stage_prediction_entered"] = stage_pred
    predictions["rule_only_stage_probability_entered"] = rule_prob
    predictions["rule_only_stage_prediction_entered"] = rule_pred
    predictions = predictions.merge(comparator, on=["case_id", "rule"], validate="one_to_one")
    predictions["stage_aware_hard"] = hard
    predictions["stage_aware_soft"] = soft

    stage_metrics = {"rf_raw9_rule": classification_metrics(y_stage, stage_pred, stage_prob),
                     "rule_only_probability": classification_metrics(y_stage, rule_pred, rule_prob),
                     "by_fold": fold_rows, "by_rule": {}}
    for rule in (1, 2, 3):
        m = rules == rule
        stage_metrics["by_rule"][str(rule)] = classification_metrics(y_stage[m], stage_pred[m], stage_prob[m])

    regression = []
    for model in ("rf9_leaf3", "rf9_leaf1", "stage_aware_hard", "stage_aware_soft"):
        pred = predictions[model].to_numpy(dtype=float)
        for scope, mask in [("overall", np.ones(len(frame), dtype=bool)),
                            ("f500", frame.is_f500.to_numpy()),
                            ("non_f500", ~frame.is_f500.to_numpy()),
                            ("entered_aperture", y_stage == 1),
                            ("not_entered_aperture", y_stage == 0)]:
            regression.append({"model": model, "scope": scope, **regression_metrics(y_f[mask], pred[mask])})
        for rule in (1, 2, 3):
            m = rules == rule
            regression.append({"model": model, "scope": f"rule_{rule}", **regression_metrics(y_f[m], pred[m])})

    contingency = []
    for rule in (1, 2, 3):
        for entered in (0, 1):
            for f500 in (False, True):
                n = int(np.sum((rules == rule) & (y_stage == entered) & (frame.is_f500.to_numpy() == f500)))
                if n:
                    contingency.append({"rule": rule, "entered_aperture": bool(entered), "f_number_500": f500, "n": n})
    minority = int(np.sum(y_stage == 0))
    stability = {"not_entered_total": minority,
                 "not_entered_fraction": float(minority / len(frame)),
                 "minimum_not_entered_training_rows": min(x["train_not_entered"] for x in fold_rows),
                 "validation_fold_without_not_entered": [x["fold"] for x in fold_rows if x["valid_not_entered"] == 0],
                 "stable_training_supported": False,
                 "decision": "Retain direct RF models; stage-aware results are feasibility evidence only because the minority class has 11 rows, all from Rule 2, and one validation fold has no minority example."}
    write(OUT / "stage_log_audit.json", {"definition": "entered iff any accepted-source result step has stage == aperture",
                                          "contingency": contingency,
                                          "observed_equivalence_in_batch_1_only": bool(np.array_equal(y_stage == 0, frame.is_f500.to_numpy())),
                                          "no_equivalence_assumed": True})
    write(OUT / "stage_predictions.json", predictions.to_dict("records"))
    write(OUT / "stage_metrics.json", stage_metrics)
    write(OUT / "regression_metrics.json", regression)
    write(OUT / "stability.json", stability)
    write(OUT / "provenance.json", {"scope": "first 100 prescriptions only", "source_sha256": source_hashes,
                                     "raw_features": RAW9, "design_columns": columns,
                                     "fold_source": "ml_rule_regression_100_v1/fold_assignments.json",
                                     "leakage_audit": leakage_audit,
                                     "no_zemax_run": True, "historical_files_modified": False})
    write(OUT / "source_hashes_after.json", {path: digest(ROOT / path) for path in source_hashes})
    if source_hashes != read(OUT / "source_hashes_after.json"):
        raise RuntimeError("Historical source changed during read-only experiment")
    print(json.dumps({"output": str(OUT), "stage": stage_metrics["rf_raw9_rule"],
                      "regression_overall": [x for x in regression if x["scope"] == "overall"],
                      "stability": stability}, indent=2))


if __name__ == "__main__":
    main()
