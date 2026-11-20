"""P0-P2 follow-up audits for the first 100 rule-conditioned regressions."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_rule_regression_v1 import RAW9, PHYSICS6, conditioned_design


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "ml_rule_regression_100_v1"
OUT = ROOT / "ml_rule_regression_followup_100_v1"
SEED = 20261118


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def metric(y, pred):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "r2": float(r2_score(y, pred))}


def build_frame():
    features = read(ROOT / "feature_engineering_100_v1/features.json")
    outcomes = read(ROOT / "feature_engineering_100_v1/outcomes.json")
    by_case = {row["case_id"]: row for row in features}
    records = []
    for outcome in outcomes:
        source = by_case[outcome["case_id"]]
        if source["prescription_sha256"] != outcome["prescription_sha256"]:
            raise RuntimeError("Feature/outcome mismatch")
        records.append({"case_id": outcome["case_id"], "rule": outcome["rule"],
                        **source["features"],
                        "final_f_number": outcome["final_working_f_number"],
                        "final_field_deg": outcome["final_max_field_deg"]})
    return pd.DataFrame(records).sort_values(["case_id", "rule"]).reset_index(drop=True)


def folds_for(frame):
    assignments = {row["case_id"]: row["fold"]
                   for row in read(BASE / "fold_assignments.json")}
    folds = []
    for fold in range(1, 6):
        valid = np.where(frame.case_id.map(assignments).to_numpy() == fold)[0]
        train = np.where(frame.case_id.map(assignments).to_numpy() != fold)[0]
        if set(frame.iloc[train].case_id) & set(frame.iloc[valid].case_id):
            raise RuntimeError("Group leakage")
        folds.append((train, valid))
    return folds


def audit_f500():
    outcomes = read(ROOT / "feature_engineering_100_v1/outcomes.json")
    state = read(ROOT / "background_runs/batch_300_20261118/state.json")
    cases = []
    for outcome in outcomes:
        if outcome["final_working_f_number"] != 500:
            continue
        key = f"case_{outcome['case_id']:03d}_rule_{outcome['rule']}"
        job = state["jobs"][key]
        path = ROOT / "background_runs/batch_300_20261118" / job["attempt"] / "experiment/result.json"
        result = read(path)
        raw_hash = digest(path)
        if raw_hash != job["acceptance"]["result_sha256"]:
            raise RuntimeError("Accepted result hash mismatch")
        final = result["final_accepted"]
        if final["working_f_number"] != 500 or not final["geometry_valid"]:
            raise RuntimeError("Unexpected F/#=500 endpoint")
        if not all(field["meets_threshold"] for field in final["fields"]):
            raise RuntimeError("F/#=500 endpoint failed MTF")
        cases.append({
            "case_id": outcome["case_id"], "rule": outcome["rule"],
            "result_path": str(path.relative_to(ROOT)), "result_sha256": raw_hash,
            "initial_step_accepted": bool(result["steps"][0]["accepted"]),
            "accepted_field_steps": sum(step["accepted"] and step["stage"] == "field"
                                        for step in result["steps"]),
            "final_field_deg": final["max_field_deg"],
            "stop_reasons": result["stop_reasons"],
            "geometry_valid": final["geometry_valid"],
            "minimum_geometry_clearance_mm": min(final["clearances_mm"].values()),
            "all_mtf_pass": all(field["meets_threshold"] for field in final["fields"]),
            "minimum_mtf_margin": min(min(field["mtf_tangential"], field["mtf_sagittal"])
                                      - field["mtf_threshold"] for field in final["fields"]),
            "saved_final_verified": result["audit"].get("saved_final_verified", False),
        })
    return {
        "count": len(cases), "rule_counts": dict(Counter(str(row["rule"]) for row in cases)),
        "code_evidence": {
            "initial_value": "p1_engine.py START_F = 500.0",
            "rule_2_order": "run_p1.py order = (field, aperture)",
            "failure_behavior": "If first stage stops before target, second stage is marked not_started_first_stage_limit",
        },
        "classification": {
            "special_failure_sentinel": False,
            "explicit_optimizer_upper_bound": False,
            "initial_continuation_boundary": True,
            "valid_under_current_geometry_and_mtf_checks": True,
            "policy_censored_endpoint": True,
            "explanation": "Rule 2 starts at F/#=500 and expands field first. In these cases field progression stopped before 5 degrees, so aperture progression never began. The last accepted prescriptions are real saved and reloaded endpoints that pass current geometry and MTF checks, but 500 reflects the continuation policy boundary rather than an independently optimized aperture optimum.",
        },
        "cases": cases,
    }


def error_audit():
    predictions = pd.DataFrame(read(BASE / "predictions.json"))
    data = predictions[(predictions.target == "final_f_number") &
                       (predictions.model == "random_forest") &
                       (predictions.feature_set == "raw9")].copy()
    data["error"] = data.predicted - data.actual
    data["absolute_error"] = data.error.abs()
    data["squared_error"] = data.error ** 2
    total_abs = data.absolute_error.sum()
    total_sq = data.squared_error.sum()
    is500 = data.actual == 500
    quantiles = {str(q): float(data.absolute_error.quantile(q))
                 for q in (0, .25, .5, .75, .9, .95, 1)}
    by_rule = []
    for rule, group in data.groupby("rule"):
        by_rule.append({"rule": int(rule), **metric(group.actual, group.predicted),
                        "median_absolute_error": float(group.absolute_error.median()),
                        "p90_absolute_error": float(group.absolute_error.quantile(.9)),
                        "maximum_absolute_error": float(group.absolute_error.max()),
                        "share_total_absolute_error": float(group.absolute_error.sum() / total_abs),
                        "share_total_squared_error": float(group.squared_error.sum() / total_sq),
                        "under_predictions": int((group.error < 0).sum()),
                        "over_predictions": int((group.error > 0).sum())})
    by_case = data.groupby("case_id").agg(
        mean_absolute_error=("absolute_error", "mean"),
        maximum_absolute_error=("absolute_error", "max"),
        sum_absolute_error=("absolute_error", "sum"),
    ).sort_values("sum_absolute_error", ascending=False).reset_index()
    return {
        "model": "random_forest/raw9 OOF", "absolute_error_quantiles": quantiles,
        "by_rule": by_rule,
        "f500_subset": {
            **metric(data.loc[is500, "actual"], data.loc[is500, "predicted"]),
            "n": int(is500.sum()),
            "mean_error": float(data.loc[is500, "error"].mean()),
            "share_total_absolute_error": float(data.loc[is500, "absolute_error"].sum() / total_abs),
            "share_total_squared_error": float(data.loc[is500, "squared_error"].sum() / total_sq),
        },
        "non_f500": metric(data.loc[~is500, "actual"], data.loc[~is500, "predicted"]),
        "top_25_rows": data.sort_values("absolute_error", ascending=False)[
            ["case_id", "rule", "fold", "actual", "predicted", "error", "absolute_error"]
        ].head(25).to_dict("records"),
        "top_20_cases": by_case.head(20).to_dict("records"),
    }


def per_rule_baseline_comparison():
    metrics = pd.DataFrame(read(BASE / "metrics.json"))
    fold = pd.DataFrame(read(BASE / "fold_metrics.json"))
    rows = []
    for target in ("final_f_number", "final_field_deg"):
        for rule in (1, 2, 3):
            scope = f"rule_{rule}"
            baseline = metrics[(metrics.target == target) & (metrics.model == "rule_mean") &
                               (metrics.split == "validation") & (metrics.scope == scope)].iloc[0]
            baseline_fold = fold[(fold.target == target) & (fold.model == "rule_mean") &
                                 (fold.split == "validation") & (fold.scope == scope)].set_index("fold")
            for model, feature_set in (("ridge", "raw9"), ("ridge", "raw9_plus_physics6"),
                                       ("random_forest", "raw9"),
                                       ("random_forest", "raw9_plus_physics6")):
                candidate = metrics[(metrics.target == target) & (metrics.model == model) &
                                    (metrics.feature_set == feature_set) &
                                    (metrics.split == "validation") &
                                    (metrics.scope == scope)].iloc[0]
                candidate_fold = fold[(fold.target == target) & (fold.model == model) &
                                      (fold.feature_set == feature_set) &
                                      (fold.split == "validation") &
                                      (fold.scope == scope)].set_index("fold")
                item = {"target": target, "rule": rule, "model": model,
                        "feature_set": feature_set}
                for name in ("mae", "rmse", "r2"):
                    item[f"baseline_{name}"] = float(baseline[name])
                    item[f"candidate_{name}"] = float(candidate[name])
                    item[f"delta_{name}_candidate_minus_baseline"] = float(candidate[name] - baseline[name])
                    delta = candidate_fold[name] - baseline_fold[name]
                    item[f"folds_better_{name}"] = int((delta < 0).sum() if name != "r2" else
                                                       (delta > 0).sum())
                rows.append(item)
    return rows


def log_target_experiment(frame, folds):
    rules = frame.rule.to_numpy(dtype=int)
    designs = {key: conditioned_design(frame, names)[0]
               for key, names in (("raw9", RAW9), ("raw9_plus_physics6", RAW9 + PHYSICS6))}
    configs = [
        ("ridge", "raw9", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("ridge", "raw9_plus_physics6", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("random_forest", "raw9", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=1)),
        ("random_forest", "raw9_plus_physics6", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=SEED, n_jobs=1)),
    ]
    existing = pd.DataFrame(read(BASE / "predictions.json"))
    predictions = []
    metrics = []
    for target in ("final_f_number", "final_field_deg"):
        y = frame[target].to_numpy(dtype=float)
        log_y = np.log(y)
        for model_name, feature_set, estimator in configs:
            oof = np.full(len(frame), np.nan)
            train_y, train_pred = [], []
            for fold_number, (train, valid) in enumerate(folds, 1):
                model = clone(estimator).fit(designs[feature_set][train], log_y[train])
                raw_train = np.exp(model.predict(designs[feature_set][train]))
                raw_valid = np.exp(model.predict(designs[feature_set][valid]))
                if not np.isfinite(raw_train).all() or not np.isfinite(raw_valid).all():
                    raise RuntimeError("Non-finite inverse log prediction")
                oof[valid] = raw_valid
                train_y.extend(y[train]); train_pred.extend(raw_train)
                for index, predicted in zip(valid, raw_valid):
                    predictions.append({"case_id": int(frame.iloc[index].case_id),
                                        "rule": int(rules[index]), "fold": fold_number,
                                        "target": target, "model": model_name,
                                        "feature_set": feature_set, "actual": float(y[index]),
                                        "predicted": float(predicted),
                                        "residual": float(predicted - y[index])})
            for split, actual, predicted in (("train", np.asarray(train_y), np.asarray(train_pred)),
                                             ("validation", y, oof)):
                metrics.append({"target": target, "model": model_name,
                                "feature_set": feature_set, "split": split,
                                "scope": "overall", **metric(actual, predicted),
                                "log_mae": float(mean_absolute_error(np.log(actual), np.log(predicted))),
                                "log_rmse": float(mean_squared_error(np.log(actual), np.log(predicted)) ** .5),
                                "log_r2": float(r2_score(np.log(actual), np.log(predicted)))})
                if split == "validation":
                    for rule in (1, 2, 3):
                        mask = rules == rule
                        metrics.append({"target": target, "model": model_name,
                                        "feature_set": feature_set, "split": split,
                                        "scope": f"rule_{rule}", **metric(y[mask], oof[mask]),
                                        "log_mae": float(mean_absolute_error(log_y[mask], np.log(oof[mask]))),
                                        "log_rmse": float(mean_squared_error(log_y[mask], np.log(oof[mask])) ** .5),
                                        "log_r2": float(r2_score(log_y[mask], np.log(oof[mask])))})
    comparison = []
    log_metrics = pd.DataFrame(metrics)
    direct_metrics = pd.DataFrame(read(BASE / "metrics.json"))
    for _, row in log_metrics[(log_metrics.split == "validation") &
                              (log_metrics.scope == "overall")].iterrows():
        direct = direct_metrics[(direct_metrics.target == row.target) &
                                (direct_metrics.model == row.model) &
                                (direct_metrics.feature_set == row.feature_set) &
                                (direct_metrics.split == "validation") &
                                (direct_metrics.scope == "overall")].iloc[0]
        comparison.append({"target": row.target, "model": row.model,
                           "feature_set": row.feature_set,
                           "direct_mae": float(direct.mae), "log_target_mae": float(row.mae),
                           "delta_mae_log_minus_direct": float(row.mae - direct.mae),
                           "direct_rmse": float(direct.rmse), "log_target_rmse": float(row.rmse),
                           "delta_rmse_log_minus_direct": float(row.rmse - direct.rmse),
                           "direct_r2": float(direct.r2), "log_target_r2": float(row.r2),
                           "delta_r2_log_minus_direct": float(row.r2 - direct.r2)})
    return predictions, metrics, comparison


def rf_sensitivity(frame, folds):
    x = conditioned_design(frame, RAW9)[0]
    rules = frame.rule.to_numpy(dtype=int)
    predictions = []
    metrics = []
    for target in ("final_f_number", "final_field_deg"):
        y = frame[target].to_numpy(dtype=float)
        for depth in (4, 8, None):
            for leaf in (1, 3, 8):
                oof = np.full(len(frame), np.nan)
                train_y, train_pred = [], []
                for fold_number, (train, valid) in enumerate(folds, 1):
                    model = RandomForestRegressor(
                        n_estimators=500, max_depth=depth, min_samples_leaf=leaf,
                        max_features="sqrt", random_state=SEED, n_jobs=1).fit(x[train], y[train])
                    p_train = model.predict(x[train]); p_valid = model.predict(x[valid])
                    oof[valid] = p_valid
                    train_y.extend(y[train]); train_pred.extend(p_train)
                    for index, predicted in zip(valid, p_valid):
                        predictions.append({"case_id": int(frame.iloc[index].case_id),
                                            "rule": int(rules[index]), "fold": fold_number,
                                            "target": target, "max_depth": depth,
                                            "min_samples_leaf": leaf, "actual": float(y[index]),
                                            "predicted": float(predicted)})
                for split, actual, predicted in (("train", np.asarray(train_y), np.asarray(train_pred)),
                                                 ("validation", y, oof)):
                    metrics.append({"target": target, "max_depth": depth,
                                    "min_samples_leaf": leaf, "split": split,
                                    "scope": "overall", **metric(actual, predicted)})
                    if split == "validation":
                        for rule in (1, 2, 3):
                            mask = rules == rule
                            metrics.append({"target": target, "max_depth": depth,
                                            "min_samples_leaf": leaf, "split": split,
                                            "scope": f"rule_{rule}", **metric(y[mask], oof[mask])})
    return predictions, metrics


def main():
    if OUT.exists():
        raise SystemExit("Output already exists")
    OUT.mkdir()
    frame = build_frame()
    folds = folds_for(frame)
    f500 = audit_f500()
    errors = error_audit()
    baseline = per_rule_baseline_comparison()
    log_predictions, log_metrics, log_comparison = log_target_experiment(frame, folds)
    rf_predictions, rf_metrics = rf_sensitivity(frame, folds)
    write(OUT / "f500_audit.json", f500)
    write(OUT / "f_number_error_audit.json", errors)
    write(OUT / "per_rule_baseline_comparison.json", baseline)
    write(OUT / "log_target_predictions.json", log_predictions)
    write(OUT / "log_target_metrics.json", log_metrics)
    write(OUT / "log_target_comparison.json", log_comparison)
    write(OUT / "rf_sensitivity_predictions.json", rf_predictions)
    write(OUT / "rf_sensitivity_metrics.json", rf_metrics)
    sources = [ROOT / "feature_engineering_100_v1/features.json",
               ROOT / "feature_engineering_100_v1/outcomes.json",
               BASE / "predictions.json", BASE / "metrics.json",
               BASE / "fold_metrics.json", BASE / "fold_assignments.json",
               ROOT / "background_runs/batch_300_20261118/state.json",
               ROOT / "p1_engine.py", ROOT / "run_p1.py"]
    write(OUT / "provenance.json", {
        "sources_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "script_sha256": digest(Path(__file__)), "random_seed": SEED,
        "second_batch_used": False, "new_zemax_run": False,
        "rf_sensitivity_is_model_analysis_not_hyperparameter_selection": True,
    })
    print(json.dumps({"f500": f500["classification"],
                      "f500_error": errors["f500_subset"],
                      "log_target": log_comparison,
                      "rf_validation": [row for row in rf_metrics if row["split"] == "validation"
                                        and row["scope"] == "overall"]}, indent=2))


if __name__ == "__main__":
    main()
