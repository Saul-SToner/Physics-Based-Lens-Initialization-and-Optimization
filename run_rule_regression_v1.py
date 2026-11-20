"""First grouped-CV regression experiment for rule-conditioned Zemax endpoints."""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20261118
RAW9 = [
    "object_l1_mm", "image_l2_mm", "entrance_pupil_l1_mm",
    "exit_pupil_l2_mm", "magnification", "thin_spacing_mm",
    "thin_power_1_per_mm", "thin_power_2_per_mm", "thin_efl_mm",
]
PHYSICS6 = [
    "thin_spacing_over_abs_efl", "thin_power_ratio", "thin_power_balance",
    "initial_gap_over_abs_efl", "object_over_abs_efl", "image_over_abs_efl",
]
TARGETS = {
    "final_f_number": "final_working_f_number",
    "final_field_deg": "final_max_field_deg",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def conditioned_design(frame, names):
    """Fixed, non-learned rule expansion; scaling remains inside each CV fold."""
    values = frame[names].to_numpy(dtype=float)
    rule = frame["rule"].to_numpy(dtype=int)
    blocks = []
    columns = []
    for value in (1, 2, 3):
        indicator = (rule == value).astype(float)[:, None]
        blocks.append(indicator)
        columns.append(f"rule_{value}")
        blocks.append(values * indicator)
        columns.extend(f"{name}__rule_{value}" for name in names)
    return np.hstack(blocks), columns


def rule_mean_predict(y_train, rule_train, rule_test):
    overall = float(np.mean(y_train))
    means = {value: float(np.mean(y_train[rule_train == value]))
             for value in (1, 2, 3) if np.any(rule_train == value)}
    return np.asarray([means.get(int(value), overall) for value in rule_test])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        parser.error("Output directory already exists")
    output.mkdir(parents=True)

    source_dir = root / "feature_engineering_100_v1"
    source_paths = [source_dir / name for name in (
        "features.json", "outcomes.json", "dictionary.json", "quality.json",
        "provenance.json")]
    before = {str(path.relative_to(root)): digest(path) for path in source_paths}
    feature_rows = load(source_paths[0])
    outcomes = load(source_paths[1])
    dictionary = load(source_paths[2])

    if len(feature_rows) != 100 or len(outcomes) != 300:
        raise RuntimeError("Expected 100 initial prescriptions and 300 rule outcomes")
    if {row["case_id"] for row in feature_rows} != set(range(1, 101)):
        raise RuntimeError("Unexpected case IDs")
    if any(dictionary["features"][name]["available_before_rule_selection"] is not True
           for name in RAW9 + PHYSICS6):
        raise RuntimeError("A selected feature is unavailable before rule selection")
    if [name for name in RAW9 if dictionary["features"][name]["group"]
            not in ("sampled", "paper_derived")]:
        raise RuntimeError("RAW9 contains a non-original feature")

    by_case = {row["case_id"]: row for row in feature_rows}
    records = []
    for outcome in outcomes:
        feature = by_case[outcome["case_id"]]
        if feature["prescription_sha256"] != outcome["prescription_sha256"]:
            raise RuntimeError("Feature/outcome prescription mismatch")
        records.append({
            "case_id": outcome["case_id"],
            "prescription_sha256": outcome["prescription_sha256"],
            "rule": outcome["rule"],
            **feature["features"],
            **{name: outcome[key] for name, key in TARGETS.items()},
        })
    frame = pd.DataFrame(records).sort_values(["case_id", "rule"]).reset_index(drop=True)
    if frame[list(TARGETS)].isna().any().any():
        raise RuntimeError("Missing regression target")
    if any(set(frame.loc[frame.case_id == case_id, "rule"]) != {1, 2, 3}
           for case_id in range(1, 101)):
        raise RuntimeError("Each case must have exactly Rules 1, 2, and 3")

    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(frame, groups=frame["case_id"]))
    fold_of_case = {}
    for fold, (train, valid) in enumerate(folds, 1):
        train_cases = set(frame.iloc[train].case_id)
        valid_cases = set(frame.iloc[valid].case_id)
        if train_cases & valid_cases or len(valid_cases) != 20:
            raise RuntimeError("Grouped fold leakage or imbalance")
        fold_of_case.update({case_id: fold for case_id in valid_cases})
    write(output / "fold_assignments.json", [
        {"case_id": case_id, "fold": fold_of_case[case_id]} for case_id in range(1, 101)
    ])

    designs = {}
    design_columns = {}
    for key, names in (("raw9", RAW9), ("raw9_plus_physics6", RAW9 + PHYSICS6)):
        designs[key], design_columns[key] = conditioned_design(frame, names)

    configs = [
        ("dummy_mean", "rule_only", DummyRegressor(strategy="mean")),
        ("rule_mean", "rule_only", None),
        ("ridge", "raw9", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("ridge", "raw9_plus_physics6", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        ("random_forest", "raw9", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=1)),
        ("random_forest", "raw9_plus_physics6", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt",
            random_state=SEED, n_jobs=1)),
    ]

    prediction_rows = []
    fold_metric_rows = []
    aggregate_rows = []
    for target in TARGETS:
        y = frame[target].to_numpy(dtype=float)
        rules = frame["rule"].to_numpy(dtype=int)
        for model_name, feature_set, estimator in configs:
            validation = np.full(len(frame), np.nan)
            train_true = []
            train_pred = []
            for fold, (train, valid) in enumerate(folds, 1):
                if model_name == "dummy_mean":
                    model = clone(estimator).fit(np.zeros((len(train), 1)), y[train])
                    p_train = model.predict(np.zeros((len(train), 1)))
                    p_valid = model.predict(np.zeros((len(valid), 1)))
                elif model_name == "rule_mean":
                    p_train = rule_mean_predict(y[train], rules[train], rules[train])
                    p_valid = rule_mean_predict(y[train], rules[train], rules[valid])
                else:
                    x = designs[feature_set]
                    model = clone(estimator).fit(x[train], y[train])
                    p_train = model.predict(x[train])
                    p_valid = model.predict(x[valid])
                validation[valid] = p_valid
                train_true.extend(y[train])
                train_pred.extend(p_train)
                for split, indices, predicted in (
                    ("train", train, p_train), ("validation", valid, p_valid)):
                    fold_metric_rows.append({
                        "target": target, "model": model_name,
                        "feature_set": feature_set, "fold": fold, "split": split,
                        "scope": "overall", **metrics(y[indices], predicted),
                    })
                    for rule in (1, 2, 3):
                        mask = rules[indices] == rule
                        fold_metric_rows.append({
                            "target": target, "model": model_name,
                            "feature_set": feature_set, "fold": fold, "split": split,
                            "scope": f"rule_{rule}",
                            **metrics(y[indices][mask], predicted[mask]),
                        })
            if not np.isfinite(validation).all():
                raise RuntimeError("Incomplete out-of-fold predictions")
            for index, predicted in enumerate(validation):
                prediction_rows.append({
                    "case_id": int(frame.iloc[index].case_id),
                    "rule": int(rules[index]), "fold": fold_of_case[int(frame.iloc[index].case_id)],
                    "target": target, "model": model_name, "feature_set": feature_set,
                    "actual": float(y[index]), "predicted": float(predicted),
                    "residual": float(predicted - y[index]),
                })
            for split, actual, predicted in (
                ("train", np.asarray(train_true), np.asarray(train_pred)),
                ("validation", y, validation)):
                aggregate_rows.append({
                    "target": target, "model": model_name, "feature_set": feature_set,
                    "split": split, "scope": "overall", **metrics(actual, predicted),
                })
                if split == "validation":
                    for rule in (1, 2, 3):
                        mask = rules == rule
                        aggregate_rows.append({
                            "target": target, "model": model_name, "feature_set": feature_set,
                            "split": split, "scope": f"rule_{rule}",
                            **metrics(y[mask], validation[mask]),
                        })

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    aggregates = pd.DataFrame(aggregate_rows)
    rule_baseline = fold_metrics[(fold_metrics.model == "rule_mean") &
                                 (fold_metrics.feature_set == "rule_only") &
                                 (fold_metrics.split == "validation") &
                                 (fold_metrics.scope == "overall")]
    comparisons = []
    for (target, model, feature_set), subset in fold_metrics[
            (fold_metrics.split == "validation") & (fold_metrics.scope == "overall") &
            (~fold_metrics.model.isin(["dummy_mean", "rule_mean"]))].groupby(
                ["target", "model", "feature_set"]):
        baseline = rule_baseline[rule_baseline.target == target].set_index("fold")
        candidate = subset.set_index("fold")
        row = {"target": target, "model": model, "feature_set": feature_set}
        for metric in ("mae", "rmse", "r2"):
            delta = candidate[metric] - baseline[metric]
            row[f"delta_{metric}_candidate_minus_rule_baseline_mean"] = float(delta.mean())
            row[f"folds_better_{metric}"] = int((delta < 0).sum() if metric != "r2" else (delta > 0).sum())
        pooled = aggregates[(aggregates.target == target) & (aggregates.model == model) &
                            (aggregates.feature_set == feature_set) &
                            (aggregates.split == "validation") & (aggregates.scope == "overall")].iloc[0]
        pooled_base = aggregates[(aggregates.target == target) & (aggregates.model == "rule_mean") &
                                 (aggregates.split == "validation") &
                                 (aggregates.scope == "overall")].iloc[0]
        row.update({f"pooled_delta_{m}_candidate_minus_rule_baseline": float(pooled[m] - pooled_base[m])
                    for m in ("mae", "rmse", "r2")})
        comparisons.append(row)

    pairwise = []
    for (target, model, feature_set), subset in predictions.groupby(["target", "model", "feature_set"]):
        table = subset.pivot(index="case_id", columns="rule", values=["actual", "predicted"])
        for a, b in combinations((1, 2, 3), 2):
            actual_delta = table["actual", a] - table["actual", b]
            predicted_delta = table["predicted", a] - table["predicted", b]
            non_ties = actual_delta != 0
            pairwise.append({
                "target": target, "model": model, "feature_set": feature_set,
                "rule_pair": f"{a}_minus_{b}", "n": 100,
                "actual_exact_ties": int((~non_ties).sum()),
                "delta_mae": float(mean_absolute_error(actual_delta, predicted_delta)),
                "delta_rmse": float(mean_squared_error(actual_delta, predicted_delta) ** 0.5),
                "delta_r2": float(r2_score(actual_delta, predicted_delta)),
                "pearson_delta": float(np.corrcoef(actual_delta, predicted_delta)[0, 1])
                    if np.std(actual_delta) and np.std(predicted_delta) else None,
                "direction_agreement_non_ties": float(np.mean(
                    np.sign(actual_delta[non_ties]) == np.sign(predicted_delta[non_ties])))
                    if non_ties.any() else None,
            })

    write(output / "dataset_manifest.json", {
        "rows": 300, "unique_case_ids": 100, "rules": [1, 2, 3],
        "raw9": RAW9, "physics6_added": PHYSICS6,
        "targets": TARGETS, "source_directory": "feature_engineering_100_v1",
        "second_batch_used": False,
        "rule_conditioning": "fixed one-hot rule indicators and rule-by-feature interactions",
    })
    write(output / "predictions.json", prediction_rows)
    write(output / "fold_metrics.json", fold_metric_rows)
    write(output / "metrics.json", aggregate_rows)
    write(output / "baseline_comparisons.json", comparisons)
    write(output / "pairwise_rule_differences.json", pairwise)
    write(output / "provenance.json", {
        "source_sha256": before, "analysis_script_sha256": digest(Path(__file__)),
        "random_seed": SEED,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    })
    if any(digest(root / name) != value for name, value in before.items()):
        raise RuntimeError("Source data changed during analysis")
    print(json.dumps({
        "rows": len(frame), "fold_sizes": [len(valid) for _, valid in folds],
        "metrics": [row for row in aggregate_rows if row["split"] == "validation" and row["scope"] == "overall"],
        "baseline_comparisons": comparisons,
    }, indent=2))


if __name__ == "__main__":
    main()
