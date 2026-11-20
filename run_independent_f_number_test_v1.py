"""Frozen one-shot batch-2 F/# test trained exclusively on batch 1."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from build_feature_dataset import build_initial_features
from prescription_registry import sample_fingerprint
from run_rule_regression_v1 import RAW9, conditioned_design

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "independent_f_number_test_batch2_v1"
SEED = 20261118


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def metric(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** 0.5),
            "r2": float(r2_score(y, p)) if len(y) > 1 and not np.all(y == y[0]) else None}


def build_train():
    features = read(ROOT / "feature_engineering_100_v1/features.json")
    outcomes = read(ROOT / "feature_engineering_100_v1/outcomes.json")
    by_case = {int(x["case_id"]): x for x in features}
    rows = []
    for outcome in outcomes:
        cid = int(outcome["case_id"])
        source = by_case[cid]
        if source["prescription_sha256"] != outcome["prescription_sha256"]:
            raise RuntimeError("Training identity mismatch")
        rows.append({"case_id": cid, "rule": int(outcome["rule"]), **source["features"],
                     "target": float(outcome["final_working_f_number"])})
    frame = pd.DataFrame(rows).sort_values(["case_id", "rule"]).reset_index(drop=True)
    if len(frame) != 300 or set(frame.case_id) != set(range(1, 101)):
        raise RuntimeError("Frozen training set mismatch")
    return frame, features


def audit_and_build_test_features(train_features):
    source_a = read(ROOT / "random_candidates_100/candidates.json")
    source_all = read(ROOT / "random_candidates_200/candidates.json")
    plan_a = read(ROOT / "background_runs/batch_300_20261118/plan.json")
    plan_b = read(ROOT / "background_runs/batch_300_20261118_02/plan.json")
    state_a = read(ROOT / "background_runs/batch_300_20261118/state.json")
    state_b = read(ROOT / "background_runs/batch_300_20261118_02/state.json")
    test_candidates = sorted(plan_b["cases"], key=lambda x: int(x["id"]))
    if [int(x["id"]) for x in test_candidates] != list(range(101, 201)):
        raise RuntimeError("Test IDs are not the frozen 101-200 interval")
    if [x["inputs"] for x in source_all["candidates"][:100]] != [x["inputs"] for x in source_a["candidates"]]:
        raise RuntimeError("Combined candidate stream does not preserve batch 1")
    train_fp = {x["prescription_sha256"] for x in train_features}
    test_fp, test_input_tuples, schemas, records = set(), set(), {}, []
    for candidate in test_candidates:
        fp, _ = sample_fingerprint(candidate)
        if fp in test_fp:
            raise RuntimeError("Duplicate inside test batch")
        test_fp.add(fp)
        test_input_tuples.add(tuple(candidate["inputs"]))
        features = build_initial_features(candidate, schemas)
        for rule in (1, 2, 3):
            records.append({"case_id": int(candidate["id"]), "rule": rule,
                            "prescription_sha256": fp, **features})
    train_inputs = {tuple(x["inputs"]) for x in source_a["candidates"]}
    overlap = train_fp & test_fp
    exact_input_overlap = train_inputs & test_input_tuples
    train_dict = read(ROOT / "feature_engineering_100_v1/dictionary.json")["features"]
    schema_fields = ("unit", "definition", "dependencies", "group", "available_before_rule_selection")
    schema_equal = set(schemas) == set(train_dict) and all(
        all(schemas[name][field] == train_dict[name][field] for field in schema_fields)
        for name in schemas)
    core_sources = ("run_p1.py", "p1_engine.py", "analyze_opticstudio.py",
                    "lens_geometry.py", "compare_opticstudio.py")
    core_equal = all(plan_a["source_sha256"][x] == plan_b["source_sha256"][x] for x in core_sources)
    wrapper_differences = {x: {"batch1": plan_a["source_sha256"].get(x),
                               "batch2": plan_b["source_sha256"].get(x)}
                           for x in set(plan_a["source_sha256"]) | set(plan_b["source_sha256"])
                           if plan_a["source_sha256"].get(x) != plan_b["source_sha256"].get(x)}
    audit = {
        "passed": not overlap and not exact_input_overlap and schema_equal and core_equal
                  and plan_a["rules"] == plan_b["rules"] and state_a["runtime_policy"] == state_b["runtime_policy"],
        "train_ids": [1, 100], "test_ids": [101, 200],
        "prescription_fingerprint_overlap_count": len(overlap),
        "raw_input_exact_overlap_count": len(exact_input_overlap),
        "near_duplicate_policy": "No arbitrary distance threshold. Sampling declares no decimal quantization; comparison at original serialized floating-point precision is exact component-wise identity.",
        "same_seed_continuation": source_a["seed"] == source_all["seed"],
        "test_is_disjoint_continuation_of_sampling_stream": True,
        "sampling_ranges_equal": source_a["sampling_ranges"] == source_all["sampling_ranges"],
        "sampling_policy_equal": source_a["sampling_policy"] == source_all["sampling_policy"],
        "feature_schema_equal": schema_equal,
        "raw9_equal_and_ordered": RAW9 == read(ROOT / "model_protocol_freeze_batch1_to_batch2_v1.json")["input_features"],
        "rules_equal": plan_a["rules"] == plan_b["rules"],
        "core_optimizer_source_hashes_equal": core_equal,
        "runtime_policy_equal": state_a["runtime_policy"] == state_b["runtime_policy"],
        "execution_sources_equal": state_a.get("execution_sources") == state_b.get("execution_sources"),
        "group_timeout_equal": plan_a["group_timeout_seconds"] == plan_b["group_timeout_seconds"],
        "infrastructure_wrapper_hash_differences": wrapper_differences,
        "interpretation": "The test prescriptions are disjoint pseudorandom-stream continuation samples, not a newly seeded independent realization. Core optical optimization protocol is identical; background orchestration wrappers changed without changing the frozen runtime policy or core optimizer hashes."
    }
    if not audit["passed"] or not audit["raw9_equal_and_ordered"]:
        raise RuntimeError("Independence/protocol audit failed; labels were not read")
    return pd.DataFrame(records).sort_values(["case_id", "rule"]).reset_index(drop=True), audit, state_b


def fit_frozen_models(train):
    x, _ = conditioned_design(train, RAW9)
    y = train.target.to_numpy(dtype=float)
    models = {}
    for name, leaf in (("rf_raw9_leaf1", 1), ("rf_raw9_leaf3", 3)):
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=leaf,
                                      max_depth=None, max_features="sqrt",
                                      random_state=SEED, n_jobs=1)
        models[name] = model.fit(x, y)
    rule_means = {rule: float(train.loc[train.rule == rule, "target"].mean()) for rule in (1, 2, 3)}
    return models, rule_means


def load_test_labels(test, state):
    # Deliberately called only after every frozen model has been fit.
    labels, hashes = [], {}
    base = ROOT / "background_runs/batch_300_20261118_02"
    for row in test[["case_id", "rule"]].itertuples(index=False):
        key = f"case_{row.case_id:03d}_rule_{row.rule}"
        info = state["jobs"][key]
        path = base / info["attempt"] / "experiment/result.json"
        value_hash = digest(path)
        if value_hash != info["acceptance"]["result_sha256"]:
            raise RuntimeError("Test result hash mismatch")
        result = read(path)
        if (int(result["case_id"]), int(result["rule"])) != (row.case_id, row.rule):
            raise RuntimeError("Test feature/target identity mismatch")
        final = result["final_accepted"]
        if final is None:
            raise RuntimeError("Missing accepted test endpoint")
        labels.append(float(final["working_f_number"]))
        hashes[str(path.relative_to(ROOT))] = value_hash
    return np.asarray(labels), hashes


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    freeze_path = ROOT / "model_protocol_freeze_batch1_to_batch2_v1.json"
    freeze_hash_before = digest(freeze_path)
    train, train_features = build_train()
    test, audit, state_b = audit_and_build_test_features(train_features)
    models, rule_means = fit_frozen_models(train)
    y_test, result_hashes = load_test_labels(test, state_b)
    x_test, _ = conditioned_design(test, RAW9)
    predictions = test[["case_id", "rule", "prescription_sha256"]].copy()
    predictions["actual"] = y_test
    predictions["is_f500"] = np.isclose(y_test, 500.0, rtol=0, atol=1e-12)
    predictions["rf_raw9_leaf1"] = models["rf_raw9_leaf1"].predict(x_test)
    predictions["rf_raw9_leaf3"] = models["rf_raw9_leaf3"].predict(x_test)
    predictions["rule_conditioned_dummy"] = [rule_means[int(r)] for r in test.rule]
    rows = []
    for model in ("rf_raw9_leaf1", "rf_raw9_leaf3", "rule_conditioned_dummy"):
        pred = predictions[model].to_numpy(dtype=float)
        scopes = [("overall", np.ones(len(test), dtype=bool)),
                  ("f_number_500", predictions.is_f500.to_numpy()),
                  ("non_f_number_500", ~predictions.is_f500.to_numpy())]
        scopes += [(f"rule_{rule}", test.rule.to_numpy() == rule) for rule in (1, 2, 3)]
        for scope, mask in scopes:
            rows.append({"model": model, "scope": scope, **metric(y_test[mask], pred[mask])})
    if digest(freeze_path) != freeze_hash_before:
        raise RuntimeError("Frozen protocol changed during evaluation")
    source_paths = [ROOT / "feature_engineering_100_v1/features.json",
                    ROOT / "feature_engineering_100_v1/outcomes.json",
                    ROOT / "random_candidates_100/candidates.json",
                    ROOT / "random_candidates_200/candidates.json",
                    ROOT / "background_runs/batch_300_20261118/plan.json",
                    ROOT / "background_runs/batch_300_20261118_02/plan.json",
                    ROOT / "background_runs/batch_300_20261118/state.json",
                    ROOT / "background_runs/batch_300_20261118_02/state.json"]
    write(OUT / "independence_protocol_audit.json", audit)
    write(OUT / "predictions.json", predictions.to_dict("records"))
    write(OUT / "metrics.json", rows)
    write(OUT / "provenance.json", {"freeze_manifest": str(freeze_path.relative_to(ROOT)),
                                     "freeze_sha256": freeze_hash_before,
                                     "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                                     "test_result_sha256": result_hashes,
                                     "models_fit_before_test_labels_loaded": True,
                                     "test_labels_used_for_selection_or_refit": False,
                                     "stage_aware_excluded": True,
                                     "zemax_run": False})
    print(json.dumps({"audit": audit, "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
