import hashlib
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_rule_regression_200_v2"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_dataset_cardinality_and_identity():
    d = load("dataset_manifest.json")
    assert d["prescription_count"] == 200 and d["rule_conditioned_rows"] == 600
    assert len(d["prescription_sha256"]) == 200
    assert len({x["prescription_sha256"] for x in d["prescription_sha256"]}) == 200


def test_grouped_folds_have_no_leakage():
    folds = load("fold_assignments.json")
    assert len(folds) == 5
    seen = []
    for fold in folds:
        train, valid = set(fold["train_prescriptions"]), set(fold["validation_prescriptions"])
        assert not train & valid and len(train) == 160 and len(valid) == 40
        seen.extend(valid)
    assert sorted(seen) == list(range(1, 201))


def test_predictions_complete():
    rows = load("predictions_oof.json")
    assert len(rows) == 1200
    assert len({(x["prescription_id"], x["rule"], x["target"]) for x in rows}) == 1200
    assert all(1 <= x["fold"] <= 5 for x in rows)


def test_final_models_load_and_match_config():
    config = load("config.json")
    assert config["model"]["min_samples_leaf"] == 1
    assert config["model"]["n_estimators"] == 500
    assert config["new_independent_test_available"] is False
    for target in config["targets"]:
        model = joblib.load(OUT / f"model_{target}.joblib")
        assert model.n_estimators == 500 and model.min_samples_leaf == 1


def test_model_and_source_hashes():
    p = load("provenance.json")
    assert all(sha(OUT / name) == value for name, value in p["model_sha256"].items())
    assert all(sha(ROOT / name) == value for name, value in p["source_sha256"].items())
    assert p["old_outputs_overwritten"] is False and p["new_independent_test_result"] is False

