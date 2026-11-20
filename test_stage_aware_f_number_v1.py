import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_stage_aware_f_number_100_v1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_stage_audit_is_observed_not_assumed():
    audit = load("stage_log_audit.json")
    assert audit["no_equivalence_assumed"] is True
    assert audit["observed_equivalence_in_batch_1_only"] is True
    assert sum(x["n"] for x in audit["contingency"]) == 300


def test_oof_predictions_complete_and_fold_grouped():
    rows = load("stage_predictions.json")
    assert len(rows) == 300
    assert len({(x["case_id"], x["rule"]) for x in rows}) == 300
    assert all({x["fold"] for x in rows if x["case_id"] == cid}.__len__() == 1 for cid in range(1, 101))


def test_validation_labels_never_used_for_prediction():
    audit = load("provenance.json")["leakage_audit"]
    assert all(x["validation_stage_used_for_prediction"] is False for x in audit)
    assert all(not (set(x["classifier_fit_rows"]) & set(x["validation_rows"])) for x in audit)
    assert all(not (set(x["regressor_fit_rows"]) & set(x["validation_rows"])) for x in audit)


def test_historical_sources_unchanged():
    assert load("provenance.json")["source_sha256"] == load("source_hashes_after.json")


def test_direct_leaf1_is_retained():
    stability = load("stability.json")
    metrics = load("regression_metrics.json")
    overall = {x["model"]: x for x in metrics if x["scope"] == "overall"}
    assert stability["stable_training_supported"] is False
    assert overall["rf9_leaf1"]["rmse"] < overall["stage_aware_hard"]["rmse"]
