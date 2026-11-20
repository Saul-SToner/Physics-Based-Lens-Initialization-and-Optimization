import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "independent_f_number_test_batch2_v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_independence_and_protocol_audit_passed():
    a = load(OUT / "independence_protocol_audit.json")
    assert a["passed"] is True
    assert a["prescription_fingerprint_overlap_count"] == 0
    assert a["raw_input_exact_overlap_count"] == 0
    assert a["feature_schema_equal"] and a["core_optimizer_source_hashes_equal"]
    assert a["rules_equal"] and a["runtime_policy_equal"]


def test_frozen_manifest_is_unchanged():
    p = load(OUT / "provenance.json")
    freeze = ROOT / p["freeze_manifest"]
    assert sha(freeze) == p["freeze_sha256"]


def test_predictions_are_complete_and_disjoint():
    rows = load(OUT / "predictions.json")
    assert len(rows) == 300
    assert {x["case_id"] for x in rows} == set(range(101, 201))
    assert len({(x["case_id"], x["rule"]) for x in rows}) == 300
    assert all(set(x) == {1, 2, 3} for x in
               ({r["rule"] for r in rows if r["case_id"] == cid} for cid in range(101, 201)))


def test_no_test_label_reuse():
    p = load(OUT / "provenance.json")
    assert p["models_fit_before_test_labels_loaded"] is True
    assert p["test_labels_used_for_selection_or_refit"] is False
    assert p["stage_aware_excluded"] is True


def test_prespecified_primary_retained_without_reselection():
    freeze = load(ROOT / "model_protocol_freeze_batch1_to_batch2_v1.json")
    primary = [x for x in freeze["models"] if x["role"] == "primary"]
    assert len(primary) == 1 and primary[0]["name"] == "rf_raw9_leaf1"
    assert freeze["test_policy"].startswith("single evaluation")

