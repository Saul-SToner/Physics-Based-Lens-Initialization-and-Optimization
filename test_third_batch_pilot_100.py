import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "third_batch_pilot_100_validation_v1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_scope_is_exactly_first_100_new_prescriptions():
    rows = load("predictions_and_actuals.json")
    assert len(rows) == 300
    assert {x["prescription_id"] for x in rows} == set(range(201, 301))
    assert len({(x["prescription_id"], x["rule"]) for x in rows}) == 300


def test_frozen_predictions_are_preoptimization_artifacts():
    receipt = json.loads((ROOT / "third_batch_validation_freeze_v1/freeze_receipt.json").read_text())
    for name, expected in receipt["artifact_sha256"].items():
        assert sha(ROOT / "third_batch_validation_freeze_v1" / name) == expected


def test_no_refit_and_remaining_scope_unread():
    p = load("provenance.json")
    assert p["model_refit"] is False and p["hyperparameters_changed"] is False
    assert p["evaluated_cases"] == [201, 300]
    assert p["remaining_cases_not_read"] == [301, 700]
    assert all("case_2" in path or "case_300" in path for path in p["result_sha256"])


def test_protocol_hash_unchanged():
    p = load("provenance.json")
    assert sha(ROOT / p["protocol"]) == p["protocol_sha256"]


def test_decision_counts_complete():
    d = load("decision_summary.json")
    counts = d["benefit_equal_harm_counts"]
    assert sum(counts.values()) == 100
    assert sum(d["selection_counts"].values()) == 100

