import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "third_batch_followup_237_validation_v1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exact_followup_scope():
    rows = load("predictions_and_actuals.json")
    assert len(rows) == 711
    assert {x["prescription_id"] for x in rows} == set(range(301, 538))
    assert len({(x["prescription_id"], x["rule"]) for x in rows}) == 711


def test_frozen_prediction_artifacts_unchanged():
    receipt = json.loads((ROOT / "third_batch_validation_freeze_v1/freeze_receipt.json").read_text())
    assert all(sha(ROOT / "third_batch_validation_freeze_v1" / name) == expected
               for name, expected in receipt["artifact_sha256"].items())


def test_no_refit_and_remaining_cases_unread():
    p = load("provenance.json")
    assert p["model_refit"] is False and p["hyperparameters_changed"] is False
    assert p["evaluated_cases"] == [301, 537]
    assert p["remaining_cases_not_read"] == [538, 700]


def test_protocol_unchanged():
    p = load("provenance.json")
    assert sha(ROOT / p["protocol"]) == p["protocol_sha256"]


def test_decision_counts_complete():
    d = load("decision_summary.json")
    counts = d["benefit_equal_harm_counts"]
    assert sum(counts.values()) == 237
    assert sum(d["selection_counts"].values()) == 237

