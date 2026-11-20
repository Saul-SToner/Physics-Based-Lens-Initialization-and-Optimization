import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_v2_oof_error_decision_analysis"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_oof_rows_and_decisions_complete():
    rows, decisions = load("analysis_rows.json"), load("decision_by_prescription.json")
    assert len(rows) == 600 and len(decisions) == 200
    assert len({(x["prescription_id"], x["rule"]) for x in rows}) == 600
    assert {x["prescription_id"] for x in decisions} == set(range(1, 201))


def test_selection_uses_oof_only():
    p = load("provenance.json")
    assert p["oof_only"] and p["prescription_grouped_outer_predictions"]
    assert p["decision_uses_only_oof_predictions_at_selection_time"]
    assert p["actual_endpoints_used_only_for_post_selection_scoring"]
    assert p["no_new_model_fit"] and p["no_zemax_run"]


def test_source_hashes_unchanged():
    assert all(sha(ROOT / path) == value for path, value in load("provenance.json")["source_sha256"].items())


def test_policy_metrics_consistent():
    d = load("decision_summary.json")
    assert 0 <= d["exact_oracle_rule_set_accuracy"] <= 1
    assert 0 <= d["selected_rule_in_true_pareto_fraction"] <= 1
    assert d["mean_gap_regret"] >= 0
    assert sum(d["choice_counts"].values()) == 200


def test_bootstrap_unit_is_prescription():
    u = load("decision_uncertainty.json")
    assert u["bootstrap_unit"] == "prescription"
    assert u["bootstrap_replicates"] == 20000
    assert u["prescriptions_better"] + u["prescriptions_equal"] + u["prescriptions_worse"] == 200

