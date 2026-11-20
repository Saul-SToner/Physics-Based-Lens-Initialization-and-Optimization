import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rule2_stage_switch_mve_v1"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_mechanism_and_case_scope():
    audit = load("mechanism_audit.json")
    assert audit["count"] == 31 and audit["common_mechanism_confirmed"]
    assert all(1 <= x["case_id"] <= 200 for x in audit["cases"])
    assert all(not x["entered_aperture"] for x in audit["cases"])


def test_ten_frozen_representatives_and_runs():
    selected = load("selected_cases.json")
    receipt = load("run_receipt.json")
    assert len(selected) == 10 and len({x["cluster"] for x in selected}) == 10
    assert receipt["completed_cases"] == 10 and receipt["case_201_plus_used"] is False


def test_protocol_and_historical_sources_unchanged():
    protocol = load("protocol_freeze.json")
    assert sha(OUT / "protocol_freeze.json") == load("run_receipt.json")["protocol_sha256"]
    assert all(sha(ROOT / name) == expected for name, expected in protocol["source_sha256"].items())
    assert all(sha(ROOT / name) == expected
               for name, expected in protocol["historical_result_and_lens_sha256"].items())


def test_all_intervention_endpoints_pass_acceptance():
    summary = load("summary.json")
    assert summary["intervention"]["all_final_endpoints_passed_original_acceptance"] is True
    assert summary["intervention"]["cases_run"] == 10


def test_comparison_rows_complete():
    rows = load("comparison_rows.json")
    assert len(rows) == 30
    assert all({x["method"] for x in rows if x["case_id"] == cid} ==
               {"historical_rule_2", "historical_rule_3", "rule_2_stage_switch"}
               for cid in {x["case_id"] for x in rows})

