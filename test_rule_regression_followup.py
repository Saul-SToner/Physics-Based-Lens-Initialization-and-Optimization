import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_rule_regression_followup_100_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


class FollowupAudit(unittest.TestCase):
    def test_f500_is_verified_and_rule2_only(self):
        audit = read("f500_audit.json")
        self.assertEqual(audit["count"], 11)
        self.assertEqual(audit["rule_counts"], {"2": 11})
        self.assertTrue(all(row["geometry_valid"] and row["all_mtf_pass"] and
                            row["saved_final_verified"] for row in audit["cases"]))
        self.assertTrue(all(row["stop_reasons"] == {
            "field": "step_limit", "aperture": "not_started_first_stage_limit"
        } for row in audit["cases"]))

    def test_followup_predictions_preserve_grouped_folds(self):
        assignments = {row["case_id"]: row["fold"] for row in
                       json.loads((ROOT / "ml_rule_regression_100_v1/fold_assignments.json").read_text())}
        for name in ("log_target_predictions.json", "rf_sensitivity_predictions.json"):
            rows = read(name)
            self.assertTrue(all(row["fold"] == assignments[row["case_id"]] for row in rows))

    def test_log_and_sensitivity_grid_are_complete(self):
        log_rows = pd.DataFrame(read("log_target_predictions.json"))
        self.assertEqual(len(log_rows), 2 * 4 * 300)
        self.assertEqual(log_rows.groupby(["target", "model", "feature_set"]).size().unique().tolist(), [300])
        rf = pd.DataFrame(read("rf_sensitivity_predictions.json"))
        self.assertEqual(len(rf), 2 * 9 * 300)
        self.assertEqual(rf.groupby(["target", "max_depth", "min_samples_leaf"], dropna=False).size().unique().tolist(), [300])

    def test_source_hashes_and_scope(self):
        provenance = read("provenance.json")
        self.assertFalse(provenance["second_batch_used"])
        self.assertFalse(provenance["new_zemax_run"])
        for name, expected in provenance["sources_sha256"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)


if __name__ == "__main__":
    unittest.main()
