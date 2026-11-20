import json
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_rule_regression_100_v1"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


class RuleRegressionAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = read("predictions.json")
        cls.metrics = read("metrics.json")
        cls.manifest = read("dataset_manifest.json")

    def test_dataset_scope_and_feature_contract(self):
        self.assertEqual(self.manifest["rows"], 300)
        self.assertEqual(self.manifest["unique_case_ids"], 100)
        self.assertFalse(self.manifest["second_batch_used"])
        self.assertEqual(len(self.manifest["raw9"]), 9)
        self.assertEqual(len(self.manifest["physics6_added"]), 6)
        provenance = read("provenance.json")
        self.assertTrue(all(name.replace("\\", "/").startswith("feature_engineering_100_v1/")
                            for name in provenance["source_sha256"]))

    def test_grouped_folds_have_no_case_leakage(self):
        assignments = read("fold_assignments.json")
        self.assertEqual(len(assignments), 100)
        self.assertEqual({row["case_id"] for row in assignments}, set(range(1, 101)))
        counts = {fold: sum(row["fold"] == fold for row in assignments) for fold in range(1, 6)}
        self.assertEqual(counts, {fold: 20 for fold in range(1, 6)})
        mapping = {row["case_id"]: row["fold"] for row in assignments}
        self.assertTrue(all(row["fold"] == mapping[row["case_id"]] for row in self.predictions))

    def test_each_configuration_has_one_oof_prediction_per_case_rule(self):
        configurations = {(row["target"], row["model"], row["feature_set"])
                          for row in self.predictions}
        self.assertEqual(len(configurations), 12)
        for configuration in configurations:
            rows = [row for row in self.predictions
                    if (row["target"], row["model"], row["feature_set"]) == configuration]
            self.assertEqual(len(rows), 300)
            self.assertEqual(len({(row["case_id"], row["rule"]) for row in rows}), 300)

    def test_pooled_metrics_recompute_exactly(self):
        for target, model, feature_set in (
            ("final_f_number", "random_forest", "raw9"),
            ("final_field_deg", "ridge", "raw9_plus_physics6"),
        ):
            rows = [row for row in self.predictions if row["target"] == target and
                    row["model"] == model and row["feature_set"] == feature_set]
            actual = np.asarray([row["actual"] for row in rows])
            predicted = np.asarray([row["predicted"] for row in rows])
            metric = next(row for row in self.metrics if row["target"] == target and
                          row["model"] == model and row["feature_set"] == feature_set and
                          row["split"] == "validation" and row["scope"] == "overall")
            self.assertAlmostEqual(metric["mae"], mean_absolute_error(actual, predicted))
            self.assertAlmostEqual(metric["rmse"], mean_squared_error(actual, predicted) ** 0.5)
            self.assertAlmostEqual(metric["r2"], r2_score(actual, predicted))


if __name__ == "__main__":
    unittest.main()
