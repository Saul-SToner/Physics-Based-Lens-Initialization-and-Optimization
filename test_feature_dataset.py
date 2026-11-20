from copy import deepcopy
import json
from pathlib import Path
import unittest

from build_feature_dataset import build_initial_features, pareto_rules


class FeatureDatasetTests(unittest.TestCase):
    def test_features_are_rebuilt_from_initial_records_only(self):
        root = Path(__file__).resolve().parent
        candidates = json.loads((root/"random_candidates_100/candidates.json").read_text())["candidates"]
        exported = json.loads((root/"feature_engineering_100_v1/features.json").read_text())
        for source, record in zip(candidates,exported):
            schema = {}
            row = build_initial_features(source,schema)
            self.assertEqual(record["case_id"],source["id"])
            self.assertEqual(row,record["features"])
            self.assertEqual(len(row),35)
            self.assertTrue(all(value["available_before_rule_selection"] for value in schema.values()))
        changed = deepcopy(candidates[0])
        changed["optimized_mtf"] = 999
        changed["optimized_cost"] = -1
        self.assertEqual(build_initial_features(changed,{}),build_initial_features(candidates[0],{}))

    def test_pareto_tradeoffs_and_ties_are_retained(self):
        def row(rule,f_number,field):
            return {"rule":rule,"has_accepted_endpoint":True,
                    "final_working_f_number":f_number,"final_max_field_deg":field}
        self.assertEqual(pareto_rules([row(1,20,2),row(2,30,5),row(3,30,5)]),[1,2,3])
        self.assertEqual(pareto_rules([row(1,30,2),row(2,20,5),row(3,20,5)]),[2,3])

    def test_exported_targets_and_labels_have_one_case_group(self):
        root=Path(__file__).resolve().parent/"feature_engineering_100_v1"
        features=json.loads((root/"features.json").read_text())
        outcomes=json.loads((root/"outcomes.json").read_text())
        labels=json.loads((root/"labels.json").read_text())
        self.assertEqual(len(features),100)
        self.assertEqual(len(outcomes),300)
        self.assertEqual(len(labels),100)
        for feature,label in zip(features,labels):
            case=feature["case_id"]
            rows=[row for row in outcomes if row["case_id"]==case]
            self.assertEqual(sorted(row["rule"] for row in rows),[1,2,3])
            self.assertEqual(feature["prescription_sha256"],label["prescription_sha256"])
            self.assertTrue(all(row["prescription_sha256"]==feature["prescription_sha256"] for row in rows))
            successful=[row for row in rows if row["target_reached"]]
            self.assertEqual(label["target_success_mask"],sum(1<<(row["rule"]-1) for row in successful))
            self.assertEqual(label["pareto_endpoint_rules"],pareto_rules(rows))
            self.assertFalse(any(name.startswith("final_") or name in ("cost","target_reached","fields")
                                 for name in feature["features"]))


if __name__=="__main__":
    unittest.main()
