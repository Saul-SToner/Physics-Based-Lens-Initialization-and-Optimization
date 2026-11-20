"""One-shot independent pilot evaluation on frozen Case 201-300 predictions only."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "background_runs/batch_1500_validation_20261118"
FREEZE = ROOT / "third_batch_validation_freeze_v1"
OUT = ROOT / "third_batch_pilot_100_validation_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    constant = len(y) < 2 or np.all(y == y[0])
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** .5),
            "r2": None if constant else float(r2_score(y, p)),
            "bias_pred_minus_actual": float(np.mean(p - y))}


def gap(f_number, field):
    return np.maximum((np.asarray(f_number) - 20.0) / 20.0, 0.0) + np.maximum((5.0 - np.asarray(field)) / 5.0, 0.0)


def pareto(f_values, field_values):
    result = set()
    for i in range(3):
        dominated = any(f_values[j] <= f_values[i] and field_values[j] >= field_values[i]
                        and (f_values[j] < f_values[i] or field_values[j] > field_values[i])
                        for j in range(3) if j != i)
        if not dominated:
            result.add(i + 1)
    return result


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    protocol = ROOT / "third_batch_pilot_100_protocol.json"
    protocol_hash = digest(protocol)
    receipt = read(FREEZE / "freeze_receipt.json")
    for name, expected in receipt["artifact_sha256"].items():
        if digest(FREEZE / name) != expected:
            raise RuntimeError("Frozen prediction artifact changed")
    predictions = pd.DataFrame(read(FREEZE / "predictions_preoptimization.json"))
    decisions = pd.DataFrame(read(FREEZE / "rule_selection_preoptimization.json"))
    predictions = predictions[predictions.prescription_id.between(201, 300)].copy()
    decisions = decisions[decisions.prescription_id.between(201, 300)].copy()
    if len(predictions) != 300 or len(decisions) != 100:
        raise RuntimeError("Pilot scope mismatch")
    state = read(RUN / "state.json")
    actual_rows, source_hashes = [], {}
    for cid in range(201, 301):
        for rule in (1, 2, 3):
            info = state["jobs"][f"case_{cid:03d}_rule_{rule}"]
            if info["status"] != "completed":
                raise RuntimeError(f"Pilot result incomplete: Case {cid} Rule {rule}")
            path = RUN / info["attempt"] / "experiment/result.json"
            value_hash = digest(path)
            if value_hash != info["acceptance"]["result_sha256"]:
                raise RuntimeError("Accepted pilot result hash mismatch")
            result = read(path)
            final = result["final_accepted"]
            actual_rows.append({"prescription_id": cid, "rule": rule,
                                "actual_final_f_number": float(final["working_f_number"]),
                                "actual_final_field_deg": float(final["max_field_deg"]),
                                "target_reached": bool(result["target_reached"])})
            source_hashes[str(path.relative_to(ROOT))] = value_hash
    frame = predictions.merge(pd.DataFrame(actual_rows), on=["prescription_id", "rule"], validate="one_to_one")
    frame["is_f500"] = np.isclose(frame.actual_final_f_number, 500.0, rtol=0, atol=1e-12)
    metric_rows = []
    for target, actual, predicted in (("final_f_number", "actual_final_f_number", "predicted_final_f_number"),
                                      ("final_field_deg", "actual_final_field_deg", "predicted_final_field_deg")):
        y, p = frame[actual].to_numpy(), frame[predicted].to_numpy()
        scopes = [("overall", np.ones(len(frame), dtype=bool)),
                  ("f_number_500", frame.is_f500.to_numpy()),
                  ("non_f_number_500", ~frame.is_f500.to_numpy())]
        scopes += [(f"rule_{r}", frame.rule.to_numpy() == r) for r in (1, 2, 3)]
        for scope, mask in scopes:
            metric_rows.append({"target": target, "scope": scope, **metrics(y[mask], p[mask])})

    decision_rows = []
    selected = {int(x.prescription_id): int(x.selected_rule) for x in decisions.itertuples(index=False)}
    for cid, group in frame.groupby("prescription_id"):
        g = group.sort_values("rule")
        af, av = g.actual_final_f_number.to_numpy(), g.actual_final_field_deg.to_numpy()
        gaps = gap(af, av)
        chosen = selected[int(cid)]
        true_pareto = pareto(af, av)
        decision_rows.append({"prescription_id": int(cid), "selected_rule": chosen,
                              "actual_gap_selected": float(gaps[chosen - 1]),
                              "actual_gap_rule_3": float(gaps[2]),
                              "paired_gap_delta_selected_minus_rule_3": float(gaps[chosen - 1] - gaps[2]),
                              "actual_oracle_gap": float(np.min(gaps)),
                              "oracle_regret": float(gaps[chosen - 1] - np.min(gaps)),
                              "selected_rule_in_true_pareto": chosen in true_pareto,
                              "selected_target_reached": bool(g.iloc[chosen - 1].target_reached),
                              "true_pareto_rules": sorted(true_pareto)})
    decision_frame = pd.DataFrame(decision_rows)
    delta = decision_frame.paired_gap_delta_selected_minus_rule_3.to_numpy()
    rng = np.random.default_rng(20261119)
    bootstrap = np.asarray([np.mean(rng.choice(delta, len(delta), replace=True)) for _ in range(20000)])
    decision_summary = {
        "n_prescriptions": 100,
        "primary_mean_paired_gap_delta_selected_minus_rule_3": float(np.mean(delta)),
        "primary_bootstrap_95_percentile_ci": list(map(float, np.quantile(bootstrap, [.025, .975]))),
        "benefit_equal_harm_counts": {"benefit": int(np.sum(delta < -1e-12)),
                                      "equal": int(np.sum(np.abs(delta) <= 1e-12)),
                                      "harm": int(np.sum(delta > 1e-12))},
        "selected_target_success_rate": float(decision_frame.selected_target_reached.mean()),
        "fixed_rule_3_target_success_rate": float(frame[frame.rule == 3].target_reached.mean()),
        "selected_rule_true_pareto_rate": float(decision_frame.selected_rule_in_true_pareto.mean()),
        "mean_oracle_regret": float(decision_frame.oracle_regret.mean()),
        "zero_oracle_regret_rate": float(np.mean(np.isclose(decision_frame.oracle_regret, 0, atol=1e-12))),
        "selection_counts": {str(k): int(v) for k, v in decisions.selected_rule.value_counts().sort_index().items()},
    }
    if digest(protocol) != protocol_hash:
        raise RuntimeError("Pilot protocol changed during evaluation")
    write(OUT / "metrics.json", metric_rows)
    write(OUT / "predictions_and_actuals.json", frame.to_dict("records"))
    write(OUT / "decision_by_prescription.json", decision_rows)
    write(OUT / "decision_summary.json", decision_summary)
    write(OUT / "provenance.json", {"protocol": str(protocol.relative_to(ROOT)),
                                     "protocol_sha256": protocol_hash,
                                     "frozen_prediction_receipt_sha256": digest(FREEZE / "freeze_receipt.json"),
                                     "result_sha256": source_hashes,
                                     "evaluated_cases": [201, 300],
                                     "remaining_cases_not_read": [301, 700],
                                     "model_refit": False, "hyperparameters_changed": False})
    print(json.dumps({"metrics": metric_rows, "decision": decision_summary}, indent=2))


if __name__ == "__main__":
    main()
