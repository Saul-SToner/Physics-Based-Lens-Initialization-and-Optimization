"""Error mechanisms and leakage-safe rule-decision feasibility from V2 OOF predictions."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from build_feature_dataset import build_initial_features
from run_rule_regression_v1 import RAW9

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ml_rule_regression_200_v2"
OUT = ROOT / "ml_v2_oof_error_decision_analysis"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def reg_metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** .5),
            "bias_pred_minus_actual": float(np.mean(p - y)),
            "median_absolute_error": float(np.median(np.abs(p - y))),
            "p90_absolute_error": float(np.quantile(np.abs(p - y), .9)),
            "r2": None if len(y) < 2 or np.all(y == y[0]) else float(r2_score(y, p))}


def target_gap(f_number, field):
    return np.maximum((np.asarray(f_number) - 20.0) / 20.0, 0.0) + np.maximum((5.0 - np.asarray(field)) / 5.0, 0.0)


def argmin_ties(values, tol=1e-12):
    minimum = float(np.min(values))
    return {i + 1 for i, value in enumerate(values) if abs(float(value) - minimum) <= tol}


def pareto_set(f_values, field_values):
    keep = set()
    for i in range(3):
        dominated = any(f_values[j] <= f_values[i] and field_values[j] >= field_values[i]
                        and (f_values[j] < f_values[i] or field_values[j] > field_values[i])
                        for j in range(3) if j != i)
        if not dominated:
            keep.add(i + 1)
    return keep


def load_oof():
    p = pd.DataFrame(read(SOURCE / "predictions_oof.json"))
    f = p[p.target == "final_f_number"].rename(columns={"actual": "actual_f", "predicted": "pred_f"})
    v = p[p.target == "final_field_deg"].rename(columns={"actual": "actual_field", "predicted": "pred_field"})
    keys = ["prescription_id", "rule", "fold"]
    out = f[keys + ["actual_f", "pred_f", "is_f500"]].merge(
        v[keys + ["actual_field", "pred_field"]], on=keys, validate="one_to_one")
    if len(out) != 600:
        raise RuntimeError("Expected 600 aligned OOF rows")
    return out.sort_values(["prescription_id", "rule"]).reset_index(drop=True)


def enrich(frame):
    candidates = {int(x["id"]): x for x in read(ROOT / "random_candidates_200/candidates.json")["candidates"]}
    schema, features = {}, {}
    for cid, candidate in candidates.items():
        features[cid] = build_initial_features(candidate, schema)
    for name in RAW9:
        frame[name] = frame.prescription_id.map(lambda x: features[int(x)][name])
    stage, target_reached, stop_signature = [], [], []
    for row in frame.itertuples(index=False):
        batch = ROOT / ("background_runs/batch_300_20261118" if row.prescription_id <= 100 else
                        "background_runs/batch_300_20261118_02")
        state = read(batch / "state.json")
        info = state["jobs"][f"case_{row.prescription_id:03d}_rule_{row.rule}"]
        result = read(batch / info["attempt"] / "experiment/result.json")
        stage.append(any(x.get("stage") == "aperture" for x in result.get("steps", [])))
        target_reached.append(bool(result["target_reached"]))
        stop_signature.append(";".join(f"{k}={v}" for k, v in sorted(result["stop_reasons"].items())))
    frame["entered_aperture"] = stage
    frame["target_reached"] = target_reached
    frame["stop_signature"] = stop_signature
    frame["f_error"] = frame.pred_f - frame.actual_f
    frame["field_error"] = frame.pred_field - frame.actual_field
    frame["f_abs_error"] = frame.f_error.abs()
    frame["field_abs_error"] = frame.field_error.abs()
    return frame


def error_analysis(frame):
    rows = []
    for target, actual, pred in (("final_f_number", "actual_f", "pred_f"),
                                 ("final_field_deg", "actual_field", "pred_field")):
        scopes = [("overall", np.ones(len(frame), dtype=bool)),
                  ("f_number_500", frame.is_f500.to_numpy()),
                  ("non_f_number_500", ~frame.is_f500.to_numpy()),
                  ("entered_aperture", frame.entered_aperture.to_numpy()),
                  ("not_entered_aperture", ~frame.entered_aperture.to_numpy())]
        scopes += [(f"rule_{r}", frame.rule.to_numpy() == r) for r in (1, 2, 3)]
        for scope, mask in scopes:
            rows.append({"target": target, "scope": scope,
                         **reg_metrics(frame.loc[mask, actual].to_numpy(), frame.loc[mask, pred].to_numpy())})
    stop_rows = []
    for (rule, signature), group in frame.groupby(["rule", "stop_signature"]):
        if len(group) >= 5:
            stop_rows.append({"rule": int(rule), "stop_signature": signature, "n": int(len(group)),
                              "f_mae": float(group.f_abs_error.mean()),
                              "field_mae": float(group.field_abs_error.mean()),
                              "f500_n": int(group.is_f500.sum()),
                              "success_n": int(group.target_reached.sum())})
    correlations = []
    for target_error in ("f_abs_error", "field_abs_error"):
        for name in RAW9:
            rho, p = spearmanr(frame[name], frame[target_error])
            correlations.append({"absolute_error": target_error, "feature": name,
                                 "spearman_rho": float(rho), "p_value_exploratory": float(p)})
    top = frame.sort_values("f_abs_error", ascending=False).head(30)[
        ["prescription_id", "rule", "fold", "actual_f", "pred_f", "f_error", "actual_field",
         "pred_field", "field_error", "is_f500", "entered_aperture", "target_reached", "stop_signature"]]
    return {"metrics": rows, "by_stop_signature_min_n5": sorted(stop_rows, key=lambda x: x["f_mae"], reverse=True),
            "feature_error_correlations_exploratory": sorted(correlations, key=lambda x: abs(x["spearman_rho"]), reverse=True),
            "top_30_f_number_errors": top.to_dict("records"),
            "error_correlation_f_vs_field": float(spearmanr(frame.f_error, frame.field_error).statistic),
            "absolute_error_correlation_f_vs_field": float(spearmanr(frame.f_abs_error, frame.field_abs_error).statistic)}


def decision_analysis(frame):
    decisions, pairwise = [], []
    for cid, group in frame.groupby("prescription_id"):
        g = group.sort_values("rule")
        af, av = g.actual_f.to_numpy(), g.actual_field.to_numpy()
        pf, pv = g.pred_f.to_numpy(), g.pred_field.to_numpy()
        actual_gap, pred_gap = target_gap(af, av), target_gap(pf, pv)
        true_pareto, predicted_pareto = pareto_set(af, av), pareto_set(pf, pv)
        oracle_gap = argmin_ties(actual_gap)
        chosen_gap = min(argmin_ties(pred_gap))  # deterministic tie break fixed before scoring
        chosen_f = min(argmin_ties(pf))
        chosen_field = min(argmin_ties(-pv))
        for a, b in ((1, 2), (1, 3), (2, 3)):
            i, j = a - 1, b - 1
            for objective, actual_delta, pred_delta in (
                ("f_number_lower", af[i] - af[j], pf[i] - pf[j]),
                ("field_higher", -(av[i] - av[j]), -(pv[i] - pv[j])),
                ("balanced_gap_lower", actual_gap[i] - actual_gap[j], pred_gap[i] - pred_gap[j])):
                tie = abs(actual_delta) <= 1e-12
                correct = tie or np.sign(actual_delta) == np.sign(pred_delta)
                pairwise.append({"prescription_id": int(cid), "pair": f"{a}_vs_{b}",
                                 "objective": objective, "actual_tie": tie, "correct": bool(correct)})
        decisions.append({"prescription_id": int(cid), "fold": int(g.fold.iloc[0]),
                          "true_pareto": sorted(true_pareto), "predicted_pareto": sorted(predicted_pareto),
                          "pareto_jaccard": len(true_pareto & predicted_pareto) / len(true_pareto | predicted_pareto),
                          "predicted_gap_choice": chosen_gap,
                          "predicted_gap_choice_in_true_oracle_set": chosen_gap in oracle_gap,
                          "predicted_gap_choice_in_true_pareto": chosen_gap in true_pareto,
                          "predicted_f_choice": chosen_f, "predicted_field_choice": chosen_field,
                          "oracle_gap_rules": sorted(oracle_gap),
                          "actual_gap_selected": float(actual_gap[chosen_gap - 1]),
                          "actual_gap_oracle": float(np.min(actual_gap)),
                          "gap_regret": float(actual_gap[chosen_gap - 1] - np.min(actual_gap)),
                          "selected_target_reached": bool(g.iloc[chosen_gap - 1].target_reached),
                          **{f"actual_gap_rule_{r}": float(actual_gap[r - 1]) for r in (1, 2, 3)},
                          **{f"actual_f_rule_{r}": float(af[r - 1]) for r in (1, 2, 3)},
                          **{f"actual_field_rule_{r}": float(av[r - 1]) for r in (1, 2, 3)}})
    d, pairs = pd.DataFrame(decisions), pd.DataFrame(pairwise)
    policies = []
    for policy in ("rule_1", "rule_2", "rule_3", "predicted_gap", "oracle_gap"):
        if policy.startswith("rule_"):
            values = d[f"actual_gap_{policy}"].to_numpy()
        elif policy == "predicted_gap":
            values = d.actual_gap_selected.to_numpy()
        else:
            values = d.actual_gap_oracle.to_numpy()
        policies.append({"policy": policy, "n": 200, "mean_target_gap": float(np.mean(values)),
                         "median_target_gap": float(np.median(values)), "p90_target_gap": float(np.quantile(values, .9)),
                         "zero_gap_fraction": float(np.mean(np.isclose(values, 0, atol=1e-12)))})
    pair_summary = []
    for (objective, pair), g in pairs.groupby(["objective", "pair"]):
        non_tie = g[~g.actual_tie]
        pair_summary.append({"objective": objective, "pair": pair, "n_non_tie": int(len(non_tie)),
                             "ranking_accuracy_excluding_actual_ties": float(non_tie.correct.mean())})
    summary = {"balanced_utility_definition": "max((F-20)/20,0) + max((5-field)/5,0); lower is better",
               "utility_is_predefined_physical_target_gap_not_learned_from_labels": True,
               "exact_oracle_rule_set_accuracy": float(d.predicted_gap_choice_in_true_oracle_set.mean()),
               "selected_rule_in_true_pareto_fraction": float(d.predicted_gap_choice_in_true_pareto.mean()),
               "mean_pareto_set_jaccard": float(d.pareto_jaccard.mean()),
               "mean_gap_regret": float(d.gap_regret.mean()), "median_gap_regret": float(d.gap_regret.median()),
               "zero_regret_fraction": float(np.mean(np.isclose(d.gap_regret, 0, atol=1e-12))),
               "selected_target_success_fraction": float(d.selected_target_reached.mean()),
               "choice_counts": dict(Counter(map(str, d.predicted_gap_choice))),
               "oracle_set_membership_counts": {str(r): int(sum(r in x for x in d.oracle_gap_rules)) for r in (1, 2, 3)},
               "policy_value": policies, "pairwise_ranking": pair_summary}
    return d, summary


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir()
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in
                     [SOURCE / "predictions_oof.json", SOURCE / "fold_assignments.json",
                      SOURCE / "config.json", ROOT / "random_candidates_200/candidates.json",
                      ROOT / "background_runs/batch_300_20261118/state.json",
                      ROOT / "background_runs/batch_300_20261118_02/state.json"]}
    frame = enrich(load_oof())
    errors = error_analysis(frame)
    decisions, decision_summary = decision_analysis(frame)
    write(OUT / "error_mechanisms.json", errors)
    write(OUT / "decision_summary.json", decision_summary)
    write(OUT / "decision_by_prescription.json", decisions.to_dict("records"))
    write(OUT / "analysis_rows.json", frame.to_dict("records"))
    write(OUT / "provenance.json", {"source_sha256": source_hashes, "oof_only": True,
                                     "prescription_grouped_outer_predictions": True,
                                     "no_new_model_fit": True, "no_zemax_run": True,
                                     "decision_uses_only_oof_predictions_at_selection_time": True,
                                     "actual_endpoints_used_only_for_post_selection_scoring": True})
    after = {path: digest(ROOT / path) for path in source_hashes}
    if after != source_hashes:
        raise RuntimeError("Source changed")
    print(json.dumps({"error": errors["metrics"], "decision": decision_summary}, indent=2))


if __name__ == "__main__":
    main()
