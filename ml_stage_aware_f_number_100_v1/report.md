# Stage-Aware F/# prediction feasibility — batch 1 (100 prescriptions)

## Decision

The stage status contains real predictive signal, but the two-stage model is **not stable enough to replace the direct RF**. Retain the direct `RF/raw9, leaf=1` model. The Stage-Aware variants are feasibility evidence only.

## Data and leakage control

- Scope: first 100 prescriptions, 3 rules each (300 rows). No second-batch data and no new Zemax run.
- Stage label: extracted from accepted result logs; `entered_aperture = any(steps[].stage == "aperture")`.
- Inputs: initial 9 parameters plus rule-conditioned encoding only.
- Five folds: exactly the existing prescription-grouped assignments. All three rows of one prescription remain in the same validation fold.
- Each fold fits its classifier on training rows only and its F/# regressor on training rows whose observed stage is “entered”. Validation stage labels are used only for scoring, never for gating or regression input.

## Log audit

| Rule | Entered aperture | F/#=500 | n |
|---:|:---:|:---:|---:|
| 1 | yes | no | 100 |
| 2 | no | yes | 11 |
| 2 | yes | no | 89 |
| 3 | yes | no | 100 |

Thus, *in this batch*, the 11 non-entered rows exactly coincide with F/#=500. This is an empirical contingency result, not a modeling assumption or a hard-coded rule. No non-entered/non-500 or entered/500 row was found.

Minority support is weak: only 11/300 rows (3.67%), all Rule 2. Validation-fold minority counts are 3, 0, 4, 2, and 2; corresponding training-fold counts are 8, 11, 7, 9, and 9.

## Stage classification (OOF)

| Model | Accuracy | Balanced accuracy | Non-entered precision | Non-entered recall | Non-entered F1 | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-only probability | 0.963 | 0.500 | 0.000 | 0.000 | 0.000 | 0.789 | 0.0335 |
| RF, raw9 + rule | 0.967 | 0.895 | 0.529 | 0.818 | 0.643 | 0.990 | 0.0214 |

RF confusion matrix (actual rows, predicted columns; order non-entered, entered): `[[9, 2], [8, 281]]`. The classifier finds 9/11 non-entered rows, but falsely gates 8 entered rows. Rule 1 and Rule 3 contain only entered rows; meaningful discrimination is therefore confined to Rule 2, where balanced accuracy is 0.864 and non-entered F1 is 0.643.

## Full F/# regression (OOF)

| Model | Overall MAE | Overall RMSE | Overall R² | F/#=500 MAE | non-500 MAE | Rule 1 MAE | Rule 2 MAE | Rule 3 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct RF/raw9, leaf=3 | 25.62 | 59.27 | 0.567 | 241.98 | 17.38 | 15.86 | 54.68 | 6.32 |
| Direct RF/raw9, leaf=1 | **23.16** | **56.81** | **0.602** | 220.74 | **15.64** | **15.11** | **48.53** | 5.83 |
| Stage-Aware hard gate | 23.26 | 82.65 | 0.159 | **82.94** | 20.99 | 15.19 | 48.79 | **5.82** |
| Stage-Aware soft mixture | 30.64 | 67.51 | 0.439 | 111.05 | 27.58 | 15.19 | 70.92 | 5.82 |

Hard gating greatly improves the F/#=500 subset, but its eight false non-entered predictions impose very large errors on valid non-500 endpoints, raising overall RMSE by 45% versus direct leaf=1. Soft mixing reduces the catastrophic gate effect but still loses on MAE, RMSE, R², and non-500 MAE.

## Conclusion

1. The initial features contain information about whether Rule 2 reaches aperture optimization; the RF classifier is materially better than a rule-only threshold baseline.
2. The class is too small and too rule-confounded to support a stable production two-stage predictor: 11 minority rows, 7–11 per training fold, and one validation fold with none.
3. Stage awareness explains the F/#=500 tail and can remain an analysis feature, but hard gating is unsafe at present.
4. Preserve `RF/raw9, leaf=1` as the current F/# predictor. Re-evaluate Stage-Aware gating only after obtaining substantially more independent non-entered and near-boundary Rule 2 cases.

## Artifacts

- `stage_log_audit.json`: observed stage/F/# contingency.
- `stage_predictions.json`: all OOF classifier and regression predictions.
- `stage_metrics.json`: pooled, fold, and rule classification metrics.
- `regression_metrics.json`: overall, F/# subset, stage subset, and rule metrics.
- `provenance.json`: source hashes and fold-level leakage audit.
- `stage_probability.{png,svg}` and `regression_error_comparison.{png,svg}`: diagnostics.
