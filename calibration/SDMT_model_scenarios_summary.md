# SDMT Model Scenarios Summary

## Compared scenarios

Three SDMT modeling scenarios were compared using the same pipeline and hold-out evaluation setup:

1. `calibration_full`
   Uses the full digital feature set, including variables closely related to the digital score.
2. `without_outcome_like`
   Excludes variables considered outcome-like: `num_err`, `num_simbolos`, `score`, `numdig1`, `numerr1`, `numdig2`, `numerr2`, `numdig3`, `numerr3`.
3. `kinematic_clinical_only`
   Keeps only kinematic and clinical covariates.

In the current SDMT matrix, scenarios 2 and 3 are identical after filtering. The retained predictors are:

- `diagonal_inches`
- `avgdur`
- `sdvdur`
- `disease_duration_years`
- `age_at_test`
- `delta_dias_digital_papel`
- `sex_binary`

## Main results

| Scenario | Best model | RMSE | MAE | R2 | CCC | ICC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `calibration_full` | `linear` | 6.44 | 4.99 | 0.641 | 0.812 | 0.823 |
| `without_outcome_like` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |
| `kinematic_clinical_only` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |

Detailed metrics by model:

| Scenario | Model | RMSE | MAE | R2 | CCC | ICC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `calibration_full` | `linear` | 6.44 | 4.99 | 0.641 | 0.812 | 0.823 |
| `calibration_full` | `rf` | 7.02 | 6.17 | 0.573 | 0.752 | 0.764 |
| `calibration_full` | `xgboost` | 7.62 | 6.86 | 0.497 | 0.717 | 0.730 |
| `without_outcome_like` | `linear` | 7.30 | 5.86 | 0.538 | 0.663 | 0.678 |
| `without_outcome_like` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |
| `without_outcome_like` | `xgboost` | 9.31 | 7.28 | 0.249 | 0.459 | 0.474 |
| `kinematic_clinical_only` | `linear` | 7.30 | 5.86 | 0.538 | 0.663 | 0.678 |
| `kinematic_clinical_only` | `rf` | 6.85 | 5.99 | 0.593 | 0.756 | 0.768 |
| `kinematic_clinical_only` | `xgboost` | 9.31 | 7.28 | 0.249 | 0.459 | 0.474 |

## Interpretation

- The best predictive setting is still `calibration_full`, confirming that the strongest SDMT model is primarily a calibration from digital performance to paper performance.
- Removing outcome-like digital variables reduces performance, which shows that these variables contain strong signal and are doing important predictive work.
- However, performance does not collapse when they are removed. This means the remaining kinematic and clinical variables still retain meaningful predictive information.
- Therefore, the current SDMT system should be interpreted as a calibration model with additional mechanistic support from execution biomarkers and clinical covariates, rather than as a purely mechanistic biomarker model.

## Ready-to-paste manuscript wording

### Short version

Three SDMT modeling scenarios were compared to distinguish direct digital-to-paper calibration from more mechanistic prediction. The full calibration model, which included digital variables closely related to performance (`score`, `num_simbolos`, and tercile-level counts/errors), achieved the best overall performance (RMSE 6.44, MAE 4.99, R2 0.641, CCC 0.812, ICC 0.823). When these outcome-like variables were removed, predictive performance decreased but remained moderate (best reduced model: Random Forest, RMSE 6.85, MAE 5.99, R2 0.593, CCC 0.756, ICC 0.768). These findings suggest that the current SDMT pipeline behaves primarily as a digital-to-paper calibration model, although kinematic and clinical covariates still preserve a non-trivial amount of predictive signal.

### Longer version

To clarify whether the SDMT model was mainly translating digital scores into paper scores or instead capturing more independent execution-related biomarkers, three modeling scenarios were evaluated under the same hold-out pipeline. The first scenario (`calibration_full`) used the complete digital feature set. The second (`without_outcome_like`) removed variables considered conceptually too close to the target score, namely `num_err`, `num_simbolos`, `score`, and tercile-specific correct/error counts. The third (`kinematic_clinical_only`) retained only kinematic and clinical covariates. In the current dataset, the latter two scenarios were identical after feature filtering, leaving `diagonal_inches`, `avgdur`, `sdvdur`, `disease_duration_years`, `age_at_test`, `delta_dias_digital_papel`, and `sex_binary`. The full calibration scenario yielded the best performance, with the linear model reaching RMSE 6.44, MAE 4.99, R2 0.641, CCC 0.812, and ICC 0.823. After removing outcome-like variables, the best reduced model was Random Forest, with RMSE 6.85, MAE 5.99, R2 0.593, CCC 0.756, and ICC 0.768. Thus, the SDMT model is best understood as a calibration framework from digital to paper performance, while the residual predictive capacity observed in the reduced models indicates that kinematic and clinical features still encode meaningful, though weaker, information about paper-based performance.
