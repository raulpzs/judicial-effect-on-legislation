# editable_validation_missing_fixed

Run status: **completed_with_unsuccessful_fits**.

Sample policy: **fixed**; requested sample: **all**. Reference model: `full2`.

Estimator: `{"method": "newton", "maxiter": 100, "tol": 1e-12, "disp": false}`.

| Model | Sample | Status | Cases | Countries |
|---|---|---|---:|---:|
| added_missing | all | unsuccessful | 825 | 43 |

`added_missing / all`: ValueError: Fixed membership retained; cannot fit because required values are missing: {'new_missing_predictor': [1]}

| full1 | all | successful | 825 | 43 |

Inspect each `fits/*__diagnostics.json` for outcomes, rank, separation, optimizer, score, Hessian and covariance checks. Unsuccessful terminal values are diagnostic only. Predictions use only successful saved fits. Regime labels select decision-year cases; differences in subset significance are not tests of regime differences.
