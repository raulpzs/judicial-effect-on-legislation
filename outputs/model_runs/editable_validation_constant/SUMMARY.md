# editable_validation_constant

Run status: **completed_with_unsuccessful_fits**.

Sample policy: **available**; requested sample: **all**. Reference model: `None`.

Estimator: `{"method": "newton", "maxiter": 100, "tol": 1e-12, "disp": false}`.

| Model | Sample | Status | Cases | Countries |
|---|---|---|---:|---:|
| constant_control | all | unsuccessful | 824 | 43 |
| full1 | all | successful | 825 | 43 |

Inspect each `fits/*__diagnostics.json` for outcomes, rank, separation, optimizer, score, Hessian and covariance checks. Unsuccessful terminal values are diagnostic only. Predictions use only successful saved fits. Regime labels select decision-year cases; differences in subset significance are not tests of regime differences.
