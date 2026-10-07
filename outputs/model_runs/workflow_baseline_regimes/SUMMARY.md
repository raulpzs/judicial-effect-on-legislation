# workflow_baseline_regimes

Run status: **completed_with_unsuccessful_fits**.

Sample policy: **available**; requested sample: **by_regime**. Reference model: `None`.

Estimator: `{"method": "newton", "maxiter": 100, "tol": 1e-12, "disp": false}`.

| Model | Sample | Status | Cases | Countries |
|---|---|---|---:|---:|
| full1 | autocracy | unsuccessful | 193 | 29 |
| full1 | democracy | successful_with_cautions | 632 | 22 |
| full2 | autocracy | unsuccessful | 193 | 29 |
| full2 | democracy | successful_with_cautions | 632 | 22 |
| full3 | autocracy | unsuccessful | 193 | 29 |
| full3 | democracy | successful_with_cautions | 632 | 22 |
| full4 | autocracy | unsuccessful | 193 | 29 |
| full4 | democracy | successful_with_cautions | 632 | 22 |

Inspect each `fits/*__diagnostics.json` for outcomes, rank, separation, optimizer, score, Hessian and covariance checks. Unsuccessful terminal values are diagnostic only. Predictions use only successful saved fits. Regime labels select decision-year cases; differences in subset significance are not tests of regime differences.
