# control_removal_comparison_01

Run status: **completed**.

Sample policy: **fixed**; requested sample: **by_regime**. Reference model: `full2`.

Estimator: `{"method": "newton", "maxiter": 100, "tol": 1e-12, "disp": false}`.

| Model | Sample | Status | Cases | Countries |
|---|---|---|---:|---:|
| full2_reduced | autocracy | successful_with_cautions | 193 | 29 |
| full2_reduced | democracy | successful_with_cautions | 632 | 22 |
| full2_reduced_no_public_speech | autocracy | successful_with_cautions | 193 | 29 |
| full2_reduced_no_public_speech | democracy | successful_with_cautions | 632 | 22 |
| full2_reduced_no_non_verbal | autocracy | successful_with_cautions | 193 | 29 |
| full2_reduced_no_non_verbal | democracy | successful_with_cautions | 632 | 22 |
| full2_reduced_no_speech_no_non_verbal | autocracy | successful_with_cautions | 193 | 29 |
| full2_reduced_no_speech_no_non_verbal | democracy | successful_with_cautions | 632 | 22 |

Inspect each `fits/*__diagnostics.json` for outcomes, rank, separation, optimizer, score, Hessian and covariance checks. Unsuccessful terminal values are diagnostic only. Predictions use only successful saved fits. Regime labels select decision-year cases; differences in subset significance are not tests of regime differences.
