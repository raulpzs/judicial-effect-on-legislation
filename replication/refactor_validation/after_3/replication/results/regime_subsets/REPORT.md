# Exploratory regime-subset full models

Reproduce from the repository root:

```sh
.venv/bin/python replication/scripts/regime_subsets.py
```

The decision-year `regime_binary` labels only select observations. No regime regressor or interactions are included. Complete-case eligibility is computed before splitting and verified against the original manifest and membership CSV. The existing preparation/design functions preserve Mixed Outcome as the base, court-specific independence, all references, controls, lagged predictors, float32 storage and fixed spline knots 2001/2016/2022. Newton uses 100 iterations and tolerance 1e-12. The country sandwich uses G/(G−1), without an extra residual-df correction.

## Samples

| Model | Regime | Contracts | Mixed | Expands | Cases | Countries | Expected counts match |
|---|---|---:|---:|---:|---:|---:|---|
| full1 | autocracy | 103 | 17 | 73 | 193 | 29 | True |
| full1 | democracy | 117 | 163 | 352 | 632 | 22 | True |
| full2 | autocracy | 103 | 17 | 73 | 193 | 29 | True |
| full2 | democracy | 117 | 163 | 352 | 632 | 22 | True |
| full3 | autocracy | 103 | 17 | 73 | 193 | 29 | True |
| full3 | democracy | 117 | 163 | 352 | 632 | 22 | True |
| full4 | autocracy | 103 | 17 | 73 | 193 | 29 | True |
| full4 | democracy | 117 | 163 | 352 | 632 | 22 | True |

Each full model has 825 eligible cases and 43 distinct countries; the two country sets overlap in eight countries. Membership is verified against the Python replication manifest, not individually identified Stata rows: the original Stata log does not list them.

## Estimation and stability

| Model | Regime | Status | Rank / columns | Converged | Iterations | Max score | Working Hessian condition | Separation |
|---|---|---|---:|---|---:|---:|---:|---|
| full1 | autocracy | unsuccessful | 17 / 17 | False | 100 | 7.29e-11 | 6.01e+11 | quasi |
| full1 | democracy | successful_with_cautions | 17 / 17 | True | 6 | 1.58e-14 | 59.5 | no_separating_direction_detected |
| full2 | autocracy | unsuccessful | 18 / 18 | False | 100 | 7.3e-11 | 6.24e+11 | quasi |
| full2 | democracy | successful_with_cautions | 18 / 18 | True | 6 | 1.86e-14 | 69.3 | no_separating_direction_detected |
| full3 | autocracy | unsuccessful | 18 / 18 | False | 100 | 7.29e-11 | 5.96e+11 | quasi |
| full3 | democracy | successful_with_cautions | 18 / 18 | True | 6 | 3.41e-14 | 61.3 | no_separating_direction_detected |
| full4 | autocracy | unsuccessful | 18 / 18 | False | 100 | 7.29e-11 | 6.2e+11 | quasi |
| full4 | democracy | successful_with_cautions | 18 / 18 | True | 6 | 1.12e-14 | 60.7 | no_separating_direction_detected |

## Interpretation and diagnostic limits

All designs are full rank with no constant non-intercept predictors, absent levels or omitted controls. Autocracy has only 17 Mixed outcomes and one intermediary case (a Mixed outcome). Other/unclear defendants and public-assembly cases have no Mixed outcomes. These empty cells motivate checking, but the separation conclusion comes from verified multinomial linear-program directions, not cell counts alone.

For each observation and alternative outcome the LP constrains the observed-minus-alternative predictor margin to be nonnegative. A direction with some strictly positive margins and ties makes the likelihood increase along an unbounded coefficient ray. A separate max-min LP tests whether every margin can be strictly positive (complete separation). All autocracy specifications have quasi-separation; the unrestricted finite MLE does not exist. For example, reducing both intermediary coefficients leaves all other observations unchanged and increasingly assigns its one case to Mixed. Terminal optimizer numbers for unsuccessful models are exported only as diagnostics, with `inference_valid=False`; they are not interpretable estimates.

All four autocracy optimizers hit 100 iterations. Their conditioned Hessian condition numbers are about 6×10¹¹. Although matrix inverses can be computed, the resulting sandwiches have negative variances, nonfinite SEs/CIs and failed covariance crosschecks. The apparently full numerical covariance rank beyond G−1 is another sign of roundoff corruption at these terminal iterates. Democracy Hessian conditions are about 59–69, scores satisfy the 1e-7 criterion, and the sandwich crosschecks pass.

LPs use [SciPy linprog/HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html), box-normalized directions, feasibility tolerance 1e-9 and strict-margin threshold 1e-7. Witness directions and observation/alternative margins are in diagnostics.json. The conditioned Hessian is used to diagnose estimation; the raw-year Hessian condition is also exported but naturally reflects year/intercept scaling. Heuristic large-value flags use absolute conditioned coefficients >20, absolute raw categorical coefficients >20, or conditioned SEs >10; they are warnings, not separation proofs. Sparse cells mean 1–5 cases. Optimizer convergence alone does not establish identification.

Democracy fits may support exploratory predicted-probability work if their recorded diagnostics pass, with only 22 country clusters. Cluster-score covariance rank is at most G−1, below the 34/36 parameters, so full-vector Wald tests are unavailable; this does not by itself invalidate individual sandwich standard errors. The eight-model comparison as a whole is not ready for probability analysis because the autocracy MLEs are not identified. No predictions or figures were generated. No controls, references, outcomes, spline knots or storage conventions were changed; no penalized/simplified fits were substituted. Significance in one subset and nonsignificance in the other is not a test of differences between regimes.

The original replication has one printed-precision discrepancy for full4 and none for full1–full3. These subset fits reuse its conventions; they are exploratory new estimates, not new claims of exact Stata validation.

## Exports

- estimation_samples.csv: all input rows, each model’s eligibility, missing required fields and subset membership.
- sample_counts.csv and country_counts.csv: regime/outcome totals and per-country composition.
- categorical_cells.csv: every expected level/outcome cell, including zero and sparse counts.
- coefficients.csv: both equations in original parameter order, clustered SEs and normal 95% CIs; explicit fit/inference flags.
- models/*: NPZ matrices, labeled covariance CSV and JSON metadata for successful fits only.
- diagnostics.json and fit_summary.csv: all eight attempts, optimizer traces, warning/score/Hessian/finite-value/separation checks.
- analysis_manifest.json: commands, exact specifications, thresholds, dependency versions, provenance and sample validation.
