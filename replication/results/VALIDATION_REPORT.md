# Validation of Analysis_20260920.log

All 17 models have the logged N, country-cluster count, outcome mapping, and parameter count. All Python fits converged, have full-column-rank designs, and finite coefficients and country-clustered covariance. All log-likelihood, AIC, and BIC comparisons pass their printed-precision tolerances.

**All printed comparisons pass:** indep1, indep2, attack5, dejure2, dejure6, full1, full2, full3.

**Printed-precision failures remain:** attack1, attack2, attack3, attack4, dejure1, dejure3, dejure4, dejure5, full4. 43 of 1515 numerical comparisons fail. These are flagged, not treated as exact matches. Validation status is retained in the figure index; presentation figures omit model names, sample sizes, and method/validation notes by request.

## Per-model checks

| Model | N | Clusters | Rank / columns | Parameters | Converged | Failed numerical cells | Max coefficient difference | Max SE difference | Max CI-endpoint difference |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| indep1 | 825 | 43 | 4/4 | 8 | True | 0 | 4.63e-06 | 4.84e-07 | 4.92e-06 |
| indep2 | 825 | 43 | 16/16 | 32 | True | 0 | 2.74e-06 | 4.88e-06 | 3.73e-06 |
| attack1 | 1074 | 47 | 4/4 | 8 | True | 1 | 2.82e-06 | 3.19e-06 | 4.02e-06 |
| attack2 | 1074 | 47 | 4/4 | 8 | True | 2 | 5.2e-07 | 4.62e-06 | 2.55e-06 |
| attack3 | 1074 | 47 | 4/4 | 8 | True | 1 | 1.99e-06 | 3.31e-06 | 2.14e-06 |
| attack4 | 1074 | 47 | 4/4 | 8 | True | 3 | 5.5e-06 | 3.72e-06 | 2.91e-06 |
| attack5 | 1074 | 47 | 16/16 | 32 | True | 0 | 4.64e-06 | 1.58e-06 | 7.29e-07 |
| dejure1 | 1074 | 47 | 4/4 | 8 | True | 2 | 2.53e-06 | 3.33e-06 | 3.51e-06 |
| dejure2 | 1074 | 47 | 16/16 | 32 | True | 0 | 2.72e-06 | 2.47e-06 | 3.55e-06 |
| dejure3 | 1074 | 47 | 4/4 | 8 | True | 16 | 1.84e-05 | 6.02e-06 | 3.42e-05 |
| dejure4 | 1074 | 47 | 16/16 | 32 | True | 1 | 1.93e-06 | 2.57e-06 | 3.07e-06 |
| dejure5 | 1074 | 47 | 4/4 | 8 | True | 16 | 2.05e-05 | 4.96e-06 | 4.08e-05 |
| dejure6 | 1074 | 47 | 16/16 | 32 | True | 0 | 3.12e-06 | 1.55e-06 | 4.77e-06 |
| full1 | 825 | 43 | 17/17 | 34 | True | 0 | 2.47e-06 | 3.63e-06 | 3.33e-06 |
| full2 | 825 | 43 | 18/18 | 36 | True | 0 | 4.84e-06 | 1.58e-06 | 4.18e-06 |
| full3 | 825 | 43 | 18/18 | 36 | True | 0 | 4.12e-06 | 2.73e-06 | 4.06e-06 |
| full4 | 825 | 43 | 18/18 | 36 | True | 1 | 4.17e-06 | 2.71e-06 | 2.77e-06 |

Maximum absolute differences over all models:

| Quantity | Maximum |
|---|---:|
| aic | 0.000466849343 |
| bic | 0.000466059979 |
| coef | 2.05010247e-05 |
| ll | 4.47620987e-05 |
| lower | 4.08020166e-05 |
| se | 6.01779331e-06 |
| upper | 1.66691324e-05 |

## Tolerances and references

Each detailed coefficient, standard error, confidence limit, log likelihood, AIC, and BIC is parsed as its original printed token. Absolute tolerance is half its last printed decimal unit plus 1e-10 for floating-point arithmetic; relative tolerance is zero. Scientific notation is handled through Decimal. For example, .1562346 allows 5.01e-8; 16.15426 allows 5.0001e-6. Trailing zeros suppressed in the log make this literal-token rule conservative for some cells. The rule was set before comparison and was not widened after failures. Regression details supply coefficients/SEs/CIs and the headline log likelihood; estat ic supplies AIC/BIC/df. No esttab number is used.

N, country clusters, parameter counts and equation/term ordering must match exactly. Design rank must equal column count, convergence must be true, and the largest absolute score in conditioned coordinates must be below 1e-7. The log has completed iteration sequences and no convergence warning, but does not expose e(converged) or a Stata design rank; those checks are computed in Python.

## Numerical diagnosis and limitations

The restricted cubic spline is reproduced directly at knots 2001, 2016, 2022. Float32 storage for both continuous imported inputs and generated spline values is used, then promoted to float64 for calculation. Stata defaults to float numeric storage; the log does not record set type, the Stata version, or the generated variables’ storage types. The spline float-storage choice is supported by the comparisons below, but cannot be independently confirmed from a saved Stata dataset. Changing storage is a precision diagnostic, not a change in formula, knots, or sample.

| Imported numeric storage | Spline storage | Failed coefficient/SE/CI cells |
|---|---|---:|
| float32 | float32 | 43 |
| float32 | float64 | 68 |
| float64 | float64 | 80 |

The largest residual differences occur in dejure3 and dejure5. Their focal coefficients differ by about 3e-6, exceeding display rounding. Both logged fits stop after iteration 3. The Python solutions satisfy their score equations to numerical precision. Refitting from the rounded logged coefficients reaches the same Python solution; across all models the largest coefficient change between these two Python starts is 7.28e-14. The evidence is consistent with optimizer stopping differences, but their exact source cannot be proved without full-precision Stata estimates and settings. Smaller residuals involve SEs and intercept confidence limits; their final digits can depend on storage and numerical evaluation. No coefficients were replaced with logged numbers, and no stopping rule was tuned to reproduce them.

The supplied CSV has 1,075 rows and 136 columns, whereas the log reports 116 columns. Outcome counts and all model sample/cluster counts match. Only executed-command variables are used. The original log does not identify individual estimation-sample rows; the reconstructed row membership is exported, but exact row identity against Stata cannot be independently certified.

Rounded logged coefficients are unsuitable for reconstructing exact predictions: year is about 2,000, so rounding a year coefficient is amplified in the linear predictor. The maximum observation-level probability difference from using rounded logged coefficients is 4.01e-05; this is only a diagnostic, not an independent validation of predictions. All exported predictions use the new Python fits and their full clustered covariance. Remaining printed-precision failures stay unresolved; no affected figure is labeled an exact validated replication.

## Prediction checks

All 24 model/predictor grids pass. Each has 51 equally spaced values including the observed sample minimum and maximum, three outcome probabilities and pointwise 95% logit-delta intervals. Every model uses its own complete-case sample. Every nonfocal value, including the year spline and other focal predictors, remains observed. Each grid has two PNG versions: one with histograms and rugs displaying predictor support, and one without that support panel.

- Largest manual-versus-MNLogit.predict difference: 3.94e-15 (limit 1e-11).
- Largest probability-sum error: 2.22e-16 (limit 1e-12).
- Observation probabilities range from 0.00639337 to 0.976534, all within [0,1].
- Largest numerical-gradient scaled error: 8.97e-10 (limit 1e-7).

Central finite differences check every parameter and every outcome at each grid’s first, middle, and last point. The scaled error is abs(analytic−numeric)/(1+abs(analytic)); the step is 1e-4 divided by max(1, max(abs(design column))) to limit linear-predictor perturbations. Internal probability columns are Mixed, Contracts, Expands. Exported labels and Stata codes are explicitly mapped; estimated equations are Contracts versus Mixed, then Expands versus Mixed.

## Audit files

- `validation_comparisons.csv`: every printed token, estimate, error, tolerance, and pass flag.
- `validation_models.csv`, `validation_summary.json`: structural checks and numerical summaries.
- `model_manifest.json`: all 17 commands/formulas, controls, references, sample row IDs, hashes, and parameter order.
- `storage_diagnostics.csv`, `optimizer_diagnostics.csv`: reproducible precision and refit diagnostics.
- `models/*.npz`: design, centering/scaling transform, coefficients, and full clustered covariance.
- `prediction_validation.csv`, `prediction_grids.csv`, `observed_support.csv`: all prediction checks and support.
- `figure_index.csv`: all 48 PNG figures (24 grids, each with and without histograms).
