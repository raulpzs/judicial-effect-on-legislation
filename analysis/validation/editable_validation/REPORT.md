# Editable workflow validation

Status: **passed**. Original files unchanged: **True**.

exact; all 17 models, original predictions and eight regime attempts.

The baseline comparisons use exact values, original sample IDs/coding/design, full covariance and terminal failed-fit arrays, normal SE/CI formulas, LL/AIC/BIC, original predictions and complete regime diagnostics. No rounding or relaxed tolerance.

Original replication sums observation scores; the regime diagnostic engine also reports model.score(). The original summed-score quantity is explicitly recomputed and compared from saved parameters. These two score algorithms are distinct diagnostics, not substituted estimates.

Smoke tests copy full2 and remove only legal system: Civil/Mixed columns and the lgl_systm missing-data requirement disappear. Available/fixed policies are independently checked. On these data removing the control need not change case eligibility. Plots use saved specifications even after the source configuration changes. Fixed missing predictors and constant controls are reported unsuccessful without dropping rows/terms; other fits continue. Duplicate run names, nonexistent plotting runs and foreign saved states are rejected.

See comparison.json for individual checks, protected file fingerprints, environment and commands; command_NN.txt contains logs.
