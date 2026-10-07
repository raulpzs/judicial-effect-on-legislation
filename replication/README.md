# Python replication for predicted probabilities

Read [the validation report](results/VALIDATION_REPORT.md) before using the figures: all 17 models match sample sizes, cluster counts, parameter counts, likelihoods and information criteria, but nine retain small printed-precision discrepancies.

All 17 models are written explicitly in `SPECIFICATIONS` near the top of [scripts/replicate.py](scripts/replicate.py). These ordered predictor lists and `controlled` flags drive estimation. `CONTROL_VARIABLES`, `CONTROL_TERMS`, `LEGAL_SYSTEM_COLUMNS`, and `DEFENDANT_COLUMNS` immediately beside them define the shared complete-case controls and ordered categorical design. `REFERENCE_CATEGORIES`, the outcome/category mappings, `CLUSTER_VARIABLE`, `SPLINE_KNOTS`, storage defaults, and `ESTIMATOR_SETTINGS` define the consequential choices in the following section.

`parse_log()` remains an independent validation reader for the entire Stata log and its printed reference estimates. `specifications_with_references()` checks every explicit command, focal-predictor list and control flag against that reader, then attaches the logged estimates for validation. The log cannot override the Python specifications. Design columns and reference-equation ordering are checked against the detailed logged coefficient rows. The existing manifest schema is retained.

Judicial independence is initialized missing, assigned the high-court value, then overwritten by the low-court value where applicable. Other/unclear defendants are combined. Complete-case deletion includes outcome, predictors, and country ID separately for every model. The unused expression-mode macro is not applied.

## Code organization and commands

The original analysis follows labeled stages in `replicate.py`: `load_data()` → `prepare_variables()` → `select_estimation_sample()` → `construct_design_matrix()` → `fit_model()` → `validate_and_export_fit()` → `predict_grid()`. The existing `prepare()` and `design()` entry points remain available to importers. Numerical conditioning, estimation, covariance, validation and prediction arithmetic stay in supporting functions. `diagnose.py` uses the same explicit specifications and estimator settings for its existing precision/start diagnostics; `report.py` and `plot_predictions.py` consume the unchanged exports.

Run the existing entry points from the repository root:

```sh
.venv/bin/python replication/scripts/replicate.py
.venv/bin/python replication/scripts/diagnose.py
.venv/bin/python replication/scripts/report.py
.venv/bin/python replication/scripts/plot_predictions.py
.venv/bin/python replication/scripts/regime_subsets.py
```

The original analysis reads `data/processed/cases_v6_short.csv`; the regime extension reads `cases_v7_short.csv`. Default commands write their existing results locations. `replicate.py --output PATH` remains available for a separate output directory; `regime_subsets.py --output PATH` accepts directories below `replication/results/regime_subsets/`.

The regime script visibly selects the shared `FULL_MODEL_NAMES` from the explicit specifications. Its stages prepare/check v7, verify complete-case membership against the original manifest, filter by decision-year `regime_binary`, diagnose the full design, attempt the original estimator, and export with inference-validity flags. `regime_binary` is never a design column. The four unsuccessful autocracy fits retain quasi-separation, nonconvergence and invalid uncertainty; no controls or outcome categories are changed to obtain a fit. It produces no predictions.

## Exact refactor equivalence

The untouched code, copied inputs, original results/figures and isolated baseline runs are preserved under `refactor_validation/`. The initial capture was run **before** editing the modeling scripts:

```sh
.venv/bin/python replication/scripts/check_refactor_equivalence.py --capture-baseline
```

It refuses to replace existing baseline evidence. Reproduce the before/after checks against the preserved baseline with:

```sh
.venv/bin/python replication/scripts/check_refactor_equivalence.py
```

This uses the current `.venv` for both versions, verifies environment/input hashes, and creates a new isolated `after_N/` project on each run. No checked-in results are refreshed during these checks. All 17 original fits, eight regime attempts, existing predictions, storage/start diagnostics, reports and 48 figures are rerun. Instrumentation records sample IDs, outcomes, cluster IDs, required variables, design matrices, optimizer state and failed-fit terminal covariance without changing fitting routines.

[The comparison report](refactor_validation/REPORT.md) and [machine-readable comparison](refactor_validation/comparison.json) compare exact stored values, with no rounding or numerical tolerance. NPZ members are compared as arrays, not ZIP-container bytes. Matching missing values and signed infinities are explicit; CSV schemas/order/textual precision and every JSON leaf are checked. Only individually identified, verified source-hash provenance changes are allowed, and they are listed in the report; whole metadata files are never excluded. Environment, dependency versions, input hashes, commands and run logs are retained with the baseline. Existing discrepancies with Stata remain documented below and in the unchanged original validation report.

## Numerical conventions and existing results

The direct Stata spline is `s1 = year` and `s2 = [(year−2001)₊³ − 3.5(year−2016)₊³ + 2.5(year−2022)₊³]/441`, constructed before deletion with fixed full-data knots. Float32 storage is emulated, with float64 arithmetic for fitting. The log omits storage settings; sensitivity diagnostics and that limitation are reported. The basis formula and knot-percentile rule follow the [official mkspline manual](https://www.stata.com/manuals15/rmkspline.pdf). Numeric defaults are documented in [generate/set type](https://www.stata.com/manuals/dgenerate.pdf) and [import delimited](https://www.stata.com/manuals/dimportdelimited.pdf).

For numerical conditioning, each nonconstant design column is centered and scaled: `W = X T`. This preserves the Stata spline basis. Exported coefficients are `B = T B_work`; the full covariance is transformed by `A = I₂ ⊗ T`, `V = A V_work Aᵀ`. The matrix `T` is saved for every model. Parameter vectors flatten by equation (Fortran order): all Contracts-versus-Mixed coefficients, then all Expands-versus-Mixed coefficients. The intercept is last in each equation.

Country covariance is computed as `G/(G−1) H⁻¹ (Σ_g s_g s_gᵀ) H⁻¹`, where `H` is negative likelihood Hessian and `s_g` sums observation scores within country. This is the maximum-likelihood correction from [Stata’s robust manual](https://www.stata.com/manuals/p_robust.pdf). It is checked against statsmodels’ uncorrected cluster sandwich times `G/(G−1)`. The library’s default extra `(N−1)/(N−K)` factor is intentionally omitted; see [cov_cluster source](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html). Coefficient confidence limits use normal 1.9599639845 quantiles, as in the logged z tables. AIC uses `−2LL + 2K`, BIC uses `−2LL + K log(N)`, with both equations’ estimated parameters counted.

Predictions are average adjusted probabilities over each model’s actual estimation sample, with one focal predictor set to each of 51 equally spaced values over its own observed range. Other values are unchanged. For outcome j and nonreference equation k, the gradient block is the sample average of `p_j (1[j=k]−p_k) x`. The variance is `gᵀ V g`, retaining all cross-equation covariance. Confidence intervals use the delta standard error of `logit(mean probability)` and inverse-logit endpoints; they are not clipped. These are pointwise intervals conditional on the observed covariate distribution, not simultaneous bands. Probability and parameter ordering are checked against [MNLogit’s documented implementation](https://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html).

Validation uses half the last printed token unit plus `1e-10`, with zero relative tolerance. All failures remain flagged; tolerances were not changed to achieve agreement. Probability agreement, normalization, and central finite-difference gradient checks have separate explicit limits in the report. The full Stata covariance is not printed, so only its logged diagonal SEs and coefficient confidence limits can be compared directly.

Outputs include [predictions.csv](results/predictions.csv) (3,672 rows), [model_manifest.json](results/model_manifest.json), [validation report](results/VALIDATION_REPORT.md), coefficient and covariance exports, and [48 PNG figures](results/figures) indexed in [figure_index.csv](results/figure_index.csv). Each of the 24 model–predictor grids has two presentation PNGs: `__with_histogram.png` and `__without_histogram.png`. Both retain the three outcome panels and confidence bands, with a plain-language predictor title. Model names, N, subtitles, and bottom notes are omitted; the figure index retains N and validation status. The histogram version also includes observed-support rugs.

Earlier annotated PNGs and their index are preserved in `results/archive_png_with_notes/`. The main `results/figures/` directory contains the 48 clean presentation images.
