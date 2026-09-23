# Python replication of Analysis_20260920.log

This directory is independent of existing repository analyses. Inputs are the root Stata log and `data/processed/cases_v6_short.csv`. Nothing is committed or pushed. Read [the validation report](results/VALIDATION_REPORT.md) before using the figures: all 17 models match sample sizes, cluster counts, parameter counts, likelihoods and information criteria, but nine retain small printed-precision discrepancies.

From the repository root, using the existing environment:

```sh
.venv/bin/python replication_20260920/scripts/replicate.py
.venv/bin/python replication_20260920/scripts/diagnose.py
.venv/bin/python replication_20260920/scripts/report.py
.venv/bin/python replication_20260920/scripts/plot_predictions.py
```

These commands overwrite only this new directory’s results. Dependencies and the versions used are in [requirements.txt](requirements.txt) and the model manifest. No Stata installation is required. `replicate.py --fit-only` validates without predicting; `--output PATH` directs replication artifacts elsewhere. Plotting, diagnostics, and reporting default to this directory’s `results`.

The manifest derives all 17 model commands from the entire log, expands only the invoked macros, and records the exact controls and reference categories. Judicial independence is initialized missing, assigned the high-court value, then overwritten by the low-court value where applicable. Other/unclear defendants are combined. Complete-case deletion includes outcome, predictors, and country ID separately for every model. The unused expression-mode macro is not applied.

The direct Stata spline is `s1 = year` and `s2 = [(year−2001)₊³ − 3.5(year−2016)₊³ + 2.5(year−2022)₊³]/441`, constructed before deletion with fixed full-data knots. Float32 storage is emulated, with float64 arithmetic for fitting. The log omits storage settings; sensitivity diagnostics and that limitation are reported. The basis formula and knot-percentile rule follow the [official mkspline manual](https://www.stata.com/manuals15/rmkspline.pdf). Numeric defaults are documented in [generate/set type](https://www.stata.com/manuals/dgenerate.pdf) and [import delimited](https://www.stata.com/manuals/dimportdelimited.pdf).

For numerical conditioning, each nonconstant design column is centered and scaled: `W = X T`. This preserves the Stata spline basis. Exported coefficients are `B = T B_work`; the full covariance is transformed by `A = I₂ ⊗ T`, `V = A V_work Aᵀ`. The matrix `T` is saved for every model. Parameter vectors flatten by equation (Fortran order): all Contracts-versus-Mixed coefficients, then all Expands-versus-Mixed coefficients. The intercept is last in each equation.

Country covariance is computed as `G/(G−1) H⁻¹ (Σ_g s_g s_gᵀ) H⁻¹`, where `H` is negative likelihood Hessian and `s_g` sums observation scores within country. This is the maximum-likelihood correction from [Stata’s robust manual](https://www.stata.com/manuals/p_robust.pdf). It is checked against statsmodels’ uncorrected cluster sandwich times `G/(G−1)`. The library’s default extra `(N−1)/(N−K)` factor is intentionally omitted; see [cov_cluster source](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html). Coefficient confidence limits use normal 1.9599639845 quantiles, as in the logged z tables. AIC uses `−2LL + 2K`, BIC uses `−2LL + K log(N)`, with both equations’ estimated parameters counted.

Predictions are average adjusted probabilities over each model’s actual estimation sample, with one focal predictor set to each of 51 equally spaced values over its own observed range. Other values are unchanged. For outcome j and nonreference equation k, the gradient block is the sample average of `p_j (1[j=k]−p_k) x`. The variance is `gᵀ V g`, retaining all cross-equation covariance. Confidence intervals use the delta standard error of `logit(mean probability)` and inverse-logit endpoints; they are not clipped. These are pointwise intervals conditional on the observed covariate distribution, not simultaneous bands. Probability and parameter ordering are checked against [MNLogit’s documented implementation](https://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html).

Validation uses half the last printed token unit plus `1e-10`, with zero relative tolerance. All failures remain flagged; tolerances were not changed to achieve agreement. Probability agreement, normalization, and central finite-difference gradient checks have separate explicit limits in the report. The full Stata covariance is not printed, so only its logged diagonal SEs and coefficient confidence limits can be compared directly.

Outputs include [predictions.csv](results/predictions.csv) (3,672 rows), [model_manifest.json](results/model_manifest.json), [validation report](results/VALIDATION_REPORT.md), coefficient and covariance exports, and [24 figure pairs](results/figures) indexed in [figure_index.csv](results/figure_index.csv). Figures display all three outcomes, confidence bands, histogram/rug support, N, and validation status. No plots are selected by significance or pattern.
