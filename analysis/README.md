# Editable modeling workflow

Run commands from the project root using `.venv/bin/python`. This workflow defaults to `data/processed/cases_v7_short.csv` and writes only to named directories under `outputs/model_runs/`. The verified `replication/` workflow remains the baseline; its scripts and results are unchanged.

## Edit a specification

Edit `analysis/specifications.py`. `SPECIFICATIONS` contains all 17 original models with their focal predictors in estimation order and explicit, independently editable control lists. `COUNTRY_CONTROLS`, `MODE_CONTROLS`, and `CONTROL_GROUPS` sit beside these definitions. A model's controls can include `@country`, `@modes`, or `@all`; each group expands before sample selection and estimation.

Create a new name when changing a specification, for example:

```python
SPECIFICATIONS['full2_no_legal_system'] = deepcopy(SPECIFICATIONS['full2'])
SPECIFICATIONS['full2_no_legal_system']['controls'].remove('lgl_systm')
```

List numeric focal predictors using source or prepared variable names. Add numeric controls directly to `controls`; define categorical controls in `CATEGORIES`, including their reference, all permitted levels, and ordered nonreference columns. Known controls retain the baseline design order in `DESIGN_CONTROL_ORDER`; additional controls follow in their listed order. Removing a control removes both its design columns and its complete-case requirement. Duplicate terms, unknown categorical values, and unidentified designs are reported rather than silently repaired.

Edit Newton optimizer settings (`maxiter`, `tol`, `disp`) in `ESTIMATOR_SETTINGS`. This implementation preserves the Newton estimator, outcome mapping, country clustering, and fixed spline knots; changing those conventions requires implementation and validation work. `NUMERIC_STORAGE` and `SPLINE_STORAGE` identify storage precision. The defaults reproduce the baseline's float32 rounding before float64 estimation.

Preparation and numerical routines reuse `replication/scripts/replicate.py` and `regime_subsets.py`: Mixed Outcome is the outcome reference; judicial independence depends on the court hearing the case; defendant Other and Unclear are combined; year splines retain knots 2001, 2016, and 2022. Conditioning, coefficient transformations, country covariance correction, and prediction arithmetic follow the baseline. Experimental specifications drive estimation directly without Stata-log specification assertions.

## Run models

The default sample policy is **available**: each model uses its own complete cases. Omitting `--models` runs all specifications in the configuration.

```sh
.venv/bin/python analysis/run_models.py \
  --models full1 full2 --sample all --run trial_01

.venv/bin/python analysis/run_models.py \
  --models full1 full2 --sample autocracy --run trial_02

.venv/bin/python analysis/run_models.py \
  --models full1 full2 --sample democracy --run trial_03

.venv/bin/python analysis/run_models.py \
  --models full1 full2 --sample by_regime --run trial_04
```

Regime samples filter decision-year `regime_binary`; regime is not added as a predictor. Any selected specification can use any of these samples. Knots are never recalculated within a subset.

Use **fixed** to retain the recorded complete-case membership of an original replication model, then subset it by regime:

```sh
.venv/bin/python analysis/run_models.py \
  --models full1 full2 --sample by_regime \
  --sample-policy fixed --reference-model full2 --run trial_05
```

Fixed sampling verifies the original source hash and every original column and row identity in the chosen input against v6 and the saved replication membership. A newly required variable missing within that sample stops the affected fit with an explanation; it does not drop those cases. Other fits continue. Available sampling can legitimately change membership after an edit. The selected policy is printed at startup and recorded in the summary.

Use `--config path/to/copied_specifications.py` for experiments isolated from the defaults, or `--input path/to/data.csv` for an intentional input change. Name every estimation run uniquely: existing names are rejected, including partially completed runs.

## Inspect fits and make predictions

Open `outputs/model_runs/<run>/SUMMARY.md` first. `run_manifest.json` distinguishes completed runs, runs with unsuccessful fits, and interrupted execution. A completed command does not imply successful estimation of every model.

Each run saves the expanded `configuration.json`, exact configuration source, input/code fingerprints, dependency versions, command, and sample policy. Under `fits/`, inspect membership CSVs, coefficient tables, diagnostics JSON, and saved numerical state. Successful fits also have labeled covariance CSVs. The combined `coefficients.csv` includes status and inference flags; unsuccessful terminal iterates are explicitly diagnostic only.

Diagnostics cover outcome/country counts, rank, constant predictors, absent levels, sparse cells, separation, convergence, score, Hessian conditioning, finite uncertainty, and covariance crosschecks. Successful fits with cautions remain distinct from unsuccessful fits. No term, reference, or outcome is automatically changed to obtain a fit.

Generate probabilities from one saved run without refitting:

```sh
.venv/bin/python analysis/plot_probabilities.py --run trial_04

.venv/bin/python analysis/plot_probabilities.py \
  --run trial_01 --models full2 --predictors court_independence_lag1
```

Selections must belong to the saved run and its focal predictors. Plotting uses the saved specification, sample, design, and working parameters even if today's configuration changes. It verifies saved artifact identities, environment, and shared numerical implementation; it never falls back to another run or replication output. Preserve the same numerical environment when plotting older runs.

Every plotting call creates a new `predictions/batch_NNN/` directory. It contains the 51-point grids, average-adjusted probabilities, pointwise 95% intervals, observed support, numerical crosschecks, figure index, figures with and without histograms, and a prediction manifest. Each grid uses the fitted sample's observed predictor range and covariate distribution. Failed or unreliable fits are skipped with recorded explanations. Figures identify the run, model, and sample; when a requested counterpart failed, usable democracy figures are explicitly democracy-only.

The default autocracy full models remain unsuccessful because of quasi-separation and unstable uncertainty. The democracy full models succeed with a small-cluster inference caution. Neither optimizer convergence alone nor a difference in subset significance establishes a regime effect difference.

## Reproduce validation

```sh
.venv/bin/python analysis/validate_workflow.py --prefix validation_02
```

Use a new prefix every time. Before experiments, the validator reproduces all 17 baseline fits and their predictions, then all eight regime attempts, including failed terminal states. Comparisons require exact full-precision equality with matching missing/nonfinite values. It checks sample/coding/design, coefficients and covariance, SEs/intervals, likelihood/information criteria, prediction grids/intervals, and regime diagnostics. It also fingerprints the protected replication tree and datasets.

The subsequent copied-specification smoke tests remove legal system, independently verify available and fixed membership, generate eligible plots, and verify that saved specifications survive later edits. Additional tests exercise fixed missing predictors, constant controls, continued fitting after failures, duplicate run rejection, nonexistent plotting runs, and foreign saved-state rejection.

Results are in `analysis/validation/<prefix>/comparison.json`, `REPORT.md`, and command logs; estimation artifacts remain separately named under `outputs/model_runs/`. The completed `editable_validation` report passed all checks with no numerical differences and no protected-file changes. If defaults have since changed, pass `--config outputs/model_runs/editable_validation_all/specifications.source.py` to validate the saved baseline configuration.
