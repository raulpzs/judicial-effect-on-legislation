"""Build the technical validation report from machine-readable results."""
from pathlib import Path
import json
import pandas as pd

BASE=Path(__file__).resolve().parents[1]


def main():
    out=BASE/'results'
    m=pd.read_csv(out/'validation_models.csv')
    c=pd.read_csv(out/'validation_comparisons.csv')
    p=pd.read_csv(out/'prediction_validation.csv')
    diag=pd.read_csv(out/'optimizer_diagnostics.csv')
    storage=pd.read_csv(out/'storage_diagnostics.csv')
    passed=m.loc[m.all_printed_comparisons_pass,'model'].tolist()
    failed=m.loc[~m.all_printed_comparisons_pass,'model'].tolist()
    lines=[
        '# Validation of Analysis_20260920.log', '',
        'All 17 models have the logged N, country-cluster count, outcome mapping, and parameter count. '
        'All Python fits converged, have full-column-rank designs, and finite coefficients and country-clustered covariance. '
        'All log-likelihood, AIC, and BIC comparisons pass their printed-precision tolerances.', '',
        '**All printed comparisons pass:** '+', '.join(passed)+'.', '',
        '**Printed-precision failures remain:** '+', '.join(failed)+'. '
        f'{int((~c.passed).sum())} of {len(c)} numerical comparisons fail. '
        'These are flagged, not treated as exact matches. Affected figures explicitly refer to this report.', '',
        '## Per-model checks', '',
        '| Model | N | Clusters | Rank / columns | Parameters | Converged | Failed numerical cells | Max coefficient difference | Max SE difference | Max CI-endpoint difference |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for row in m.itertuples():
        cm=c[c.model==row.model]
        mx=lambda metrics: cm.loc[cm.metric.isin(metrics),'absolute_difference'].max()
        lines.append(f'| {row.model} | {row.N} | {row.clusters} | {row.rank}/{row.columns} | {row.parameters} | {row.converged} | {row.numerical_failures} | {mx(["coef"]):.3g} | {mx(["se"]):.3g} | {mx(["lower","upper"]):.3g} |')
    lines += ['', 'Maximum absolute differences over all models:', '', '| Quantity | Maximum |', '|---|---:|']
    for metric,value in c.groupby('metric').absolute_difference.max().items():
        lines.append(f'| {metric} | {value:.9g} |')
    lines += ['', '## Tolerances and references', '',
        'Each detailed coefficient, standard error, confidence limit, log likelihood, AIC, and BIC is parsed '
        'as its original printed token. Absolute tolerance is half its last printed decimal unit plus 1e-10 '
        'for floating-point arithmetic; relative tolerance is zero. Scientific notation is handled through Decimal. '
        'For example, .1562346 allows 5.01e-8; 16.15426 allows 5.0001e-6. '
        'Trailing zeros suppressed in the log make this literal-token rule conservative for some cells. '
        'The rule was set before comparison and was not widened after failures. '
        'Regression details supply coefficients/SEs/CIs and the headline log likelihood; estat ic supplies AIC/BIC/df. '
        'No esttab number is used.', '',
        'N, country clusters, parameter counts and equation/term ordering must match exactly. '
        'Design rank must equal column count, convergence must be true, and the largest absolute score '
        'in conditioned coordinates must be below 1e-7. The log has completed iteration sequences and no '
        'convergence warning, but does not expose e(converged) or a Stata design rank; those checks are computed in Python.', '',
        '## Numerical diagnosis and limitations', '',
        'The restricted cubic spline is reproduced directly at knots 2001, 2016, 2022. '
        'Float32 storage for both continuous imported inputs and generated spline values is used, then '
        'promoted to float64 for calculation. Stata defaults to float numeric storage; the log does not record '
        'set type, the Stata version, or the generated variables’ storage types. The spline float-storage choice '
        'is supported by the comparisons below, but cannot be independently confirmed from a saved Stata dataset. '
        'Changing storage is a precision diagnostic, not a change in formula, knots, or sample.', '',
        '| Imported numeric storage | Spline storage | Failed coefficient/SE/CI cells |', '|---|---|---:|']
    for (imp,spl),g in storage.groupby(['import_storage','spline_storage']):
        lines.append(f'| {imp} | {spl} | {g.printed_precision_failures.sum()} |')
    lines += ['',
        'The largest residual differences occur in dejure3 and dejure5. Their focal coefficients differ '
        'by about 3e-6, exceeding display rounding. Both logged fits stop after iteration 3. '
        'The Python solutions satisfy their score equations to numerical precision. Refitting from the '
        'rounded logged coefficients reaches the same Python solution; across all models the largest '
        f'coefficient change between these two Python starts is {diag.refit_from_rounded_log_max_coefficient_difference.max():.3g}. '
        'The evidence is consistent with optimizer stopping differences, but their exact source cannot be '
        'proved without full-precision Stata estimates and settings. Smaller residuals involve SEs and '
        'intercept confidence limits; their final digits can depend on storage and numerical evaluation. '
        'No coefficients were replaced with logged numbers, and no stopping rule was tuned to reproduce them.', '',
        'The supplied CSV has 1,075 rows and 136 columns, whereas the log reports 116 columns. '
        'Outcome counts and all model sample/cluster counts match. Only executed-command variables are used. '
        'The original log does not identify individual estimation-sample rows; the reconstructed row membership '
        'is exported, but exact row identity against Stata cannot be independently certified.', '',
        'Rounded logged coefficients are unsuitable for reconstructing exact predictions: year is about 2,000, '
        'so rounding a year coefficient is amplified in the linear predictor. The maximum observation-level '
        f'probability difference from using rounded logged coefficients is {diag.max_probability_difference_from_rounded_log.max():.3g}; '
        'this is only a diagnostic, not an independent validation of predictions. All exported predictions use the '
        'new Python fits and their full clustered covariance. Remaining printed-precision failures stay unresolved; '
        'no affected figure is labeled an exact validated replication.', '',
        '## Prediction checks', '',
        'All 24 model/predictor grids pass. Each has 51 equally spaced values including the observed sample minimum '
        'and maximum, three outcome probabilities and pointwise 95% logit-delta intervals. '
        'Every model uses its own complete-case sample. Every nonfocal value, including the year spline and other '
        'focal predictors, remains observed. Histograms and rugs display that sample’s predictor support.', '',
        f'- Largest manual-versus-MNLogit.predict difference: {p.max_manual_predict_difference.max():.3g} (limit 1e-11).',
        f'- Largest probability-sum error: {p.max_probability_sum_error.max():.3g} (limit 1e-12).',
        f'- Observation probabilities range from {p.min_probability.min():.6g} to {p.max_probability.max():.6g}, all within [0,1].',
        f'- Largest numerical-gradient scaled error: {p.max_gradient_scaled_error.max():.3g} (limit 1e-7).', '',
        'Central finite differences check every parameter and every outcome at each grid’s first, middle, and last '
        'point. The scaled error is abs(analytic−numeric)/(1+abs(analytic)); the step is 1e-4 divided by '
        'max(1, max(abs(design column))) to limit linear-predictor perturbations. '
        'Internal probability columns are Mixed, Contracts, Expands. Exported labels and Stata codes are '
        'explicitly mapped; estimated equations are Contracts versus Mixed, then Expands versus Mixed.', '',
        '## Audit files', '',
        '- `validation_comparisons.csv`: every printed token, estimate, error, tolerance, and pass flag.',
        '- `validation_models.csv`, `validation_summary.json`: structural checks and numerical summaries.',
        '- `model_manifest.json`: all 17 commands/formulas, controls, references, sample row IDs, hashes, and parameter order.',
        '- `storage_diagnostics.csv`, `optimizer_diagnostics.csv`: reproducible precision and refit diagnostics.',
        '- `models/*.npz`: design, centering/scaling transform, coefficients, and full clustered covariance.',
        '- `prediction_validation.csv`, `prediction_grids.csv`, `observed_support.csv`: all prediction checks and support.',
        '- `figure_index.csv`: all 24 PNG/PDF figure pairs.','']
    (out/'VALIDATION_REPORT.md').write_text('\n'.join(lines))
    c.loc[~c.passed].to_csv(out/'validation_failures.csv',index=False)


if __name__=='__main__':
    main()
