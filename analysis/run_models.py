"""Run editable models, preserving each named experiment independently."""
import argparse
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import workflow as work


def save_fit(directory, manifest, spec, sample, matrix, names, required, result, info, cells, working_parameters):
    beta, covariance, transform, working_covariance = result
    stem = spec['name']+'__'+info['regime']
    state_path = directory/'fits'/f'{stem}.npz'
    spec_hash = work.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    np.savez(state_path, run_name=np.array(manifest['run_name']), input_sha256=np.array(manifest['input']['sha256']),
             specification_sha256=np.array(spec_hash), X=matrix, beta=beta, covariance=covariance,
             transform=np.array([]) if transform is None else transform,
             working_covariance=np.array([]) if working_covariance is None else working_covariance,
             working_parameters=working_parameters, row_ids=sample.index.to_numpy()+1,
             case_ids=sample.case_id.astype(str).to_numpy(dtype=str), outcome=sample.y.to_numpy(),
             country_ids=sample.country_id.to_numpy(), columns=np.array(names), required_variables=np.array(required),
             focal_names=np.array(spec['focal_predictors']),
             focal_values=sample[spec['focal_predictors']].to_numpy(float))
    membership_path = directory/'fits'/f'{stem}__sample.csv'
    membership = sample[['case_id', 'country_id', 'country', 'year', 'decision_direction', 'regime_binary']].copy()
    membership.insert(0, 'row_id', sample.index+1)
    membership.to_csv(membership_path, index=False)
    info_path = directory/'fits'/f'{stem}__diagnostics.json'
    work.write_json(info_path, {**info, 'categorical_cells': cells})
    with np.errstate(invalid='ignore'):
        se = np.sqrt(np.diag(covariance))
    coefficients = beta.ravel(order='F')
    rows = [{'model': spec['name'], 'sample': info['regime'], 'equation': equation, 'term': term,
             'coefficient': coefficients[i], 'clustered_se': se[i],
             'lower': coefficients[i]-work.numerics.Z95*se[i], 'upper': coefficients[i]+work.numerics.Z95*se[i],
             'status': info['status'], 'inference_valid': info['successful'],
             'estimate_type': 'MLE' if info['successful'] else 'diagnostic_terminal_iterate_only'}
            for i, (equation, term) in enumerate((e, t) for e in work.numerics.EQUATIONS for t in names)]
    for row in rows:
        valid = (row['inference_valid'] and np.isfinite(row['coefficient'])
                 and np.isfinite(row['clustered_se']) and row['clustered_se'] > 0)
        row['p_value'] = float(2 * norm.sf(abs(row['coefficient'] / row['clustered_se']))) if valid else np.nan
        row['significant_5pct'] = row['p_value'] < 0.05 if valid else None
    pd.DataFrame(rows).to_csv(directory/'fits'/f'{stem}__coefficients.csv', index=False)
    if info['successful']:
        labels = [e+':'+t for e in work.numerics.EQUATIONS for t in names]
        pd.DataFrame(covariance, index=labels, columns=labels).to_csv(directory/'fits'/f'{stem}__covariance.csv')
    entry = {'model': spec['name'], 'sample': info['regime'], 'status': info['status'],
             'successful': info['successful'], 'specification': spec, 'specification_sha256': spec_hash,
             'prediction_allowed': info['successful'] and info.get('reliable_unrestricted_fit', False),
             'counts': info['sample'], 'prediction_readiness': info.get('prediction_readiness'),
             'state': str(state_path.relative_to(directory)), 'state_sha256': work.fingerprint(state_path),
             'membership': str(membership_path.relative_to(directory)), 'membership_sha256': work.fingerprint(membership_path),
             'diagnostics': str(info_path.relative_to(directory)), 'diagnostics_sha256': work.fingerprint(info_path)}
    return entry, rows


def summary(directory, manifest):
    lines = ['# '+manifest['run_name'], '', 'Run status: **'+manifest['status']+'**.', '',
             f"Sample policy: **{manifest['sample_policy']}**; requested sample: **{manifest['sample_selection']}**. "
             f"Reference model: `{manifest.get('reference_model')}`.", '',
             'Estimator: `'+json.dumps(manifest.get('estimator', {}))+'`.', '',
             '| Model | Sample | Status | Cases | Countries |', '|---|---|---|---:|---:|']
    for fit in manifest['fits']:
        counts = fit.get('counts', {})
        lines.append(f"| {fit['model']} | {fit['sample']} | {fit['status']} | {counts.get('cases', '—')} | {counts.get('countries', '—')} |")
        if fit.get('error'):
            lines += ['', f"`{fit['model']} / {fit['sample']}`: {fit['error']}", '']
    lines += ['', 'Inspect each `fits/*__diagnostics.json` for outcomes, rank, separation, optimizer, score, Hessian and covariance checks. '
              'Unsuccessful terminal values are diagnostic only. Predictions use only successful saved fits. '
              'Regime labels select decision-year cases; differences in subset significance are not tests of regime differences.', '']
    if manifest.get('error'):
        lines += ['Execution error: '+manifest['error'], '']
    (directory/'SUMMARY.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--sample', choices=['all', 'autocracy', 'democracy', 'by_regime'], default='all')
    parser.add_argument('--sample-policy', choices=['available', 'fixed'], default='available')
    parser.add_argument('--reference-model', default='full2')
    parser.add_argument('--run', required=True)
    parser.add_argument('--input', type=Path, default=work.ROOT/'data/processed/cases_v7_short.csv')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('specifications.py'))
    args = parser.parse_args()
    directory = work.run_path(args.run)
    directory.mkdir(parents=True, exist_ok=False)  # Existing runs are immutable.
    (directory/'fits').mkdir()
    manifest = {'schema_version': 1, 'run_name': args.run, 'status': 'running', 'fits': [],
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'command': shlex.join([sys.executable, *sys.argv]), 'environment': work.environment(),
                'sample_policy': args.sample_policy, 'sample_selection': args.sample,
                'reference_model': args.reference_model if args.sample_policy == 'fixed' else None}
    coefficients = []
    try:
        print(f'SAMPLE POLICY: {args.sample_policy.upper()} | sample={args.sample} | run={args.run}', flush=True)
        config, specifications = work.load_configuration(args.config.resolve(), args.models)
        manifest.update(input={'path': str(args.input.resolve()), 'sha256': work.fingerprint(args.input)},
                        config={'path': str(args.config.resolve()), 'sha256': work.fingerprint(args.config)},
                        estimator=config.ESTIMATOR_SETTINGS.copy(), specifications=specifications,
                        numeric_storage=config.NUMERIC_STORAGE, spline_storage=config.SPLINE_STORAGE,
                        outcome_mapping=config.OUTCOME_MAPPING, cluster_variable=config.CLUSTER_VARIABLE,
                        spline_knots=config.SPLINE_KNOTS, diagnostic_thresholds=work.diagnostics.THRESHOLDS,
                        numerical_sources={str(Path(m.__file__).relative_to(work.ROOT)): work.fingerprint(m.__file__)
                                           for m in [work.numerics, work.diagnostics]},
                        workflow_sources={str(p.relative_to(work.ROOT)): work.fingerprint(p) for p in
                                          [Path(__file__).resolve(), Path(work.__file__).resolve()]})
        work.write_json(directory/'configuration.json', {'specifications': specifications, 'estimator': manifest['estimator'],
                        'categories': config.CATEGORIES, 'control_groups': config.CONTROL_GROUPS,
                        'design_control_order': config.DESIGN_CONTROL_ORDER, 'conventions': {k: manifest[k] for k in
                        ['numeric_storage', 'spline_storage', 'outcome_mapping', 'cluster_variable', 'spline_knots']}})
        manifest['configuration_sha256'] = work.fingerprint(directory/'configuration.json')
        shutil.copy2(args.config, directory/'specifications.source.py')
        work.write_json(directory/'run_manifest.json', manifest)
        data = work.prepare_data(args.input, config, specifications)
        fixed_rows = None
        if args.sample_policy == 'fixed':
            fixed_rows, manifest['fixed_reference'] = work.fixed_reference(args.input, args.reference_model)
        labels = ['autocracy', 'democracy'] if args.sample == 'by_regime' else [args.sample]
        for spec in specifications:
            for label in labels:
                try:
                    sample, matrix, names, required = work.select_and_design(
                        data, spec, label, args.sample_policy, fixed_rows, config)
                    result, info, cells, theta = work.assess_and_fit(
                        sample, matrix, names, spec, label, config.ESTIMATOR_SETTINGS, config.CATEGORIES)
                    entry, rows = save_fit(directory, manifest, spec, sample, matrix, names, required, result, info, cells, theta)
                    coefficients.extend(rows)
                except Exception as error:
                    entry = {'model': spec['name'], 'sample': label, 'status': 'unsuccessful', 'successful': False,
                             'specification': spec, 'error': f'{type(error).__name__}: {error}'}
                    if args.sample_policy == 'fixed':
                        candidate = data.loc[np.array(fixed_rows)-1]
                        if label != 'all':
                            candidate = candidate.loc[candidate.regime_binary.eq(label)]
                        entry.update(counts=work.diagnostics.counts(candidate),
                                     fixed_row_ids=(candidate.index+1).tolist(), fixed_case_ids=candidate.case_id.astype(str).tolist())
                    work.write_json(directory/'fits'/f"{spec['name']}__{label}__failure.json", entry)
                manifest['fits'].append(entry)
                pd.DataFrame(coefficients).to_csv(directory/'coefficients.csv', index=False)
                work.write_json(directory/'run_manifest.json', manifest)
                print(spec['name'], label, entry['status'], entry.get('error', ''), flush=True)
        manifest['status'] = ('completed' if all(f['successful'] for f in manifest['fits']) else 'completed_with_unsuccessful_fits')
    except BaseException as error:
        manifest.update(status='execution_failed_partial_results_preserved', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        work.write_json(directory/'run_manifest.json', manifest)
        summary(directory, manifest)


if __name__ == '__main__':
    main()
