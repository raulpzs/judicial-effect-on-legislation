"""Exact baseline comparisons followed by isolated smoke and safety tests.

Use a fresh prefix each time:
    .venv/bin/python analysis/validate_workflow.py --prefix editable_validation
"""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import workflow as work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('specifications.py'))
    args = parser.parse_args()
    work.run_path(args.prefix)
    audit = work.ROOT/'analysis/validation'/args.prefix
    audit.mkdir(parents=True, exist_ok=False)
    protected = [p for p in (work.ROOT/'replication').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected += [work.ROOT/'data/processed'/name for name in ['cases_v6_short.csv', 'cases_v7_short.csv']]
    hashes = {str(p.relative_to(work.ROOT)): work.fingerprint(p) for p in protected}
    record = {'status': 'running', 'checks': [], 'commands': [], 'environment': work.environment(),
              'protected_files_sha256': hashes, 'numerical_policy': 'exact values, zero tolerance; NaN/Inf masks explicit'}

    def check(name, left, right):
        if isinstance(left, np.ndarray):
            passed = left.dtype == right.dtype and left.shape == right.shape and (
                np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left, right))
            difference = float(np.max(np.abs(left[np.isfinite(left)]-right[np.isfinite(left)]))) if passed and left.dtype.kind in 'fc' and np.isfinite(left).any() else None
        else:
            passed, difference = left == right, None
        record['checks'].append({'name': name, 'passed': bool(passed), 'maximum_finite_absolute_difference': difference})
        if not passed:
            raise AssertionError('Exact comparison failed: '+name)

    def command(arguments, expect_success=True):
        completed = subprocess.run([sys.executable, *arguments], cwd=work.ROOT, text=True, capture_output=True)
        index = len(record['commands'])+1
        (audit/f'command_{index:02d}.txt').write_text(completed.stdout+completed.stderr)
        record['commands'].append({'argv': [sys.executable, *arguments], 'returncode': completed.returncode})
        if (completed.returncode == 0) != expect_success:
            raise AssertionError(f'Unexpected command result; see command_{index:02d}.txt')
        return completed

    runner = 'analysis/run_models.py'
    plotter = 'analysis/plot_probabilities.py'
    names = {label: args.prefix+'_'+label for label in ['all', 'regimes', 'smoke_available', 'smoke_fixed', 'missing_fixed', 'constant']}
    try:
        # First reproduce the verified baseline, before editing any specification.
        command([runner, '--sample', 'all', '--run', names['all'], '--config', str(args.config)])
        command([plotter, '--run', names['all']])
        command([runner, '--models', 'full1', 'full2', 'full3', 'full4', '--sample', 'by_regime',
                 '--run', names['regimes'], '--config', str(args.config)])
        command([plotter, '--run', names['regimes']])
        reference = json.loads((work.ROOT/'replication/results/model_manifest.json').read_text())
        data = work.numerics.prepare(work.ROOT/'data/processed/cases_v6_short.csv')
        all_directory, regime_directory = work.run_path(names['all']), work.run_path(names['regimes'])
        all_manifest = json.loads((all_directory/'run_manifest.json').read_text())
        check('17 attempted full-sample models', len(all_manifest['fits']), 17)
        check('17 successful full-sample models', sum(f['successful'] for f in all_manifest['fits']), 17)
        for spec in reference['models']:
            sample, matrix, columns, required = work.numerics.design(data, spec)
            fit, beta, covariance, T, working_V, groups, score = work.numerics.fit_model(sample, matrix)
            state = np.load(all_directory/'fits'/f"{spec['name']}__all.npz")
            for key, expected in [('X', matrix), ('beta', beta), ('covariance', covariance), ('transform', T),
                                  ('working_covariance', working_V), ('working_parameters', fit.params),
                                  ('row_ids', sample.index.to_numpy()+1), ('outcome', sample.y.to_numpy()),
                                  ('country_ids', sample.country_id.to_numpy()), ('columns', np.array(columns)),
                                  ('required_variables', np.array(required))]:
                check(spec['name']+'/'+key, state[key], expected)
            info = json.loads((all_directory/'fits'/f"{spec['name']}__all__diagnostics.json").read_text())
            check(spec['name']+'/converged', info['optimizer']['converged'], bool(fit.mle_retvals['converged']))
            check(spec['name']+'/iterations', info['optimizer']['iterations'], int(fit.mle_retvals['iterations']))
            check(spec['name']+'/loglik', info['estimates']['terminal_loglik'], float(fit.llf))
            k = beta.size
            # Same LL and parameter dimension imply the same explicit ML ICs.
            check(spec['name']+'/aic', -2*info['estimates']['terminal_loglik']+2*k, -2*fit.llf+2*k)
            check(spec['name']+'/bic', -2*info['estimates']['terminal_loglik']+np.log(len(sample))*k, -2*fit.llf+np.log(len(sample))*k)
            scores = fit.model.score_obs(state['working_parameters'].ravel(order='F'))
            check(spec['name']+'/original_score_sum', float(np.max(np.abs(scores.sum(axis=0)))), score)
        baseline_predictions = work.ROOT/'replication/results'
        for filename in ['predictions.csv', 'prediction_grids.csv', 'prediction_validation.csv', 'observed_support.csv']:
            new = pd.read_csv(all_directory/'predictions/batch_001'/filename, float_precision='round_trip').drop(columns=['run', 'sample'])
            old = pd.read_csv(baseline_predictions/filename, float_precision='round_trip')
            pd.testing.assert_frame_equal(new, old, check_exact=True)
            check('exact baseline '+filename, True, True)
        old_regime = json.loads((work.ROOT/'replication/results/regime_subsets/diagnostics.json').read_text())
        for spec in reference['models'][-4:]:
            eligible, X, columns, required = work.numerics.design(data, spec)
            for label in ['autocracy', 'democracy']:
                mask = eligible.regime_binary.eq(label) if 'regime_binary' in eligible else None
                # Decision-year labels are in v7; preserve v6 row positions.
                prepared_v7 = work.numerics.prepare(work.ROOT/'data/processed/cases_v7_short.csv')
                mask = prepared_v7.loc[eligible.index, 'regime_binary'].eq(label).to_numpy()
                sample, matrix = eligible.loc[mask], X[mask]
                diagnosis, _ = work.diagnostics.diagnose_subset(sample, matrix, columns, spec['name'], label)
                result = work.diagnostics.fit_attempt(sample, matrix, columns, diagnosis)
                beta, covariance, T, working_V = result
                stem = spec['name']+'__'+label
                state = np.load(regime_directory/'fits'/f'{stem}.npz')
                for key, expected in [('X', matrix), ('beta', beta), ('covariance', covariance), ('transform', T),
                                      ('working_covariance', working_V), ('row_ids', sample.index.to_numpy()+1),
                                      ('outcome', sample.y.to_numpy()), ('country_ids', sample.country_id.to_numpy())]:
                    check(stem+'/'+key, state[key], expected)
                actual = json.loads((regime_directory/'fits'/f'{stem}__diagnostics.json').read_text())
                actual.pop('categorical_cells')
                check(stem+'/all numerical diagnostics', actual, work.diagnostics.clean(diagnosis))
                check(stem+'/existing saved diagnostics', actual, next(d for d in old_regime if d['model'] == spec['name'] and d['regime'] == label))
        regime_prediction = json.loads((regime_directory/'predictions/batch_001/prediction_manifest.json').read_text())
        check('four autocracy fits skipped explicitly', len(regime_prediction['skipped_fits']), 4)
        figures = pd.read_csv(regime_directory/'predictions/batch_001/figure_index.csv')
        check('regime figures labeled democracy-only', figures.sample_label.str.contains('democracy-only').all().item(), True)
        record['baseline_equivalence'] = 'exact; all 17 models, original predictions and eight regime attempts'
        work.write_json(audit/'comparison.json', record)
        print('Exact baseline comparisons passed; starting clearly labeled experiments.', flush=True)

        # Copied specification: remove the legal-system control only.
        config_file = audit/'smoke_specifications.py'
        config_file.write_text(args.config.read_text()+"\nSPECIFICATIONS['full2_no_legal_system'] = deepcopy(SPECIFICATIONS['full2'])\n"
                               "SPECIFICATIONS['full2_no_legal_system']['controls'].remove('lgl_systm')\n")
        for policy, label in [('available', 'smoke_available'), ('fixed', 'smoke_fixed')]:
            command([runner, '--models', 'full2_no_legal_system', '--sample', 'all', '--sample-policy', policy,
                     '--reference-model', 'full2', '--config', str(config_file), '--run', names[label]])
            directory = work.run_path(names[label])
            manifest = json.loads((directory/'run_manifest.json').read_text())
            entry = manifest['fits'][0]
            state = np.load(directory/'fits/full2_no_legal_system__all.npz')
            check(policy+'/removed missing-data requirement', 'lgl_systm' in state['required_variables'], False)
            check(policy+'/removed Civil/Mixed columns', set(['Civil', 'Mixed']) & set(state['columns']), set())
            check(policy+'/16 design columns', state['X'].shape[1], 16)
            expected = data.dropna(subset=state['required_variables'].tolist()).index.to_numpy()+1 if policy == 'available' else np.array(reference['models'][-3]['sample_row_ids'])
            check(policy+'/sample policy membership', state['row_ids'], expected)
            if policy == 'available':
                # Today’s edited config must never control an older named run.
                config_file.write_text(config_file.read_text()+"\nSPECIFICATIONS['full2_no_legal_system']['focal_predictors'] = ['not_in_saved_run']\n")
                command([plotter, '--run', names[label]])
                generated = json.loads((directory/'predictions/batch_001/prediction_manifest.json').read_text())
                check('plots depend on saved specification, not edited source', generated['figures'] > 0, bool(entry['prediction_allowed']))
                config_file.write_text(args.config.read_text()+"\nSPECIFICATIONS['full2_no_legal_system'] = deepcopy(SPECIFICATIONS['full2'])\n"
                                       "SPECIFICATIONS['full2_no_legal_system']['controls'].remove('lgl_systm')\n")
            command([plotter, '--run', names[label]])
        # Fixed membership must not shrink when an added predictor is missing.
        missing_input = audit/'fixed_missing.csv'
        source = pd.read_csv(work.ROOT/'data/processed/cases_v7_short.csv', dtype=str, keep_default_na=False)
        source['new_missing_predictor'] = '1'
        missing_row = reference['models'][-3]['sample_row_ids'][0]
        source.loc[missing_row-1, 'new_missing_predictor'] = ''
        source.to_csv(missing_input, index=False)
        missing_config = audit/'missing_specifications.py'
        missing_config.write_text(args.config.read_text()+"\nSPECIFICATIONS['added_missing'] = deepcopy(SPECIFICATIONS['full2'])\n"
                                  "SPECIFICATIONS['added_missing']['focal_predictors'].append('new_missing_predictor')\n")
        command([runner, '--models', 'added_missing', 'full1', '--sample-policy', 'fixed', '--reference-model', 'full2',
                 '--config', str(missing_config), '--input', str(missing_input), '--run', names['missing_fixed']])
        missing_manifest = json.loads((work.run_path(names['missing_fixed'])/'run_manifest.json').read_text())
        check('fixed missing predictor stops only affected fit', [f['successful'] for f in missing_manifest['fits']], [False, True])
        check('fixed missing membership retained', missing_manifest['fits'][0]['fixed_row_ids'], reference['models'][-3]['sample_row_ids'])
        check('fixed missing failure explained', 'required values are missing' in missing_manifest['fits'][0]['error'], True)

        # Constant/redundant controls are reported, not dropped to obtain success.
        constant_config = audit/'constant_specifications.py'
        constant_config.write_text(args.config.read_text()+"\nSPECIFICATIONS['constant_control'] = deepcopy(SPECIFICATIONS['full1'])\n"
                                   "SPECIFICATIONS['constant_control']['controls'].append('new_missing_predictor')\n")
        command([runner, '--models', 'constant_control', 'full1', '--config', str(constant_config), '--input', str(missing_input),
                 '--run', names['constant']])
        constant_manifest = json.loads((work.run_path(names['constant'])/'run_manifest.json').read_text())
        check('rank failure continues other fits', [f['successful'] for f in constant_manifest['fits']], [False, True])
        constant_diag = json.loads((work.run_path(names['constant'])/'fits/constant_control__all__diagnostics.json').read_text())
        check('constant predictor documented', 'new_missing_predictor' in constant_diag['design']['constant_nonintercept_predictors'], True)

        # Run immutability and fail-closed plotting checks.
        before = work.fingerprint(all_directory/'run_manifest.json')
        command([runner, '--run', names['all']], expect_success=False)
        check('duplicate run rejected without modification', work.fingerprint(all_directory/'run_manifest.json'), before)
        command([plotter, '--run', args.prefix+'_never_created'], expect_success=False)
        smoke = work.run_path(names['smoke_fixed'])
        state_path = smoke/'fits/full2_no_legal_system__all.npz'
        saved = state_path.read_bytes()
        try:
            shutil.copyfile(all_directory/'fits/full2__all.npz', state_path)
            command([plotter, '--run', names['smoke_fixed']], expect_success=False)
        finally:
            state_path.write_bytes(saved)
        check('foreign fit rejected; own state restored', work.fingerprint(state_path), json.loads((smoke/'run_manifest.json').read_text())['fits'][0]['state_sha256'])
        record['status'] = 'passed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['protected_files_unchanged'] = all(work.fingerprint(work.ROOT/p) == value for p, value in hashes.items())
        record['runs'] = names
        work.write_json(audit/'comparison.json', record)
        lines = ['# Editable workflow validation', '', f"Status: **{record['status']}**. Original files unchanged: **{record['protected_files_unchanged']}**.", '',
                 record.get('baseline_equivalence', 'Baseline verification incomplete.')+'.', '',
                 'The baseline comparisons use exact values, original sample IDs/coding/design, full covariance and terminal failed-fit arrays, '
                 'normal SE/CI formulas, LL/AIC/BIC, original predictions and complete regime diagnostics. No rounding or relaxed tolerance.', '',
                 'Original replication sums observation scores; the regime diagnostic engine also reports model.score(). '
                 'The original summed-score quantity is explicitly recomputed and compared from saved parameters. '
                 'These two score algorithms are distinct diagnostics, not substituted estimates.', '',
                 'Smoke tests copy full2 and remove only legal system: Civil/Mixed columns and the lgl_systm missing-data requirement disappear. '
                 'Available/fixed policies are independently checked. On these data removing the control need not change case eligibility. '
                 'Plots use saved specifications even after the source configuration changes. '
                 'Fixed missing predictors and constant controls are reported unsuccessful without dropping rows/terms; other fits continue. '
                 'Duplicate run names, nonexistent plotting runs and foreign saved states are rejected.', '',
                 'See comparison.json for individual checks, protected file fingerprints, environment and commands; command_NN.txt contains logs.', '']
        if record.get('error'):
            lines += ['Unresolved difference/error: '+record['error'], '']
        (audit/'REPORT.md').write_text('\n'.join(lines))
    if not record['protected_files_unchanged']:
        raise AssertionError('Original replication/input files changed')
    print('Workflow validation passed:', audit, flush=True)


if __name__ == '__main__':
    main()
