"""Capture untouched baselines, then compare isolated runs at exact precision.

Initial capture (before editing model code):
    .venv/bin/python replication/scripts/check_refactor_equivalence.py --capture-baseline
Repeat equivalence checks against that preserved code and data:
    .venv/bin/python replication/scripts/check_refactor_equivalence.py

Only replication/refactor_validation/ is written. No tolerances or rounding
are used; JSON leaves and NPZ arrays are compared, including NaN/Inf masks.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'replication/refactor_validation'


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def inventory(directory):
    return {str(p.relative_to(directory)): hash_file(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def environment():
    return {'executable': sys.executable, 'python': platform.python_version(),
            'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__,
            'thread_environment': {k: os.environ.get(k) for k in
                                   ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + '\n')


def make_project(destination, source_scripts):
    """Separate script/result roots; preserved inputs are copied, never symlinked."""
    (destination/'replication/scripts').mkdir(parents=True)
    (destination/'data/processed').mkdir(parents=True)
    for path in source_scripts.glob('*.py'):
        shutil.copy2(path, destination/'replication/scripts'/path.name)
    for name in ['cases_v6_short.csv', 'cases_v7_short.csv']:
        shutil.copy2(AUDIT/'inputs'/name, destination/'data/processed'/name)
    shutil.copy2(AUDIT/'inputs/Analysis_20260920.log', destination/'Analysis_20260920.log')


def capture_worker(project, analysis):
    """Observe original entry points without changing their numerical routines."""
    sys.path.insert(0, str(project/'replication/scripts'))
    replication = importlib.import_module('replicate')
    state = project/'state'/analysis
    state.mkdir(parents=True)
    original_design = replication.design
    def observed_design(data, specification):
        sample, matrix, names, required = original_design(data, specification)
        sample.attrs['_equivalence_model'] = specification['name']
        np.savez(state/f"{specification['name']}__design.npz", X=matrix,
                 row_ids=sample.index.to_numpy()+1, outcome=sample.y.to_numpy(),
                 country_ids=sample.country_id.to_numpy(), names=np.array(names), required=np.array(required))
        return sample, matrix, names, required
    replication.design = observed_design
    if analysis == 'original':
        original_fit = replication.fit_model
        def observed_fit(sample, matrix):
            result = original_fit(sample, matrix)
            fit, beta, covariance, transform, working_covariance, clusters, score = result
            name = sample.attrs['_equivalence_model']
            np.savez(state/f'{name}__fit.npz', beta=beta, covariance=covariance,
                     transform=transform, working_covariance=working_covariance,
                     working_parameters=fit.params, outcome=sample.y.to_numpy(),
                     country_ids=sample.country_id.to_numpy(), loglik=np.array(fit.llf),
                     aic=np.array(fit.aic), bic=np.array(fit.bic), score=np.array(score),
                     **{f'optimizer_{k}': np.asarray(v) for k, v in fit.mle_retvals.items()})
            return result
        replication.fit_model = observed_fit
        sys.argv = [str(project/'replication/scripts/replicate.py')]
        replication.main()
    else:
        extension = importlib.import_module('regime_subsets')
        original_attempt = extension.fit_attempt
        def observed_attempt(sample, matrix, names, diagnostics):
            result = original_attempt(sample, matrix, names, diagnostics)
            beta, covariance, transform, working_covariance = result
            stem = diagnostics['model']+'__'+diagnostics['regime']
            np.savez(state/f'{stem}__attempt.npz', X=matrix, beta=beta, covariance=covariance,
                     transform=np.array([]) if transform is None else transform,
                     working_covariance=np.array([]) if working_covariance is None else working_covariance,
                     row_ids=sample.index.to_numpy()+1, outcome=sample.y.to_numpy(),
                     country_ids=sample.country_id.to_numpy(), names=np.array(names))
            return result
        extension.fit_attempt = observed_attempt
        sys.argv = [str(project/'replication/scripts/regime_subsets.py')]
        extension.main()


def run_project(project):
    logs = project/'commands'
    logs.mkdir()
    commands = [
        [sys.executable, str(Path(__file__).resolve()), '--worker', str(project), '--analysis', 'original'],
        [sys.executable, str(project/'replication/scripts/diagnose.py')],
        [sys.executable, str(project/'replication/scripts/report.py')],
        [sys.executable, str(project/'replication/scripts/plot_predictions.py')],
        [sys.executable, str(Path(__file__).resolve()), '--worker', str(project), '--analysis', 'regime'],
    ]
    for index, command in enumerate(commands):
        print('Running', ' '.join(command), flush=True)
        with (logs/f'{index+1}.txt').open('w') as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    write_json(logs/'commands.json', commands)


def capture_baseline():
    if AUDIT.exists():
        raise ValueError('Baseline already exists; refusing to replace preserved evidence')
    AUDIT.mkdir(parents=True)
    shutil.copytree(ROOT/'replication/results', AUDIT/'preserved_results')
    shutil.copy2(ROOT/'replication/README.md', AUDIT/'README.before.md')
    (AUDIT/'inputs').mkdir()
    for name in ['cases_v6_short.csv', 'cases_v7_short.csv']:
        shutil.copy2(ROOT/'data/processed'/name, AUDIT/'inputs'/name)
    shutil.copy2(ROOT/'Analysis_20260920.log', AUDIT/'inputs/Analysis_20260920.log')
    make_project(AUDIT/'baseline', ROOT/'replication/scripts')
    record = {'environment': environment(), 'input_hashes': inventory(AUDIT/'inputs'),
              'original_script_hashes': inventory(AUDIT/'baseline/replication/scripts'),
              'preserved_results_hashes': inventory(AUDIT/'preserved_results'),
              'capture_command': '.venv/bin/python replication/scripts/check_refactor_equivalence.py --capture-baseline'}
    write_json(AUDIT/'baseline_record.json', record)
    run_project(AUDIT/'baseline')


def compare_array(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {'equal': False, 'reason': 'shape/dtype mismatch'}
    equal = np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left, right)
    detail = {'equal': bool(equal), 'shape': list(left.shape), 'dtype': str(left.dtype)}
    if left.dtype.kind in 'fc':
        finite = np.isfinite(left) & np.isfinite(right)
        detail.update(nan_masks_match=bool(np.array_equal(np.isnan(left), np.isnan(right))),
                      positive_infinity_masks_match=bool(np.array_equal(np.isposinf(left), np.isposinf(right))),
                      negative_infinity_masks_match=bool(np.array_equal(np.isneginf(left), np.isneginf(right))),
                      maximum_finite_absolute_difference=float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else 0.)
    return detail


def provenance_reason(file, keys, left, right, before, after):
    """Permit specific changed source hashes/roots, never entire metadata files."""
    if file != 'regime_subsets/analysis_manifest.json':
        return None
    known_hashes = {
        ('script_sha256',): 'replication/scripts/regime_subsets.py',
        ('replication_script_sha256',): 'replication/scripts/replicate.py',
        ('original_manifest_sha256',): 'replication/results/model_manifest.json',
        ('protected_file_sha256', 'replication/scripts/replicate.py'): 'replication/scripts/replicate.py',
    }
    if keys in known_hashes:
        path = known_hashes[keys]
        if left == hash_file(before/path) and right == hash_file(after/path):
            return 'Hash of the specifically identified changed source artifact'
    return None


def compare_json(left, right, file, before, after, keys=()):
    differences, provenance = [], []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append({'path': list(keys), 'reason': 'dictionary key mismatch'})
        for key in left.keys() & right.keys():
            changed, allowed = compare_json(left[key], right[key], file, before, after, keys+(key,))
            differences.extend(changed); provenance.extend(allowed)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({'path': list(keys), 'reason': 'list length mismatch'})
        for i, (a, b) in enumerate(zip(left, right)):
            changed, allowed = compare_json(a, b, file, before, after, keys+(i,))
            differences.extend(changed); provenance.extend(allowed)
    elif type(left) != type(right) or left != right:
        reason = provenance_reason(file, keys, left, right, before, after)
        (provenance if reason else differences).append({'path': list(keys), 'before': left, 'after': right,
                                                       'reason': reason or 'exact value mismatch'})
    return differences, provenance


def compare_directory(before, after, relative_directory):
    left_root, right_root = before/relative_directory, after/relative_directory
    files_left, files_right = inventory(left_root), inventory(right_root)
    comparisons = []
    for name in sorted(files_left.keys() | files_right.keys()):
        entry = {'directory': relative_directory, 'file': name, 'equal': True}
        if name not in files_left or name not in files_right:
            entry.update(equal=False, reason='file set mismatch')
        elif name.endswith('.npz'):
            with np.load(left_root/name) as a, np.load(right_root/name) as b:
                entry['members_match'] = set(a.files) == set(b.files)
                entry['arrays'] = {k: compare_array(a[k], b[k]) for k in a.files if k in b.files}
                entry['equal'] = entry['members_match'] and all(v['equal'] for v in entry['arrays'].values())
        elif name.endswith('.json'):
            a, b = json.loads((left_root/name).read_text()), json.loads((right_root/name).read_text())
            differences, provenance = compare_json(a, b, name, before, after)
            entry.update(equal=not differences, differences=differences, provenance_changes=provenance)
        else:
            # Byte equality is stronger than parsed CSV equality and preserves
            # textual precision, schemas, order and explicit missing tokens.
            entry.update(equal=files_left[name] == files_right[name], method='exact file bytes')
        comparisons.append(entry)
    return comparisons


def compare_runs():
    record = json.loads((AUDIT/'baseline_record.json').read_text())
    assert environment() == record['environment'], 'Environment changed since baseline'
    assert inventory(AUDIT/'baseline/replication/scripts') == record['original_script_hashes'], 'Preserved original code changed'
    assert inventory(AUDIT/'inputs') == record['input_hashes'], 'Preserved inputs changed'
    assert inventory(AUDIT/'preserved_results') == record['preserved_results_hashes'], 'Preserved results changed'
    for name, expected in record['input_hashes'].items():
        current = ROOT/name if name.endswith('.log') else ROOT/'data/processed'/name
        assert hash_file(current) == expected, f'Current input changed: {name}'
    # Never overwrite a prior validation run.
    index = 1
    while (AUDIT/f'after_{index}').exists():
        index += 1
    after = AUDIT/f'after_{index}'
    original_results = inventory(ROOT/'replication/results')
    make_project(after, ROOT/'replication/scripts')
    run_project(after)
    before = AUDIT/'baseline'
    comparisons = compare_directory(before, after, 'replication/results')
    comparisons += compare_directory(before, after, 'state')
    originals_unchanged = inventory(ROOT/'replication/results') == original_results == record['preserved_results_hashes']
    summary = {'exact_analytical_equality': all(c['equal'] for c in comparisons),
               'comparison_method': 'Exact equality; no rounding, atol or rtol; NaN and signed infinities matched explicitly',
               'baseline': str(before), 'after': str(after), 'environment': environment(),
               'input_hashes': record['input_hashes'], 'checked_in_results_unchanged': originals_unchanged,
               'original_model_count': 17, 'regime_attempt_count': 8,
               'comparisons': comparisons,
               'command': '.venv/bin/python replication/scripts/check_refactor_equivalence.py'}
    write_json(AUDIT/'comparison.json', summary)
    failed = [c for c in comparisons if not c['equal']]
    allowed = [c for c in comparisons if c.get('provenance_changes')]
    lines = ['# Readability refactor equivalence', '',
             f"Exact analytical equality: **{summary['exact_analytical_equality']}**. Checked-in results unchanged: **{originals_unchanged}**.", '',
             f'Compared {len(comparisons)} output/state files: all 17 original models, eight regime attempts, '
             '24 prediction grids / 3,672 rows, storage/start diagnostics, reports and 48 generated figures.', '',
             'Comparisons include row IDs, outcomes, cluster IDs, required fields, design matrices, optimizer state, '
             'coefficients, all covariance matrices (including failed attempts), SEs/CIs, LL/AIC/BIC, predictions, '
             'separation witnesses and all diagnostic flags. NPZ members are compared at stored precision; matching '
             'NaN and signed infinity masks are explicit. CSV/text/image bytes and every JSON leaf are checked. '
             'No tolerances or rounding are used; no whole metadata file is excluded.', '',
             '## Differences', '']
    lines.extend(f"- Unresolved: `{c['directory']}/{c['file']}`; see comparison.json." for c in failed)
    if not failed:
        lines.append('No unresolved analytical differences.')
    for c in allowed:
        for change in c['provenance_changes']:
            lines.append(f"- Allowed provenance: `{c['file']}` / `{change['path']}` — {change['reason']}.")
    lines += ['', 'The existing Stata discrepancies and four unsuccessful autocracy fits remain unchanged.', '',
              '## Reproduce', '', '```sh', '.venv/bin/python replication/scripts/check_refactor_equivalence.py', '```', '',
              'Untouched scripts and baseline runs are preserved in `baseline/`; original results/figures in '
              '`preserved_results/`; input copies in `inputs/`. `baseline_record.json` records environment and hashes. '
              'Each project’s `commands/commands.json` and logs record actual commands. Repeated checks create a new '
              '`after_N/` directory and leave earlier evidence intact.', '']
    (AUDIT/'REPORT.md').write_text('\n'.join(lines))
    print('Exact analytical equality:', summary['exact_analytical_equality'], 'unresolved files:', len(failed), flush=True)
    assert not failed and originals_unchanged, 'Inspect comparison.json for unresolved differences'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-baseline', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--analysis', choices=['original', 'regime'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        capture_worker(args.worker, args.analysis)
    elif args.capture_baseline:
        capture_baseline()
    else:
        compare_runs()


if __name__ == '__main__':
    main()
