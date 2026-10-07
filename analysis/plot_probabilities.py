"""Predict from one named run's saved specification/sample; never refit."""
import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'judicial_analysis_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import workflow as work

COLORS = {'Contracts Expression': '#C45B28', 'Mixed Outcome': '#666666', 'Expands Expression': '#167BA3'}


def saved_prediction(directory, manifest, entry, predictors):
    spec = entry['specification'].copy()
    expected_hash = work.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if expected_hash != entry['specification_sha256']:
        raise ValueError('Saved fit specification identity mismatch')
    work.verify_artifact(directory, entry, 'membership')
    work.verify_artifact(directory, entry, 'diagnostics')
    state_path = work.verify_artifact(directory, entry, 'state')
    with np.load(state_path) as archive:
        state = {key: archive[key] for key in archive.files}
    for key, expected in [('run_name', manifest['run_name']), ('input_sha256', manifest['input']['sha256']),
                          ('specification_sha256', expected_hash)]:
        if state[key].item() != expected:
            raise ValueError(f'Saved state belongs to a different run/input/specification: {key}')
    requested = predictors or spec['focal_predictors']
    if not set(requested) <= set(spec['focal_predictors']):
        raise ValueError(f"{entry['model']}: predictor is not a saved focal predictor")
    sample = pd.DataFrame(state['focal_values'], columns=state['focal_names'], index=state['row_ids']-1)
    spec['focal_predictors'] = [p for p in spec['focal_predictors'] if p in requested]
    beta, covariance, X, T = (state[key] for key in ['beta', 'covariance', 'X', 'transform'])
    # Constructing a likelihood object is not fitting. Its library prediction
    # provides the original manual-versus-library check at stored theta.
    model = work.numerics.MNLogit(state['outcome'].astype(int), X @ T, check_rank=True, missing='raise')
    adapter = SimpleNamespace(predict=lambda working_X: model.predict(state['working_parameters'], working_X))
    return work.numerics.predict_grid(spec, sample, X, state['columns'].tolist(), adapter, beta, covariance, T)


def draw_figures(batch, rows, support, manifest, skipped):
    indices = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    for (model, sample, predictor), group in rows.groupby(['model', 'sample', 'predictor'], sort=False):
        observed = support.loc[(support.model == model) & (support['sample'] == sample) & (support.predictor == predictor), 'value'].to_numpy()
        unusable = [s['sample'] for s in skipped if s['model'] == model]
        subset_label = sample + ('-only; other requested subset unusable' if sample != 'all' and unusable else '')
        for histogram in [False, True]:
            variant = 'with_histogram' if histogram else 'without_histogram'
            fig = plt.figure(figsize=(12, 6 if histogram else 5.2))
            gs = fig.add_gridspec(2 if histogram else 1, 3, height_ratios=[5, 1] if histogram else [1], hspace=.12, wspace=.16)
            for j, (label, color) in enumerate(COLORS.items()):
                axis = fig.add_subplot(gs[0, j])
                frame = group.loc[group.outcome.eq(label)].sort_values('grid_value')
                axis.fill_between(frame.grid_value, frame.lower, frame.upper, color=color, alpha=.18, linewidth=0)
                axis.plot(frame.grid_value, frame.probability, color=color, lw=2)
                axis.set(title=label, ylim=(0, 1), yticks=np.linspace(0, 1, 6), xlim=(frame.grid_value.min(), frame.grid_value.max()))
                axis.grid(axis='y', alpha=.18)
                if j == 0:
                    axis.set_ylabel('Predicted probability')
                if histogram:
                    axis.tick_params(axis='x', labelbottom=False)
                    dist = fig.add_subplot(gs[1, j], sharex=axis)
                    dist.hist(observed, bins=25, color='#b8bdc3', edgecolor='white', linewidth=.3)
                    dist.plot(observed, np.zeros(len(observed)), '|', color='#555555', alpha=.13, markersize=5)
                    dist.set_ylabel('Count' if j == 0 else '')
            fig.suptitle(f"{manifest['run_name']} · {model} · {subset_label}\n{predictor}", y=.97, fontsize=13)
            fig.text(.5, .01, f"Average adjusted over {int(group.N.iloc[0])} fitted {sample} cases; this sample's observed range. "
                     'Pointwise 95% intervals; no formal regime-difference test.', ha='center', fontsize=8)
            fig.subplots_adjust(top=.80, bottom=.11, left=.07, right=.98)
            filename = f'{model}__{sample}__{predictor}__{variant}.png'
            fig.savefig(batch/'figures'/filename, dpi=180)
            plt.close(fig)
            indices.append({'run': manifest['run_name'], 'model': model, 'sample': sample, 'predictor': predictor,
                            'variant': variant, 'sample_label': subset_label, 'file': 'figures/'+filename})
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--predictors', nargs='+')
    args = parser.parse_args()
    directory = work.run_path(args.run)
    manifest = json.loads((directory/'run_manifest.json').read_text())
    if manifest['run_name'] != args.run or work.fingerprint(directory/'configuration.json') != manifest['configuration_sha256']:
        raise ValueError('Requested run/configuration identity mismatch')
    if work.environment() != manifest['environment']:
        raise ValueError('Prediction environment differs from the saved run')
    for path, expected in manifest['numerical_sources'].items():
        if work.fingerprint(work.ROOT/path) != expected:
            raise ValueError('Shared numerical implementation changed since this run: '+path)
    known_models = {fit['model'] for fit in manifest['fits']}
    if args.models and not set(args.models) <= known_models:
        raise ValueError('Requested model does not belong to this run')
    generation = 1
    while (directory/'predictions'/f'batch_{generation:03d}').exists():
        generation += 1
    batch = directory/'predictions'/f'batch_{generation:03d}'
    (batch/'figures').mkdir(parents=True)
    record = {'run_name': args.run, 'status': 'running', 'command': shlex.join([sys.executable, *sys.argv]),
              'fit_manifest_sha256': work.fingerprint(directory/'run_manifest.json'), 'skipped_fits': [], 'consumed_states': []}
    rows, checks, grids, support = [], [], [], []
    try:
        for fit in manifest['fits']:
            if args.models and fit['model'] not in args.models:
                continue
            if not fit['successful']:
                skipped = {'model': fit['model'], 'sample': fit['sample'], 'reason': fit.get('error', 'Fit diagnostics prohibit inference: '+fit['status'])}
                record['skipped_fits'].append(skipped)
                print('SKIPPED', skipped, flush=True)
                continue
            diagnosis = json.loads(work.verify_artifact(directory, fit, 'diagnostics').read_text())
            if not diagnosis.get('reliable_unrestricted_fit', False):
                skipped = {'model': fit['model'], 'sample': fit['sample'], 'reason': diagnosis['prediction_readiness']}
                record['skipped_fits'].append(skipped)
                print('SKIPPED', skipped, flush=True)
                continue
            prediction, validation, ranges, observed = saved_prediction(directory, manifest, fit, args.predictors)
            if not all(check['passed'] for check in validation):
                raise ValueError(f"Prediction checks failed: {fit['model']} / {fit['sample']}")
            for destination, values in [(rows, prediction), (checks, validation), (grids, ranges), (support, observed)]:
                destination.extend({**value, 'run': args.run, 'sample': fit['sample']} for value in values)
            record['consumed_states'].append({'model': fit['model'], 'sample': fit['sample'], 'sha256': fit['state_sha256']})
        for filename, values in [('predictions.csv', rows), ('prediction_validation.csv', checks),
                                 ('prediction_grids.csv', grids), ('observed_support.csv', support)]:
            pd.DataFrame(values).to_csv(batch/filename, index=False)
        index = draw_figures(batch, pd.DataFrame(rows), pd.DataFrame(support), manifest, record['skipped_fits']) if rows else []
        pd.DataFrame(index).to_csv(batch/'figure_index.csv', index=False)
        record.update(status='completed' if rows else 'completed_no_usable_fits', prediction_rows=len(rows), figures=len(index),
                      file_sha256={str(p.relative_to(batch)): work.fingerprint(p) for p in batch.rglob('*') if p.is_file()})
    except BaseException as error:
        record.update(status='failed_partial_results_preserved', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        work.write_json(batch/'prediction_manifest.json', record)
    print('Predictions and figures:', batch, flush=True)


if __name__ == '__main__':
    main()
