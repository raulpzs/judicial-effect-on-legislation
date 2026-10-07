"""Shared experiment plumbing; verified numerical routines remain read-only."""
import importlib.util
import json
import platform
import re
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'outputs/model_runs'
sys.path.insert(0, str(ROOT/'replication/scripts'))
import replicate as numerics
import regime_subsets as diagnostics


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(diagnostics.clean(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_path(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Run/model names must contain only letters, numbers, underscores and hyphens')
    return RUNS/name


def environment():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'statsmodels': statsmodels.__version__, 'platform': platform.platform()}


def load_configuration(path, selected):
    module_spec = importlib.util.spec_from_file_location('experiment_configuration', path)
    config = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config)
    if config.ESTIMATOR_SETTINGS['method'] != 'newton':
        raise ValueError('This engine preserves Newton estimation; edit maxiter/tol/disp in ESTIMATOR_SETTINGS')
    for variable, encoding in config.CATEGORIES.items():
        levels = [level for level, _ in encoding['columns']]
        if set(levels) != set(encoding['levels']) - {encoding['reference']} or len(levels) != len(set(levels)):
            raise ValueError(f'{variable}: encoded levels must include every nonreference category exactly once')
    names = selected or list(config.SPECIFICATIONS)
    expanded = []
    for name in names:
        run_path(name)  # Also guarantees safe artifact stems.
        entry = config.SPECIFICATIONS[name]
        controls = []
        for control in entry['controls']:
            controls.extend(config.CONTROL_GROUPS[control[1:]] if control.startswith('@') else [control])
        focal = list(entry['focal_predictors'])
        if len(set(controls+focal)) != len(controls+focal):
            raise ValueError(f'{name}: duplicate focal/control variables')
        if set(controls+focal) & set(['y', 'country_id', 'yearspline1', 'yearspline2', '_cons']):
            raise ValueError(f'{name}: outcome, cluster, fixed spline and intercept are handled by the engine')
        if any(v in config.CATEGORIES for v in focal):
            raise ValueError(f'{name}: put categorical factors in controls; focal grids require numeric predictors')
        expanded.append({'name': name, 'focal_predictors': focal, 'controls': controls})
    # Fixed conventions are deliberate; this workflow edits models/settings,
    # rather than changing the baseline outcome, clustering or spline method.
    if (config.OUTCOME_MAPPING != numerics.OUTCOME_MAPPING or
        config.CLUSTER_VARIABLE != numerics.CLUSTER_VARIABLE or config.SPLINE_KNOTS != numerics.SPLINE_KNOTS):
        raise ValueError('Outcome mapping, country clustering and fixed spline knots must retain replication conventions')
    return config, expanded


def prepare_data(path, config, specifications):
    data = numerics.prepare(path, config.NUMERIC_STORAGE, config.SPLINE_STORAGE)
    # New continuous terms follow Stata float storage too. Already-prepared
    # independence must not be rounded a second time after its court assignment.
    prepared = set(numerics.FOCAL + ['v2juhcind_lag1', 'v2juncind_lag1'] + numerics.SPLINE_COLUMNS)
    for variable in dict.fromkeys(v for s in specifications for v in s['focal_predictors']+s['controls']):
        if variable in data and variable not in prepared and variable not in config.CATEGORIES:
            try:
                data[variable] = pd.to_numeric(data[variable], errors='raise').astype(config.NUMERIC_STORAGE).astype('float64')
            except (ValueError, TypeError) as error:
                data.attrs.setdefault('invalid_numeric_variables', {})[variable] = str(error)
    return data


def fixed_reference(input_path, reference_model):
    manifest_path = ROOT/'replication/results/model_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = ROOT/'data/processed/cases_v6_short.csv'
    if fingerprint(source) != manifest['source_sha256']['data/processed/cases_v6_short.csv']:
        raise ValueError('Original manifest source fingerprint no longer matches v6')
    original = pd.read_csv(source, dtype=str, keep_default_na=False)
    current = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(original, current[original.columns])
    membership = pd.read_csv(ROOT/'replication/results/estimation_samples.csv', dtype=str, keep_default_na=False)
    if membership.case_id.tolist() != original.case_id.tolist():
        raise ValueError('Recorded case IDs differ from the original source')
    spec = next(s for s in manifest['models'] if s['name'] == reference_model)
    row_ids = spec['sample_row_ids']
    if (len(row_ids) != len(set(row_ids)) or
        membership.loc[membership[reference_model].eq('True'), 'row_id'].astype(int).tolist() != row_ids):
        raise ValueError('Fixed reference row IDs disagree with the recorded membership CSV')
    return row_ids, {'model': reference_model, 'manifest_sha256': fingerprint(manifest_path),
                     'source_sha256': fingerprint(source), 'case_identity_validated': True}


def select_and_design(data, specification, sample_label, policy, fixed_rows, config):
    required = ['y', 'country_id', 'yearspline1', 'yearspline2'] + specification['focal_predictors'] + specification['controls']
    absent_columns = [v for v in required if v not in data]
    if absent_columns:
        raise ValueError(f'Missing required source columns: {absent_columns}')
    invalid = {v: data.attrs.get('invalid_numeric_variables', {})[v] for v in required
               if v in data.attrs.get('invalid_numeric_variables', {})}
    if invalid:
        raise ValueError(f'Invalid numeric variables; no values were silently coerced: {invalid}')
    eligible = data.dropna(subset=required).copy() if policy == 'available' else data.loc[np.array(fixed_rows)-1].copy()
    sample = eligible if sample_label == 'all' else eligible.loc[eligible.regime_binary.eq(sample_label)].copy()
    if policy == 'fixed' and sample[required].isna().any().any():
        missing = {v: (sample.index[sample[v].isna()]+1).tolist() for v in required if sample[v].isna().any()}
        raise ValueError(f'Fixed membership retained; cannot fit because required values are missing: {missing}')
    if sample.empty:
        raise ValueError('No eligible cases in this sample')
    columns = {v: sample[v].to_numpy(float) for v in specification['focal_predictors']}
    controls = specification['controls']
    ordered = [v for v in config.DESIGN_CONTROL_ORDER if v in controls] + [v for v in controls if v not in config.DESIGN_CONTROL_ORDER]
    for variable in ordered:
        if variable in config.CATEGORIES:
            encoding = config.CATEGORIES[variable]
            unexpected = set(sample[variable].unique()) - set(encoding['levels'])
            if unexpected:
                raise ValueError(f'Unexpected levels for {variable}: {unexpected}')
            for level, name in encoding['columns']:
                if name in columns:
                    raise ValueError('Duplicate encoded design name: '+name)
                columns[name] = sample[variable].eq(level).to_numpy(float)
        else:
            columns[variable] = sample[variable].to_numpy(float)
    columns.update({v: sample[v].to_numpy(float) for v in numerics.SPLINE_COLUMNS})
    columns['_cons'] = np.ones(len(sample))
    if len(columns) != len(set(columns)):
        raise ValueError('Duplicate encoded design columns')
    return sample, np.column_stack(list(columns.values())), list(columns), required


def assess_and_fit(sample, matrix, names, specification, label, settings, categories):
    cells, absent = [], []
    for variable in specification['controls']:
        if variable not in categories:
            continue
        for level, text in categories[variable]['levels'].items():
            selected = sample[variable].eq(level)
            if not selected.any():
                absent.append({'variable': variable, 'level': level, 'label': text})
            for outcome in numerics.LABELS:
                count = int((selected & sample.decision_direction.eq(outcome)).sum())
                cells.append({'variable': variable, 'level': level, 'label': text, 'outcome': outcome,
                              'cases': count, 'empty': count == 0, 'sparse': 0 < count <= diagnostics.THRESHOLDS['sparse_cell_count']})
    constant = [names[i] for i in range(len(names)-1) if np.ptp(matrix[:, i]) == 0]
    W = matrix if constant else diagnostics.condition(matrix)[0]
    info = {'model': specification['name'], 'regime': label, 'sample': diagnostics.counts(sample),
            'design': {'rows': len(sample), 'columns': len(names), 'rank': int(np.linalg.matrix_rank(matrix)),
                       'working_rank': int(np.linalg.matrix_rank(W)), 'condition_raw': np.linalg.cond(matrix),
                       'condition_working': np.linalg.cond(W), 'constant_nonintercept_predictors': constant,
                       'absent_categorical_levels': absent, 'empty_cell_count': sum(c['empty'] for c in cells),
                       'sparse_cell_count': sum(c['sparse'] for c in cells), 'column_names': names},
            'separation': diagnostics.separation(W, sample.y.to_numpy(int), names, sample.index+1)}
    if info['design']['rank'] < matrix.shape[1]:
        _, _, vectors = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
        info['design']['unidentified_relationships'] = [
            {name: float(weight) for name, weight in zip(names, vector) if abs(weight) > 1e-8}
            for vector in vectors[info['design']['rank']:]]
    # Capture the actual conditioned terminal parameters without an inverse
    # coefficient transformation (which would lose a few stored bits).
    constructor = numerics.MNLogit
    captured = {}
    def observed_constructor(*args, **kwargs):
        model = constructor(*args, **kwargs)
        original_fit = model.fit
        def observed_fit(*fit_args, **fit_kwargs):
            original_callback = fit_kwargs.get('callback')
            def callback(parameters):
                captured['parameters'] = parameters.reshape((matrix.shape[1], 2), order='F').copy()
                if original_callback:
                    original_callback(parameters)
            fit_kwargs['callback'] = callback
            result = original_fit(*fit_args, **fit_kwargs)
            captured['parameters'] = result.params.copy()
            return result
        model.fit = observed_fit
        return model
    previous_settings = numerics.ESTIMATOR_SETTINGS
    try:
        numerics.ESTIMATOR_SETTINGS = settings.copy()
        numerics.MNLogit = observed_constructor
        try:
            result = diagnostics.fit_attempt(sample, matrix, names, info)
        except Exception as error:
            info.update(status='unsuccessful', successful=False, reliable_unrestricted_fit=False,
                        prediction_readiness='Not ready: estimation/diagnostic exception')
            info.setdefault('optimizer', {})['exception'] = f'{type(error).__name__}: {error}'
            info.setdefault('stability_concerns', []).append(info['optimizer']['exception'])
            info['finiteness'] = {v: False for v in ['coefficients', 'covariance', 'clustered_standard_errors', 'confidence_intervals']}
            transform = diagnostics.condition(matrix)[1] if not constant else None
            beta = transform @ captured['parameters'] if transform is not None and 'parameters' in captured else np.full((matrix.shape[1], 2), np.nan)
            result = (beta, np.full((2*matrix.shape[1], 2*matrix.shape[1]), np.nan), transform, None)
    finally:
        numerics.MNLogit = constructor
        numerics.ESTIMATOR_SETTINGS = previous_settings
    return result, info, cells, captured.get('parameters', np.full((matrix.shape[1], 2), np.nan))


def verify_artifact(directory, entry, key):
    relative = Path(entry[key])
    path = (directory/relative).resolve()
    if not path.is_relative_to(directory.resolve()) or fingerprint(path) != entry[key+'_sha256']:
        raise ValueError(f'Run artifact identity/fingerprint mismatch: {relative}')
    return path
