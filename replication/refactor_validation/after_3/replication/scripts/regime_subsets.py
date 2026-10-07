"""Exploratory full1--full4 estimation by decision-year regime; no predictions.

Reproduce: .venv/bin/python replication/scripts/regime_subsets.py
Preparation/design are imported from replicate.py; estimation follows its
Newton settings and country sandwich exactly, with additional failure logging.
Nothing outside results/regime_subsets/ is written.
"""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
import statsmodels

import replicate as replication


ROOT = replication.ROOT
RESULTS = ROOT / 'replication/results'
DEFAULT_OUTPUT = RESULTS / 'regime_subsets'

# =========================
# Shared full models, regime selection and diagnostic settings
# =========================
REGIMES = ['autocracy', 'democracy']
EXPECTED = {
    'autocracy': {'Contracts Expression': 103, 'Mixed Outcome': 17,
                  'Expands Expression': 73, 'cases': 193, 'countries': 29},
    'democracy': {'Contracts Expression': 117, 'Mixed Outcome': 163,
                  'Expands Expression': 352, 'cases': 632, 'countries': 22},
}
CATEGORIES = {
    'lgl_systm': {0: 'Common (reference)', 1: 'Civil', 2: 'Mixed'},
    'defendant5': {1: 'Citizen', 2: 'Press', 3: 'Intermediary',
                   4: 'Government (reference)', 5: 'Other/unclear'},
    'high_court': {0: '0 (reference)', 1: '1'},
    **{v: {0: '0 (reference)', 1: '1'} for v in replication.MODES},
}
THRESHOLDS = {'score': 1e-7, 'lp_feasibility': 1e-9, 'separation_margin': 1e-7,
              'hessian_condition_warning': 1e10, 'large_working_coefficient': 20,
              'large_raw_categorical_coefficient': 20,
              'large_working_se': 10, 'sparse_cell_count': 5,
              'few_clusters_warning': 30}


def full_specifications(original_manifest):
    """Use shared explicit full models; the manifest supplies validation metadata."""
    explicit_models = {spec['name']: spec for spec in replication.specifications_with_references(
        ROOT/'Analysis_20260920.log')}
    original_models = {spec['name']: spec for spec in original_manifest['models']}
    specifications = []
    for name in replication.FULL_MODEL_NAMES:
        reference = original_models[name]
        explicit = explicit_models[name]
        for field in ['expanded_command', 'focal_predictors', 'controlled']:
            assert explicit[field] == reference[field]
        specification = reference.copy()
        # These fields now drive design/estimation from SPECIFICATIONS, while
        # row IDs and design columns remain independent manifest checks.
        specification.update(explicit)
        specifications.append(specification)
    return specifications


# =========================
# Supporting diagnostics: strict JSON, counts, conditioning and separation
# =========================

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(obj):
    """Strict JSON: nonfinite attempted-fit diagnostics become null, not NaN."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')


def counts(sample):
    return {**{label: int(sample.decision_direction.eq(label).sum())
               for label in ['Contracts Expression', 'Mixed Outcome', 'Expands Expression']},
            'cases': len(sample), 'countries': int(sample.country_id.nunique())}


def condition(X):
    p = X.shape[1]
    T = np.eye(p)
    scale = X[:, :-1].std(axis=0)
    # A constant non-intercept is not silently dropped or rescaled.
    if (scale == 0).any():
        raise ValueError('Constant non-intercept design column; no controls omitted')
    T[np.arange(p-1), np.arange(p-1)] = 1 / scale
    T[-1, :-1] = -X[:, :-1].mean(axis=0) / scale
    return X @ T, T


def separation(W, y, names, row_ids):
    """Multinomial recession direction, with Mixed equation fixed at zero.

    D theta gives observed-minus-alternative linear-predictor margins. If
    all margins >=0 and at least one >0, moving along theta increases the
    likelihood without a finite maximizer. Strictly positive margins for
    every pair imply complete separation; positive plus tied pairs imply
    quasi-separation. Box bounds only normalize a homogeneous direction.
    Two LPs distinguish these cases without inferring them from empty cells.
    """
    p = W.shape[1]
    D, pairs = [], []
    for row_id, x, observed in zip(row_ids, W, y):
        for alternative in range(3):
            if alternative == observed:
                continue
            blocks = np.zeros((2, p))
            if observed:
                blocks[observed-1] += x
            if alternative:
                blocks[alternative-1] -= x
            D.append(blocks.ravel())
            pairs.append({'row_id': int(row_id), 'observed': replication.LABELS[observed],
                          'alternative': replication.LABELS[alternative]})
    D = np.array(D)
    options = {'primal_feasibility_tolerance': THRESHOLDS['lp_feasibility'],
               'dual_feasibility_tolerance': THRESHOLDS['lp_feasibility']}
    complete = linprog(np.r_[np.zeros(2*p), -1.],
                       A_ub=np.column_stack([-D, np.ones(len(D))]),
                       b_ub=np.zeros(len(D)), bounds=[(-1, 1)]*(2*p)+[(0, None)],
                       method='highs', options=options)
    recession = linprog(-D.sum(axis=0), A_ub=-D, b_ub=np.zeros(len(D)),
                        bounds=[(-1, 1)]*(2*p), method='highs', options=options)
    result = {'complete_lp_success': complete.success, 'complete_lp_message': complete.message,
              'recession_lp_success': recession.success, 'recession_lp_message': recession.message,
              'complete_maximum_minimum_margin': complete.x[-1] if complete.success else None,
              'status': 'indeterminate', 'parameter_direction': [], 'strict_pairs': []}
    if not (complete.success and recession.success):
        return result
    margins = D @ recession.x
    tol = THRESHOLDS['separation_margin']
    feasible = margins.min() >= -THRESHOLDS['lp_feasibility']
    strict = margins > tol
    result.update(minimum_recession_margin=margins.min(), sum_recession_margins=margins.sum(),
                  strict_pair_count=int(strict.sum()), tied_pair_count=int((~strict).sum()),
                  verified_nonnegative_margins=feasible)
    if not feasible:
        return result
    result['status'] = ('complete' if complete.x[-1] > tol else
                        'quasi' if strict.any() else 'no_separating_direction_detected')
    result['parameter_direction'] = [
        {'equation': equation, 'term': term, 'working_direction': float(value)}
        for equation, block in zip(replication.EQUATIONS, recession.x.reshape(2, p))
        for term, value in zip(names, block) if abs(value) > tol]
    result['strict_pairs'] = [{**pair, 'margin': float(margin)}
                              for pair, margin in zip(pairs, margins) if margin > tol]
    return result


def categorical_diagnostics(sample, model, regime):
    cells, absent = [], []
    for variable, levels in CATEGORIES.items():
        for level, label in levels.items():
            present = sample[variable].eq(level)
            n = int(present.sum())
            if n == 0:
                absent.append({'variable': variable, 'level': level, 'label': label})
            for outcome in replication.LABELS:
                count = int((present & sample.decision_direction.eq(outcome)).sum())
                cells.append({'model': model, 'regime': regime, 'variable': variable,
                              'level': level, 'label': label, 'outcome': outcome, 'cases': count,
                              'level_total': n, 'empty': count == 0,
                              'sparse': 0 < count <= THRESHOLDS['sparse_cell_count']})
    return cells, absent


def fit_attempt(sample, X, names, info):
    """Retain optimizer diagnostics even when replicate.fit_model would raise.

    Settings, transform and covariance are identical to that function; the
    explicit steps let unsuccessful fits return without losing their state.
    """
    p = X.shape[1]
    settings = replication.ESTIMATOR_SETTINGS
    info.update(optimizer={'method': settings['method'], 'maxiter': settings['maxiter'], 'tol': settings['tol'],
                           'converged': False, 'iterations': None, 'exception': None},
                iteration_trace=[], warnings=[], hessian={},
                finiteness={k: False for k in ['coefficients', 'covariance', 'clustered_standard_errors', 'confidence_intervals']},
                omissions=[], stability_concerns=[], successful=False, reliable_unrestricted_fit=False,
                prediction_readiness='Not ready: identification/numerical concerns')
    beta = np.full((p, 2), np.nan)
    V = np.full((2*p, 2*p), np.nan)
    if info['design']['rank'] != p:
        info['optimizer']['exception'] = 'Rank-deficient design; exact specification cannot be identified'
        info['status'] = 'unsuccessful'
        info['stability_concerns'].append(info['optimizer']['exception'])
        return beta, V, None, None
    W, T = condition(X)
    model = replication.MNLogit(sample.y.to_numpy(int), W, check_rank=True, missing='raise')
    assert list(model._ynames_map.values()) == ['0', '1', '2']
    theta = np.zeros(2*p)
    fitted = None
    working_V = None
    def callback(parameters):
        nonlocal theta
        theta = parameters.copy()
        info['iteration_trace'].append({'iteration': len(info['iteration_trace'])+1,
                                       'loglik': float(model.loglike(theta)),
                                       'max_abs_working_parameter': np.max(np.abs(theta)),
                                       'max_abs_score': np.max(np.abs(model.score(theta)))})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fitted = model.fit(**settings, callback=callback)
            theta = fitted.params.ravel(order='F')
            info['optimizer'].update(converged=bool(fitted.mle_retvals['converged']),
                                     iterations=int(fitted.mle_retvals['iterations']))
        except Exception as error:
            info['optimizer']['exception'] = f'{type(error).__name__}: {error}'
            info['optimizer']['iterations'] = len(info['iteration_trace'])
        # This is a terminal optimizer iterate, never a certified MLE by itself.
        beta = T @ theta.reshape((p, 2), order='F')
        scores = model.score_obs(theta)
        score = model.score(theta)
        info['score'] = {'maximum_absolute_working_score': np.max(np.abs(score)),
                         'l2_working_score': np.linalg.norm(score),
                         'maximum_absolute_score_per_case': np.max(np.abs(score))/len(sample),
                         'observation_scores_finite': bool(np.isfinite(scores).all())}
        H = -model.hessian(theta)
        finite_H = bool(np.isfinite(H).all())
        bread = None
        if finite_H:
            eig = np.linalg.eigvalsh((H+H.T)/2)
            info['hessian'].update(working_rank=int(np.linalg.matrix_rank(H)), dimensions=list(H.shape),
                                   condition_working=np.linalg.cond(H),
                                   condition_original=np.linalg.cond(-replication.MNLogit(
                                       sample.y.to_numpy(int), X).hessian(beta.ravel(order='F'))),
                                   minimum_eigenvalue_working=eig.min(), maximum_eigenvalue_working=eig.max())
            try:
                bread = np.linalg.inv(H)
                info['hessian']['inverse_computed'] = bool(np.isfinite(bread).all())
                info['hessian']['inverse_residual_inf'] = np.linalg.norm(H @ bread - np.eye(2*p), ord=np.inf)
            except np.linalg.LinAlgError as error:
                info['hessian'].update(inverse_computed=False, inverse_error=str(error))
        else:
            info['hessian']['inverse_computed'] = False
        info['hessian']['finite'] = finite_H
        groups, ids = np.unique(sample.country_id.to_numpy(), return_inverse=True)
        if bread is not None and np.isfinite(scores).all():
            sums = np.zeros((len(groups), 2*p))
            np.add.at(sums, ids, scores)
            working_V = len(groups)/(len(groups)-1) * bread @ (sums.T @ sums) @ bread
            working_V = (working_V+working_V.T)/2
            A = np.kron(np.eye(2), T)
            V = A @ working_V @ A.T
            if np.isfinite(V).all():
                info['covariance'] = {
                    'clusters': len(groups), 'correction': len(groups)/(len(groups)-1),
                    'rank_working': int(np.linalg.matrix_rank(working_V)),
                    'maximum_rank_from_cluster_scores': len(groups)-1,
                    'minimum_diagonal': np.diag(V).min(),
                    'minimum_eigenvalue_working': np.linalg.eigvalsh(working_V).min(),
                    'negative_variance_count': int((np.diag(V) < 0).sum()),
                    'negative_working_variance_count': int((np.diag(working_V) < 0).sum()),
                }
            if fitted is not None:
                try:
                    check = replication.cov_cluster(fitted, ids, use_correction=False)*len(groups)/(len(groups)-1)
                    info.setdefault('covariance', {})['statsmodels_crosscheck_pass'] = bool(
                        np.allclose(working_V, check, rtol=1e-9, atol=1e-11))
                except Exception as error:
                    info.setdefault('covariance', {})['crosscheck_error'] = str(error)
        se = np.sqrt(np.diag(V))
        b = beta.ravel(order='F')
        info['finiteness'] = {'coefficients': bool(np.isfinite(b).all()),
                              'covariance': bool(np.isfinite(V).all()),
                              'clustered_standard_errors': bool(np.isfinite(se).all()),
                              'confidence_intervals': bool(np.isfinite(b-replication.Z95*se).all() and
                                                           np.isfinite(b+replication.Z95*se).all())}
        info['estimates'] = {'terminal_loglik': float(model.loglike(theta)),
                             'maximum_absolute_raw_coefficient': np.max(np.abs(b)),
                             'maximum_absolute_working_coefficient': np.max(np.abs(theta)),
                             'large_working_parameter_indices': np.flatnonzero(
                                 np.abs(theta)>THRESHOLDS['large_working_coefficient']).tolist(),
                             'large_raw_categorical_coefficients': [
                                 {'equation': e, 'term': t, 'coefficient': float(beta[j, k])}
                                 for k, e in enumerate(replication.EQUATIONS) for j, t in enumerate(names)
                                 if t not in sample.columns and t != '_cons' and
                                 abs(beta[j, k]) > THRESHOLDS['large_raw_categorical_coefficient']],
                             'maximum_raw_clustered_se': np.max(se),
                             'maximum_working_clustered_se': np.sqrt(np.diag(working_V)).max()
                                 if working_V is not None else None,
                             'linear_predictor_transform_pass': bool(np.allclose(
                                 X @ beta, W @ theta.reshape(p, 2, order='F'), atol=1e-11, rtol=1e-11))}
        info['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
    concerns = info['stability_concerns']
    if info['separation']['status'] in ['complete', 'quasi']:
        concerns.append('Verified '+info['separation']['status']+'-separation: no finite unrestricted MLE')
    elif info['separation']['status'] == 'indeterminate':
        concerns.append('Separation LP inconclusive')
    if not info['optimizer']['converged']:
        concerns.append('Optimizer did not converge')
    if info['optimizer']['exception']:
        concerns.append(info['optimizer']['exception'])
    if not all(info['finiteness'].values()):
        concerns.append('At least one estimate/uncertainty component is nonfinite; see finiteness flags')
    if not finite_H or info['hessian'].get('working_rank', 0) < 2*p:
        concerns.append('Likelihood Hessian is numerically rank-deficient/nonfinite')
    if info['hessian'].get('condition_working', np.inf) > THRESHOLDS['hessian_condition_warning']:
        concerns.append('Ill-conditioned working Hessian')
    if not np.isfinite(score).all() or np.max(np.abs(score)) > THRESHOLDS['score']:
        concerns.append('Score tolerance not satisfied')
    if info['estimates']['maximum_absolute_working_coefficient'] > THRESHOLDS['large_working_coefficient']:
        concerns.append('Unusually large conditioned coefficients')
    if info['estimates']['large_raw_categorical_coefficients']:
        concerns.append('Unusually large raw categorical coefficients')
    maximum_working_se = info['estimates']['maximum_working_clustered_se']
    if maximum_working_se is not None and np.isfinite(maximum_working_se) and maximum_working_se > THRESHOLDS['large_working_se']:
        concerns.append('Unusually large conditioned uncertainty')
    if info.get('covariance', {}).get('negative_variance_count', 0):
        concerns.append('Computed sandwich has negative variances; uncertainty is invalid')
    if info.get('covariance', {}).get('statsmodels_crosscheck_pass') is False:
        concerns.append('Country sandwich crosscheck failed at the unstable terminal iterate')
    core = (fitted is not None and info['optimizer']['converged'] and all(info['finiteness'].values())
            and finite_H and info['hessian'].get('working_rank') == 2*p
            and info['hessian'].get('inverse_residual_inf', np.inf) < 1e-8
            and np.isfinite(score).all() and np.max(np.abs(score)) < THRESHOLDS['score']
            and info.get('covariance', {}).get('statsmodels_crosscheck_pass', False)
            and info['separation']['status'] == 'no_separating_direction_detected')
    info['successful'] = bool(core)
    info['reliable_unrestricted_fit'] = bool(core and not concerns)
    if len(groups) < THRESHOLDS['few_clusters_warning']:
        concerns.append(f'Only {len(groups)} country clusters; normal-based inference needs caution')
    if len(groups)-1 < 2*p:
        concerns.append('Cluster covariance cannot have full parameter rank; full-vector joint Wald tests unavailable')
    info['status'] = 'successful_with_cautions' if core and concerns else 'successful' if core else 'unsuccessful'
    info['prediction_readiness'] = ('Exploratory subset probabilities possible with cluster/sample cautions'
                                   if info['reliable_unrestricted_fit'] else 'Not ready: identification/numerical concerns')
    return beta, V, T, working_V


def report(out, diagnostics, sample_counts):
    lines = ['# Exploratory regime-subset full models', '',
             'Reproduce from the repository root:', '',
             '```sh', '.venv/bin/python replication/scripts/regime_subsets.py', '```', '',
             'The decision-year `regime_binary` labels only select observations. No regime regressor or interactions are included. '
             'Complete-case eligibility is computed before splitting and verified against the original manifest and membership CSV. '
             'The existing preparation/design functions preserve Mixed Outcome as the base, court-specific independence, all references, '
             'controls, lagged predictors, float32 storage and fixed spline knots 2001/2016/2022. '
             'Newton uses 100 iterations and tolerance 1e-12. The country sandwich uses G/(G−1), without an extra residual-df correction.', '',
             '## Samples', '', '| Model | Regime | Contracts | Mixed | Expands | Cases | Countries | Expected counts match |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in sample_counts:
        lines.append(f"| {row['model']} | {row['regime']} | {row['Contracts Expression']} | {row['Mixed Outcome']} | "
                     f"{row['Expands Expression']} | {row['cases']} | {row['countries']} | {row['expected_counts_match']} |")
    lines += ['', 'Each full model has 825 eligible cases and 43 distinct countries; the two country sets overlap in eight countries. '
              'Membership is verified against the Python replication manifest, not individually identified Stata rows: the original Stata log does not list them.', '',
              '## Estimation and stability', '',
              '| Model | Regime | Status | Rank / columns | Converged | Iterations | Max score | Working Hessian condition | Separation |',
              '|---|---|---|---:|---|---:|---:|---:|---|']
    for d in diagnostics:
        score = d.get('score', {}).get('maximum_absolute_working_score', np.nan)
        cond = d.get('hessian', {}).get('condition_working', np.nan)
        lines.append(f"| {d['model']} | {d['regime']} | {d['status']} | {d['design']['rank']} / {d['design']['columns']} | "
                     f"{d['optimizer']['converged']} | {d['optimizer']['iterations']} | {score:.3g} | {cond:.3g} | {d['separation']['status']} |")
    lines += ['', '## Interpretation and diagnostic limits', '',
              'All designs are full rank with no constant non-intercept predictors, absent levels or omitted controls. '
              'Autocracy has only 17 Mixed outcomes and one intermediary case (a Mixed outcome). '
              'Other/unclear defendants and public-assembly cases have no Mixed outcomes. '
              'These empty cells motivate checking, but the separation conclusion comes from verified multinomial linear-program directions, not cell counts alone.', '',
              'For each observation and alternative outcome the LP constrains the observed-minus-alternative predictor margin to be nonnegative. '
              'A direction with some strictly positive margins and ties makes the likelihood increase along an unbounded coefficient ray. '
              'A separate max-min LP tests whether every margin can be strictly positive (complete separation). '
              'All autocracy specifications have quasi-separation; the unrestricted finite MLE does not exist. '
              'For example, reducing both intermediary coefficients leaves all other observations unchanged and increasingly assigns its one case to Mixed. '
              'Terminal optimizer numbers for unsuccessful models are exported only as diagnostics, with `inference_valid=False`; they are not interpretable estimates.', '',
              'All four autocracy optimizers hit 100 iterations. Their conditioned Hessian condition numbers are about 6×10¹¹. '
              'Although matrix inverses can be computed, the resulting sandwiches have negative variances, nonfinite SEs/CIs and failed covariance crosschecks. '
              'The apparently full numerical covariance rank beyond G−1 is another sign of roundoff corruption at these terminal iterates. '
              'Democracy Hessian conditions are about 59–69, scores satisfy the 1e-7 criterion, and the sandwich crosschecks pass.', '',
              'LPs use [SciPy linprog/HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html), '
              'box-normalized directions, feasibility tolerance 1e-9 and strict-margin threshold 1e-7. '
              'Witness directions and observation/alternative margins are in diagnostics.json. '
              'The conditioned Hessian is used to diagnose estimation; the raw-year Hessian condition is also exported but naturally reflects year/intercept scaling. '
              'Heuristic large-value flags use absolute conditioned coefficients >20, absolute raw categorical coefficients >20, '
              'or conditioned SEs >10; they are warnings, not separation proofs. '
              'Sparse cells mean 1–5 cases. Optimizer convergence alone does not establish identification.', '',
              'Democracy fits may support exploratory predicted-probability work if their recorded diagnostics pass, with only 22 country clusters. '
              'Cluster-score covariance rank is at most G−1, below the 34/36 parameters, so full-vector Wald tests are unavailable; '
              'this does not by itself invalidate individual sandwich standard errors. '
              'The eight-model comparison as a whole is not ready for probability analysis because the autocracy MLEs are not identified. '
              'No predictions or figures were generated. No controls, references, outcomes, spline knots or storage conventions were changed; '
              'no penalized/simplified fits were substituted. '
              'Significance in one subset and nonsignificance in the other is not a test of differences between regimes.', '',
              'The original replication has one printed-precision discrepancy for full4 and none for full1–full3. '
              'These subset fits reuse its conventions; they are exploratory new estimates, not new claims of exact Stata validation.', '',
              '## Exports', '',
              '- estimation_samples.csv: all input rows, each model’s eligibility, missing required fields and subset membership.',
              '- sample_counts.csv and country_counts.csv: regime/outcome totals and per-country composition.',
              '- categorical_cells.csv: every expected level/outcome cell, including zero and sparse counts.',
              '- coefficients.csv: both equations in original parameter order, clustered SEs and normal 95% CIs; explicit fit/inference flags.',
              '- models/*: NPZ matrices, labeled covariance CSV and JSON metadata for successful fits only.',
              '- diagnostics.json and fit_summary.csv: all eight attempts, optimizer traces, warning/score/Hessian/finite-value/separation checks.',
              '- analysis_manifest.json: commands, exact specifications, thresholds, dependency versions, provenance and sample validation.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


# =========================
# Readable analysis stages: data checks, membership, diagnostics and export
# =========================

def prepare_regime_analysis(original_manifest, data_path):
    """Load v7 and verify old cells, decision-year labels and fixed spline basis."""
    specifications = full_specifications(original_manifest)
    data = replication.prepare(data_path, original_manifest['numeric_import_storage'], original_manifest['spline']['storage'])
    # Validate all original cells/row positions before relying on original row IDs.
    v6 = pd.read_csv(ROOT/'data/processed/cases_v6_short.csv', dtype=str, keep_default_na=False)
    v7 = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(v6, v7[v6.columns])
    expected_regime = pd.to_numeric(data.v2x_regime).map({0: 'autocracy', 1: 'autocracy', 2: 'democracy', 3: 'democracy'})
    assert expected_regime.equals(data.regime_binary), 'Decision-year regime mapping mismatch'
    assert original_manifest['spline']['knots'] == [2001, 2016, 2022]
    saved_basis = pd.read_csv(RESULTS/'spline_basis.csv').set_index('year')
    actual_basis = data[['year', 'yearspline1', 'yearspline2']].drop_duplicates().set_index('year').sort_index()
    assert np.allclose(actual_basis, saved_basis.loc[actual_basis.index], rtol=0, atol=1e-12)
    memberships_original = pd.read_csv(RESULTS/'estimation_samples.csv')
    assert memberships_original.row_id.tolist() == list(data.index+1)
    assert memberships_original.case_id.astype(str).tolist() == data.case_id.astype(str).tolist()
    return data, specifications, memberships_original


def validate_sample_membership(data, sample, specification, memberships_original, output_directory, previous_checks):
    """Check original eligible row IDs before splitting, then record every case."""
    rows = []
    required_variables = specification['complete_case_variables']
    membership_ok = list(sample.index+1) == specification['sample_row_ids']
    csv_membership_ok = memberships_original.loc[memberships_original[specification['name']], 'row_id'].tolist() == list(sample.index+1)
    check = {'model': specification['name'], 'manifest_membership_matches': membership_ok,
                          'membership_csv_matches': csv_membership_ok, 'eligible_counts': counts(sample),
                          'union_countries': int(sample.country_id.nunique()),
                          'overlap_country_ids': sorted(set(sample.loc[sample.regime_binary.eq('autocracy'), 'country_id']) &
                                                        set(sample.loc[sample.regime_binary.eq('democracy'), 'country_id']))}
    if not (membership_ok and csv_membership_ok):
        write_json(output_directory/'sample_validation_failure.json', previous_checks + [check])
        raise ValueError(f"{specification['name']}: estimation membership differs from original replication; stopping rather than forcing agreement")
    for index, row in data.iterrows():
        eligible = index in sample.index
        rows.append({'model': specification['name'], 'row_id': int(index+1), 'case_id': row.case_id,
                            'country_id': row.country_id, 'country': row.country, 'year': row.year,
                            'outcome': row.decision_direction, 'regime': row.regime_binary,
                            'eligible': eligible, 'subset_member': eligible and row.regime_binary in REGIMES,
                            'missing_required_fields': '|'.join(v for v in required_variables if pd.isna(row[v]))})
    return check, rows


def diagnose_subset(subset_sample, subset_matrix, column_names, model_name, regime):
    """Inspect full design and separation; never remove controls or levels."""
    observed = counts(subset_sample)
    local_cells, absent = categorical_diagnostics(subset_sample, model_name, regime)
    constant = [column_names[i] for i in range(len(column_names)-1) if np.ptp(subset_matrix[:, i]) == 0]
    # Retain rank-deficient specifications as unsuccessful, without dropping columns.
    W = subset_matrix if constant else condition(subset_matrix)[0]
    fit_diagnostics = {'model': model_name, 'regime': regime, 'sample': observed,
            'design': {'rows': len(subset_sample), 'columns': len(column_names), 'rank': int(np.linalg.matrix_rank(subset_matrix)),
                       'working_rank': int(np.linalg.matrix_rank(W)),
                       'condition_raw': np.linalg.cond(subset_matrix), 'condition_working': np.linalg.cond(W),
                       'constant_nonintercept_predictors': constant, 'absent_categorical_levels': absent,
                       'empty_cell_count': sum(c['empty'] for c in local_cells),
                       'sparse_cell_count': sum(c['sparse'] for c in local_cells), 'column_names': column_names},
            'separation': separation(W, subset_sample.y.to_numpy(int), column_names, subset_sample.index+1)}
    return fit_diagnostics, local_cells


def export_subset_fit(output_directory, specification, subset_sample, subset_matrix, column_names, fit_diagnostics, fitted_result):
    """Flag terminal failed estimates; archive covariance only for successful fits."""
    beta, covariance, transform, working_V = fitted_result
    regime = fit_diagnostics['regime']
    observed = fit_diagnostics['sample']
    rows = []
    with np.errstate(invalid='ignore'):
        se = np.sqrt(np.diag(covariance))
    b = beta.ravel(order='F')
    parameter_labels = [e+':'+t for e in replication.EQUATIONS for t in column_names]
    for i, (equation, term) in enumerate((e, t) for e in replication.EQUATIONS for t in column_names):
        rows.append({'model': specification['name'], 'regime': regime, 'equation': equation, 'term': term,
                             'coefficient': b[i], 'clustered_se': se[i],
                             'lower': b[i]-replication.Z95*se[i], 'upper': b[i]+replication.Z95*se[i],
                             'fit_status': fit_diagnostics['status'], 'inference_valid': fit_diagnostics['successful'],
                             'estimate_type': 'MLE' if fit_diagnostics['successful'] else 'terminal_optimizer_iterate_only',
                             'stability_concerns': '; '.join(fit_diagnostics['stability_concerns'])})
    stem = specification['name']+'__'+regime
    if fit_diagnostics['successful']:
        np.savez(output_directory/'models'/f'{stem}.npz', beta=beta, covariance=covariance, X=subset_matrix,
                 transform=transform, working_covariance=working_V, row_ids=subset_sample.index.to_numpy()+1,
                 columns=np.array(column_names), equation_labels=np.array(replication.EQUATIONS),
                 probability_labels=np.array(replication.LABELS))
        pd.DataFrame(covariance, index=parameter_labels, columns=parameter_labels).to_csv(output_directory/'models'/f'{stem}__covariance.csv')
        write_json(output_directory/'models'/f'{stem}.json', {'original_specification': specification, 'diagnostics': fit_diagnostics,
                                               'subset_sample_row_ids': (subset_sample.index+1).tolist(),
                                               'parameter_order': parameter_labels,
                                               'covariance_correction': observed['countries']/(observed['countries']-1)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_directory = args.output.resolve()
    if not output_directory.is_relative_to(DEFAULT_OUTPUT.resolve()):
        parser.error('Outputs must stay under replication/results/regime_subsets/')
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory/'models').mkdir(exist_ok=True)
    data_path = ROOT/'data/processed/cases_v7_short.csv'
    manifest_path = RESULTS/'model_manifest.json'
    protected = [p for p in RESULTS.rglob('*') if p.is_file() and not p.is_relative_to(DEFAULT_OUTPUT)]
    protected += [data_path, ROOT/'data/processed/cases_v6_short.csv',
                  ROOT/'replication/scripts/replicate.py', ROOT/'Analysis_20260920.log']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    original_manifest = json.loads(manifest_path.read_text())
    # 1–2. Load v7 and prepare the same variables as the original replication.
    data, specifications, memberships_original = prepare_regime_analysis(original_manifest, data_path)
    sample_counts, memberships, country_counts, cells, diagnostics, coefficients, sample_checks = [], [], [], [], [], [], []
    for specification in specifications:
        # 3–4. Complete cases first, then the original ordered design matrix.
        sample, design_matrix, column_names, required_variables = replication.design(data, specification)
        assert column_names == specification['design_columns'] and required_variables == specification['complete_case_variables']
        check, membership_rows = validate_sample_membership(
            data, sample, specification, memberships_original, output_directory, sample_checks)
        sample_checks.append(check)
        memberships.extend(membership_rows)
        for regime in REGIMES:
            # Decision-year regime is a row filter, never a regressor.
            mask = sample.regime_binary.eq(regime).to_numpy()
            subset_sample, subset_matrix = sample.loc[mask], design_matrix[mask]
            observed = counts(subset_sample)
            sample_counts.append({'model': specification['name'], 'regime': regime, **observed,
                                  'expected_counts_match': observed == EXPECTED[regime]})
            for cid, group in subset_sample.groupby('country_id', sort=True):
                country_counts.append({'model': specification['name'], 'regime': regime, 'country_id': cid,
                                       'country': '|'.join(sorted(group.country.unique())), **counts(group)})
            fit_diagnostics, local_cells = diagnose_subset(subset_sample, subset_matrix, column_names, specification['name'], regime)
            cells.extend(local_cells)
            # 5. Attempt the exact specification, retaining failed-fit diagnostics.
            try:
                beta, covariance, transform, working_V = fit_attempt(subset_sample, subset_matrix, column_names, fit_diagnostics)
            except Exception as error:
                # Unexpected optimizer/diagnostic failure in one subset must not suppress the others.
                fit_diagnostics.update(status='unsuccessful', successful=False, reliable_unrestricted_fit=False,
                            prediction_readiness='Not ready: estimation/diagnostic exception')
                fit_diagnostics.setdefault('optimizer', {})['exception'] = f'{type(error).__name__}: {error}'
                fit_diagnostics.setdefault('stability_concerns', []).append(fit_diagnostics['optimizer']['exception'])
                fit_diagnostics['finiteness'] = {k: False for k in ['coefficients', 'covariance', 'clustered_standard_errors', 'confidence_intervals']}
                beta = np.full((len(column_names), 2), np.nan)
                covariance = np.full((2*len(column_names), 2*len(column_names)), np.nan)
                transform = working_V = None
            # 6. Validate/flag and export; there is no prediction stage here.
            diagnostics.append(fit_diagnostics)
            coefficients.extend(export_subset_fit(output_directory, specification, subset_sample, subset_matrix, column_names, fit_diagnostics,
                                                  (beta, covariance, transform, working_V)))
            stem = specification['name']+'__'+regime
            print(stem, fit_diagnostics['status'], fit_diagnostics['optimizer'], 'separation:', fit_diagnostics['separation']['status'], flush=True)
    pd.DataFrame(memberships).to_csv(output_directory/'estimation_samples.csv', index=False)
    pd.DataFrame(sample_counts).to_csv(output_directory/'sample_counts.csv', index=False)
    pd.DataFrame(country_counts).to_csv(output_directory/'country_counts.csv', index=False)
    pd.DataFrame(cells).to_csv(output_directory/'categorical_cells.csv', index=False)
    pd.DataFrame(coefficients).to_csv(output_directory/'coefficients.csv', index=False)
    write_json(output_directory/'diagnostics.json', diagnostics)
    pd.DataFrame([{'model': data['model'], 'regime': data['regime'], 'status': data['status'],
                   'successful': data['successful'], **data['sample'], 'design_columns': data['design']['columns'],
                   'design_rank': data['design']['rank'], 'converged': data['optimizer']['converged'],
                   'iterations': data['optimizer']['iterations'], 'separation': data['separation']['status'],
                   'max_absolute_score': data.get('score', {}).get('maximum_absolute_working_score'),
                   'hessian_condition_working': data.get('hessian', {}).get('condition_working'),
                   'all_estimates_uncertainty_finite': all(data['finiteness'].values()),
                   'prediction_readiness': data['prediction_readiness'],
                   'stability_concerns': '; '.join(data['stability_concerns'])} for data in diagnostics]).to_csv(output_directory/'fit_summary.csv', index=False)
    unchanged = all(digest(ROOT/path) == value for path, value in hashes.items())
    metadata = {'command': '.venv/bin/python replication/scripts/regime_subsets.py',
                'data': str(data_path.relative_to(ROOT)), 'data_sha256': digest(data_path),
                'script_sha256': digest(Path(__file__)), 'replication_script_sha256': digest(Path(replication.__file__)),
                'original_manifest_sha256': digest(manifest_path), 'specifications': specifications,
                'spline': original_manifest['spline'], 'numeric_import_storage': original_manifest['numeric_import_storage'],
                'regime_definition': 'decision-year v2x_regime: 0/1 autocracy; 2/3 democracy; subset only',
                'sample_validation': sample_checks, 'expected_counts': EXPECTED,
                'all_expected_counts_match': all(s['expected_counts_match'] for s in sample_counts),
                'thresholds': THRESHOLDS, 'protected_files_unchanged': unchanged,
                'protected_file_sha256': hashes,
                'successful_fits': sum(data['successful'] for data in diagnostics), 'attempted_fits': len(diagnostics),
                'dependencies': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'pandas': pd.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__}}
    write_json(output_directory/'analysis_manifest.json', metadata)
    report(output_directory, diagnostics, sample_counts)
    assert unchanged, 'A protected original file changed during this run'
    assert len(diagnostics) == 8


if __name__ == '__main__':
    main()
