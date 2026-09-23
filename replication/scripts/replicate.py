"""Reconstruct the executed Stata models; validate before producing predictions.

Run from any directory. No repository model or prediction code is imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from decimal import Decimal
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax, expit, logit
from scipy.stats import norm
import statsmodels
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'replication'
FOCAL = ['court_independence_lag1', 'v2jupoatck_lag1', 'v2jupack_lag1',
         'v2jureform_lag1', 'v2jupurge_lag1', 'wdj_expression_lag1',
         'wdj_press_lag1', 'wdj_citizen_lag1']
MODES = ['mode_electronic_internet', 'mode_press_newspapers', 'mode_public_assembly',
         'mode_public_speech', 'mode_non_verbal_expression']
LABELS = ['Mixed Outcome', 'Contracts Expression', 'Expands Expression']
EQUATIONS = ['Contracts_Expression', 'Expands_Expression']
Z95 = norm.ppf(.975)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def parse_log(path):
    """Read the ENTIRE log; extract only detailed fits and estat ic, never esttab."""
    log = path.read_text()
    models = []
    macros = {}
    # Join Stata's wrapped command lines, then expand only invoked macros.
    for match in re.finditer(r'^\. local (\w+) "(.*?)"', log, re.M | re.S):
        macros[match[1]] = re.sub(r'\n>\s?', '', match[2])
    starts = list(re.finditer(r'^\. mlogit\s+', log, re.M))
    for i, start in enumerate(starts):
        block = log[start.start():starts[i+1].start() if i+1 < len(starts) else len(log)]
        name = re.search(r'estimates store (\w+)', block)[1]
        raw = block.split('\n\n', 1)[0]
        command = re.sub(r'\s+', ' ', re.sub(r'^>|///', ' ', raw, flags=re.M)).strip()[2:]
        expanded = command
        for macro, value in macros.items():
            expanded = expanded.replace('`' + macro + "'", value)
        focal = [v for v in FOCAL if v in expanded]
        controlled = 'i.lgl_systm' in expanded
        expected = 'mlogit decision ' + ' '.join(focal)
        if controlled:
            expected += ' i.lgl_systm i.high_court ib4.defendant5 ' + ' '.join('i.'+v for v in MODES)
        expected += ' yearspline* , vce(cluster country_id) baseoutcome(2)'
        assert re.sub(r'\s+', ' ', expanded) == expected, expanded
        n = int(re.search(r'Number of obs\s*=\s*([\d,]+)', block)[1].replace(',', ''))
        clusters = int(re.search(r'adjusted for (\d+) clusters', block)[1])
        ll = re.search(r'^Log pseudolikelihood\s*=\s*([-\d.]+)', block, re.M)[1]
        ic = re.search(r'^\s*' + name + r'\s*\|\s*(.+)$', block, re.M)[1].split()
        rows = []
        equation = None
        regression = block.split('estimates store')[0]
        assert re.search(r'Mixed_Outcome\s*\|\s*\(base outcome\)', regression)
        for line in regression.splitlines():
            if '|' not in line:
                continue
            term, values = (s.strip() for s in line.split('|', 1))
            if term in EQUATIONS:
                equation = term
            tokens = values.split()
            if len(tokens) == 6 and equation:
                try:
                    list(map(float, tokens))
                except ValueError:
                    continue
                rows.append(dict(equation=equation, term=term,
                                 coef=tokens[0], se=tokens[1], lower=tokens[4], upper=tokens[5]))
        models.append(dict(name=name, command=command, expanded_command=expanded,
                           focal_predictors=focal, controlled=controlled, logged_N=n,
                           logged_clusters=clusters, logged_ll=ll,
                           logged_k=int(ic[3]), logged_aic=ic[4], logged_bic=ic[5],
                           logged_rows=rows,
                           log_lines=[log[:start.start()].count('\n')+1,
                                      log[:start.start()+block.index('estimates store')].count('\n')+1]))
    assert [m['name'] for m in models] == ([f'indep{i}' for i in range(1,3)] +
        [f'attack{i}' for i in range(1,6)] + [f'dejure{i}' for i in range(1,7)] +
        [f'full{i}' for i in range(1,5)])
    return models


def prepare(path, storage='float32', spline_storage='float32'):
    d = pd.read_csv(path, na_values=['.']).copy()
    # Stata's default numeric storage is float; arithmetic/estimation uses double.
    for v in FOCAL[1:] + ['v2juhcind_lag1', 'v2juncind_lag1']:
        d[v] = d[v].astype(storage).astype('float64')
    d = d.copy()
    d['court_independence_lag1'] = np.nan
    d.loc[d.high_court.eq(1), 'court_independence_lag1'] = d.v2juhcind_lag1
    d.loc[d.low_court.eq(1), 'court_independence_lag1'] = d.v2juncind_lag1
    d['y'] = d.decision_direction.map({v:i for i,v in enumerate(LABELS)})
    d['defendant5'] = d.defendant_classification_correct.map(
        {'citizen':1, 'press':2, 'intermediary':3, 'government':4, 'other':5, 'unclear':5})
    d['lgl_systm'] = d.legal_system.map({'Common':0, 'Civil':1, 'Mixed':2})
    x = d.year.astype(float)
    d['yearspline1'] = x
    d['yearspline2'] = ((np.maximum(x-2001, 0)**3 -
        21/6*np.maximum(x-2016, 0)**3 + 15/6*np.maximum(x-2022, 0)**3) / 21**2
        ).astype(spline_storage).astype('float64')
    return d


def design(d, spec):
    required = ['y', 'country_id', 'yearspline1', 'yearspline2'] + spec['focal_predictors']
    if spec['controlled']:
        required += ['lgl_systm', 'defendant5', 'high_court'] + MODES
    sample = d.dropna(subset=required).copy()
    columns = {v:sample[v].to_numpy(float) for v in spec['focal_predictors']}
    if spec['controlled']:
        columns.update({v:sample.lgl_systm.eq(k).to_numpy(float) for k,v in [(1,'Civil'),(2,'Mixed')]})
        columns['1.high_court'] = sample.high_court.to_numpy(float)
        columns.update({v:sample.defendant5.eq(k).to_numpy(float) for k,v in
                        [(1,'Citizen'),(2,'Press'),(3,'Intermediary'),(5,'Other/unclear')]})
        columns.update({'1.'+v:sample[v].to_numpy(float) for v in MODES})
    columns.update({v:sample[v].to_numpy(float) for v in ['yearspline1','yearspline2']})
    columns['_cons'] = np.ones(len(sample))
    return sample, np.column_stack(list(columns.values())), list(columns), required


def fit_model(sample, X):
    """Condition columns without changing spline basis or fitted model.

    W=X T, beta_stata=T beta_work; V_stata=A V_work A', A=I_2 kron T.
    """
    p = X.shape[1]
    T = np.eye(p)
    scale = X[:,:-1].std(axis=0)
    T[np.arange(p-1), np.arange(p-1)] = 1/scale
    T[-1,:-1] = -X[:,:-1].mean(axis=0)/scale
    W = X @ T
    model = MNLogit(sample.y.to_numpy(int), W, check_rank=True, missing='raise')
    fit = model.fit(method='newton', maxiter=100, tol=1e-12, disp=False)
    theta = fit.params.ravel(order='F')
    scores = model.score_obs(theta)
    bread = np.linalg.inv(-model.hessian(theta))
    groups, ids = np.unique(sample.country_id.to_numpy(), return_inverse=True)
    sums = np.zeros((len(groups), len(theta)))
    np.add.at(sums, ids, scores)
    V = (len(groups)/(len(groups)-1)) * bread @ (sums.T @ sums) @ bread
    V = (V+V.T)/2
    A = np.kron(np.eye(2), T)
    beta, covariance = T @ fit.params, A @ V @ A.T
    check_cov = cov_cluster(fit, ids, use_correction=False)*len(groups)/(len(groups)-1)
    assert np.allclose(V, check_cov, rtol=1e-9, atol=1e-11)
    assert np.allclose(X@beta, W@fit.params, atol=1e-11, rtol=1e-11)
    return fit, beta, covariance, T, V, len(groups), float(np.max(np.abs(scores.sum(axis=0))))


def tolerance(token):
    """Half of the last printed decimal unit; tiny arithmetic slack only.

    Uses the literal token, including scientific notation. Stripped trailing zeros
    can make this conservative. No tolerance is fitted to replication residuals.
    """
    return float(Decimal(10) ** Decimal(Decimal(token).as_tuple().exponent))/2 + 1e-10


def compare(name, metric, value, token, term='', equation=''):
    tol = tolerance(token)
    return dict(model=name, metric=metric, equation=equation, term=term,
                python=float(value), stata=float(token), printed_token=token,
                absolute_difference=abs(float(value)-float(token)), tolerance=tol,
                passed=bool(abs(float(value)-float(token)) <= tol))


def probabilities(X, beta):
    return softmax(np.column_stack([np.zeros(len(X)), X@beta]), axis=1)


def average_gradient(X, beta):
    P = probabilities(X, beta)
    gradients = []
    for j in range(3):
        gradients.append(np.concatenate([np.mean(
            (P[:,j] * ((j == k)-P[:,k]))[:,None]*X, axis=0) for k in [1,2]]))
    return P, np.array(gradients)


def predict_grid(spec, sample, X, names, fit, beta, V, T, size=51):
    rows, checks, grids, support = [], [], [], []
    for focal in spec['focal_predictors']:
        observed = sample[focal].to_numpy()
        grid = np.linspace(observed.min(), observed.max(), size)
        grids.append(dict(model=spec['name'], predictor=focal, minimum=float(grid[0]),
                          maximum=float(grid[-1]), N=len(sample), grid_size=size,
                          definition='numpy.linspace(minimum, maximum, grid_size), endpoints included'))
        support.extend(dict(model=spec['name'], predictor=focal, row_id=int(i)+1, value=float(v))
                       for i,v in zip(sample.index, observed))
        max_predict_error = max_sum_error = max_gradient_error = 0.
        min_probability, max_probability = 1., 0.
        for g, value in enumerate(grid):
            Xg = X.copy()
            Xg[:,names.index(focal)] = value
            P, G = average_gradient(Xg, beta)
            avg = P.mean(axis=0)
            max_predict_error = max(max_predict_error, float(np.max(np.abs(P-fit.predict(Xg@T)))))
            max_sum_error = max(max_sum_error, float(np.max(np.abs(P.sum(axis=1)-1))))
            min_probability = min(min_probability, float(P.min()))
            max_probability = max(max_probability, float(P.max()))
            if g in {0, size//2, size-1}:
                # Central numerical derivatives in exported Stata parameter order.
                flat = beta.ravel(order='F')
                numerical = np.empty_like(G)
                for h in range(len(flat)):
                    # Bound change in each linear predictor to ~1e-4.
                    step = 1e-4/max(1., np.max(np.abs(Xg[:,h % len(names)])))
                    plus, minus = flat.copy(), flat.copy()
                    plus[h] += step
                    minus[h] -= step
                    numerical[:,h] = (probabilities(Xg, plus.reshape(beta.shape, order='F')).mean(axis=0)-
                                      probabilities(Xg, minus.reshape(beta.shape, order='F')).mean(axis=0))/(2*step)
                err = np.max(np.abs(G-numerical)/(1+np.abs(G)))
                max_gradient_error = max(max_gradient_error, float(err))
            variances = np.einsum('ij,jk,ik->i', G, V, G)
            assert np.all(variances >= 0)
            se = np.sqrt(variances)
            # Delta method on logit of the AVERAGED probability; inverse-logit CI.
            logit_se = se/(avg*(1-avg))
            lo, hi = expit(logit(avg)-Z95*logit_se), expit(logit(avg)+Z95*logit_se)
            for j, label in enumerate(LABELS):
                rows.append(dict(model=spec['name'], predictor=focal, grid_value=float(value),
                                 outcome=label, outcome_code_stata=[2,1,3][j], probability=float(avg[j]),
                                 lower=float(lo[j]), upper=float(hi[j]), se_probability=float(se[j]), N=len(sample)))
        checks.append(dict(model=spec['name'], predictor=focal,
                           max_manual_predict_difference=max_predict_error,
                           max_probability_sum_error=max_sum_error,
                           min_probability=min_probability, max_probability=max_probability,
                           max_gradient_scaled_error=max_gradient_error,
                           passed=bool(max_predict_error<1e-11 and max_sum_error<1e-12 and
                                       0<=min_probability<=max_probability<=1 and max_gradient_error<1e-7)))
    return rows, checks, grids, support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE/'results')
    parser.add_argument('--storage', choices=['float32','float64'], default='float32')
    parser.add_argument('--spline-storage', choices=['float32','float64'], default='float32')
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    log_path, data_path = ROOT/'Analysis_20260920.log', ROOT/'data/processed/cases_v6_short.csv'
    specs = parse_log(log_path)
    d = prepare(data_path, args.storage, args.spline_storage)
    comparisons, summaries, fitted, estimates = [], [], [], []
    memberships = pd.DataFrame({'row_id':np.arange(1,len(d)+1), 'case_id':d.case_id})
    for spec in specs:
        sample, X, names, required = design(d, spec)
        fit, beta, V, T, working_V, clusters, score = fit_model(sample, X)
        assert list(fit.model._ynames_map.values()) == ['0','1','2']
        assert [(r['equation'], r['term']) for r in spec['logged_rows']] == [(e,t) for e in EQUATIONS for t in names]
        b = beta.ravel(order='F')
        se = np.sqrt(np.diag(V))
        for j, ref in enumerate(spec['logged_rows']):
            for metric, value in [('coef',b[j]),('se',se[j]),('lower',b[j]-Z95*se[j]),('upper',b[j]+Z95*se[j])]:
                comparisons.append(compare(spec['name'],metric,value,ref[metric],ref['term'],ref['equation']))
            estimates.append(dict(model=spec['name'], equation=ref['equation'], term=ref['term'],
                                  coefficient=b[j], clustered_se=se[j], lower=b[j]-Z95*se[j], upper=b[j]+Z95*se[j]))
        k = len(b)
        for metric, value, token in [('ll', fit.llf, spec['logged_ll']),
                                     ('aic', -2*fit.llf+2*k, spec['logged_aic']),
                                     ('bic', -2*fit.llf+np.log(len(sample))*k, spec['logged_bic'])]:
            comparisons.append(compare(spec['name'],metric,value,token))
        expected_n = 825 if spec['name'].startswith(('indep','full')) else 1074
        structural = dict(N=len(sample), logged_N=spec['logged_N'], clusters=clusters,
            logged_clusters=spec['logged_clusters'], parameters=k, logged_parameters=spec['logged_k'],
            rank=int(np.linalg.matrix_rank(X)), columns=len(names),
            converged=bool(fit.mle_retvals['converged']), finite=bool(np.isfinite(b).all() and np.isfinite(V).all()),
            max_absolute_working_score=score, outcome_mapping_pass=True)
        structural_pass = (len(sample)==spec['logged_N']==expected_n and clusters==spec['logged_clusters'] and
            k==spec['logged_k'] and structural['rank']==len(names) and structural['converged'] and structural['finite'] and score<1e-7)
        local = [v for v in comparisons if v['model']==spec['name']]
        summaries.append(dict(model=spec['name'], **structural, structural_pass=bool(structural_pass),
            numerical_failures=sum(not r['passed'] for r in local), all_printed_comparisons_pass=all(r['passed'] for r in local)))
        spec.update(design_columns=names, complete_case_variables=required,
                    sample_row_ids=(sample.index+1).tolist(), actual_N=len(sample), actual_clusters=clusters,
                    formula='decision ~ ' + spec['expanded_command'].split('decision ',1)[1].split(' ,')[0].replace('yearspline*','yearspline1 yearspline2'),
                    reference_categories={'outcome':'Mixed Outcome (Stata 2; Python 0)', 'legal_system':'Common (0)',
                                          'defendant5':'Government (4)', 'high_court':0, **{v:0 for v in MODES}},
                    parameter_order=[e+':'+t for e in EQUATIONS for t in names],
                    covariance_correction=clusters/(clusters-1))
        memberships[spec['name']] = d.index.isin(sample.index)
        np.savez(out/'models'/f"{spec['name']}.npz", beta=beta, covariance=V, X=X, transform=T,
                 working_covariance=working_V, row_ids=sample.index.to_numpy()+1, columns=np.array(names),
                 equation_labels=np.array(EQUATIONS), probability_labels=np.array(LABELS))
        fitted.append((spec,sample,X,names,fit,beta,V,T))
        print(spec['name'], summaries[-1], flush=True)
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(out/'validation_comparisons.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'validation_models.csv', index=False)
    pd.DataFrame(estimates).to_csv(out/'coefficients.csv', index=False)
    memberships.to_csv(out/'estimation_samples.csv', index=False)
    d[['year','yearspline1','yearspline2']].drop_duplicates().sort_values('year').to_csv(out/'spline_basis.csv',index=False)
    metadata = dict(models=specs, spline=dict(knots=[2001,2016,2022], construction_sample_N=len(d),
        basis=['year','((year-2001)_+^3 - 3.5*(year-2016)_+^3 + 2.5*(year-2022)_+^3)/441'],
        storage=args.spline_storage), numeric_import_storage=args.storage,
        outcome_codes={'1':'Contracts Expression','2':'Mixed Outcome','3':'Expands Expression'},
        python_probability_order=LABELS, parameter_flatten_order='F (equation-major)',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [log_path,data_path]},
        data_columns=len(pd.read_csv(data_path,nrows=0).columns), logged_data_columns=116,
        dependencies={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                      'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        comparison_tolerance='half last printed token unit + 1e-10; no relative tolerance',
        notes=['CSV has 136 columns; log import reported 116. Only executed-command variables used.',
               'Stata set type and version were not recorded; default float storage assumed for imported continuous predictors.',
               'Models fitted on centered/scaled columns of exact Stata basis; T and full transformed covariance exported.'])
    write_json(out/'model_manifest.json', metadata)
    # Validation artifacts are persisted BEFORE predictions, including all failures.
    write_json(out/'validation_summary.json',dict(models=summaries,
        maxima_by_metric=comparison_df.groupby('metric').absolute_difference.max().to_dict(),
        failed_comparisons=int((~comparison_df.passed).sum())))
    if args.fit_only:
        return
    assert all(s['structural_pass'] for s in summaries), 'Structural failure; inspect validation before prediction'
    rows, checks, grids, support = [], [], [], []
    for entry in fitted:
        r,c,g,s = predict_grid(*entry)
        rows.extend(r); checks.extend(c); grids.extend(g); support.extend(s)
    pd.DataFrame(rows).to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'prediction_validation.csv', index=False)
    pd.DataFrame(grids).to_csv(out/'prediction_grids.csv', index=False)
    pd.DataFrame(support).to_csv(out/'observed_support.csv', index=False)
    assert all(c['passed'] for c in checks), 'Prediction validation failed'
    assert len(grids)==24 and len(rows)==24*51*3
    print(f'Exported {len(grids)} grids and {len(rows)} prediction rows to {out}')


if __name__ == '__main__':
    main()
