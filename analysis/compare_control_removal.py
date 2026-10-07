"""Audit the four saved fixed-sample models and render only final histogram figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import workflow as work
from plot_probabilities import saved_prediction, COLORS
import matplotlib.pyplot as plt

RUN = 'control_removal_comparison_01'
BASE = 'full2_reduced'
SELECTED = 'full2_reduced_no_speech_no_non_verbal'
TITLES = {'court_independence_lag1': ('Judicial Independence', 'judicial_independence'),
          'v2jupurge_lag1': ('Judicial Purges', 'judicial_purges'),
          'wdj_expression_lag1': ('De Jure Expression', 'de_jure_expression')}


def main():
    directory = work.run_path(RUN)
    out = directory/'comparison'
    out.mkdir(exist_ok=False)
    (out/'figures').mkdir()
    manifest = json.loads((directory/'run_manifest.json').read_text())
    assert manifest['sample_policy'] == 'fixed' and manifest['reference_model'] == 'full2'
    assert work.environment() == manifest['environment']
    assert work.fingerprint(directory/'configuration.json') == manifest['configuration_sha256']
    for path, digest in manifest['numerical_sources'].items():
        assert work.fingerprint(work.ROOT/path) == digest
    protected = [p for root in [work.ROOT/'data', work.ROOT/'replication',
                  work.run_path('autocracy_reduced_01'), work.run_path('democracy_reduced_01')]
                  for p in root.rglob('*') if p.is_file()]
    before = {str(p.relative_to(work.ROOT)): work.fingerprint(p) for p in protected}
    tables, coefficients, predictions, validations, grids, supports = [], [], [], [], [], []
    samples, states, checks = {}, {}, {}
    for fit in manifest['fits']:
        regime, name = fit['sample'], fit['model']
        assert fit['successful'] and fit['prediction_allowed']
        d = json.loads(work.verify_artifact(directory, fit, 'diagnostics').read_text())
        member = pd.read_csv(work.verify_artifact(directory, fit, 'membership'))
        with np.load(work.verify_artifact(directory, fit, 'state')) as z:
            state = {key: z[key] for key in z.files}
        if regime not in samples:
            samples[regime] = member
            states[regime] = state
            old = work.run_path(regime+'_reduced_01')
            old_manifest = json.loads((old/'run_manifest.json').read_text())
            old_fit = old_manifest['fits'][0]
            old_config = json.loads((old/'configuration.json').read_text())
            new_config = json.loads((directory/'configuration.json').read_text())
            for key in ['estimator', 'categories', 'control_groups', 'design_control_order', 'conventions']:
                assert old_config[key] == new_config[key]
            assert fit['specification'] == old_fit['specification']
            pd.testing.assert_frame_equal(member, pd.read_csv(work.verify_artifact(old, old_fit, 'membership')))
            with np.load(work.verify_artifact(old, old_fit, 'state')) as previous:
                for key in ['X','beta','covariance','transform','working_covariance','working_parameters','row_ids','outcome','focal_values']:
                    np.testing.assert_array_equal(state[key], previous[key])
            checks[regime] = {'baseline_reproduction_exact': True, 'cases': len(member),
                              'membership_sha256': fit['membership_sha256']}
        pd.testing.assert_frame_equal(member, samples[regime])
        for key in ['row_ids','outcome','country_ids','focal_values']:
            np.testing.assert_array_equal(state[key], states[regime][key])
        common = [c for c in state['columns'] if c in states[regime]['columns']]
        for c in common:
            np.testing.assert_array_equal(state['X'][:, list(state['columns']).index(c)],
                                          states[regime]['X'][:, list(states[regime]['columns']).index(c)])
        k = state['beta'].size  # Both nonreference outcome equations, including intercepts.
        ll = float(work.numerics.MNLogit(state['outcome'].astype(int), state['X']@state['transform']).loglike(state['working_parameters']))
        assert abs(ll-d['estimates']['terminal_loglik']) < 1e-10
        tables.append(dict(specification=name, regime=regime, cases=len(member), countries=member.country_id.nunique(),
                           parameters=k, converged=d['optimizer']['converged'], iterations=d['optimizer']['iterations'],
                           separation=d['separation']['status'], working_hessian_condition=d['hessian']['condition_working'],
                           finite_se=d['finiteness']['clustered_standard_errors'], finite_ci=d['finiteness']['confidence_intervals'],
                           covariance_crosscheck=d['covariance']['statsmodels_crosscheck_pass'],
                           covariance_rank=d['covariance']['rank_working'], covariance_correction=d['covariance']['correction'],
                           linear_predictor_crosscheck=d['estimates']['linear_predictor_transform_pass'],
                           log_likelihood=ll, AIC=-2*ll+2*k, BIC=-2*ll+k*np.log(len(member))))
        coef = pd.read_csv(directory/'fits'/f'{name}__{regime}__coefficients.csv')
        coefficients.append(coef.loc[coef.term.isin(TITLES)])
        p,v,g,s = saved_prediction(directory, manifest, fit, None)
        assert all(check['passed'] for check in v)
        for dest, rows in [(predictions,p),(validations,v),(grids,g),(supports,s)]:
            dest.extend({**row, 'sample': regime} for row in rows)
    table = pd.DataFrame(tables)
    for metric in ['log_likelihood','AIC','BIC']:
        base = table.loc[table.specification.eq(BASE)].set_index('regime')[metric]
        table['delta_'+metric] = table[metric]-table.regime.map(base)
    table.to_csv(out/'model_comparison.csv', index=False)
    coef = pd.concat(coefficients)
    basecoef = coef.loc[coef.model.eq(BASE)].set_index(['sample','equation','term'])
    for metric in ['coefficient','clustered_se','lower','upper']:
        coef['delta_'+metric] = [row[metric]-basecoef.loc[(row['sample'],row.equation,row.term),metric] for _,row in coef.iterrows()]
    coef.to_csv(out/'focal_coefficient_sensitivity.csv', index=False)
    pred = pd.DataFrame(predictions)
    pred.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(validations).to_csv(out/'prediction_validation.csv', index=False)
    pd.DataFrame(grids).to_csv(out/'prediction_grids.csv', index=False)
    support = pd.DataFrame(supports)
    support.to_csv(out/'observed_support.csv', index=False)
    sensitivity = []
    keys = ['predictor','outcome','grid_value']
    for regime, frame in pred.groupby('sample'):
        ref = frame.loc[frame.model.eq(BASE)].set_index(keys).sort_index()
        for name, rows in frame.groupby('model'):
            other = rows.set_index(keys).sort_index()
            pd.testing.assert_index_equal(ref.index, other.index)
            for (predictor,outcome), part in other.groupby(level=[0,1]):
                basepart = ref.loc[part.index]
                sensitivity.append(dict(regime=regime, specification=name, predictor=predictor, outcome=outcome,
                    max_abs_probability_change=float((part.probability-basepart.probability).abs().max()),
                    max_abs_lower_change=float((part.lower-basepart.lower).abs().max()),
                    max_abs_upper_change=float((part.upper-basepart.upper).abs().max()),
                    low_probability=part.probability.iloc[0], high_probability=part.probability.iloc[-1],
                    endpoint_change=part.probability.iloc[-1]-part.probability.iloc[0]))
    pd.DataFrame(sensitivity).to_csv(out/'probability_sensitivity.csv', index=False)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'savefig.facecolor':'white'})
    index = []
    for (regime,predictor), rows in pred.loc[pred.model.eq(SELECTED)].groupby(['sample','predictor']):
        title, slug = TITLES[predictor]
        observed = support.loc[support.model.eq(SELECTED)&support['sample'].eq(regime)&support.predictor.eq(predictor),'value'].to_numpy()
        fig = plt.figure(figsize=(12,5.4))
        gs = fig.add_gridspec(2,3,height_ratios=[5,1],hspace=.12,wspace=.18)
        for j,(label,color) in enumerate(COLORS.items()):
            ax = fig.add_subplot(gs[0,j])
            frame = rows.loc[rows.outcome.eq(label)].sort_values('grid_value')
            ax.fill_between(frame.grid_value,frame.lower,frame.upper,color=color,alpha=.18,linewidth=0)
            ax.plot(frame.grid_value,frame.probability,color=color,lw=2)
            ax.set(title=label,ylim=(0,1),yticks=np.linspace(0,1,6),xlim=(frame.grid_value.min(),frame.grid_value.max()))
            ax.grid(axis='y',alpha=.18)
            ax.tick_params(axis='x',labelbottom=False)
            if j==0: ax.set_ylabel('Predicted probability')
            dist = fig.add_subplot(gs[1,j],sharex=ax)
            dist.hist(observed,bins=25,color='#b8bdc3',edgecolor='white',linewidth=.3)
            dist.plot(observed,np.zeros(len(observed)),'|',color='#555555',alpha=.13,markersize=5)
            dist.set_xlabel(title)
            if j==0: dist.set_ylabel('Count')
            dist.tick_params(labelsize=9)
        fig.suptitle(title,y=.97,fontsize=16)
        fig.subplots_adjust(top=.86,bottom=.105,left=.07,right=.98)
        filename=f'{regime}_{SELECTED}_{slug}.png'
        fig.savefig(out/'figures'/filename,dpi=180)
        plt.close(fig)
        index.append(dict(regime=regime,specification=SELECTED,predictor=predictor,file='figures/'+filename))
    pd.DataFrame(index).to_csv(out/'figure_index.csv',index=False)
    assert len(index)==6
    assert before == {str(p.relative_to(work.ROOT)):work.fingerprint(p) for p in protected}
    work.write_json(out/'verification.json', {'sample_checks':checks,'identical_membership_all_specifications':True,
        'identical_common_design_columns':True,'identical_prediction_grids':True,'prediction_checks_pass':True,
        'protected_files_unchanged_during_comparison':True,'protected_sha256':before,
        'selected':SELECTED,'figure_count':len(index),
        'output_sha256':{str(p.relative_to(out)):work.fingerprint(p) for p in out.rglob('*') if p.is_file()}})
    print(table.to_string(index=False))
    print('\nSelected focal coefficients:\n',coef.loc[coef.model.eq(SELECTED)].to_string(index=False))
    print('\nProbability sensitivity:\n',pd.DataFrame(sensitivity).loc[lambda x:x.specification.eq(SELECTED)].to_string(index=False))

if __name__=='__main__': main()
