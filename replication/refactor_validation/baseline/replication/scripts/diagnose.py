"""Storage and optimizer diagnostics; never substitutes these fits for predictions."""
from replicate import *


def main():
    out = BASE/'results'
    specs = parse_log(ROOT/'Analysis_20260920.log')
    comparisons, optimizer = [], []
    for storage, spline_storage in [('float32','float32'),('float32','float64'),('float64','float64')]:
        d = prepare(ROOT/'data/processed/cases_v6_short.csv', storage, spline_storage)
        for spec in specs:
            s,X,names,_ = design(d,spec)
            fit,b,V,T,_,_,_ = fit_model(s,X)
            values = dict(coef=b.ravel(order='F'), se=np.sqrt(np.diag(V)))
            values['lower']=values['coef']-Z95*values['se']
            values['upper']=values['coef']+Z95*values['se']
            errors = [compare(spec['name'],metric,values[metric][i],ref[metric],ref['term'],ref['equation'])
                      for i,ref in enumerate(spec['logged_rows']) for metric in values]
            comparisons.append(dict(model=spec['name'], import_storage=storage, spline_storage=spline_storage,
                printed_precision_failures=sum(not e['passed'] for e in errors),
                max_coef_error=max(e['absolute_difference'] for e in errors if e['metric']=='coef'),
                max_se_error=max(e['absolute_difference'] for e in errors if e['metric']=='se')))
            if (storage,spline_storage)!=('float32','float32'):
                continue
            rounded = np.array([float(r['coef']) for r in spec['logged_rows']]).reshape(b.shape,order='F')
            theta = np.linalg.solve(T,rounded).ravel(order='F')
            refit = fit.model.fit(start_params=theta, method='newton', maxiter=100,tol=1e-12,disp=False)
            refit_b = T@refit.params
            # Compare full-sample probabilities and focal slope error; cancellation in
            # rounded year coefficients can amplify intercept/likelihood diagnostics.
            optimizer.append(dict(model=spec['name'],
                loglik_at_rounded_logged_coefficients=fit.model.loglike(theta),
                loglik_at_python_optimum=fit.llf,
                loglik_improvement_from_rounded_log=fit.llf-fit.model.loglike(theta),
                refit_from_rounded_log_max_coefficient_difference=float(np.max(np.abs(refit_b-b))),
                max_probability_difference_from_rounded_log=float(np.max(np.abs(probabilities(X,b)-probabilities(X,rounded)))),
                max_focal_coefficient_difference=float(np.max(np.abs((b-rounded)[:len(spec['focal_predictors'])])))))
    pd.DataFrame(comparisons).to_csv(out/'storage_diagnostics.csv',index=False)
    pd.DataFrame(optimizer).to_csv(out/'optimizer_diagnostics.csv',index=False)


if __name__=='__main__':
    main()
