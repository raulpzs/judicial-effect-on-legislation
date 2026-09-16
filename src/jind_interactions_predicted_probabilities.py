"""
Average adjusted predicted probabilities for saved WDJ × judicial independence models.
Adapted from mnl_margins.py; no models are refitted.
"""
from pathlib import Path
import ast
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jind_interactions"
MODEL_DIR = OUTPUT_DIR / "fitted_models_mixed"
MATRIX_DIR = OUTPUT_DIR / "model_matrices_mixed"
POSTEST_DIR = OUTPUT_DIR / "postestimation"
PLOT_DIR = POSTEST_DIR / "plots"


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

OUTCOME_LABELS = {
    0: "Mixed Outcome",
    1: "Contracts Expression",
    2: "Expands Expression",
}

WDJ_LABELS = {
    "expression": "Expression",
    "citizen": "Citizen expression",
    "press": "Press expression",
}
CONTEXT_LABELS = {
    "reform": "Judicial reform",
    "attack": "Judicial attack",
    "pack": "Judicial packing",
    "purge": "Judicial purge",
}
SELECTED_EFFECTS = {
    f"spec_{measure}_jind_{context}": f"wdj_{measure}_lag1"
    for measure in WDJ_LABELS
    for context in CONTEXT_LABELS
}
WDJ_LEVELS = ["Low (p25)", "Median (p50)", "High (p75)"]
N_SIM = 1000
CI_LEVEL = 0.95
RANDOM_SEED = 12345


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_result(spec_name: str):
    """Load fitted statsmodels result object."""
    path = MODEL_DIR / f"{spec_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Could not find fitted model: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_X(spec_name: str) -> pd.DataFrame:
    """Load saved design matrix for a model spec."""
    path = MATRIX_DIR / f"{spec_name}_X.csv"

    if not path.exists():
        raise FileNotFoundError(f"Could not find model matrix: {path}")

    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------

def _predict_probs(result, X: pd.DataFrame, outcome_labels=None) -> pd.DataFrame:
    """
    Return predicted probabilities from the fitted model.
    """

    probs = pd.DataFrame(result.predict(X), index=X.index)

    if outcome_labels is not None:
        probs = probs.rename(columns=outcome_labels)

    return probs


def _predict_probs_from_params(
    result,
    X: pd.DataFrame,
    params_2d: np.ndarray,
    outcome_labels=None,
) -> pd.DataFrame:
    """
    Return predicted probabilities using supplied parameter values.

    This is used for simulated confidence intervals. The params_2d array
    should have the same shape as result.params.
    """

    probs = pd.DataFrame(
        result.model.predict(params_2d, exog=X),
        index=X.index,
    )

    if outcome_labels is not None:
        probs = probs.rename(columns=outcome_labels)

    return probs


def _make_counterfactual_matrix(X, variable, wdj_value, j_ind_value):
    """Change only the two constituent columns and their interaction."""
    interaction = f"{variable}:j_ind_lag1"
    changed = [variable, "j_ind_lag1", interaction]
    if not set(changed).issubset(X.columns):
        raise ValueError(f"Missing constituent/interaction columns: {changed}")
    X_cf = X.copy()
    X_cf[variable] = wdj_value
    X_cf["j_ind_lag1"] = j_ind_value
    X_cf[interaction] = X_cf[variable] * X_cf["j_ind_lag1"]
    assert np.array_equal(X_cf[interaction], X_cf[variable] * X_cf["j_ind_lag1"])
    assert X_cf.drop(columns=changed).equals(X.drop(columns=changed))
    return X_cf


# ---------------------------------------------------------------------
# Parameter simulation helpers
# ---------------------------------------------------------------------

def _get_param_vector_and_cov(result):
    """Map covariance labels explicitly to positions in the coefficient matrix."""
    params = result.params
    cov = result.cov_params()
    if not isinstance(params, pd.DataFrame) or not isinstance(cov, pd.DataFrame):
        raise ValueError("Labeled parameters/covariance required to verify ordering")
    if not isinstance(cov.index, pd.MultiIndex) or cov.index.nlevels != 2:
        raise ValueError("Cannot verify covariance equation/variable labels")
    if not cov.index.equals(cov.columns) or not cov.index.is_unique:
        raise ValueError("Covariance axes must have identical, unique labels")
    if list(params.columns) != [0, 1] or not params.index.is_unique:
        raise ValueError("Unexpected MNLogit parameter structure")

    # MNLogit columns 0 and 1 are equations for outcome codes 1 and 2.
    # Outcome coding is independently checked against the estimator and data.
    positions = {
        (str(result.model._ynames_map[j + 1]), variable): (i, j)
        for j in range(params.shape[1])
        for i, variable in enumerate(params.index)
    }
    if set(cov.index) != set(positions) or len(cov) != params.size:
        raise ValueError("Covariance labels do not map uniquely to coefficients")
    param_order = [positions[label] for label in cov.index]
    beta_hat = np.array([params.iloc[i, j] for i, j in param_order])
    bse = np.array([result.bse.iloc[i, j] for i, j in param_order])
    if not np.allclose(np.diag(cov), bse ** 2, rtol=1e-8, atol=1e-12):
        raise ValueError("Covariance ordering fails stored standard-error check")
    reconstructed = np.empty(params.shape)
    for value, (i, j) in zip(beta_hat, param_order):
        reconstructed[i, j] = value
    assert np.array_equal(reconstructed, params.to_numpy())
    if not np.isfinite(cov.to_numpy()).all() or not np.isfinite(beta_hat).all():
        raise ValueError("Non-finite coefficients/covariance")
    print("  Parameter/covariance ordering verified by equation and variable labels")
    return beta_hat, cov.to_numpy(), params.shape, param_order


def _draw_parameter_matrices(result, n_sim=1000, random_seed=12345):
    """Draw coefficients using the stored country-clustered covariance."""
    beta_hat, cov, param_shape, param_order = _get_param_vector_and_cov(result)
    rng = np.random.default_rng(random_seed)
    # Keep numerical covariance warnings visible; do not repair the covariance.
    draws = rng.multivariate_normal(
        mean=beta_hat, cov=cov, size=n_sim, check_valid="warn",
    )
    param_draws = []
    for draw in draws:
        params_2d = np.empty(param_shape)
        for value, (i, j) in zip(draw, param_order):
            params_2d[i, j] = value
        param_draws.append(params_2d)
    return param_draws


def _summarize_simulated_effects(sim_effects, ci_level=0.95):
    """Percentile simulation intervals, as in mnl_margins.py."""
    alpha = 1 - ci_level
    low_q = alpha / 2
    high_q = 1 - alpha / 2
    summary = (
        sim_effects.groupby("outcome")["estimate"]
        .quantile([low_q, high_q]).unstack().reset_index()
    )
    return summary.rename(columns={low_q: "ci_low", high_q: "ci_high"})


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_probabilities(probs):
    values = np.asarray(probs)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    if not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-10):
        raise ValueError("Outcome probabilities do not sum to one")


def _estimator_outcome_map():
    """Read only mapping definitions; never execute the estimation script."""
    path = PROJECT_ROOT / "src" / "multinom_jind_interactions.py"
    tree = ast.parse(path.read_text())
    names = {"OUTCOME_LEVELS", "BASE_CATEGORY_MAP", "BASE", "get_y_map"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )) or (isinstance(node, ast.FunctionDef) and node.name == "get_y_map")
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    mapping = namespace["get_y_map"](namespace["BASE"])
    if namespace["BASE"] != "mixed" or mapping != {v: k for k, v in OUTCOME_LABELS.items()}:
        raise ValueError("Current estimator does not use the required Mixed-base coding")
    return mapping


def _validate_model(result, X, spec_name, source, y_map):
    if result.cov_type != "cluster":
        raise ValueError(f"{spec_name}: stored covariance is not cluster robust")
    if result.model._ynames_map != {0: "0", 1: "1", 2: "2"}:
        raise ValueError(f"{spec_name}: unexpected fitted outcome codes")
    if not result.mle_retvals["converged"]:
        raise ValueError(f"{spec_name}: saved fit did not converge")
    if list(X.columns) != list(result.params.index):
        raise ValueError(f"{spec_name}: matrix columns differ from fitted parameters")
    if not X.index.equals(pd.Index(result.model.data.row_labels)):
        raise ValueError(f"{spec_name}: saved matrix row labels differ from fitted sample")
    if not np.allclose(X.to_numpy(), result.model.exog, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{spec_name}: saved matrix differs from fitted matrix")
    y = pd.read_csv(MATRIX_DIR / f"{spec_name}_y.csv", index_col=0)["outcome"]
    expected = source.loc[X.index, "decision_direction"].map(y_map)
    if not y.index.equals(X.index) or not np.array_equal(y, expected):
        raise ValueError(f"{spec_name}: outcome labels cannot be verified against saved y")
    if not np.array_equal(y, result.model.endog):
        raise ValueError(f"{spec_name}: fitted outcome codes differ from saved y")
    if not np.array_equal(result.cov_kwds["groups"], source.loc[X.index, "country"]):
        raise ValueError(f"{spec_name}: saved clusters differ from sample countries")
    if int(result.nobs) != len(X):
        raise ValueError(f"{spec_name}: model/matrix sample sizes differ")
    if len(X) != 825:
        warnings.warn(f"{spec_name}: saved estimation N={len(X)}, expected 825")
    print(f"  N={len(X)}; outcome coding and country clusters verified")


# ---------------------------------------------------------------------
# Main prediction functions
# ---------------------------------------------------------------------

def predicted_probabilities_with_ci(result, X, variable, wdj_values, j_ind_grid,
                                    random_seed=12345):
    """Average over the full estimation sample at each interaction-grid point."""
    param_draws = _draw_parameter_matrices(result, N_SIM, random_seed)
    rows = []
    cached_levels = {}
    for level, wdj_value in zip(WDJ_LEVELS, wdj_values):
        # Identical empirical quantiles imply identical predictions and intervals.
        # Keep each requested label, reusing the same model-level simulation draws.
        if wdj_value in cached_levels:
            for cached in cached_levels[wdj_value]:
                out = cached.copy()
                out["wdj_level"] = level
                rows.append(out)
            print(f"  Completed {level}: {wdj_value:.12g} (duplicate value)", flush=True)
            continue
        level_rows = []
        for j_ind_value in j_ind_grid:
            X_cf = _make_counterfactual_matrix(X, variable, wdj_value, j_ind_value)
            probs = _predict_probs(result, X_cf, outcome_labels=OUTCOME_LABELS)
            _validate_probabilities(probs)
            point = probs.mean().reset_index()
            point.columns = ["outcome", "predicted_probability"]
            _validate_probabilities(point["predicted_probability"].to_numpy())
            sim_estimates = []
            for sim_id, params_2d in enumerate(param_draws):
                sim_probs = _predict_probs_from_params(
                    result, X_cf, params_2d, outcome_labels=OUTCOME_LABELS,
                )
                _validate_probabilities(sim_probs)
                sim_estimates.append(sim_probs.to_numpy().mean(axis=0))
            sim_effects = pd.DataFrame({
                "outcome": np.tile(list(OUTCOME_LABELS.values()), len(param_draws)),
                "estimate": np.asarray(sim_estimates).ravel(),
            })
            ci = _summarize_simulated_effects(sim_effects, CI_LEVEL)
            out = point.merge(ci, on="outcome", how="left")
            out["wdj_variable"] = variable
            out["wdj_level"] = level
            out["wdj_value"] = wdj_value
            out["j_ind_value"] = j_ind_value
            out["n_obs"] = len(X)
            out["n_sim"] = N_SIM
            out["ci_level"] = CI_LEVEL
            rows.append(out)
            level_rows.append(out)
        cached_levels[wdj_value] = level_rows
        print(f"  Completed {level}: {wdj_value:.12g}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    values = out[["predicted_probability", "ci_low", "ci_high"]].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Probabilities or CI bounds outside [0, 1]")
    assert (out["ci_low"] <= out["ci_high"]).all()
    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_margins(effects_df, output_path, title="Average predicted probabilities"):
    """Three outcome panels with WDJ-level lines and 95% simulation ribbons."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    outcomes = ["Contracts Expression", "Mixed Outcome", "Expands Expression"]
    for ax, outcome in zip(axes, outcomes):
        for level in WDJ_LEVELS:
            sub = effects_df[(effects_df["outcome"] == outcome) &
                             (effects_df["wdj_level"] == level)].sort_values("j_ind_value")
            x = sub["j_ind_value"].to_numpy()
            line, = ax.plot(x, sub["predicted_probability"].to_numpy(),
                            label=f"{level}: {sub['wdj_value'].iloc[0]:.6g}")
            ax.fill_between(x, sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(),
                            color=line.get_color(), alpha=0.2)
        ax.set_ylim(0, 1)
        ax.set_xlim(effects_df["j_ind_value"].min(), effects_df["j_ind_value"].max())
        ax.set_ylabel("Average predicted probability")
        ax.set_title(outcome)
        ax.legend(title="WDJ value")
    axes[-1].set_xlabel("Judicial independence (j_ind_lag1)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    warnings.simplefilter("always")
    if {p.stem for p in MODEL_DIR.glob("*.pkl")} != set(SELECTED_EFFECTS):
        raise ValueError("Expected exactly the 12 judicial-independence fitted models")
    # Refuse to overwrite any prior post-estimation outputs.
    POSTEST_DIR.mkdir(parents=True, exist_ok=False)
    PLOT_DIR.mkdir()
    source = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cases_v6_short.csv")
    y_map = _estimator_outcome_map()
    all_probs = []
    support_rows = []
    for spec_i, (spec_name, variable) in enumerate(SELECTED_EFFECTS.items()):
        print(f"Processing {spec_name}", flush=True)
        result = load_result(spec_name)
        X = load_X(spec_name)
        _validate_model(result, X, spec_name, source, y_map)
        wdj_values = X[variable].quantile([0.25, 0.50, 0.75]).to_numpy()
        j_low, j_high = X["j_ind_lag1"].quantile([0.05, 0.95]).to_numpy()
        print(f"  {variable}: p25={wdj_values[0]:.12g}, p50={wdj_values[1]:.12g}, "
              f"p75={wdj_values[2]:.12g}; j_ind_lag1: p5={j_low:.12g}, p95={j_high:.12g}")
        if len(np.unique(wdj_values)) != 3:
            warnings.warn(f"{spec_name}: duplicate WDJ quantiles: {dict(zip(WDJ_LEVELS, wdj_values))}")
        j_ind_grid = np.linspace(j_low, j_high, 50)
        assert ((j_ind_grid >= j_low) & (j_ind_grid <= j_high)).all()
        support_rows.append(dict(spec=spec_name, n_obs=len(X), wdj_variable=variable,
                                 wdj_p25=wdj_values[0], wdj_p50=wdj_values[1],
                                 wdj_p75=wdj_values[2], j_ind_p5=j_low, j_ind_p95=j_high))
        # Preserve the template's reproducible, distinct model seed offsets.
        probs = predicted_probabilities_with_ci(
            result, X, variable, wdj_values, j_ind_grid,
            random_seed=RANDOM_SEED + 1000 * spec_i,
        )
        probs.insert(0, "spec", spec_name)
        all_probs.append(probs)
        _, measure, _, context = spec_name.split("_")
        plot_margins(probs, PLOT_DIR / f"{spec_name}_predicted_probabilities_mixed.png",
                     f"{WDJ_LABELS[measure]} × judicial independence — {CONTEXT_LABELS[context]} model")

    probs_all = pd.concat(all_probs, ignore_index=True)
    assert probs_all["spec"].nunique() == 12 and len(probs_all) == 5400
    assert np.allclose(probs_all.groupby(["spec", "wdj_level", "j_ind_value"])
                       ["predicted_probability"].sum(), 1, rtol=0, atol=1e-10)
    columns = ["spec", "wdj_variable", "wdj_level", "wdj_value", "j_ind_value", "outcome",
               "predicted_probability", "ci_low", "ci_high", "n_obs", "n_sim", "ci_level"]
    probs_all[columns].to_csv(POSTEST_DIR / "predicted_probabilities_mixed.csv", index=False)
    support = pd.DataFrame(support_rows)
    support.to_csv(POSTEST_DIR / "prediction_values_used.csv", index=False)
    print(support.to_string(index=False))
    print("Validated 12 models: interaction products, observed controls, outcome sums, "
          "probability/CI bounds, and simulated parameter ordering.")
    print("Saved 2 CSV files and 12 PNG figures.")
    print(f"Saved outputs to: {POSTEST_DIR}")


if __name__ == "__main__":
    main()
