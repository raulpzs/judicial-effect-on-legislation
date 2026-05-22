"""
Post-estimation predicted probability contrasts for multinomial logit models.

This script:
1. Loads fitted statsmodels MNLogit result objects.
2. Loads the matching model matrix for each spec.
3. Computes average predicted-probability changes for selected variables.
4. Uses coefficient simulation to add confidence intervals.
5. Generates a margins plot with confidence intervals.

Interpretation alternatives:
- Continuous variables: mean -> mean + 1 SD, p25 -> p75, or p10 -> p90.
- Binary variables: 0 -> 1.
- Estimates are reported in probability units and percentage points.
"""

from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "fitted_models"
MATRIX_DIR = OUTPUT_DIR / "model_matrices"
POSTEST_DIR = OUTPUT_DIR / "postestimation"
PLOT_DIR = POSTEST_DIR / "plots"

for path in [OUTPUT_DIR, MODEL_DIR, MATRIX_DIR, POSTEST_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

OUTCOME_LABELS = {
    0: "Mixed Outcome",
    1: "Contracts Expression",
    2: "Expands Expression",
}


SELECTED_EFFECTS = {
    "spec_p_reform": {
        "wdj_press_lag1": "sd",
        "v2jureform_lag1": "sd",
    },
    "spec_p_attack": {
        "wdj_press_lag1": "sd",
        "v2jupoatck_lag1": "sd",
    },
    "spec_p_reform_decision": {
        "wdj_press_lag1": "sd",
        "v2jureform_lag1": "sd",
    },
}


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


def _make_contrast_matrices(
    X: pd.DataFrame,
    variable: str,
    contrast: str,
):
    """
    Create two counterfactual design matrices for a variable contrast.

    Supported contrasts:
    - "sd": mean -> mean + 1 SD
    - "iqr": p25 -> p75
    - "p10_p90": p10 -> p90
    - "binary": 0 -> 1
    """

    if variable not in X.columns:
        raise ValueError(f"Variable not found in X: {variable}")

    X0 = X.copy()
    X1 = X.copy()

    x = pd.to_numeric(X[variable], errors="coerce")

    if x.notna().sum() == 0:
        raise ValueError(f"Variable is entirely non-numeric or missing: {variable}")

    if contrast == "sd":
        low = x.mean()
        high = x.mean() + x.std()
        contrast_label = "mean_to_mean_plus_1sd"

    elif contrast == "iqr":
        low = x.quantile(0.25)
        high = x.quantile(0.75)
        contrast_label = "p25_to_p75"

    elif contrast == "p10_p90":
        low = x.quantile(0.10)
        high = x.quantile(0.90)
        contrast_label = "p10_to_p90"

    elif contrast == "binary":
        low = 0
        high = 1
        contrast_label = "0_to_1"

    else:
        raise ValueError(
            "contrast must be one of: 'sd', 'iqr', 'p10_p90', 'binary'"
        )

    X0[variable] = low
    X1[variable] = high

    contrast_info = {
        "contrast": contrast_label,
        "low_value": low,
        "high_value": high,
    }

    return X0, X1, contrast_info


# ---------------------------------------------------------------------
# Parameter simulation helpers
# ---------------------------------------------------------------------

def _get_param_vector_and_cov(result):
    """
    Extract parameter vector, covariance matrix, and original parameter shape.

    This uses row-major flattening and reshaping consistently. For standard
    statsmodels MNLogit results, this should align with result.cov_params().
    """

    params = np.asarray(result.params)
    param_shape = params.shape

    beta_hat = params.ravel(order="C")
    cov = np.asarray(result.cov_params())

    if cov.shape != (beta_hat.size, beta_hat.size):
        raise ValueError(
            "Covariance matrix shape does not match flattened parameter vector. "
            f"params size={beta_hat.size}, cov shape={cov.shape}. "
            "Check statsmodels parameter ordering for this result object."
        )

    return beta_hat, cov, param_shape


def _draw_parameter_matrices(
    result,
    n_sim: int = 1000,
    random_seed: int = 12345,
):
    """
    Draw simulated coefficient matrices from the estimated sampling distribution.
    """

    beta_hat, cov, param_shape = _get_param_vector_and_cov(result)

    rng = np.random.default_rng(random_seed)

    # Small numerical issues can make covariance matrices appear non-PSD.
    # This is common enough that we warn instead of failing immediately.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = rng.multivariate_normal(
            mean=beta_hat,
            cov=cov,
            size=n_sim,
            check_valid="warn",
        )

    param_draws = [
        draw.reshape(param_shape, order="C")
        for draw in draws
    ]

    return param_draws


def _summarize_simulated_effects(
    sim_effects: pd.DataFrame,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize simulated effects into lower and upper confidence bounds.
    """

    alpha = 1 - ci_level
    low_q = alpha / 2
    high_q = 1 - alpha / 2

    summary = (
        sim_effects
        .groupby("outcome")["estimate"]
        .quantile([low_q, high_q])
        .unstack()
        .reset_index()
    )

    summary = summary.rename(
        columns={
            low_q: "ci_low",
            high_q: "ci_high",
        }
    )

    summary["ci_low_pp"] = summary["ci_low"] * 100
    summary["ci_high_pp"] = summary["ci_high"] * 100

    return summary


# ---------------------------------------------------------------------
# Main contrast functions
# ---------------------------------------------------------------------

def predicted_probability_contrast(
    result,
    X: pd.DataFrame,
    variable: str,
    contrast: str = "sd",
    outcome_labels=None,
) -> pd.DataFrame:
    """
    Estimate average predicted-probability change for one variable.

    Returns point estimates only. For confidence intervals, use
    predicted_probability_contrast_with_ci().
    """

    X0, X1, contrast_info = _make_contrast_matrices(
        X=X,
        variable=variable,
        contrast=contrast,
    )

    p0 = _predict_probs(result, X0, outcome_labels=outcome_labels)
    p1 = _predict_probs(result, X1, outcome_labels=outcome_labels)

    diff = p1 - p0

    out = diff.mean().reset_index()
    out.columns = ["outcome", "estimate"]

    out["estimate_pp"] = out["estimate"] * 100
    out["variable"] = variable
    out["contrast"] = contrast_info["contrast"]
    out["low_value"] = contrast_info["low_value"]
    out["high_value"] = contrast_info["high_value"]

    return out[
        [
            "variable",
            "contrast",
            "low_value",
            "high_value",
            "outcome",
            "estimate",
            "estimate_pp",
        ]
    ]


def predicted_probability_contrast_with_ci(
    result,
    X: pd.DataFrame,
    variable: str,
    contrast: str = "sd",
    outcome_labels=None,
    n_sim: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 12345,
) -> pd.DataFrame:
    """
    Estimate average predicted-probability change with simulation-based CIs.

    The point estimate uses the fitted model. Confidence intervals come from
    repeated draws from the estimated coefficient distribution.
    """

    X0, X1, contrast_info = _make_contrast_matrices(
        X=X,
        variable=variable,
        contrast=contrast,
    )

    # Point estimate from fitted model
    point = predicted_probability_contrast(
        result=result,
        X=X,
        variable=variable,
        contrast=contrast,
        outcome_labels=outcome_labels,
    )

    # Simulated effects
    param_draws = _draw_parameter_matrices(
        result=result,
        n_sim=n_sim,
        random_seed=random_seed,
    )

    sim_rows = []

    for sim_id, params_2d in enumerate(param_draws):
        p0 = _predict_probs_from_params(
            result=result,
            X=X0,
            params_2d=params_2d,
            outcome_labels=outcome_labels,
        )

        p1 = _predict_probs_from_params(
            result=result,
            X=X1,
            params_2d=params_2d,
            outcome_labels=outcome_labels,
        )

        diff = p1 - p0

        sim_effect = diff.mean().reset_index()
        sim_effect.columns = ["outcome", "estimate"]
        sim_effect["sim_id"] = sim_id

        sim_rows.append(sim_effect)

    sim_effects = pd.concat(sim_rows, ignore_index=True)

    ci = _summarize_simulated_effects(
        sim_effects=sim_effects,
        ci_level=ci_level,
    )

    out = point.merge(ci, on="outcome", how="left")

    out["n_sim"] = n_sim
    out["ci_level"] = ci_level

    return out[
        [
            "variable",
            "contrast",
            "low_value",
            "high_value",
            "outcome",
            "estimate",
            "estimate_pp",
            "ci_low",
            "ci_high",
            "ci_low_pp",
            "ci_high_pp",
            "n_sim",
            "ci_level",
        ]
    ]


def predicted_probabilities_long(
    result,
    X: pd.DataFrame,
    outcome_labels=None,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Return observation-level predicted probabilities in long format.
    """

    probs = _predict_probs(result, X, outcome_labels=outcome_labels)

    probs_long = (
        probs
        .reset_index()
        .melt(
            id_vars="index",
            var_name="outcome",
            value_name="predicted_probability",
        )
        .rename(columns={"index": "row_id"})
    )

    if model is not None:
        probs_long.insert(0, "model", model)

    return probs_long


# ---------------------------------------------------------------------
# Statsmodels get_margeff output
# ---------------------------------------------------------------------

def statsmodels_marginal_effects(result, spec_name: str) -> pd.DataFrame:
    """
    Save statsmodels average marginal effects.

    This is useful as a comparison, but for interpretation in the paper
    the finite-difference predicted-probability contrasts are usually clearer.
    """

    mfx = result.get_margeff(at="overall", method="dydx", dummy=True)
    df = mfx.summary_frame().reset_index()
    df.insert(0, "model", spec_name)

    return df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_margins(
    effects_df: pd.DataFrame,
    output_path: Path,
    title: str = "Predicted probability changes",
):
    """
    Generate a margins plot with confidence intervals.

    Assumes effects_df contains:
    - model
    - variable
    - outcome
    - estimate_pp
    - ci_low_pp
    - ci_high_pp
    """

    required = {
        "model",
        "variable",
        "outcome",
        "estimate_pp",
        "ci_low_pp",
        "ci_high_pp",
    }

    missing = required - set(effects_df.columns)

    if missing:
        raise ValueError(f"effects_df is missing required columns: {missing}")

    plot_df = effects_df.copy()

    plot_df["label"] = (
        plot_df["model"].astype(str)
        + "\n"
        + plot_df["variable"].astype(str)
    )

    outcomes = list(plot_df["outcome"].dropna().unique())

    y_labels = list(plot_df["label"].drop_duplicates())
    y_pos = np.arange(len(y_labels))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(y_labels))))

    if len(outcomes) == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.22, 0.22, len(outcomes))

    label_to_y = {label: i for i, label in enumerate(y_labels)}

    for outcome, offset in zip(outcomes, offsets):
        sub = plot_df[plot_df["outcome"] == outcome].copy()

        sub["y"] = sub["label"].map(label_to_y) + offset

        x = sub["estimate_pp"].to_numpy()
        y = sub["y"].to_numpy()

        xerr_low = x - sub["ci_low_pp"].to_numpy()
        xerr_high = sub["ci_high_pp"].to_numpy() - x

        ax.errorbar(
            x=x,
            y=y,
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="o",
            capsize=3,
            label=outcome,
        )

    ax.axvline(0, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Change in predicted probability, percentage points")
    ax.set_title(title)
    ax.legend(title="Outcome")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    all_effects = []
    all_probs = []
    all_mfx = []

    for spec_i, (spec_name, variable_map) in enumerate(SELECTED_EFFECTS.items()):
        print(f"Processing {spec_name}")

        result = load_result(spec_name)
        X = load_X(spec_name)

        # Observation-level predicted probabilities
        probs_long = predicted_probabilities_long(
            result=result,
            X=X,
            outcome_labels=OUTCOME_LABELS,
            model=spec_name,
        )
        all_probs.append(probs_long)

        # Selected finite-difference predicted-probability contrasts with CIs
        for var_i, (variable, contrast) in enumerate(variable_map.items()):
            print(f"  - {variable}: {contrast}")

            # Offset seed so each model-variable pair gets reproducible,
            # but distinct, simulation draws.
            seed = RANDOM_SEED + (1000 * spec_i) + var_i

            effect_df = predicted_probability_contrast_with_ci(
                result=result,
                X=X,
                variable=variable,
                contrast=contrast,
                outcome_labels=OUTCOME_LABELS,
                n_sim=N_SIM,
                ci_level=CI_LEVEL,
                random_seed=seed,
            )

            effect_df.insert(0, "model", spec_name)
            all_effects.append(effect_df)

        # Optional statsmodels marginal effects
        try:
            mfx_df = statsmodels_marginal_effects(result, spec_name)
            all_mfx.append(mfx_df)
        except Exception as e:
            print(f"Could not compute statsmodels get_margeff for {spec_name}: {e}")

    # Save selected contrasts with confidence intervals
    effects_all = pd.concat(all_effects, ignore_index=True)

    effects_all.to_csv(
        POSTEST_DIR / "selected_predicted_probability_contrasts_with_ci.csv",
        index=False,
    )

    # Save observation-level predicted probabilities
    probs_all = pd.concat(all_probs, ignore_index=True)

    probs_all.to_csv(
        POSTEST_DIR / "predicted_probabilities_long.csv",
        index=False,
    )

    # Save optional statsmodels marginal effects
    if all_mfx:
        mfx_all = pd.concat(all_mfx, ignore_index=True)
        mfx_all.to_csv(
            POSTEST_DIR / "statsmodels_average_marginal_effects.csv",
            index=False,
        )

    # Plot selected contrasts with CIs
    plot_margins(
        effects_df=effects_all,
        output_path=PLOT_DIR / "selected_predicted_probability_contrasts_with_ci.png",
        title="Selected predicted probability contrasts with 95% CIs",
    )

    print("Done.")
    print(f"Saved outputs to: {POSTEST_DIR}")


if __name__ == "__main__":
    main()