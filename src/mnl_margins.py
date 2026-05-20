"""
Post-estimation predicted probability contrasts for multinomial logit models.

This script:
1. Loads fitted statsmodels MNLogit result objects.
2. Loads the matching model matrix for each spec.
3. Computes average predicted-probability changes for selected variables.
4. Optionally generates a margins plot.

Interpretation alternatives:
- Continuous variables: mean -> mean + 1 SD, p25 -> p75, or p10 -> p90.
- Binary variables: 0 -> 1.
- Estimates are reported in probability units and percentage points.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR "fitted_models"
MATRIX_DIR = OUTPUT_DIR / "outputs/model_matrices"
POSTEST_DIR = OUTPUT_DIR / "postestimation"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

OUTCOME_LABELS = {
    0: "Mixed Outcome",
    1: "Contracts Expression",
    2: "Expands Expression",
}


# Edit this block as needed.
# Keys are model/spec names. Values are variable: contrast pairs.
SELECTED_EFFECTS = {
    "spec_extended_4": {
        "weighted_de_jure_expression": "sd",
        "judicial_action_index": "sd",
    },
    "spec_extended_4_high_court_1": {
        "weighted_de_jure_expression": "sd",
        "judicial_action_index": "sd",
    },
    "spec_extended_4_high_court_0": {
        "weighted_de_jure_expression": "sd",
        "judicial_action_index": "sd",
    },
}


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
    Return predicted probabilities as a DataFrame.

    Parameters
    ----------
    result:
        Fitted statsmodels MNLogit result.
    X:
        Design matrix with the same columns used in estimation.
    outcome_labels:
        Optional mapping from outcome codes to readable labels.

    Returns
    -------
    pd.DataFrame
        Predicted probabilities, one column per outcome.
    """

    probs = pd.DataFrame(result.predict(X), index=X.index)

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


def predicted_probability_contrast(
    result,
    X: pd.DataFrame,
    variable: str,
    contrast: str = "sd",
    outcome_labels=None,
) -> pd.DataFrame:
    """
    Estimate average predicted-probability change for one variable.

    The function changes the focal variable from a low value to a high value,
    predicts probabilities under both scenarios, and averages the difference
    over the observed sample.

    Returns
    -------
    pd.DataFrame
        One row per outcome.
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


def predicted_probabilities_long(
    result,
    X: pd.DataFrame,
    outcome_labels=None,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Return observation-level predicted probabilities in long format.

    Useful for diagnostics, descriptive summaries, or plotting predicted
    probabilities directly.
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

    # statsmodels supports get_margeff for discrete models; at="overall"
    # averages marginal effects over observations.
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
    Generate a margins plot from the selected predicted-probability contrasts.

    Assumes effects_df contains:
    - model
    - variable
    - outcome
    - estimate_pp
    """

    plot_df = effects_df.copy()

    plot_df["label"] = (
        plot_df["model"].astype(str)
        + "\n"
        + plot_df["variable"].astype(str)
    )

    outcomes = list(plot_df["outcome"].dropna().unique())

    y_labels = list(plot_df["label"].drop_duplicates())
    y_pos = np.arange(len(y_labels))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(y_labels))))

    # Offset points by outcome so they do not overlap
    if len(outcomes) == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.18, 0.18, len(outcomes))

    for outcome, offset in zip(outcomes, offsets):
        sub = plot_df[plot_df["outcome"] == outcome].copy()

        sub["y"] = sub["label"].map({label: i for i, label in enumerate(y_labels)})

        ax.scatter(
            sub["estimate_pp"],
            sub["y"] + offset,
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

    for spec_name, variable_map in SELECTED_EFFECTS.items():
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

        # Selected finite-difference predicted-probability contrasts
        for variable, contrast in variable_map.items():
            print(f"  - {variable}: {contrast}")

            effect_df = predicted_probability_contrast(
                result=result,
                X=X,
                variable=variable,
                contrast=contrast,
                outcome_labels=OUTCOME_LABELS,
            )

            effect_df.insert(0, "model", spec_name)
            all_effects.append(effect_df)

        # Optional statsmodels marginal effects
        try:
            mfx_df = statsmodels_marginal_effects(result, spec_name)
            all_mfx.append(mfx_df)
        except Exception as e:
            print(f"Could not compute statsmodels get_margeff for {spec_name}: {e}")

    # Save selected contrasts
    effects_all = pd.concat(all_effects, ignore_index=True)

    effects_all.to_csv(
        POSTEST_DIR / "selected_predicted_probability_contrasts.csv",
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

    # Plot selected contrasts
    plot_margins(
        effects_df=effects_all,
        output_path=PLOT_DIR / "selected_predicted_probability_contrasts.png",
        title="Selected predicted probability contrasts",
    )

    print("Done.")
    print(f"Saved outputs to: {POSTEST_DIR}")


if __name__ == "__main__":
    main()