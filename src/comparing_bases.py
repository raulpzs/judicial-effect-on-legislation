from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "outputs" 
OUTPUT_DIR = BASE_DIR / "outputs" / "judicial_models" / "base_category_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "mixed": INPUT_DIR / "param_table_all_specs_mixed.csv",
    "contract": INPUT_DIR / "param_table_all_specs_contract.csv",
    "expand": INPUT_DIR / "param_table_all_specs_expand.csv",
}

BASE_LABELS = {
    "mixed": "Mixed Outcome",
    "contract": "Contracts Expression",
    "expand": "Expands Expression",
}

OUTCOME_SHORT = {
    "Mixed Outcome": "mixed",
    "Contracts Expression": "contract",
    "Expands Expression": "expand",
}

KEY_CONTRASTS = [
    "contract_vs_mixed",
    "expand_vs_mixed",
    "contract_vs_expand",
]

# Optional: narrow this list if the full output is too much.
FOCAL_VARIABLES = [
    "wdj_expression_lag1",
    "v2jureform_lag1",
    "v2jupoatck_lag1",
    "v2jupack_lag1",
    "v2x_polyarchy_lag1",
    "v2jureview_lag1",
    "legal_common",
    "legal_civil",
    "high_court",
    "j_ind_lag1",
    "wdj_citizen_lag1",
    "wdj_press_lag1",
    "wdj_intermediaries_lag1",
    "wdj_govprot_lag1",
]

# Optional: narrow this list to your main specs.
FOCAL_SPECS = None
# Example:
# FOCAL_SPECS = [
#     "spec_extended_1_spline",
#     "spec_extended_2_spline",
#     "spec_extended_3_spline",
# ]


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def stars_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def sign_stars(coef, p):
    stars = stars_from_p(p)
    if stars == "":
        return "0"
    return ("+" if coef > 0 else "-") + stars


def load_all_param_tables():
    frames = []

    for base_key, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df["source_base"] = base_key
        df["source_base_label"] = BASE_LABELS[base_key]

        required = ["variable", "coef", "outcome", "std_err", "p_value", "spec"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")

        df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
        df["std_err"] = pd.to_numeric(df["std_err"], errors="coerce")
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")

        frames.append(df)

    all_params = pd.concat(frames, ignore_index=True)

    all_params["outcome_short"] = all_params["outcome"].map(OUTCOME_SHORT)
    all_params["base_short"] = all_params["source_base_label"].map(OUTCOME_SHORT)

    # Remove intercept unless you explicitly want it.
    all_params = all_params[all_params["variable"] != "Intercept"].copy()

    if FOCAL_VARIABLES is not None:
        all_params = all_params[all_params["variable"].isin(FOCAL_VARIABLES)].copy()

    if FOCAL_SPECS is not None:
        all_params = all_params[all_params["spec"].isin(FOCAL_SPECS)].copy()

    return all_params


def build_pairwise_contrasts(params):
    """
    Each row from a multinomial model is outcome vs base.
    This creates both:
      outcome_vs_base = coef
      base_vs_outcome = -coef

    That makes all base-category versions directly comparable.
    """
    rows = []

    for _, r in params.iterrows():
        left = r["outcome_short"]
        right = r["base_short"]

        if pd.isna(left) or pd.isna(right):
            continue

        if left == right:
            continue

        common = {
            "source_base": r["source_base"],
            "spec": r["spec"],
            "variable": r["variable"],
            "std_err": r["std_err"],
            "p_value": r["p_value"],
        }

        rows.append({
            **common,
            "contrast": f"{left}_vs_{right}",
            "coef": r["coef"],
            "direction": "raw",
        })

        rows.append({
            **common,
            "contrast": f"{right}_vs_{left}",
            "coef": -r["coef"],
            "direction": "reversed",
        })

    pairwise = pd.DataFrame(rows)
    pairwise["stars"] = pairwise["p_value"].apply(stars_from_p)
    pairwise["sign_stars"] = pairwise.apply(
        lambda x: sign_stars(x["coef"], x["p_value"]),
        axis=1
    )

    return pairwise


def equivalence_check(pairwise):
    """
    Checks whether equivalent contrasts match across base-category reruns.
    For example:
      contract_vs_mixed from mixed-base model
      contract_vs_mixed from reversed contract-base model

    These should be numerically very close.
    """
    records = []

    for (spec, variable, contrast), g in pairwise.groupby(["spec", "variable", "contrast"]):
        if len(g) < 2:
            continue

        records.append({
            "spec": spec,
            "variable": variable,
            "contrast": contrast,
            "n_versions": len(g),
            "sources": ", ".join(sorted(g["source_base"].unique())),
            "coef_min": g["coef"].min(),
            "coef_max": g["coef"].max(),
            "coef_mean": g["coef"].mean(),
            "max_abs_diff": g["coef"].max() - g["coef"].min(),
            "sign_patterns": " / ".join(sorted(g["sign_stars"].unique())),
        })

    out = pd.DataFrame(records)

    if not out.empty:
        out = out.sort_values("max_abs_diff", ascending=False)

    return out


def make_significance_summary(pairwise):
    """
    Creates a wide sign/significance table for the three core contrasts.
    If multiple base-category files give the same contrast, it collapses them.
    If signs/significance disagree, it shows the disagreement.
    """
    temp = pairwise[pairwise["contrast"].isin(KEY_CONTRASTS)].copy()

    def collapse(x):
        vals = sorted(set(x.dropna().astype(str)))
        if len(vals) == 1:
            return vals[0]
        return " / ".join(vals)

    wide = (
        temp
        .groupby(["spec", "variable", "contrast"])["sign_stars"]
        .apply(collapse)
        .reset_index()
        .pivot_table(
            index=["spec", "variable"],
            columns="contrast",
            values="sign_stars",
            aggfunc="first"
        )
        .reset_index()
    )

    for c in KEY_CONTRASTS:
        if c not in wide.columns:
            wide[c] = ""

    wide = wide[["spec", "variable"] + KEY_CONTRASTS]
    wide = wide.sort_values(["spec", "variable"])

    return wide


def make_coefficient_summary(pairwise):
    """
    Creates a coefficient table for the same contrasts.
    Averages equivalent coefficients across base-category reruns.
    If the reruns are consistent, min/max differences should be tiny.
    """
    temp = pairwise[pairwise["contrast"].isin(KEY_CONTRASTS)].copy()

    out = (
        temp
        .groupby(["spec", "variable", "contrast"])
        .agg(
            coef_mean=("coef", "mean"),
            coef_min=("coef", "min"),
            coef_max=("coef", "max"),
            std_err_mean=("std_err", "mean"),
            p_value_mean=("p_value", "mean"),
            n_versions=("coef", "size"),
        )
        .reset_index()
    )

    out["max_abs_diff"] = out["coef_max"] - out["coef_min"]
    out["stars"] = out["p_value_mean"].apply(stars_from_p)
    out["coef_se"] = out.apply(
        lambda x: f"{x['coef_mean']:.3f}{x['stars']} ({x['std_err_mean']:.3f})",
        axis=1
    )

    return out.sort_values(["spec", "variable", "contrast"])


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    params = load_all_param_tables()

    print(f"Loaded parameter rows: {len(params):,}")
    print("Source bases found:")
    print(params["source_base"].value_counts())

    pairwise = build_pairwise_contrasts(params)

    eq = equivalence_check(pairwise)
    signs = make_significance_summary(pairwise)
    coefs = make_coefficient_summary(pairwise)

    params.to_csv(OUTPUT_DIR / "combined_param_tables.csv", index=False)
    pairwise.to_csv(OUTPUT_DIR / "pairwise_contrasts_all_bases.csv", index=False)
    eq.to_csv(OUTPUT_DIR / "base_category_equivalence_check.csv", index=False)
    signs.to_csv(OUTPUT_DIR / "robustness_significance_summary.csv", index=False)
    coefs.to_csv(OUTPUT_DIR / "robustness_coefficient_summary.csv", index=False)

    print("\nLargest equivalence differences:")
    if eq.empty:
        print("No duplicated equivalent contrasts found. Check that all three base files loaded correctly.")
    else:
        print(
            eq[
                ["spec", "variable", "contrast", "max_abs_diff", "sources", "sign_patterns"]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\nWrote outputs to:")
    print(OUTPUT_DIR)

    print("\nMost useful files:")
    print(" - base_category_equivalence_check.csv")
    print(" - robustness_significance_summary.csv")
    print(" - robustness_coefficient_summary.csv")


if __name__ == "__main__":
    main()