import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from patsy import dmatrix

from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# Directories
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PRED_PATH = OUTPUT_DIR / "predictions_all_specs.csv"
OUTPUT_MODEL_PATH = OUTPUT_DIR / "model_comparison.csv"
OUTPUT_PARAM_PATH = OUTPUT_DIR / "param_table_all_specs.csv"
OUTPUT_LATEX_PATH = OUTPUT_DIR / "mnl_results.tex"
OUTPUT_HTML_PATH = OUTPUT_DIR / "mnl_results.html"

OUTPUT_VIF_PATH = OUTPUT_DIR / "vif_table_all_specs.csv"
OUTPUT_VIF_LATEX_PATH = OUTPUT_DIR / "vif_table_all_specs.tex"
OUTPUT_VIF_HTML_PATH = OUTPUT_DIR / "vif_table_all_specs.html"

# =========================
# Config
# =========================

DATA_PATH = DATA_DIR / "processed" / "cases_v4_short_merged.csv"

Y_VAR = "decision_direction"

CLUSTER_VAR = "country"

BASE_CATEGORY = "Mixed Outcome"

# =========================
# Model specifications
# =========================

CONTROLS = [
    "v2x_polyarchy_lag1",
    "v2jureview_lag1",
    "legal_common",
    "legal_civil",
    "high_court",
    "j_ind_lag1",
]

SPECIFICATIONS = {
    "spec_base_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"],
        "spline": False,
        "interaction": False,
    },
    "spec_base_2": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"],
        "spline": False,
        "interaction": False,
    },
    "spec_base_3": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"],
        "spline": False,
        "interaction": False,
    },
    "spec_extended_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_1_spline": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": True,
        "interaction": False,
    },
    "spec_extended_1_spline_interact": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": True,
        "interaction": True,
    },
    "spec_extended_2": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_2_spline": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"] + CONTROLS,
        "spline": True,
        "interaction": False,
    },
    "spec_extended_2_spline_interact": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"] + CONTROLS,
        "spline": True,
        "interaction": True,
    },
    "spec_extended_3": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_3_spline": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": True,
        "interaction": False,
    },
    "spec_extended_3_spline_interact": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": True,
        "interaction": True,
    },
    "spec_extended_4": {
        "vars": ["wdj_citizen_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_5": {
        "vars": ["wdj_intermediaries_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_6": {
        "vars": ["wdj_press_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
    "spec_extended_7": {
        "vars": ["wdj_govprot_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False,
    },
}

# =========================
# Loading data
# =========================

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[Y_VAR])

# =========================
# Feature engineering
# =========================

df["legal_common"] = (df["legal_system"] == "Common").astype(int)
df["legal_civil"] = (df["legal_system"] == "Civil").astype(int)

# =========================
# Encoding Y
# =========================

y_map = {
    "Contracts Expression": 0,
    "Expands Expression": 1,
    "Mixed Outcome": 2,
}

y_reverse_map = {v: k for k, v in y_map.items()}

# =========================
# Storage
# =========================

all_preds = []
model_summary_rows = []
all_params = []
all_vifs = []

# =========================
# Helper: stars
# =========================

def stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""

# =========================
# Matrix builder
# =========================

def build_X(data, spec):
    """
    Creates design matrix with optional spline + interaction.
    Patsy includes an intercept by default, so do NOT add another constant.
    """

    rhs_terms = []

    if spec["spline"]:
        if spec["interaction"]:
            rhs_terms.append("bs(year, df=4) * v2x_polyarchy_lag1")
        else:
            rhs_terms.append("bs(year, df=4)")

    for v in spec["vars"]:
        if v == "v2x_polyarchy_lag1" and spec["spline"] and spec["interaction"]:
            continue
        rhs_terms.append(v)

    formula = " + ".join(rhs_terms)

    X = dmatrix(formula, data, return_type="dataframe")

    return X

# =========================
# VIF helper
# =========================

def compute_vif(X):
    vif = pd.DataFrame()
    vif["variable"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif

# =========================
# Model loop
# =========================

for spec_name, spec in SPECIFICATIONS.items():

    print(f"\nRunning {spec_name}")

    required_vars = [Y_VAR] + spec["vars"] + [CLUSTER_VAR]

    if spec["spline"]:
        required_vars += ["year"]

    if spec["interaction"]:
        required_vars += ["v2x_polyarchy_lag1"]

    data = df.dropna(subset=required_vars).copy()

    # outcome
    y = data[Y_VAR].map(y_map)

    if y.isna().any():
        raise ValueError("Unmapped categories found in Y")

    # design matrix
    X = build_X(data, spec)

    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        print(f"\nRank deficient X in {spec_name}")
        print(f"Rank: {rank}, Columns: {X.shape[1]}")
        print(X.columns.tolist())
        continue

    # =========================
    # VIF table
    # =========================

    vif_table = compute_vif(X)
    vif_table["spec"] = spec_name
    vif_table = vif_table[["spec", "variable", "VIF"]]

    all_vifs.append(vif_table)

    high_vif = vif_table[vif_table["VIF"] > 10]

    if len(high_vif) > 0:
        print(f"\nHigh multicollinearity in {spec_name}")
        print(high_vif)

    # model
    model = sm.MNLogit(y, X)

    result = model.fit(
        method="newton",
        maxiter=200,
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data[CLUSTER_VAR]},
    )

    # =========================
    # Model-level stats
    # =========================

    model_summary_rows.append({
        "spec": spec_name,
        "n_obs": int(result.nobs),
        "log_likelihood": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "pseudo_r2": 1 - result.llf / result.llnull,
        "base_category": BASE_CATEGORY,
    })

    # =========================
    # Parameter table
    # =========================

    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    param_table = (
        params.stack()
        .reset_index()
        .rename(columns={
            "level_0": "variable",
            "level_1": "class_id",
            0: "coef",
        })
    )

    param_table["class_id"] = param_table["class_id"].astype(int)

    mnl_class_map = {
        0: "Contracts Expression",
        1: "Expands Expression",
    }

    param_table["class_label"] = param_table["class_id"].map(mnl_class_map)
    param_table["outcome"] = param_table["class_id"].map(mnl_class_map)

    param_table = param_table.drop(columns=["class_id"])

    param_table["std_err"] = bse.stack().values
    param_table["p_value"] = pvalues.stack().values

    param_table["spec"] = spec_name
    param_table["base_category"] = BASE_CATEGORY

    param_table["stars"] = param_table["p_value"].apply(stars)

    param_table["coef_se"] = (
        param_table["coef"].round(3).astype(str)
        + param_table["stars"]
        + " ("
        + param_table["std_err"].round(3).astype(str)
        + ")"
    )

    all_params.append(param_table)

    # =========================
    # Prediction
    # =========================

    preds = result.predict(X)
    pred_class = np.argmax(preds.values, axis=1)

    out = data.copy()
    out["spec"] = spec_name

    out["pred_class_id"] = pred_class
    out["pred_class_label"] = pd.Series(pred_class).map(y_reverse_map).values

    for i in range(preds.shape[1]):
        out[f"prob_{y_reverse_map[i]}"] = preds.iloc[:, i]

    all_preds.append(out)

# =========================
# Combine outputs
# =========================

pred_df = pd.concat(all_preds, ignore_index=True)
model_df = pd.DataFrame(model_summary_rows)
param_df = pd.concat(all_params, ignore_index=True)

vif_long_df = pd.concat(all_vifs, ignore_index=True)

vif_df = (
    vif_long_df
    .pivot_table(
        index="spec",
        columns="variable",
        values="VIF",
        aggfunc="first",
    )
    .reset_index()
)

vif_df.columns.name = None
vif_df = vif_df.round(3)

# =========================
# Save outputs
# =========================

pred_df.to_csv(OUTPUT_PRED_PATH, index=False)
model_df.to_csv(OUTPUT_MODEL_PATH, index=False)
param_df.to_csv(OUTPUT_PARAM_PATH, index=False)

vif_df.to_csv(OUTPUT_VIF_PATH, index=False)

# =========================
# Publication coefficient table
# =========================

table = param_df.pivot_table(
    index="variable",
    columns=["outcome", "spec"],
    values="coef_se",
    aggfunc="first",
)

latex_str = table.to_latex(
    multicolumn=True,
    multicolumn_format="c",
    escape=False,
)

with open(OUTPUT_LATEX_PATH, "w") as f:
    f.write(latex_str)

html_str = table.to_html()

with open(OUTPUT_HTML_PATH, "w") as f:
    f.write(html_str)

# =========================
# VIF publication table
# =========================

vif_latex_str = vif_df.to_latex(
    index=False,
    float_format="%.3f",
    escape=False,
)

with open(OUTPUT_VIF_LATEX_PATH, "w") as f:
    f.write(vif_latex_str)

vif_html_str = vif_df.to_html(
    index=False,
    float_format="{:.3f}".format,
    escape=False,
)

with open(OUTPUT_VIF_HTML_PATH, "w") as f:
    f.write(vif_html_str)

# =========================
# Done
# =========================

print("\nSaved:")
print(f"- {OUTPUT_PRED_PATH.name}")
print(f"- {OUTPUT_MODEL_PATH.name}")
print(f"- {OUTPUT_PARAM_PATH.name}")
print(f"- {OUTPUT_LATEX_PATH.name}")
print(f"- {OUTPUT_HTML_PATH.name}")
print(f"- {OUTPUT_VIF_PATH.name}")
print(f"- {OUTPUT_VIF_LATEX_PATH.name}")
print(f"- {OUTPUT_VIF_HTML_PATH.name}")

# =========================
# Separate split-sample model: spec_extended_4 by high_court
# =========================

split_params = []
split_model_rows = []

split_spec = SPECIFICATIONS["spec_extended_4"].copy()
split_spec["vars"] = [v for v in split_spec["vars"] if v != "high_court"]

for high_court_value, sample_label in [
    (1, "high_court_1"),
    (0, "high_court_0"),
]:

    print(f"\nRunning spec_extended_4 split sample: {sample_label}")

    data_split = df[df["high_court"] == high_court_value].copy()

    required_vars = [Y_VAR] + split_spec["vars"] + [CLUSTER_VAR]
    data_split = data_split.dropna(subset=required_vars).copy()

    y = data_split[Y_VAR].map(y_map)

    if y.isna().any():
        raise ValueError("Unmapped categories found in Y")

    X = build_X(data_split, split_spec)

    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        print(f"\nRank deficient X in spec_extended_4_{sample_label}")
        print(f"Rank: {rank}, Columns: {X.shape[1]}")
        print(X.columns.tolist())
        continue

    model = sm.MNLogit(y, X)

    result = model.fit(
        method="newton",
        maxiter=200,
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data_split[CLUSTER_VAR]},
    )

    split_model_rows.append({
        "spec": "spec_extended_4",
        "sample": sample_label,
        "high_court_value": high_court_value,
        "n_obs": int(result.nobs),
        "log_likelihood": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "pseudo_r2": 1 - result.llf / result.llnull,
        "base_category": BASE_CATEGORY,
    })

    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    param_table = (
        params.stack()
        .reset_index()
        .rename(columns={
            "level_0": "variable",
            "level_1": "class_id",
            0: "coef",
        })
    )

    param_table["class_id"] = param_table["class_id"].astype(int)

    mnl_class_map = {
        0: "Contracts Expression",
        1: "Expands Expression",
    }

    param_table["class_label"] = param_table["class_id"].map(mnl_class_map)
    param_table["outcome"] = param_table["class_id"].map(mnl_class_map)
    param_table = param_table.drop(columns=["class_id"])

    param_table["std_err"] = bse.stack().values
    param_table["p_value"] = pvalues.stack().values

    param_table["spec"] = "spec_extended_4"
    param_table["sample"] = sample_label
    param_table["high_court_value"] = high_court_value
    param_table["base_category"] = BASE_CATEGORY
    param_table["stars"] = param_table["p_value"].apply(stars)

    param_table["coef_se"] = (
        param_table["coef"].round(3).astype(str)
        + param_table["stars"]
        + " ("
        + param_table["std_err"].round(3).astype(str)
        + ")"
    )

    split_params.append(param_table)

# save split-sample outputs
if split_params:
    split_param_df = pd.concat(split_params, ignore_index=True)
    split_model_df = pd.DataFrame(split_model_rows)

    split_param_df.to_csv(
        OUTPUT_DIR / "param_table_spec_extended_4_high_court_split.csv",
        index=False,
    )

    split_model_df.to_csv(
        OUTPUT_DIR / "model_comparison_spec_extended_4_high_court_split.csv",
        index=False,
    )

    print("\nSaved split-sample outputs:")
    print("- param_table_spec_extended_4_high_court_split.csv")
    print("- model_comparison_spec_extended_4_high_court_split.csv")