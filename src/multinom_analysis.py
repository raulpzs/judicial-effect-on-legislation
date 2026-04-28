# this is the script for running the multinomial logit models of de jure+executive factors affecting judicial decisions

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

# =========================
# Config
# =========================

DATA_PATH = DATA_DIR / "processed" / "cases_v4_short_merged.csv"

Y_VAR = "decision_direction"

CLUSTER_VAR = "country"  # IMPORTANT: must be set

BASE_CATEGORY = "Mixed Outcome"

# =========================
# Model specifications
# =========================

CONTROLS = [
    "v2x_polyarchy_lag1", 
#    "v2juhcind_lag_1", 
#    "v2juncind_lag_1", 
    "v2jureview_lag1",
    "legal_common",
    "legal_civil", # court-specific independence
    "high_court",
    "j_ind_lag1",
]
# need to add high and low court dummies + court-specific independence 

SPECIFICATIONS = {
    "spec_base_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"],
        "spline": False,
        "interaction": False
    },
    "spec_base_2": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"],
        "spline": False,
        "interaction": False
    },
    "spec_base_3": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"],
        "spline": False,
        "interaction": False
    },

    "spec_extended_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False
    },
    "spec_extended_2": {
        "vars": ["wdj_expression_lag1", "v2jupoatck_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False
    },
    "spec_extended_3": {
        "vars": ["wdj_expression_lag1", "v2jupack_lag1"] + CONTROLS,
        "spline": False,
        "interaction": False
    },

    # spline specs (global time trend)
    "spec_spline_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": True,
        "interaction": False
    },

    # spline × democracy interaction version
    "spec_spline_interact_1": {
        "vars": ["wdj_expression_lag1", "v2jureform_lag1"] + CONTROLS,
        "spline": True,
        "interaction": True
    }
}

# =========================
# Loading data
# =========================

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[Y_VAR])

# =========================
# Feature engineering
# =========================

# legal system
df["legal_common"] = (df["legal_system"] == "Common").astype(int)
df["legal_civil"] = (df["legal_system"] == "Civil").astype(int)

# =========================
# Encoding Y
# =========================

y_map = {
    "Contracts Expression": 0,
    "Expands Expression": 1,
    "Mixed Outcome": 2
}

y_reverse_map = {v: k for k, v in y_map.items()}

# =========================
# Storage
# =========================

all_preds = []
model_summary_rows = []
all_params = []

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
    
    vif_table = compute_vif(X)

    high_vif = vif_table[vif_table["VIF"] > 10]
    
    if len(high_vif) > 0:
        print(f"\nHigh multicollinearity in {spec_name}")
        print(high_vif)

    # model
    model = sm.MNLogit(y, X)

    # cluster robust SE
    result = model.fit(
        method="newton",
        maxiter=200,
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data[CLUSTER_VAR]}
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
        "base_category": BASE_CATEGORY
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
            0: "coef"
        })
    )
    
    param_table["class_id"] = param_table["class_id"].astype(int)
    
    # MNLogit estimates equations for non-base outcome categories.
    # Here class_id 0 and 1 correspond to the non-base alternatives.
    # Adjust this if your actual output order differs.
    mnl_class_map = {
        0: "Contracts Expression",
        1: "Expands Expression"
    }
    
    param_table["class_label"] = param_table["class_id"].map(mnl_class_map)

    param_table["outcome"] = param_table["class_id"].map(mnl_class_map)

    # dropping this since we'll use the labels instead
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

# =========================
# Save outputs
# =========================

pred_df.to_csv(OUTPUT_PRED_PATH, index=False)
model_df.to_csv(OUTPUT_MODEL_PATH, index=False)
param_df.to_csv(OUTPUT_PARAM_PATH, index=False)

# =========================
# Publication table
# =========================

table = param_df.pivot_table(
    index="variable",
    columns=["outcome", "spec"],
    values="coef_se",
    aggfunc="first"
)

latex_str = table.to_latex(
    multicolumn=True,
    multicolumn_format="c",
    escape=False
)

with open(OUTPUT_LATEX_PATH, "w") as f:
    f.write(latex_str)

html_str = table.to_html()

with open(OUTPUT_HTML_PATH, "w") as f:
    f.write(html_str)

# =========================
# Done
# =========================

print("\nSaved:")
print(f"- {OUTPUT_PRED_PATH.name}")
print(f"- {OUTPUT_MODEL_PATH.name}")
print(f"- {OUTPUT_PARAM_PATH.name}")
print(f"- {OUTPUT_LATEX_PATH.name}")
print(f"- {OUTPUT_HTML_PATH.name}")