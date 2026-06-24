import pandas as pd
import numpy as np
from pathlib import Path


VERSION = "main"
# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" 

PARAM_PATH = OUTPUT_DIR / "param_table_all_specs_mixed.csv"
MODEL_PATH = OUTPUT_DIR / "model_comparison_mixed.csv"

OUT_LATEX_PATH = OUTPUT_DIR / "tables" / f"mnl_main_models_table_{VERSION}.tex"

# ============================================================
# User settings
# ============================================================

CAPTION = (
    "Multinomial Logit Models Predicting Judicial Decisions that Contract or Expand Freedom of Expression"
)

LABEL = f"tab:mnl_expression_{VERSION}"

OUTCOMES = [
    "Contracts Expression",
    "Expands Expression",
]

BASE_CATEGORY = "Mixed Outcome"

# Each tuple is:
# (spec_name, model_group_label, model_number, spline_included, spline_polyarchy_interaction)

MODEL_SPECS = [
    ("spec_base_1", "Base", "(1)", "No", "No"),
    ("spec_base_2", "Base", "(2)", "No", "No"),
    ("spec_base_3", "Base", "(3)", "No", "No"),

    ("spec_extended_1", "Extended", "(4)", "No", "No"),
    ("spec_extended_2", "Extended", "(5)", "No", "No"),
    ("spec_extended_3", "Extended", "(6)", "No", "No"),

    ("spec_extended_1_spline", "Spline", "(7)", "Yes", "No"),
    ("spec_extended_1_spline_interact", "Spline", "(8)", "Yes", "Yes"),
]

# Each tuple is:
# (variable_name_in_param_table, display_label, vertical_space_after_block)

VARIABLES = [
    ("wdj_expression_lag1", "De Jure Expression, lag", "0.5cm"),

    ("v2jureform_lag1", "Judicial Reform, lag", "0.1cm"),
    ("v2jupoatck_lag1", "Judicial Political Attacks, lag", "0.1cm"),
    ("v2jupack_lag1", "Judicial Packing, lag", "0.5cm"),

    ("high_court", "High Court", "0.1cm"),
    ("j_ind_lag1", "Judicial Independence, lag", "0.5cm"),

    ("v2x_polyarchy_lag1", "Polyarchy, lag", "0.1cm"),

    ("legal_civil", "Legal System: Civil", "0.1cm"),
    ("legal_common", "Legal System: Common", "0.5cm"),
]

NOTES = (
    "Table reports multinomial logistic regression coefficients predicting whether "
    "judicial decisions contract or expand freedom of expression relative to the "
    "omitted baseline category. Robust standard errors are reported in parentheses. "
    "All substantive predictors are lagged one year. Extended models include "
    "institutional and regime-level controls. Spline specifications model non-linear "
    "temporal dynamics using cubic B-splines over year; interaction specifications "
    "additionally interact spline terms with lagged polyarchy scores. Legal system "
    "indicators are relative to the omitted reference category of mixed or other "
    "legal systems. "
    "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

# ============================================================
# Helpers
# ============================================================

def latex_escape(value):
    """Escape LaTeX special characters in labels."""
    if pd.isna(value):
        return ""
    value = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


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


def fmt_coef(coef, stars):
    if pd.isna(coef):
        return ""
    if pd.isna(stars):
        stars = ""
    return f"{coef:.3f}{stars}"


def fmt_se(se):
    if pd.isna(se):
        return ""
    return f"({se:.3f})"


def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(x):,}"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_r2(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


def make_cmidrule(start, end):
    return rf"\cmidrule(lr){{{start}-{end}}}"


# ============================================================
# Load data
# ============================================================

param_df = pd.read_csv(PARAM_PATH)
model_df = pd.read_csv(MODEL_PATH)

spec_order = [x[0] for x in MODEL_SPECS]
n_models = len(spec_order)
n_outcomes = len(OUTCOMES)
n_model_cols = n_models * n_outcomes

if "stars" not in param_df.columns:
    param_df["stars"] = param_df["p_value"].apply(stars_from_p)

# Keep requested specs and outcomes only
param_df = param_df[
    param_df["spec"].isin(spec_order)
    & param_df["outcome"].isin(OUTCOMES)
].copy()

model_df = model_df[model_df["spec"].isin(spec_order)].copy()

# Make lookup dictionaries
param_lookup = {}
for _, row in param_df.iterrows():
    key = (row["variable"], row["outcome"], row["spec"])
    param_lookup[key] = {
        "coef": row.get("coef", np.nan),
        "std_err": row.get("std_err", np.nan),
        "stars": row.get("stars", ""),
    }

model_lookup = model_df.set_index("spec").to_dict(orient="index")

# ============================================================
# Header construction
# ============================================================

col_spec = "l" + "c" * n_model_cols

lines = []

lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append("")
lines.append(rf"\caption{{{latex_escape(CAPTION)}}}")
lines.append(rf"\label{{{LABEL}}}")
lines.append("")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{2pt}")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
lines.append("")

# Outcome header row
outcome_header = [""]

col_start = 2
outcome_cmidrules = []

for outcome in OUTCOMES:
    col_end = col_start + n_models - 1
    outcome_header.append(
        rf"\multicolumn{{{n_models}}}{{c}}{{\textbf{{{latex_escape(outcome)}}}}}"
    )
    outcome_cmidrules.append(make_cmidrule(col_start, col_end))
    col_start = col_end + 1

lines.append(" & ".join(outcome_header) + r" \\")
lines.extend(outcome_cmidrules)
lines.append("")

# Group header row
group_header = [""]
group_cmidrules = []

for outcome_i, outcome in enumerate(OUTCOMES):
    start_offset = 2 + outcome_i * n_models

    current_group = None
    group_start = None

    for j, (_, group_label, _, _, _) in enumerate(MODEL_SPECS):
        if group_label != current_group:
            if current_group is not None:
                group_end = start_offset + j - 1
                group_header.append(
                    rf"\multicolumn{{{group_end - group_start + 1}}}{{c}}{{{latex_escape(current_group)}}}"
                )
                group_cmidrules.append(make_cmidrule(group_start, group_end))

            current_group = group_label
            group_start = start_offset + j

    group_end = start_offset + n_models - 1
    group_header.append(
        rf"\multicolumn{{{group_end - group_start + 1}}}{{c}}{{{latex_escape(current_group)}}}"
    )
    group_cmidrules.append(make_cmidrule(group_start, group_end))

lines.append(" & ".join(group_header) + r" \\")
lines.extend(group_cmidrules)
lines.append("")

# Model number row
model_number_row = [""]
for _outcome in OUTCOMES:
    model_number_row.extend([model_num for _, _, model_num, _, _ in MODEL_SPECS])

lines.append(" & ".join(model_number_row) + r" \\")
lines.append("")
lines.append(r"\midrule")
lines.append("")

# ============================================================
# Coefficient rows
# ============================================================

for variable, label, vspace in VARIABLES:
    coef_row = [latex_escape(label)]
    se_row = [""]

    for outcome in OUTCOMES:
        for spec in spec_order:
            entry = param_lookup.get((variable, outcome, spec), None)

            if entry is None:
                coef_row.append("")
                se_row.append("")
            else:
                coef_row.append(fmt_coef(entry["coef"], entry["stars"]))
                se_row.append(fmt_se(entry["std_err"]))

    lines.append(" & ".join(coef_row) + r" \\")
    lines.append(rf"\vspace{{{vspace}}}")
    lines.append(" & ".join(se_row) + r" \\")
    lines.append("")

# ============================================================
# Model feature rows
# ============================================================

def add_spec_feature_row(label, values_from_model_specs):
    row = [latex_escape(label)]
    for _outcome in OUTCOMES:
        row.extend(values_from_model_specs)
    lines.append(" & ".join(row) + r" \\")

spline_values = [spline for _, _, _, spline, _ in MODEL_SPECS]
interaction_values = [interaction for _, _, _, _, interaction in MODEL_SPECS]

add_spec_feature_row("Spline Terms Included", spline_values)
lines.append("")
add_spec_feature_row(r"Spline $\times$ Polyarchy Interaction", interaction_values)

lines.append(r"\addlinespace")
lines.append(r"\midrule")

# ============================================================
# Model statistics rows
# ============================================================

def add_model_stat_row(label, key, formatter):
    row = [latex_escape(label)]

    values = []
    for spec in spec_order:
        value = model_lookup.get(spec, {}).get(key, np.nan)
        values.append(formatter(value))

    for _outcome in OUTCOMES:
        row.extend(values)

    lines.append(" & ".join(row) + r" \\")

add_model_stat_row("Observations", "n_obs", fmt_int)
add_model_stat_row("Log Likelihood", "log_likelihood", lambda x: fmt_num(x, 2))
add_model_stat_row("AIC", "aic", lambda x: fmt_num(x, 2))
add_model_stat_row("BIC", "bic", lambda x: fmt_num(x, 2))
add_model_stat_row(r"Pseudo-$R^2$", "pseudo_r2", fmt_r2)

# ============================================================
# Footer
# ============================================================

lines.append("")
lines.append(r"\bottomrule")
lines.append("")
lines.append(r"\end{tabular}")
lines.append("")
lines.append(r"\vspace{0.4em}")
lines.append("")
lines.append(r"\begin{minipage}{0.97\linewidth}")
lines.append(r"\footnotesize")
lines.append(rf"\textit{{Notes:}} {NOTES}")
lines.append(r"\end{minipage}")
lines.append("")
lines.append(r"\end{table}")

latex = "\n".join(lines)

OUT_LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_LATEX_PATH.write_text(latex)

print(f"Saved LaTeX table to: {OUT_LATEX_PATH}")