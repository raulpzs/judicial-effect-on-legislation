import pandas as pd
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PARAM_PATH = OUTPUT_DIR / "param_table_all_specs_mixed.csv"

TABLE_DIR = OUTPUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_LATEX_PATH = TABLE_DIR / "focal_direction_across_models.tex"

# ============================================================
# User settings
# ============================================================

CAPTION = "Direction and Significance of Focal Judicial-Context Variables Across Model Families"
LABEL = "tab:focal-direction-across-models"

OUTCOMES = {
    "Contracts": "Contracts Expression",
    "Expands": "Expands Expression",
}

# Column groups: display label -> variable name in param table
FOCAL_VARIABLES = {
    "Judicial reform": "v2jureform_lag1",
    "Judicial attack": "v2jupoatck_lag1",
    "Judicial packing": "v2jupack_lag1",
}

# Panels and rows:
# Each row is (row label, specs for reform/attack/packing)
#
# You can set a spec to None if that variable/model does not exist.
PANELS = [
    (
        "Panel A: Main judicial-context models",
        [
            (
                "Extended spline model",
                {
                    "Judicial reform": "spec_extended_1_spline",
                    "Judicial attack": "spec_extended_2_spline",
                    "Judicial packing": "spec_extended_3_spline",
                },
            ),
        ],
    ),
    (
        "Panel B: Actor-specific models",
        [
            (
                "Citizen",
                {
                    "Judicial reform": "spec_c_reform",
                    "Judicial attack": "spec_c_attack",
                    "Judicial packing": "spec_c_pack",
                },
            ),
            (
                "Intermediary",
                {
                    "Judicial reform": "spec_im_reform",
                    "Judicial attack": "spec_im_attack",
                    "Judicial packing": "spec_im_pack",
                },
            ),
            (
                "Press",
                {
                    "Judicial reform": "spec_p_reform",
                    "Judicial attack": "spec_p_attack",
                    "Judicial packing": "spec_p_pack",
                },
            ),
            (
                "Gov.-protected actor",
                {
                    "Judicial reform": "spec_govprot_reform",
                    "Judicial attack": "spec_govprot_attack",
                    "Judicial packing": "spec_govprot_pack",
                },
            ),
        ],
    ),
    (
        "Panel C: Actor-specific decision-level models",
        [
            (
                "Citizen decision model",
                {
                    "Judicial reform": "spec_c_reform_decision",
                    "Judicial attack": "spec_c_attack_decision",
                    "Judicial packing": "spec_c_pack_decision",
                },
            ),
            (
                "Intermediary decision model",
                {
                    "Judicial reform": "spec_im_reform_decision",
                    "Judicial attack": "spec_im_attack_decision",
                    "Judicial packing": "spec_im_pack_decision",
                },
            ),
            (
                "Press decision model",
                {
                    "Judicial reform": "spec_p_reform_decision",
                    "Judicial attack": "spec_p_attack_decision",
                    "Judicial packing": "spec_p_pack_decision",
                },
            ),
            (
                "Gov.-protected actor decision model",
                {
                    "Judicial reform": "spec_govprot_reform_decision",
                    "Judicial attack": "spec_govprot_attack_decision",
                    "Judicial packing": "spec_govprot_pack_decision",
                },
            ),
        ],
    ),
]

NOTES = (
    "Entries summarize coefficient direction and statistical significance for focal "
    "judicial-context variables. A positive sign indicates a positive coefficient; "
    "a negative sign indicates a negative coefficient. 0 indicates that the coefficient "
    "was not statistically significant at the 10 percent level. "
    "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$."
)

# ============================================================
# Helpers
# ============================================================

def latex_escape(value):
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


def direction_sig(coef, p_value):
    """
    Return LaTeX-formatted direction/significance:
    $+$***, $-$**, $+$*, or 0.
    """
    if pd.isna(coef) or pd.isna(p_value):
        return ""

    stars = stars_from_p(p_value)

    if stars == "":
        return "0"

    sign = r"$+$" if coef > 0 else r"$-$"
    return f"{sign}{stars}"


def get_direction_cell(param_df, spec, variable, outcome):
    if spec is None:
        return ""

    sub = param_df[
        (param_df["spec"] == spec)
        & (param_df["variable"] == variable)
        & (param_df["outcome"] == outcome)
    ]

    if sub.empty:
        return ""

    row = sub.iloc[0]
    return direction_sig(row["coef"], row["p_value"])


# ============================================================
# Load data
# ============================================================

param_df = pd.read_csv(PARAM_PATH)

required_cols = {"spec", "variable", "outcome", "coef", "p_value"}
missing = required_cols - set(param_df.columns)

if missing:
    raise ValueError(f"param table is missing required columns: {missing}")

# ============================================================
# Build LaTeX
# ============================================================

n_cols = 1 + len(FOCAL_VARIABLES) * len(OUTCOMES)

lines = []

lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append("")
lines.append(rf"\caption{{{latex_escape(CAPTION)}}}")
lines.append(rf"\label{{{LABEL}}}")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{4pt}")
lines.append(r"\renewcommand{\arraystretch}{1.1}")
lines.append(r"\begin{tabular}{p{0.25\textwidth}cccccc}")
lines.append(r"\toprule")

# Header row 1
header_1 = [""]

for focal_label in FOCAL_VARIABLES.keys():
    header_1.append(rf"\multicolumn{{2}}{{c}}{{{latex_escape(focal_label)}}}")

lines.append(" & ".join(header_1) + r" \\")

# cmidrules
cmidrules = []
start = 2

for _ in FOCAL_VARIABLES:
    cmidrules.append(rf"\cmidrule(lr){{{start}-{start + 1}}}")
    start += 2

lines.append(" ".join(cmidrules))

# Header row 2
header_2 = ["Model family"]

for _ in FOCAL_VARIABLES:
    header_2.extend(["Contracts", "Expands"])

lines.append(" & ".join(header_2) + r" \\")
lines.append(r"\midrule")

# Body panels
for panel_idx, (panel_label, rows) in enumerate(PANELS):
    if panel_idx > 0:
        lines.append(r"\addlinespace")

    lines.append(
        rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{latex_escape(panel_label)}}}}} \\"
    )

    for row_label, specs_by_focal in rows:
        row_cells = [latex_escape(row_label)]

        for focal_label, variable in FOCAL_VARIABLES.items():
            spec = specs_by_focal.get(focal_label)

            for outcome_short, outcome_full in OUTCOMES.items():
                row_cells.append(
                    get_direction_cell(
                        param_df=param_df,
                        spec=spec,
                        variable=variable,
                        outcome=outcome_full,
                    )
                )

        lines.append(" & ".join(row_cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

lines.append("")
lines.append(r"\vspace{0.4em}")
lines.append(r"\begin{minipage}{0.95\linewidth}")
lines.append(r"\footnotesize")
lines.append(rf"\textit{{Notes:}} {NOTES}")
lines.append(r"\end{minipage}")
lines.append("")
lines.append(r"\end{table}")

latex = "\n".join(lines)

OUT_LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_LATEX_PATH.write_text(latex)

print(f"Saved LaTeX table to: {OUT_LATEX_PATH.resolve()}")