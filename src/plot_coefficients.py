# script for generating coefficient plots for main models

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARAM_PATH = OUTPUT_DIR / "param_table_all_specs_contract.csv"

OUT_PATH = FIG_DIR / "coef_plot_main_models_grouped_contract.png"

MAIN_SPECS = [
    "spec_extended_1",
    "spec_extended_2",
    "spec_extended_3",
    "spec_c_pack",
    "spec_im_pack",
    "spec_p_pack",
    "spec_govprot_pack",
    "spec_c_reform",
    "spec_im_reform",
    "spec_p_reform",
    "spec_govprot_reform",
    "spec_c_attack",
    "spec_im_attack",
    "spec_p_attack",
    "spec_govprot_attack",
    "spec_c_reform_decision",
    "spec_im_reform_decision",
    "spec_p_reform_decision",
    "spec_govprot_reform_decision",
    "spec_c_pack_decision",
    "spec_im_pack_decision",
    "spec_p_pack_decision",
    "spec_govprot_pack_decision",
]

MAIN_VARIABLES = [
    "wdj_expression_lag1",
    "wdj_citizen_lag1",
    "wdj_intermediaries_lag1",
    "wdj_press_lag1",
    "wdj_govprot_lag1",
    "v2jureform_lag1",
    "v2jupoatck_lag1",
    "v2jupack_lag1",
]

VARIABLE_LABELS = {
    "wdj_expression_lag1": "Expression decisions",
    "wdj_citizen_lag1": "Citizen-related decisions",
    "wdj_intermediaries_lag1": "Intermediary-related decisions",
    "wdj_press_lag1": "Press-related decisions",
    "wdj_govprot_lag1": "Government protection decisions",
    "v2jureform_lag1": "Judicial reform",
    "v2jupoatck_lag1": "Judicial attacks",
    "v2jupack_lag1": "Judicial packing",
}

SPEC_LABELS = {
    "spec_extended_1": "Extended 1",
    "spec_extended_2": "Extended 2",
    "spec_extended_3": "Extended 3",

    "spec_c_pack": "Citizen × Packing",
    "spec_im_pack": "Intermediary × Packing",
    "spec_p_pack": "Press × Packing",
    "spec_govprot_pack": "Gov. protection × Packing",

    "spec_c_reform": "Citizen × Reform",
    "spec_im_reform": "Intermediary × Reform",
    "spec_p_reform": "Press × Reform",
    "spec_govprot_reform": "Gov. protection × Reform",

    "spec_c_attack": "Citizen × Attacks",
    "spec_im_attack": "Intermediary × Attacks",
    "spec_p_attack": "Press × Attacks",
    "spec_govprot_attack": "Gov. protection × Attacks",

    "spec_c_reform_decision": "Citizen × Reform + decision controls",
    "spec_im_reform_decision": "Intermediary × Reform + decision controls",
    "spec_p_reform_decision": "Press × Reform + decision controls",
    "spec_govprot_reform_decision": "Gov. protection × Reform + decision controls",

    "spec_c_pack_decision": "Citizen × Packing + decision controls",
    "spec_im_pack_decision": "Intermediary × Packing + decision controls",
    "spec_p_pack_decision": "Press × Packing + decision controls",
    "spec_govprot_pack_decision": "Gov. protection × Packing + decision controls",
}

OUTCOME_ORDER = [
    "Expands Expression",
]

df = pd.read_csv(PARAM_PATH)

df = df[
    df["spec"].isin(MAIN_SPECS)
    & df["variable"].isin(MAIN_VARIABLES)
    & df["outcome"].isin(OUTCOME_ORDER)
].copy()

df["ci_low"] = df["coef"] - 1.96 * df["std_err"]
df["ci_high"] = df["coef"] + 1.96 * df["std_err"]

df["spec"] = pd.Categorical(df["spec"], categories=MAIN_SPECS, ordered=True)
df["variable"] = pd.Categorical(df["variable"], categories=MAIN_VARIABLES, ordered=True)
df["outcome"] = pd.Categorical(df["outcome"], categories=OUTCOME_ORDER, ordered=True)

df = df.sort_values(["variable", "spec", "outcome"]).copy()

# -------------------------------------------------------------------
# Build grouped y-axis:
# variable block -> specs within variable -> two outcomes per spec
# -------------------------------------------------------------------

groups = (
    df[["variable", "spec"]]
    .drop_duplicates()
    .sort_values(["variable", "spec"])
    .reset_index(drop=True)
)

y_positions = []
current_y = 0
gap_between_variables = 1.1

last_var = None

for _, row in groups.iterrows():
    var = row["variable"]

    if last_var is not None and var != last_var:
        current_y += gap_between_variables

    y_positions.append(current_y)
    current_y += 1
    last_var = var

groups["group_y"] = y_positions

df = df.merge(groups, on=["variable", "spec"], how="left")

offsets = {
    "Contracts Expression": -0.15,
    "Expands Expression": 0.15,
}

markers = {
    "Contracts Expression": "o",
    "Expands Expression": "s",
}

df["y"] = df["group_y"] + df["outcome"].map(offsets).astype(float)

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig_height = max(9, 0.34 * len(groups))
fig, ax = plt.subplots(figsize=(12, fig_height))

# Shade each variable block
variable_blocks = (
    groups.groupby("variable", observed=True)["group_y"]
    .agg(["min", "max"])
    .reset_index()
)

for i, row in variable_blocks.iterrows():
    if i % 2 == 0:
        ax.axhspan(row["min"] - 0.5, row["max"] + 0.5, alpha=0.08)

# Add variable labels as block headers
for _, row in variable_blocks.iterrows():
    var = row["variable"]
    label = VARIABLE_LABELS.get(str(var), str(var))

    ax.text(
        x=ax.get_xlim()[0],
        y=row["min"] - 0.65,
        s=label,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

# Plot coefficients by outcome
for outcome in OUTCOME_ORDER:
    sub = df[df["outcome"] == outcome].copy()

    ax.errorbar(
        sub["coef"],
        sub["y"],
        xerr=[
            sub["coef"] - sub["ci_low"],
            sub["ci_high"] - sub["coef"],
        ],
        fmt=markers[outcome],
        capsize=3,
        linestyle="none",
        label=outcome,
    )

ax.axvline(0, linestyle="--", linewidth=1)

# y-axis labels should now be specs only, because variables are block headers
groups["label"] = groups["spec"].astype(str).map(SPEC_LABELS).fillna(groups["spec"].astype(str))

ax.set_yticks(groups["group_y"])
ax.set_yticklabels(groups["label"], fontsize=8)

ax.set_xlabel("Coefficient estimate with 95% CI")
ax.set_ylabel("")
ax.set_title("Coefficient estimates across main model specifications, grouped by variable")
ax.legend(title="Outcome vs. Mixed Outcome")

ax.invert_yaxis()
fig.tight_layout()

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUT_PATH}")