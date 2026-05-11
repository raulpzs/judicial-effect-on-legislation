# script for generating coefficient plots for main models

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PARAM_PATH = OUTPUT_DIR / "param_table_all_specs.csv"

OUT_PATH = FIG_DIR / "coef_plot_main_models_grouped.png"

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

OUTCOME_ORDER = [
    "Contracts Expression",
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

df = df.sort_values(["spec", "variable", "outcome"]).copy()

groups = (
    df[["spec", "variable"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

groups["group_y"] = range(len(groups))

df = df.merge(groups, on=["spec", "variable"], how="left")

offsets = {
    "Contracts Expression": -0.15,
    "Expands Expression": 0.15,
}

markers = {
    "Contracts Expression": "o",
    "Expands Expression": "s",
}

df["y"] = df["group_y"] + df["outcome"].map(offsets).astype(float)

fig_height = max(8, 0.35 * len(groups))
fig, ax = plt.subplots(figsize=(11, fig_height))

# alternating shaded bands for variable/spec groups
for i, row in groups.iterrows():
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, alpha=0.08)

# plot coefficients by outcome
for outcome in OUTCOME_ORDER:
    sub = df[df["outcome"] == outcome]

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

groups["label"] = groups["variable"].astype(str) + " | " + groups["spec"].astype(str)

ax.set_yticks(groups["group_y"])
ax.set_yticklabels(groups["label"])

ax.set_xlabel("Coefficient estimate with 95% CI")
ax.set_ylabel("")
ax.set_title("Coefficient plot for main model specifications")
ax.legend(title="Outcome vs. Mixed Outcome")

ax.invert_yaxis()
fig.tight_layout()

fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUT_PATH}")