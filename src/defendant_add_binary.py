from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_CSV = BASE_DIR / "data" / "processed" / "cases_v5_short_modes.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "cases_v6_short.csv"

CLASS_COL = "defendant_classification_correct"  # matches your spelling

BINARY_VARS = {
    "citizen": "defendant_citizen",
    "press": "defendant_press",
    "government": "defendant_government",
    "intermediary": "defendant_intermediary",
    "other": "defendant_other",
    "unclear": "defendant_unclear",
}

def clean_label(x):
    if pd.isna(x):
        return "unclear"
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)

    replacements = {
        "member_of_the_press": "press",
        "media": "press",
        "journalist": "press",
        "author": "press",        "broadcasting_company": "intermediary",
        "broadcasting_companies": "intermediary",
        "internet_intermediary": "intermediary",
        "state": "government",
        "gov": "government",
        "not_found": "unclear",
        "": "unclear",
        "nan": "unclear",
    }

    return replacements.get(x, x)

def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")

    if CLASS_COL not in df.columns:
        raise ValueError(
            f"Column '{CLASS_COL}' not found. Available columns are:\n{df.columns.tolist()}"
        )

    clean_class = df[CLASS_COL].apply(clean_label)

    for category, new_var in BINARY_VARS.items():
        df[new_var] = (clean_class == category).astype(int)

    valid_categories = set(BINARY_VARS.keys())
    df["defendant_classification_unexpected"] = (~clean_class.isin(valid_categories)).astype(int)

    unexpected = sorted(clean_class[~clean_class.isin(valid_categories)].dropna().unique())

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print("\nBinary variable counts:")
    for _, new_var in BINARY_VARS.items():
        print(f"{new_var}: {df[new_var].sum()}")

    if unexpected:
        print("\nUnexpected labels found:")
        for label in unexpected:
            print(f"- {label}")
    else:
        print("\nNo unexpected labels found.")

if __name__ == "__main__":
    main()