from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = BASE_DIR / "outputs" / "defendant_pipeline"

INPUT_CSV = WORKDIR / "cases_v5_with_defendant_full.csv"
OUTPUT_DIR = WORKDIR / "summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Looking for:", INPUT_CSV)
if not INPUT_CSV.exists():
    print("Available CSVs:")
    for f in WORKDIR.glob("*.csv"):
        print("-", f.name)
    raise FileNotFoundError(INPUT_CSV)

df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")

for col in [
    "defendant",
    "defendant_evidence",
    "defendant_explanation",
    "defendant_classification",
    "defendant_classification_correct",
    "classification_evidence",
    "classification_explanation",
    "defendant_error",
]:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str).str.strip()

n = len(df)

def pct(x):
    return round((x / n) * 100, 2) if n else 0

class_counts = (
    df["defendant_classification_correct"]
    .replace("", "blank")
    .value_counts(dropna=False)
    .rename_axis("defendant_classification_correct")
    .reset_index(name="n")
)
class_counts["percent"] = class_counts["n"].apply(pct)

problem_rows = df[
    df["defendant"].isin(["", "not_found", "unclear", "error"])
    | df["defendant_classification_correct"].isin(["", "unclear"])
    | df["defendant_error"].ne("")
    | df["defendant_evidence"].eq("")
    | df["classification_evidence"].eq("")
].copy()

sample_rows = df.sample(n=min(50, len(df)), random_state=42)

class_counts.to_csv(OUTPUT_DIR / "defendant_classification_counts_v2.csv", index=False)
problem_rows.to_csv(OUTPUT_DIR / "defendant_problem_rows_v2.csv", index=False)
sample_rows.to_csv(OUTPUT_DIR / "defendant_random_sample_50_v2.csv", index=False)

summary = f"""
DEFENDANT EXTRACTION SUMMARY

Input:
{INPUT_CSV}

Total rows: {n}

Rows with errors: {(df["defendant_error"].ne("")).sum()} ({pct((df["defendant_error"].ne("")).sum())}%)
Defendant not_found/unclear/error/blank: {(df["defendant"].isin(["", "not_found", "unclear", "error"])).sum()} ({pct((df["defendant"].isin(["", "not_found", "unclear", "error"])).sum())}%)
Classification unclear/blank: {(df["defendant_classification_correct"].isin(["", "unclear"])).sum()} ({pct((df["defendant_classification_correct"].isin(["", "unclear"])).sum())}%)
Missing defendant evidence: {(df["defendant_evidence"].eq("")).sum()} ({pct((df["defendant_evidence"].eq("")).sum())}%)
Missing classification evidence: {(df["classification_evidence"].eq("")).sum()} ({pct((df["classification_evidence"].eq("")).sum())}%)

Classification counts:
{class_counts.to_string(index=False)}

Files written:
- {OUTPUT_DIR / "defendant_classification_counts.csv"}
- {OUTPUT_DIR / "defendant_problem_rows.csv"}
- {OUTPUT_DIR / "defendant_random_sample_50.csv"}
""".strip()

print(summary)
(OUTPUT_DIR / "defendant_results_summary.txt").write_text(summary, encoding="utf-8")