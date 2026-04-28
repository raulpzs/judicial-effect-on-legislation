# basic eda for judicial decision project

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Directories
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "processed" / "cases_v4_short_merged.csv"

# =========================
# Loading data
# =========================

df = pd.read_csv(DATA_PATH)

print(df.head())

# =========================
# Clipping long strings
# =========================

def truncate_for_display(df, max_len=50):
    df_copy = df.copy()
    obj_cols = df_copy.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        df_copy[col] = df_copy[col].astype(str).apply(
            lambda x: x if len(x) <= max_len else x[:max_len] + "..."
        )

    return df_copy

# =========================
# Categorical frequency tables (top 5)
# =========================

cat_blocks = []

cat_blocks.insert(0, "<h1>Categorical Variables: Top Frequencies</h1>")

obj_cols = df.select_dtypes(include=["object", "string"]).columns

for col in obj_cols:
    
    # --- compute frequencies on raw data (IMPORTANT) ---
    freq = (
        df[col]
        .value_counts(dropna=False)
        .head(5)
        .to_frame("count")
    )

    # --- clip ONLY for display ---
    def clip_index(val, max_len=50):
        if pd.isna(val):
            return "__missing__"
        val = str(val)
        return val if len(val) <= max_len else val[:max_len] + "..."

    freq.index = [clip_index(v) for v in freq.index]

    # --- ensure missing is explicit and correct ---
    if df[col].isna().any():
        freq.loc["__missing__"] = df[col].isna().sum()

    styled = (
        freq.style
        .set_caption(f"Top 5 values: {col}")
        .set_table_styles([
            # Header styling
            {"selector": "th", "props": [
                ("background-color", "#2c3e50"),
                ("color", "white"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("padding", "8px")
            ]},
    
            # Cell styling
            {"selector": "td", "props": [
                ("padding", "6px 10px"),
                ("border", "1px solid #ddd")
            ]},
    
            # Table layout
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "60%"),
                ("margin", "10px 0")
            ]},
    
            # Caption styling
            {"selector": "caption", "props": [
                ("caption-side", "top"),
                ("font-size", "16px"),
                ("font-weight", "bold"),
                ("padding", "5px")
            ]}
        ])
    )

    cat_blocks.append(styled.to_html())


# =========================
# Descriptive statistics
# =========================

html_blocks = []

steps = 5

left = 0

while left < df.shape[1]:
    
    right = left + steps

    chunk_raw = df.iloc[:, left:right]
    chunk_display = truncate_for_display(chunk_raw)
    
    chunk = chunk_display.describe(include="all")

    styled = (
        chunk.style
        .set_caption(f"Descriptive stats: Columns {left}–{right-1}")
        .set_table_styles([
            # Table container
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "80%"),
                ("margin", "10px 0"),
                ("font-size", "13px")
            ]},
    
            # Header styling
            {"selector": "th", "props": [
                ("background-color", "#2c3e50"),
                ("color", "white"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("padding", "8px"),
                ("border", "1px solid #ddd")
            ]},
    
            # Cell styling
            {"selector": "td", "props": [
                ("padding", "6px 10px"),
                ("border", "1px solid #ddd"),
                ("text-align", "center")
            ]},
    
            # Row striping
            {"selector": "tr:nth-child(even)", "props": [
                ("background-color", "#f9f9f9")
            ]},
    
            # Hover effect
            {"selector": "tr:hover", "props": [
                ("background-color", "#f1f1f1")
            ]},
    
            # Caption styling
            {"selector": "caption", "props": [
                ("caption-side", "top"),
                ("font-size", "16px"),
                ("font-weight", "bold"),
                ("padding", "6px"),
                ("color", "#2c3e50")
            ]}
        ])
    )

    
    html_blocks.append(
        f"<h2>Columns {left}–{right-1}</h2>" + styled.to_html()
    )

    left += steps

full_html = f"""
<html>
<head>
    <title>EDA Summary</title>
</head>
<body>
    {"".join(cat_blocks + html_blocks)}
</body>
</html>
"""

OUTPUT_FILE = OUTPUT_DIR / "summary.html"

with open(OUTPUT_FILE, "w") as f:
    f.write(full_html)