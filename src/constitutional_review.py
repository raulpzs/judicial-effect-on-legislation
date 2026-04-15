import os
import re
import json
import time
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-5.1"

INPUT_CSV = Path("data/raw/judicial_decisions_matched.csv")
WORKDIR = Path("outputs/constitutional_review_work_2")
WORKDIR.mkdir(parents=True, exist_ok=True)

BATCH_INPUT_JSONL = WORKDIR / "constitutional_review_batch_2.jsonl"
BATCH_OUTPUT_JSONL = WORKDIR / "constitutional_review_output_2.jsonl"
BATCH_ERROR_JSONL = WORKDIR / "constitutional_review_errors_2.jsonl"
MERGED_OUTPUT_CSV = Path("outputs/judicial_decisions_with_constitutional_review_2.csv")

LIMIT_ROWS: Optional[int] = None
POLL_INTERVAL_SECONDS = 30

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dtapi.openai.azure.com/openai/v1/",
)

DEVELOPER_PROMPT = """
You are extracting one specific piece of information from legal case summaries.

Task:
Identify whether the court explicitly reviewed the validity or meaning of a law,
regulation, or legal provision.

Definitions:
- invalidated: The court explicitly struck down, nullified, or declared a law unconstitutional.
- upheld: The court explicitly confirmed that a law is valid or constitutional.
- interpreted: The court explicitly clarified, limited, or defined the meaning or scope of a law or provision, beyond routine application, without invalidating it.

You must return:
- constitutional_review:
    - a list containing one or more of "invalidated", "upheld", "interpreted" if status is "found"
    - null if status is "not_found" or "unclear"
- constitutional_review_status: one of "found", "not_found", "unclear"
- constitutional_review_evidence: a short verbatim snippet copied exactly from the text. If no exact supporting phrase exists, leave empty.
- constitutional_review_explanation: a brief explanation

Rules:
- Only classify actions that are explicitly stated in the text.
- Do not infer constitutional review from general reasoning or outcomes.
- Do not classify cases that only apply a law without reviewing it.
- If multiple actions are clearly present (e.g., part of a law invalidated and part upheld), include all applicable labels.

Ambiguous cases:
If no explicit constitutional or legal review of a law is present, return "not_found".
If the text is ambiguous or contradictory, return "unclear".

Output must follow the JSON schema exactly.
""".strip()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "constitutional_review": {
            "type": ["array", "null"],
            "items": {
                "type": "string",
                "enum": ["invalidated", "upheld", "interpreted"]
            },
            "minItems": 1,
            "uniqueItems": True
        },
        "constitutional_review_status": {
            "type": "string",
            "enum": ["found", "not_found", "unclear", "error"],
        },
        "constitutional_review_evidence": {
            "type": "string"
        },
        "constitutional_review_explanation": {
            "type": "string"
        }
    },
    "required": [
        "constitutional_review",
        "constitutional_review_status",
        "constitutional_review_evidence",
        "constitutional_review_explanation"
    ],
    "additionalProperties": False
}

def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text(value: Any) -> str:
    text = safe_text(value)
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_prompt(row: pd.Series) -> str:
    summary = clean_text(row.get("summary_outcome"))
    decision = clean_text(row.get("decision_overview"))

    return f"""
Identify the constitutional review outcome from this case.

Case metadata:
- Judicial body: {clean_text(row.get("Judicial Body"))}
- Country: {clean_text(row.get("Country"))}
- Decision date: {clean_text(row.get("Decision Date"))}

Case text:

DECISION OVERVIEW:
{decision}

SUMMARY:
{summary}
""".strip()

def has_substantive_text(row: pd.Series) -> bool:
    fields = [
        clean_text(row.get("summary_outcome")),
        clean_text(row.get("decision_overview")),
        clean_text(row.get("facts")),
    ]
    return any(len(x) >= 20 for x in fields)

def make_batch_request(row_index: int, row: pd.Series) -> Dict[str, Any]:
    prompt = build_prompt(row)

    return {
        "custom_id": f"case_{row_index}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "constitutional_review_extraction",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        },
    }

def write_batch_jsonl(df: pd.DataFrame, path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            if not has_substantive_text(row):
                continue
            req = make_batch_request(idx, row)
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            count += 1
    return count


def submit_batch(batch_jsonl_path: Path):
    with batch_jsonl_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return input_file, batch


def poll_batch(batch_id: str):
    terminal_states = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch.status}")

        if getattr(batch, "request_counts", None):
            counts = batch.request_counts
            print(
                f"Counts - total: {getattr(counts, 'total', None)}, "
                f"completed: {getattr(counts, 'completed', None)}, "
                f"failed: {getattr(counts, 'failed', None)}"
            )

        if batch.status in terminal_states:
            return batch

        time.sleep(POLL_INTERVAL_SECONDS)


def download_file(file_id: str, destination: Path) -> None:
    content = client.files.content(file_id)
    data = content.read()
    destination.write_bytes(data)


def parse_response_output_text(response_body: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(response_body, dict) and "output_text" in response_body:
        return json.loads(response_body["output_text"])

    output = response_body.get("output", [])
    for item in output:
        for block in item.get("content", []):
            if block.get("type") in ("output_text", "text") and "text" in block:
                return json.loads(block["text"])

    raise ValueError("Could not find structured JSON text in response body.")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_results_back(
    original_df: pd.DataFrame,
    output_jsonl_path: Path,
    error_jsonl_path: Optional[Path],
) -> pd.DataFrame:

    results_by_index = {}
    errors_by_index = {}

    if output_jsonl_path.exists():
        for line in load_jsonl(output_jsonl_path):
            m = re.match(r"case_(\d+)$", line.get("custom_id", ""))
            if not m:
                continue

            idx = int(m.group(1))
            try:
                parsed = parse_response_output_text(line["response"]["body"])
                results_by_index[idx] = parsed
            except Exception as e:
                errors_by_index[idx] = f"Parse error: {e}"

    if error_jsonl_path and error_jsonl_path.exists():
        for line in load_jsonl(error_jsonl_path):
            m = re.match(r"case_(\d+)$", line.get("custom_id", ""))
            if not m:
                continue
            errors_by_index[int(m.group(1))] = json.dumps(line)

    col_val, col_status, col_ev, col_exp, col_err = [], [], [], [], []

    for idx in original_df.index:
        r = results_by_index.get(idx)

        if r:
            value = r.get("constitutional_review")
            if isinstance(value, list):
                col_val.append(json.dumps(value, ensure_ascii=False))
            else:
                col_val.append(value)
            col_status.append(r.get("constitutional_review_status", "unclear"))
            col_ev.append(r.get("constitutional_review_evidence", ""))
            col_exp.append(r.get("constitutional_review_explanation", ""))
            col_err.append(errors_by_index.get(idx, ""))
        else:
            col_val.append(None)
            col_status.append("error")
            col_ev.append("")
            col_exp.append("")
            col_err.append(errors_by_index.get(idx, "No result"))

    merged = original_df.copy()
    merged["constitutional_review"] = col_val
    merged["constitutional_review_status"] = col_status
    merged["constitutional_review_evidence"] = col_ev
    merged["constitutional_review_explanation"] = col_exp
    merged["constitutional_review_error"] = col_err

    return merged


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")

    if LIMIT_ROWS:
        df = df.head(LIMIT_ROWS)

    count = write_batch_jsonl(df, BATCH_INPUT_JSONL)
    print(f"Requests: {count}")

    input_file, batch = submit_batch(BATCH_INPUT_JSONL)
    final = poll_batch(batch.id)

    if final.output_file_id:
        download_file(final.output_file_id, BATCH_OUTPUT_JSONL)

    if final.error_file_id:
        download_file(final.error_file_id, BATCH_ERROR_JSONL)

    merged = merge_results_back(
        df,
        BATCH_OUTPUT_JSONL,
        BATCH_ERROR_JSONL if BATCH_ERROR_JSONL.exists() else None,
    )

    merged.to_csv(MERGED_OUTPUT_CSV, index=False)
    print(f"Saved: {MERGED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()