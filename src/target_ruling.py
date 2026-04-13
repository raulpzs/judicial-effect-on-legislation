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

INPUT_CSV = Path("judicial_decisions_with_constitutional_review.csv")
WORKDIR = Path("target_of_ruling_work")
WORKDIR.mkdir(parents=True, exist_ok=True)

BATCH_INPUT_JSONL = WORKDIR / "target_of_ruling_batch.jsonl"
BATCH_OUTPUT_JSONL = WORKDIR / "target_of_ruling_output.jsonl"
BATCH_ERROR_JSONL = WORKDIR / "target_of_ruling_errors.jsonl"
MERGED_OUTPUT_CSV = Path("judicial_decisions_with_constitutional_review_and_target.csv")

LIMIT_ROWS: Optional[int] = None
POLL_INTERVAL_SECONDS = 30

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dtapi.openai.azure.com/openai/v1/",
)

DEVELOPER_PROMPT = """
You are extracting one specific piece of information from legal case summaries.

Task:
Identify the single primary target of the ruling, only when that target is explicitly stated in the text.

Definition:
The target of the ruling is the main institution, actor, or decision-maker whose action, law, decision, or authority is explicitly reviewed, constrained, directed, invalidated, upheld, or otherwise acted upon by the court.

Allowed values:
- executive: The ruling explicitly targets actions or decisions of the executive branch, government officials, ministries, police, prosecutors, agencies, regulators, or administrative authorities.
- legislature: The ruling explicitly targets a law, statute, provision, decree, code, or legislative act enacted by a legislature or equivalent lawmaking authority.
- lower_court: The ruling explicitly targets, reverses, remands, criticizes, or reviews a lower court or subordinate judicial decision.
- private_actor: The ruling explicitly targets a company, media platform, employer, private institution, or other non-state actor.
- mixed: Multiple targets are explicitly present and no single primary target is clear.
- not_found: No single explicit target can be identified.
- unclear: The text is ambiguous about the target.

Rules:
1. Only identify a target if it is explicitly supported by the text.
2. Do NOT infer the target from background context alone.
3. Do NOT guess based on who the parties are.
4. If the court is reviewing the validity of a law or provision, classify as legislature only if the law or provision itself is the explicit object of review.
5. If the court is reviewing an arrest, sanction, censorship action, permit denial, prosecution, administrative decision, police conduct, or executive enforcement action, classify as executive.
6. If the court is reviewing a lower court judgment, classify as lower_court.
7. If the court is reviewing conduct by a company, publisher, platform, employer, or other private body, classify as private_actor.
8. If multiple targets appear and one is not clearly primary, return "mixed".
9. If no explicit target is present, return "not_found".
10. If the text is contradictory or too ambiguous, return "unclear".
11. Be conservative. Do not guess.
12. target_of_ruling_evidence must be a short verbatim snippet copied exactly from the text.
13. target_of_ruling_explanation must briefly explain why the result is chosen.
14. If no exact supporting phrase exists, leave target_of_ruling_evidence as an empty string.

Output must follow the JSON schema exactly.
""".strip()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_of_ruling": {
            "type": ["string", "null"],
            "enum": ["executive", "legislature", "lower_court", "private_actor", "mixed", None],
        },
        "target_of_ruling_status": {
            "type": "string",
            "enum": ["found", "not_found", "unclear", "error"],
        },
        "target_of_ruling_evidence": {
            "type": "string"
        },
        "target_of_ruling_explanation": {
            "type": "string"
        }
    },
    "required": [
        "target_of_ruling",
        "target_of_ruling_status",
        "target_of_ruling_evidence",
        "target_of_ruling_explanation"
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


def shorten(text: str, max_chars: int = 8000) -> str:
    return text


def build_prompt(row: pd.Series) -> str:
    summary = shorten(clean_text(row.get("summary_outcome")))
    decision = shorten(clean_text(row.get("decision_overview")))
    facts = shorten(clean_text(row.get("facts")))

    return f"""
Identify the primary target of the ruling from this case.

Case metadata:
- Judicial body: {clean_text(row.get("Judicial Body"))}
- Country: {clean_text(row.get("Country"))}
- Decision date: {clean_text(row.get("Decision Date"))}

Case text:

DECISION OVERVIEW:
{decision}

SUMMARY:
{summary}

FACTS:
{facts}
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
                    "name": "target_of_ruling_extraction",
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

    results_by_index: Dict[int, Dict[str, Any]] = {}
    errors_by_index: Dict[int, str] = {}

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
            errors_by_index[int(m.group(1))] = json.dumps(line, ensure_ascii=False)

    col_val, col_status, col_ev, col_exp, col_err = [], [], [], [], []

    for idx in original_df.index:
        r = results_by_index.get(idx)

        if r:
            col_val.append(r.get("target_of_ruling"))
            col_status.append(r.get("target_of_ruling_status", "unclear"))
            col_ev.append(r.get("target_of_ruling_evidence", ""))
            col_exp.append(r.get("target_of_ruling_explanation", ""))
            col_err.append(errors_by_index.get(idx, ""))
        else:
            col_val.append(None)
            col_status.append("error")
            col_ev.append("")
            col_exp.append("")
            col_err.append(errors_by_index.get(idx, "No result"))

    merged = original_df.copy()
    merged["target_of_ruling"] = col_val
    merged["target_of_ruling_status"] = col_status
    merged["target_of_ruling_evidence"] = col_ev
    merged["target_of_ruling_explanation"] = col_exp
    merged["target_of_ruling_error"] = col_err

    return merged


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")

    if LIMIT_ROWS is not None:
        df = df.head(LIMIT_ROWS).copy()
        print(f"Testing mode: using first {len(df)} rows")
    else:
        print(f"Full mode: using all {len(df)} rows")

    count = write_batch_jsonl(df, BATCH_INPUT_JSONL)
    print(f"Requests: {count}")

    if count == 0:
        print("No usable rows found. Exiting.")
        return

    print("Uploading input file and creating batch...")
    input_file, batch = submit_batch(BATCH_INPUT_JSONL)
    print(f"Input file id: {input_file.id}")
    print(f"Batch id: {batch.id}")
    print(f"Initial batch status: {batch.status}")

    final = poll_batch(batch.id)
    print(f"Final batch status: {final.status}")

    if final.output_file_id:
        print(f"Downloading output file: {final.output_file_id}")
        download_file(final.output_file_id, BATCH_OUTPUT_JSONL)

    if final.error_file_id:
        print(f"Downloading error file: {final.error_file_id}")
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