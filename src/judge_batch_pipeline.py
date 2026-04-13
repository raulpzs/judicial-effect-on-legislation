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

INPUT_CSV = Path("cases-may-2026.csv")
WORKDIR = Path("judge_batch_work")
WORKDIR.mkdir(parents=True, exist_ok=True)

BATCH_INPUT_JSONL = WORKDIR / "judge_extraction_batch.jsonl"
BATCH_OUTPUT_JSONL = WORKDIR / "judge_extraction_output.jsonl"
BATCH_ERROR_JSONL = WORKDIR / "judge_extraction_errors.jsonl"
MERGED_OUTPUT_CSV = Path("cases-may-2026_with_judges.csv")
LIMIT_ROWS: Optional[int] = None

POLL_INTERVAL_SECONDS = 30

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://dtapi.openai.azure.com/openai/v1/")

DEVELOPER_PROMPT = """
You are extracting one specific piece of information from legal case summaries.

Task:
Identify the single judge explicitly named in the text as writing, authoring,
delivering, or issuing the main ruling or opinion.

Rules:
1. Only return a judge if the text explicitly identifies that person as the
   author or issuer of the ruling.
2. Do NOT infer from court membership.
3. Do NOT return panel members unless one is explicitly identified as the author.
4. Do NOT return concurring or dissenting judges unless the text explicitly says
   that judge authored the main ruling.
5. Do NOT return parties, lawyers, prosecutors, academics, or judges from lower courts.
6. If no single ruling judge is explicitly named, return "not_found".
7. If multiple judges are named and authorship is unclear, return "unclear".
8. Be conservative. Do not guess.
9. ruling_judge_evidence must be a short verbatim snippet copied exactly from the text.
10. ruling_judge_explanation must briefly explain why the result is found, not_found, or unclear.
11. If no exact supporting phrase exists, leave ruling_judge_evidence as an empty string.

Output must follow the JSON schema exactly.
""".strip()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ruling_judge": {
            "type": ["string", "null"]
        },
        "ruling_judge_status": {
            "type": "string",
            "enum": ["found", "not_found", "unclear", "error"]
        },
        "ruling_judge_evidence": {
            "type": "string"
        },
        "ruling_judge_explanation": {
            "type": "string"
        }
    },
    "required": [
        "ruling_judge",
        "ruling_judge_status",
        "ruling_judge_evidence",
        "ruling_judge_explanation"
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
    """
    Keeps prompts from becoming too large.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[TRUNCATED]"


def build_prompt(row: pd.Series) -> str:
    summary = shorten(clean_text(row.get("Case Summary and Outcome")))
    facts = shorten(clean_text(row.get("Facts")))
    decision = shorten(clean_text(row.get("Decision Overview")))

    return f"""
Identify the ruling judge from this case.

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
        clean_text(row.get("Case Summary and Outcome")),
        clean_text(row.get("Facts")),
        clean_text(row.get("Decision Overview")),
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
                    "name": "judge_extraction",
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
    terminal_states = {
        "completed",
        "failed",
        "expired",
        "cancelled",
    }

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
    """
    Batch output structure can vary by SDK/object serialization path.
    This tries the most common locations for output text.
    """
    if isinstance(response_body, dict) and "output_text" in response_body:
        return json.loads(response_body["output_text"])

    # Responses API content blocks
    output = response_body.get("output", [])
    for item in output:
        content_blocks = item.get("content", [])
        for block in content_blocks:
            if block.get("type") in ("output_text", "text") and "text" in block:
                return json.loads(block["text"])

    raise ValueError("Could not find structured JSON text in response body.")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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
            custom_id = line.get("custom_id", "")
            m = re.match(r"case_(\d+)$", custom_id)
            if not m:
                continue

            row_idx = int(m.group(1))
            try:
                response_body = line["response"]["body"]
                parsed = parse_response_output_text(response_body)
                results_by_index[row_idx] = parsed
            except Exception as e:
                errors_by_index[row_idx] = f"Parse error: {e}"

    if error_jsonl_path and error_jsonl_path.exists():
        for line in load_jsonl(error_jsonl_path):
            custom_id = line.get("custom_id", "")
            m = re.match(r"case_(\d+)$", custom_id)
            if not m:
                continue
            row_idx = int(m.group(1))
            errors_by_index[row_idx] = json.dumps(line, ensure_ascii=False)

    ruling_judge = []
    ruling_status = []
    ruling_evidence = []
    ruling_explanation = []
    error_notes = []

    for idx in original_df.index:
        result = results_by_index.get(idx)

        if result:
            ruling_judge.append(result.get("ruling_judge"))
            ruling_status.append(result.get("ruling_judge_status", "unclear"))
            ruling_evidence.append(result.get("ruling_judge_evidence", ""))
            ruling_explanation.append(result.get("ruling_judge_explanation", ""))
            error_notes.append(errors_by_index.get(idx, ""))
        else:
            ruling_judge.append(None)
            ruling_status.append("error")
            ruling_evidence.append("")
            ruling_explanation.append("")
            error_notes.append(errors_by_index.get(idx, "No batch result for this row"))
    
    merged = original_df.copy()
    merged["ruling_judge"] = ruling_judge
    merged["ruling_judge_status"] = ruling_status
    merged["ruling_judge_evidence"] = ruling_evidence
    merged["ruling_judge_explanation"] = ruling_explanation
    merged["ruling_judge_error"] = error_notes

    return merged


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")

    if LIMIT_ROWS is not None:
        df = df.head(LIMIT_ROWS).copy()
        print(f"Testing mode: using first {len(df)} rows")
    else:
        print(f"Full mode: using all {len(df)} rows")

    print(f"Writing batch JSONL: {BATCH_INPUT_JSONL}")
    request_count = write_batch_jsonl(df, BATCH_INPUT_JSONL)
    print(f"Batch requests written: {request_count}")

    if request_count == 0:
        print("No usable rows found. Exiting.")
        return

    print("Uploading input file and creating batch...")
    input_file, batch = submit_batch(BATCH_INPUT_JSONL)
    print(f"Input file id: {input_file.id}")
    print(f"Batch id: {batch.id}")
    print(f"Initial batch status: {batch.status}")

    print("Polling batch until terminal state...")
    final_batch = poll_batch(batch.id)
    print(f"Final batch status: {final_batch.status}")

    output_file_id = getattr(final_batch, "output_file_id", None)
    error_file_id = getattr(final_batch, "error_file_id", None)

    if output_file_id:
        print(f"Downloading output file: {output_file_id}")
        download_file(output_file_id, BATCH_OUTPUT_JSONL)
        print(f"Saved output JSONL to: {BATCH_OUTPUT_JSONL}")
    else:
        print("No output file id found.")

    if error_file_id:
        print(f"Downloading error file: {error_file_id}")
        download_file(error_file_id, BATCH_ERROR_JSONL)
        print(f"Saved error JSONL to: {BATCH_ERROR_JSONL}")
    else:
        print("No error file id found.")

    print("Merging results back into CSV...")
    merged_df = merge_results_back(
        original_df=df,
        output_jsonl_path=BATCH_OUTPUT_JSONL if BATCH_OUTPUT_JSONL.exists() else Path("missing"),
        error_jsonl_path=BATCH_ERROR_JSONL if BATCH_ERROR_JSONL.exists() else None,
    )
    merged_df.to_csv(MERGED_OUTPUT_CSV, index=False)
    print(f"Saved merged CSV to: {MERGED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()