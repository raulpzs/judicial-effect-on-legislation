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

#load_dotenv()

MODEL = "gpt-5.1"

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

INPUT_CSV = BASE_DIR / "data" /"processed" / "cases_v5_full.csv"
WORKDIR = BASE_DIR / "outputs" / "defendant_pipeline"
WORKDIR.mkdir(parents=True, exist_ok=True)

LIMIT_ROWS: Optional[int] = None
RESUME_BATCH_ID: Optional[str] = None


RUN_LABEL = f"test_{LIMIT_ROWS}" if LIMIT_ROWS is not None else "full"

BATCH_INPUT_JSONL = WORKDIR / f"defendant_extraction_batch_{RUN_LABEL}.jsonl"
BATCH_OUTPUT_JSONL = WORKDIR / f"defendant_extraction_output_{RUN_LABEL}.jsonl"
BATCH_ERROR_JSONL = WORKDIR / f"defendant_extraction_errors_{RUN_LABEL}.jsonl"
MERGED_OUTPUT_CSV = WORKDIR / f"cases_v5_with_defendant_{RUN_LABEL}.csv"

print("Script file:", Path(__file__).resolve())
print("BASE_DIR:", BASE_DIR)
print("CSV exists:", INPUT_CSV.exists(), INPUT_CSV)
print(".env exists:", (BASE_DIR / ".env").exists())
print("Output dir:", WORKDIR)

POLL_INTERVAL_SECONDS = 30

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://dtapi.openai.azure.com/openai/v1/")

DEVELOPER_PROMPT = """
You are extracting and classifying defendants from legal case summaries for a structured legal dataset.

Your task is to identify the defendant/respondent/accused party and classify the defendant type. Do this in sequence: first determine the defendant, then classify only that identified defendant. Do not choose a defendant because they fit a classification category.

Definitions:
- In a civil case, the defendant is the individual or organization that the plaintiff claims caused harm or sued for relief.
- In a criminal case, the defendant is the individual or organization charged, prosecuted, indicted, convicted, or accused by a government, prosecutor, or international court.
- In an appeal, constitutional review, interlocutory proceeding, or procedural dispute, identify the defendant from the underlying lawsuit or prosecution, not merely the party labeled appellant, respondent, applicant, petitioner, or appellee.
- Do not classify the plaintiff/claimant/applicant as the defendant merely because they are defending a ruling or advancing a legal issue on appeal.

Extraction rules:
1. Return only defendants explicitly named in the text.
2. Do not infer a defendant from background facts alone.
3. If one defendant is clearly named, return that name.
4. If multiple defendants are clearly named and treated collectively, return all named defendants separated by semicolons.
5. If multiple defendants are named but the defendant role is unclear, return "unclear".
6. If no defendant, respondent, accused, charged, prosecuted, or sued party is explicitly named, return "not_found".
7. Be conservative. Do not guess.
8. defendant_evidence must be a short verbatim snippet copied exactly from the text.
9. If no exact supporting phrase exists, leave defendant_evidence as an empty string.
10. defendant_explanation must briefly explain why the result is found, not_found, or unclear.

Defendant classification:
Classify the identified defendant as exactly one of:
- "citizen"
- "press"
- "intermediary"
- "government"
- "other"
- "unclear"

Classification rules:
1. Classify based on the capacity in which the defendant is being sued, charged, prosecuted, or accused, not only their general identity.
2. Use "press" for journalists, authors, editors, newspapers, magazines, publishers, media outlets, or individuals/organizations sued or charged for writing, publishing, reporting, editing, or distributing news, commentary, books, articles, letters, or other expressive publications.
3. Use "intermediary" for social media platforms, internet service providers, telecommunications providers, hosting services, search engines, broadcasting companies, or other communications infrastructure/distribution entities when the case concerns transmitting, hosting, broadcasting, moderating, blocking, or failing to remove content.
4. Use "government" for state institutions, public officials, courts, prosecutors, police, ministries, regulatory bodies, legislatures, or state-owned entities acting in an official capacity.
5. Use "citizen" for private individuals acting as ordinary speakers, protesters, activists, artists, demonstrators, voters, or participants in public debate, unless the case concerns their press, publishing, intermediary, or official government role.
6. Use "other" for corporations, NGOs, unions, political parties, religious organizations, schools, universities, or other entities that do not fit the above categories.
7. If the summary says “X brought defamation suits against Y,” then Y is the defendant, even if X later appears as an appellant, respondent, applicant, or party seeking constitutional relief.
8. If multiple defendants fall into different categories, return "unclear" for defendant_classification unless the case clearly treats them collectively in one shared capacity. For instance, if the defendants are a newspaper and the author of an article, classify them together as "press".
9. classification_evidence must be a short verbatim snippet copied exactly from the text.
10. classification_explanation must briefly explain the classification.
11. If defendant is "not_found" or "unclear", defendant_classification should usually be "unclear" unless the text clearly supports one collective classification.

Return only valid JSON. Do not include markdown, comments, or extra text.
""".strip()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "defendant": {
            "type": "string"
        },
        "defendant_evidence": {
            "type": "string"
        },
        "defendant_explanation": {
            "type": "string"
        },
        "defendant_classification": {
            "type": ["string", "null"],
            "enum": [
                "citizen",
                "press",
                "intermediary",
                "government",
                "other",
                "unclear",
                None
            ]
        },
        "classification_evidence": {
            "type": "string"
        },
        "classification_explanation": {
            "type": "string"
        }
    },
    "required": [
        "defendant",
        "defendant_evidence",
        "defendant_explanation",
        "defendant_classification",
        "classification_evidence",
        "classification_explanation"
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
    summary = shorten(clean_text(row.get("summary_outcome")))
    facts = shorten(clean_text(row.get("facts")))

    return f"""
Extract and classify the defendant from this case summary.

Case metadata:
- Country: {clean_text(row.get("country"))}
- Decision date: {clean_text(row.get("decision_date_raw"))}
- Case ID: {clean_text(row.get("case_id_words"))}

summary_outcome:
{summary}

facts:
{facts}
""".strip()


def has_substantive_text(row: pd.Series) -> bool:
    fields = [
        clean_text(row.get("summary_outcome")),
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
                    "name": "defendant_extraction",
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

    defendant = []
    defendant_evidence = []
    defendant_explanation = []
    defendant_classification = []
    classification_evidence = []
    classification_explanation = []
    error_notes = []

    for idx in original_df.index:
        result = results_by_index.get(idx)

        if result:
            defendant.append(result.get("defendant"))
            defendant_evidence.append(result.get("defendant_evidence", ""))
            defendant_explanation.append(result.get("defendant_explanation", ""))
            defendant_classification.append(result.get("defendant_classification", "unclear"))
            classification_evidence.append(result.get("classification_evidence", ""))
            classification_explanation.append(result.get("classification_explanation", ""))
            error_notes.append(errors_by_index.get(idx, ""))
        else:
            defendant.append(None)
            defendant_evidence.append("")
            defendant_explanation.append("")
            defendant_classification.append("unclear")
            classification_evidence.append("")
            classification_explanation.append("")
            error_notes.append(errors_by_index.get(idx, "No batch result for this row"))

    merged = original_df.copy()
    merged["defendant"] = defendant
    #merged["defendant_status"] = defendant_status
    merged["defendant_evidence"] = defendant_evidence
    merged["defendant_explanation"] = defendant_explanation
    merged["defendant_classification"] = defendant_classification
    merged["classification_evidence"] = classification_evidence
    merged["classification_explanation"] = classification_explanation
    merged["defendant_error"] = error_notes

    return merged


def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, engine="python", on_bad_lines="warn")
    df.columns = df.columns.str.strip()
    
    if LIMIT_ROWS is not None:
        df = df.sample(n=LIMIT_ROWS, random_state=48).copy()
        print(f"Testing mode: using random sample of {len(df)} rows")
    else:
        print(f"Full mode: using all {len(df)} rows")

    print(f"Writing batch JSONL: {BATCH_INPUT_JSONL}")
    request_count = write_batch_jsonl(df, BATCH_INPUT_JSONL)
    print(f"Batch requests written: {request_count}")

    if request_count == 0:
        print("No usable rows found. Exiting.")
        return

    if RESUME_BATCH_ID:
        print(f"Resuming existing batch: {RESUME_BATCH_ID}")
        batch = client.batches.retrieve(RESUME_BATCH_ID)
    else:
        print("Uploading input file and creating batch...")
        input_file, batch = submit_batch(BATCH_INPUT_JSONL)
        print(f"Input file id: {input_file.id}")
        print(f"Batch id: {batch.id}")
        (WORKDIR / "latest_batch_id.txt").write_text(batch.id, encoding="utf-8")

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