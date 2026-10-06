"""Append two V-Dem variables at t, t-1, t-2 and current-year regime_binary.

Run from any directory: python3 src/merge_vdem_v7.py
Optional --input, --source, --output and --log paths override repository defaults.
Use --add-regime-binary to append only regime_binary to the existing --output
file, retaining its cells and the existing validation log.
Uses only the standard library. CSV cells are retained as strings, including
empty cells and original numeric formatting. No existing V-Dem values are read
from the source into the output. Missing source values remain missing.

Matching follows python_notebooks/vdem_merge.ipynb and add_press_freedom.ipynb:
country-year lookups, with actual source year = decision year minus lag.
Source keys must be unique; country names and available V-Dem IDs must agree.
"""

import argparse
import csv
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("v2x_regime", "v2x_libdem")
NEW_COLUMNS = [v + (f"_lag{lag}" if lag else "") for v in VARIABLES for lag in range(3)]
DERIVED_COLUMN = "regime_binary"
MISSING = {"", "NA", "NaN", "nan"}
# The notebook's crosswalk, with Turkey aliases resolving to one temporary key.
ALIASES = {
    "united states": "united states of america",
    "myanmar": "burma/myanmar",
    "gambia": "the gambia",
    "moldova, republic of": "moldova",
    "iran, islamic republic of": "iran",
    "korea, republic of": "south korea",
    "russian federation": "russia",
    "republic of north macedonia": "north macedonia",
    "syrian arab republic": "syria",
    "czech republic": "czechia",
    "lao people's democratic republic": "laos",
    "drc": "democratic republic of the congo",
    "suecia": "sweden",
    "venezuela, bolivarian republic of": "venezuela",
    "palestine": "palestine/west bank",
    "palestine, state of": "palestine/west bank",
    "turkey": "türkiye",
    "t√ºrkiye": "türkiye",
}


def country_key(value):
    key = unicodedata.normalize("NFC", value.strip()).casefold()
    return ALIASES.get(key, key)


def integer(value):
    number = Decimal(value)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"Expected integer, got {value!r}")
    return int(number)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(header) != len(set(header)) or any(len(r) != len(header) for r in rows):
        raise ValueError("Duplicate headers or malformed case rows")
    return header, rows


def regime_label(value):
    if value.strip() in MISSING:
        return ""
    try:
        number = Decimal(value)
        if number.is_finite():
            if number in {0, 1}:
                return "autocracy"
            if number in {2, 3}:
                return "democracy"
    except InvalidOperation:
        pass
    raise ValueError(f"Unexpected current-year v2x_regime value: {value!r}")


def append_regime_binary(header, rows, log):
    if DERIVED_COLUMN in header:
        raise ValueError("regime_binary already exists; refusing to overwrite it")
    position = header.index("v2x_regime")
    counts = Counter()
    unexpected = []
    result = []
    for row_index, row in enumerate(rows):
        try:
            label = regime_label(row[position])
        except ValueError:
            unexpected.append({"row_index": row_index, "value": row[position]})
            continue
        counts[label] += 1
        result.append(row + [label])
    log["regime_binary"] = {
        "source_column": "v2x_regime", "time_point": "current year",
        "mapping": {"0": "autocracy", "1": "autocracy", "2": "democracy", "3": "democracy"},
        "counts": {label: counts[label] for label in ["autocracy", "democracy"]},
        "missing_values": counts[""], "unexpected_source_values": unexpected,
        "mapping_valid": not unexpected,
    }
    if unexpected:
        raise ValueError(f"Found {len(unexpected)} unexpected current-year regime values; see log")
    return result


def update_regime_binary(args, log):
    """Change only the existing v7 output, without rereading V-Dem or v6."""
    prior_log = json.loads(args.log.read_text(encoding="utf-8"))
    log.update(prior_log)
    log["status"] = "running"
    header, rows = read_cases(args.output)
    before_hash = sha256(args.output)
    output_rows = append_regime_binary(header, rows, log)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=args.output.parent, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header + [DERIVED_COLUMN])
        writer.writerows(output_rows)
    try:
        saved_header, saved_rows = read_cases(temporary)
        checks = {
            "same_row_count": len(saved_rows) == len(rows),
            "row_order_and_every_existing_cell_unchanged": [r[:-1] for r in saved_rows] == rows,
            "only_regime_binary_appended_as_final_column": saved_header == header + [DERIVED_COLUMN],
            "every_label_matches_current_year_regime": all(
                r[-1] == regime_label(r[header.index("v2x_regime")]) for r in saved_rows),
            "expected_1075_rows_142_existing_143_output_columns":
                len(rows) == 1075 and len(header) == 142 and len(saved_header) == 143,
            "output_unchanged_during_update": sha256(args.output) == before_hash,
        }
        log["regime_binary_update"] = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_output_sha256": before_hash, "existing_columns": len(header),
            "output_columns": len(saved_header), "rows": len(saved_rows), "validation": checks,
        }
        if not all(checks.values()):
            raise ValueError("In-place regime_binary validation failed")
        temporary.replace(args.output)
        log["dimensions"]["output_columns"] = len(saved_header)
        log["added_columns"] = NEW_COLUMNS + [DERIVED_COLUMN]
        log["validation"]["regime_binary_mapping_valid"] = True
        log["output_sha256"] = sha256(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_source(path, log):
    """Stream the full CSV; retain only keys and the two original value strings."""
    by_name, by_id, by_text = {}, {}, {}
    identities = {}
    duplicate_counts = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        required = ["country_name", "country_id", "country_text_id", "year", *VARIABLES]
        if len(header) != len(set(header)):
            raise ValueError("Duplicate source headers")
        indices = [header.index(c) for c in required]
        count = 0
        for row in reader:
            count += 1
            if len(row) != len(header):
                raise ValueError(f"Malformed source row {count}")
            name, cid, text, year, regime, libdem = [row[i] for i in indices]
            cid, year = integer(cid), integer(year)
            text = text.strip()
            record = (name, cid, text, year, regime, libdem)
            identity = (country_key(name), text)
            if cid in identities and identities[cid] != identity:
                raise ValueError(f"Inconsistent source identity for country_id {cid}")
            identities[cid] = identity
            for label, index, key in [
                ("normalized_country_name_year", by_name, (country_key(name), year)),
                ("country_id_year", by_id, (cid, year)),
                ("country_text_id_year", by_text, (text, year)),
            ]:
                if key in index:
                    duplicate_counts[label] += 1
                index[key] = record
    log["source_key_validation"] = {
        "rows": count,
        "countries": len(identities),
        "duplicate_counts": {k: duplicate_counts[k] for k in
                             ["normalized_country_name_year", "country_id_year", "country_text_id_year"]},
        "unique": not any(duplicate_counts.values()),
    }
    if duplicate_counts:
        raise ValueError("Source country-year keys are not unique")
    return by_name, by_id, by_text


def merge(args, log):
    if args.source is None:
        candidates = sorted((ROOT / "data/raw").glob("V-Dem-CY-Full*.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one full V-Dem CSV; use --source. Found {candidates}")
        args.source = candidates[0]
    paths = [p.resolve() for p in (args.input, args.source, args.output, args.log)]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, source, output and log paths must be distinct")
    version = re.search(r"(?:^|[-_])v(\d+(?:\.\d+)*)", args.source.name, re.I)
    input_hash = sha256(args.input)
    log["provenance"] = {
        "input": str(args.input.resolve()), "input_sha256": input_hash,
        "source": str(args.source.resolve()), "source_filename": args.source.name,
        "source_sha256": sha256(args.source),
        "vdem_version": version.group(1) if version else None,
        "version_identification": "source filename" if version else "not identifiable",
        "output": str(args.output.resolve()),
        "convention_references": ["python_notebooks/vdem_merge.ipynb", "python_notebooks/add_press_freedom.ipynb"],
    }
    header, rows = read_cases(args.input)
    if set(NEW_COLUMNS + [DERIVED_COLUMN]) & set(header):
        raise ValueError("Input already contains one or more requested new columns")
    for field in ("country", "year"):
        if field not in header:
            raise ValueError(f"Missing required case column: {field}")
    by_name, by_id, by_text = read_source(args.source, log)
    output_rows, audit, mappings = [], {}, {}
    missing = Counter()
    invalid = {column: [] for column in NEW_COLUMNS}
    id_checks = Counter()
    for row_index, values in enumerate(rows):
        case = dict(zip(header, values))
        year = integer(case["year"])
        name = case.get("country_clean") or case.get("country_name") or case["country"]
        key = country_key(name)
        extras = {}
        for lag in range(3):
            target_year = year - lag
            record = by_name.get((key, target_year))
            audit_key = (key, target_year, lag)
            if audit_key not in audit:
                audit[audit_key] = {
                    "matching_country_key": key, "source_year": target_year, "lag": lag,
                    "matched": record is not None, "case_row_indices_zero_based": [],
                    "source_country_name": record[0] if record else None,
                    "source_country_id": record[1] if record else None,
                    "source_country_text_id": record[2] if record else None,
                }
            audit[audit_key]["case_row_indices_zero_based"].append(row_index)
            # Name, numeric V-Dem ID and V-Dem text ID must resolve to the same row.
            for field, index in [("country_id", by_id), ("country_text_id", by_text)]:
                if case.get(field, "") not in MISSING:
                    identifier = integer(case[field]) if field == "country_id" else case[field].strip()
                    id_record = index.get((identifier, target_year))
                    id_checks[field + "_checks"] += 1
                    if id_record != record:
                        raise ValueError(f"Country name/ID mismatch: row {row_index}, {field}, year {target_year}")
            if record:
                for field in ("country_clean", "country_name", "country"):
                    if case.get(field) and country_key(case[field]) != country_key(record[0]):
                        raise ValueError(f"Conflicting {field} at case row {row_index}")
                mapping = (case["country"], name, record[0], record[1], record[2])
                mappings[mapping] = mappings.get(mapping, 0) + (lag == 0)
            for offset, variable in enumerate(VARIABLES, start=4):
                column = variable + (f"_lag{lag}" if lag else "")
                value = record[offset] if record else ""
                extras[column] = value
                if value.strip() in MISSING:
                    missing[column] += 1
                else:
                    number = Decimal(value)
                    valid = number.is_finite() and (
                        number in {0, 1, 2, 3} if variable == "v2x_regime" else 0 <= number <= 1)
                    if not valid:
                        invalid[column].append({"row_index": row_index, "value": value})
        output_rows.append(values + [extras[column] for column in NEW_COLUMNS])
    log["country_mappings"] = [
        {"original_country": k[0], "original_matching_field": k[1],
         "source_country_name": k[2], "source_country_id": k[3],
         "source_country_text_id": k[4], "case_count": count}
        for k, count in sorted(mappings.items())
    ]
    log["country_id_verification"] = dict(id_checks)
    log["country_year_audit"] = list(audit.values())
    log["unmatched_country_years"] = [entry for entry in audit.values() if not entry["matched"]]
    log["columns"] = {
        c: {"missing_values": missing[c], "nonmissing_values": len(rows) - missing[c],
            "unmatched_case_rows": sum(len(e["case_row_indices_zero_based"]) for e in audit.values()
                                       if not e["matched"] and e["lag"] == (int(c[-1]) if "_lag" in c else 0)),
            "invalid_values": invalid[c]}
        for c in NEW_COLUMNS
    }
    log["time_points"] = {
        str(lag): {"source_year_rule": f"decision year - {lag}",
                   "matched_case_rows": sum(len(e["case_row_indices_zero_based"]) for e in audit.values()
                                            if e["matched"] and e["lag"] == lag)}
        for lag in range(3)
    }
    if any(invalid.values()):
        raise ValueError("New values violate variable domains")
    output_rows = append_regime_binary(header + NEW_COLUMNS, output_rows, log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=args.output.parent, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header + NEW_COLUMNS + [DERIVED_COLUMN])
        writer.writerows(output_rows)
    try:
        saved_header, saved_rows = read_cases(temporary)
        checks = {
            "same_row_count": len(saved_rows) == len(rows),
            "row_order_and_every_original_cell_unchanged": [r[:len(header)] for r in saved_rows] == rows,
            "exactly_six_columns_appended": saved_header[:-1] == header + NEW_COLUMNS,
            "regime_binary_appended_as_final_column": saved_header[-1] == DERIVED_COLUMN,
            "regime_binary_mapping_valid": all(
                r[-1] == regime_label(r[saved_header.index("v2x_regime")]) for r in saved_rows),
            "all_new_cells_equal_source_lookup": saved_rows == output_rows,
            "input_file_unchanged": sha256(args.input) == input_hash,
            "value_domains_valid": not any(invalid.values()),
        }
        log["validation"] = checks
        log["dimensions"] = {"input_rows": len(rows), "output_rows": len(saved_rows),
                             "input_columns": len(header), "output_columns": len(saved_header),
                             "expected_case_count": 1075, "expected_case_count_matches": len(rows) == 1075}
        log["complete_coverage"] = not log["unmatched_country_years"] and not any(missing.values())
        log["warnings"] = []
        if len(rows) != 1075:
            log["warnings"].append(f"Expected 1075 cases; found {len(rows)}")
        if not log["complete_coverage"]:
            log["warnings"].append("Complete coverage was not reproduced; no imputation or year substitution applied")
        if not all(checks.values()):
            raise ValueError("Output validation failed")
        temporary.replace(args.output)
        log["output_sha256"] = sha256(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/cases_v6_short.csv")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/cases_v7_short.csv")
    parser.add_argument("--log", type=Path, default=ROOT / "outputs/vdem_merge/cases_v7_short_validation.json")
    parser.add_argument("--add-regime-binary", action="store_true",
                        help="Append only regime_binary to existing v7 and extend its existing log")
    args = parser.parse_args()
    # Reject unsafe aliases before even attempting to write a failure log.
    if args.log.resolve() in {p.resolve() for p in [args.input, args.output, *([args.source] if args.source else [])]}:
        parser.error("Log path must differ from data paths")
    log = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "added_columns": NEW_COLUMNS + [DERIVED_COLUMN], "missing_value_tokens": sorted(MISSING)}
    try:
        if args.add_regime_binary:
            update_regime_binary(args, log)
        else:
            merge(args, log)
        log["status"] = "passed" if not log["warnings"] else "passed_with_warnings"
    except Exception as error:
        log["status"] = "failed"
        log["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        log["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": log["status"], "output": str(args.output), "log": str(args.log),
                      "dimensions": log["dimensions"], "missing_counts": dict(missing_counts(log)),
                      "regime_binary": log["regime_binary"]}, indent=2))


def missing_counts(log):
    return ((column, details["missing_values"]) for column, details in log["columns"].items())


if __name__ == "__main__":
    main()
