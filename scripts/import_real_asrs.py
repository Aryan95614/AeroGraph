#!/usr/bin/env python3
"""Import real ASRS data from HuggingFace datasets.

Replaces synthetic reports with real NASA ASRS incident narratives.
Supports multiple HuggingFace ASRS datasets and normalizes them
into the AeroGraph report format.

Usage:
    pip install datasets
    python scripts/import_real_asrs.py [--max-reports 2000]
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def try_aviation_qa():
    """Try Timilehin674/Aviation_QA — 13,500+ ASRS + NTSB narratives."""
    from datasets import load_dataset
    print("Trying Timilehin674/Aviation_QA...")
    try:
        ds = load_dataset("Timilehin674/Aviation_QA", split="train")
        print(f"  Loaded {len(ds)} records")
        return ds, "aviation_qa"
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def try_asrs_chatgpt():
    """Try archanatikayatray/ASRS-ChatGPT — 9,984 real ASRS records."""
    from datasets import load_dataset
    print("Trying archanatikayatray/ASRS-ChatGPT...")
    try:
        ds = load_dataset("archanatikayatray/ASRS-ChatGPT", split="train")
        print(f"  Loaded {len(ds)} records")
        return ds, "asrs_chatgpt"
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def try_asrs_reports():
    """Try elihoole/asrs-aviation-reports."""
    from datasets import load_dataset
    print("Trying elihoole/asrs-aviation-reports...")
    try:
        ds = load_dataset("elihoole/asrs-aviation-reports", split="train")
        print(f"  Loaded {len(ds)} records")
        return ds, "asrs_reports"
    except Exception as e:
        print(f"  Failed: {e}")
        return None, None


def normalize_record(record: dict, source: str) -> dict | None:
    """Normalize a record from any source into AeroGraph report format."""
    from aerograph.ingest import normalize_aircraft, extract_phase, extract_anomaly, clean_narrative

    # Extract narrative text — combine both reporter narratives if available
    # elihoole/asrs-aviation-reports uses "Report 1_Narrative", "Report 2_Narrative", "Report 1.2_Synopsis"
    parts = []
    for field in ["Report 1_Narrative", "Report 2_Narrative", "Report 1.1_Callback",
                   "narrative", "text", "Report Narrative", "NARRATIVE", "report_narrative",
                   "Synopsis", "synopsis", "Report_Narrative", "Report 1.2_Synopsis", "content"]:
        if field in record and record[field]:
            val = str(record[field]).strip()
            if len(val) > 30:
                parts.append(val)

    text = " ".join(parts) if parts else ""

    if not text or len(text) < 50:
        return None

    text = clean_narrative(text)
    if len(text.split()) < 30:
        return None

    # Extract ID
    report_id = ""
    for field in ["acn_num_ACN", "Person 1.10_ASRS Report Number.Accession Number",
                   "ACN", "acn", "Report Number", "report_number", "id", "ID",
                   "report_id", "Accession Number"]:
        if field in record and record[field]:
            report_id = str(record[field]).strip()
            # Handle float ACNs
            try:
                report_id = str(int(float(report_id)))
            except (ValueError, OverflowError):
                pass
            break
    if not report_id:
        report_id = f"REAL_{hash(text) % 10000000:07d}"

    # Extract metadata — elihoole dataset uses "Aircraft 1.2_Make Model Name"
    aircraft = ""
    for field in ["Aircraft 1.2_Make Model Name", "Aircraft Type", "aircraft_type",
                   "Make Model", "Aircraft", "AC Type", "AcftType"]:
        if field in record and record[field]:
            aircraft = normalize_aircraft(str(record[field]))
            break

    # Use structured phase field if available, fall back to text extraction
    phase = ""
    for field in ["Aircraft 1.9_Flight Phase"]:
        if field in record and record[field]:
            phase = str(record[field]).strip()
            break
    if not phase:
        phase = extract_phase(text)

    # Use structured anomaly field if available
    anomaly = ""
    for field in ["Events_Anomaly"]:
        if field in record and record[field]:
            anomaly = str(record[field]).strip()
            break
    if not anomaly:
        anomaly = extract_anomaly(text)

    return {
        "id": report_id,
        "text": text,
        "metadata": {
            "aircraft_type": aircraft,
            "phase_of_flight": phase,
            "anomaly_type": anomaly,
            "source": f"asrs_real_{source}",
            "is_synthetic": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Import real ASRS data from HuggingFace")
    parser.add_argument("--max-reports", type=int, default=2000,
                        help="Maximum number of reports to import (default: 2000)")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Install datasets first: pip install datasets")
        sys.exit(1)

    # Try datasets in order of quality
    ds, source = try_aviation_qa()
    if ds is None:
        ds, source = try_asrs_chatgpt()
    if ds is None:
        ds, source = try_asrs_reports()
    if ds is None:
        print("\nNo ASRS dataset available. Check your internet connection.")
        sys.exit(1)

    print(f"\nUsing {source} dataset ({len(ds)} raw records)")
    print(f"Normalizing up to {args.max_reports} reports...")

    # Save raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"real_asrs_{source}.jsonl"
    with open(raw_path, "w") as f:
        for i, record in enumerate(ds):
            if i >= args.max_reports * 2:  # fetch extra to account for filtering
                break
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  Saved raw data to {raw_path}")

    # Normalize into report format
    reports = []
    seen_ids = set()
    for record in ds:
        if len(reports) >= args.max_reports:
            break
        normalized = normalize_record(dict(record), source)
        if normalized and normalized["id"] not in seen_ids:
            seen_ids.add(normalized["id"])
            reports.append(normalized)

    print(f"  Normalized {len(reports)} reports (filtered from {len(ds)} raw)")

    # Save to processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "reports.jsonl"

    # Backup existing
    if output_path.exists():
        backup = PROCESSED_DIR / "reports_synthetic_backup.jsonl"
        output_path.rename(backup)
        print(f"  Backed up synthetic reports to {backup}")

    with open(output_path, "w") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone! {len(reports)} real ASRS reports saved to {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Clear old extractions: rm data/processed/extractions.jsonl")
    print(f"  2. Re-run extraction:     python -m aerograph extract")
    print(f"  3. Rebuild pipeline:      python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
