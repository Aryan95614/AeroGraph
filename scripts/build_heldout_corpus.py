"""Build held-out ASRS corpus for cross-corpus validation.

Pulls 2,000 ASRS records from elihoole/asrs-aviation-reports that are
disjoint from the primary corpus (data/processed/reports.jsonl) and
writes them to data_heldout/processed/reports.jsonl.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

PRIMARY_PATH = Path("data/processed/reports.jsonl")
HELDOUT_DIR = Path("data_heldout")
HELDOUT_RAW = HELDOUT_DIR / "raw"
HELDOUT_PROC = HELDOUT_DIR / "processed"
TARGET = 2000


def main():
    from datasets import load_dataset
    from import_real_asrs import normalize_record  # type: ignore

    HELDOUT_RAW.mkdir(parents=True, exist_ok=True)
    HELDOUT_PROC.mkdir(parents=True, exist_ok=True)

    used_ids: set[str] = set()
    with open(PRIMARY_PATH) as f:
        for line in f:
            used_ids.add(json.loads(line)["id"])
    print(f"Primary corpus has {len(used_ids)} reports (excluding their IDs)")

    ds = load_dataset("elihoole/asrs-aviation-reports", split="train")
    print(f"HF dataset size: {len(ds)}")

    reports = []
    raw_saved = 0
    raw_fh = open(HELDOUT_RAW / "real_asrs_heldout.jsonl", "w")

    # Skip the first 2000 we likely consumed; start scanning from index 2000
    start = 0
    for i in range(start, len(ds)):
        record = dict(ds[i])
        normalized = normalize_record(record, "heldout")
        if normalized is None:
            continue
        if normalized["id"] in used_ids:
            continue
        if any(r["id"] == normalized["id"] for r in reports):
            continue
        raw_fh.write(json.dumps(record, default=str) + "\n")
        raw_saved += 1
        reports.append(normalized)
        if len(reports) >= TARGET:
            break
        if len(reports) % 500 == 0:
            print(f"  collected {len(reports)}/{TARGET}")
    raw_fh.close()

    out = HELDOUT_PROC / "reports.jsonl"
    with open(out, "w") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(reports)} held-out reports to {out}")
    print(f"Raw kept: {raw_saved} records")


if __name__ == "__main__":
    main()
