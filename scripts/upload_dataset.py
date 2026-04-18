#!/usr/bin/env python3
"""Upload AeroGraph ASRS dataset to HuggingFace Hub.

Uploads both the processed reports and entity/relation extractions
as a structured HuggingFace dataset.

Usage:
    pip install huggingface_hub datasets
    huggingface-cli login
    python scripts/upload_dataset.py [--repo-id AryanDhawan/aerograph-asrs]
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_reports(path: Path) -> list[dict]:
    """Load reports from JSONL."""
    reports = []
    with open(path) as f:
        for line in f:
            reports.append(json.loads(line))
    return reports


def load_extractions(path: Path) -> list[dict]:
    """Load extractions from JSONL."""
    extractions = []
    with open(path) as f:
        for line in f:
            extractions.append(json.loads(line))
    return extractions


def main():
    parser = argparse.ArgumentParser(description="Upload AeroGraph dataset to HuggingFace Hub")
    parser.add_argument("--repo-id", default="AryanDhawan/aerograph-asrs",
                        help="HuggingFace repo ID (default: AryanDhawan/aerograph-asrs)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    args = parser.parse_args()

    try:
        from datasets import Dataset, DatasetDict, Features, Value, Sequence
        from huggingface_hub import HfApi
    except ImportError:
        print("Install dependencies: pip install datasets huggingface_hub")
        sys.exit(1)

    # --- Load data ---
    reports_path = PROCESSED_DIR / "reports.jsonl"
    extractions_path = PROCESSED_DIR / "extractions.jsonl"

    if not reports_path.exists():
        print(f"Reports file not found: {reports_path}")
        sys.exit(1)

    print("Loading data...")
    reports = load_reports(reports_path)
    print(f"  {len(reports)} reports")

    extractions = []
    if extractions_path.exists():
        extractions = load_extractions(extractions_path)
        print(f"  {len(extractions)} extractions")

    # --- Build reports dataset ---
    report_records = []
    for r in reports:
        meta = r.get("metadata", {})
        report_records.append({
            "id": r["id"],
            "text": r["text"],
            "aircraft_type": meta.get("aircraft_type", ""),
            "phase_of_flight": meta.get("phase_of_flight", ""),
            "anomaly_type": meta.get("anomaly_type", ""),
            "source": meta.get("source", ""),
            "is_synthetic": meta.get("is_synthetic", meta.get("synthetic", False)),
        })

    # --- Build extractions dataset ---
    extraction_records = []
    for ex in extractions:
        entities = ex.get("entities", [])
        relations = ex.get("relations", [])
        extraction_records.append({
            "report_id": ex["report_id"],
            "num_entities": len(entities),
            "num_relations": len(relations),
            "entity_names": [e["name"] for e in entities],
            "entity_types": [e["type"] for e in entities],
            "entity_canonical_names": [e.get("canonical_name", e["name"].lower()) for e in entities],
            "relation_sources": [r["source"] for r in relations],
            "relation_targets": [r["target"] for r in relations],
            "relation_types": [r["type"] for r in relations],
        })

    # --- Create HF datasets ---
    print("\nCreating HuggingFace datasets...")

    reports_ds = Dataset.from_list(report_records)
    print(f"  Reports dataset: {reports_ds}")

    splits = {"reports": reports_ds}

    if extraction_records:
        extractions_ds = Dataset.from_list(extraction_records)
        print(f"  Extractions dataset: {extractions_ds}")
        splits["extractions"] = extractions_ds

    dataset_dict = DatasetDict(splits)

    # --- Upload ---
    print(f"\nUploading to {args.repo_id}...")
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message="Upload AeroGraph ASRS dataset: 2000 real NASA reports + entity/relation extractions",
    )

    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{args.repo_id}")

    # --- Create dataset card ---
    card_content = f"""---
dataset_info:
  - config_name: reports
    features:
      - name: id
        dtype: string
      - name: text
        dtype: string
      - name: aircraft_type
        dtype: string
      - name: phase_of_flight
        dtype: string
      - name: anomaly_type
        dtype: string
      - name: source
        dtype: string
      - name: synthetic
        dtype: bool
    num_examples: {len(report_records)}
  - config_name: extractions
    features:
      - name: report_id
        dtype: string
      - name: num_entities
        dtype: int32
      - name: num_relations
        dtype: int32
      - name: entity_names
        sequence: string
      - name: entity_types
        sequence: string
      - name: relation_types
        sequence: string
    num_examples: {len(extraction_records)}
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - aviation
  - safety
  - graphrag
  - knowledge-graph
  - asrs
  - nasa
size_categories:
  - 1K<n<10K
---

# AeroGraph ASRS Dataset

2,000 real NASA Aviation Safety Reporting System (ASRS) incident reports
with LLM-extracted entities and relations for knowledge graph construction.

## Dataset Description

This dataset contains processed ASRS incident narratives along with
structured entity and relation extractions conforming to an aviation
safety ontology (10 entity types, 8 edge types).

### Reports Split
- **{len(report_records)} reports** from the NASA ASRS database
- Fields: id, text, aircraft_type, phase_of_flight, anomaly_type
- Average narrative length: ~250 words

### Extractions Split
- **{len(extraction_records)} extraction results** (one per report)
- **{sum(e['num_entities'] for e in extraction_records):,} total entities** extracted
- **{sum(e['num_relations'] for e in extraction_records):,} total relations** extracted
- Entity types: Aircraft, Event, Phase, Factor, Component, Outcome,
  Recommendation, ATC_Facility, Weather, TimePeriod
- Edge types: CAUSED_BY, CONTRIBUTED_TO, OCCURRED_DURING, INVOLVED,
  RESOLVED_BY, PRECEDED_BY, CO_OCCURRED_WITH, TEMPORAL_SEQUENCE

## Usage

```python
from datasets import load_dataset

# Load reports
reports = load_dataset("{args.repo_id}", "reports", split="reports")

# Load extractions
extractions = load_dataset("{args.repo_id}", "extractions", split="extractions")
```

## Citation

```bibtex
@misc{{dhawan2026aerograph,
  title={{AeroGraph: Graph-Augmented Retrieval for Multi-Hop Causal
         Reasoning over Aviation Safety Reports}},
  author={{Dhawan, Aryan}},
  year={{2026}},
  url={{https://github.com/AryanDhawan/AeroGraph}}
}}
```

## License

MIT
"""

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print("Dataset card uploaded.")


if __name__ == "__main__":
    main()
