"""Quick exploration of downloaded ASRS reports."""
import json
from pathlib import Path
from collections import Counter

def main():
    reports_path = Path("data/processed/reports.jsonl")
    if not reports_path.exists():
        print("No reports found. Run: python -m aerograph ingest")
        return

    reports = []
    with open(reports_path) as f:
        for line in f:
            reports.append(json.loads(line))

    print(f"Total reports: {len(reports)}")

    phases = Counter(r.get("flight_phase", "unknown") for r in reports)
    print("\nFlight phases:")
    for phase, count in phases.most_common(10):
        print(f"  {phase}: {count}")

    event_types = Counter(r.get("event_type", "unknown") for r in reports)
    print("\nEvent types:")
    for etype, count in event_types.most_common(10):
        print(f"  {etype}: {count}")

if __name__ == "__main__":
    main()
