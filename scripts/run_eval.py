#!/usr/bin/env python3
"""Run the AeroGraph evaluation suite."""

from aerograph.eval import run_evaluation, generate_eval_queries
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    # Ensure eval queries exist
    queries_path = DATA_DIR / "eval_queries.jsonl"
    if not queries_path.exists():
        print("Generating evaluation queries...")
        generate_eval_queries(queries_path)

    print("Running evaluation suite...")
    metrics = run_evaluation(queries_path)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for system, vals in metrics.items():
        print(f"\n{system.upper()}:")
        for k, v in vals.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
