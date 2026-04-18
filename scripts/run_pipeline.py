#!/usr/bin/env python3
"""Run the full post-extraction pipeline.

Assumes extraction is complete (data/processed/extractions.jsonl exists).
Runs: graph build → embedding index → evaluation → paper figures.
"""

import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    start = time.time()

    # Check prereqs
    extractions = DATA_DIR / "processed" / "extractions.jsonl"
    reports = DATA_DIR / "processed" / "reports.jsonl"

    if not extractions.exists():
        print("ERROR: No extractions found. Run extraction first:")
        print("  python -m aerograph extract")
        sys.exit(1)

    n_extractions = sum(1 for _ in open(extractions))
    n_reports = sum(1 for _ in open(reports)) if reports.exists() else 0
    print(f"Found {n_extractions} extractions from {n_reports} reports\n")

    # Step 1: Build knowledge graph
    print("=" * 60)
    print("[1/4] Building knowledge graph...")
    print("=" * 60)
    from aerograph.graph import build_graph
    backend = build_graph()
    print(f"  Nodes: {backend.node_count()}")
    print(f"  Edges: {backend.edge_count()}")
    print()

    # Step 2: Build embedding index
    print("=" * 60)
    print("[2/4] Building vector index...")
    print("=" * 60)
    from aerograph.embed import build_index
    chunk_count = build_index()
    print(f"  Chunks indexed: {chunk_count}")
    print()

    # Step 3: Run evaluation
    print("=" * 60)
    print("[3/4] Running 4-system evaluation (this takes a while)...")
    print("=" * 60)
    from aerograph.eval import run_evaluation
    metrics = run_evaluation()
    print()

    # Step 4: Populate paper
    print("=" * 60)
    print("[4/4] Generating paper artifacts...")
    print("=" * 60)
    try:
        # Re-generate figures with actual data
        from aerograph.eval import generate_figures, EvalResult, RESULTS_DIR
        import json
        results_path = RESULTS_DIR / "eval_results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            results = [EvalResult(**r) for r in data["results"]]
            generate_figures(data["metrics"], results)
    except Exception as e:
        print(f"  Figure generation: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")

    # Print summary
    print("\nRESULTS SUMMARY:")
    for sys_name in ["graphrag", "baseline", "bm25", "graph_only"]:
        m = metrics.get(sys_name, {})
        if not m:
            continue
        print(f"\n  {sys_name.upper()}:")
        print(f"    Faithfulness:  {m.get('faithfulness', 0):.3f} +/- {m.get('faithfulness_ci95', 0):.3f}")
        print(f"    Relevance:     {m.get('answer_relevance', 0):.3f} +/- {m.get('answer_relevance_ci95', 0):.3f}")
        print(f"    Causal Acc:    {m.get('causal_chain_accuracy', 0):.3f}")
        print(f"    Latency:       {m.get('mean_latency_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
