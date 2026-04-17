#!/usr/bin/env python3
"""Extract compelling qualitative examples where graph retrieval outperforms baseline.

Finds multi-hop queries with the largest P@10 gap between graph systems and baseline,
then prints side-by-side answer excerpts for paper inclusion.

Usage:
    python scripts/extract_examples.py
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "paper" / "results"


def main():
    path = RESULTS_DIR / "eval_results.json"
    if not path.exists():
        print("No results found. Run eval first.")
        return

    with open(path) as f:
        data = json.load(f)

    results = data["results"]

    # Group by query
    by_query = defaultdict(dict)
    for r in results:
        by_query[r["query_id"]][r["system"]] = r

    # Find multi-hop queries where graph systems beat baseline the most
    gaps = []
    for qid, systems in by_query.items():
        if "baseline" not in systems:
            continue
        base_r = systems["baseline"]
        if base_r["query_type"] != "multi_hop":
            continue

        for graph_sys in ["hybrid_4way", "graphrag", "ppr_only"]:
            if graph_sys not in systems:
                continue
            graph_r = systems[graph_sys]
            p10_gap = graph_r["precision_at_10"] - base_r["precision_at_10"]
            faith_gap = graph_r["faithfulness"] - base_r["faithfulness"]
            gaps.append({
                "query_id": qid,
                "system": graph_sys,
                "p10_gap": p10_gap,
                "faith_gap": faith_gap,
                "graph_p10": graph_r["precision_at_10"],
                "base_p10": base_r["precision_at_10"],
                "graph_faith": graph_r["faithfulness"],
                "base_faith": base_r["faithfulness"],
                "graph_answer": graph_r["answer"],
                "base_answer": base_r["answer"],
                "graph_acns": graph_r["source_acns"],
                "base_acns": base_r["source_acns"],
            })

    # Sort by combined gap
    gaps.sort(key=lambda x: x["p10_gap"] + x["faith_gap"], reverse=True)

    print("=" * 80)
    print("TOP QUALITATIVE EXAMPLES: Graph > Baseline on Multi-Hop")
    print("=" * 80)

    for i, g in enumerate(gaps[:5]):
        print(f"\n{'='*80}")
        print(f"Example {i+1}: {g['query_id']} ({g['system']})")
        print(f"P@10: {g['system']}={g['graph_p10']:.2f} vs baseline={g['base_p10']:.2f} (Δ={g['p10_gap']:+.2f})")
        print(f"Faith: {g['system']}={g['graph_faith']:.2f} vs baseline={g['base_faith']:.2f} (Δ={g['faith_gap']:+.2f})")
        print(f"Sources: {g['system']}={len(g['graph_acns'])} ACNs, baseline={len(g['base_acns'])} ACNs")

        # Print query
        q = by_query[g["query_id"]].get("baseline", {})
        print(f"\nQuery: (see eval_queries.jsonl for {g['query_id']})")

        print(f"\n--- {g['system'].upper()} ANSWER (first 500 chars) ---")
        print(g["graph_answer"][:500])

        print(f"\n--- BASELINE ANSWER (first 500 chars) ---")
        print(g["base_answer"][:500])

    # Also find worst cases — where graph hurts
    gaps.sort(key=lambda x: x["p10_gap"] + x["faith_gap"])
    print(f"\n\n{'='*80}")
    print("WORST CASES: Graph < Baseline (for honest limitations section)")
    print("=" * 80)

    for i, g in enumerate(gaps[:3]):
        print(f"\n{g['query_id']} ({g['system']}): P@10 Δ={g['p10_gap']:+.2f}, Faith Δ={g['faith_gap']:+.2f}")

    # Summary stats
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    wins = sum(1 for g in gaps if g["p10_gap"] > 0)
    losses = sum(1 for g in gaps if g["p10_gap"] < 0)
    ties = sum(1 for g in gaps if g["p10_gap"] == 0)
    print(f"Graph beats baseline: {wins}/{len(gaps)} ({wins/len(gaps)*100:.0f}%)")
    print(f"Baseline beats graph: {losses}/{len(gaps)} ({losses/len(gaps)*100:.0f}%)")
    print(f"Ties: {ties}/{len(gaps)} ({ties/len(gaps)*100:.0f}%)")


if __name__ == "__main__":
    main()
