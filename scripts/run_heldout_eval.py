"""Run the 6-system evaluation on the held-out ASRS corpus.

Assumes AEROGRAPH_DATA_DIR=data_heldout is set so retrievers load the
held-out graph, chroma index, and entity embeddings. Uses the same
50-query benchmark from data/eval_queries.jsonl and writes results to
paper/results/eval_results_heldout.json.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("AEROGRAPH_DATA_DIR", "data_heldout")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aerograph.eval import (  # noqa: E402
    run_evaluation, RESULTS_DIR,
)
from aerograph.retrieve import (  # noqa: E402
    GraphRAGRetriever, BaselineRetriever, BM25Retriever,
    GraphOnlyRetriever, PPROnlyRetriever, HippoRAGRetriever,
)


def main():
    queries_path = Path("data/eval_queries.jsonl")
    output_path = RESULTS_DIR / "eval_results_heldout.json"

    retrievers = {
        "baseline": BaselineRetriever(),
        "bm25": BM25Retriever(),
        "graph_only": GraphOnlyRetriever(),
        "graphrag": GraphRAGRetriever(),
        "ppr_only": PPROnlyRetriever(),
        "hybrid_4way": HippoRAGRetriever(),
    }

    results = run_evaluation(
        queries_path=queries_path,
        output_path=output_path,
        retrievers_dict=retrievers,
    )
    print(f"\nResults written to {output_path}")
    return results


if __name__ == "__main__":
    main()
