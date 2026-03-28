#!/usr/bin/env python3
"""Demo script: run example queries against AeroGraph."""

from aerograph.retrieve import GraphRAGRetriever
from aerograph.generate import generate_answer


DEMO_QUERIES = [
    "What are the most common contributing factors to runway incursions?",
    "How does weather affect go-around decisions during approach phase?",
    "What causal chain links bird strikes to engine failure outcomes?",
    "Compare maintenance-related incidents between B737 and A320 aircraft.",
    "What ATC communication failures have led to altitude deviations?",
]


def main():
    retriever = GraphRAGRetriever()
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")
        result = retriever.retrieve(query)
        answer = generate_answer(query, result)
        print(f"\nAnswer: {answer.text}")
        print(f"Sources: {answer.source_acns}")


if __name__ == "__main__":
    main()
