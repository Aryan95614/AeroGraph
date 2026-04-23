"""Generate cached demo answers for 10 showcase queries.

Writes data/demo_cache.jsonl with lines:
  {"question": str, "answer": str, "sources": [acn, ...], "chunks": [text, ...]}

Runs once with a live ANTHROPIC_API_KEY. app.py uses this as the offline
fallback so the HF Space works without secrets.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SHOWCASE = [
    "What are the most common contributing factors in bird strike incidents?",
    "Trace the causal chain from engine failure to go-around decisions",
    "Compare contributing factors in B737 vs A320 turbulence encounters",
    "What role does ATC communication play in runway incursion incidents?",
    "What weather conditions most frequently contribute to approach-phase incidents?",
    "How do pilot fatigue factors relate to altitude deviation events?",
    "What is the relationship between icing conditions and autopilot disconnects?",
    "Describe the sequence of events in TCAS resolution advisory incidents",
    "What causal factors connect thunderstorm encounters to turbulence injuries?",
    "How does crew fatigue contribute to communication failures leading to altitude deviations?",
]

OUT = Path("data/demo_cache.jsonl")


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY missing; cannot build live demo cache.")
        sys.exit(1)

    from aerograph.retrieve import GraphRAGRetriever
    from aerograph.generate import generate_answer

    retriever = GraphRAGRetriever()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as out:
        for i, q in enumerate(SHOWCASE, 1):
            print(f"[{i}/{len(SHOWCASE)}] {q[:60]}...")
            result = retriever.retrieve(q, top_k=10)
            gen = generate_answer(q, result)
            chunks = [c.text[:600] for c in result.chunks[:8]]
            entry = {
                "question": q,
                "answer": gen.text,
                "sources": gen.source_acns[:8],
                "chunks": chunks,
                "retrieval_latency_ms": result.latency_ms,
                "generation_latency_ms": gen.latency_ms,
            }
            out.write(json.dumps(entry) + "\n")
            out.flush()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
