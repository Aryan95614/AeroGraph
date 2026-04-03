"""Benchmark retrieval latency across different index sizes."""
import time
from aerograph.retrieve import GraphRAGRetriever

QUERIES = [
    "runway incursion causes",
    "bird strike engine failure",
    "weather go-around decision",
    "ATC communication altitude deviation",
    "maintenance B737 incidents",
]

def bench():
    retriever = GraphRAGRetriever()
    for q in QUERIES:
        start = time.perf_counter()
        result = retriever.retrieve(q)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{elapsed:7.1f}ms | {len(result.chunks):2d} chunks | {q}")

if __name__ == "__main__":
    bench()
