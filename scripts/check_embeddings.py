"""Spot-check embedding quality by looking at nearest neighbors."""
import numpy as np
from aerograph.embed import get_chroma_client, COLLECTION_NAME

def check():
    client = get_chroma_client()
    col = client.get_collection(COLLECTION_NAME)
    print(f"Collection size: {col.count()}")

    test_queries = [
        "engine failure during takeoff",
        "pilot fatigue night operations",
        "runway incursion taxiway confusion",
    ]
    for q in test_queries:
        results = col.query(query_texts=[q], n_results=5)
        print(f"\nQuery: {q}")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"  [{i+1}] dist={dist:.4f} | {doc[:100]}...")

if __name__ == "__main__":
    check()
