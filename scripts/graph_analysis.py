"""Analyze graph structure and connectivity patterns."""
import networkx as nx
from pathlib import Path
from collections import Counter

def load_and_analyze():
    graph_path = Path("data/processed/graph.graphml")
    if not graph_path.exists():
        print("No graph found. Run: python -m aerograph build")
        return

    G = nx.read_graphml(graph_path)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")

    degrees = dict(G.degree())
    top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop 20 nodes by degree:")
    for node, deg in top:
        print(f"  {node}: {deg}")

    types = Counter(nx.get_node_attributes(G, "type").values())
    print("\nNode types:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")

if __name__ == "__main__":
    load_and_analyze()
