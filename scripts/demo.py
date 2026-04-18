#!/usr/bin/env python3
"""AeroGraph demonstration script.

Showcases the full GraphRAG pipeline:
  1. Graph statistics
  2. Entity exploration
  3. Five example queries with GraphRAG retrieval
  4. Side-by-side comparison: GraphRAG vs vector-only
"""

import time


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def show_graph_stats():
    """Display knowledge graph statistics."""
    print_header("KNOWLEDGE GRAPH OVERVIEW")

    from aerograph.graph import detect_backend
    backend = detect_backend()

    print(f"  Nodes: {backend.node_count():,}")
    print(f"  Edges: {backend.edge_count():,}")

    if hasattr(backend, "get_type_counts"):
        print(f"\n  Entity type distribution:")
        for ntype, count in sorted(backend.get_type_counts().items(), key=lambda x: x[1], reverse=True):
            bar = "█" * min(count // 5, 40)
            print(f"    {ntype:20s} {count:5d}  {bar}")

    if hasattr(backend, "get_edge_type_counts"):
        print(f"\n  Relation type distribution:")
        for etype, count in sorted(backend.get_edge_type_counts().items(), key=lambda x: x[1], reverse=True):
            bar = "█" * min(count // 5, 40)
            print(f"    {etype:20s} {count:5d}  {bar}")

    top = backend.get_high_centrality_nodes(10)
    if top:
        print(f"\n  Top entities by PageRank:")
        for name, score in top:
            node = backend.get_node(name)
            ntype = node.type if node else "?"
            n_reports = len(node.report_ids) if node else 0
            print(f"    {name:30s} ({ntype:12s}) PR={score:.6f}  reports={n_reports}")


def show_entity_exploration():
    """Explore a few entity neighborhoods."""
    print_header("ENTITY NEIGHBORHOOD EXPLORATION")

    from aerograph.graph import detect_backend
    backend = detect_backend()

    entities_to_explore = ["bird strike", "engine failure", "go-around", "b737", "turbulence"]

    for entity in entities_to_explore:
        node = backend.get_node(entity)
        if not node:
            continue

        subgraph = backend.get_neighbors(entity, depth=1)
        neighbors = [n for n in subgraph.nodes if n.canonical_name != entity]

        print(f"  {entity} ({node.type}, {len(node.report_ids)} reports)")
        if neighbors:
            for n in neighbors[:5]:
                print(f"    -> {n.canonical_name} ({n.type})")
            if len(neighbors) > 5:
                print(f"    ... and {len(neighbors)-5} more")
        print()


def run_demo_queries():
    """Run 5 example queries through GraphRAG."""
    print_header("GRAPHRAG QUERY DEMONSTRATION")

    from aerograph.retrieve import GraphRAGRetriever
    from aerograph.generate import generate_answer

    queries = [
        ("Single-hop", "What aircraft type is most frequently involved in bird strike incidents?"),
        ("Multi-hop", "What causal chain links bird strikes to engine failure and subsequent go-around decisions?"),
        ("Multi-hop", "How does crew fatigue contribute to communication failures that lead to altitude deviations?"),
        ("Comparative", "Compare the contributing factors in B737 vs A320 engine failure incidents."),
        ("Single-hop", "What ATC communication failures have led to near midair collisions?"),
    ]

    retriever = GraphRAGRetriever()

    for i, (qtype, query) in enumerate(queries, 1):
        print(f"  [{i}/5] ({qtype}) {query}")
        print(f"  {'-'*66}")

        start = time.time()
        result = retriever.retrieve(query, top_k=5)
        answer = generate_answer(query, result)
        total_ms = (time.time() - start) * 1000

        # Provenance breakdown
        prov_counts = {}
        for c in result.chunks:
            prov_counts[c.provenance] = prov_counts.get(c.provenance, 0) + 1
        prov_str = ", ".join(f"{k}={v}" for k, v in prov_counts.items())

        print(f"  Entities: {result.query_entities}")
        print(f"  Sources: {answer.source_acns[:5]}")
        print(f"  Provenance: {prov_str}")
        print(f"  Latency: {total_ms:.0f}ms")
        print()
        # Show first ~300 chars of answer
        preview = answer.text[:400].replace("\n", "\n  ")
        print(f"  {preview}...")
        print()


def run_comparison():
    """Side-by-side: GraphRAG vs vector-only on a multi-hop query."""
    print_header("GRAPHRAG vs VECTOR-ONLY COMPARISON")

    from aerograph.retrieve import GraphRAGRetriever, BaselineRetriever
    from aerograph.generate import generate_answer

    query = "What sequence of factors connects weather deterioration to runway incursion events?"
    print(f"  Query: {query}\n")

    for sys_name, retriever in [("GraphRAG", GraphRAGRetriever()), ("Vector-Only", BaselineRetriever())]:
        result = retriever.retrieve(query, top_k=5)
        answer = generate_answer(query, result)

        prov_counts = {}
        for c in result.chunks:
            prov_counts[c.provenance] = prov_counts.get(c.provenance, 0) + 1

        unique_reports = len({c.report_id for c in result.chunks})

        print(f"  --- {sys_name} ---")
        print(f"  Reports retrieved: {unique_reports}")
        print(f"  Provenance: {prov_counts}")
        print(f"  Latency: {result.latency_ms:.0f}ms retrieval + {answer.latency_ms:.0f}ms generation")
        print(f"  Answer preview: {answer.text[:250]}...")
        print()


def main():
    print("\n" + "=" * 70)
    print("  AEROGRAPH DEMO")
    print("  GraphRAG over Aviation Safety Incident Reports")
    print("=" * 70)

    show_graph_stats()
    show_entity_exploration()
    run_demo_queries()
    run_comparison()

    print_header("DEMO COMPLETE")
    print("  To explore interactively:")
    print("    make serve      # FastAPI at http://localhost:8000")
    print("    make dashboard  # Streamlit at http://localhost:8501")
    print()


if __name__ == "__main__":
    main()
