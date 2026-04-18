"""AeroGraph CLI entry point.

Usage:
    python -m aerograph ingest      — Download/generate ASRS reports
    python -m aerograph extract     — Run entity extraction
    python -m aerograph build       — Build knowledge graph from extractions
    python -m aerograph embed       — Build ChromaDB vector index
    python -m aerograph normalize   — Run taxonomy-based entity resolution
    python -m aerograph community   — Detect and summarize graph communities
    python -m aerograph eval        — Run 50-query evaluation suite
    python -m aerograph serve       — Start FastAPI server
    python -m aerograph demo        — Run 5 example queries
    python -m aerograph dashboard   — Launch Streamlit dashboard
    python -m aerograph search      — Interactive CLI search
    python -m aerograph pipeline    — Run full pipeline (extract → build → embed)
    python -m aerograph stats       — Print graph and index statistics
    python -m aerograph --help      — Show this help message
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        from aerograph.ingest import run_ingestion
        run_ingestion()

    elif command == "extract":
        from aerograph.extract import run_extraction
        run_extraction()

    elif command == "build":
        from aerograph.graph import build_graph
        build_graph()

    elif command == "embed":
        from aerograph.embed import build_index
        count = build_index()
        print(f"Indexed {count} chunks")

    elif command == "normalize":
        from aerograph.taxonomy import run_normalize
        stats = run_normalize()

    elif command == "community":
        from aerograph.community import detect_communities, summarize_all_communities, community_stats
        from aerograph.graph import load_graph
        G = load_graph()
        communities = detect_communities(G)
        for level, comms in sorted(communities.items()):
            sizes = [len(m) for m in comms.values()]
            print(f"  Level {level}: {len(comms)} communities, "
                  f"mean size {sum(sizes)/len(sizes):.1f}")
        if "--summarize" in sys.argv:
            import anthropic
            client = anthropic.Anthropic()
            summaries = summarize_all_communities(G, communities, client)
            print(f"Generated {len(summaries)} summaries")

    elif command == "eval":
        from aerograph.eval import run_evaluation, generate_figures
        if "--figures-only" in sys.argv:
            import json
            from pathlib import Path
            from aerograph.eval import EvalResult, RESULTS_DIR
            results_path = RESULTS_DIR / "eval_results.json"
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                results = [EvalResult(**r) for r in data["results"]]
                generate_figures(data["metrics"], results)
            else:
                print("No eval results found. Run full evaluation first.")
        else:
            run_evaluation()

    elif command == "serve":
        import uvicorn
        uvicorn.run("aerograph.api:app", host="0.0.0.0", port=8000, reload=True)

    elif command == "demo":
        from aerograph.retrieve import GraphRAGRetriever
        from aerograph.generate import generate_answer
        queries = [
            "What are the most common contributing factors to runway incursions?",
            "How does weather affect go-around decisions during approach phase?",
            "What causal chain links bird strikes to engine failure outcomes?",
            "Compare maintenance-related incidents between B737 and A320 aircraft.",
            "What ATC communication failures have led to altitude deviations?",
        ]
        retriever = GraphRAGRetriever()
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print(f"{'='*60}")
            result = retriever.retrieve(query)
            answer = generate_answer(query, result)
            print(f"\nAnswer: {answer.text[:500]}...")
            print(f"Sources: {answer.source_acns[:5]}")
            print(f"Latency: {result.latency_ms:.0f}ms retrieval + {answer.latency_ms:.0f}ms generation")

    elif command == "dashboard":
        try:
            import subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "src/aerograph/dashboard.py",
                "--server.port", "8501",
            ])
        except KeyboardInterrupt:
            pass

    elif command == "search":
        from aerograph.retrieve import GraphRAGRetriever
        from aerograph.generate import generate_answer
        retriever = GraphRAGRetriever()
        print("AeroGraph interactive search (Ctrl+C to exit)\n")
        try:
            while True:
                query = input("query> ").strip()
                if not query:
                    continue
                result = retriever.retrieve(query)
                answer = generate_answer(query, result)
                print(f"\n{answer.text}\n")
                print(f"Sources: {', '.join(answer.source_acns[:5])}")
                print(f"Latency: {result.latency_ms:.0f}ms retrieval\n")
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")

    elif command == "pipeline":
        print("=== AeroGraph Full Pipeline ===\n")

        print("[1/4] Running entity extraction...")
        from aerograph.extract import run_extraction
        extractions = run_extraction()
        print(f"  Extracted from {len(extractions)} reports\n")

        print("[2/4] Building knowledge graph from extractions...")
        from aerograph.graph import build_graph
        backend = build_graph()
        print(f"  Graph: {backend.node_count()} nodes, {backend.edge_count()} edges\n")

        print("[3/4] Building vector index...")
        from aerograph.embed import build_index
        count = build_index()
        print(f"  Index: {count} chunks\n")

        print("[4/4] Pipeline complete. Ready for queries.")
        print("  Run: python -m aerograph serve")
        print("  Or:  python -m aerograph demo")

    elif command == "stats":
        _print_stats()

    elif command in ("--help", "-h", "help"):
        print(__doc__)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def _print_stats():
    """Print graph and index statistics."""
    print("=== AeroGraph Statistics ===\n")

    # Graph stats
    try:
        from aerograph.graph import detect_backend
        backend = detect_backend()
        print(f"Knowledge Graph:")
        print(f"  Nodes: {backend.node_count()}")
        print(f"  Edges: {backend.edge_count()}")
        if hasattr(backend, "get_type_counts"):
            for ntype, count in sorted(backend.get_type_counts().items()):
                print(f"    {ntype}: {count}")
        if hasattr(backend, "get_edge_type_counts"):
            print(f"  Edge types:")
            for etype, count in sorted(backend.get_edge_type_counts().items()):
                print(f"    {etype}: {count}")
        top = backend.get_high_centrality_nodes(10)
        if top:
            print(f"  Top entities by PageRank:")
            for name, score in top:
                print(f"    {name}: {score:.6f}")
    except Exception as e:
        print(f"  Graph unavailable: {e}")

    # Index stats
    print()
    try:
        from aerograph.embed import get_chroma_client, COLLECTION_NAME
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Vector Index (ChromaDB):")
        print(f"  Chunks: {collection.count()}")
    except Exception as e:
        print(f"  Index unavailable: {e}")

    # Report stats
    print()
    try:
        from pathlib import Path
        reports_path = Path("data/processed/reports.jsonl")
        if reports_path.exists():
            with open(reports_path) as f:
                n_reports = sum(1 for _ in f)
            print(f"Reports: {n_reports}")
        extractions_path = Path("data/processed/extractions.jsonl")
        if extractions_path.exists():
            with open(extractions_path) as f:
                n_extractions = sum(1 for _ in f)
            print(f"Extractions: {n_extractions}")
    except Exception as e:
        print(f"  Data unavailable: {e}")


if __name__ == "__main__":
    main()
