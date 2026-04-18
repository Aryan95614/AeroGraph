"""AeroGraph — Gradio app for HuggingFace Spaces.

Interactive demo of GraphRAG over 2,000 real NASA ASRS aviation safety
incident reports. Query the knowledge graph, explore entities, and
compare retrieval systems.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import gradio as gr

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first use
# ---------------------------------------------------------------------------

_graph_backend = None
_graphrag_retriever = None


def _get_graph():
    global _graph_backend
    if _graph_backend is None:
        from aerograph.graph import detect_backend
        _graph_backend = detect_backend()
    return _graph_backend


def _get_retriever():
    global _graphrag_retriever
    if _graphrag_retriever is None:
        from aerograph.retrieve import GraphRAGRetriever
        _graphrag_retriever = GraphRAGRetriever()
    return _graphrag_retriever


# ---------------------------------------------------------------------------
# Query tab
# ---------------------------------------------------------------------------

def query_aerograph(question: str, top_k: int) -> tuple[str, str, str]:
    """Run a GraphRAG query and return (answer, sources, debug info)."""
    if not question.strip():
        return "", "", ""

    try:
        from aerograph.retrieve import GraphRAGRetriever
        from aerograph.generate import generate_answer

        retriever = _get_retriever()
        t0 = time.time()
        retrieval_result = retriever.retrieve(question, top_k=int(top_k))
        gen_result = generate_answer(question, retrieval_result)
        total_ms = (time.time() - t0) * 1000

        # Format sources
        sources_lines = []
        for i, chunk in enumerate(retrieval_result.chunks, 1):
            preview = chunk.text[:300].replace("\n", " ")
            sources_lines.append(
                f"**{i}. ACN {chunk.report_id}** ({chunk.provenance})\n"
                f"> {preview}..."
            )
        sources_md = "\n\n".join(sources_lines) if sources_lines else "*No sources retrieved*"

        # Debug info
        trace = retrieval_result.retrieval_trace
        debug = (
            f"**Latency:** {total_ms:.0f}ms | "
            f"**Entities detected:** {', '.join(trace.get('query_entities', []))} | "
            f"**Vector hits:** {trace.get('vector_results_count', 0)} | "
            f"**Graph-scored reports:** {trace.get('graph_scored_reports', 0)}"
        )

        return gen_result.text, sources_md, debug

    except Exception as e:
        return f"Error: {e}\n\n```\n{traceback.format_exc()}\n```", "", ""


# ---------------------------------------------------------------------------
# Entity explorer tab
# ---------------------------------------------------------------------------

def explore_entity(entity_name: str) -> tuple[str, str]:
    """Explore an entity's neighborhood in the knowledge graph."""
    if not entity_name.strip():
        return "", ""

    try:
        backend = _get_graph()
        node = backend.get_node(entity_name.lower().strip())
        if not node:
            return f"Entity **{entity_name}** not found in the knowledge graph.", ""

        subgraph = backend.get_neighbors(entity_name, depth=1)

        # Entity info
        info_lines = [
            f"## {node.canonical_name}",
            f"**Type:** {node.type}",
            f"**Reports:** {len(node.report_ids)}",
            f"**Neighbors:** {len(subgraph.nodes) - 1}",
            f"**Edges:** {len(subgraph.edges)}",
            "",
        ]

        # Neighbor table
        if subgraph.nodes:
            info_lines.append("### Connected Entities\n")
            info_lines.append("| Entity | Type | Reports | Relation |")
            info_lines.append("|--------|------|---------|----------|")

            # Build edge lookup
            edge_map: dict[str, list[str]] = {}
            for e in subgraph.edges:
                if e.source == node.canonical_name:
                    edge_map.setdefault(e.target, []).append(f"--[{e.type}]-->")
                elif e.target == node.canonical_name:
                    edge_map.setdefault(e.source, []).append(f"<--[{e.type}]--")

            for n in sorted(subgraph.nodes, key=lambda x: len(x.report_ids), reverse=True):
                if n.canonical_name == node.canonical_name:
                    continue
                relations = ", ".join(edge_map.get(n.canonical_name, ["?"]))
                info_lines.append(
                    f"| {n.canonical_name} | {n.type} | {len(n.report_ids)} | {relations} |"
                )

        info_md = "\n".join(info_lines)

        # Causal chains (if any edges are causal)
        causal_lines = []
        causal_types = {"CAUSED_BY", "CONTRIBUTED_TO", "PRECEDED_BY", "TEMPORAL_SEQUENCE"}
        causal_edges = [e for e in subgraph.edges if e.type in causal_types]
        if causal_edges:
            causal_lines.append("### Causal / Temporal Edges\n")
            for e in causal_edges[:20]:
                causal_lines.append(f"- `{e.source}` --[**{e.type}**]--> `{e.target}` (weight: {e.weight})")

        causal_md = "\n".join(causal_lines) if causal_lines else ""

        return info_md, causal_md

    except Exception as e:
        return f"Error: {e}", ""


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------

def get_stats() -> str:
    """Return knowledge graph statistics as markdown."""
    try:
        backend = _get_graph()
        node_types = backend.get_type_counts() if hasattr(backend, "get_type_counts") else {}
        edge_types = backend.get_edge_type_counts() if hasattr(backend, "get_edge_type_counts") else {}
        top_nodes = backend.get_high_centrality_nodes(top_n=15)

        # Count unique reports
        all_report_ids = set()
        if hasattr(backend, "graph"):
            for _, data in backend.graph.nodes(data=True):
                all_report_ids.update(data.get("report_ids", []))

        lines = [
            "## Knowledge Graph Overview\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Total Nodes** | {backend.node_count():,} |",
            f"| **Total Edges** | {backend.edge_count():,} |",
            f"| **Reports Indexed** | {len(all_report_ids):,} |",
            f"| **Entity Types** | {len(node_types)} |",
            f"| **Edge Types** | {len(edge_types)} |",
            "",
            "### Entity Types\n",
            "| Type | Count |",
            "|------|-------|",
        ]
        for t, c in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {t} | {c:,} |")

        lines.extend([
            "",
            "### Edge Types\n",
            "| Type | Count |",
            "|------|-------|",
        ])
        for t, c in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {t} | {c:,} |")

        if top_nodes:
            lines.extend([
                "",
                "### Highest PageRank Entities\n",
                "| Rank | Entity | Score |",
                "|------|--------|-------|",
            ])
            for i, (name, score) in enumerate(top_nodes, 1):
                lines.append(f"| {i} | {name} | {score:.6f} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading stats: {e}"


def get_eval_results() -> str:
    """Load and format evaluation results."""
    eval_path = DATA_DIR.parent / "paper" / "results" / "eval_results.json"
    if not eval_path.exists():
        return "*Evaluation results not found.*"

    with open(eval_path) as f:
        data = json.load(f)

    lines = [
        "## 4-System Ablation Results (50-query benchmark)\n",
        "| System | Faithfulness | Relevance | Causal Acc | Latency |",
        "|--------|-------------|-----------|------------|---------|",
    ]

    system_names = {
        "graphrag": "GraphRAG",
        "baseline": "Vector-Only",
        "bm25": "BM25",
        "graph_only": "Graph-Only",
    }

    for key, display in system_names.items():
        m = data["metrics"].get(key, {})
        faith = m.get("faithfulness", 0)
        rel = m.get("answer_relevance", 0)
        causal = m.get("causal_chain_accuracy", 0)
        lat = m.get("mean_latency_ms", 0) / 1000
        lines.append(f"| **{display}** | {faith:.3f} | {rel:.3f} | {causal:.3f} | {lat:.1f}s |")

    lines.extend([
        "",
        "### Per Query Type — Faithfulness\n",
        "| System | Single-Hop | Multi-Hop | Comparative |",
        "|--------|-----------|-----------|-------------|",
    ])

    for key, display in system_names.items():
        m = data["metrics"].get(key, {})
        sh = m.get("single_hop_faithfulness", 0)
        mh = m.get("multi_hop_faithfulness", 0)
        comp = m.get("comparative_faithfulness", 0)
        lines.append(f"| **{display}** | {sh:.3f} | {mh:.3f} | {comp:.3f} |")

    lines.extend([
        "",
        "### Per Query Type — Relevance\n",
        "| System | Single-Hop | Multi-Hop | Comparative |",
        "|--------|-----------|-----------|-------------|",
    ])

    for key, display in system_names.items():
        m = data["metrics"].get(key, {})
        sh = m.get("single_hop_relevance", 0)
        mh = m.get("multi_hop_relevance", 0)
        comp = m.get("comparative_relevance", 0)
        lines.append(f"| **{display}** | {sh:.3f} | {mh:.3f} | {comp:.3f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What are the most common contributing factors in bird strike incidents?",
    "Trace the causal chain from engine failure to go-around decisions",
    "Compare contributing factors in B737 vs A320 turbulence encounters",
    "What role does ATC communication play in runway incursion incidents?",
    "What weather conditions most frequently contribute to approach-phase incidents?",
    "How do pilot fatigue factors relate to altitude deviation events?",
    "What is the relationship between icing conditions and autopilot disconnects?",
    "Describe the sequence of events in TCAS resolution advisory incidents",
]

EXAMPLE_ENTITIES = [
    "bird strike", "engine failure", "go-around", "turbulence",
    "b737", "runway incursion", "wind shear", "pilot deviation",
    "tcas", "atc", "icing", "autopilot",
]


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="AeroGraph — GraphRAG for Aviation Safety",
    ) as app:
        gr.Markdown(
            """
            # AeroGraph
            ### Graph-Augmented Retrieval for Aviation Safety Analysis

            Query a knowledge graph built from **2,000 real NASA ASRS incident reports**
            containing **29,000+ entities** and **43,000+ relations** across 10 entity types.

            Built with Claude Sonnet for extraction + generation, ChromaDB for vector search,
            NetworkX for graph traversal, and Reciprocal Rank Fusion for hybrid retrieval.

            [[Paper]](https://github.com/AryanDhawan/AeroGraph) | [[GitHub]](https://github.com/AryanDhawan/AeroGraph) | [[Dataset]](https://huggingface.co/datasets/AryanDhawan/aerograph-asrs)
            """,
            elem_classes=["header-text"],
        )

        with gr.Tabs():
            # ---- Query Tab ----
            with gr.Tab("Query", id="query"):
                gr.Markdown("Ask questions about aviation safety incidents. GraphRAG combines vector search with knowledge graph traversal for multi-hop causal reasoning.")

                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Question",
                            placeholder="e.g., What causal chain links bird strikes to engine failure?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k = gr.Slider(
                            minimum=3, maximum=20, value=10, step=1,
                            label="Top-K chunks",
                        )
                        query_btn = gr.Button("Ask AeroGraph", variant="primary", size="lg")

                gr.Examples(
                    examples=[[q, 10] for q in EXAMPLE_QUERIES],
                    inputs=[query_input, top_k],
                    label="Example Queries",
                )

                debug_output = gr.Markdown(label="Retrieval Info")
                answer_output = gr.Markdown(label="Answer")

                with gr.Accordion("Source Evidence", open=False):
                    sources_output = gr.Markdown()

                query_btn.click(
                    fn=query_aerograph,
                    inputs=[query_input, top_k],
                    outputs=[answer_output, sources_output, debug_output],
                )
                query_input.submit(
                    fn=query_aerograph,
                    inputs=[query_input, top_k],
                    outputs=[answer_output, sources_output, debug_output],
                )

            # ---- Entity Explorer Tab ----
            with gr.Tab("Graph Explorer", id="explorer"):
                gr.Markdown("Explore entity neighborhoods in the aviation safety knowledge graph.")

                with gr.Row():
                    entity_input = gr.Textbox(
                        label="Entity Name",
                        placeholder="e.g., bird strike, engine failure, b737",
                    )
                    explore_btn = gr.Button("Explore", variant="primary")

                gr.Examples(
                    examples=[[e] for e in EXAMPLE_ENTITIES],
                    inputs=[entity_input],
                    label="Example Entities",
                )

                entity_info = gr.Markdown(label="Entity Neighborhood")
                causal_info = gr.Markdown(label="Causal/Temporal Edges")

                explore_btn.click(
                    fn=explore_entity,
                    inputs=[entity_input],
                    outputs=[entity_info, causal_info],
                )
                entity_input.submit(
                    fn=explore_entity,
                    inputs=[entity_input],
                    outputs=[entity_info, causal_info],
                )

            # ---- Stats Tab ----
            with gr.Tab("Statistics", id="stats"):
                with gr.Row():
                    with gr.Column():
                        stats_output = gr.Markdown()
                        stats_btn = gr.Button("Load Graph Stats", variant="secondary")
                        stats_btn.click(fn=get_stats, outputs=[stats_output])
                    with gr.Column():
                        eval_output = gr.Markdown()
                        eval_btn = gr.Button("Load Eval Results", variant="secondary")
                        eval_btn.click(fn=get_eval_results, outputs=[eval_output])

            # ---- About Tab ----
            with gr.Tab("About", id="about"):
                gr.Markdown("""
## Architecture

```
ASRS Reports (2,000 real NASA reports)
    |
    v
[Entity Extraction] ── Claude Sonnet ── 43K entities, 39K relations
    |
    v
[Knowledge Graph] ── NetworkX DiGraph ── 29K nodes, 43K edges
    |                                         |
    v                                         v
[ChromaDB Index]                    [Graph Traversal]
  4,710 chunks                      2-hop BFS expansion
  all-MiniLM-L6-v2                  PageRank centrality
    |                                         |
    +────── Reciprocal Rank Fusion ───────────+
                      |
                      v
              [Claude Generation]
              Grounded answers with
              source ACN citations
```

## Key Results (50-query benchmark)

| System | Faithfulness | Relevance | Causal Acc |
|--------|-------------|-----------|------------|
| **GraphRAG** | **0.390** | **0.854** | 0.550 |
| Vector-Only | 0.306 | 0.850 | 0.550 |
| BM25 | 0.230 | 0.775 | 0.562 |
| Graph-Only | 0.528 | 0.598 | 0.537 |

**GraphRAG achieves +27% faithfulness over vector-only retrieval** while
maintaining top relevance. On comparative queries specifically, GraphRAG
achieves 0.61 faithfulness vs 0.34 for the vector-only baseline (+79%).

## Ontology

- **10 Entity Types:** Aircraft, Event, Phase, Factor, Component, Outcome,
  Recommendation, ATC_Facility, Weather, TimePeriod
- **8 Edge Types:** CAUSED_BY, CONTRIBUTED_TO, OCCURRED_DURING, INVOLVED,
  RESOLVED_BY, PRECEDED_BY, CO_OCCURRED_WITH, TEMPORAL_SEQUENCE

## Citation

```bibtex
@misc{dhawan2026aerograph,
  title={AeroGraph: Graph-Augmented Retrieval for Multi-Hop Causal
         Reasoning over Aviation Safety Reports},
  author={Dhawan, Aryan},
  year={2026},
  url={https://github.com/AryanDhawan/AeroGraph}
}
```
                """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False,
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))
