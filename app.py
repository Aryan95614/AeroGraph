"""AeroGraph — Gradio app for HuggingFace Spaces.

Interactive demo of hybrid retrieval over 2,000 real NASA ASRS aviation safety
incident reports. Query the knowledge graph, compare retrieval systems
side-by-side, and inspect citations.

Runs in two modes:
  - Live mode (ANTHROPIC_API_KEY set): queries go through the full pipeline.
  - Cached mode (no API key): three preset queries return pre-computed answers
    from the benchmark evaluation. Arbitrary queries return a friendly notice.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import gradio as gr

DATA_DIR = Path(__file__).parent / "data"
REPO_URL = "https://github.com/Aryan95614/AeroGraph"
PAPER_URL = "https://github.com/Aryan95614/AeroGraph/blob/main/paper/abstract.md"
DATASET_URL = "https://huggingface.co/datasets/Aryan95614/aerograph-asrs"

CACHED_MODE = not bool(os.getenv("ANTHROPIC_API_KEY"))

_cached_answers: dict | None = None


def _load_cache() -> dict:
    global _cached_answers
    if _cached_answers is None:
        path = DATA_DIR / "cached_demo_answers.json"
        _cached_answers = json.loads(path.read_text()) if path.exists() else {}
    return _cached_answers


# ---------------------------------------------------------------------------
# Lazy singletons — only loaded in live mode
# ---------------------------------------------------------------------------

_graph_backend = None
_retrievers: dict = {}


def _get_graph():
    global _graph_backend
    if _graph_backend is None:
        from aerograph.graph import detect_backend
        _graph_backend = detect_backend()
    return _graph_backend


def _get_retriever(mode: str):
    if mode in _retrievers:
        return _retrievers[mode]
    if mode == "hybrid_4way":
        from aerograph.retrieve import HippoRAGRetriever
        _retrievers[mode] = HippoRAGRetriever()
    elif mode == "baseline":
        from aerograph.retrieve import VectorOnlyRetriever
        _retrievers[mode] = VectorOnlyRetriever()
    else:
        from aerograph.retrieve import GraphRAGRetriever
        _retrievers[mode] = GraphRAGRetriever()
    return _retrievers[mode]


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

PRESET_QUERIES = [
    "What causal chain links bird strikes to engine failure and subsequent go-around decisions?",
    "Compare the contributing factors in B737 vs A320 engine failure incidents.",
    "What weather condition is most commonly associated with go-around decisions?",
]


def _cached_answer(question: str, system: str) -> tuple[str, str, str]:
    cache = _load_cache()
    entry = cache.get(question)
    if not entry:
        msg = (
            "**Cached-demo mode.** This deployment runs without an API key and serves "
            "pre-computed answers for the three preset queries below. "
            f"For custom queries, clone the repo and set `ANTHROPIC_API_KEY`: {REPO_URL}"
        )
        return msg, "", ""
    payload = entry.get(system)
    if not payload:
        return "*No cached answer for this system.*", "", ""
    acns = payload.get("source_acns") or []
    sources_md = "\n".join(f"- ACN **{a}**" for a in acns[:8]) if acns else "*no sources*"
    debug = (
        f"**System:** {system} · **Latency (benchmark run):** {payload['latency_ms']:.0f}ms · "
        f"**Sources cited:** {len(acns)} · **Ollama faith:** {payload.get('faithfulness_ollama')} · "
        f"**Claude faith:** {payload.get('faithfulness_claude')}"
    )
    return payload["answer"], sources_md, debug


def query_aerograph(question: str, top_k: int) -> tuple[str, str, str]:
    if not question.strip():
        return "", "", ""
    if CACHED_MODE:
        return _cached_answer(question.strip(), "hybrid_4way")

    try:
        from aerograph.generate import generate_answer
        retriever = _get_retriever("hybrid_4way")
        t0 = time.time()
        retrieval = retriever.retrieve(question, top_k=int(top_k))
        gen = generate_answer(question, retrieval)
        total_ms = (time.time() - t0) * 1000

        sources_lines = []
        for i, chunk in enumerate(retrieval.chunks, 1):
            preview = chunk.text[:300].replace("\n", " ")
            sources_lines.append(
                f"**{i}. ACN {chunk.report_id}** ({chunk.provenance})\n> {preview}..."
            )
        sources_md = "\n\n".join(sources_lines) if sources_lines else "*No sources retrieved*"

        trace = retrieval.retrieval_trace
        debug = (
            f"**Latency:** {total_ms:.0f}ms · "
            f"**Entities:** {', '.join(trace.get('query_entities', []))} · "
            f"**Vector hits:** {trace.get('vector_results_count', 0)} · "
            f"**Graph-scored reports:** {trace.get('graph_scored_reports', 0)}"
        )
        return gen.text, sources_md, debug
    except Exception as e:
        return f"Error: {e}\n\n```\n{traceback.format_exc()}\n```", "", ""


def compare_systems(question: str) -> tuple[str, str, str, str]:
    """Return (hybrid_answer, hybrid_sources, baseline_answer, baseline_sources)."""
    if not question.strip():
        return "", "", "", ""
    if CACHED_MODE:
        h_ans, h_src, _ = _cached_answer(question.strip(), "hybrid_4way")
        b_ans, b_src, _ = _cached_answer(question.strip(), "baseline")
        return h_ans, h_src, b_ans, b_src

    try:
        from aerograph.generate import generate_answer

        def run(mode: str) -> tuple[str, str]:
            retriever = _get_retriever(mode)
            retrieval = retriever.retrieve(question, top_k=10)
            gen = generate_answer(question, retrieval)
            src = "\n".join(
                f"- ACN **{c.report_id}**" for c in retrieval.chunks[:6]
            ) or "*no sources*"
            return gen.text, src

        h_ans, h_src = run("hybrid_4way")
        b_ans, b_src = run("baseline")
        return h_ans, h_src, b_ans, b_src
    except Exception as e:
        err = f"Error: {e}"
        return err, "", err, ""


# ---------------------------------------------------------------------------
# Entity explorer
# ---------------------------------------------------------------------------

def explore_entity(entity_name: str) -> tuple[str, str]:
    if not entity_name.strip():
        return "", ""
    if CACHED_MODE:
        return (
            "*Entity explorer requires live graph access. Clone the repo to run.*",
            "",
        )
    try:
        backend = _get_graph()
        node = backend.get_node(entity_name.lower().strip())
        if not node:
            return f"Entity **{entity_name}** not in the graph.", ""
        subgraph = backend.get_neighbors(entity_name, depth=1)
        lines = [
            f"## {node.canonical_name}",
            f"**Type:** {node.type} · **Reports:** {len(node.report_ids)} · "
            f"**Neighbors:** {len(subgraph.nodes) - 1} · **Edges:** {len(subgraph.edges)}",
            "",
        ]
        if subgraph.nodes:
            lines.append("### Connected Entities\n")
            lines.append("| Entity | Type | Reports | Relation |")
            lines.append("|---|---|---|---|")
            edge_map: dict[str, list[str]] = {}
            for e in subgraph.edges:
                if e.source == node.canonical_name:
                    edge_map.setdefault(e.target, []).append(f"--[{e.type}]-->")
                elif e.target == node.canonical_name:
                    edge_map.setdefault(e.source, []).append(f"<--[{e.type}]--")
            for n in sorted(subgraph.nodes, key=lambda x: len(x.report_ids), reverse=True):
                if n.canonical_name == node.canonical_name:
                    continue
                rel = ", ".join(edge_map.get(n.canonical_name, ["?"]))
                lines.append(f"| {n.canonical_name} | {n.type} | {len(n.report_ids)} | {rel} |")
        info_md = "\n".join(lines)

        causal_types = {"CAUSED_BY", "CONTRIBUTED_TO", "PRECEDED_BY", "TEMPORAL_SEQUENCE"}
        causal_edges = [e for e in subgraph.edges if e.type in causal_types]
        causal_md = ""
        if causal_edges:
            cl = ["### Causal / Temporal Edges\n"]
            for e in causal_edges[:20]:
                cl.append(f"- `{e.source}` --[**{e.type}**]--> `{e.target}` (w={e.weight})")
            causal_md = "\n".join(cl)
        return info_md, causal_md
    except Exception as e:
        return f"Error: {e}", ""


# ---------------------------------------------------------------------------
# Stats / eval results
# ---------------------------------------------------------------------------

def get_stats() -> str:
    if CACHED_MODE:
        return (
            "| Metric | Value |\n|---|---|\n"
            "| Reports benchmarked | 2,000 |\n"
            "| Raw graph — nodes | 29,244 |\n"
            "| Raw graph — edges | 43,505 |\n"
            "| Canonicalized graph — nodes | 23,948 |\n"
            "| Canonicalized graph — edges | 48,479 |\n"
            "| Chunks indexed | 4,710 |\n"
            "| Entity types | 10 |\n"
            "| Edge types | 8 |\n"
            "| Communities (L0/L1/L2) | 311 / 393 / 296 |\n"
        )
    try:
        backend = _get_graph()
        node_types = backend.get_type_counts() if hasattr(backend, "get_type_counts") else {}
        edge_types = backend.get_edge_type_counts() if hasattr(backend, "get_edge_type_counts") else {}
        top_nodes = backend.get_high_centrality_nodes(top_n=15)
        all_report_ids: set = set()
        if hasattr(backend, "graph"):
            for _, data in backend.graph.nodes(data=True):
                all_report_ids.update(data.get("report_ids", []))
        lines = [
            "## Knowledge Graph\n",
            "| Metric | Value |",
            "|---|---|",
            f"| Total nodes | {backend.node_count():,} |",
            f"| Total edges | {backend.edge_count():,} |",
            f"| Reports indexed | {len(all_report_ids):,} |",
            f"| Entity types | {len(node_types)} |",
            f"| Edge types | {len(edge_types)} |",
            "",
            "### Entity Types\n",
            "| Type | Count |",
            "|---|---|",
        ]
        for t, c in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {t} | {c:,} |")
        if top_nodes:
            lines += ["", "### Top PageRank Entities\n", "| Rank | Entity | Score |", "|---|---|---|"]
            for i, (name, score) in enumerate(top_nodes, 1):
                lines.append(f"| {i} | {name} | {score:.6f} |")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_eval_results() -> str:
    path = DATA_DIR.parent / "paper" / "results" / "eval_results_claude_judge.json"
    if not path.exists():
        return "*Evaluation results not found.*"
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    display = [
        ("hybrid_4way", "HippoRAG 4-way"),
        ("graphrag", "GraphRAG"),
        ("baseline", "Vector-Only"),
        ("bm25", "BM25"),
        ("ppr_only", "PPR-Only"),
        ("graph_only", "Graph-Only"),
    ]
    lines = [
        "## Benchmark — 6 systems × 50 queries\n",
        "| System | P@10 | nDCG@10 | Faith (Ollama) | Faith (Claude) | Latency |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in display:
        m = metrics.get(key, {})
        p = m.get("precision_at_10"); n = m.get("ndcg_at_10")
        of = m.get("faithfulness"); cf = m.get("claude_faithfulness")
        lat = m.get("mean_latency_ms", 0) / 1000 if m.get("mean_latency_ms") else None
        def f(v, spec=".3f"):
            return f"{v:{spec}}" if v is not None else "—"
        lines.append(
            f"| **{label}** | {f(p)} | {f(n)} | {f(of)} | {f(cf)} | "
            f"{f(lat, '.1f')}s |"
        )
    lines += [
        "",
        "### Pairwise blind comparisons (Claude judge)",
        "",
        "| Comparison | Left wins | Right wins | Ties | Win rate |",
        "|---|---|---|---|---|",
    ]
    pw = metrics.get("_pairwise", {})
    for name, res in pw.items():
        if not isinstance(res, dict):
            continue
        wins_keys = [k for k in res if k.endswith("_wins")]
        wr_key = next((k for k in res if "win_rate" in k), None)
        if len(wins_keys) >= 2:
            lw, rw = res.get(wins_keys[0], "?"), res.get(wins_keys[1], "?")
        else:
            lw, rw = res.get(wins_keys[0], "?"), "?"
        wr = res.get(wr_key, 0) if wr_key else 0
        display_name = name.replace("_vs_", " vs. ")
        lines.append(f"| {display_name} | {lw} | {rw} | {res.get('ties', '?')} | {wr:.0%} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EXAMPLE_ENTITIES = [
    "bird strike", "engine failure", "go-around", "turbulence",
    "b737", "runway incursion", "wind shear", "pilot deviation",
    "tcas", "atc", "icing", "autopilot",
]


def build_app() -> gr.Blocks:
    mode_banner = (
        "**Cached-demo mode** · Three preset queries return benchmark-run answers. "
        f"Clone the [repo]({REPO_URL}) and set `ANTHROPIC_API_KEY` for live queries."
        if CACHED_MODE else
        "**Live mode** · Queries go through the full hybrid retrieval pipeline."
    )

    with gr.Blocks(title="AeroGraph — Hybrid Retrieval for Aviation Safety") as app:
        gr.Markdown(
            f"""# AeroGraph
### Hybrid knowledge-graph + vector retrieval over 2,000 NASA ASRS incident reports

{mode_banner}

[Repo]({REPO_URL}) · [Paper]({PAPER_URL}) · [Dataset]({DATASET_URL})
"""
        )

        with gr.Tabs():
            with gr.Tab("Query"):
                gr.Markdown("Ask a question. Hybrid retrieval fuses vector, BM25, PPR, and community-summary signals.")
                with gr.Row():
                    with gr.Column(scale=3):
                        q_input = gr.Textbox(
                            label="Question",
                            placeholder=PRESET_QUERIES[0],
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k = gr.Slider(3, 20, value=10, step=1, label="Top-K")
                        q_btn = gr.Button("Ask AeroGraph", variant="primary", size="lg")

                gr.Markdown("**Preset demo queries (work in cached mode):**")
                with gr.Row():
                    for pq in PRESET_QUERIES:
                        gr.Button(pq[:60] + ("…" if len(pq) > 60 else ""), size="sm").click(
                            lambda q=pq: q, outputs=q_input
                        )

                debug_output = gr.Markdown(label="Retrieval info")
                answer_output = gr.Markdown(label="Answer")
                with gr.Accordion("Source citations (ACN)", open=True):
                    sources_output = gr.Markdown()

                q_btn.click(query_aerograph, [q_input, top_k],
                            [answer_output, sources_output, debug_output])
                q_input.submit(query_aerograph, [q_input, top_k],
                               [answer_output, sources_output, debug_output])

            with gr.Tab("Compare — Hybrid vs Vector-Only"):
                gr.Markdown(
                    "Same query, two retrieval stacks side-by-side. "
                    "Hybrid fuses 4 signals; vector is the dense-only baseline."
                )
                cmp_input = gr.Textbox(
                    label="Question",
                    placeholder=PRESET_QUERIES[0],
                    lines=2,
                )
                with gr.Row():
                    for pq in PRESET_QUERIES:
                        gr.Button(pq[:50] + ("…" if len(pq) > 50 else ""), size="sm").click(
                            lambda q=pq: q, outputs=cmp_input
                        )
                cmp_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Hybrid (4-signal RRF)")
                        h_ans = gr.Markdown()
                        h_src = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### Vector-Only baseline")
                        b_ans = gr.Markdown()
                        b_src = gr.Markdown()
                cmp_btn.click(compare_systems, [cmp_input], [h_ans, h_src, b_ans, b_src])

            with gr.Tab("Graph Explorer"):
                gr.Markdown("Explore entity neighborhoods in the knowledge graph.")
                with gr.Row():
                    entity_input = gr.Textbox(
                        label="Entity", placeholder="e.g., bird strike, engine failure, b737",
                    )
                    explore_btn = gr.Button("Explore", variant="primary")
                gr.Examples(examples=[[e] for e in EXAMPLE_ENTITIES], inputs=[entity_input])
                entity_info = gr.Markdown()
                causal_info = gr.Markdown()
                explore_btn.click(explore_entity, [entity_input], [entity_info, causal_info])
                entity_input.submit(explore_entity, [entity_input], [entity_info, causal_info])

            with gr.Tab("Benchmark Results"):
                with gr.Row():
                    with gr.Column():
                        stats_md = gr.Markdown()
                        gr.Button("Load graph stats", variant="secondary").click(
                            get_stats, outputs=stats_md)
                    with gr.Column():
                        eval_md = gr.Markdown()
                        gr.Button("Load benchmark", variant="secondary").click(
                            get_eval_results, outputs=eval_md)

            with gr.Tab("About"):
                gr.Markdown(f"""
## What this is

A working implementation of hybrid retrieval over a domain knowledge graph,
benchmarked across six retrieval systems with statistical significance testing.
Built on 2,000 real NASA ASRS aviation incident reports.

## Stack

- **Extraction:** Claude Sonnet with structured JSON output, aviation ontology
  (10 entity types, 8 edge types), taxonomy normalization (HFACS, ICAO)
- **Graph:** NetworkX DiGraph, Leiden community detection, Personalized PageRank
- **Vector:** ChromaDB + sentence-transformers/all-MiniLM-L6-v2
- **Sparse:** from-scratch BM25
- **Fusion:** Reciprocal Rank Fusion across 4 signals (vector, BM25, PPR, community)
- **Generation:** Claude Sonnet with source ACN citations
- **API:** FastAPI · **Tests:** 135 pytest cases

## Benchmark (50 queries × 6 systems)

Best single retriever: **Personalized PageRank** at P@10 = 0.200, nDCG@10 = 0.221.
Best hybrid: **HippoRAG 4-way RRF** at P@10 = 0.210.
Best answer quality (pairwise blind preference): **hybrid systems win 62–72%** vs
the vector-only baseline under a Claude judge.

Under a local Ollama judge, hybrid faithfulness reaches 0.685; under a stricter
Claude judge it drops to 0.310 — a consistent 0.2–0.4 gap across systems.
The ranking mostly holds; absolute numbers are judge-sensitive.

## Source

Repo: {REPO_URL}
""")

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
