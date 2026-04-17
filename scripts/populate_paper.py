#!/usr/bin/env python3
"""Populate paper tables and statistics with actual evaluation results.

Reads paper/results/eval_results.json and updates:
  - paper/results_table.md (LaTeX tables with actual numbers)

Run after: make eval
"""

import json
from pathlib import Path

PAPER_DIR = Path(__file__).parent.parent / "paper"
RESULTS_DIR = PAPER_DIR / "results"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_results():
    path = RESULTS_DIR / "eval_results.json"
    if not path.exists():
        print(f"No results found at {path}. Run 'make eval' first.")
        return None
    with open(path) as f:
        return json.load(f)


def format_val(v, fmt=".2f"):
    if v is None:
        return "--"
    return f"{v:{fmt}}"


def generate_results_table(data):
    metrics = data["metrics"]

    systems = [
        ("graphrag", "GraphRAG"),
        ("baseline", "Vector-Only"),
        ("bm25", "BM25"),
        ("graph_only", "Graph-Only"),
    ]

    # Main results table
    lines = ["# Results Table (LaTeX-Ready)", "", "## Main Results — 4-System Ablation", "", "```latex"]
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study: four retrieval systems on 50-query benchmark.")
    lines.append(r"Scores are means $\pm$ std across all queries. Faithfulness and relevance")
    lines.append(r"scored by Claude-as-judge (0--1). Causal chain accuracy scored only for")
    lines.append(r"multi-hop queries. Reference similarity uses ROUGE-L F1.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{Ctx P.} & \textbf{Ctx R.} & \textbf{Causal} & \textbf{Ref. Sim.} & \textbf{Lat. (s)} \\")
    lines.append(r"\midrule")

    for sys_key, label in systems:
        m = metrics.get(sys_key, {})
        faith = format_val(m.get("faithfulness"))
        faith_std = format_val(m.get("faithfulness_std"))
        rel = format_val(m.get("answer_relevance"))
        rel_std = format_val(m.get("answer_relevance_std"))
        prec = format_val(m.get("context_precision"))
        rec = format_val(m.get("context_recall"))
        causal = format_val(m.get("causal_chain_accuracy"))
        ref = format_val(m.get("reference_similarity"))
        lat = format_val(m.get("mean_latency_ms", 0) / 1000, ".1f")

        lines.append(
            f"{label:16s} & ${faith} \\pm {faith_std}$ & ${rel} \\pm {rel_std}$ "
            f"& {prec} & {rec} & {causal} & {ref} & {lat} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")

    # Per query type breakdown
    lines.extend(["", "## Per Query Type Breakdown", "", "```latex"])
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Faithfulness and answer relevance by query type.}")
    lines.append(r"\label{tab:query-type}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Query Type} & \textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{n} \\")
    lines.append(r"\midrule")

    query_types = [("single_hop", "Single-Hop", 20), ("multi_hop", "Multi-Hop", 20), ("comparative", "Comparative", 10)]

    for qt, qt_label, n in query_types:
        first = True
        for sys_key, label in systems:
            m = metrics.get(sys_key, {})
            faith = format_val(m.get(f"{qt}_faithfulness"))
            rel = format_val(m.get(f"{qt}_relevance"))
            prefix = f"\\multirow{{4}}{{*}}{{{qt_label}}}" if first else "  "
            lines.append(f"{prefix:30s} & {label:16s} & {faith} & {rel} & {n} \\\\")
            first = False
        lines.append(r"\midrule")

    # Remove last midrule, add bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")

    # Graph statistics
    lines.extend(["", "## Graph Statistics", ""])

    try:
        from aerograph.graph import detect_backend
        backend = detect_backend()
        node_count = backend.node_count()
        edge_count = backend.edge_count()
        type_counts = backend.get_type_counts() if hasattr(backend, "get_type_counts") else {}
        n_types = len(type_counts)
        edge_type_counts = backend.get_edge_type_counts() if hasattr(backend, "get_edge_type_counts") else {}
        n_edge_types = len(edge_type_counts)
    except Exception:
        node_count = edge_count = n_types = n_edge_types = "--"

    try:
        from aerograph.embed import get_chroma_client, COLLECTION_NAME
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        chunk_count = collection.count()
    except Exception:
        chunk_count = "--"

    # Compute mean entities/relations per report
    try:
        import json as json_mod
        ext_path = DATA_DIR / "processed" / "extractions.jsonl"
        total_ents = 0
        total_rels = 0
        n_reports = 0
        with open(ext_path) as f:
            for line in f:
                d = json_mod.loads(line)
                total_ents += len(d.get("entities", []))
                total_rels += len(d.get("relations", []))
                n_reports += 1
        mean_ents = f"{total_ents / n_reports:.1f}" if n_reports > 0 else "--"
        mean_rels = f"{total_rels / n_reports:.1f}" if n_reports > 0 else "--"
    except Exception:
        mean_ents = mean_rels = "--"
        n_reports = "--"

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total nodes | {node_count} |")
    lines.append(f"| Total edges | {edge_count} |")
    lines.append(f"| Entity types | {n_types} |")
    lines.append(f"| Edge types | {n_edge_types} |")
    lines.append(f"| Reports indexed | {n_reports} |")
    lines.append(f"| Chunks indexed | {chunk_count} |")
    lines.append(f"| Mean entities/report | {mean_ents} |")
    lines.append(f"| Mean relations/report | {mean_rels} |")
    lines.append("")
    lines.append("*Auto-populated by `scripts/populate_paper.py` from evaluation results.*")

    return "\n".join(lines)


def main():
    data = load_results()
    if data is None:
        return

    content = generate_results_table(data)
    out_path = PAPER_DIR / "results_table.md"
    with open(out_path, "w") as f:
        f.write(content + "\n")
    print(f"Updated {out_path}")

    # Print summary
    metrics = data["metrics"]
    print("\n=== Summary ===")
    for sys_name in ["graphrag", "baseline", "bm25", "graph_only"]:
        m = metrics.get(sys_name, {})
        print(f"\n{sys_name.upper()}:")
        print(f"  Faithfulness:  {m.get('faithfulness', 0):.3f} +/- {m.get('faithfulness_std', 0):.3f}")
        print(f"  Relevance:     {m.get('answer_relevance', 0):.3f} +/- {m.get('answer_relevance_std', 0):.3f}")
        print(f"  Causal Acc:    {m.get('causal_chain_accuracy', 0):.3f}")
        print(f"  Latency:       {m.get('mean_latency_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
