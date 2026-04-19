#!/usr/bin/env python3
"""Populate paper tables and statistics with actual evaluation results.

Reads paper/results/eval_results_claude_judge.json (authoritative — contains
both Ollama and Claude-as-judge scores for all 6 systems) and updates:
  - paper/results_table.md (LaTeX tables with actual numbers)

Run after: make eval && python scripts/claude_judge_pass.py
"""

import json
from pathlib import Path

PAPER_DIR = Path(__file__).parent.parent / "paper"
RESULTS_DIR = PAPER_DIR / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

RESULT_FILES_PRIORITY = [
    "eval_results_claude_judge.json",
    "eval_results_v2.json",
    "eval_results.json",
]

SYSTEMS = [
    ("hybrid_4way", "HippoRAG-4way"),
    ("graphrag",    "GraphRAG"),
    ("baseline",    "Vector-Only"),
    ("bm25",        "BM25"),
    ("ppr_only",    "PPR-Only"),
    ("graph_only",  "Graph-Only"),
]


def load_results():
    for fname in RESULT_FILES_PRIORITY:
        path = RESULTS_DIR / fname
        if path.exists():
            print(f"Loading {path.name}")
            return json.loads(path.read_text()), fname
    print("No eval results found.")
    return None, None


def fmt(v, spec=".3f"):
    if v is None:
        return "--"
    try:
        return f"{v:{spec}}"
    except (TypeError, ValueError):
        return "--"


def main_table(metrics, source):
    lines = [
        "# Results Table (LaTeX-Ready)", "",
        f"*Auto-generated from `paper/results/{source}` by `scripts/populate_paper.py`.*",
        "",
        "## Table 1 — Retrieval Quality (6 systems, 50 queries)", "",
        "```latex",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Retrieval quality across six systems on the 50-query benchmark. "
        r"P@10, R@10, nDCG@10 computed against graph-derived relevance labels. "
        r"Means $\pm$ std. Best per column in \textbf{bold}.}",
        r"\label{tab:retrieval}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{P@10} & \textbf{R@10} & \textbf{nDCG@10} & \textbf{Lat.\ (s)} \\",
        r"\midrule",
    ]
    best = {
        "p": max(metrics[s].get("precision_at_10", 0) for s, _ in SYSTEMS if s in metrics),
        "r": max(metrics[s].get("recall_at_10", 0) for s, _ in SYSTEMS if s in metrics),
        "n": max(metrics[s].get("ndcg_at_10", 0) for s, _ in SYSTEMS if s in metrics),
    }
    for key, label in SYSTEMS:
        m = metrics.get(key, {})
        p = m.get("precision_at_10"); ps = m.get("precision_at_10_std")
        r = m.get("recall_at_10");    rs = m.get("recall_at_10_std")
        n = m.get("ndcg_at_10");      ns = m.get("ndcg_at_10_std")
        lat = m.get("mean_latency_ms", 0) / 1000 if m.get("mean_latency_ms") else None
        def b(val, top):
            s = fmt(val)
            return f"\\textbf{{{s}}}" if val is not None and abs(val - top) < 1e-9 else s
        lines.append(
            f"{label:14s} & ${b(p, best['p'])} \\pm {fmt(ps)}$ "
            f"& ${b(r, best['r'])} \\pm {fmt(rs)}$ "
            f"& ${b(n, best['n'])} \\pm {fmt(ns)}$ "
            f"& {fmt(lat, '.1f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "```"]
    return lines


def judge_table(metrics):
    lines = [
        "", "## Table 2 — Answer Quality (Dual-Judge Evaluation)", "",
        "Faithfulness and answer relevance scored by two independent LLM judges:",
        "Ollama (`qwen2.5:7b`, local) and Claude (`claude-sonnet-4`).",
        "Scores are means $\\pm$ std on [0, 1]. Causal accuracy is scored only on multi-hop queries.",
        "", "```latex",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Answer quality with dual-judge evaluation. "
        r"Ollama and Claude judges are reported independently; agreement between judges "
        r"is discussed in Section~\ref{sec:judge-agreement}.}",
        r"\label{tab:answer-quality}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Ollama Judge}} "
        r"& \multicolumn{2}{c}{\textbf{Claude Judge}} & & \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{System} & \textbf{Faith.} & \textbf{Rel.} "
        r"& \textbf{Faith.} & \textbf{Rel.} & \textbf{Causal} & \textbf{Ctx P.} \\",
        r"\midrule",
    ]
    for key, label in SYSTEMS:
        m = metrics.get(key, {})
        of   = fmt(m.get("faithfulness"));        ofs   = fmt(m.get("faithfulness_std"))
        ore  = fmt(m.get("answer_relevance"));    ores  = fmt(m.get("answer_relevance_std"))
        cf   = fmt(m.get("claude_faithfulness")); cfs   = fmt(m.get("claude_faithfulness_std"))
        cre  = fmt(m.get("claude_relevance"));    cres  = fmt(m.get("claude_relevance_std"))
        causal = fmt(m.get("causal_chain_accuracy"))
        ctxp   = fmt(m.get("context_precision"))
        lines.append(
            f"{label:14s} & ${of} \\pm {ofs}$ & ${ore} \\pm {ores}$ "
            f"& ${cf} \\pm {cfs}$ & ${cre} \\pm {cres}$ "
            f"& {causal} & {ctxp} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "```"]
    return lines


def query_type_table(metrics):
    lines = [
        "", "## Table 3 — Performance by Query Type (Ollama Judge)", "",
        "```latex",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ollama-judged faithfulness and relevance stratified by query type. "
        r"Single-hop (n=20): fact retrieval from one report. Multi-hop (n=20): causal "
        r"reasoning spanning multiple reports. Comparative (n=10): cross-corpus comparison.}",
        r"\label{tab:query-type}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Query Type} & \textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{n} \\",
        r"\midrule",
    ]
    for qt, qt_label, n in [("single_hop", "Single-Hop", 20),
                            ("multi_hop", "Multi-Hop", 20),
                            ("comparative", "Comparative", 10)]:
        first = True
        for key, label in SYSTEMS:
            m = metrics.get(key, {})
            faith = fmt(m.get(f"{qt}_faithfulness"))
            rel = fmt(m.get(f"{qt}_relevance"))
            prefix = f"\\multirow{{6}}{{*}}{{{qt_label}}}" if first else ""
            lines.append(f"{prefix:32s} & {label:14s} & {faith} & {rel} & {n} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", "```"]
    return lines


def pairwise_table(metrics):
    pw = metrics.get("_pairwise", {})
    if not pw:
        return []
    lines = [
        "", "## Table 4 — Pairwise Blind Comparisons", "",
        "Each pair judged by Claude with blinded system labels. Reports win rate of the left system.",
        "", "| Comparison | Left Wins | Right Wins | Ties | Total | Win Rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in pw.items():
        if not isinstance(res, dict):
            continue
        keys = list(res.keys())
        left_wins = next((res[k] for k in keys if k.endswith("_wins") and not k.startswith(("baseline_", "bm25_"))), None)
        wr_key = next((k for k in keys if "win_rate" in k), None)
        if left_wins is None:
            lw_key = [k for k in keys if k.endswith("_wins")][0]
            left_wins = res[lw_key]
        rw_key = next((k for k in keys if k.endswith("_wins") and k != (wr_key or "")
                       and not k.startswith(name.split("_vs_")[0])), None)
        right_wins = res.get(rw_key, "--") if rw_key else "--"
        total = res.get("total", "--")
        ties = res.get("ties", "--")
        wr = res.get(wr_key, 0) if wr_key else None
        wr_s = f"{wr:.2%}" if wr else "--"
        display = name.replace("_vs_", " vs. ")
        lines.append(f"| {display} | {left_wins} | {right_wins} | {ties} | {total} | {wr_s} |")
    return lines


def graph_stats_table():
    lines = ["", "## Graph Statistics", ""]
    # Prefer the evaluation-time graph (clean variant)
    try:
        import pickle
        from pathlib import Path as P
        pkl = P("data/graphs/aerograph_clean.pkl")
        raw_pkl = P("data/graphs/aerograph.pkl")
        info = {}
        if pkl.exists():
            with open(pkl, "rb") as f:
                G = pickle.load(f)
            info["Clean graph — nodes"] = G.number_of_nodes()
            info["Clean graph — edges"] = G.number_of_edges()
        if raw_pkl.exists():
            with open(raw_pkl, "rb") as f:
                Gr = pickle.load(f)
            info["Raw graph — nodes"] = Gr.number_of_nodes()
            info["Raw graph — edges"] = Gr.number_of_edges()
        # Chunk count
        try:
            import chromadb
            client = chromadb.PersistentClient(path="data/chroma_db")
            info["Chunks indexed"] = sum(c.count() for c in client.list_collections())
        except Exception:
            info["Chunks indexed"] = "--"
        # Report count and mean counts
        ext_path = DATA_DIR / "processed" / "extractions.jsonl"
        if ext_path.exists():
            total_ents = total_rels = n = 0
            with open(ext_path) as f:
                for line in f:
                    d = json.loads(line)
                    total_ents += len(d.get("entities", []))
                    total_rels += len(d.get("relations", []))
                    n += 1
            info["Reports ingested"] = n
            info["Mean entities / report"] = f"{total_ents / n:.1f}" if n else "--"
            info["Mean relations / report"] = f"{total_rels / n:.1f}" if n else "--"
        # Community structure
        comm_pkl = P("data/graphs/aerograph_communities.pkl")
        if comm_pkl.exists():
            with open(comm_pkl, "rb") as f:
                Gc = pickle.load(f)
            for level in ("community_L0", "community_L1", "community_L2"):
                comms = {Gc.nodes[n].get(level) for n in Gc.nodes() if level in Gc.nodes[n]}
                comms.discard(None)
                if comms:
                    info[f"Communities ({level.replace('community_', '').upper()})"] = len(comms)
    except Exception as e:
        info = {"error": str(e)}

    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in info.items():
        lines.append(f"| {k} | {v} |")
    return lines


def main():
    data, source = load_results()
    if data is None:
        return
    metrics = data["metrics"]

    lines = []
    lines += main_table(metrics, source)
    lines += judge_table(metrics)
    lines += query_type_table(metrics)
    lines += pairwise_table(metrics)
    lines += graph_stats_table()

    out = PAPER_DIR / "results_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Updated {out}")

    print("\n=== Judge disagreement (Ollama - Claude, faithfulness) ===")
    for key, label in SYSTEMS:
        m = metrics.get(key, {})
        of = m.get("faithfulness")
        cf = m.get("claude_faithfulness")
        if of is not None and cf is not None:
            delta = of - cf
            print(f"  {label:14s}: Ollama {of:.3f} - Claude {cf:.3f} = {delta:+.3f}")


if __name__ == "__main__":
    main()
