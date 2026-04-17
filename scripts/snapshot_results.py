#!/usr/bin/env python3
"""Read eval checkpoint and produce full paper-ready results without waiting for completion.

Usage:
    python scripts/snapshot_results.py          # reads checkpoint, prints tables
    python scripts/snapshot_results.py --save   # also writes eval_results.json + figures
"""

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

PAPER_DIR = Path(__file__).parent.parent / "paper"
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"
CHECKPOINT_PATH = RESULTS_DIR / "eval_checkpoint.json"
FINAL_PATH = RESULTS_DIR / "eval_results.json"


def load_checkpoint():
    path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else FINAL_PATH
    if not path.exists():
        print("No checkpoint or results found.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    completed = data.get("completed_query_ids", [])
    print(f"Source: {path.name}")
    print(f"Completed queries: {len(completed)}/50")
    print(f"Total result rows: {len(results)}")
    return results, completed


def safe_mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def safe_std(vals):
    if len(vals) < 2:
        return 0.0
    m = safe_mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def ci95(vals):
    if len(vals) < 2:
        return 0.0
    return 1.96 * safe_std(vals) / math.sqrt(len(vals))


def aggregate(results):
    systems = defaultdict(list)
    for r in results:
        systems[r["system"]].append(r)

    metrics = {}
    for name, rows in systems.items():
        faith = [r["faithfulness"] for r in rows]
        rel = [r["answer_relevance"] for r in rows]
        p10 = [r["precision_at_10"] for r in rows]
        r10 = [r["recall_at_10"] for r in rows]
        ndcg = [r["ndcg_at_10"] for r in rows]
        causal = [r["causal_chain_accuracy"] for r in rows if r["causal_chain_accuracy"] >= 0]
        lat = [r["latency_ms"] for r in rows]

        metrics[name] = {
            "precision_at_10": safe_mean(p10),
            "precision_at_10_std": safe_std(p10),
            "precision_at_10_ci95": ci95(p10),
            "recall_at_10": safe_mean(r10),
            "recall_at_10_std": safe_std(r10),
            "ndcg_at_10": safe_mean(ndcg),
            "ndcg_at_10_std": safe_std(ndcg),
            "ndcg_at_10_ci95": ci95(ndcg),
            "faithfulness": safe_mean(faith),
            "faithfulness_std": safe_std(faith),
            "faithfulness_ci95": ci95(faith),
            "answer_relevance": safe_mean(rel),
            "answer_relevance_std": safe_std(rel),
            "answer_relevance_ci95": ci95(rel),
            "causal_chain_accuracy": safe_mean(causal),
            "mean_latency_ms": safe_mean(lat),
            "n_queries": len(rows),
        }

        for qt in ["single_hop", "multi_hop", "comparative"]:
            qt_rows = [r for r in rows if r["query_type"] == qt]
            if qt_rows:
                metrics[name][f"{qt}_faithfulness"] = safe_mean([r["faithfulness"] for r in qt_rows])
                metrics[name][f"{qt}_relevance"] = safe_mean([r["answer_relevance"] for r in qt_rows])
                metrics[name][f"{qt}_n"] = len(qt_rows)

    return metrics


def print_tables(metrics):
    order = ["hybrid_4way", "graphrag", "baseline", "bm25", "ppr_only", "graph_only"]
    labels = {
        "hybrid_4way": "HippoRAG (4-way)",
        "graphrag": "GraphRAG",
        "baseline": "Vector-Only",
        "bm25": "BM25",
        "ppr_only": "PPR Only",
        "graph_only": "Graph-Only",
    }

    print("\n" + "=" * 70)
    print("TIER 1: RETRIEVAL QUALITY")
    print("=" * 70)
    print(f"{'System':20s} {'P@10':>8s} {'R@10':>8s} {'nDCG@10':>8s}")
    print("-" * 46)
    for key in order:
        if key not in metrics:
            continue
        m = metrics[key]
        print(f"{labels[key]:20s} {m['precision_at_10']:8.3f} {m['recall_at_10']:8.3f} {m['ndcg_at_10']:8.3f}")

    print("\n" + "=" * 70)
    print("TIER 2: ANSWER QUALITY (Llama 3.1 8B judge, strict rubric)")
    print("=" * 70)
    print(f"{'System':20s} {'Faith':>10s} {'Relev':>10s} {'Causal':>8s} {'Lat(s)':>8s}")
    print("-" * 58)
    for key in order:
        if key not in metrics:
            continue
        m = metrics[key]
        f_str = f"{m['faithfulness']:.3f}±{m['faithfulness_std']:.3f}"
        r_str = f"{m['answer_relevance']:.3f}±{m['answer_relevance_std']:.3f}"
        c_str = f"{m['causal_chain_accuracy']:.3f}"
        l_str = f"{m['mean_latency_ms']/1000:.1f}"
        print(f"{labels[key]:20s} {f_str:>10s} {r_str:>10s} {c_str:>8s} {l_str:>8s}")

    # Per query type
    print("\n" + "=" * 70)
    print("PER QUERY TYPE BREAKDOWN")
    print("=" * 70)
    for qt, qt_label in [("single_hop", "Single-Hop"), ("multi_hop", "Multi-Hop"), ("comparative", "Comparative")]:
        print(f"\n  {qt_label}:")
        print(f"  {'System':20s} {'Faith':>8s} {'Relev':>8s} {'n':>5s}")
        print(f"  {'-'*43}")
        for key in order:
            if key not in metrics:
                continue
            m = metrics[key]
            f = m.get(f"{qt}_faithfulness", 0)
            r = m.get(f"{qt}_relevance", 0)
            n = m.get(f"{qt}_n", 0)
            print(f"  {labels[key]:20s} {f:8.3f} {r:8.3f} {n:5d}")

    # Delta table (vs baseline)
    print("\n" + "=" * 70)
    print("DELTAS vs VECTOR-ONLY BASELINE")
    print("=" * 70)
    if "baseline" in metrics:
        base = metrics["baseline"]
        print(f"{'System':20s} {'ΔP@10':>8s} {'ΔnDCG':>8s} {'ΔFaith':>8s} {'ΔRelev':>8s}")
        print("-" * 46)
        for key in order:
            if key not in metrics or key == "baseline":
                continue
            m = metrics[key]
            dp = m["precision_at_10"] - base["precision_at_10"]
            dn = m["ndcg_at_10"] - base["ndcg_at_10"]
            df = m["faithfulness"] - base["faithfulness"]
            dr = m["answer_relevance"] - base["answer_relevance"]
            print(f"{labels[key]:20s} {dp:+8.3f} {dn:+8.3f} {df:+8.3f} {dr:+8.3f}")


def print_significance(results):
    """Paired Wilcoxon signed-rank tests: each graph system vs baseline."""
    from collections import defaultdict
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("\n(scipy not installed — skipping significance tests)")
        return

    order = ["hybrid_4way", "graphrag", "ppr_only"]
    labels = {"hybrid_4way": "HippoRAG", "graphrag": "GraphRAG", "ppr_only": "PPR Only"}

    # Group scores by (query_id, system)
    by_qs = defaultdict(dict)
    for r in results:
        by_qs[r["query_id"]][r["system"]] = r

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank vs Vector-Only)")
    print("=" * 70)
    print(f"{'System':20s} {'Metric':>10s} {'p-value':>10s} {'Sig?':>6s}")
    print("-" * 48)

    for sys in order:
        for metric in ["precision_at_10", "ndcg_at_10", "faithfulness"]:
            pairs_sys = []
            pairs_base = []
            for qid, systems in by_qs.items():
                if sys in systems and "baseline" in systems:
                    pairs_sys.append(systems[sys][metric])
                    pairs_base.append(systems["baseline"][metric])

            if len(pairs_sys) < 10:
                continue

            diffs = [a - b for a, b in zip(pairs_sys, pairs_base)]
            # Skip if all diffs are zero
            if all(d == 0 for d in diffs):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(pairs_sys, pairs_base, alternative="greater")
                except ValueError:
                    p = 1.0

            metric_short = metric.replace("precision_at_10", "P@10").replace("ndcg_at_10", "nDCG").replace("faithfulness", "Faith")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{labels[sys]:20s} {metric_short:>10s} {p:10.4f} {sig:>6s}")


def print_multihop_table(results):
    """Dedicated multi-hop results table — the key paper finding."""
    from collections import defaultdict

    order = ["ppr_only", "hybrid_4way", "graphrag", "baseline", "bm25", "graph_only"]
    labels = {
        "hybrid_4way": "HippoRAG (4-way)", "graphrag": "GraphRAG",
        "baseline": "Vector-Only", "bm25": "BM25",
        "ppr_only": "PPR Only", "graph_only": "Graph-Only",
    }

    mh = defaultdict(lambda: {"p10": [], "ndcg": [], "faith": [], "rel": []})
    for r in results:
        if r["query_type"] == "multi_hop":
            mh[r["system"]]["p10"].append(r["precision_at_10"])
            mh[r["system"]]["ndcg"].append(r["ndcg_at_10"])
            mh[r["system"]]["faith"].append(r["faithfulness"])
            mh[r["system"]]["rel"].append(r["answer_relevance"])

    base_p = safe_mean(mh["baseline"]["p10"])
    base_n = safe_mean(mh["baseline"]["ndcg"])

    print("\n" + "=" * 70)
    print("MULTI-HOP QUERIES ONLY (n=20) — PRIMARY PAPER RESULT")
    print("=" * 70)
    print(f"{'System':20s} {'P@10':>7s} {'Δ%':>7s} {'nDCG':>7s} {'Δ%':>7s} {'Faith':>7s} {'Rel':>7s}")
    print("-" * 63)
    for key in order:
        if key not in mh:
            continue
        s = mh[key]
        p = safe_mean(s["p10"])
        n = safe_mean(s["ndcg"])
        f = safe_mean(s["faith"])
        r = safe_mean(s["rel"])
        dp = ((p - base_p) / base_p * 100) if base_p > 0 else 0
        dn = ((n - base_n) / base_n * 100) if base_n > 0 else 0
        print(f"{labels[key]:20s} {p:7.3f} {dp:+6.0f}% {n:7.3f} {dn:+6.0f}% {f:7.3f} {r:7.3f}")


def save_results(metrics, results):
    out = {"metrics": metrics, "results": results}
    out_path = RESULTS_DIR / "eval_results_snapshot.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved snapshot to {out_path}")


def main():
    results, completed = load_checkpoint()
    metrics = aggregate(results)
    print_tables(metrics)
    print_multihop_table(results)
    print_significance(results)

    if "--save" in sys.argv:
        save_results(metrics, results)


if __name__ == "__main__":
    main()
