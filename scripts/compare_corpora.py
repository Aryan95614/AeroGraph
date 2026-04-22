"""Compare primary vs held-out ASRS evaluation results.

Loads paper/results/eval_results_claude_judge.json (primary corpus) and
paper/results/eval_results_heldout.json (held-out corpus) and prints a
side-by-side table. Computes whether retrieval improvements replicate.
"""
from __future__ import annotations
import json
from pathlib import Path
from scipy.stats import wilcoxon

PRIMARY = Path("paper/results/eval_results_claude_judge.json")
HELDOUT = Path("paper/results/eval_results_heldout.json")


def paired_p(results, sys_a, sys_b, metric="precision_at_10"):
    by_q = {}
    for r in results:
        by_q.setdefault(r["query_id"], {})[r["system"]] = r.get(metric)
    diffs = [by_q[q][sys_a] - by_q[q][sys_b] for q in by_q
             if sys_a in by_q[q] and sys_b in by_q[q]
             and by_q[q][sys_a] is not None and by_q[q][sys_b] is not None]
    nz = [x for x in diffs if x != 0]
    if not nz:
        return None, len(diffs), 0.0
    try:
        w, p = wilcoxon(nz, alternative="greater")
    except Exception:
        return None, len(diffs), sum(diffs) / len(diffs)
    return p, len(diffs), sum(diffs) / len(diffs)


def main():
    if not PRIMARY.exists():
        print(f"Missing: {PRIMARY}")
        return
    if not HELDOUT.exists():
        print(f"Missing: {HELDOUT} — run scripts/run_heldout_eval.py first.")
        return

    primary = json.loads(PRIMARY.read_text())
    heldout = json.loads(HELDOUT.read_text())
    systems = ["baseline", "bm25", "graph_only", "graphrag", "ppr_only", "hybrid_4way"]
    labels = ["Vector", "BM25", "Graph-only", "GraphRAG", "PPR-only", "Hybrid RRF"]

    print("\n" + "=" * 88)
    print(" CROSS-CORPUS VALIDATION — Primary ASRS (n=2000) vs Held-Out ASRS (n=2000)")
    print("=" * 88)

    for metric, label in [
        ("precision_at_10", "P@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("faithfulness", "Faith (Ollama)"),
    ]:
        print(f"\n{label}")
        print(f"  {'System':<14} {'Primary':>12} {'Held-out':>12} {'Δ':>10} {'replicates':>12}")
        for sys_key, sys_label in zip(systems, labels):
            p_val = primary["metrics"].get(sys_key, {}).get(metric)
            h_val = heldout["metrics"].get(sys_key, {}).get(metric)
            if p_val is None or h_val is None:
                continue
            delta = h_val - p_val
            replicates = "yes" if abs(delta) < max(abs(p_val) * 0.2, 0.05) else "diverges"
            print(f"  {sys_label:<14} {p_val:>12.3f} {h_val:>12.3f} {delta:>+10.3f} {replicates:>12}")

    print("\n" + "=" * 88)
    print(" PAIRED WILCOXON SIGNIFICANCE — Hybrid and GraphRAG vs Vector")
    print("=" * 88)
    for name, results in [("Primary", primary["results"]), ("Held-out", heldout["results"])]:
        print(f"\n{name} corpus (P@10 gains):")
        for a, b in [("hybrid_4way", "baseline"), ("graphrag", "baseline"),
                     ("ppr_only", "baseline"), ("hybrid_4way", "ppr_only")]:
            p, n, md = paired_p(results, a, b)
            sig = "*" if p and p < 0.05 else " "
            p_str = f"{p:.4f}" if p is not None else "n/a"
            print(f"  {a:>12} > {b:<10}  n={n}  Δmean={md:+.4f}  p={p_str} {sig}")


if __name__ == "__main__":
    main()
