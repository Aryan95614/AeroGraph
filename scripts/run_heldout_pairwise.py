"""Run the 6 paired pairwise comparisons on held-out corpus results.

Same six pairs as the primary corpus:
  GraphRAG vs Vector, Hybrid vs BM25, Hybrid vs Vector,
  Hybrid vs PPR, Hybrid vs GraphRAG, PPR vs GraphRAG
All 50 queries, position randomized.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aerograph.eval import evaluate_pairwise

RESULTS = Path("paper/results/eval_results_heldout.json")
PAIRS = [
    ("graphrag", "baseline"),
    ("hybrid_4way", "bm25"),
    ("hybrid_4way", "baseline"),
    ("hybrid_4way", "ppr_only"),
    ("hybrid_4way", "graphrag"),
    ("ppr_only", "graphrag"),
]


def main():
    random.seed(42)
    data = json.loads(RESULTS.read_text())
    by_qs: dict = {}
    for r in data["results"]:
        by_qs[(r["query_id"], r["system"])] = r["answer"]
    # Load questions from the canonical benchmark file
    questions: dict = {}
    with open("data/eval_queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q["question"]
    qids = sorted(questions.keys())
    print(f"Loaded {len(qids)} queries from data/eval_queries.jsonl")

    existing_pw = data["metrics"].get("_pairwise", {})
    new_pw: dict = {}
    for sys_a, sys_b in PAIRS:
        key = f"{sys_a}_vs_{sys_b}"
        if key in existing_pw and existing_pw[key].get("total", 0) >= 50:
            print(f"\n=== {key} === already complete ({existing_pw[key].get('total')} queries), skipping")
            continue
        print(f"\n=== {key} ===")
        wins_a = wins_b = ties = 0
        per_q = []
        for i, qid in enumerate(qids):
            ans_a = by_qs.get((qid, sys_a))
            ans_b = by_qs.get((qid, sys_b))
            if not ans_a or not ans_b:
                continue
            if random.random() < 0.5:
                left, right = ans_a, ans_b
                ls, rs = sys_a, sys_b
            else:
                left, right = ans_b, ans_a
                ls, rs = sys_b, sys_a
            try:
                v = evaluate_pairwise(questions[qid], [left], [right])
            except Exception as e:
                print(f"  err {qid}: {e}")
                ties += 1
                continue
            if v == "A":
                winner = ls
            elif v == "B":
                winner = rs
            else:
                winner = "tie"
            if winner == sys_a:
                wins_a += 1
            elif winner == sys_b:
                wins_b += 1
            else:
                ties += 1
            per_q.append({"qid": qid, "winner": winner, "left_sys": ls})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(qids)}: {sys_a}={wins_a} {sys_b}={wins_b} ties={ties}")
        total = wins_a + wins_b + ties
        new_pw[key] = {
            f"{sys_a}_wins": wins_a,
            f"{sys_b}_wins": wins_b,
            "ties": ties,
            "total": total,
            f"{sys_a}_win_rate": round(wins_a / total, 4) if total else 0.0,
            "per_query": per_q,
        }
        print(f"  FINAL: {sys_a}={wins_a} ({wins_a/total:.1%}), {sys_b}={wins_b}, ties={ties}")

    data["metrics"].setdefault("_pairwise", {}).update(new_pw)
    RESULTS.write_text(json.dumps(data, indent=2))
    print(f"\nWrote updated {RESULTS}")


if __name__ == "__main__":
    main()
