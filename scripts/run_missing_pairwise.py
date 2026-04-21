"""Run the 3 missing pairwise comparisons over all 50 queries.

Pairs: Hybrid vs PPR, Hybrid vs GraphRAG, PPR vs GraphRAG.

Reads cached answers from paper/results/eval_results_claude_judge.json and
writes pairwise results back into its _pairwise metrics block.
"""
from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aerograph.eval import evaluate_pairwise

RESULTS = Path("paper/results/eval_results_claude_judge.json")
PAIRS = [
    ("hybrid_4way", "ppr_only"),
    ("hybrid_4way", "graphrag"),
    ("ppr_only", "graphrag"),
]


def load_queries_and_answers():
    data = json.loads(RESULTS.read_text())
    by_qs = {}  # (qid, system) -> answer text
    questions = {}
    for r in data["results"]:
        by_qs[(r["query_id"], r["system"])] = r["answer"]
        if r["query_id"] not in questions:
            first_line = r["answer"].split("\n", 1)[0]
            if first_line.startswith("**Retrieved evidence for:**"):
                q = first_line.replace("**Retrieved evidence for:**", "").strip()
                questions[r["query_id"]] = q
    return data, by_qs, questions


def main():
    random.seed(42)
    data, by_qs, questions = load_queries_and_answers()
    qids = sorted(questions.keys())
    print(f"Loaded {len(qids)} queries, {len(by_qs)} (query, system) answer cells")

    new_pairwise = {}
    for sys_a, sys_b in PAIRS:
        key = f"{sys_a}_vs_{sys_b}"
        print(f"\n=== {key} ===")
        wins_a = 0
        wins_b = 0
        ties = 0
        per_query = []
        for i, qid in enumerate(qids):
            q_text = questions[qid]
            ans_a = by_qs.get((qid, sys_a))
            ans_b = by_qs.get((qid, sys_b))
            if not ans_a or not ans_b:
                continue
            # Randomize which side each system is shown on per query
            if random.random() < 0.5:
                left, right = ans_a, ans_b
                left_sys, right_sys = sys_a, sys_b
            else:
                left, right = ans_b, ans_a
                left_sys, right_sys = sys_b, sys_a
            try:
                verdict = evaluate_pairwise(q_text, [left], [right])
            except Exception as e:
                print(f"  [{qid}] error: {e}")
                ties += 1
                per_query.append({"qid": qid, "verdict": "error"})
                continue
            # Map A/B back to system name
            if verdict == "A":
                winner = left_sys
            elif verdict == "B":
                winner = right_sys
            else:
                winner = "tie"
            if winner == sys_a:
                wins_a += 1
            elif winner == sys_b:
                wins_b += 1
            else:
                ties += 1
            per_query.append({"qid": qid, "winner": winner, "left_sys": left_sys})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(qids)}: {sys_a}={wins_a} {sys_b}={wins_b} ties={ties}")

        total = wins_a + wins_b + ties
        win_rate_a = wins_a / total if total else 0.0
        new_pairwise[key] = {
            f"{sys_a}_wins": wins_a,
            f"{sys_b}_wins": wins_b,
            "ties": ties,
            "total": total,
            f"{sys_a}_win_rate": round(win_rate_a, 4),
            "per_query": per_query,
        }
        print(f"  FINAL: {sys_a}={wins_a} ({win_rate_a:.1%}), {sys_b}={wins_b}, ties={ties}")

    # Merge into results file
    pw = data["metrics"].setdefault("_pairwise", {})
    pw.update(new_pairwise)
    RESULTS.write_text(json.dumps(data, indent=2))
    print(f"\nWrote updated {RESULTS}")


if __name__ == "__main__":
    main()
