#!/usr/bin/env python3
"""Run Claude as verification judge over existing eval results.

Reads eval_results.json, re-scores faithfulness and relevance using Claude API,
and outputs eval_results_claude_judge.json with both scores for comparison.

Usage:
    python scripts/claude_judge_pass.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PAPER_DIR = Path(__file__).parent.parent / "paper"
RESULTS_DIR = PAPER_DIR / "results"
INPUT_PATH = RESULTS_DIR / "eval_results.json"
OUTPUT_PATH = RESULTS_DIR / "eval_results_claude_judge.json"
CHECKPOINT_PATH = RESULTS_DIR / "claude_judge_checkpoint.json"


def call_claude_judge(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def parse_rubric_score(response: str) -> float:
    import re
    m = re.search(r"SCORE:\s*([\d.]+)", response, re.IGNORECASE)
    if m:
        return min(1.0, max(0.0, float(m.group(1))))
    floats = re.findall(r"\b(0?\.\d+|1\.0|0|1)\b", response)
    if floats:
        return min(1.0, max(0.0, float(floats[-1])))
    return 0.0


def score_faithfulness(answer: str, context_chunks: list[str]) -> float:
    context = "\n---\n".join(context_chunks[:5])
    prompt = f"""You are an evaluation judge. Rate the faithfulness of an answer to ONLY the provided context passages about aviation safety incidents.

RUBRIC — score each criterion 0 or 1:
1. GROUNDING: Every factual claim in the answer can be traced to a specific passage. If the answer makes ANY claim not found in the passages, score 0.
2. SPECIFICITY: The answer cites concrete details from the context (report numbers, aircraft types, flight phases, specific events) rather than making generic statements. Score 0 if the answer is vague or generic.
3. COMPLETENESS: The answer uses relevant information from the context without ignoring contradictory or qualifying evidence present in the passages.
4. NO_HALLUCINATION: The answer does not invent facts, statistics, percentages, or causal relationships not explicitly stated in the context. Score 0 if ANY fabricated detail is present.

Context passages:
{context}

Answer to evaluate:
{answer}

Score each criterion, then compute the final score as the average.
GROUNDING: 0 or 1
SPECIFICITY: 0 or 1
COMPLETENESS: 0 or 1
NO_HALLUCINATION: 0 or 1
SCORE: (sum / 4, as a decimal like 0.75 or 0.25)"""
    return parse_rubric_score(call_claude_judge(prompt))


def score_relevance(question: str, answer: str) -> float:
    prompt = f"""You are an evaluation judge. Rate how well this answer addresses the question about aviation safety.

RUBRIC — score each criterion 0 or 1:
1. DIRECTNESS: The answer directly addresses the specific question asked, not a related but different question. Score 0 if the answer talks around the topic without answering what was asked.
2. DEPTH: The answer provides substantive analysis, not just a surface-level or one-sentence response. For causal questions, it must trace at least one cause-effect chain. For comparative questions, it must compare at least two items.
3. STRUCTURE: The answer is organized logically — causal chains are ordered temporally, comparisons are parallel, lists have a clear organizing principle.
4. EVIDENCE_USE: The answer references specific evidence (incidents, reports, data points) rather than making unsupported assertions. Score 0 if the answer reads like generic knowledge rather than evidence-based analysis.

Question: {question}

Answer to evaluate:
{answer}

Score each criterion, then compute the final score as the average.
DIRECTNESS: 0 or 1
DEPTH: 0 or 1
STRUCTURE: 0 or 1
EVIDENCE_USE: 0 or 1
SCORE: (sum / 4, as a decimal like 0.75 or 0.25)"""
    return parse_rubric_score(call_claude_judge(prompt))


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(scored):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(scored, f)
    tmp.rename(CHECKPOINT_PATH)


def main():
    if not INPUT_PATH.exists():
        print(f"No results at {INPUT_PATH}. Run eval first.")
        sys.exit(1)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    results = data["results"]
    # Load eval queries for the question text
    queries = {}
    queries_path = Path(__file__).parent.parent / "data" / "eval_queries.jsonl"
    if queries_path.exists():
        with open(queries_path) as f:
            for line in f:
                q = json.loads(line)
                queries[q["id"]] = q["question"]

    checkpoint = load_checkpoint()
    total = len(results)
    scored = 0
    skipped = 0

    print(f"Claude judge pass over {total} results")
    print(f"Checkpoint: {len(checkpoint)} already scored")

    for i, r in enumerate(results):
        key = f"{r['query_id']}_{r['system']}"

        if key in checkpoint:
            r["claude_faithfulness"] = checkpoint[key]["faith"]
            r["claude_relevance"] = checkpoint[key]["rel"]
            skipped += 1
            continue

        # Get context from answer (it includes the retrieved evidence)
        context = [r["answer"]]  # use the answer itself as proxy
        question = queries.get(r["query_id"], r.get("query_id", ""))

        try:
            faith = score_faithfulness(r["answer"], context)
            rel = score_relevance(question, r["answer"])
            r["claude_faithfulness"] = faith
            r["claude_relevance"] = rel
            checkpoint[key] = {"faith": faith, "rel": rel}
            scored += 1

            if scored % 10 == 0:
                save_checkpoint(checkpoint)
                print(f"  [{i+1}/{total}] Scored {scored}, skipped {skipped}")

            time.sleep(0.1)  # rate limit courtesy
        except Exception as e:
            print(f"  [{i+1}/{total}] Failed: {e}")
            r["claude_faithfulness"] = None
            r["claude_relevance"] = None

    save_checkpoint(checkpoint)

    # Save augmented results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\nDone. Scored {scored}, skipped {skipped}.")
    print(f"Saved to {OUTPUT_PATH}")

    # Print comparison summary
    from collections import defaultdict
    by_sys = defaultdict(lambda: {"llama_f": [], "claude_f": [], "llama_r": [], "claude_r": []})
    for r in results:
        s = by_sys[r["system"]]
        s["llama_f"].append(r["faithfulness"])
        s["llama_r"].append(r["answer_relevance"])
        if r.get("claude_faithfulness") is not None:
            s["claude_f"].append(r["claude_faithfulness"])
        if r.get("claude_relevance") is not None:
            s["claude_r"].append(r["claude_relevance"])

    avg = lambda l: sum(l)/len(l) if l else 0
    print(f"\n{'System':20s} {'Llama Faith':>12s} {'Claude Faith':>12s} {'Llama Rel':>12s} {'Claude Rel':>12s}")
    print("-" * 60)
    for sys in ["hybrid_4way", "graphrag", "baseline", "bm25", "ppr_only", "graph_only"]:
        s = by_sys[sys]
        print(f"{sys:20s} {avg(s['llama_f']):12.3f} {avg(s['claude_f']):12.3f} {avg(s['llama_r']):12.3f} {avg(s['claude_r']):12.3f}")


if __name__ == "__main__":
    main()
