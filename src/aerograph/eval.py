"""Evaluation framework with benchmark queries and metrics.

Three-tier evaluation:
  1. Deterministic IR metrics (P@k, R@k, nDCG@k)
  2. LLM-as-judge rubric scoring (Ollama local / Claude fallback)
  3. Pairwise blind comparison across retrieval systems

Runs six retrievers on a 50-query benchmark suite for ablation study:
  - GraphRAG (hybrid vector + graph with RRF)
  - Baseline (vector-only)
  - BM25 (lexical-only)
  - Graph-only (graph neighborhood, no vector)
  - HippoRAG (4-way hybrid with PPR)
  - PPR-only (personalized PageRank)

Metrics:
  - Faithfulness (LLM judge, 4-criterion rubric)
  - Answer relevance (LLM judge, 4-criterion rubric)
  - Precision@10, Recall@10, nDCG@10 (deterministic)
  - Context precision / recall (deterministic)
  - Multi-hop causal chain accuracy
  - Pairwise preference (blind A/B comparison)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PAPER_DIR = Path(__file__).parent.parent.parent / "paper"
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"


@dataclass
class EvalQuery:
    id: str
    question: str
    query_type: str  # "single_hop", "multi_hop", "comparative"
    reference_answer: str = ""
    expected_entities: list[str] = field(default_factory=list)
    expected_acns: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    query_id: str
    query_type: str
    system: str  # "graphrag" | "baseline" | "bm25" | "graph_only" | "hybrid_4way" | "ppr_only"
    answer: str
    source_acns: list[str]
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    causal_chain_accuracy: float = 0.0
    reference_similarity: float = 0.0
    latency_ms: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """Mean of list, 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    """Sample standard deviation, 0.0 if empty or single value."""
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _ci95(values: list[float]) -> float:
    """95% confidence interval half-width (mean +/- this value)."""
    if len(values) < 2:
        return 0.0
    std = _safe_std(values)
    return 1.96 * std / math.sqrt(len(values))


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set | list, k: int = 10) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set | list, k: int = 10) -> float:
    """Fraction of relevant items found in top-k."""
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved: list[str], relevant: set | list, k: int = 10) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, doc in enumerate(top_k):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs at the top
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Rubric score parsing
# ---------------------------------------------------------------------------

def _parse_rubric_score(response: str) -> float:
    """Extract a 0-1 score from LLM judge output."""
    # Primary: look for "SCORE: X.XX"
    m = re.search(r"SCORE:\s*([\d.]+)", response, re.IGNORECASE)
    if m:
        return min(1.0, max(0.0, float(m.group(1))))
    # Fallback: last float in 0-1 range
    floats = re.findall(r"\b(0?\.\d+|1\.0{0,2}|0|1)\b", response)
    if floats:
        return min(1.0, max(0.0, float(floats[-1])))
    return 0.0


# ---------------------------------------------------------------------------
# LLM judge infrastructure
# ---------------------------------------------------------------------------

def _call_judge(prompt: str) -> str:
    """Call LLM judge: try Ollama (llama3.1:8b) first, fall back to Claude."""
    # Try Ollama first
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "")
    except Exception:
        pass

    # Fall back to Claude
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Score faithfulness via 4-criterion binary rubric."""
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

    response = _call_judge(prompt)
    if response:
        return _parse_rubric_score(response)
    return _keyword_faithfulness(answer, context_chunks)


def evaluate_relevance(question: str, answer: str) -> float:
    """Score answer relevance via 4-criterion rubric."""
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

    response = _call_judge(prompt)
    if response:
        return _parse_rubric_score(response)
    return _keyword_relevance(question, answer)


def evaluate_context_precision(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Precision: fraction of retrieved chunks from relevant reports."""
    if not relevant_ids:
        return -1.0  # N/A: no expected ACNs to evaluate against
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in retrieved_ids if rid in relevant_set)
    return hits / len(retrieved_ids)


def evaluate_context_recall(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Recall: fraction of relevant reports represented in retrieval."""
    if not relevant_ids:
        return -1.0  # N/A: no expected ACNs to evaluate against
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def evaluate_causal_chain(answer: str, query_type: str) -> float:
    """Score multi-hop causal reasoning quality (heuristic)."""
    if query_type != "multi_hop":
        return -1.0  # N/A

    causal_indicators = [
        "caused by", "led to", "resulted in", "contributing factor",
        "because", "therefore", "consequently", "as a result",
        "chain", "sequence", "cascade", "preceded by",
    ]
    lower = answer.lower()
    score = sum(1 for ind in causal_indicators if ind in lower)
    return min(1.0, score / 4.0)


def evaluate_reference_similarity(answer: str, reference: str) -> float:
    """Token-level F1 (ROUGE-L style) against reference answer."""
    if not reference or not reference.strip():
        return 0.0

    answer_tokens = answer.lower().split()
    ref_tokens = reference.lower().split()

    if not answer_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(answer_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == answer_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_query_relevance(query: str, graph_backend=None) -> set[str]:
    """Compute relevant report_ids for a query using graph structure."""
    from aerograph.retrieve import _keyword_entity_extract

    if graph_backend is None:
        try:
            from aerograph.graph import detect_backend
            graph_backend = detect_backend()
        except Exception:
            return set()

    entities = _keyword_entity_extract(query)
    if not entities:
        return set()

    entity_report_sets: list[set[str]] = []

    for entity in entities:
        node = graph_backend.get_node(entity)
        if node is None:
            continue

        # Skip hub nodes with too many reports
        if len(node.report_ids) > 500:
            continue

        report_ids = set(node.report_ids)

        # Expand 1-hop neighbors
        subgraph = graph_backend.get_neighbors(entity, depth=1, max_degree=200)
        for neighbor_node in subgraph.nodes:
            if len(neighbor_node.report_ids) <= 500:
                report_ids.update(neighbor_node.report_ids)

        entity_report_sets.append(report_ids)

    if not entity_report_sets:
        return set()

    # Multi-entity intersection when 2+ entities match
    if len(entity_report_sets) >= 2:
        relevant = entity_report_sets[0]
        for s in entity_report_sets[1:]:
            relevant = relevant & s
        # Fall back to union if intersection is too small
        if len(relevant) < 5:
            relevant = set()
            for s in entity_report_sets:
                relevant.update(s)
    else:
        relevant = entity_report_sets[0]

    # Cap at 100 reports
    if len(relevant) > 100:
        relevant = set(list(relevant)[:100])

    return relevant


# ---------------------------------------------------------------------------
# Keyword fallbacks (when no LLM judge available)
# ---------------------------------------------------------------------------

def _keyword_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Fallback faithfulness via keyword overlap."""
    context_words = set()
    for chunk in context_chunks:
        context_words.update(chunk.lower().split())
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words) / len(answer_words)
    return min(1.0, overlap * 1.5)


def _keyword_relevance(question: str, answer: str) -> float:
    """Fallback relevance via keyword overlap with question."""
    q_words = set(question.lower().split()) - {"what", "how", "is", "the", "a", "an", "of", "in", "to", "and", "for"}
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.5
    overlap = len(q_words & a_words) / len(q_words)
    return min(1.0, overlap * 1.2)


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path) -> dict:
    """Load evaluation checkpoint, returns empty dict if missing."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"results": [], "completed_query_ids": []}
    return {"results": [], "completed_query_ids": []}


def _save_checkpoint(path: Path, data: dict) -> None:
    """Atomic save: write to tmp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp_path).rename(path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def load_eval_queries(path: Optional[Path] = None) -> list[EvalQuery]:
    """Load evaluation queries from JSONL file."""
    if path is None:
        path = DATA_DIR / "eval_queries.jsonl"
    queries = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            queries.append(EvalQuery(**data))
    return queries


def generate_eval_queries(output_path: Optional[Path] = None) -> list[EvalQuery]:
    """Generate the 50-query benchmark suite."""
    if output_path is None:
        output_path = DATA_DIR / "eval_queries.jsonl"

    queries = []

    single_hop = [
        ("sh01", "What aircraft type was involved in the most bird strike incidents?"),
        ("sh02", "During which phase of flight do runway incursions most commonly occur?"),
        ("sh03", "What is the most frequently cited contributing factor in engine failure reports?"),
        ("sh04", "How many incident reports mention turbulence encounters during cruise phase?"),
        ("sh05", "What weather condition is most commonly associated with go-around decisions?"),
        ("sh06", "Which ATC facility type appears most frequently in communication failure reports?"),
        ("sh07", "What is the most common outcome reported for hydraulic system failures?"),
        ("sh08", "During which phase of flight do TCAS RAs most frequently occur?"),
        ("sh09", "What aircraft component is most often cited in maintenance error reports?"),
        ("sh10", "How frequently do altitude deviations occur during the descent phase?"),
        ("sh11", "What is the most common pilot recommendation in icing encounter reports?"),
        ("sh12", "Which aircraft type has the highest frequency of autopilot disconnect events?"),
        ("sh13", "What time-related factor is most cited in pilot deviation reports?"),
        ("sh14", "How often do gear malfunction reports mention prior maintenance actions?"),
        ("sh15", "What is the primary contributing factor in near midair collision reports?"),
        ("sh16", "During which phase do pressurization loss events most commonly occur?"),
        ("sh17", "What weather phenomenon is most associated with wind shear encounters?"),
        ("sh18", "How many reports mention fatigue as a contributing factor?"),
        ("sh19", "What is the most common ATC response to pilot-reported emergencies?"),
        ("sh20", "Which airport facilities appear most in ground conflict reports?"),
    ]

    for qid, question in single_hop:
        queries.append(EvalQuery(id=qid, question=question, query_type="single_hop"))

    multi_hop = [
        ("mh01", "What causal chain links bird strikes to engine failure and subsequent go-around decisions?"),
        ("mh02", "How does crew fatigue contribute to communication failures that lead to altitude deviations?"),
        ("mh03", "What sequence of factors connects weather deterioration to runway incursion events?"),
        ("mh04", "How do maintenance errors lead to hydraulic failures that affect landing gear operation?"),
        ("mh05", "What causal path connects ATC workload to communication breakdowns and near midair collisions?"),
        ("mh06", "How does icing accumulation on approach lead to autopilot disconnects and unstabilized approaches?"),
        ("mh07", "What chain of events links fuel management errors to engine power loss during climb?"),
        ("mh08", "How do training deficiencies contribute to procedural deviations that cause runway excursions?"),
        ("mh09", "What causal factors connect thunderstorm encounters to turbulence injuries in the cabin?"),
        ("mh10", "How does time pressure lead to abbreviated checklists that result in configuration errors?"),
        ("mh11", "What sequence connects poor CRM to missed callouts and subsequent CFIT risk?"),
        ("mh12", "How do equipment malfunctions cascade from initial failure to emergency declarations?"),
        ("mh13", "What causal chain links inadequate weather briefing to unexpected wind shear on approach?"),
        ("mh14", "How do scheduling factors contribute to fatigue-related errors in multi-leg operations?"),
        ("mh15", "What path connects navigation system errors to airspace deviations and ATC interventions?"),
        ("mh16", "How do language barriers between pilots and controllers lead to heading or altitude errors?"),
        ("mh17", "What causal factors link deferred maintenance items to in-flight system failures?"),
        ("mh18", "How does high-density traffic contribute to controller errors and separation violations?"),
        ("mh19", "What sequence of events connects electrical failures to loss of communication and navigation?"),
        ("mh20", "How do organizational pressures contribute to normalized deviance and safety incidents?"),
    ]

    for qid, question in multi_hop:
        queries.append(EvalQuery(id=qid, question=question, query_type="multi_hop"))

    comparative = [
        ("cp01", "Compare the contributing factors in B737 vs A320 engine failure incidents."),
        ("cp02", "How do bird strike outcomes differ between GA aircraft and commercial jets?"),
        ("cp03", "Compare the frequency of go-arounds at high-altitude vs sea-level airports."),
        ("cp04", "How do pilot-reported vs controller-reported communication failures differ?"),
        ("cp05", "Compare maintenance-related incidents between regional jets and widebody aircraft."),
        ("cp06", "How do winter vs summer weather contribute differently to approach incidents?"),
        ("cp07", "Compare the causal chains in runway incursions at towered vs non-towered airports."),
        ("cp08", "How do fatigue-related incidents differ between short-haul and long-haul operations?"),
        ("cp09", "Compare TCAS RA compliance rates between different aircraft type categories."),
        ("cp10", "How do automation-related incidents differ between older and newer aircraft types?"),
    ]

    for qid, question in comparative:
        queries.append(EvalQuery(id=qid, question=question, query_type="comparative"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in queries:
            f.write(json.dumps(asdict(q)) + "\n")

    print(f"Generated {len(queries)} evaluation queries")
    return queries


# ---------------------------------------------------------------------------
# Single query evaluation
# ---------------------------------------------------------------------------

def run_single_eval(
    query: EvalQuery,
    retriever,
    system_name: str,
    generator_fn=None,
    relevant_ids: Optional[set[str]] = None,
) -> EvalResult:
    """Run evaluation on a single query with a given retriever."""
    if generator_fn is None:
        from aerograph.generate import generate_answer
        generator_fn = generate_answer

    start = time.time()
    retrieval_result = retriever.retrieve(query.question)
    gen_result = generator_fn(query.question, retrieval_result)
    elapsed = (time.time() - start) * 1000

    context_chunks = [c.text for c in retrieval_result.chunks]
    retrieved_report_ids = [c.report_id for c in retrieval_result.chunks]

    faithfulness = evaluate_faithfulness(gen_result.text, context_chunks)
    relevance = evaluate_relevance(query.question, gen_result.text)
    precision = evaluate_context_precision(retrieved_report_ids, query.expected_acns)
    recall = evaluate_context_recall(retrieved_report_ids, query.expected_acns)
    causal = evaluate_causal_chain(gen_result.text, query.query_type)
    ref_sim = evaluate_reference_similarity(gen_result.text, query.reference_answer)

    # IR metrics against graph-derived relevance set
    if relevant_ids is None:
        relevant_ids = set(query.expected_acns) if query.expected_acns else set()
    p10 = precision_at_k(retrieved_report_ids, relevant_ids, k=10) if relevant_ids else 0.0
    r10 = recall_at_k(retrieved_report_ids, relevant_ids, k=10) if relevant_ids else 0.0
    ndcg10 = ndcg_at_k(retrieved_report_ids, relevant_ids, k=10) if relevant_ids else 0.0

    return EvalResult(
        query_id=query.id,
        query_type=query.query_type,
        system=system_name,
        answer=gen_result.text,
        source_acns=gen_result.source_acns,
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_precision=precision,
        context_recall=recall,
        causal_chain_accuracy=causal,
        reference_similarity=ref_sim,
        latency_ms=elapsed,
        precision_at_10=p10,
        ndcg_at_10=ndcg10,
        recall_at_10=r10,
    )


# ---------------------------------------------------------------------------
# Full evaluation run with checkpoint resume
# ---------------------------------------------------------------------------

def run_evaluation(
    queries_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    retrievers_dict: Optional[dict] = None,
) -> dict:
    """Run full evaluation suite with checkpoint resume and pairwise comparison."""
    from aerograph.retrieve import (
        GraphRAGRetriever, BaselineRetriever,
        BM25Retriever, GraphOnlyRetriever,
    )

    if queries_path is None:
        queries_path = DATA_DIR / "eval_queries.jsonl"

    if not queries_path.exists():
        print("Generating evaluation queries...")
        generate_eval_queries(queries_path)

    queries = load_eval_queries(queries_path)
    print(f"Running evaluation on {len(queries)} queries")

    if retrievers_dict is None:
        retrievers_dict = {
            "graphrag": GraphRAGRetriever(),
            "baseline": BaselineRetriever(),
            "bm25": BM25Retriever(),
            "graph_only": GraphOnlyRetriever(),
        }

    retriever_names = list(retrievers_dict.keys())
    n_retrievers = len(retriever_names)

    # Checkpoint setup
    if output_path is None:
        output_path = RESULTS_DIR / "eval_results.json"
    checkpoint_path = output_path.with_name("eval_checkpoint.json")
    checkpoint = _load_checkpoint(checkpoint_path)
    completed_ids = set(checkpoint.get("completed_query_ids", []))
    all_results: list[EvalResult] = [EvalResult(**r) for r in checkpoint.get("results", [])]

    for i, query in enumerate(queries, 1):
        if query.id in completed_ids:
            continue

        print(f"  [{i}/{len(queries)}] {query.id}: {query.question[:60]}...")

        # Compute graph-derived relevance for IR metrics
        relevant_ids = compute_query_relevance(query.question)

        n_succeeded = 0
        query_results: list[EvalResult] = []

        for system_name in retriever_names:
            retriever = retrievers_dict[system_name]
            try:
                result = run_single_eval(
                    query, retriever, system_name,
                    relevant_ids=relevant_ids,
                )
                query_results.append(result)
                n_succeeded += 1
            except Exception as e:
                print(f"    {system_name} failed: {e}")

        # Only mark complete when all retrievers succeed
        if n_succeeded == n_retrievers:
            all_results.extend(query_results)
            completed_ids.add(query.id)

            _save_checkpoint(checkpoint_path, {
                "results": [asdict(r) for r in all_results],
                "completed_query_ids": list(completed_ids),
            })
        else:
            print(f"    Incomplete: {n_succeeded}/{n_retrievers} succeeded, skipping checkpoint")
            all_results.extend(query_results)

    # Aggregate metrics
    metrics = _aggregate_metrics(all_results)

    # Pairwise comparisons between systems
    pairwise = _run_pairwise_comparisons(queries, all_results, retriever_names)
    if pairwise:
        metrics["pairwise"] = pairwise

    # Save final results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)
    print(f"Results saved to {output_path}")

    # Clean up checkpoint on completion
    if len(completed_ids) >= len(queries) and checkpoint_path.exists():
        checkpoint_path.unlink()

    generate_figures(metrics, all_results)
    return metrics


def _aggregate_metrics(results: list[EvalResult]) -> dict:
    """Aggregate per-query metrics into summary statistics with variance."""
    systems: dict[str, list[EvalResult]] = {}
    for r in results:
        systems.setdefault(r.system, []).append(r)

    metrics = {}
    for system_name, system_results in systems.items():
        if not system_results:
            continue

        faithfulness_scores = [r.faithfulness for r in system_results]
        relevance_scores = [r.answer_relevance for r in system_results]
        precision_scores = [r.context_precision for r in system_results]
        recall_scores = [r.context_recall for r in system_results]
        causal_scores = [r.causal_chain_accuracy for r in system_results if r.causal_chain_accuracy >= 0]
        ref_scores = [r.reference_similarity for r in system_results if r.reference_similarity > 0]
        latency = [r.latency_ms for r in system_results]
        p10_scores = [r.precision_at_10 for r in system_results]
        r10_scores = [r.recall_at_10 for r in system_results]
        ndcg_scores = [r.ndcg_at_10 for r in system_results]

        metrics[system_name] = {
            "faithfulness": _safe_mean(faithfulness_scores),
            "faithfulness_std": _safe_std(faithfulness_scores),
            "faithfulness_ci95": _ci95(faithfulness_scores),
            "answer_relevance": _safe_mean(relevance_scores),
            "answer_relevance_std": _safe_std(relevance_scores),
            "answer_relevance_ci95": _ci95(relevance_scores),
            "context_precision": _safe_mean(precision_scores),
            "context_precision_std": _safe_std(precision_scores),
            "context_recall": _safe_mean(recall_scores),
            "context_recall_std": _safe_std(recall_scores),
            "causal_chain_accuracy": _safe_mean(causal_scores),
            "causal_chain_accuracy_std": _safe_std(causal_scores),
            "causal_chain_accuracy_ci95": _ci95(causal_scores),
            "reference_similarity": _safe_mean(ref_scores),
            "precision_at_10": _safe_mean(p10_scores),
            "precision_at_10_std": _safe_std(p10_scores),
            "recall_at_10": _safe_mean(r10_scores),
            "recall_at_10_std": _safe_std(r10_scores),
            "ndcg_at_10": _safe_mean(ndcg_scores),
            "ndcg_at_10_std": _safe_std(ndcg_scores),
            "mean_latency_ms": _safe_mean(latency),
            "latency_std_ms": _safe_std(latency),
            "n_queries": len(system_results),
        }

        # Per query type breakdown
        for qt in ["single_hop", "multi_hop", "comparative"]:
            qt_results = [r for r in system_results if r.query_type == qt]
            if qt_results:
                qt_faith = [r.faithfulness for r in qt_results]
                qt_rel = [r.answer_relevance for r in qt_results]
                qt_p10 = [r.precision_at_10 for r in qt_results]
                metrics[system_name][f"{qt}_faithfulness"] = _safe_mean(qt_faith)
                metrics[system_name][f"{qt}_faithfulness_std"] = _safe_std(qt_faith)
                metrics[system_name][f"{qt}_relevance"] = _safe_mean(qt_rel)
                metrics[system_name][f"{qt}_relevance_std"] = _safe_std(qt_rel)
                metrics[system_name][f"{qt}_precision_at_10"] = _safe_mean(qt_p10)

    return metrics


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(metrics: dict, results: list[EvalResult]) -> None:
    """Generate paper-ready evaluation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    _plot_metrics_comparison(metrics, plt, np)
    _plot_query_type_breakdown(metrics, results, plt, np)
    _plot_graph_stats(plt, np)
    _plot_pairwise_comparison(metrics, plt, np)


def _plot_metrics_comparison(metrics: dict, plt, np) -> None:
    """Bar chart comparing all retriever systems with 95% CI error bars."""
    metric_names = ["Faithfulness", "Answer\nRelevance", "Context\nPrecision", "Context\nRecall", "Causal Chain\nAccuracy"]
    metric_keys = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "causal_chain_accuracy"]

    systems = [
        ("graphrag", "GraphRAG", "#2196F3"),
        ("baseline", "Baseline (Vector)", "#FF9800"),
        ("bm25", "BM25", "#4CAF50"),
        ("graph_only", "Graph Only", "#9C27B0"),
        ("hybrid_4way", "HippoRAG", "#E91E63"),
        ("ppr_only", "PPR Only", "#009688"),
    ]
    systems = [(k, label, c) for k, label, c in systems if k in metrics]
    n_systems = len(systems)

    x = np.arange(len(metric_names))
    width = 0.8 / max(n_systems, 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, (sys_key, label, color) in enumerate(systems):
        vals = [metrics.get(sys_key, {}).get(k, 0) for k in metric_keys]
        ci = [metrics.get(sys_key, {}).get(f"{k}_ci95", 0) for k in metric_keys]
        offset = (idx - (n_systems - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=ci,
                      label=label, color=color, edgecolor="white",
                      capsize=2, error_kw={"linewidth": 1})
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Retriever Ablation: Evaluation Metrics (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bar_chart_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR / 'bar_chart_metrics.png'}")


def _plot_query_type_breakdown(metrics: dict, results: list[EvalResult], plt, np) -> None:
    """Grouped bar chart: performance by query type for all retriever systems."""
    query_types = ["single_hop", "multi_hop", "comparative"]
    type_labels = ["Single-Hop", "Multi-Hop", "Comparative"]

    systems = [
        ("graphrag", "GraphRAG", "#2196F3"),
        ("baseline", "Baseline (Vector)", "#FF9800"),
        ("bm25", "BM25", "#4CAF50"),
        ("graph_only", "Graph Only", "#9C27B0"),
        ("hybrid_4way", "HippoRAG", "#E91E63"),
        ("ppr_only", "PPR Only", "#009688"),
    ]
    present_systems = {r.system for r in results}
    systems = [(k, label, c) for k, label, c in systems if k in present_systems]
    n_systems = len(systems)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.8 / max(n_systems, 1)

    for idx, (system, label, color) in enumerate(systems):
        system_results = [r for r in results if r.system == system]
        faith_by_type = []
        rel_by_type = []
        for qt in query_types:
            qt_results = [r for r in system_results if r.query_type == qt]
            faith_by_type.append(_safe_mean([r.faithfulness for r in qt_results]))
            rel_by_type.append(_safe_mean([r.answer_relevance for r in qt_results]))

        x = np.arange(len(type_labels))
        offset = (idx - (n_systems - 1) / 2) * width

        axes[0].bar(x + offset, faith_by_type, width, label=label, color=color, edgecolor="white")
        axes[1].bar(x + offset, rel_by_type, width, label=label, color=color, edgecolor="white")

    for ax, title in zip(axes, ["Faithfulness by Query Type", "Answer Relevance by Query Type"]):
        ax.set_title(title)
        ax.set_xticks(np.arange(len(type_labels)))
        ax.set_xticklabels(type_labels)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "query_type_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR / 'query_type_breakdown.png'}")


def _plot_graph_stats(plt, np) -> None:
    """Graph statistics visualization."""
    try:
        from aerograph.graph import detect_backend
        backend = detect_backend()

        if hasattr(backend, "get_type_counts"):
            type_counts = backend.get_type_counts()
        else:
            type_counts = {}

        if hasattr(backend, "get_edge_type_counts"):
            edge_counts = backend.get_edge_type_counts()
        else:
            edge_counts = {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            ax1.barh(types, counts, color=colors, edgecolor="white")
            ax1.set_title("Entity Types in Knowledge Graph")
            ax1.set_xlabel("Count")
        else:
            ax1.text(0.5, 0.5, "No graph data available", transform=ax1.transAxes,
                     ha="center", va="center")
            ax1.set_title("Entity Types")

        if edge_counts:
            edges = list(edge_counts.keys())
            ecounts = list(edge_counts.values())
            colors = plt.cm.Set2(np.linspace(0, 1, len(edges)))
            ax2.barh(edges, ecounts, color=colors, edgecolor="white")
            ax2.set_title("Relation Types in Knowledge Graph")
            ax2.set_xlabel("Count")
        else:
            ax2.text(0.5, 0.5, "No graph data available", transform=ax2.transAxes,
                     ha="center", va="center")
            ax2.set_title("Relation Types")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "graph_stats.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {FIGURES_DIR / 'graph_stats.png'}")

    except Exception as e:
        print(f"Could not generate graph stats figure: {e}")


