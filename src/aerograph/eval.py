"""Evaluation framework with benchmark queries and metrics.

Runs both GraphRAG and baseline (vector-only) retrievers on a 50-query
benchmark suite. Produces paper-ready metrics and figures.

Metrics:
  - Faithfulness (Claude judge)
  - Answer relevance (Claude judge)
  - Context precision (deterministic)
  - Context recall (deterministic)
  - Multi-hop causal chain accuracy
"""

from __future__ import annotations

import json
import os
import sys
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
    system: str  # "graphrag" or "baseline"
    answer: str
    source_acns: list[str]
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    causal_chain_accuracy: float = 0.0
    latency_ms: float = 0.0


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
    """Generate the 50-query benchmark suite.

    20 single-hop factual, 20 multi-hop causal, 10 comparative.
    """
    if output_path is None:
        output_path = DATA_DIR / "eval_queries.jsonl"

    queries = []

    # 20 single-hop factual queries
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
        queries.append(EvalQuery(
            id=qid, question=question, query_type="single_hop",
        ))

    # 20 multi-hop causal queries
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
        queries.append(EvalQuery(
            id=qid, question=question, query_type="multi_hop",
        ))

    # 10 comparative queries
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
        queries.append(EvalQuery(
            id=qid, question=question, query_type="comparative",
        ))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in queries:
            f.write(json.dumps(asdict(q)) + "\n")

    print(f"Generated {len(queries)} evaluation queries")
    return queries


def evaluate_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Score faithfulness: are claims in the answer supported by context?

    Uses Claude as judge. Falls back to keyword overlap.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _keyword_faithfulness(answer, context_chunks)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        context = "\n\n".join(context_chunks[:5])
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": f"""Rate the faithfulness of this answer to the provided context.
Faithfulness means every claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Rate from 0.0 to 1.0 where 1.0 means perfectly faithful.
Return ONLY a decimal number, nothing else."""}],
        )

        try:
            return min(1.0, max(0.0, float(response.content[0].text.strip())))
        except ValueError:
            return _keyword_faithfulness(answer, context_chunks)
    except Exception:
        return _keyword_faithfulness(answer, context_chunks)


def evaluate_relevance(question: str, answer: str) -> float:
    """Score answer relevance: does the answer address the question?"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _keyword_relevance(question, answer)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": f"""Rate how relevant this answer is to the question.
Relevance means the answer directly addresses what was asked.

Question: {question}
Answer: {answer}

Rate from 0.0 to 1.0 where 1.0 means perfectly relevant.
Return ONLY a decimal number, nothing else."""}],
        )

        try:
            return min(1.0, max(0.0, float(response.content[0].text.strip())))
        except ValueError:
            return _keyword_relevance(question, answer)
    except Exception:
        return _keyword_relevance(question, answer)


def evaluate_context_precision(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Precision: fraction of retrieved chunks from relevant reports."""
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
        return 1.0  # no expected = vacuously recalled
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def evaluate_causal_chain(answer: str, query_type: str) -> float:
    """Score multi-hop causal reasoning quality.

    Only scored for multi_hop queries. Checks for causal language and
    chain structure in the answer.
    """
    if query_type != "multi_hop":
        return -1.0  # N/A

    causal_indicators = [
        "caused by", "led to", "resulted in", "contributing factor",
        "because", "therefore", "consequently", "as a result",
        "chain", "sequence", "cascade", "preceded by",
    ]
    lower = answer.lower()
    score = sum(1 for ind in causal_indicators if ind in lower)
    return min(1.0, score / 4.0)  # Normalize: 4+ indicators = 1.0


def _keyword_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Fallback faithfulness via keyword overlap."""
    context_words = set()
    for chunk in context_chunks:
        context_words.update(chunk.lower().split())
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0
    overlap = len(answer_words & context_words) / len(answer_words)
    return min(1.0, overlap * 1.5)  # Scale up slightly


def _keyword_relevance(question: str, answer: str) -> float:
    """Fallback relevance via keyword overlap with question."""
    q_words = set(question.lower().split()) - {"what", "how", "is", "the", "a", "an", "of", "in", "to", "and", "for"}
    a_words = set(answer.lower().split())
    if not q_words:
        return 0.5
    overlap = len(q_words & a_words) / len(q_words)
    return min(1.0, overlap * 1.2)


def run_single_eval(
    query: EvalQuery,
    retriever,
    system_name: str,
) -> EvalResult:
    """Run evaluation on a single query with a given retriever."""
    from aerograph.generate import generate_answer

    start = time.time()
    retrieval_result = retriever.retrieve(query.question)
    gen_result = generate_answer(query.question, retrieval_result)

    context_chunks = [c.text for c in retrieval_result.chunks]
    retrieved_report_ids = [c.report_id for c in retrieval_result.chunks]

    faithfulness = evaluate_faithfulness(gen_result.text, context_chunks)
    relevance = evaluate_relevance(query.question, gen_result.text)
    precision = evaluate_context_precision(retrieved_report_ids, query.expected_acns)
    recall = evaluate_context_recall(retrieved_report_ids, query.expected_acns)
    causal = evaluate_causal_chain(gen_result.text, query.query_type)

    elapsed = (time.time() - start) * 1000

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
        latency_ms=elapsed,
    )


def run_evaluation(
    queries_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Run full evaluation suite: GraphRAG vs baseline on all queries."""
    from aerograph.retrieve import GraphRAGRetriever, BaselineRetriever

    if queries_path is None:
        queries_path = DATA_DIR / "eval_queries.jsonl"

    if not queries_path.exists():
        print("Generating evaluation queries...")
        generate_eval_queries(queries_path)

    queries = load_eval_queries(queries_path)
    print(f"Running evaluation on {len(queries)} queries")

    graphrag = GraphRAGRetriever()
    baseline = BaselineRetriever()

    all_results: list[EvalResult] = []

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query.id}: {query.question[:60]}...")

        try:
            gr_result = run_single_eval(query, graphrag, "graphrag")
            all_results.append(gr_result)
        except Exception as e:
            print(f"    GraphRAG failed: {e}")

        try:
            bl_result = run_single_eval(query, baseline, "baseline")
            all_results.append(bl_result)
        except Exception as e:
            print(f"    Baseline failed: {e}")

    # Aggregate metrics
    metrics = _aggregate_metrics(all_results)

    # Save results
    if output_path is None:
        output_path = RESULTS_DIR / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)
    print(f"Results saved to {output_path}")

    # Generate figures
    generate_figures(metrics, all_results)

    return metrics


def _aggregate_metrics(results: list[EvalResult]) -> dict:
    """Aggregate per-query metrics into summary statistics."""
    systems = {"graphrag": [], "baseline": []}
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
        latency = [r.latency_ms for r in system_results]

        metrics[system_name] = {
            "faithfulness": _safe_mean(faithfulness_scores),
            "answer_relevance": _safe_mean(relevance_scores),
            "context_precision": _safe_mean(precision_scores),
            "context_recall": _safe_mean(recall_scores),
            "causal_chain_accuracy": _safe_mean(causal_scores),
            "mean_latency_ms": _safe_mean(latency),
            "n_queries": len(system_results),
        }

        # Per query type breakdown
        for qt in ["single_hop", "multi_hop", "comparative"]:
            qt_results = [r for r in system_results if r.query_type == qt]
            if qt_results:
                metrics[system_name][f"{qt}_faithfulness"] = _safe_mean([r.faithfulness for r in qt_results])
                metrics[system_name][f"{qt}_relevance"] = _safe_mean([r.answer_relevance for r in qt_results])

    return metrics


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_figures(metrics: dict, results: list[EvalResult]) -> None:
    """Generate paper-ready evaluation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Academic styling
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


def _plot_metrics_comparison(metrics: dict, plt, np) -> None:
    """Bar chart comparing GraphRAG vs baseline across all metrics."""
    metric_names = ["Faithfulness", "Answer\nRelevance", "Context\nPrecision", "Context\nRecall", "Causal Chain\nAccuracy"]
    metric_keys = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "causal_chain_accuracy"]

    graphrag_vals = [metrics.get("graphrag", {}).get(k, 0) for k in metric_keys]
    baseline_vals = [metrics.get("baseline", {}).get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, graphrag_vals, width, label="GraphRAG", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + width / 2, baseline_vals, width, label="Baseline (Vector)", color="#FF9800", edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("GraphRAG vs Baseline: Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bar_chart_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR / 'bar_chart_metrics.png'}")


def _plot_query_type_breakdown(metrics: dict, results: list[EvalResult], plt, np) -> None:
    """Grouped bar chart: performance by query type."""
    query_types = ["single_hop", "multi_hop", "comparative"]
    type_labels = ["Single-Hop", "Multi-Hop", "Comparative"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (system, color) in enumerate([("graphrag", "#2196F3"), ("baseline", "#FF9800")]):
        system_results = [r for r in results if r.system == system]
        faith_by_type = []
        rel_by_type = []
        for qt in query_types:
            qt_results = [r for r in system_results if r.query_type == qt]
            faith_by_type.append(_safe_mean([r.faithfulness for r in qt_results]))
            rel_by_type.append(_safe_mean([r.answer_relevance for r in qt_results]))

        x = np.arange(len(type_labels))
        width = 0.35
        offset = -width / 2 if idx == 0 else width / 2

        axes[0].bar(x + offset, faith_by_type, width, label=system.title(), color=color, edgecolor="white")
        axes[1].bar(x + offset, rel_by_type, width, label=system.title(), color=color, edgecolor="white")

    axes[0].set_title("Faithfulness by Query Type")
    axes[0].set_xticks(np.arange(len(type_labels)))
    axes[0].set_xticklabels(type_labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    axes[1].set_title("Answer Relevance by Query Type")
    axes[1].set_xticks(np.arange(len(type_labels)))
    axes[1].set_xticklabels(type_labels)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

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


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        # Load existing results and regenerate figures
        results_path = RESULTS_DIR / "eval_results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            results = [EvalResult(**r) for r in data["results"]]
            generate_figures(data["metrics"], results)
        else:
            print("No eval results found. Run full evaluation first.")
    else:
        run_evaluation()
