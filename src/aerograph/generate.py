"""Claude-powered answer generation with source tracing.

Takes retrieved chunks + graph context and generates an answer with
full provenance tracking (source ACNs + reasoning trace).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an aviation safety expert with access to NASA ASRS incident reports.
Your role is to analyze safety patterns, causal chains, and contributing factors
from real incident data.

When answering:
- Ground all claims in the provided incident report evidence
- Cite specific ACN numbers when referencing reports
- Distinguish between established facts and inferred patterns
- Note when evidence is limited or conflicting
- Use standard aviation terminology

If the retrieved evidence is insufficient to fully answer the question,
say so explicitly rather than speculating."""


@dataclass
class GenerationResult:
    text: str
    source_acns: list[str]
    reasoning_trace: str
    latency_ms: float = 0.0
    model: str = "claude-sonnet-4-20250514"
    metadata: dict = field(default_factory=dict)


def _build_context(retrieval_result) -> str:
    """Build the context string from retrieved chunks and graph context."""
    lines = []

    # Retrieved evidence
    lines.append("## Retrieved Evidence\n")
    seen_reports = set()
    for i, chunk in enumerate(retrieval_result.chunks, 1):
        report_tag = f"[ACN {chunk.report_id}]"
        provenance_tag = f"(source: {chunk.provenance})"
        lines.append(f"### Chunk {i} {report_tag} {provenance_tag}")
        lines.append(chunk.text)
        lines.append("")
        seen_reports.add(chunk.report_id)

    # Graph context
    if retrieval_result.chunks and retrieval_result.chunks[0].graph_context:
        lines.append("\n## Knowledge Graph Context\n")
        lines.append("Entity relationships from the aviation safety knowledge graph:")
        lines.append(retrieval_result.chunks[0].graph_context)
        lines.append("")

    # Query entities
    if retrieval_result.query_entities:
        lines.append(f"\nIdentified entities: {', '.join(retrieval_result.query_entities)}")

    return "\n".join(lines)


def generate_answer(
    query: str,
    retrieval_result,
    model: str = "claude-sonnet-4-20250514",
) -> GenerationResult:
    """Generate an answer using Claude with retrieved context.

    Args:
        query: The user's question
        retrieval_result: RetrievalResult from the retriever
        model: Claude model to use

    Returns:
        GenerationResult with answer text, source ACNs, and reasoning trace
    """
    start_time = time.time()

    context = _build_context(retrieval_result)

    user_prompt = f"""Based on the following aviation safety incident report evidence and knowledge graph context, answer the question below.

{context}

## Question
{query}

## Instructions
1. Answer the question thoroughly using the provided evidence
2. Cite specific ACN numbers for each claim
3. If the evidence supports causal reasoning, trace the causal chain
4. End with a "Sources" section listing all ACNs referenced

Provide your answer in this format:

**Analysis:**
[Your detailed answer]

**Causal Factors:**
[If applicable, list the causal chain]

**Sources:**
[List of ACN numbers used]"""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_generation(query, retrieval_result, start_time)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer_text = response.content[0].text

        # Extract source ACNs from the answer
        source_acns = _extract_acns(answer_text, retrieval_result)

        # Build reasoning trace
        reasoning_trace = (
            f"Query entities: {retrieval_result.query_entities}\n"
            f"Chunks retrieved: {len(retrieval_result.chunks)}\n"
            f"Provenance mix: {_provenance_summary(retrieval_result)}\n"
            f"Sources cited: {source_acns}"
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            text=answer_text,
            source_acns=source_acns,
            reasoning_trace=reasoning_trace,
            latency_ms=elapsed_ms,
            model=model,
            metadata={
                "retrieval_latency_ms": retrieval_result.latency_ms,
                "total_latency_ms": elapsed_ms,
                "chunk_count": len(retrieval_result.chunks),
            },
        )

    except Exception as e:
        return _fallback_generation(query, retrieval_result, start_time, error=str(e))


def _fallback_generation(
    query: str,
    retrieval_result,
    start_time: float,
    error: Optional[str] = None,
) -> GenerationResult:
    """Fallback when Claude API is unavailable — return retrieved context directly."""
    chunks_text = "\n\n".join(
        f"[ACN {c.report_id}] {c.text}" for c in retrieval_result.chunks[:5]
    )

    source_acns = list({c.report_id for c in retrieval_result.chunks})

    text = f"**Retrieved evidence for:** {query}\n\n{chunks_text}"
    if error:
        text += f"\n\n(Note: LLM generation unavailable — {error}. Showing raw retrieved context.)"

    elapsed_ms = (time.time() - start_time) * 1000

    return GenerationResult(
        text=text,
        source_acns=source_acns,
        reasoning_trace=f"Fallback mode. Error: {error}" if error else "Fallback mode (no API key)",
        latency_ms=elapsed_ms,
        model="fallback",
    )


def _extract_acns(text: str, retrieval_result) -> list[str]:
    """Extract ACN numbers mentioned in the generated answer."""
    import re
    # Match ACN patterns in text
    mentioned = set(re.findall(r"ACN\s*(\d+)", text, re.IGNORECASE))
    # Also include all ACNs from retrieved chunks
    all_acns = {c.report_id for c in retrieval_result.chunks}
    # Return intersection + any directly mentioned
    return sorted(mentioned | (all_acns & mentioned) or all_acns)


def _provenance_summary(retrieval_result) -> str:
    """Summarize provenance distribution of retrieved chunks."""
    counts = {"vector": 0, "graph": 0, "both": 0}
    for chunk in retrieval_result.chunks:
        counts[chunk.provenance] = counts.get(chunk.provenance, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
