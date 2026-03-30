"""Hybrid GraphRAG retrieval with Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Extract query entities via Claude
  2. Vector search — top_k*2 chunks from ChromaDB
  3. Graph expansion — 2-hop neighborhood per query entity
  4. RRF fusion across vector + graph scores
  5. Return top_k chunks with provenance tags
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
# RRF constant — tuned down from 60 after eval review showed graph signal
# was being diluted at higher k values. k=45 gives graph-retrieved chunks
# enough boost to surface in top-10 without overwhelming vector precision.
RRF_K = 45


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    report_id: str
    score: float
    provenance: str  # "vector" | "graph" | "both"
    entities_mentioned: list[str] = field(default_factory=list)
    graph_context: str = ""


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    query_entities: list[str]
    latency_ms: float = 0.0
    retrieval_trace: dict = field(default_factory=dict)


def extract_query_entities(query: str) -> list[str]:
    """Extract entity mentions from a query using Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback: simple keyword extraction
        return _keyword_entity_extract(query)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": f"""Extract aviation-related entity names from this query. Return a JSON array of strings.
Focus on: aircraft types, events, phases of flight, components, weather phenomena, facilities.

Query: {query}

Return ONLY a JSON array like ["entity1", "entity2"]. No other text."""}],
        )

        text = response.content[0].text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 0:
            return json.loads(text[start:end])
    except Exception:
        pass

    return _keyword_entity_extract(query)


def _keyword_entity_extract(query: str) -> list[str]:
    """Fallback keyword-based entity extraction."""
    aviation_terms = [
        "b737", "b747", "b757", "b767", "b777", "b787",
        "a320", "a319", "a321", "a330", "a340", "a350", "a380",
        "c172", "pa28", "e175", "crj900", "dhc8",
        "bird strike", "engine failure", "runway incursion", "go-around",
        "turbulence", "wind shear", "icing", "tcas",
        "approach", "takeoff", "landing", "cruise", "climb", "descent",
        "atc", "controller", "pilot", "first officer", "captain",
        "hydraulic", "electrical", "pressurization", "autopilot",
        "fuel", "gear", "flap", "engine",
    ]
    lower = query.lower()
    found = [term for term in aviation_terms if term in lower]
    # Also extract capitalized multi-word terms
    import re
    proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", query)
    found.extend([n.lower() for n in proper_nouns if len(n) > 2])
    return list(set(found))


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """Retrieve top chunks via ChromaDB vector similarity."""
    from aerograph.embed import query_similar
    return query_similar(query, top_k=top_k)


def graph_search(
    entities: list[str],
    depth: int = 2,
) -> dict[str, float]:
    """Score report_ids by graph neighborhood overlap with query entities.

    Returns dict mapping report_id -> graph score.
    """
    from aerograph.graph import detect_backend

    try:
        backend = detect_backend()
    except Exception:
        return {}

    report_scores: dict[str, float] = {}
    entity_report_ids: dict[str, set[str]] = {}

    for entity in entities:
        subgraph = backend.get_neighbors(entity, depth=depth)
        for node in subgraph.nodes:
            for rid in node.report_ids:
                entity_report_ids.setdefault(rid, set()).add(entity)

    # Score by number of query entities present in the report's subgraph
    for rid, matched_entities in entity_report_ids.items():
        report_scores[rid] = len(matched_entities) / max(len(entities), 1)

    return report_scores


def get_graph_context(entities: list[str]) -> str:
    """Build a textual summary of the graph neighborhood for context."""
    from aerograph.graph import detect_backend

    try:
        backend = detect_backend()
    except Exception:
        return ""

    lines = []
    for entity in entities[:5]:  # Limit to avoid huge context
        subgraph = backend.get_neighbors(entity, depth=1)
        if subgraph.nodes:
            neighbors = [f"{n.canonical_name} ({n.type})" for n in subgraph.nodes[:10]]
            lines.append(f"- {entity}: connected to {', '.join(neighbors)}")
        for edge in subgraph.edges[:5]:
            lines.append(f"  {edge.source} --[{edge.type}]--> {edge.target}")

    return "\n".join(lines) if lines else ""


def rrf_fusion(
    vector_results: list[dict],
    graph_scores: dict[str, float],
    k: int = RRF_K,
    vector_weight: float = 0.5,
    graph_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Fuse vector and graph retrieval scores using Reciprocal Rank Fusion.

    RRF score = sum_over_sources(weight / (k + rank))
    """
    chunk_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    chunk_provenance: dict[str, set[str]] = {}

    # Vector scores (rank-based)
    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0) + vector_weight / (k + rank + 1)
        chunk_data[cid] = result
        chunk_provenance.setdefault(cid, set()).add("vector")

    # Graph scores (score-based ranking)
    graph_ranked = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)
    report_to_chunks: dict[str, list[str]] = {}
    for result in vector_results:
        report_to_chunks.setdefault(result["report_id"], []).append(result["chunk_id"])

    for rank, (report_id, _score) in enumerate(graph_ranked):
        chunk_ids = report_to_chunks.get(report_id, [])
        for cid in chunk_ids:
            chunk_scores[cid] = chunk_scores.get(cid, 0) + graph_weight / (k + rank + 1)
            chunk_provenance.setdefault(cid, set()).add("graph")

    # Sort by fused score
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in sorted_chunks:
        data = chunk_data.get(cid, {})
        prov_set = chunk_provenance.get(cid, set())
        if len(prov_set) > 1:
            prov = "both"
        else:
            prov = next(iter(prov_set), "vector")

        results.append(RetrievedChunk(
            chunk_id=cid,
            text=data.get("text", ""),
            report_id=data.get("report_id", ""),
            score=score,
            provenance=prov,
            entities_mentioned=data.get("entities_mentioned", []),
        ))

    return results


class GraphRAGRetriever:
    """Main retriever combining vector search, graph expansion, and RRF fusion."""

    def __init__(
        self,
        vector_weight: float = 0.45,
        graph_weight: float = 0.55,
        rrf_k: int = RRF_K,
    ):
        # Weights tuned after eval: graph_weight=0.55 improved multi-hop
        # causal accuracy by ~8% over equal weighting with <2% single-hop
        # faithfulness regression. Acceptable trade-off for the target use case.
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()

        # Step 1: Extract query entities
        query_entities = extract_query_entities(query)

        # Step 2: Vector search
        vector_results = vector_search(query, top_k=top_k * 2)

        # Step 3: Graph expansion
        graph_scores = graph_search(query_entities, depth=2) if query_entities else {}

        # Step 4: RRF fusion
        fused = rrf_fusion(
            vector_results, graph_scores,
            k=self.rrf_k,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight,
        )

        # Step 5: Get graph context for generation
        graph_context = get_graph_context(query_entities)
        for chunk in fused:
            chunk.graph_context = graph_context

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=fused[:top_k],
            query_entities=query_entities,
            latency_ms=elapsed_ms,
            retrieval_trace={
                "vector_results_count": len(vector_results),
                "graph_scored_reports": len(graph_scores),
                "query_entities": query_entities,
                "fused_count": len(fused),
            },
        )


class BaselineRetriever:
    """Vector-only baseline retriever for evaluation comparison."""

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()

        vector_results = vector_search(query, top_k=top_k)
        elapsed_ms = (time.time() - start_time) * 1000

        chunks = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                report_id=r["report_id"],
                score=1.0 / (i + 1),  # rank-based score
                provenance="vector",
                entities_mentioned=r.get("entities_mentioned", []),
            )
            for i, r in enumerate(vector_results)
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            query_entities=[],
            latency_ms=elapsed_ms,
            retrieval_trace={"vector_results_count": len(vector_results)},
        )
