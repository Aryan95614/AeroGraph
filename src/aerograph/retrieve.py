"""Hybrid GraphRAG retrieval with Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Extract query entities via Claude or embedding-based linking
  2. Vector search — top_k*2 chunks from ChromaDB
  3. Graph expansion — 2-hop neighborhood per query entity
  4. RRF fusion across vector + graph scores
  5. Return top_k chunks with provenance tags

Extended with BM25, PPR, community-based global search, and HippoRAG
multi-signal fusion (Gutierrez et al., arXiv:2405.14831).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))

# RRF constant — standard value per Cormack et al. 2009
RRF_K = 60
# Tuned k for graph-heavy fusion. Lower k gives more weight to top ranks,
# which helps when graph signals are sparse but precise.
RRF_K_GRAPH = 45


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    report_id: str
    score: float
    provenance: str  # "vector" | "graph" | "both" | "bm25" | "ppr" | "multi"
    entities_mentioned: list[str] = field(default_factory=list)
    graph_context: str = ""


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    query_entities: list[str]
    latency_ms: float = 0.0
    retrieval_trace: dict = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Tunable knobs for the HippoRAG retriever."""
    use_vector: bool = True
    use_bm25: bool = True
    use_ppr: bool = True
    use_community: bool = False
    ppr_alpha: float = 0.15
    ppr_top_k: int = 20
    rrf_k: int = RRF_K_GRAPH
    community_top_k: int = 10
    final_top_k: int = 0


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_query_entities(query: str) -> list[str]:
    """Extract entity mentions from a query using Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
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
    proper_nouns = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", query)
    found.extend([n.lower() for n in proper_nouns if len(n) > 2])
    return list(set(found))


# ---------------------------------------------------------------------------
# Entity embedding index (for v2 entity linking)
# ---------------------------------------------------------------------------

def build_entity_index(
    graph,
    cache_path: Optional[Path] = None,
) -> tuple[list[str], np.ndarray]:
    """Build or load cached embedding index of all entity names in the graph."""
    if cache_path is None:
        cache_path = DATA_DIR / "entity_embeddings.npz"

    entity_names = sorted(graph.nodes())

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_names = list(data["names"])
        cached_embeddings = data["embeddings"]
        if cached_names == entity_names:
            return cached_names, cached_embeddings

    from aerograph.embed import get_embedding_model
    model = get_embedding_model()
    embeddings = model.encode(entity_names, show_progress_bar=False, convert_to_numpy=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, names=np.array(entity_names, dtype=object), embeddings=embeddings)

    return entity_names, embeddings


def extract_query_entities_v2(
    query: str,
    entity_names: list[str],
    embeddings: np.ndarray,
    model=None,
    similarity_threshold: float = 0.65,
) -> list[tuple[str, float]]:
    """Two-pass entity linking: exact match then embedding similarity."""
    if model is None:
        from aerograph.embed import get_embedding_model
        model = get_embedding_model()

    matched: dict[str, float] = {}
    query_lower = query.lower()

    # Pass 1: exact word-boundary match
    for name in entity_names:
        if len(name) <= 2:
            continue
        pattern = r'\b' + re.escape(name.lower()) + r'\b'
        if re.search(pattern, query_lower):
            matched[name] = 1.0

    # Pass 2: embedding similarity for unmatched entities
    query_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed_embs = embeddings / norms
    query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_norm = np.where(query_norm == 0, 1, query_norm)
    normed_query = query_emb / query_norm
    similarities = (normed_embs @ normed_query.T).flatten()

    for i, sim in enumerate(similarities):
        name = entity_names[i]
        if name not in matched and sim >= similarity_threshold:
            matched[name] = float(sim)

    results = sorted(matched.items(), key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Synonym / near-duplicate detection via embeddings
# ---------------------------------------------------------------------------

def detect_synonym_edges(
    graph,
    entity_names: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> list[tuple[str, str, float]]:
    """Find near-duplicate entities by embedding cosine similarity within same type."""
    type_map: dict[str, list[int]] = {}
    for i, name in enumerate(entity_names):
        ntype = graph.nodes[name].get("type", "unknown") if graph.has_node(name) else "unknown"
        type_map.setdefault(ntype, []).append(i)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms

    pairs = []
    for ntype, indices in type_map.items():
        if len(indices) < 2:
            continue
        sub = normed[indices]
        sims = sub @ sub.T
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                sim = float(sims[a_pos, b_pos])
                if sim >= threshold:
                    pairs.append((entity_names[indices[a_pos]], entity_names[indices[b_pos]], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Personalized PageRank search
# ---------------------------------------------------------------------------

def ppr_search(
    graph,
    seed_entities: list[tuple[str, float]],
    top_k: int = 20,
    alpha: float = 0.15,
) -> list[dict]:
    """Personalized PageRank over the graph, seeded on query entities."""
    import networkx as nx

    undirected = graph.to_undirected() if graph.is_directed() else graph

    personalization = {}
    seed_names = set()
    for name, weight in seed_entities:
        if undirected.has_node(name):
            personalization[name] = weight
            seed_names.add(name)

    if not personalization:
        return []

    try:
        ppr = nx.pagerank(undirected, alpha=alpha, personalization=personalization, max_iter=100)
    except Exception:
        return []

    scored = []
    for node, score in ppr.items():
        if node in seed_names:
            continue
        data = graph.nodes[node] if graph.has_node(node) else {}
        scored.append({
            "node_name": node,
            "ppr_score": score,
            "report_ids": data.get("report_ids", []),
            "type": data.get("type", "unknown"),
        })

    scored.sort(key=lambda x: x["ppr_score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Vector + graph search (original)
# ---------------------------------------------------------------------------

def vector_search(query: str, top_k: int = 20) -> list[dict]:
    """Retrieve top chunks via ChromaDB vector similarity."""
    from aerograph.embed import query_similar
    return query_similar(query, top_k=top_k)


def graph_search(
    entities: list[str],
    depth: int = 2,
    max_degree: int = 200,
) -> dict[str, float]:
    """Score report_ids by graph neighborhood overlap with query entities.

    Caps expansion at nodes with degree > max_degree to prevent hub node
    explosion (e.g., "b737" connecting to thousands of reports).
    """
    from aerograph.graph import detect_backend

    try:
        backend = detect_backend()
    except Exception:
        return {}

    report_scores: dict[str, float] = {}
    entity_report_ids: dict[str, set[str]] = {}

    for entity in entities:
        subgraph = backend.get_neighbors(entity, depth=depth, max_degree=max_degree)
        for node in subgraph.nodes:
            for rid in node.report_ids:
                entity_report_ids.setdefault(rid, set()).add(entity)

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
    for entity in entities[:5]:
        subgraph = backend.get_neighbors(entity, depth=1)
        if subgraph.nodes:
            neighbors = [f"{n.canonical_name} ({n.type})" for n in subgraph.nodes[:10]]
            lines.append(f"- {entity}: connected to {', '.join(neighbors)}")
        for edge in subgraph.edges[:5]:
            lines.append(f"  {edge.source} --[{edge.type}]--> {edge.target}")

    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# RRF fusion (original 2-signal)
# ---------------------------------------------------------------------------

def rrf_fusion(
    vector_results: list[dict],
    graph_scores: dict[str, float],
    k: int = RRF_K_GRAPH,
    vector_weight: float = 0.5,
    graph_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Fuse vector and graph retrieval scores using Reciprocal Rank Fusion."""
    chunk_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    chunk_provenance: dict[str, set[str]] = {}

    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0) + vector_weight / (k + rank + 1)
        chunk_data[cid] = result
        chunk_provenance.setdefault(cid, set()).add("vector")

    graph_ranked = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)
    report_to_chunks: dict[str, list[str]] = {}
    for result in vector_results:
        report_to_chunks.setdefault(result["report_id"], []).append(result["chunk_id"])

    for rank, (report_id, _score) in enumerate(graph_ranked):
        chunk_ids = report_to_chunks.get(report_id, [])
        if not chunk_ids:
            try:
                from aerograph.embed import get_collection
                collection = get_collection()
                results = collection.get(
                    where={"report_id": report_id},
                    include=["documents", "metadatas"],
                )
                for i, doc_id in enumerate(results["ids"]):
                    text = results["documents"][i] if results["documents"] else ""
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    chunk_data[doc_id] = {
                        "chunk_id": doc_id,
                        "text": text,
                        "report_id": report_id,
                        "entities_mentioned": json.loads(meta.get("entities_mentioned", "[]")),
                    }
                    chunk_ids.append(doc_id)
            except Exception:
                pass
        for cid in chunk_ids:
            chunk_scores[cid] = chunk_scores.get(cid, 0) + graph_weight / (k + rank + 1)
            chunk_provenance.setdefault(cid, set()).add("graph")

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


# ---------------------------------------------------------------------------
# Multi-signal RRF fusion (for HippoRAG-style pipelines)
# ---------------------------------------------------------------------------

def multi_rrf_fusion(
    signal_lists: list[tuple[str, list[dict]]],
    k: int = RRF_K_GRAPH,
    signal_weights: Optional[dict[str, float]] = None,
    top_k: int = 0,
) -> list[RetrievedChunk]:
    """Weighted RRF across N retrieval signals.

    signal_lists: [("vector", [chunk_dicts...]), ("bm25", [...]), ...]
    Each chunk_dict must have at least chunk_id, text, report_id.
    """
    if signal_weights is None:
        signal_weights = {}

    chunk_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    chunk_signals: dict[str, set[str]] = {}

    for signal_name, ranked_list in signal_lists:
        w = signal_weights.get(signal_name, 1.0)
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            chunk_scores[cid] = chunk_scores.get(cid, 0) + w / (k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = chunk
            chunk_signals.setdefault(cid, set()).add(signal_name)

    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in sorted_chunks:
        data = chunk_data.get(cid, {})
        signals = chunk_signals.get(cid, set())
        if len(signals) > 1:
            prov = "multi"
        else:
            prov = next(iter(signals), "unknown")

        results.append(RetrievedChunk(
            chunk_id=cid,
            text=data.get("text", ""),
            report_id=data.get("report_id", ""),
            score=score,
            provenance=prov,
            entities_mentioned=data.get("entities_mentioned", []),
        ))

    if top_k > 0:
        results = results[:top_k]
    return results


# ---------------------------------------------------------------------------
# Global / community-based search
# ---------------------------------------------------------------------------

_GLOBAL_PATTERNS = [
    r"most common",
    r"overall",
    r"trends?\s+across",
    r"how often",
    r"how frequent",
    r"what percentage",
    r"on average",
    r"typically",
    r"general\s+pattern",
    r"across all",
    r"summary of",
    r"statistics",
]

_GLOBAL_RE = re.compile("|".join(_GLOBAL_PATTERNS), re.IGNORECASE)


def is_global_query(query: str) -> bool:
    """Detect whether a query asks for corpus-level aggregation."""
    return bool(_GLOBAL_RE.search(query))


def global_search(
    query: str,
    community_summaries: dict[str, dict],
    top_k: int = 5,
) -> list[dict]:
    """BM25-style scoring of query against community summary texts."""
    if not community_summaries:
        return []

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens:
        return []

    scored = []
    for comm_id, info in community_summaries.items():
        summary = info.get("summary", "")
        summary_tokens = re.findall(r"[a-z0-9]+", summary.lower())
        if not summary_tokens:
            continue

        tf_counts = Counter(summary_tokens)
        doc_len = len(summary_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_counts.get(qt, 0)
            if tf > 0:
                score += (1 + math.log(1 + tf)) / (1 + math.log(1 + doc_len))

        scored.append({
            "community_id": comm_id,
            "score": score,
            "summary": summary,
            "report_ids": info.get("report_ids", []),
            "node_count": info.get("node_count", 0),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

def route_query(query: str) -> str:
    """Auto-select retriever type based on query characteristics."""
    if is_global_query(query):
        return "hipporag"

    causal_indicators = [
        r"caus(e|ed|es|al|ing)",
        r"led?\s+to",
        r"result(ed|s|ing)?\s+in",
        r"chain",
        r"because",
        r"contribut",
        r"factor",
        r"why\s+did",
        r"what\s+caused",
        r"how\s+did\s+.+\s+lead",
    ]
    causal_re = re.compile("|".join(causal_indicators), re.IGNORECASE)
    if causal_re.search(query):
        return "graphrag"

    comparative_re = re.compile(r"\bcompar|vs\.?\b|\bversus\b|\bdifference\b", re.IGNORECASE)
    if comparative_re.search(query):
        return "graphrag"

    return "graphrag"


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class GraphRAGRetriever:
    """Main retriever combining vector search, graph expansion, and RRF fusion."""

    def __init__(
        self,
        vector_weight: float = 0.45,
        graph_weight: float = 0.55,
        rrf_k: int = RRF_K_GRAPH,
    ):
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()

        query_entities = extract_query_entities(query)
        vector_results = vector_search(query, top_k=top_k * 2)
        graph_scores = graph_search(query_entities, depth=2) if query_entities else {}

        fused = rrf_fusion(
            vector_results, graph_scores,
            k=self.rrf_k,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight,
        )

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

        vector_results = vector_search(query, top_k=top_k * 2)
        elapsed_ms = (time.time() - start_time) * 1000

        chunks = [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                report_id=r["report_id"],
                score=1.0 / (i + 1),
                provenance="vector",
                entities_mentioned=r.get("entities_mentioned", []),
            )
            for i, r in enumerate(vector_results)
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks[:top_k],
            query_entities=[],
            latency_ms=elapsed_ms,
            retrieval_trace={"vector_results_count": len(vector_results)},
        )


class BM25Retriever:
    """Okapi BM25 lexical retriever, built from scratch over ChromaDB chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[dict] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._inverted_index: dict[str, list[tuple[int, int]]] = {}
        self._initialized = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        from aerograph.embed import get_collection

        collection = get_collection()
        all_data = collection.get(include=["documents", "metadatas"])

        self._docs = []
        for i, doc_id in enumerate(all_data["ids"]):
            text = all_data["documents"][i] if all_data["documents"] else ""
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            self._docs.append({
                "chunk_id": doc_id,
                "text": text,
                "report_id": meta.get("report_id", ""),
                "entities_mentioned": json.loads(meta["entities_mentioned"])
                if meta.get("entities_mentioned") else [],
            })

        n = len(self._docs)
        self._doc_freqs = {}
        self._inverted_index = {}
        self._doc_lens = []
        total_len = 0

        for idx, doc in enumerate(self._docs):
            tokens = self._tokenize(doc["text"])
            self._doc_lens.append(len(tokens))
            total_len += len(tokens)

            tf_counts = Counter(tokens)
            seen_terms: set[str] = set()
            for term, tf in tf_counts.items():
                self._inverted_index.setdefault(term, []).append((idx, tf))
                if term not in seen_terms:
                    self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
                    seen_terms.add(term)

        self._avgdl = total_len / n if n > 0 else 1.0
        self._initialized = True

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        n = len(self._docs)
        dl = self._doc_lens[doc_idx]
        score = 0.0

        for term in query_tokens:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            tf = 0
            for didx, t in self._inverted_index.get(term, []):
                if didx == doc_idx:
                    tf = t
                    break

            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            score += idf * tf_norm

        return score

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()

        if not self._initialized:
            self._build_index()

        query_tokens = self._tokenize(query)

        candidate_indices: set[int] = set()
        for term in query_tokens:
            for doc_idx, _tf in self._inverted_index.get(term, []):
                candidate_indices.add(doc_idx)

        scored = []
        for idx in candidate_indices:
            s = self._bm25_score(query_tokens, idx)
            if s > 0:
                scored.append((idx, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.time() - start_time) * 1000

        chunks = []
        for idx, s in scored[:top_k]:
            doc = self._docs[idx]
            chunks.append(RetrievedChunk(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                report_id=doc["report_id"],
                score=s,
                provenance="bm25",
                entities_mentioned=doc.get("entities_mentioned", []),
            ))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            query_entities=[],
            latency_ms=elapsed_ms,
            retrieval_trace={"bm25_candidates": len(candidate_indices)},
        )


class GraphOnlyRetriever:
    """Graph-only retriever: neighborhood expansion without vector similarity."""

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()

        query_entities = extract_query_entities(query)

        if not query_entities:
            elapsed_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query,
                chunks=[],
                query_entities=query_entities,
                latency_ms=elapsed_ms,
                retrieval_trace={"graph_report_ids": 0},
            )

        graph_scores = graph_search(query_entities, depth=2)
        ranked_reports = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)

        from aerograph.embed import get_collection
        collection = get_collection()

        chunks = []
        for report_id, g_score in ranked_reports:
            if len(chunks) >= top_k:
                break
            try:
                results = collection.get(
                    where={"report_id": report_id},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue

            for i, doc_id in enumerate(results["ids"]):
                if len(chunks) >= top_k:
                    break
                text = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}
                chunks.append(RetrievedChunk(
                    chunk_id=doc_id,
                    text=text,
                    report_id=report_id,
                    score=g_score,
                    provenance="graph",
                    entities_mentioned=json.loads(meta["entities_mentioned"])
                    if meta.get("entities_mentioned") else [],
                ))

        elapsed_ms = (time.time() - start_time) * 1000

        graph_context = get_graph_context(query_entities)
        for chunk in chunks:
            chunk.graph_context = graph_context

        return RetrievalResult(
            query=query,
            chunks=chunks,
            query_entities=query_entities,
            latency_ms=elapsed_ms,
            retrieval_trace={
                "graph_report_ids": len(graph_scores),
                "query_entities": query_entities,
            },
        )


class PPROnlyRetriever:
    """PPR-only retriever: Personalized PageRank without vector similarity."""

    def __init__(self, alpha: float = 0.15, ppr_top_k: int = 20):
        self.alpha = alpha
        self.ppr_top_k = ppr_top_k
        self._graph = None
        self._entity_names: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None

    def _ensure_graph(self):
        if self._graph is not None:
            return
        from aerograph.graph import load_graph
        self._graph = load_graph()
        self._entity_names, self._embeddings = build_entity_index(self._graph)
        from aerograph.embed import get_embedding_model
        self._model = get_embedding_model()

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()
        self._ensure_graph()

        linked = extract_query_entities_v2(
            query, self._entity_names, self._embeddings, self._model,
        )
        seed_entities = [(name, score) for name, score in linked[:10]]
        query_entity_names = [name for name, _ in seed_entities]

        ppr_results = ppr_search(
            self._graph, seed_entities,
            top_k=self.ppr_top_k, alpha=self.alpha,
        )

        report_ids_scored: dict[str, float] = {}
        for node_info in ppr_results:
            for rid in node_info["report_ids"]:
                report_ids_scored[rid] = max(
                    report_ids_scored.get(rid, 0),
                    node_info["ppr_score"],
                )

        ranked_reports = sorted(report_ids_scored.items(), key=lambda x: x[1], reverse=True)

        from aerograph.embed import get_collection
        collection = get_collection()

        chunks = []
        for report_id, ppr_score in ranked_reports:
            if len(chunks) >= top_k:
                break
            try:
                results = collection.get(
                    where={"report_id": report_id},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue
            for i, doc_id in enumerate(results["ids"]):
                if len(chunks) >= top_k:
                    break
                text = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}
                chunks.append(RetrievedChunk(
                    chunk_id=doc_id,
                    text=text,
                    report_id=report_id,
                    score=ppr_score,
                    provenance="ppr",
                    entities_mentioned=json.loads(meta["entities_mentioned"])
                    if meta.get("entities_mentioned") else [],
                ))

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=chunks,
            query_entities=query_entity_names,
            latency_ms=elapsed_ms,
            retrieval_trace={
                "ppr_nodes": len(ppr_results),
                "ppr_reports": len(report_ids_scored),
                "linked_entities": len(seed_entities),
            },
        )


class HippoRAGRetriever:
    """4-way fusion retriever: vector + BM25 + PPR + community."""

    SIGNAL_WEIGHTS = {
        "ppr": 1.0,
        "community": 1.0,
        "vector": 1.0,
        "bm25": 1.0,
    }

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._bm25 = BM25Retriever()
        self._graph = None
        self._entity_names: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None
        self._community_summaries: Optional[dict] = None

    def _ensure_graph(self):
        if self._graph is not None:
            return
        from aerograph.graph import load_graph
        self._graph = load_graph()
        self._entity_names, self._embeddings = build_entity_index(self._graph)
        from aerograph.embed import get_embedding_model
        self._model = get_embedding_model()

    def _load_community_summaries(self) -> dict:
        if self._community_summaries is not None:
            return self._community_summaries
        path = DATA_DIR / "community_summaries.json"
        if path.exists():
            with open(path) as f:
                self._community_summaries = json.load(f)
        else:
            self._community_summaries = {}
        return self._community_summaries

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start_time = time.time()
        self._ensure_graph()

        linked = extract_query_entities_v2(
            query, self._entity_names, self._embeddings, self._model,
        )
        seed_entities = [(name, score) for name, score in linked[:10]]
        query_entity_names = [name for name, _ in seed_entities]

        signal_lists: list[tuple[str, list[dict]]] = []

        if self.config.use_vector:
            vec_results = vector_search(query, top_k=top_k * 2)
            signal_lists.append(("vector", vec_results))

        if self.config.use_bm25:
            bm25_result = self._bm25.retrieve(query, top_k=top_k * 2)
            bm25_dicts = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "report_id": c.report_id,
                    "entities_mentioned": c.entities_mentioned,
                }
                for c in bm25_result.chunks
            ]
            signal_lists.append(("bm25", bm25_dicts))

        if self.config.use_ppr:
            ppr_results = ppr_search(
                self._graph, seed_entities,
                top_k=self.config.ppr_top_k, alpha=self.config.ppr_alpha,
            )
            ppr_report_ids: dict[str, float] = {}
            for node_info in ppr_results:
                for rid in node_info["report_ids"]:
                    ppr_report_ids[rid] = max(
                        ppr_report_ids.get(rid, 0),
                        node_info["ppr_score"],
                    )
            ranked_ppr = sorted(ppr_report_ids.items(), key=lambda x: x[1], reverse=True)

            ppr_chunks: list[dict] = []
            try:
                from aerograph.embed import get_collection
                collection = get_collection()
                for report_id, _ in ranked_ppr[:top_k]:
                    try:
                        results = collection.get(
                            where={"report_id": report_id},
                            include=["documents", "metadatas"],
                        )
                    except Exception:
                        continue
                    for i, doc_id in enumerate(results["ids"]):
                        text = results["documents"][i] if results["documents"] else ""
                        meta = results["metadatas"][i] if results["metadatas"] else {}
                        ppr_chunks.append({
                            "chunk_id": doc_id,
                            "text": text,
                            "report_id": report_id,
                            "entities_mentioned": json.loads(meta["entities_mentioned"])
                            if meta.get("entities_mentioned") else [],
                        })
            except Exception:
                pass
            if ppr_chunks:
                signal_lists.append(("ppr", ppr_chunks))

        if self.config.use_community:
            summaries = self._load_community_summaries()
            if summaries:
                comm_results = global_search(query, summaries, top_k=self.config.community_top_k)
                comm_chunks: list[dict] = []
                try:
                    from aerograph.embed import get_collection
                    collection = get_collection()
                    for cr in comm_results:
                        for rid in cr.get("report_ids", [])[:3]:
                            try:
                                results = collection.get(
                                    where={"report_id": rid},
                                    include=["documents", "metadatas"],
                                )
                            except Exception:
                                continue
                            for i, doc_id in enumerate(results["ids"]):
                                text = results["documents"][i] if results["documents"] else ""
                                meta = results["metadatas"][i] if results["metadatas"] else {}
                                comm_chunks.append({
                                    "chunk_id": doc_id,
                                    "text": text,
                                    "report_id": rid,
                                    "entities_mentioned": json.loads(meta["entities_mentioned"])
                                    if meta.get("entities_mentioned") else [],
                                })
                except Exception:
                    pass
                if comm_chunks:
                    signal_lists.append(("community", comm_chunks))

        fused = multi_rrf_fusion(
            signal_lists,
            k=self.config.rrf_k,
            signal_weights=self.SIGNAL_WEIGHTS,
            top_k=self.config.final_top_k if self.config.final_top_k > 0 else top_k,
        )

        graph_context = get_graph_context(query_entity_names)
        for chunk in fused:
            chunk.graph_context = graph_context

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=fused,
            query_entities=query_entity_names,
            latency_ms=elapsed_ms,
            retrieval_trace={
                "signals": [name for name, _ in signal_lists],
                "signal_counts": {name: len(lst) for name, lst in signal_lists},
                "linked_entities": len(seed_entities),
                "fused_count": len(fused),
            },
        )
