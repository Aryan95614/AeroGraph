"""FastAPI server for AeroGraph queries.

Endpoints:
  POST /query      — Run a GraphRAG query
  GET  /graph/entity/{name} — Entity neighborhood as JSON
  GET  /stats      — Node/edge/report counts by type
  GET  /health     — Liveness check
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AeroGraph",
    description="GraphRAG over aviation safety incident reports",
    version="0.1.0",
)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=10, ge=1, le=50)


class SourceInfo(BaseModel):
    acn: str
    text_preview: str
    provenance: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    retrieval_trace: dict
    latency_ms: float


class EntityNeighborhood(BaseModel):
    entity: str
    entity_type: str
    neighbors: list[dict]
    edges: list[dict]


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    total_reports: int


class HealthResponse(BaseModel):
    status: str
    version: str
    graph_available: bool
    index_available: bool


# --- Endpoints ---

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run a GraphRAG query over the aviation safety knowledge base."""
    start_time = time.time()

    try:
        from aerograph.retrieve import GraphRAGRetriever
        from aerograph.generate import generate_answer

        retriever = GraphRAGRetriever()
        retrieval_result = retriever.retrieve(req.question, top_k=req.top_k)
        gen_result = generate_answer(req.question, retrieval_result)

        sources = [
            SourceInfo(
                acn=chunk.report_id,
                text_preview=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                provenance=chunk.provenance,
            )
            for chunk in retrieval_result.chunks
        ]

        total_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            answer=gen_result.text,
            sources=sources,
            retrieval_trace=retrieval_result.retrieval_trace,
            latency_ms=total_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/graph/entity/{name}", response_model=EntityNeighborhood)
def get_entity(name: str):
    """Get entity neighborhood from the knowledge graph."""
    try:
        from aerograph.graph import detect_backend

        backend = detect_backend()
        node = backend.get_node(name)
        if not node:
            raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

        subgraph = backend.get_neighbors(name, depth=1)

        neighbors = [
            {
                "name": n.canonical_name,
                "type": n.type,
                "report_count": len(n.report_ids),
            }
            for n in subgraph.nodes
            if n.canonical_name != name.lower().strip()
        ]

        edges = [
            {
                "source": e.source,
                "target": e.target,
                "type": e.type,
                "weight": e.weight,
            }
            for e in subgraph.edges
        ]

        return EntityNeighborhood(
            entity=node.canonical_name,
            entity_type=node.type,
            neighbors=neighbors,
            edges=edges,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=GraphStats)
def get_stats():
    """Get graph statistics: node/edge/report counts by type."""
    try:
        from aerograph.graph import detect_backend

        backend = detect_backend()
        node_types = backend.get_type_counts() if hasattr(backend, "get_type_counts") else {}
        edge_types = backend.get_edge_type_counts() if hasattr(backend, "get_edge_type_counts") else {}

        # Count unique reports
        all_report_ids = set()
        if hasattr(backend, "graph"):
            for _, data in backend.graph.nodes(data=True):
                all_report_ids.update(data.get("report_ids", []))

        return GraphStats(
            total_nodes=backend.node_count(),
            total_edges=backend.edge_count(),
            node_types=node_types,
            edge_types=edge_types,
            total_reports=len(all_report_ids),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check."""
    graph_ok = False
    index_ok = False

    try:
        from aerograph.graph import detect_backend
        backend = detect_backend()
        graph_ok = backend.node_count() > 0
    except Exception:
        pass

    try:
        from aerograph.embed import get_chroma_client, COLLECTION_NAME
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
        index_ok = collection.count() > 0
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="0.1.0",
        graph_available=graph_ok,
        index_available=index_ok,
    )
