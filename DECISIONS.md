# AeroGraph Architecture Decisions

## AD-001: Graph Store — Neo4j with NetworkX Fallback

Neo4j provides production-grade graph traversal and Cypher queries, but requires
Docker. NetworkX serves as a zero-dependency fallback for development and CI.
Graph module auto-detects Neo4j availability at runtime.

## AD-002: Vector Store — ChromaDB (Embedded)

ChromaDB runs in-process with no external server. Good enough for our corpus size
(~5k reports) and avoids infrastructure overhead vs. Pinecone/Weaviate.

## AD-003: Embeddings — Local sentence-transformers

all-MiniLM-L6-v2 runs locally with zero API cost. 384-dim embeddings are
sufficient for semantic chunk retrieval at our scale.

## AD-004: LLM — Claude claude-sonnet-4-20250514 via Anthropic API

Used for entity extraction and answer generation. Structured JSON output mode
for extraction reliability. ANTHROPIC_API_KEY from environment.

## AD-005: Retrieval — Reciprocal Rank Fusion (RRF)

Hybrid retrieval combining vector similarity and graph neighborhood scoring.
RRF is parameter-light (single k constant) and empirically robust for
heterogeneous score distributions.

## AD-006: Evaluation — Custom Claude-as-Judge

RAGAS dependency proved fragile. Custom evaluation with Claude as faithfulness
and relevance judge, plus deterministic context precision/recall metrics.

## AD-007: Data — Synthetic ASRS Fallback

ASRS search API is unreliable for bulk download. Synthetic reports generated
with Claude as hard fallback, clearly labeled in all downstream artifacts.
