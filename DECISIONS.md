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

## AD-008: BM25 Baseline — From-Scratch Implementation

Implemented Okapi BM25 without external library dependencies (no rank-bm25 or
Pyserini) to keep the dependency footprint minimal. The implementation builds
an inverted index at query time from the ChromaDB chunk corpus and uses
standard IDF/TF saturation scoring (k1=1.5, b=0.75).

## AD-009: 4-System Ablation Study

Evaluation runs 4 retrieval systems (GraphRAG, Vector-Only, BM25, Graph-Only)
rather than just GraphRAG vs. baseline. This isolates the contribution of each
retrieval signal and strengthens the empirical claims. All systems use the same
generation backend for fair comparison.

## AD-010: Hub Node Explosion Prevention

Graph expansion (2-hop BFS) skips nodes with degree > 200 to prevent hub
entities like "b737" from pulling in thousands of reports. The max_degree
parameter is configurable per query. This trades recall for tractability.

## AD-011: Number-Aware Entity Deduplication

Entity normalization skips merging entities that differ only by a numeric
identifier (e.g., "engine 1 failure" vs "engine 2 failure", "runway 28L" vs
"runway 28R"). The _differ_only_by_number check uses regex to detect
alphanumeric tokens and prevent false merges.

## AD-012: Temporal Graph Edges

Added TEMPORAL_SEQUENCE edge type and TimePeriod entity type to capture
narrative temporal ordering. Temporal chains use topological sort over
directed temporal edges. This enables within-report timeline reconstruction
via get_report_timeline() and cross-report temporal pattern analysis.

## AD-013: Statistical Reporting

All evaluation metrics report mean, standard deviation, and 95% confidence
intervals. Figures include error bars. This addresses the concern that
point estimates without variance are uninterpretable for n=50 queries.

## AD-014: Real Data Migration — HuggingFace Hub

Switched from synthetic data to 2,000 real NASA ASRS reports sourced from
elihoole/asrs-aviation-reports on HuggingFace Hub. The dataset provides
pre-structured fields (ACN, narrative, synopsis, contributing factors)
that map directly to our ingestion schema. Synthetic generation retained
as a fallback for offline environments.

## AD-015: Gradio for Demo Interface

Chose Gradio over Streamlit for the public-facing demo. Gradio integrates
natively with HuggingFace Spaces (zero-config deployment), provides built-in
API endpoints for programmatic access, and handles the graph explorer
visualization with standard Plotly components.

## AD-016: Modal Serverless Deployment

Modal provides CPU autoscaling with cold-start under 5 seconds. The
deployment wraps the FastAPI server, entity extraction pipeline, and graph
build as independent Modal functions. Chosen over AWS Lambda for Python
dependency compatibility and over Railway for cost structure.
