# Methods

## 3.1 System Overview

AeroGraph is a GraphRAG system for multi-hop causal reasoning over aviation
safety incident reports. The pipeline consists of five stages: (1) data
ingestion and normalization, (2) LLM-based entity and relation extraction,
(3) knowledge graph construction, (4) hybrid retrieval via Reciprocal Rank
Fusion, and (5) grounded answer generation. We describe each stage below.

## 3.2 Data Ingestion

We use 2,000 real NASA ASRS (Aviation Safety Reporting System) incident reports
sourced from the elihoole/asrs-aviation-reports dataset on HuggingFace
(38,655 total records). ASRS is the largest voluntary safety reporting database
in aviation, containing narrative descriptions of safety events from pilots,
air traffic controllers, and maintenance personnel. Each report includes
structured metadata (aircraft type, phase of flight, anomaly type) alongside
free-text narratives describing the incident and contributing factors.

Reports are imported with both reporter narratives (Report 1 and Report 2)
combined into a single text field, along with structured metadata fields
including aircraft make/model, flight phase, and anomaly classification. The
ingestion pipeline normalizes aircraft type strings to canonical forms
(e.g., "Boeing 737-800", "B737-800", "737" all map to "B737"), extracts
structured metadata fields, and filters reports with narratives shorter than
50 words. Float-format ACN identifiers from the source dataset are
normalized to integers for consistent cross-referencing.

## 3.3 Entity and Relation Extraction

We define a domain-specific aviation safety ontology comprising 10 node types
(Aircraft, Event, Phase, Factor, Component, Outcome, Recommendation,
ATC_Facility, Weather, TimePeriod) and 8 edge types (CAUSED_BY, CONTRIBUTED_TO,
OCCURRED_DURING, INVOLVED, RESOLVED_BY, PRECEDED_BY, CO_OCCURRED_WITH,
TEMPORAL_SEQUENCE). This ontology is intentionally coarser than formal aviation
taxonomies (e.g., ECCAIRS) to remain within the extraction capabilities of
current LLMs.

For each report, we prompt Claude Sonnet with the full ontology specification
and the report text, requesting structured JSON output containing extracted
entities (with type and canonical name) and relations (with source, target, and
edge type). The extraction prompt explicitly instructs temporal ordering:
when the narrative describes events in sequence ("X happened before Y", "after
the go-around, Z occurred"), the model produces TEMPORAL_SEQUENCE edges from
the earlier event to the later event.

We process reports in batches of 20 with exponential backoff retry logic
(tenacity library) for API rate limit handling. Results are cached to enable
idempotent re-runs — already-processed reports are skipped on re-execution.

Post-extraction normalization applies two stages:

1. **Name canonicalization:** Lowercase, whitespace collapsing, trailing
   punctuation removal, and aviation abbreviation normalization (e.g.,
   "frequency" → "freq").

2. **Fuzzy deduplication:** Within each entity type, we merge entities whose
   canonical names exceed a SequenceMatcher similarity threshold of 0.85. This
   collapses near-duplicates like "bird strike" / "bird strikes" / "birdstrike"
   into a single canonical entity. Critically, we apply a number-aware guard:
   entities that differ only by a numeric or alphanumeric identifier (e.g.,
   "engine 1 failure" vs. "engine 2 failure", "runway 28L" vs. "runway 28R")
   are never merged, regardless of string similarity.

## 3.4 Knowledge Graph Construction

Extracted entities and relations are merged into a directed knowledge graph
using MERGE semantics: nodes are keyed on (type, canonical_name) and edges on
(source, target, type). Duplicate edges increment a weight counter and
accumulate report ID lists, enabling frequency-based analysis.

We implement two interchangeable graph backends:

- **Neo4j** (preferred): Accessed via the official Python driver at
  bolt://localhost:7687. Provides production-grade Cypher queries, ACID
  transactions, and native graph traversal.

- **NetworkX** (fallback): A zero-dependency in-process directed graph stored
  as a pickle file. Automatically selected when Neo4j is unreachable.

Both backends expose the same query interface: `get_neighbors()` (BFS with
configurable depth and max_degree filter), `get_causal_chain()` (shortest
causal paths via CAUSED_BY/CONTRIBUTED_TO/PRECEDED_BY edges),
`get_high_centrality_nodes()` (PageRank), `get_subgraph()` (per-report entity
subgraph), and `get_temporal_chain()` (topological sort over TEMPORAL_SEQUENCE
edges).

The max_degree parameter (default 200) in `get_neighbors()` prevents hub node
explosion: when expanding the 2-hop neighborhood, nodes with degree exceeding
the threshold are skipped during BFS traversal. This prevents high-frequency
entities like common aircraft types from pulling in thousands of weakly
related reports.

## 3.5 Hybrid Retrieval

AeroGraph's retrieval pipeline combines three signals:

**Vector search.** Report text is chunked into 256-token overlapping windows
(128-token overlap) and embedded with sentence-transformers all-MiniLM-L6-v2
(384-dimensional embeddings). Chunks are stored in ChromaDB with metadata
linking each chunk to its source report and mentioned entities. At query time,
the top 2k chunks (where k is the requested result count) are retrieved by
cosine similarity.

**Graph expansion.** Query entities are extracted (via Claude or keyword
fallback), and for each entity, a 2-hop neighborhood is expanded in the
knowledge graph. Report IDs appearing in the neighborhood are scored by the
fraction of query entities they overlap with: `score = |matched_entities| /
|query_entities|`.

**Reciprocal Rank Fusion (RRF).** Vector and graph scores are fused using
RRF (Cormack et al., 2009):

$$\text{RRF}(d) = \sum_{s \in \{v, g\}} \frac{w_s}{k + \text{rank}_s(d)}$$

where $k = 45$, $w_v = 0.45$ (vector weight), and $w_g = 0.55$ (graph weight).
The graph weight exceeds the vector weight based on empirical tuning: a 0.55
graph weight improved multi-hop causal reasoning accuracy by approximately 8%
over equal weighting, with less than 2% regression on single-hop faithfulness.

For ablation, we implement three additional retrievers: **Vector-Only**
(ChromaDB semantic similarity only), **BM25** (from-scratch Okapi BM25 with
k1=1.5, b=0.75 over the chunk corpus), and **Graph-Only** (graph neighborhood
expansion without vector similarity).

## 3.6 Answer Generation

Retrieved chunks and graph context (entity neighborhood summaries) are passed
to Claude Sonnet with a system prompt establishing the role of an aviation
safety expert. The generation prompt instructs the model to ground all claims
in the provided evidence, cite specific ACN (ASRS report) numbers, trace
causal chains when evidence supports them, and explicitly note when evidence
is insufficient. The response includes structured sections for analysis,
causal factors, and source citations.

## 3.7 Evaluation

We construct a 50-query benchmark suite: 20 single-hop factual queries
(e.g., "What aircraft type was involved in the most bird strike incidents?"),
20 multi-hop causal queries (e.g., "What causal chain links bird strikes to
engine failure and subsequent go-around decisions?"), and 10 comparative
queries (e.g., "Compare contributing factors in B737 vs. A320 engine failure
incidents").

All four retrieval systems are evaluated on each query. Metrics:

- **Faithfulness** (Claude-as-judge, 0-1): Are claims in the answer supported
  by the retrieved context?
- **Answer relevance** (Claude-as-judge, 0-1): Does the answer directly address
  the question?
- **Context precision** (deterministic): Fraction of retrieved chunks from
  relevant reports.
- **Context recall** (deterministic): Fraction of relevant reports represented
  in retrieval.
- **Causal chain accuracy** (heuristic, multi-hop only): Presence of causal
  language indicators in the answer.
- **Reference similarity** (token-level ROUGE-L F1): Longest common subsequence
  overlap with reference answers.

We report all metrics with mean, standard deviation, and 95% confidence
intervals across the query set.

**Evaluation transparency.** The same LLM (Claude Sonnet) performs entity
extraction, answer generation, and answer judging. This circularity may inflate
faithfulness scores; we report them for relative comparison across retrieval
systems rather than absolute quality claims. Reference answers were generated
by Claude over the full corpus and are LLM-generated references, not human
gold-standard annotations. Future work should validate with a held-out judge
model or human annotators.
