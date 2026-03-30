# AeroGraph: Graph-Augmented Retrieval for Multi-Hop Causal Reasoning over Aviation Safety Reports

## Abstract

Aviation safety analysis requires reasoning over complex causal chains spanning
multiple incident reports — a task at which standard retrieval-augmented
generation (RAG) systems consistently underperform. We present AeroGraph, a
GraphRAG system that constructs a domain-specific knowledge graph from NASA
Aviation Safety Reporting System (ASRS) incident reports using LLM-based entity
and relation extraction guided by an aviation safety ontology. AeroGraph
combines vector similarity search over report chunks with graph-structural
retrieval via entity neighborhood expansion, fusing both signals through
Reciprocal Rank Fusion (RRF) to surface evidence that spans report boundaries.
We evaluate on a 50-query benchmark suite comprising single-hop factual,
multi-hop causal, and comparative questions, measuring faithfulness, answer
relevance, context precision, and causal chain accuracy. Our results show that
GraphRAG retrieval improves multi-hop causal reasoning accuracy over a
vector-only baseline while maintaining competitive single-hop performance. We
analyze failure modes, discuss the limitations of synthetic evaluation data, and
release the full pipeline — ingestion, extraction, graph construction, hybrid
retrieval, and evaluation — as an open-source toolkit for safety-critical domain
RAG research.
