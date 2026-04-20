# awesome-modal PR

## List entry to add

Add under the "AI / ML" or "RAG" section:

```markdown
- [AeroGraph](https://github.com/AryanDhawan/AeroGraph) - GraphRAG over 2,000 NASA ASRS aviation safety reports. Knowledge graph with 29K entities, hybrid retrieval via RRF fusion, 4-system ablation eval. Deploys FastAPI + Gradio on Modal with persistent volumes.
```

## PR title

Add AeroGraph: GraphRAG over aviation safety incident reports

## PR body

```markdown
**AeroGraph** is a GraphRAG system that builds a domain-specific knowledge graph from NASA Aviation Safety Reporting System (ASRS) incident reports for multi-hop causal reasoning.

### What it does
- Extracts 29K+ entities and 43K+ relations from 2,000 real ASRS reports using Claude Sonnet
- Constructs a knowledge graph with aviation safety ontology (10 entity types, 8 edge types)
- Hybrid retrieval: vector search (ChromaDB) + graph traversal (NetworkX) fused via Reciprocal Rank Fusion
- 4-system ablation evaluation (GraphRAG, Vector-only, BM25, Graph-only)

### Modal integration
- `modal_app.py` deploys FastAPI server + Gradio demo as serverless endpoints
- Persistent Modal Volume for graph data and embeddings
- Extraction and graph build can run as Modal functions for scalability
- Uses `modal.Secret` for Anthropic API key management

### Links
- [GitHub](https://github.com/AryanDhawan/AeroGraph)
- [HuggingFace Dataset](https://huggingface.co/datasets/AryanDhawan/aerograph-asrs)
- [Live Demo (HuggingFace Spaces)](https://huggingface.co/spaces/AryanDhawan/aerograph)
```

## Steps to submit

1. Fork https://github.com/modal-labs/awesome-modal
2. Add the entry in alphabetical order under the appropriate section
3. Open PR with the title and body above
