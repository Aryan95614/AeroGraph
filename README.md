# AeroGraph

**GraphRAG over aviation safety incident reports for causal reasoning and safety pattern extraction.**

AeroGraph demonstrates that graph-structured retrieval outperforms vanilla RAG on multi-hop causal queries in the aviation safety domain. Built on NASA ASRS (Aviation Safety Reporting System) incident reports.

```
                    ┌──────────────────────────────────────────────────┐
                    │                  AeroGraph                       │
                    │                                                  │
  ASRS Reports      │   ┌─────────┐    ┌───────────┐    ┌──────────┐  │
  ──────────────────►   │ Ingest  │───►│  Extract  │───►│  Graph   │  │
                    │   │ (clean) │    │ (Claude)  │    │(Neo4j/NX)│  │
                    │   └─────────┘    └───────────┘    └────┬─────┘  │
                    │                                        │        │
                    │   ┌─────────┐    ┌───────────┐         │        │
                    │   │ Embed   │───►│ ChromaDB  │─────────┤        │
                    │   │ (SBERT) │    │ (vectors) │         │        │
                    │   └─────────┘    └───────────┘         │        │
                    │                                        ▼        │
  Query             │                  ┌───────────┐    ┌──────────┐  │
  ──────────────────►──────────────────►  Retrieve │───►│ Generate │──►── Answer
                    │                  │ (RRF fuse)│    │ (Claude) │  │
                    │                  └───────────┘    └──────────┘  │
                    │                                                  │
                    │   ┌─────────┐    ┌───────────┐                  │
                    │   │  Eval   │───►│  Figures  │                  │
                    │   │(50-Q)   │    │  (paper)  │                  │
                    │   └─────────┘    └───────────┘                  │
                    └──────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/aerograph.git
cd aerograph
make install

# 2. Set up environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 3. Run the full pipeline
make ingest    # Download/generate ASRS reports
make build     # Extract entities + build knowledge graph
make eval      # Run 50-query evaluation benchmark
make paper     # Generate figures and results tables

# 4. Start the API server
make serve     # FastAPI on http://localhost:8000
```

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Ingestion** | Python + httpx | ASRS download with synthetic fallback |
| **Extraction** | Claude claude-sonnet-4-20250514 | Entity/relation extraction with aviation ontology |
| **Graph Store** | Neo4j / NetworkX | Knowledge graph with causal edges |
| **Vector Store** | ChromaDB | Semantic chunk retrieval |
| **Embeddings** | all-MiniLM-L6-v2 | Local sentence embeddings (384-dim) |
| **Retrieval** | RRF Fusion | Hybrid vector + graph scoring |
| **Generation** | Claude claude-sonnet-4-20250514 | Answer generation with source tracing |
| **Evaluation** | Claude-as-Judge | Faithfulness, relevance, causal accuracy |
| **API** | FastAPI | REST endpoints for queries and graph exploration |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Run a GraphRAG query |
| GET | `/graph/entity/{name}` | Entity neighborhood as JSON |
| GET | `/stats` | Graph node/edge/report counts |
| GET | `/health` | Liveness check |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What causal chain links bird strikes to engine failure?", "top_k": 5}'
```

## Evaluation

The benchmark suite contains 50 queries:
- **20 single-hop factual** — direct lookups against the corpus
- **20 multi-hop causal** — requires reasoning across multiple reports
- **10 comparative** — cross-entity or cross-category analysis

Metrics: faithfulness, answer relevance, context precision, context recall, causal chain accuracy.

## Project Structure

```
aerograph/
├── src/aerograph/
│   ├── ingest.py      # ASRS data download + cleaning
│   ├── extract.py     # LLM entity/relation extraction
│   ├── graph.py       # Neo4j/NetworkX graph construction
│   ├── embed.py       # ChromaDB chunking + embedding
│   ├── retrieve.py    # Hybrid GraphRAG retrieval (RRF)
│   ├── generate.py    # Claude answer generation
│   ├── eval.py        # Evaluation framework + figures
│   └── api.py         # FastAPI server
├── scripts/
│   ├── build_graph.py # Graph construction entry point
│   ├── run_eval.py    # Evaluation entry point
│   └── demo.py        # Demo with 5 example queries
├── tests/
├── paper/
│   ├── abstract.md
│   ├── results_table.md
│   ├── figures/
│   └── results/
├── data/
│   ├── raw/
│   ├── processed/
│   └── graphs/
├── pyproject.toml
├── Makefile
└── DECISIONS.md
```

## Limitations

See [Limitations](#limitations-1) section below.

### Limitations

1. **Synthetic data caveat**: If ASRS download fails, the system falls back to
   synthetic reports generated by Claude. These lack the nuance and diversity of
   real incident reports. All downstream metrics should be interpreted with this
   in mind.

2. **Evaluation with LLM-as-judge**: Faithfulness and relevance scores use
   Claude as evaluator, not human annotations. This creates a circular
   dependency when Claude also generates the answers. We report these metrics
   transparently but do not claim they substitute for human evaluation.

3. **Graph coverage**: Entity extraction quality depends on Claude's domain
   understanding. Extraction errors propagate to the graph structure and
   downstream retrieval.

4. **Single embedding model**: all-MiniLM-L6-v2 is a general-purpose model, not
   fine-tuned on aviation text. Domain-specific embeddings would likely improve
   vector retrieval quality.

5. **No temporal reasoning**: The graph structure does not encode temporal
   relationships between events. Causal chains are inferred from co-occurrence
   and LLM extraction, not temporal ordering.

## License

MIT
