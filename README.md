# AeroGraph

**GraphRAG over aviation safety incident reports for causal reasoning and safety pattern extraction.**

AeroGraph demonstrates that graph-structured retrieval outperforms vanilla RAG on multi-hop causal queries in the aviation safety domain. Built on NASA ASRS (Aviation Safety Reporting System) incident reports, with a full 4-system ablation study (GraphRAG vs. Vector-only vs. BM25 vs. Graph-only).

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
  Query             │   ┌───────┐      ┌───────────┐    ┌──────────┐  │
  ──────────────────►   │ BM25  │─────►│  Retrieve │───►│ Generate │──►── Answer
                    │   │(lexic)│      │ (RRF fuse)│    │ (Claude) │  │
                    │   └───────┘      └───────────┘    └──────────┘  │
                    │                                                  │
                    │   ┌─────────┐    ┌───────────┐    ┌──────────┐  │
                    │   │  Eval   │───►│  Figures  │    │Dashboard │  │
                    │   │(50-Q)   │    │  (paper)  │    │(Streamlt)│  │
                    │   └─────────┘    └───────────┘    └──────────┘  │
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
make eval      # Run 50-query evaluation benchmark (4-system ablation)
make paper     # Generate figures and results tables

# 4. Start the API server or dashboard
make serve     # FastAPI on http://localhost:8000
make dashboard # Streamlit UI on http://localhost:8501
```

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Ingestion** | Python + httpx | 2,000 real NASA ASRS reports via HuggingFace Hub |
| **Extraction** | Claude Sonnet | Entity/relation extraction with aviation ontology (10 node types, 8 edge types) |
| **Graph Store** | Neo4j / NetworkX | Knowledge graph with causal + temporal edges |
| **Vector Store** | ChromaDB | Semantic chunk retrieval (256-token, 128 overlap) |
| **Embeddings** | all-MiniLM-L6-v2 | Local sentence embeddings (384-dim) |
| **Retrieval** | RRF Fusion | Hybrid vector + graph scoring (k=45, weights 0.45/0.55) |
| **BM25** | From scratch | Okapi BM25 lexical retrieval (k1=1.5, b=0.75) |
| **Generation** | Claude Sonnet | Answer generation with source tracing |
| **Evaluation** | Claude-as-Judge | Faithfulness, relevance, causal accuracy with 95% CI |
| **API** | FastAPI | REST endpoints for queries and graph exploration |
| **Dashboard** | Streamlit | Interactive query, graph explorer, causal chain tracer |

## Aviation Safety Ontology

**Node types (10):** Aircraft, Event, Phase, Factor, Component, Outcome, Recommendation, ATC_Facility, Weather, TimePeriod

**Edge types (8):** CAUSED_BY, CONTRIBUTED_TO, OCCURRED_DURING, INVOLVED, RESOLVED_BY, PRECEDED_BY, CO_OCCURRED_WITH, TEMPORAL_SEQUENCE

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

The benchmark suite contains 50 queries evaluated across **4 retrieval systems**:

| System | Description |
|--------|-------------|
| **GraphRAG** | Hybrid vector + graph with RRF fusion |
| **Vector-Only** | ChromaDB semantic similarity only |
| **BM25** | Okapi BM25 lexical matching (from scratch) |
| **Graph-Only** | Graph neighborhood expansion only |

Query types:
- **20 single-hop factual** — direct lookups against the corpus
- **20 multi-hop causal** — requires reasoning across multiple reports
- **10 comparative** — cross-entity or cross-category analysis

Metrics: faithfulness, answer relevance, context precision, context recall, causal chain accuracy, reference similarity (ROUGE-L F1). All reported with mean, std, and 95% CI.

### Key Results (2,000 Real ASRS Reports)

| System | Faithfulness | Answer Relevance | Causal Accuracy | Avg Latency |
|--------|:----------:|:----------:|:----------:|:----------:|
| **GraphRAG** | **0.390** | **0.854** | 0.550 | 19.8s |
| Vector-Only | 0.306 | 0.850 | 0.550 | 18.5s |
| BM25 | 0.230 | 0.775 | 0.562 | 17.0s |
| Graph-Only | 0.528 | 0.598 | 0.537 | 14.0s |

GraphRAG achieves +27% faithfulness over vector-only retrieval and +79% on comparative queries. Full results in `paper/results_table.md`.

## Deployment

### HuggingFace Spaces

```bash
# Install Spaces requirements and run locally
pip install -r spaces_requirements.txt
python app.py

# Or deploy to HuggingFace Spaces — push repo to a HF Space with Gradio SDK
```

### Modal (Serverless)

```bash
modal deploy modal_app.py
```

### Dataset Upload

```bash
huggingface-cli login
python scripts/upload_dataset.py
```

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
│   ├── api.py         # FastAPI server
│   └── dashboard.py   # Streamlit interactive UI
├── scripts/
│   ├── build_graph.py # Graph construction entry point
│   ├── run_eval.py    # Evaluation entry point
│   └── demo.py        # Demo with 5 example queries
├── tests/
│   ├── test_ingest.py
│   ├── test_extract.py
│   └── test_retrieve.py
├── paper/
│   ├── abstract.md
│   ├── results_table.md
│   ├── related_work.md
│   ├── figures/
│   └── results/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── graphs/
│   └── chroma_db/
├── pyproject.toml
├── Makefile
├── DECISIONS.md
└── README.md
```

## Limitations

1. **Data scope**: Evaluated on 2,000 real NASA ASRS reports sourced from
   HuggingFace Hub (elihoole/asrs-aviation-reports). Synthetic report generation
   is retained as a fallback for environments without internet access, but all
   published results use real incident data.

2. **Evaluation circularity**: Faithfulness and relevance scores use Claude as
   evaluator for answers also generated by Claude. We report these for relative
   comparison across systems, not as absolute quality measures.

3. **Graph coverage**: Entity extraction quality depends on LLM domain
   understanding. Extraction errors propagate to the graph and downstream
   retrieval. The 0.85 fuzzy dedup threshold mitigates but does not eliminate
   entity fragmentation.

4. **Single embedding model**: all-MiniLM-L6-v2 is general-purpose, not
   fine-tuned on aviation text. Domain-specific embeddings would likely improve
   vector retrieval quality.

5. **Temporal reasoning**: TEMPORAL_SEQUENCE edges are inferred from narrative
   structure, not ground-truth timestamps. Temporal ordering reflects author
   description order, which may not perfectly match chronological sequence.

## License

MIT
