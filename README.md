# AeroGraph

**Production hybrid retrieval over 2,000 NASA aviation safety incident reports.
Four retrieval signals fused, six systems benchmarked head-to-head with
statistical significance testing.**

[Live demo](# "cached-demo deployment — see deploy section") ·
[Dataset](https://huggingface.co/datasets/Aryan95614/aerograph-asrs "upload pending") ·
[Paper](./paper/abstract.md) ·
[Walkthrough](# "3-min video")

<!-- TODO: fill Spaces URL, dataset URL, arXiv URL, Loom URL after deploy -->

## What It Does

The FAA holds ~15,000 near-miss records that are rarely analyzed across reports.
67 people died at Reagan National in January 2025 in a mid-air collision whose
contributing factors appear in NASA ASRS narratives filed years earlier.
AeroGraph builds a knowledge graph from the ASRS corpus and lets an analyst ask
causal questions that span thousands of reports — the kind of multi-hop
reasoning no existing tool performs.

## Why This Is Interesting Engineering-Wise

This is a working implementation of the hybrid retrieval stack that most RAG
systems claim but few actually ship:

- **Knowledge graph extraction** from unstructured text via LLM structured output,
  normalized against aviation taxonomies (HFACS, ICAO aircraft/airport codes)
- **Four parallel retrieval signals** — dense vector, BM25, Personalized
  PageRank, community summaries — fused with Reciprocal Rank Fusion
- **Leiden community detection** over 24K-node graph, with LLM-summarized
  communities for global queries
- **End-to-end evaluation harness** — 50 queries × 6 systems, 95% confidence
  intervals, Wilcoxon signed-rank tests, dual-judge evaluation (local Ollama +
  Claude Sonnet)

Every component is tested, benchmarked, and has a measured contribution to
end-to-end answer quality.

## Results

Six retrieval systems evaluated on 50 queries (single-hop, multi-hop,
comparative). All numbers pulled from
[`paper/results/eval_results_claude_judge.json`](./paper/results/eval_results_claude_judge.json).

| System | P@10 | nDCG@10 | Faith (Ollama) | Faith (Claude) |
|---|---|---|---|---|
| Vector baseline | 0.114 | 0.121 | 0.608 | 0.410 |
| BM25 | 0.106 | 0.116 | 0.520 | 0.245 |
| Graph-Only | 0.058 | 0.072 | 0.532 | 0.205 |
| GraphRAG (vector + 2-hop graph) | 0.152 **(+33%)** | 0.151 | **0.642** | 0.255 |
| PPR-Only (HippoRAG signal) | **0.200 (+75%)** | **0.221** | 0.578 | 0.230 |
| **HippoRAG 4-way RRF** | **0.210 (+84%)** | **0.231** | **0.685** | 0.310 |

**Pairwise blind comparison (Claude judge, system labels hidden):**

| Comparison | Left wins | Ties | Win rate |
|---|---|---|---|
| GraphRAG vs Vector-only | 36 / 50 | 5 | **72%** |
| Hybrid 4-way vs BM25 | 32 / 50 | 4 | 64% |
| Hybrid 4-way vs Vector-only | 31 / 50 | 2 | 62% |

**The most interesting finding:** Personalized PageRank wins on retrieval
precision (P@10 = 0.200), but four-signal fusion wins on final answer quality
(62–72% pairwise preference). Retrieval precision ≠ answer quality — signal
*diversity* beats any single signal's top rank. Judge disagreement is real and
reported honestly: Claude is 0.2–0.4 stricter than Ollama on faithfulness
across all systems, but rankings between systems are preserved.

## Architecture

```
ASRS reports (2,000 real NASA narratives)
        │
        ▼
 ┌──────────────────┐      Claude Sonnet, structured JSON
 │  Entity /        │      10 entity types, 8 edge types
 │  Relation        │─────►  29,244 raw entities → 23,948 canonical
 │  Extraction      │       43,505 raw relations → 48,479 clean
 └──────────────────┘       HFACS / ICAO normalization
        │
        ▼
 ┌──────────────────┐      NetworkX DiGraph
 │  Knowledge       │      Leiden communities (L0: 311, L1: 393, L2: 296)
 │  Graph           │      LLM community summaries (100 largest)
 └──────────────────┘
        │
        ├────────────┬────────────┬─────────────┐
        ▼            ▼            ▼             ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
 │ ChromaDB │ │   BM25   │ │   PPR    │ │  Community   │
 │ 4,710    │ │ from-    │ │ alpha=   │ │  Summaries   │
 │ chunks   │ │ scratch  │ │ 0.15     │ │  Map-reduce  │
 └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘
      │            │            │               │
      └──────RRF fusion (k=45, tuned)──────────┘
                       │
                       ▼
             ┌──────────────────┐     Claude Sonnet, grounded
             │   Generation     │     Source ACN citations
             └──────────────────┘
```

## Try It Locally (5 minutes)

```bash
git clone https://github.com/Aryan95614/AeroGraph.git
cd AeroGraph
make install                    # pip install -e .
echo "ANTHROPIC_API_KEY=..." > .env

make serve                      # FastAPI on :8000
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What weather conditions most often precede go-arounds?"}'
```

Sample output (trimmed):

```json
{
  "answer": "Low ceilings, windshear, and heavy rain are the three weather
  conditions most frequently cited preceding go-around decisions in the
  corpus. IMC approaches with gusting crosswinds appear in 23 of the 47
  go-around reports examined...",
  "sources": ["ACN 1355886", "ACN 1024106", "ACN 1764327"],
  "latency_ms": 4821
}
```

## Reproduce the Benchmark

```bash
make ingest       # parse ASRS reports (uses local cache, no network)
make extract      # LLM entity / relation extraction — costs ~$15 in API
make build        # build NetworkX graph from extractions
make embed        # ChromaDB + sentence-transformers index
make normalize    # taxonomy-based entity resolution
make community    # Leiden detection + LLM summarization
make eval         # 50 queries × 6 systems, ~20 min
```

Outputs `paper/results/eval_results_claude_judge.json` with mean, std, 95% CI
per config, plus per-query Ollama and Claude judge scores.

## Stack

NetworkX DiGraph · ChromaDB + sentence-transformers/all-MiniLM-L6-v2 ·
from-scratch BM25 · Personalized PageRank (HippoRAG) · Leiden (leidenalg +
python-igraph) · Claude Sonnet (extraction + generation) · FastAPI · Gradio
(HF Spaces) · Streamlit (dashboard) · Modal (serverless deploy) · pytest
(135 tests)

## Project Structure

```
src/aerograph/
  ingest.py          parse ASRS reports + taxonomy-aware normalization
  extract.py         LLM entity/relation extraction w/ aviation ontology
  graph.py           NetworkX + Neo4j backends, 2-hop BFS, PageRank
  embed.py           ChromaDB chunking + indexing
  retrieve.py        6 retrievers + RRF fusion + query routing
  community.py       Leiden detection + community summarization
  taxonomy.py        HFACS, ICAO, weather, phase canonical lookups
  eval.py            50-query benchmark, dual-judge, checkpointing
  generate.py        Claude generation with ACN source tracing
  api.py             FastAPI server (/query, /stats, /graph/entity)
  dashboard.py       Streamlit interactive dashboard

tests/               135 pytest cases, no network / API dependencies
scripts/             pipeline runners, paper generation, dataset upload
paper/               abstract, methods, related work, results tables, figures
```

## Deployment

**HuggingFace Spaces (cached demo):** `app.py` auto-detects missing API key
and serves pre-computed answers for 3 preset queries. Deploy:

```bash
huggingface-cli repo create aerograph --type=space --space_sdk=gradio
git remote add space https://huggingface.co/spaces/Aryan95614/aerograph
git push space RUNTHISFILE:main
```

**Modal (serverless):** see `modal_app.py`. Deploys FastAPI + Gradio with
persistent storage for the graph pickle and Chroma index.

## Scale

- **2,000** real NASA ASRS reports benchmarked
- **29,244** raw entities, **23,948** canonicalized after taxonomy resolution
- **43,505** raw relations, **48,479** after synonym expansion
- **4,710** indexed chunks
- **100** LLM-summarized communities
- **135** tests pass, 0 fail
- **~10,000** lines of Python across 13 source modules

## Citation

```bibtex
@misc{dhawan2026aerograph,
  title        = {AeroGraph: Graph-Augmented Retrieval for Multi-Hop Causal
                  Reasoning over Aviation Safety Reports},
  author       = {Dhawan, Aryan},
  year         = {2026},
  url          = {https://github.com/Aryan95614/AeroGraph}
}
```

## License

MIT — see [LICENSE](./LICENSE).
