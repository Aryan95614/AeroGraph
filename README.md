# AeroGraph

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-TBA-red.svg)](# "arXiv id pending")

**AeroGraph is a benchmark-pathology study of graph-augmented RAG over aviation
safety narratives.** We build a hybrid retrieval system over 2,000 NASA ASRS
incident reports and evaluate six retrieval configurations on a 50-query
benchmark with paired Wilcoxon significance tests and dual LLM judges. Within
the primary corpus, graph-augmented systems significantly beat a vector
baseline ($p=0.047$). On a disjoint held-out sample from the same release,
significance evaporates, rankings scramble (Kendall $\tau \approx -0.07$), and
pairwise preference shifts toward the baseline. We name this failure mode
**Extractive Oracle Circularity**: when the relevance oracle is defined by
entity overlap and the entities come from one LLM extraction pipeline,
graph-aware retrievers have a within-corpus advantage that swap-corpus
evaluation exposes as artifact.

## Try the Live Demo

<p align="center">
  <a href="https://huggingface.co/spaces/Aryan95614/aerograph-v2" target="_blank" rel="noopener">
    <img
      src="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/Aryan95614/aerograph-v2.png"
      alt="Open the AeroGraph demo on HuggingFace Spaces"
      width="720"
    />
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Aryan95614/aerograph-v2" target="_blank" rel="noopener">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg" alt="Open in HuggingFace Spaces" />
  </a>
  &nbsp;
  <a href="https://huggingface.co/datasets/Aryan95614/aerograph-asrs" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/🤗%20Dataset-aerograph--asrs-yellow" alt="HF Dataset" />
  </a>
  &nbsp;
  <a href="paper/aerograph_v0.1.pdf">
    <img src="https://img.shields.io/badge/Paper-v0.1%20PDF-blue" alt="Paper PDF" />
  </a>
</p>

The Space runs in **cached mode**: click any of the 10 preset showcase queries
and see a grounded answer with ACN citations. No API key required. For live
queries against the full hybrid retriever, clone the repo and set
`ANTHROPIC_API_KEY`.

## Architecture

```
ASRS reports (2,000 real narratives)
       │
       ▼
 ┌───────────────────┐
 │ LLM extraction    │  Claude Sonnet, structured JSON
 │ 10 entity types,  │  HFACS / ICAO taxonomy normalization
 │ 8 edge types      │
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │ Knowledge graph   │  NetworkX / Neo4j
 │ 23,948 nodes      │  Leiden communities
 │ 48,479 edges      │
 └─────────┬─────────┘
           │
    ┌──────┴──────┬──────────┬───────────┐
    ▼             ▼          ▼           ▼
 ChromaDB     BM25       Personalized   Community
 4,710 chunks inverted   PageRank       summaries
                idx       α=0.15        top 100
    │             │          │           │
    └────────RRF fusion (k=45, tuned)────┘
                   │
                   ▼
         Claude generation with ACN citations
```

## Headline Results

| Metric | Primary corpus | Held-out corpus |
|---|---|---|
| Hybrid vs Vector P@10 (paired Wilcoxon $p$) | **0.047** | 0.14 |
| GraphRAG vs Vector P@10 $p$ | **0.018** | 0.08 |
| Bootstrap 95% CI on $\Delta$P@10 (Hybrid − Vector) | $[+0.022, +0.180]$ | $[-0.004, +0.098]$ |
| Kendall $\tau$, system ranking primary ↔ held-out | — | $-0.07$ |
| Spearman $\rho$, per-query $\Delta$ primary ↔ held-out | — | $+0.15$ |
| Multi-hop $\Delta$P@10(Hybrid − Vector) | $+0.090$ | **$-0.020$** (sign flip) |
| Ollama faithfulness (Hybrid) | 0.685 | 0.722 |
| Oracle-dead queries (all systems P@10=0) | 12/50 | 24/50 |

Full numbers: `paper/results/eval_results_claude_judge.json` (primary) and
`paper/results/eval_results_heldout.json` (held-out).

## Try It

**Cached demo** (no API key required):
```bash
git clone https://github.com/Aryan95614/AeroGraph.git
cd AeroGraph
make repro
```
`make repro` installs the package, verifies data artifacts, runs the
integration test, and launches the Gradio demo on `localhost:7860` in
cached mode with 10 showcase queries pre-answered.

**Live mode** (requires `ANTHROPIC_API_KEY`):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

**Reproduce the benchmark from scratch** (requires API credits, ~2h):
```bash
make ingest      # parse 2,000 reports from HF
make extract     # Claude entity/relation extraction (~$25)
make build       # build canonicalized graph
make embed       # ChromaDB index
make eval        # 50-query × 6-system benchmark (~$15)
make paper       # regenerate figures + populate LaTeX tables
```

## Repository Layout

```
src/aerograph/
  ingest.py       parse ASRS narratives + aircraft normalization
  extract.py      LLM-based entity/relation extraction
  taxonomy.py     HFACS / ICAO / phase / weather canonicalization
  graph.py        NetworkX + Neo4j backends, 2-hop BFS, PageRank
  embed.py        ChromaDB chunking and indexing
  community.py    Leiden detection + LLM summarization
  retrieve.py     6 retrievers + RRF fusion
  generate.py     Claude answer generation with ACN citations
  eval.py         50-query benchmark, dual-judge, checkpointing
  api.py          FastAPI server (/query, /stats, /graph/entity)
  dashboard.py    Streamlit interactive dashboard

tests/            integration and unit tests
scripts/          pipeline runners, HF upload, figure generation
paper/            LaTeX source, figures, results JSONs
```

## Citation

```bibtex
@misc{dhawan2026aerograph,
  title         = {Extractive Oracle Circularity: Why Graph-RAG Benchmarks
                   Fail Cross-Corpus Replication},
  author        = {Dhawan, Aryan},
  year          = {2026},
  eprint        = {TBA},
  archivePrefix = {arXiv},
  url           = {https://github.com/Aryan95614/AeroGraph}
}
```

## License

MIT — see [LICENSE](LICENSE). ASRS narratives are NASA public-domain data;
our extracted knowledge graph and benchmark queries are released under CC-BY-4.0.
