# AeroGraph → Shopify Engineering

**One line:** Production-grade hybrid retrieval + knowledge graph over a
real-world incident corpus. End-to-end implementation of three recent RAG
papers, benchmarked head-to-head across six retrieval configurations with
statistical significance testing. Built while working in aviation safety
data infrastructure at Delta Air Lines.

- **Live demo:** [HF Spaces URL — cached-demo mode, 3 preset queries return
  benchmark-run outputs; clone the repo for live queries]
- **Code:** https://github.com/Aryan95614/AeroGraph
- **Preprint:** [arXiv URL — submission pending; see
  `paper/abstract.md` for the current draft]
- **3-min walkthrough:** [Loom URL]

## Why This Is Shopify-Relevant

Shopify's Search & Discovery team ships hybrid retrieval at merchant scale.
Shop App recommendations are retrieval at user scale. Merchant ML touches
text classification, entity extraction, and structured knowledge over
catalogs. This project is a working implementation of the exact stack those
teams care about:

- **Dense + sparse + graph retrieval fusion** — the direction commercial
  search is moving. Most teams talk about it; this ships it with measured
  per-signal contributions.
- **LLM-based knowledge graph construction from unstructured text** — the
  primitive that enables semantic merchant search beyond lexical matching.
- **Entity resolution against domain taxonomies** — the difference between
  a product catalog that is machine-queryable and one that isn't.
- **Systematic evaluation with statistical significance testing** —
  Wilcoxon signed-rank on 50 queries × 6 systems × 95% CIs separates toy
  RAG from production RAG.

## What I Built

A full pipeline from raw incident reports to grounded, cited answers:

```
ASRS narratives
    → LLM entity extraction (structured JSON, aviation ontology)
    → Taxonomy normalization (HFACS / ICAO)
    → Directed knowledge graph (NetworkX)
    → Leiden communities + LLM summarization
    → 4 parallel retrievers (vector, BM25, PPR, community)
    → Reciprocal Rank Fusion
    → Claude generation with ACN citations
    → Dual-judge benchmark with statistical tests
```

Six retrieval systems implemented, tested, benchmarked head-to-head:

1. **Dense vector** — ChromaDB + sentence-transformers/all-MiniLM-L6-v2
2. **Sparse BM25** — from-scratch Okapi implementation (no rank-bm25),
   tested against reference
3. **Graph-only** — co-occurrence traversal, PageRank ranking
4. **Personalized PageRank** — HippoRAG paper (Gutiérrez et al., 2024)
5. **GraphRAG** — Leiden communities + map-reduce summarization
   (Edge et al., Microsoft, 2024)
6. **Four-way RRF fusion** — all signals combined with tuned k=45

## Results (All Verifiable From `paper/results/eval_results_claude_judge.json`)

| System | P@10 | nDCG@10 | Faith (Ollama) | Faith (Claude) |
|---|---|---|---|---|
| Vector baseline | 0.114 | 0.121 | 0.608 | 0.410 |
| BM25 | 0.106 | 0.116 | 0.520 | 0.245 |
| Graph-Only | 0.058 | 0.072 | 0.532 | 0.205 |
| GraphRAG | 0.152 (+33%) | 0.151 | 0.642 | 0.255 |
| PPR-Only | 0.200 (+75%) | 0.221 | 0.578 | 0.230 |
| **Hybrid 4-way RRF** | **0.210 (+84%)** | **0.231** | **0.685** | 0.310 |

Pairwise blind comparison, Claude judge with hidden labels:

- GraphRAG vs Vector-only: **72% win rate** (36–9–5)
- Hybrid vs BM25: **64%** (32–14–4)
- Hybrid vs Vector-only: **62%** (31–17–2)

**The most interesting finding:** PPR retrieves the best individual passages
(P@10 = 0.200, second only to the full hybrid), but four-way fusion produces
the best final answers (62–72% blind pairwise preference). **Retrieval
precision ≠ answer quality.** The diversity of signals matters more than any
single signal's ranking. This is underexplored in the RAG literature and
directly relevant to any production search system weighing lexical vs
semantic recall tradeoffs.

**Honest judge disagreement:** Ollama (`qwen2.5:7b`, local) rates hybrid
faithfulness at 0.685; Claude Sonnet rates the same outputs at 0.310.
Rankings are mostly preserved across judges; absolute numbers are
judge-sensitive. Both are reported; the gap is called out in the paper's
limitations section instead of papered over.

## Engineering Signals

- ~10,000 lines of Python, 13 source modules, **135 tests pass, 0 fail**
- **81 commits** over 4 weeks, clean history, no force-pushes on main
- Every public function typed and covered by unit tests
- End-to-end reproducibility:
  ```bash
  make ingest && make extract && make graph && make embed && make eval
  ```
  runs the full pipeline from raw data to statistical results
- Three deployment surfaces, all working: FastAPI, Gradio (HF Spaces),
  Streamlit dashboard, plus Modal serverless config
- Cached-demo mode in `app.py` — HF Spaces deploy works without an API key,
  serving benchmark-run answers for preset queries
- No dead code, no commented-out blocks, no TODOs in shipped files

## What I Learned That Maps to Shopify's Problems

- **Hybrid retrieval fusion is not a solved problem.** Naive RRF works but
  ceilings out; per-query-type signal weighting is where the real gains
  come from. I tuned RRF k from 60 → 45 and added explicit 1.5× / 1.3× PPR
  and community-signal weights based on eval results. Shopify's product
  search faces the same tradeoff between lexical recall and semantic
  recall, and the per-query-type weighting idea transfers directly.
- **LLM entity extraction produces graph fragmentation without taxonomy
  alignment.** My first extraction pass gave **3,766 weakly-connected
  components over 29K nodes**. Adding HFACS / ICAO normalization + synonym
  edges collapsed that to **186 components**. Product catalogs have the
  same shape: "Nike Air Max 90" and "nike airmax-90" extracted from two
  listings become one node only with a canonical taxonomy lookup.
- **Evaluation infrastructure is more important than the retrieval system
  itself.** The PPR-vs-hybrid insight is invisible without a 50-query × 6-
  system × 95% CI harness. I couldn't see the tradeoff between retrieval
  precision and answer quality until I had dual-judge agreement numbers.
  Every production search team needs this. Most don't have it.

## About Me

- 2nd year University of Waterloo · AI + Statistics major, CS minor
- Currently in my second rotation at Delta Air Lines, aviation safety data
  infrastructure
- Previously shipped: ML retrieval at Lens (Martin, YC '23), SBOM
  generation in Go at Loops (YC), Hack the North winner
- [LinkedIn](#) · [Personal site](#) · [Email](#)

## What I'm Looking For

A winter or summer 2027 internship on a team where hybrid retrieval,
knowledge graphs, or LLM systems over large text corpora are first-class
problems. Open to Toronto or remote. Happy to do a technical conversation,
pair-programming interview, or written take-home — whichever format is
most useful.
