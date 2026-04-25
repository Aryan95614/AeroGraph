# AeroGraph → Shopify Engineering

**One sentence:** A working hybrid graph-vector retrieval system over 2,000 NASA aviation safety reports — and a methodological finding (*Extractive Oracle Circularity*) that exposes a structural flaw in how graph-RAG benchmarks are scored.

| | |
|---|---|
| Live demo | https://huggingface.co/spaces/Aryan95614/aerograph-v2 |
| Code | https://github.com/Aryan95614/AeroGraph |
| Paper (arXiv-formatted) | [`paper/aerograph_v0.1.pdf`](../paper/aerograph_v0.1.pdf) |
| Dataset | https://huggingface.co/datasets/Aryan95614/aerograph-asrs |

> **Endorsed by Delta's Managing Director of Tools and Technology** in person — she asked specifically how the cross-corpus replication methodology would map to Delta's internal incident review pipeline. Presented in 20 minutes, no slides, just the running demo and the paper.

---

## The numbers

| | |
|---|---|
| Reports ingested | **2,000** real NASA ASRS narratives |
| Held-out replication corpus | **2,000** disjoint reports, same source |
| Knowledge graph (canonicalized) | **23,948 entities · 48,479 relations · 186 weakly-connected components** |
| Retrievers benchmarked head-to-head | **6** (vector, BM25, graph-only, GraphRAG, PPR-only, 4-way RRF) |
| Benchmark queries | **50** (single-hop, multi-hop, comparative) |
| LLM judges | **2 independent** (local `llama3.1:8b` + Claude Sonnet) |
| Total scored evaluation runs | **300 generations × 2 judges + 6 pairwise pairs × 50 queries = 900+ judgments** |
| Statistical tests reported | Paired Wilcoxon signed-rank, **10,000-iter bootstrap 95% CIs**, Kendall τ, Spearman ρ |
| Result-table figures | 4 (bar chart, pairwise win-rate, query-type breakdown, graph stats) |
| Tests passing | **135** (pytest, 0 failures, 0 skips) |
| Lines of Python | **~10,000** across 13 source modules |
| Time from blank repo to v0.1 | **31 days**, ~80 commits, no force-pushes on main |

---

## The methodological finding

Within the primary corpus, hybrid retrieval significantly improves P@10 over a vector baseline (**+0.096, paired Wilcoxon p = 0.047**), and pairwise blind preference favors graph-augmented systems 62–72% under a Claude judge.

**On a held-out sample drawn from the same data source, the significance evaporates.** Bootstrap 95% CIs cross zero. System rankings are uncorrelated across corpora (**Kendall τ ≈ −0.07**). The multi-hop sign flips: +0.090 → −0.020. Faithfulness, the one metric that doesn't consult the entity-overlap oracle, is stable.

**Named contribution:** when a relevance oracle is defined by entity overlap and the entities come from a single LLM extraction pipeline, retrievers that walk the same extracted graph have an artifactual within-corpus advantage that swap-corpus evaluation exposes. We call this **Extractive Oracle Circularity** and propose held-out swap as a minimum falsifiability test for any graph-RAG benchmark whose relevance labels derive from structure the retriever can also exploit.

---

## Process discipline (Shopify-relevant signals)

- **Pre-deploy verification gate.** `make verify-space` runs five checks before any HF Space deploy: byte-compile, README↔requirements parity, demo-cache JSON validation, clean-venv install of pinned dependencies, headless `build_app()` probe. `make deploy-space` hard-depends on it (`SKIP_VERIFY=1` overrides only for CI bootstrap).
- **Full reproducibility from a fresh clone.** `make repro` installs deps, verifies artifacts, runs the integration test, launches the cached-mode Gradio demo on localhost. ~3 minutes from `git clone` to a working app.
- **Idempotent pipeline scripts.** Every stage (`extract`, `build`, `embed`, `community`, `eval`) is checkpoint-resumable. Re-running on partial state is safe and cheap.
- **Honest negative-result reporting.** The within-corpus result is not retracted; it's contextualized. The held-out non-replication isn't buried; it's the headline of the rewritten abstract and Section 6.5.
- **Released artifacts.** Per-query outputs, dual-judge scores, taxonomy tables, benchmark queries, and reference answers all public. Nothing is "available upon request."

---

## Why this maps to Shopify

- **Search & Discovery** is shipping hybrid lexical+semantic retrieval at merchant catalog scale. The PPR-vs-Hybrid result here (statistically tied at *n*=50) is a pre-existing data point for the tradeoff between fusion complexity and retriever simplicity. Latency: **2.6s vs 23.1s** for indistinguishable answer quality.
- **Merchant ML / Catalog Intelligence** runs LLM extraction over unstructured text and structures it into a graph (products, attributes, relations). The taxonomy resolution cascade implemented here (HFACS / ICAO / phase / weather, three tiers: exact match → `rapidfuzz` → embedding similarity) collapsed graph fragmentation from 3,766 weakly-connected components to 186. The pattern transfers directly to product canonicalization.
- **Evaluation infrastructure.** Shopify presumably runs internal RAG benchmarks. The ExOC finding applies broadly: any benchmark whose relevance oracle reuses structure the retriever exploits gives biased within-corpus scores. The held-out swap protocol is one extra line in their existing pipeline and surfaces this immediately.

---

## What I'm asking for

A 30-minute conversation with someone on Search & Discovery, Merchant ML, or Polaris ML — anyone whose retrieval/eval stack would be improved by an external pair of eyes that just shipped this. Internship interest is real (winter or summer 2027); a technical pair-programming round, take-home, or systems-design conversation works equally well.

Aryan Dhawan · 2nd year CS+Stats, University of Waterloo · currently in 2nd co-op rotation at Delta Air Lines, aviation safety data infrastructure
