# LinkedIn Post — AeroGraph

*Draft. Review and edit before posting.*

---

Spent the last month building **AeroGraph** — a hybrid retrieval system that
reads NASA's aviation safety reports the way an accident investigator would.

The problem: the FAA holds ~15,000 near-miss records. Most are never analyzed
across reports. When 67 people died at Reagan National in January 2025 in a
mid-air collision, some of the contributing factors were already present in
ASRS narratives filed years earlier. No tool existed to surface them.

I built the tool.

The stack implements three recent RAG papers end-to-end:
• **Microsoft GraphRAG** — Leiden community detection + hierarchical
  summarization
• **HippoRAG (NeurIPS 2024)** — Personalized PageRank + 4-signal RRF fusion
• **Agarwal et al. (LREC 2022)** — HFACS / ICAO taxonomy normalization for
  aviation knowledge graphs

2,000 real NASA ASRS reports in. A knowledge graph with 23,948 canonicalized
entities and 48,479 relations out. Then six retrieval systems benchmarked
head-to-head on 50 queries with 95% confidence intervals and Wilcoxon
signed-rank tests.

Headline numbers:

| System | P@10 | Faithfulness |
|---|---|---|
| Vector baseline | 0.114 | 0.608 |
| BM25 | 0.106 | 0.520 |
| GraphRAG | 0.152 (+33%) | 0.642 |
| PPR-Only | 0.200 (+75%) | 0.578 |
| **Hybrid 4-way RRF** | **0.210 (+84%)** | **0.685** |

Under blind pairwise comparison (Claude judge, labels hidden), GraphRAG
beats vector-only 72% of the time.

The most interesting finding: **retrieval precision ≠ answer quality**.
Personalized PageRank wins on P@10, but signal *diversity* wins on final
answers. The four-way fusion isn't the best retriever on any one query —
it's the most consistent across query types.

135 tests passing. ~10,000 lines of Python. All code, data, and results
public.

Repo: https://github.com/Aryan95614/AeroGraph
Live demo: [HF Spaces URL]
Preprint: [arXiv URL]

Built while working at Delta Air Lines on aviation safety infrastructure.
Looking for 2027 internships in ML / retrieval / search systems.
