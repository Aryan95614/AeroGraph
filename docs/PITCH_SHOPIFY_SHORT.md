My friend Aryan (2nd year Waterloo, currently at Delta Air Lines) built a
hybrid retrieval + knowledge graph system over 2,000 NASA aviation safety
reports — implements three recent RAG papers end-to-end (GraphRAG, HippoRAG,
taxonomy-normalized entity resolution), benchmarks six retrieval
configurations with Wilcoxon significance testing, and ships as a live
HF Spaces demo with 135 pytest cases passing. Key finding: PPR wins
retrieval P@10 (0.200, +75%), but 4-way RRF fusion wins blind pairwise
answer-quality preference (62–72% vs vector baseline). He's looking for a
2027 ML / retrieval internship.

Live demo: [HF Spaces URL]
Repo: https://github.com/Aryan95614/AeroGraph
One-pager: [repo]/docs/PITCH_SHOPIFY.md

Worth 20 minutes with someone on Search & Discovery or Merchant ML?
