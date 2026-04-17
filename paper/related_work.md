# Related Work

AeroGraph draws on several intersecting lines of research: retrieval-augmented
generation, graph-structured retrieval, NLP for aviation safety, LLM-based
information extraction, and evaluation methodology for RAG systems. We survey
each in turn.

## 2.1 Retrieval-Augmented Generation

Lewis et al. (2020) introduced RAG as a general framework for grounding language
model generation in retrieved evidence, combining a dense passage retriever with
a seq2seq generator and training the two components end-to-end. Their
formulation established the retrieve-then-read paradigm that now underlies most
open-domain QA systems. Subsequent work has extended the basic pattern in several
directions. Izacard and Grave (2021) proposed Fusion-in-Decoder (FiD), which
encodes multiple retrieved passages independently before fusing them in the
decoder, improving performance on multi-document questions. Borgeaud et al.
(2022) scaled retrieval-augmented pretraining to trillion-token datastores with
RETRO, demonstrating that retrieval augmentation reduces the parameter count
needed for competitive perplexity.

More recent work targets specific failure modes of flat retrieval. Gao et al.
(2023) introduced HyDE (Hypothetical Document Embeddings), which generates a
hypothetical answer to the query and retrieves against the hypothetical
document's embedding rather than the query embedding directly, improving recall
for queries whose surface form diverges from relevant passages. Sarthi et al.
(2024) proposed RAPTOR, which recursively clusters and summarizes document
chunks into a tree structure, enabling retrieval at multiple levels of
abstraction. Asai et al. (2024) introduced Self-RAG, which trains a language
model to adaptively decide when to retrieve, what to retrieve, and whether
retrieved passages are relevant, using reflection tokens to control generation.

Despite these advances, a fundamental limitation persists: flat retrieval
operates over independent text chunks and cannot follow multi-hop reasoning
chains that span document boundaries. Trivedi et al. (2023) demonstrated this
concretely with IRCoT, showing that interleaving retrieval with chain-of-thought
reasoning steps substantially improves multi-hop QA, precisely because single
retrieval calls fail to surface the complete evidence chain. This limitation
motivates graph-structured retrieval approaches, including ours.

## 2.2 Graph-Based Retrieval and GraphRAG

The integration of knowledge graphs with language models predates the RAG
paradigm. Early work on knowledge-grounded dialogue and QA used structured
knowledge bases (Freebase, Wikidata) as retrieval targets, with graph traversal
replacing or supplementing document retrieval (Sun et al., 2018; Yasunaga et
al., 2021). Yasunaga et al. (2021) proposed QA-GNN, which jointly reasons over
a language model's contextual representations and a knowledge graph subgraph
using graph neural networks, achieving strong results on commonsense QA
benchmarks.

The term "GraphRAG" was crystallized by Edge et al. (2024), who proposed a
two-stage pipeline: first, build a knowledge graph from a text corpus via
LLM-based entity and relation extraction; second, use graph community detection
(Leiden algorithm) to create hierarchical summaries that serve as retrieval
targets. Their key insight is that global queries --- those requiring synthesis
across many documents --- benefit from graph-level summarization rather than
chunk-level retrieval. Microsoft's open-source implementation demonstrated
substantial improvements on global sensemaking tasks over corpora of news
articles and podcast transcripts.

Our work differs from Edge et al. (2024) in several respects. We use
domain-specific ontology-guided extraction rather than open-domain entity
extraction, which yields a more constrained and interpretable graph. We do not
use community-based summarization; instead, we perform entity-anchored subgraph
expansion at query time and fuse graph-structural evidence with vector retrieval
via Reciprocal Rank Fusion (Cormack et al., 2009). This design choice favors
local causal chain queries over global summarization queries --- a deliberate
tradeoff given our target use case of incident-level causal reasoning.

Other recent graph-augmented retrieval work includes SURGE (Kang et al.,
2023) [?], which generates responses grounded in knowledge graph subgraphs for
dialogue, and G-Retriever (He et al., 2024) [?], which combines graph neural
networks with LLMs for textual graph reasoning. The broader trend is clear:
graph structure provides a complementary retrieval signal to vector similarity,
particularly for queries requiring relational or multi-hop reasoning.

## 2.3 NLP for Aviation Safety

The aviation safety domain has a long history of structured reporting. The NASA
Aviation Safety Reporting System (ASRS), established in 1976, is the largest
voluntary safety reporting database in aviation, containing over 1.9 million
incident and near-miss reports as of 2025 (NASA, 2025) [?]. The system's
confidential and non-punitive design has made it a foundational data source for
safety trend analysis.

Automated analysis of ASRS reports has been explored primarily through text
classification and topic modeling. Kuhn (2018) [?] applied latent Dirichlet
allocation to ASRS narratives to identify recurring safety themes. Rose et al.
(2015) [?] used supervised learning to classify ASRS reports by anomaly type,
finding that NLP-based classification could approximate human analyst
categorizations. More recently, the FAA and MITRE have developed text analytics
pipelines for ASRS and ATSAP (Air Traffic Safety Action Program) data, though
much of this work remains in internal technical reports rather than peer-reviewed
publications.

On the ontology side, ECCAIRS (European Co-ordination Centre for Accident and
Incident Reporting Systems) defines a taxonomy for aviation occurrence
categories that is widely used in European safety reporting (EUROCONTROL, 2020)
[?]. The CAST (Commercial Aviation Safety Team) taxonomy provides a
complementary framework focused on contributing factors in accidents and
incidents. Our aviation safety ontology (ten node types, eight edge types, including
temporal constructs) is intentionally simpler than these formal taxonomies,
designed for LLM extraction feasibility rather than complete domain coverage.

Knowledge graph approaches to aviation safety remain sparse. Aidan et al.
(2020) [?] constructed a knowledge graph from aviation maintenance records for
fault diagnosis, but used rule-based extraction rather than LLM-based methods.
To our knowledge, AeroGraph is the first system to apply the GraphRAG paradigm
--- LLM extraction, knowledge graph construction, and hybrid graph-vector
retrieval --- specifically to aviation safety incident reports.

## 2.4 LLM-Based Information Extraction

Traditional named entity recognition and relation extraction rely on supervised
models trained on annotated corpora, using architectures such as BiLSTM-CRF
(Lample et al., 2016) or span-based extractors built on pretrained
transformers (Zhong and Chen, 2021) [?]. These approaches require
domain-specific training data, which is scarce in specialized domains like
aviation safety.

LLM-based extraction offers a zero-shot or few-shot alternative. Wei et al.
(2023) [?] systematically evaluated GPT-3.5 and GPT-4 on standard NER
benchmarks, finding that LLMs achieve competitive performance in zero-shot
settings, particularly for coarse entity types, but lag behind supervised models
on fine-grained entity distinctions and boundary detection. Wadhwa et al. (2023)
[?] extended this analysis to relation extraction, observing similar tradeoffs:
LLMs excel at recall (finding entities and relations that exist) but suffer
on precision (generating entities and relations that do not).

Our extraction pipeline follows the structured-output prompting approach, where
the LLM is instructed to produce JSON conforming to a predefined schema. This
is related to function calling and tool use capabilities now standard in
commercial LLMs (Anthropic, 2024; OpenAI, 2023). We use ontology-constrained
prompts --- specifying valid node and edge types --- to improve precision, and
apply post-hoc normalization (lowercase canonicalization, fuzzy deduplication at
threshold 0.85) to address entity fragmentation, a well-documented failure mode
of LLM-based extraction.

## 2.5 Evaluation of RAG Systems

Evaluation of RAG systems requires assessing both the retrieval component
(did the system find the right evidence?) and the generation component (did the
system produce a faithful and relevant answer?). Es et al. (2024) introduced
RAGAS, a framework that decomposes RAG evaluation into four axes: faithfulness,
answer relevance, context precision, and context recall. RAGAS uses LLM-based
scoring for faithfulness and relevance, while context precision and recall are
computed against reference context sets.

The use of LLMs as evaluation judges --- "LLM-as-judge" --- was systematically
studied by Zheng et al. (2024), who showed that strong LLMs (GPT-4 in their
study) achieve high agreement with human evaluators on open-ended generation
tasks but exhibit systematic biases, including position bias (favoring the first
response in pairwise comparisons), verbosity bias (favoring longer responses),
and self-enhancement bias (an LLM rates its own outputs higher than those of
other models). The self-enhancement concern is directly relevant to our setup,
where Claude serves as both generator and judge.

Alternative evaluation approaches include human evaluation, which remains the
gold standard but is expensive and slow, and reference-based metrics such as
BERTScore (Zhang et al., 2020) and ROUGE (Lin, 2004), which measure lexical
or semantic overlap with reference answers but do not assess factual
faithfulness. Min et al. (2023) proposed FActScore, which decomposes generated
text into atomic claims and verifies each against a knowledge source, providing
a more granular faithfulness measure than holistic LLM judgments.

For multi-hop reasoning specifically, evaluation is complicated by the need to
assess not just the final answer but the reasoning chain. Yang et al. (2018)
introduced HotpotQA with supporting-fact annotations, enabling evaluation of
both answer correctness and evidence selection. Our evaluation includes a causal
chain accuracy metric that assesses whether the retrieved evidence supports the
multi-hop reasoning path, though we acknowledge (see Limitations) that our
reference answers are themselves LLM-generated, introducing evaluation
circularity.

## Summary

AeroGraph sits at the intersection of these research threads. It applies the
GraphRAG paradigm (Edge et al., 2024) to a domain --- aviation safety reporting
--- that has received limited attention from the NLP community relative to its
practical importance. Its hybrid retrieval design is motivated by the documented
limitations of flat retrieval for multi-hop reasoning (Trivedi et al., 2023),
while its evaluation methodology reflects the practical constraints and known
biases of LLM-as-judge approaches (Zheng et al., 2024).

---

## References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *ICLR 2024*.

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving language models by retrieving from trillions of tokens. *ICML 2022*.

Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR 2009*.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. *EACL 2024 System Demonstrations*.

Gao, L., Ma, X., Lin, J., & Callan, J. (2023). Precise zero-shot dense retrieval without relevance labels. *ACL 2023*.

Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for open domain question answering. *EACL 2021*.

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. *NAACL 2016*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out, ACL 2004 Workshop*.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *EMNLP 2023*.

Sarthi, P., Abdullah, S., Tuli, A., Khushi, S., Golber, A. [?], & Nushi, B. [?] (2024). RAPTOR: Recursive abstractive processing for tree-organized retrieval. *ICLR 2024*.

Sun, H., Dhingra, B., Zaheer, M., Mazaitis, K., Salakhutdinov, R., & Cohen, W. W. (2018). Open domain question answering using early fusion of knowledge bases and text. *EMNLP 2018*.

Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. *ACL 2023*.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *EMNLP 2018*.

Yasunaga, M., Ren, H., Bosselut, A., Liang, P., & Leskovec, J. (2021). QA-GNN: Reasoning with language models and knowledge graphs for question answering. *NAACL 2021*.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *ICLR 2020*.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2024). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
