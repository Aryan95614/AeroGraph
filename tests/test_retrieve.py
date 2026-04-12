"""Tests for hybrid retrieval."""

import math

from aerograph.retrieve import (
    rrf_fusion,
    _keyword_entity_extract,
    BM25Retriever,
    RetrievedChunk,
    RRF_K,
)
from aerograph.eval import (
    evaluate_reference_similarity,
    _safe_mean,
    _safe_std,
    _ci95,
)


class TestKeywordEntityExtract:
    def test_finds_aircraft(self):
        entities = _keyword_entity_extract("What caused the B737 engine failure?")
        assert "b737" in entities
        assert "engine failure" in entities or "engine" in entities

    def test_finds_phase(self):
        entities = _keyword_entity_extract("Problems during approach and landing")
        assert "approach" in entities
        assert "landing" in entities

    def test_empty_query(self):
        entities = _keyword_entity_extract("")
        assert isinstance(entities, list)


class TestRRFFusion:
    def test_basic_fusion(self):
        vector_results = [
            {"chunk_id": "c1", "text": "text1", "report_id": "r1",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.1},
            {"chunk_id": "c2", "text": "text2", "report_id": "r2",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.2},
        ]
        graph_scores = {"r1": 0.8, "r2": 0.3}

        fused = rrf_fusion(vector_results, graph_scores)
        assert len(fused) == 2
        # r1 should rank higher (top in both vector and graph)
        assert fused[0].chunk_id == "c1"
        assert fused[0].provenance == "both"

    def test_vector_only(self):
        vector_results = [
            {"chunk_id": "c1", "text": "text1", "report_id": "r1",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.1},
        ]
        fused = rrf_fusion(vector_results, {})
        assert len(fused) == 1
        assert fused[0].provenance == "vector"

    def test_empty_inputs(self):
        fused = rrf_fusion([], {})
        assert len(fused) == 0

    def test_score_ordering(self):
        vector_results = [
            {"chunk_id": f"c{i}", "text": f"t{i}", "report_id": f"r{i}",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.1 * i}
            for i in range(5)
        ]
        fused = rrf_fusion(vector_results, {})
        # Scores should be monotonically decreasing
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score

    def test_weight_influence(self):
        vector_results = [
            {"chunk_id": "c1", "text": "t1", "report_id": "r1",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.1},
            {"chunk_id": "c2", "text": "t2", "report_id": "r2",
             "chunk_index": 0, "entities_mentioned": [], "distance": 0.2},
        ]
        # r2 is top in graph, r1 is top in vector
        graph_scores = {"r2": 0.9, "r1": 0.1}

        # Heavy vector weight: c1 should win
        fused_v = rrf_fusion(vector_results, graph_scores, vector_weight=0.9, graph_weight=0.1)
        assert fused_v[0].chunk_id == "c1"

        # Heavy graph weight: c2 should win
        fused_g = rrf_fusion(vector_results, graph_scores, vector_weight=0.1, graph_weight=0.9)
        assert fused_g[0].chunk_id == "c2"


class TestReferenceSimilarity:
    def test_identical(self):
        score = evaluate_reference_similarity("hello world", "hello world")
        assert score == 1.0

    def test_no_overlap(self):
        score = evaluate_reference_similarity("alpha beta", "gamma delta")
        assert score == 0.0

    def test_partial_overlap(self):
        score = evaluate_reference_similarity(
            "bird strikes cause engine failure during approach",
            "engine failure from bird strike on approach phase",
        )
        assert 0.3 < score < 0.9

    def test_empty_reference(self):
        assert evaluate_reference_similarity("some answer", "") == 0.0

    def test_empty_answer(self):
        assert evaluate_reference_similarity("", "some reference") == 0.0


class TestBM25Scoring:
    """Test BM25 scoring math on a small synthetic corpus."""

    def _build_retriever(self, docs: list[str]) -> BM25Retriever:
        """Build a BM25Retriever with manually injected documents (no ChromaDB)."""
        r = BM25Retriever(k1=1.5, b=0.75)
        r._docs = [
            {"chunk_id": f"c{i}", "text": d, "report_id": f"r{i}", "entities_mentioned": []}
            for i, d in enumerate(docs)
        ]
        # Build index internals manually
        from collections import Counter
        n = len(r._docs)
        total_len = 0
        for idx, doc in enumerate(r._docs):
            tokens = r._tokenize(doc["text"])
            r._doc_lens.append(len(tokens))
            total_len += len(tokens)
            tf_counts = Counter(tokens)
            seen: set[str] = set()
            for term, tf in tf_counts.items():
                r._inverted_index.setdefault(term, []).append((idx, tf))
                if term not in seen:
                    r._doc_freqs[term] = r._doc_freqs.get(term, 0) + 1
                    seen.add(term)
        r._avgdl = total_len / n if n > 0 else 1.0
        r._initialized = True
        return r

    def test_tokenizer(self):
        r = BM25Retriever()
        tokens = r._tokenize("B737 engine failure during approach!")
        assert tokens == ["b737", "engine", "failure", "during", "approach"]

    def test_exact_match_scores_highest(self):
        docs = [
            "bird strike engine failure on takeoff",
            "hydraulic system maintenance report",
            "weather turbulence during cruise",
        ]
        r = self._build_retriever(docs)
        query_tokens = r._tokenize("engine failure")
        scores = [r._bm25_score(query_tokens, i) for i in range(3)]
        # Doc 0 contains both query terms, should score highest
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_no_match_scores_zero(self):
        docs = ["bird strike on takeoff", "weather turbulence cruise"]
        r = self._build_retriever(docs)
        query_tokens = r._tokenize("hydraulic failure")
        scores = [r._bm25_score(query_tokens, i) for i in range(2)]
        assert all(s == 0.0 for s in scores)

    def test_idf_weights_rare_terms_higher(self):
        # "rare" appears in 1 doc, "common" appears in all 3
        docs = [
            "common common common rare",
            "common common common",
            "common common common",
        ]
        r = self._build_retriever(docs)
        score_rare = r._bm25_score(r._tokenize("rare"), 0)
        score_common = r._bm25_score(r._tokenize("common"), 0)
        # Rare term should have higher IDF and thus higher score on doc 0
        assert score_rare > score_common

    def test_tf_saturation(self):
        # BM25 should saturate: doubling TF should not double score
        docs = ["engine", "engine engine engine engine engine"]
        r = self._build_retriever(docs)
        query_tokens = r._tokenize("engine")
        s0 = r._bm25_score(query_tokens, 0)
        s1 = r._bm25_score(query_tokens, 1)
        # s1 should be higher but not 5x higher due to saturation
        assert s1 > s0
        assert s1 < s0 * 5

    def test_bm25_score_positive(self):
        docs = ["bird strike caused engine failure during approach phase"]
        r = self._build_retriever(docs)
        query_tokens = r._tokenize("bird strike engine")
        score = r._bm25_score(query_tokens, 0)
        assert score > 0

    def test_length_normalization(self):
        # Shorter doc with same term should score higher than long doc (b > 0)
        docs = [
            "engine failure",
            "engine failure and also many other words about various topics in aviation safety",
        ]
        r = self._build_retriever(docs)
        query_tokens = r._tokenize("engine failure")
        s_short = r._bm25_score(query_tokens, 0)
        s_long = r._bm25_score(query_tokens, 1)
        assert s_short > s_long


class TestStatisticalFunctions:
    def test_safe_mean(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert _safe_mean([]) == 0.0

    def test_safe_std(self):
        std = _safe_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert 1.9 < std < 2.2  # expected ~2.07
        assert _safe_std([]) == 0.0
        assert _safe_std([5.0]) == 0.0

    def test_ci95(self):
        ci = _ci95([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ci > 0
        assert _ci95([]) == 0.0
