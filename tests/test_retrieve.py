"""Tests for hybrid retrieval."""

from aerograph.retrieve import (
    rrf_fusion,
    _keyword_entity_extract,
    RetrievedChunk,
    RRF_K,
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
