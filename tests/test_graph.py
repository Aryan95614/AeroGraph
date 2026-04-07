"""Tests for knowledge graph construction and queries."""

import tempfile
from pathlib import Path

from aerograph.graph import (
    NetworkXBackend,
    GraphNode,
    GraphEdge,
    SubgraphResult,
)


class TestNetworkXBackend:
    def _make_backend(self, tmp_path=None):
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp()) / "test.pkl"
        return NetworkXBackend(path=tmp_path)

    def test_add_and_get_node(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(
            name="B737", type="Aircraft", canonical_name="b737",
            report_ids=["ACN001"],
        ))
        node = backend.get_node("b737")
        assert node is not None
        assert node.type == "Aircraft"
        assert "ACN001" in node.report_ids

    def test_merge_duplicate_nodes(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(
            name="B737", type="Aircraft", canonical_name="b737",
            report_ids=["ACN001"],
        ))
        backend.add_node(GraphNode(
            name="B737", type="Aircraft", canonical_name="b737",
            report_ids=["ACN002"],
        ))
        node = backend.get_node("b737")
        assert set(node.report_ids) == {"ACN001", "ACN002"}
        assert backend.node_count() == 1

    def test_add_edge(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="Bird Strike", type="Event", canonical_name="bird strike", report_ids=["1"]))
        backend.add_node(GraphNode(name="Engine Failure", type="Event", canonical_name="engine failure", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="bird strike", target="engine failure", type="CAUSED_BY", report_ids=["1"]))
        assert backend.edge_count() == 1

    def test_merge_duplicate_edges(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="B", type="Event", canonical_name="b", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["2"]))
        assert backend.edge_count() == 1
        edge_data = backend.graph.edges["a", "b"]
        assert edge_data["weight"] == 2
        assert set(edge_data["report_ids"]) == {"1", "2"}

    def test_get_neighbors(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="B", type="Factor", canonical_name="b", report_ids=["1"]))
        backend.add_node(GraphNode(name="C", type="Outcome", canonical_name="c", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="b", target="c", type="CONTRIBUTED_TO", report_ids=["1"]))
        result = backend.get_neighbors("a", depth=2)
        node_names = {n.canonical_name for n in result.nodes}
        assert "a" in node_names
        assert "b" in node_names
        assert "c" in node_names  # 2 hops away

    def test_get_neighbors_depth_1(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="B", type="Factor", canonical_name="b", report_ids=["1"]))
        backend.add_node(GraphNode(name="C", type="Outcome", canonical_name="c", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="b", target="c", type="CONTRIBUTED_TO", report_ids=["1"]))
        result = backend.get_neighbors("a", depth=1)
        node_names = {n.canonical_name for n in result.nodes}
        assert "a" in node_names
        assert "b" in node_names
        assert "c" not in node_names  # too far

    def test_hub_node_skipped(self):
        backend = self._make_backend()
        # Create a hub node with many connections
        backend.add_node(GraphNode(name="Hub", type="Aircraft", canonical_name="hub", report_ids=["1"]))
        backend.add_node(GraphNode(name="Start", type="Event", canonical_name="start", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="start", target="hub", type="INVOLVED", report_ids=["1"]))
        # Add many neighbors to hub to exceed max_degree
        for i in range(10):
            n = f"spoke{i}"
            backend.add_node(GraphNode(name=n, type="Event", canonical_name=n, report_ids=["1"]))
            backend.add_edge(GraphEdge(source="hub", target=n, type="INVOLVED", report_ids=["1"]))
        # With max_degree=5, hub should be skipped during expansion
        result = backend.get_neighbors("start", depth=2, max_degree=5)
        node_names = {n.canonical_name for n in result.nodes}
        assert "hub" in node_names  # direct neighbor, still included
        assert len([n for n in node_names if n.startswith("spoke")]) == 0  # spokes not reached

    def test_causal_chain(self):
        backend = self._make_backend()
        for name in ["a", "b", "c"]:
            backend.add_node(GraphNode(name=name, type="Event", canonical_name=name, report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="b", target="c", type="CONTRIBUTED_TO", report_ids=["1"]))
        paths = backend.get_causal_chain("a", "c")
        assert len(paths) > 0
        assert paths[0] == ["a", "b", "c"]

    def test_causal_chain_no_path(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="a", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="z", type="Event", canonical_name="z", report_ids=["1"]))
        paths = backend.get_causal_chain("a", "z")
        assert paths == []

    def test_get_subgraph(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["R1"]))
        backend.add_node(GraphNode(name="B", type="Factor", canonical_name="b", report_ids=["R1"]))
        backend.add_node(GraphNode(name="C", type="Event", canonical_name="c", report_ids=["R2"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["R1"]))
        result = backend.get_subgraph("R1")
        node_names = {n.canonical_name for n in result.nodes}
        assert "a" in node_names
        assert "b" in node_names
        assert "c" not in node_names  # different report

    def test_high_centrality(self):
        backend = self._make_backend()
        # Star graph: spokes point TO center (center has highest in-degree → highest PageRank)
        backend.add_node(GraphNode(name="center", type="Event", canonical_name="center", report_ids=["1"]))
        for i in range(5):
            n = f"spoke{i}"
            backend.add_node(GraphNode(name=n, type="Factor", canonical_name=n, report_ids=["1"]))
            backend.add_edge(GraphEdge(source=n, target="center", type="INVOLVED", report_ids=["1"]))
        top = backend.get_high_centrality_nodes(3)
        assert len(top) == 3
        assert top[0][0] == "center"  # highest centrality (most incoming edges)

    def test_temporal_chain(self):
        backend = self._make_backend()
        for name in ["takeoff", "climb", "cruise"]:
            backend.add_node(GraphNode(name=name, type="Phase", canonical_name=name, report_ids=["1"]))
        backend.add_edge(GraphEdge(source="takeoff", target="climb", type="TEMPORAL_SEQUENCE", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="climb", target="cruise", type="TEMPORAL_SEQUENCE", report_ids=["1"]))
        chain = backend.get_temporal_chain("takeoff")
        assert chain == ["takeoff", "climb", "cruise"]

    def test_report_timeline(self):
        backend = self._make_backend()
        for name in ["event_a", "event_b", "event_c"]:
            backend.add_node(GraphNode(name=name, type="Event", canonical_name=name, report_ids=["R1"]))
        backend.add_edge(GraphEdge(source="event_a", target="event_b", type="TEMPORAL_SEQUENCE", report_ids=["R1"]))
        backend.add_edge(GraphEdge(source="event_b", target="event_c", type="TEMPORAL_SEQUENCE", report_ids=["R1"]))
        timeline = backend.get_report_timeline("R1")
        assert timeline == ["event_a", "event_b", "event_c"]

    def test_save_and_load(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "test_save.pkl"
        backend = self._make_backend(tmp)
        backend.add_node(GraphNode(name="Test", type="Event", canonical_name="test", report_ids=["1"]))
        backend.save()
        # Load from same path
        loaded = NetworkXBackend(path=tmp)
        assert loaded.node_count() == 1
        node = loaded.get_node("test")
        assert node is not None
        assert node.type == "Event"

    def test_type_counts(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="B", type="Event", canonical_name="b", report_ids=["1"]))
        backend.add_node(GraphNode(name="C", type="Factor", canonical_name="c", report_ids=["1"]))
        counts = backend.get_type_counts()
        assert counts["Event"] == 2
        assert counts["Factor"] == 1

    def test_edge_type_counts(self):
        backend = self._make_backend()
        backend.add_node(GraphNode(name="A", type="Event", canonical_name="a", report_ids=["1"]))
        backend.add_node(GraphNode(name="B", type="Factor", canonical_name="b", report_ids=["1"]))
        backend.add_edge(GraphEdge(source="a", target="b", type="CAUSED_BY", report_ids=["1"]))
        counts = backend.get_edge_type_counts()
        assert counts["CAUSED_BY"] == 1

    def test_node_not_found(self):
        backend = self._make_backend()
        assert backend.get_node("nonexistent") is None

    def test_empty_graph(self):
        backend = self._make_backend()
        assert backend.node_count() == 0
        assert backend.edge_count() == 0
        assert backend.get_high_centrality_nodes() == []
