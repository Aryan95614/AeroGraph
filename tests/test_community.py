"""Tests for community detection and global search."""

import networkx as nx
import numpy as np

from aerograph.community import (
    networkx_to_igraph,
    detect_communities,
    get_community_subgraph,
    community_stats,
)
from aerograph.graph import clean_graph
from aerograph.retrieve import (
    global_search,
    is_global_query,
)


def _make_test_graph() -> nx.DiGraph:
    """Build a small test graph with clear community structure."""
    G = nx.DiGraph()
    # Community 1: bird strike cluster
    G.add_node("bird_strike", type="Event", report_ids=["r1", "r2"])
    G.add_node("engine_failure", type="Event", report_ids=["r1", "r3"])
    G.add_node("go_around", type="Outcome", report_ids=["r1"])
    G.add_edge("bird_strike", "engine_failure", type="CAUSED_BY", weight=3)
    G.add_edge("engine_failure", "go_around", type="CAUSED_BY", weight=2)
    G.add_edge("bird_strike", "go_around", type="CONTRIBUTED_TO", weight=1)

    # Community 2: weather cluster
    G.add_node("thunderstorm", type="Weather", report_ids=["r4", "r5"])
    G.add_node("wind_shear", type="Weather", report_ids=["r4", "r6"])
    G.add_node("turbulence", type="Weather", report_ids=["r5", "r6"])
    G.add_edge("thunderstorm", "wind_shear", type="CO_OCCURRED_WITH", weight=5)
    G.add_edge("thunderstorm", "turbulence", type="CAUSED_BY", weight=4)
    G.add_edge("wind_shear", "turbulence", type="CO_OCCURRED_WITH", weight=2)

    # Community 3: ATC cluster
    G.add_node("atc_katl", type="ATC_Facility", report_ids=["r7"])
    G.add_node("runway_incursion", type="Event", report_ids=["r7", "r8"])
    G.add_node("communication_failure", type="Event", report_ids=["r8"])
    G.add_edge("atc_katl", "runway_incursion", type="INVOLVED", weight=2)
    G.add_edge("communication_failure", "runway_incursion", type="CAUSED_BY", weight=1)

    # Weak inter-community link
    G.add_edge("go_around", "wind_shear", type="CAUSED_BY", weight=1)

    return G


class TestNetworkXToIgraph:
    def test_preserves_node_count(self):
        G = _make_test_graph()
        ig, node_list = networkx_to_igraph(G)
        assert ig.vcount() == G.number_of_nodes()

    def test_preserves_node_attributes(self):
        G = _make_test_graph()
        ig, node_list = networkx_to_igraph(G)
        # Check type attribute preserved
        types = ig.vs["type"]
        assert "Event" in types
        assert "Weather" in types
        assert "ATC_Facility" in types

    def test_preserves_report_ids(self):
        G = _make_test_graph()
        ig, node_list = networkx_to_igraph(G)
        report_ids = ig.vs["report_ids"]
        # bird_strike should have r1, r2
        bird_idx = node_list.index("bird_strike")
        assert "r1" in report_ids[bird_idx]
        assert "r2" in report_ids[bird_idx]

    def test_edge_count(self):
        G = _make_test_graph()
        ig, node_list = networkx_to_igraph(G)
        # Undirected dedup: 10 directed edges -> <= 10 undirected edges
        assert ig.ecount() <= G.number_of_edges()
        assert ig.ecount() > 0

    def test_edge_weights_preserved(self):
        G = _make_test_graph()
        ig, _ = networkx_to_igraph(G)
        weights = ig.es["weight"]
        assert all(w >= 1 for w in weights)

    def test_empty_graph(self):
        G = nx.DiGraph()
        ig, node_list = networkx_to_igraph(G)
        assert ig.vcount() == 0
        assert node_list == []


class TestDetectCommunities:
    def test_non_empty_at_each_level(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        for level in range(3):
            assert level in hierarchy
            assert len(hierarchy[level]) > 0

    def test_every_node_in_one_community(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        total_nodes = G.number_of_nodes()
        for level in range(3):
            all_members = []
            for members in hierarchy[level].values():
                all_members.extend(members)
            # Every node appears exactly once
            assert len(all_members) == total_nodes
            assert len(set(all_members)) == total_nodes

    def test_node_attributes_set(self):
        G = _make_test_graph()
        detect_communities(G)
        for node in G.nodes():
            assert "community_L0" in G.nodes[node]
            assert "community_L1" in G.nodes[node]
            assert "community_L2" in G.nodes[node]

    def test_coarser_has_fewer_or_equal_communities(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        # Level 0 (resolution=1.0) should have >= communities as level 2 (resolution=0.1)
        assert len(hierarchy[0]) >= len(hierarchy[2])


class TestGetCommunitySubgraph:
    def test_subgraph_nodes(self):
        G = _make_test_graph()
        sub = get_community_subgraph(G, ["bird_strike", "engine_failure", "go_around"])
        assert sub.number_of_nodes() == 3

    def test_subgraph_edges(self):
        G = _make_test_graph()
        sub = get_community_subgraph(G, ["bird_strike", "engine_failure", "go_around"])
        # All 3 edges within this cluster should be present
        assert sub.number_of_edges() == 3

    def test_no_external_edges(self):
        G = _make_test_graph()
        sub = get_community_subgraph(G, ["thunderstorm", "wind_shear", "turbulence"])
        # go_around -> wind_shear edge should NOT be in subgraph
        assert not sub.has_edge("go_around", "wind_shear")

    def test_empty_community(self):
        G = _make_test_graph()
        sub = get_community_subgraph(G, [])
        assert sub.number_of_nodes() == 0


class TestCommunityStats:
    def test_returns_per_level(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        stats = community_stats(G, hierarchy)
        assert 0 in stats
        assert 1 in stats
        assert 2 in stats

    def test_stat_keys(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        stats = community_stats(G, hierarchy)
        for level, s in stats.items():
            assert "num_communities" in s
            assert "mean_size" in s
            assert "median_size" in s
            assert "max_size" in s
            assert "modularity" in s

    def test_modularity_range(self):
        G = _make_test_graph()
        hierarchy = detect_communities(G)
        stats = community_stats(G, hierarchy)
        for level, s in stats.items():
            # Modularity should be between -0.5 and 1.0
            assert -0.5 <= s["modularity"] <= 1.0


class TestGraphCleanup:
    def test_removes_placeholder(self):
        G = nx.DiGraph()
        G.add_node("aircraft x", type="Aircraft", report_ids=["r1"])
        G.add_node("engine_failure", type="Event", report_ids=["r1"])
        G.add_node("go_around", type="Outcome", report_ids=["r1"])
        G.add_edge("aircraft x", "engine_failure", type="INVOLVED")
        G.add_edge("engine_failure", "go_around", type="CAUSED_BY")
        import tempfile, pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(G, f)
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name
        from pathlib import Path
        cleaned = clean_graph(Path(input_path), Path(output_path))
        assert not cleaned.has_node("aircraft x")
        assert cleaned.has_node("engine_failure")

    def test_removes_isolated_nodes(self):
        G = nx.DiGraph()
        G.add_node("connected_a", type="Event", report_ids=[])
        G.add_node("connected_b", type="Event", report_ids=[])
        G.add_node("isolated", type="Event", report_ids=[])
        G.add_edge("connected_a", "connected_b", type="CAUSED_BY")
        import tempfile, pickle
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(G, f)
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name
        cleaned = clean_graph(Path(input_path), Path(output_path))
        assert not cleaned.has_node("isolated")
        assert cleaned.has_node("connected_a")

    def test_removes_self_loops(self):
        G = nx.DiGraph()
        G.add_node("a", type="Event", report_ids=[])
        G.add_node("b", type="Event", report_ids=[])
        G.add_edge("a", "b", type="CAUSED_BY")
        G.add_edge("a", "a", type="SELF")
        import tempfile, pickle
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(G, f)
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name
        cleaned = clean_graph(Path(input_path), Path(output_path))
        assert not list(nx.selfloop_edges(cleaned))

    def test_fixes_missing_type(self):
        G = nx.DiGraph()
        G.add_node("a", report_ids=[])  # no type attribute
        G.add_node("b", type="Event", report_ids=[])
        G.add_edge("a", "b", type="CAUSED_BY")
        import tempfile, pickle
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(G, f)
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            output_path = f.name
        cleaned = clean_graph(Path(input_path), Path(output_path))
        assert cleaned.nodes["a"]["type"] == "unknown"


class TestIsGlobalQuery:
    def test_global_patterns(self):
        assert is_global_query("What are the most common causes of runway incursions?")
        assert is_global_query("Overall patterns in bird strike incidents")
        assert is_global_query("Trends across all reports for engine failure")
        assert is_global_query("How often do TCAS RAs occur during approach?")

    def test_local_queries(self):
        assert not is_global_query("What happened in ACN 1850000?")
        assert not is_global_query("Explain the causal chain from bird strike to go-around")
        assert not is_global_query("TCAS RA during approach at ATL")


class TestGlobalSearch:
    def test_with_summaries(self):
        summaries = {
            "L1_C0": {
                "summary": "Bird strike events causing engine failures during takeoff phase.",
                "node_count": 5,
                "edge_count": 8,
                "report_ids": ["r1", "r2"],
                "level": 1,
            },
            "L1_C1": {
                "summary": "Weather-related incidents involving thunderstorms and wind shear on approach.",
                "node_count": 4,
                "edge_count": 6,
                "report_ids": ["r3", "r4"],
                "level": 1,
            },
        }
        results = global_search("bird strike engine failure", summaries, top_k=2)
        assert len(results) == 2
        # Bird strike summary should rank higher for this query
        assert results[0]["community_id"] == "L1_C0"
        assert results[0]["score"] > results[1]["score"]

    def test_empty_summaries(self):
        results = global_search("any query", {}, top_k=5)
        assert results == []

    def test_returns_community_ids(self):
        summaries = {
            "L1_C5": {
                "summary": "Runway incursion events at major airports.",
                "node_count": 3,
                "edge_count": 2,
                "report_ids": ["r10"],
                "level": 1,
            },
        }
        results = global_search("runway incursion", summaries)
        assert results[0]["community_id"] == "L1_C5"
        assert "report_ids" in results[0]
