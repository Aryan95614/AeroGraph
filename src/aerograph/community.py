"""Community detection and summarization (Edge et al., arXiv:2404.16130).

Implements the Microsoft GraphRAG community layer:
  1. Leiden hierarchical community detection via leidenalg/igraph
  2. LLM-powered community summarization
  3. Community metadata for global search
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
GRAPHS_DIR = DATA_DIR / "graphs"


def networkx_to_igraph(G: nx.DiGraph):
    """Convert a NetworkX DiGraph to an igraph Graph.

    Preserves node attributes (type, name, report_ids, canonical_name) and
    edge attributes (type, weight, report_ids). igraph works on undirected
    graphs for community detection, so directionality is dropped.

    Returns (ig_graph, node_list) where node_list maps igraph vertex indices
    to NetworkX node names.
    """
    import igraph as ig

    node_list = list(G.nodes())
    node_index = {name: i for i, name in enumerate(node_list)}

    ig_graph = ig.Graph(n=len(node_list), directed=False)

    # Copy node attributes
    ig_graph.vs["name"] = node_list
    ig_graph.vs["type"] = [G.nodes[n].get("type", "unknown") for n in node_list]
    ig_graph.vs["report_ids"] = [G.nodes[n].get("report_ids", []) for n in node_list]
    ig_graph.vs["canonical_name"] = [G.nodes[n].get("canonical_name", n) for n in node_list]

    # Build edge list (undirected, deduplicate)
    seen_edges: set[tuple[int, int]] = set()
    edges = []
    weights = []
    edge_types = []
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        pair = (min(i, j), max(i, j))
        if pair not in seen_edges:
            seen_edges.add(pair)
            edges.append(pair)
            weights.append(data.get("weight", 1))
            edge_types.append(data.get("type", "unknown"))

    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = weights
    ig_graph.es["type"] = edge_types

    return ig_graph, node_list


def detect_communities(
    G: nx.DiGraph,
    resolution_params: Optional[list[float]] = None,
    n_levels: int = 3,
) -> dict[int, dict[int, list[str]]]:
    """Run Leiden community detection at multiple resolutions.

    Produces a hierarchy where level 0 is finest (highest resolution)
    and level n_levels-1 is coarsest (lowest resolution).

    Args:
        G: NetworkX DiGraph to detect communities in.
        resolution_params: Resolution parameters per level. Default [1.0, 0.5, 0.1].
        n_levels: Number of hierarchy levels (must match len(resolution_params)).

    Returns:
        dict mapping level -> {community_id: [node_names]}
        Each node appears in exactly one community per level.
    """
    import igraph as ig
    import leidenalg

    if resolution_params is None:
        resolution_params = [1.0, 0.5, 0.1]
    n_levels = len(resolution_params)

    ig_graph, node_list = networkx_to_igraph(G)

    hierarchy: dict[int, dict[int, list[str]]] = {}

    for level, resolution in enumerate(resolution_params):
        print(f"  Level {level}: resolution={resolution}")

        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
            seed=42,
        )

        communities: dict[int, list[str]] = {}
        for node_idx, comm_id in enumerate(partition.membership):
            communities.setdefault(comm_id, []).append(node_list[node_idx])

        hierarchy[level] = communities

        # Store as node attributes on the original graph
        attr_name = f"community_L{level}"
        for comm_id, members in communities.items():
            for node_name in members:
                if G.has_node(node_name):
                    G.nodes[node_name][attr_name] = comm_id

        print(f"    {len(communities)} communities "
              f"(median size: {sorted(len(m) for m in communities.values())[len(communities)//2]})")

    return hierarchy


def get_community_subgraph(G: nx.DiGraph, community_nodes: list[str]) -> nx.DiGraph:
    """Extract the induced subgraph for a community.

    Returns a DiGraph containing only the specified nodes and all edges
    between them from the original graph.
    """
    return G.subgraph(community_nodes).copy()


def community_stats(
    G: nx.DiGraph,
    communities: dict[int, dict[int, list[str]]],
) -> dict:
    """Compute per-level statistics for the community hierarchy.

    Returns dict with per-level: num_communities, mean/median/max size,
    modularity score (computed on undirected graph).
    """
    import igraph as ig
    import leidenalg

    ig_graph, node_list = networkx_to_igraph(G)
    node_index = {name: i for i, name in enumerate(node_list)}

    stats: dict[int, dict] = {}

    for level, comms in communities.items():
        sizes = sorted(len(members) for members in comms.values())
        n = len(sizes)

        # Compute modularity using igraph membership vector
        membership = [0] * len(node_list)
        for comm_id, members in comms.items():
            for node_name in members:
                if node_name in node_index:
                    membership[node_index[node_name]] = comm_id

        modularity = ig_graph.modularity(membership, weights="weight")

        stats[level] = {
            "num_communities": n,
            "mean_size": sum(sizes) / n if n else 0,
            "median_size": sizes[n // 2] if n else 0,
            "max_size": sizes[-1] if sizes else 0,
            "min_size": sizes[0] if sizes else 0,
            "modularity": round(modularity, 4),
        }

    return stats


