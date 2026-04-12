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


