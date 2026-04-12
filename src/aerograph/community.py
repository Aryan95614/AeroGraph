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


def summarize_community(
    G: nx.DiGraph,
    community_nodes: list[str],
    client,
) -> str:
    """Generate a natural-language summary of a community using Claude.

    Builds a prompt with all entities, relations, and report IDs in the
    community, then asks Claude to summarize safety themes and causal patterns.
    """
    subgraph = get_community_subgraph(G, community_nodes)

    # Collect entity info
    entities = []
    all_report_ids: set[str] = set()
    for node in subgraph.nodes():
        data = subgraph.nodes[node]
        ntype = data.get("type", "unknown")
        entities.append(f"- {node} ({ntype})")
        for rid in data.get("report_ids", []):
            all_report_ids.add(rid)

    # Collect relations
    relations = []
    for u, v, data in subgraph.edges(data=True):
        etype = data.get("type", "RELATED")
        relations.append(f"- {u} --[{etype}]--> {v}")

    entity_block = "\n".join(entities[:100])  # Cap for token budget
    relation_block = "\n".join(relations[:100])
    report_block = ", ".join(sorted(all_report_ids)[:50])

    prompt = f"""Summarize the key safety themes, causal patterns, and notable incidents in this cluster of aviation safety entities. Be specific about aircraft types, airports, failure modes, and contributing factors. Reference specific ASRS accession numbers.

## Entities ({len(entities)} total)
{entity_block}

## Relations ({len(relations)} total)
{relation_block}

## Associated ASRS Reports
{report_block}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def summarize_all_communities(
    G: nx.DiGraph,
    communities: dict[int, dict[int, list[str]]],
    client,
    level: int = 1,
    max_communities: int = 100,
    cache_path: Optional[str] = None,
) -> dict[str, dict]:
    """Batch summarize communities at a given hierarchy level.

    Caches results to JSON. Skips communities with fewer than 3 nodes.
    Rate limits to max 10 requests/minute. Resumes from cache if interrupted.

    Returns dict mapping community_id -> {summary, node_count, edge_count,
    report_ids, level}.
    """
    if cache_path is None:
        cache_path = str(DATA_DIR / "community_summaries.json")

    # Load existing cache
    cache: dict[str, dict] = {}
    if Path(cache_path).exists():
        with open(cache_path) as f:
            cache = json.load(f)

    if level not in communities:
        print(f"Level {level} not found in community hierarchy")
        return cache

    level_comms = communities[level]

    # Filter and sort by size (largest first)
    eligible = [
        (cid, members) for cid, members in level_comms.items()
        if len(members) >= 3
    ]
    eligible.sort(key=lambda x: len(x[1]), reverse=True)
    eligible = eligible[:max_communities]

    print(f"Summarizing {len(eligible)} communities at level {level} "
          f"(of {len(level_comms)} total, {len(level_comms) - len(eligible)} skipped < 3 nodes)")

    summarized = 0
    request_times: list[float] = []

    for i, (cid, members) in enumerate(eligible):
        cache_key = f"L{level}_C{cid}"

        if cache_key in cache:
            print(f"  [{i+1}/{len(eligible)}] C{cid} ({len(members)} nodes) — cached")
            continue

        # Rate limiting: max 10 requests per minute
        now = time.time()
        request_times = [t for t in request_times if now - t < 60]
        if len(request_times) >= 10:
            wait = 60 - (now - request_times[0]) + 0.5
            print(f"  Rate limit: waiting {wait:.1f}s")
            time.sleep(wait)

        print(f"  [{i+1}/{len(eligible)}] C{cid} ({len(members)} nodes)...", end=" ", flush=True)

        subgraph = get_community_subgraph(G, members)
        all_report_ids = set()
        for node in subgraph.nodes():
            for rid in subgraph.nodes[node].get("report_ids", []):
                all_report_ids.add(rid)

        try:
            summary = summarize_community(G, members, client)
            request_times.append(time.time())
            summarized += 1
            print("done")
        except Exception as e:
            print(f"error: {e}")
            summary = f"[Summarization failed: {e}]"

        cache[cache_key] = {
            "summary": summary,
            "node_count": len(members),
            "edge_count": subgraph.number_of_edges(),
            "report_ids": sorted(all_report_ids),
            "level": level,
        }

        # Save after each summary for resilience
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    print(f"\nSummarized {summarized} new communities ({len(cache)} total cached)")
    return cache


def run_community_detection(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> tuple[nx.DiGraph, dict]:
    """Run community detection pipeline and save annotated graph.

    Returns (graph_with_community_attrs, hierarchy).
    """
    if input_path is None:
        clean_path = GRAPHS_DIR / "aerograph_clean.pkl"
        normalized_path = GRAPHS_DIR / "aerograph_normalized.pkl"
        original_path = GRAPHS_DIR / "aerograph.pkl"
        for p in [clean_path, normalized_path, original_path]:
            if p.exists():
                input_path = p
                break
    if input_path is None or not input_path.exists():
        raise FileNotFoundError("No graph pickle found. Run `aerograph build` first.")

    if output_path is None:
        output_path = GRAPHS_DIR / "aerograph_communities.pkl"

    import pickle
    with open(input_path, "rb") as f:
        G: nx.DiGraph = pickle.load(f)

    print(f"Running Leiden community detection on {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    hierarchy = detect_communities(G)
    stats = community_stats(G, hierarchy)

    print(f"\nCommunity hierarchy:")
    for level, s in stats.items():
        print(f"  Level {level}: {s['num_communities']} communities, "
              f"modularity={s['modularity']}, "
              f"sizes: median={s['median_size']}, max={s['max_size']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved community-annotated graph to {output_path}")

    return G, hierarchy


def run_summarization(
    level: int = 1,
    max_communities: int = 100,
) -> dict:
    """Run community summarization pipeline."""
    import pickle

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Cannot summarize communities."
        )

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Load community-annotated graph
    communities_path = GRAPHS_DIR / "aerograph_communities.pkl"
    if not communities_path.exists():
        raise FileNotFoundError(
            "No community-annotated graph. Run `aerograph communities` first."
        )

    with open(communities_path, "rb") as f:
        G: nx.DiGraph = pickle.load(f)

    # Reconstruct hierarchy from node attributes
    hierarchy: dict[int, dict[int, list[str]]] = {}
    for node in G.nodes():
        data = G.nodes[node]
        for l in range(3):
            attr = f"community_L{l}"
            if attr in data:
                comm_id = data[attr]
                hierarchy.setdefault(l, {}).setdefault(comm_id, []).append(node)

    if level not in hierarchy:
        print(f"Level {level} not found. Available levels: {list(hierarchy.keys())}")
        return {}

    return summarize_all_communities(
        G, hierarchy, client,
        level=level,
        max_communities=max_communities,
    )
