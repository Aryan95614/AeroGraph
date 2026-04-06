"""Knowledge graph construction and query module.

Supports Neo4j (preferred) with automatic fallback to NetworkX.
Graph is built from entity extraction results with MERGE semantics.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
GRAPHS_DIR = DATA_DIR / "graphs"


@dataclass
class GraphNode:
    name: str
    type: str
    canonical_name: str
    report_ids: list[str] = field(default_factory=list)
    embedding_id: Optional[str] = None
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str
    weight: int = 1
    report_ids: list[str] = field(default_factory=list)


@dataclass
class SubgraphResult:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphBackend(ABC):
    """Abstract graph backend interface."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def get_node(self, canonical_name: str) -> Optional[GraphNode]: ...

    @abstractmethod
    def get_neighbors(
        self, entity: str, edge_types: Optional[list[str]] = None, depth: int = 2,
        max_degree: int = 200,
    ) -> SubgraphResult: ...

    @abstractmethod
    def get_causal_chain(
        self, start_entity: str, end_entity: str
    ) -> list[list[str]]: ...

    @abstractmethod
    def get_high_centrality_nodes(self, top_n: int = 20) -> list[tuple[str, float]]: ...

    @abstractmethod
    def get_subgraph(self, report_id: str) -> SubgraphResult: ...

    @abstractmethod
    def get_temporal_chain(self, entity: str) -> list[str]: ...

    @abstractmethod
    def get_report_timeline(self, report_id: str) -> list[str]: ...

    @abstractmethod
    def node_count(self) -> int: ...

    @abstractmethod
    def edge_count(self) -> int: ...

    @abstractmethod
    def save(self) -> None: ...


class NetworkXBackend(GraphBackend):
    """NetworkX-based graph backend (zero-dependency fallback)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or GRAPHS_DIR / "aerograph.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.graph: nx.DiGraph = pickle.load(f)
        else:
            self.graph = nx.DiGraph()

    def add_node(self, node: GraphNode) -> None:
        key = node.canonical_name
        if self.graph.has_node(key):
            existing = self.graph.nodes[key]
            existing_ids = set(existing.get("report_ids", []))
            existing_ids.update(node.report_ids)
            existing["report_ids"] = list(existing_ids)
            if node.embedding_id:
                existing["embedding_id"] = node.embedding_id
        else:
            self.graph.add_node(key, **{
                "name": node.name,
                "type": node.type,
                "canonical_name": node.canonical_name,
                "report_ids": list(node.report_ids),
                "embedding_id": node.embedding_id,
                **node.properties,
            })

    def add_edge(self, edge: GraphEdge) -> None:
        if self.graph.has_edge(edge.source, edge.target):
            existing = self.graph.edges[edge.source, edge.target]
            existing["weight"] = existing.get("weight", 1) + 1
            existing_ids = set(existing.get("report_ids", []))
            existing_ids.update(edge.report_ids)
            existing["report_ids"] = list(existing_ids)
            # Track all relation types seen on this edge
            types = existing.get("types", [existing.get("type", "")])
            if edge.type not in types:
                types.append(edge.type)
            existing["types"] = types
            return
        edge_attrs = {
            "type": edge.type,
            "types": [edge.type],
            "weight": edge.weight,
            "report_ids": list(edge.report_ids),
        }
        # Store report date on temporal edges for cross-report temporal reasoning
        if edge.type == "TEMPORAL_SEQUENCE":
            for rid in edge.report_ids:
                if self.graph.has_node(edge.source):
                    src_data = self.graph.nodes[edge.source]
                    if "report_date" in src_data:
                        edge_attrs["report_date"] = src_data["report_date"]
                        break
        self.graph.add_edge(edge.source, edge.target, **edge_attrs)

    def get_node(self, canonical_name: str) -> Optional[GraphNode]:
        key = canonical_name.lower().strip()
        if not self.graph.has_node(key):
            return None
        data = self.graph.nodes[key]
        return GraphNode(
            name=data.get("name", key),
            type=data.get("type", "unknown"),
            canonical_name=key,
            report_ids=data.get("report_ids", []),
            embedding_id=data.get("embedding_id"),
        )

    def get_neighbors(
        self, entity: str, edge_types: Optional[list[str]] = None, depth: int = 2,
        max_degree: int = 200,
    ) -> SubgraphResult:
        key = entity.lower().strip()
        if not self.graph.has_node(key):
            return SubgraphResult(nodes=[], edges=[])

        # BFS to collect neighbors up to depth
        # Skip hub nodes (degree > max_degree) to prevent explosion
        visited = {key}
        frontier = {key}
        all_nodes = set()
        all_edges = []

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                node_degree = self.graph.degree(node)
                if node != key and node_degree > max_degree:
                    continue  # skip hub nodes during expansion
                for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                    edge_data = self.graph.edges.get((node, neighbor)) or self.graph.edges.get((neighbor, node))
                    if edge_data and (edge_types is None or edge_data.get("type") in edge_types):
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
                            visited.add(neighbor)
                        all_nodes.add(neighbor)
                        all_edges.append(GraphEdge(
                            source=node if self.graph.has_edge(node, neighbor) else neighbor,
                            target=neighbor if self.graph.has_edge(node, neighbor) else node,
                            type=edge_data.get("type", ""),
                            weight=edge_data.get("weight", 1),
                            report_ids=edge_data.get("report_ids", []),
                        ))
            frontier = next_frontier

        all_nodes.add(key)
        nodes = []
        for n in all_nodes:
            node = self.get_node(n)
            if node:
                nodes.append(node)

        return SubgraphResult(nodes=nodes, edges=all_edges)

    def get_causal_chain(
        self, start_entity: str, end_entity: str
    ) -> list[list[str]]:
        start = start_entity.lower().strip()
        end = end_entity.lower().strip()
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []

        causal_types = {"CAUSED_BY", "CONTRIBUTED_TO", "PRECEDED_BY"}
        causal_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") in causal_types:
                causal_graph.add_edge(u, v)

        # Try directed paths first (respects causal direction)
        try:
            paths = list(nx.all_simple_paths(causal_graph, start, end, cutoff=5))
            if not paths:
                # Fall back to undirected search if no directed path exists
                paths = list(nx.all_simple_paths(
                    causal_graph.to_undirected(), start, end, cutoff=5
                ))
            return [list(p) for p in paths[:10]]  # cap at 10 paths
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

    def get_high_centrality_nodes(self, top_n: int = 20) -> list[tuple[str, float]]:
        if self.graph.number_of_nodes() == 0:
            return []
        try:
            pr = nx.pagerank(self.graph, max_iter=100)
            sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_n]
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality
            dc = nx.degree_centrality(self.graph)
            sorted_nodes = sorted(dc.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:top_n]

    def get_subgraph(self, report_id: str) -> SubgraphResult:
        nodes = []
        for n, data in self.graph.nodes(data=True):
            if report_id in data.get("report_ids", []):
                nodes.append(GraphNode(
                    name=data.get("name", n),
                    type=data.get("type", "unknown"),
                    canonical_name=n,
                    report_ids=data.get("report_ids", []),
                ))

        node_names = {n.canonical_name for n in nodes}
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_names and v in node_names:
                edges.append(GraphEdge(
                    source=u, target=v,
                    type=data.get("type", ""),
                    weight=data.get("weight", 1),
                    report_ids=data.get("report_ids", []),
                ))

        return SubgraphResult(nodes=nodes, edges=edges)

    def get_temporal_chain(self, entity: str) -> list[str]:
        """Find all nodes connected via TEMPORAL_SEQUENCE edges and return in topological order."""
        key = entity.lower().strip()
        if not self.graph.has_node(key):
            return []

        # Build subgraph of only TEMPORAL_SEQUENCE edges
        temporal_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "TEMPORAL_SEQUENCE":
                temporal_graph.add_edge(u, v)

        if not temporal_graph.has_node(key):
            return []

        # Collect all nodes reachable from or reaching this entity via temporal edges
        reachable_forward = nx.descendants(temporal_graph, key)
        reachable_backward = nx.ancestors(temporal_graph, key)
        connected = reachable_forward | reachable_backward | {key}

        # Extract the subgraph and topologically sort it
        sub = temporal_graph.subgraph(connected)
        try:
            return list(nx.topological_sort(sub))
        except nx.NetworkXUnfeasible:
            # Cycle in temporal edges -- fall back to BFS order from entity
            return list(nx.bfs_tree(sub, key))

    def get_report_timeline(self, report_id: str) -> list[str]:
        """Return entities from a single report ordered by their temporal relationships."""
        # Collect nodes belonging to this report
        report_nodes = set()
        for n, data in self.graph.nodes(data=True):
            if report_id in data.get("report_ids", []):
                report_nodes.add(n)

        if not report_nodes:
            return []

        # Build temporal subgraph restricted to this report's nodes
        temporal_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if (
                data.get("type") == "TEMPORAL_SEQUENCE"
                and u in report_nodes
                and v in report_nodes
            ):
                temporal_graph.add_edge(u, v)

        # Add report nodes that have no temporal edges so they appear at the end
        for n in report_nodes:
            if not temporal_graph.has_node(n):
                temporal_graph.add_node(n)

        try:
            return list(nx.topological_sort(temporal_graph))
        except nx.NetworkXUnfeasible:
            return list(report_nodes)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def get_edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            t = data.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved: {self.node_count()} nodes, {self.edge_count()} edges")


class Neo4jBackend(GraphBackend):
    """Neo4j-based graph backend."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        from neo4j import GraphDatabase

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()

    def add_node(self, node: GraphNode) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MERGE (n:Entity {canonical_name: $canonical_name})
                SET n.name = $name, n.type = $type, n.embedding_id = $embedding_id
                WITH n
                UNWIND $report_ids AS rid
                SET n.report_ids = coalesce(n.report_ids, []) + rid
                """,
                canonical_name=node.canonical_name,
                name=node.name,
                type=node.type,
                embedding_id=node.embedding_id,
                report_ids=node.report_ids,
            )

    def add_edge(self, edge: GraphEdge) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Entity {canonical_name: $source})
                MATCH (b:Entity {canonical_name: $target})
                MERGE (a)-[r:RELATES {type: $type}]->(b)
                SET r.weight = coalesce(r.weight, 0) + 1
                WITH r
                UNWIND $report_ids AS rid
                SET r.report_ids = coalesce(r.report_ids, []) + rid
                """,
                source=edge.source, target=edge.target,
                type=edge.type, report_ids=edge.report_ids,
            )

    def get_node(self, canonical_name: str) -> Optional[GraphNode]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity {canonical_name: $name}) RETURN n",
                name=canonical_name.lower().strip(),
            )
            record = result.single()
            if not record:
                return None
            n = record["n"]
            return GraphNode(
                name=n.get("name", ""),
                type=n.get("type", ""),
                canonical_name=n.get("canonical_name", ""),
                report_ids=n.get("report_ids", []),
                embedding_id=n.get("embedding_id"),
            )

    def get_neighbors(
        self, entity: str, edge_types: Optional[list[str]] = None, depth: int = 2,
        max_degree: int = 200,
    ) -> SubgraphResult:
        key = entity.lower().strip()
        edge_filter = ""
        params: dict = {"name": key, "depth": depth, "max_degree": max_degree}
        if edge_types:
            edge_filter = "WHERE ALL(rel IN relationships(path) WHERE rel.type IN $edge_types)"
            params["edge_types"] = edge_types

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Entity {{canonical_name: $name}})-[r:RELATES*1..{depth}]-(end:Entity)
                {edge_filter}
                WITH path, nodes(path) AS ns
                WHERE ALL(n IN ns WHERE n = start OR size([(n)-[]-() | 1]) <= $max_degree)
                UNWIND ns AS n
                UNWIND relationships(path) AS rel
                RETURN DISTINCT n, rel
                """,
                **params,
            )
            nodes_dict = {}
            edges_list = []
            for record in result:
                n = record["n"]
                cn = n.get("canonical_name", "")
                if cn not in nodes_dict:
                    nodes_dict[cn] = GraphNode(
                        name=n.get("name", ""),
                        type=n.get("type", ""),
                        canonical_name=cn,
                        report_ids=n.get("report_ids", []),
                    )
                rel = record["rel"]
                edges_list.append(GraphEdge(
                    source=rel.start_node.get("canonical_name", ""),
                    target=rel.end_node.get("canonical_name", ""),
                    type=rel.get("type", ""),
                    weight=rel.get("weight", 1),
                    report_ids=rel.get("report_ids", []),
                ))
            return SubgraphResult(
                nodes=list(nodes_dict.values()),
                edges=edges_list,
            )

    def get_causal_chain(
        self, start_entity: str, end_entity: str
    ) -> list[list[str]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (a:Entity {canonical_name: $start})-[r:RELATES*..5]-(b:Entity {canonical_name: $end})
                )
                WHERE ALL(rel IN relationships(path) WHERE rel.type IN ['CAUSED_BY', 'CONTRIBUTED_TO', 'PRECEDED_BY'])
                RETURN [n IN nodes(path) | n.canonical_name] AS chain
                LIMIT 10
                """,
                start=start_entity.lower().strip(),
                end=end_entity.lower().strip(),
            )
            return [record["chain"] for record in result]

    def get_high_centrality_nodes(self, top_n: int = 20) -> list[tuple[str, float]]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Entity)
                WITH n, size([(n)-[]-() | 1]) AS degree
                RETURN n.canonical_name AS name, toFloat(degree) AS centrality
                ORDER BY centrality DESC
                LIMIT $top_n
                """,
                top_n=top_n,
            )
            return [(r["name"], r["centrality"]) for r in result]

    def get_subgraph(self, report_id: str) -> SubgraphResult:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Entity) WHERE $report_id IN n.report_ids
                OPTIONAL MATCH (n)-[r:RELATES]-(m:Entity)
                WHERE $report_id IN m.report_ids
                RETURN DISTINCT n, r, m
                """,
                report_id=report_id,
            )
            nodes_dict = {}
            edges_list = []
            for record in result:
                for node_key in ["n", "m"]:
                    n = record[node_key]
                    if n:
                        cn = n.get("canonical_name", "")
                        if cn and cn not in nodes_dict:
                            nodes_dict[cn] = GraphNode(
                                name=n.get("name", ""),
                                type=n.get("type", ""),
                                canonical_name=cn,
                                report_ids=n.get("report_ids", []),
                            )
                rel = record["r"]
                if rel:
                    edges_list.append(GraphEdge(
                        source=rel.start_node.get("canonical_name", ""),
                        target=rel.end_node.get("canonical_name", ""),
                        type=rel.get("type", ""),
                        weight=rel.get("weight", 1),
                    ))
            return SubgraphResult(
                nodes=list(nodes_dict.values()),
                edges=edges_list,
            )

    def get_temporal_chain(self, entity: str) -> list[str]:
        """Find all nodes connected via TEMPORAL_SEQUENCE edges and return in topological order."""
        key = entity.lower().strip()
        with self.driver.session() as session:
            # Fetch all temporal edges connected to this entity
            result = session.run(
                """
                MATCH (start:Entity {canonical_name: $name})
                OPTIONAL MATCH path_fwd = (start)-[r1:RELATES*1..10]->(fwd:Entity)
                WHERE ALL(rel IN relationships(path_fwd) WHERE rel.type = 'TEMPORAL_SEQUENCE')
                OPTIONAL MATCH path_bwd = (bwd:Entity)-[r2:RELATES*1..10]->(start)
                WHERE ALL(rel IN relationships(path_bwd) WHERE rel.type = 'TEMPORAL_SEQUENCE')
                WITH collect(DISTINCT fwd) + collect(DISTINCT bwd) + collect(DISTINCT start) AS all_nodes
                UNWIND all_nodes AS n
                WITH DISTINCT n
                WHERE n IS NOT NULL
                OPTIONAL MATCH (n)-[r:RELATES]->(m)
                WHERE r.type = 'TEMPORAL_SEQUENCE' AND m IN all_nodes
                RETURN n.canonical_name AS src, m.canonical_name AS tgt
                """,
                name=key,
            )
            # Build adjacency and topological sort via Kahn's algorithm
            all_names: set[str] = set()
            adj: dict[str, list[str]] = {}
            in_degree: dict[str, int] = {}
            for record in result:
                src = record["src"]
                tgt = record["tgt"]
                if src:
                    all_names.add(src)
                    adj.setdefault(src, [])
                    in_degree.setdefault(src, 0)
                if src and tgt:
                    all_names.add(tgt)
                    adj.setdefault(tgt, [])
                    adj[src].append(tgt)
                    in_degree[tgt] = in_degree.get(tgt, 0) + 1
                    in_degree.setdefault(src, 0)

            queue = [n for n in all_names if in_degree.get(n, 0) == 0]
            ordered: list[str] = []
            while queue:
                node = queue.pop(0)
                ordered.append(node)
                for neighbor in adj.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            remaining = [n for n in all_names if n not in set(ordered)]
            return ordered + remaining

    def get_report_timeline(self, report_id: str) -> list[str]:
        """Return entities from a single report ordered by temporal relationships."""
        with self.driver.session() as session:
            # Get temporally ordered entities first
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                WHERE r.type = 'TEMPORAL_SEQUENCE'
                  AND $report_id IN a.report_ids
                  AND $report_id IN b.report_ids
                WITH collect({src: a.canonical_name, tgt: b.canonical_name}) AS edges
                MATCH (n:Entity) WHERE $report_id IN n.report_ids
                WITH n.canonical_name AS name, edges
                RETURN DISTINCT name, edges
                """,
                report_id=report_id,
            )
            records = list(result)
            if not records:
                return []

            # Extract edges and all node names
            edges = records[0]["edges"] if records else []
            all_names = [r["name"] for r in records]

            # Build adjacency and topological sort via Kahn's algorithm
            in_degree: dict[str, int] = {n: 0 for n in all_names}
            adj: dict[str, list[str]] = {n: [] for n in all_names}
            for e in edges:
                src, tgt = e["src"], e["tgt"]
                if src in adj and tgt in in_degree:
                    adj[src].append(tgt)
                    in_degree[tgt] += 1

            queue = [n for n in all_names if in_degree[n] == 0]
            ordered = []
            while queue:
                node = queue.pop(0)
                ordered.append(node)
                for neighbor in adj.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # Append any remaining nodes not reached (cycle or disconnected)
            remaining = [n for n in all_names if n not in set(ordered)]
            return ordered + remaining

    def node_count(self) -> int:
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS c")
            return result.single()["c"]

    def edge_count(self) -> int:
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS c")
            return result.single()["c"]

    def get_type_counts(self) -> dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) RETURN n.type AS type, count(n) AS c"
            )
            return {r["type"]: r["c"] for r in result}

    def get_edge_type_counts(self) -> dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH ()-[r:RELATES]->() RETURN r.type AS type, count(r) AS c"
            )
            return {r["type"]: r["c"] for r in result}

    def save(self) -> None:
        pass  # Neo4j persists automatically


def detect_backend() -> GraphBackend:
    """Auto-detect available graph backend."""
    # Try Neo4j first
    try:
        backend = Neo4jBackend()
        print(f"Connected to Neo4j at {backend.uri}")
        return backend
    except Exception:
        pass

    # Fall back to NetworkX
    print("Neo4j unavailable, using NetworkX backend")
    return NetworkXBackend()


def load_graph(path: Optional[Path] = None) -> nx.DiGraph:
    """Load the raw NetworkX graph from pickle, trying cleaned/normalized/original in order."""
    if path and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    for name in ["aerograph_communities.pkl", "aerograph_clean.pkl",
                  "aerograph_normalized.pkl", "aerograph.pkl"]:
        p = GRAPHS_DIR / name
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("No graph pickle found. Run `aerograph build` first.")


def build_graph(
    extractions_path: Optional[Path] = None,
    backend: Optional[GraphBackend] = None,
) -> GraphBackend:
    """Build knowledge graph from extraction results."""
    if extractions_path is None:
        extractions_path = PROCESSED_DIR / "extractions.jsonl"
    if backend is None:
        backend = detect_backend()

    from aerograph.extract import load_extractions, normalize_entities, normalize_relations
    extractions = load_extractions(extractions_path)
    print(f"Building graph from {len(extractions)} extraction results")

    # Collect all entities and relations for cross-report normalization
    all_entities = []
    all_relations = []
    for result in extractions:
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)

    # Normalize: deduplicate entities and remap relation endpoints
    merged_entities = normalize_entities(all_entities)
    entity_map = {}
    for e in all_entities:
        entity_map[e.canonical_name] = e.canonical_name
    for e in merged_entities:
        entity_map[e.canonical_name] = e.canonical_name
    merged_relations = normalize_relations(all_relations, entity_map)

    print(f"  Normalized: {len(all_entities)} -> {len(merged_entities)} entities, "
          f"{len(all_relations)} -> {len(merged_relations)} relations")

    total_entities = 0
    total_relations = 0

    for entity in merged_entities:
        backend.add_node(GraphNode(
            name=entity.name,
            type=entity.type,
            canonical_name=entity.canonical_name,
            report_ids=entity.report_ids,
        ))
        total_entities += 1

    for relation in merged_relations:
        backend.add_edge(GraphEdge(
            source=relation.source,
            target=relation.target,
            type=relation.type,
            report_ids=relation.report_ids,
        ))
        total_relations += 1

    backend.save()
    print(f"Graph built: {backend.node_count()} nodes, {backend.edge_count()} edges")
    print(f"  (from {total_entities} entity mentions, {total_relations} relation mentions)")
    return backend


def build_normalized_graph(normalized_path: Path) -> GraphBackend:
    """Rebuild the graph from taxonomy-normalized extractions."""
    return build_graph(extractions_path=normalized_path)


