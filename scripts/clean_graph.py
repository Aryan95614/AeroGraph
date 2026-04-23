"""Graph surgery on existing data. Does NOT re-run extraction or embeddings.

Steps:
  1. Dedupe data/processed/extractions.jsonl in place (line-level).
  2. Rebuild data/graphs/aerograph.pkl from the deduped extractions.
  3. Remove placeholder nodes, isolated nodes, self-loops, and null-type nodes.
  4. Save the cleaned graph back to data/graphs/aerograph.pkl.

Prints before/after counts. Exits non-zero if acceptance invariants fail.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import networkx as nx

REPO = Path(__file__).parent.parent
EXTRACTIONS = REPO / "data/processed/extractions.jsonl"
GRAPH_PATH = REPO / "data/graphs/aerograph.pkl"

PLACEHOLDER_EXACT = {"", "unknown", "x", "y", "z"}
PLACEHOLDER_RE = re.compile(r"^aircraft\s*x$", re.I)


def is_placeholder_name(name: str, node_type: str | None) -> bool:
    if name is None:
        return True
    n = str(name).strip()
    if n == "":
        return True
    nl = n.lower()
    if nl in PLACEHOLDER_EXACT:
        return True
    if PLACEHOLDER_RE.match(nl):
        return True
    # Single-letter aircraft types: "aircraft a", "aircraft b", ...
    if re.match(r"^aircraft\s+[a-z]$", nl):
        return True
    # Type literal "unknown" means the node's type field is the string "unknown"
    if (node_type or "").strip().lower() == "unknown":
        return True
    return False


def dedupe_extractions(path: Path) -> tuple[int, int]:
    """Line-level dedup of the JSONL file, preserving order. Returns (before, after)."""
    if not path.exists():
        return (0, 0)
    lines = path.read_text().splitlines()
    before = len(lines)
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        key = ln.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ln)
    path.write_text("\n".join(out) + "\n")
    return (before, len(out))


def rebuild_graph_from_extractions() -> nx.DiGraph:
    """Use the existing build pipeline to produce a fresh graph."""
    sys.path.insert(0, str(REPO / "src"))
    from aerograph.graph import build_graph, load_graph

    # build_graph writes through to NetworkXBackend.path which is aerograph.pkl
    backend = build_graph(extractions_path=EXTRACTIONS)
    # Reload the pickle as a nx.DiGraph for surgery
    return load_graph(GRAPH_PATH)


def surgical_cleanup(G: nx.DiGraph) -> dict:
    """Apply placeholder / isolated / self-loop / null-type filters."""
    stats = {
        "removed_placeholders": 0,
        "removed_null_type": 0,
        "removed_self_loops": 0,
        "removed_isolated": 0,
    }
    # 1. Placeholders
    to_drop = []
    for n in list(G.nodes()):
        node_type = G.nodes[n].get("type")
        if is_placeholder_name(n, node_type):
            to_drop.append(n)
    G.remove_nodes_from(to_drop)
    stats["removed_placeholders"] = len(to_drop)

    # 2. Null type
    to_drop = [n for n in list(G.nodes()) if not G.nodes[n].get("type")]
    G.remove_nodes_from(to_drop)
    stats["removed_null_type"] = len(to_drop)

    # 3. Self-loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    stats["removed_self_loops"] = len(self_loops)

    # 4. Isolated (must be last — edge removals may create new isolates)
    isolated = [n for n in list(G.nodes()) if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    stats["removed_isolated"] = len(isolated)

    return stats


def summarize(G: nx.DiGraph, label: str) -> dict:
    n_placeholders = sum(
        1 for n in G.nodes() if is_placeholder_name(n, G.nodes[n].get("type"))
    )
    n_null_type = sum(1 for n in G.nodes() if not G.nodes[n].get("type"))
    n_self = sum(1 for u, v in G.edges() if u == v)
    n_iso = sum(1 for n in G.nodes() if G.degree(n) == 0)
    n_comp = nx.number_weakly_connected_components(G)
    info = {
        "label": label,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": n_comp,
        "isolated": n_iso,
        "self_loops": n_self,
        "placeholders": n_placeholders,
        "null_type": n_null_type,
    }
    print(f"{label}: nodes={info['nodes']}, edges={info['edges']}, "
          f"components={info['components']}, isolated={info['isolated']}, "
          f"self_loops={info['self_loops']}, placeholders={info['placeholders']}, "
          f"null_type={info['null_type']}")
    return info


def main():
    # --- BEFORE snapshot ---
    with open(GRAPH_PATH, "rb") as f:
        G_before = pickle.load(f)
    summarize(G_before, "BEFORE")

    # --- Dedupe extractions ---
    ext_before, ext_after = dedupe_extractions(EXTRACTIONS)
    print(f"extractions.jsonl: {ext_before} -> {ext_after} lines ({ext_before - ext_after} dups removed)")

    # --- Rebuild graph from deduped extractions ---
    print("Rebuilding graph from deduped extractions...")
    G = rebuild_graph_from_extractions()
    summarize(G, "REBUILT")

    # --- Surgical cleanup ---
    stats = surgical_cleanup(G)
    print(f"Removed: {stats}")

    # --- Save ---
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved cleaned graph to {GRAPH_PATH}")

    # --- AFTER ---
    info = summarize(G, "AFTER")

    # --- Acceptance invariants ---
    fails = []
    if info["isolated"] != 0:
        fails.append(f"isolated count is {info['isolated']}, expected 0")
    if info["self_loops"] != 0:
        fails.append(f"self_loops count is {info['self_loops']}, expected 0")
    if info["placeholders"] != 0:
        fails.append(f"placeholders count is {info['placeholders']}, expected 0")
    if info["null_type"] != 0:
        fails.append(f"null_type count is {info['null_type']}, expected 0")

    # Check extraction dedupe result
    with open(EXTRACTIONS) as f:
        post_lines = f.read().splitlines()
    seen: set[str] = set()
    dups = 0
    for ln in post_lines:
        key = ln.strip()
        if not key:
            continue
        if key in seen:
            dups += 1
        seen.add(key)
    if dups != 0:
        fails.append(f"extractions.jsonl still has {dups} duplicates")

    if fails:
        print("\nACCEPTANCE FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nACCEPTANCE OK")


if __name__ == "__main__":
    main()
