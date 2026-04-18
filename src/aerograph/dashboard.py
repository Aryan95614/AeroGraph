"""AeroGraph Interactive Dashboard.

Streamlit-based explorer for the AeroGraph knowledge graph, providing:
  - Natural language query interface with GraphRAG retrieval
  - Knowledge graph neighborhood visualization
  - Graph statistics overview
  - Causal chain tracing between entities
"""

from __future__ import annotations

import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PAGE_TITLE = "AeroGraph"
PAGE_ICON = "\u2708"  # airplane
LAYOUT = "wide"

NODE_TYPE_COLORS = {
    "Aircraft":       "#2563EB",
    "Event":          "#DC2626",
    "Phase":          "#059669",
    "Factor":         "#D97706",
    "Component":      "#7C3AED",
    "Outcome":        "#DB2777",
    "Recommendation": "#0891B2",
    "ATC_Facility":   "#4F46E5",
    "Weather":        "#0284C7",
    "TimePeriod":     "#F59E0B",
    "unknown":        "#6B7280",
}

EDGE_TYPE_LABELS = {
    "CAUSED_BY":        "caused by",
    "CONTRIBUTED_TO":   "contributed to",
    "OCCURRED_DURING":  "occurred during",
    "INVOLVED":         "involved",
    "RESOLVED_BY":      "resolved by",
    "PRECEDED_BY":      "preceded by",
    "CO_OCCURRED_WITH": "co-occurred with",
    "TEMPORAL_SEQUENCE": "temporal sequence",
}

# ---------------------------------------------------------------------------
# Graph backend (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_graph_backend():
    """Load and cache the graph backend for the session."""
    from aerograph.graph import detect_backend
    return detect_backend()


def get_backend():
    """Return the cached graph backend, handling load errors gracefully."""
    try:
        return load_graph_backend()
    except Exception as exc:
        st.error(f"Could not load graph backend: {exc}")
        return None

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Top-level title area */
    .block-container { padding-top: 2rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #475569;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
    }

    /* Sidebar section headers */
    .sidebar .stRadio > label { font-weight: 600; }

    /* Source cards */
    .source-card {
        background: #f1f5f9;
        border-left: 3px solid #2563EB;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .source-card .acn { font-weight: 700; color: #1e40af; }
    .source-card .prov {
        display: inline-block;
        font-size: 0.75rem;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .prov-vector { background: #dbeafe; color: #1e40af; }
    .prov-graph  { background: #d1fae5; color: #065f46; }
    .prov-both   { background: #ede9fe; color: #5b21b6; }

    /* Causal chain path */
    .chain-step {
        display: inline-block;
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 2px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .chain-arrow {
        display: inline-block;
        color: #6b7280;
        margin: 0 2px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("## AeroGraph")
st.sidebar.caption("GraphRAG over Aviation Safety Reports")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Query", "Graph Explorer", "Statistics", "Causal Chains"],
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_color(node_type: str) -> str:
    return NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["unknown"])


def _render_subgraph(subgraph, center_entity: Optional[str] = None, title: str = ""):
    """Render a SubgraphResult as a matplotlib figure embedded in Streamlit."""
    if not subgraph.nodes:
        st.info("No nodes to display.")
        return

    G = nx.DiGraph()
    for node in subgraph.nodes:
        G.add_node(node.canonical_name, type=node.type, label=node.name)
    for edge in subgraph.edges:
        if G.has_node(edge.source) and G.has_node(edge.target):
            G.add_edge(edge.source, edge.target, type=edge.type, weight=edge.weight)

    if G.number_of_nodes() == 0:
        st.info("No nodes to display.")
        return

    # Layout
    if G.number_of_nodes() <= 3:
        pos = nx.spring_layout(G, k=3.0, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    # Sizing
    node_count = G.number_of_nodes()
    fig_w = max(8, min(14, 6 + node_count * 0.4))
    fig_h = max(6, min(10, 4 + node_count * 0.3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # Node colors and sizes
    node_list = list(G.nodes())
    colors = [_node_color(G.nodes[n].get("type", "unknown")) for n in node_list]
    sizes = []
    for n in node_list:
        if center_entity and n == center_entity.lower().strip():
            sizes.append(1800)
        else:
            sizes.append(900)

    # Draw edges
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        etype = data.get("type", "")
        label = EDGE_TYPE_LABELS.get(etype, etype.lower().replace("_", " "))
        edge_labels[(u, v)] = label

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#94a3b8",
        width=1.2,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        alpha=0.7,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=node_list,
        node_color=colors,
        node_size=sizes,
        alpha=0.92,
        edgecolors="white",
        linewidths=1.5,
    )

    # Node labels (use short display names)
    label_map = {}
    for n in node_list:
        raw = G.nodes[n].get("label", n)
        # Truncate long labels
        label_map[n] = raw[:20] + "..." if len(raw) > 20 else raw

    nx.draw_networkx_labels(
        G, pos, labels=label_map, ax=ax,
        font_size=8, font_weight="bold", font_color="white",
    )

    # Edge labels
    if G.number_of_edges() <= 30:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=6, font_color="#64748b",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            rotate=False,
        )

    # Legend
    seen_types = sorted({G.nodes[n].get("type", "unknown") for n in node_list})
    legend_patches = [
        mpatches.Patch(color=_node_color(t), label=t)
        for t in seen_types
    ]
    ax.legend(
        handles=legend_patches, loc="upper left", framealpha=0.9,
        fontsize=8, title="Node types", title_fontsize=9,
        fancybox=True, edgecolor="#e2e8f0",
    )

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a", pad=12)

    ax.set_axis_off()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _provenance_class(prov: str) -> str:
    return f"prov-{prov}"


# ---------------------------------------------------------------------------
# PAGE: Query
# ---------------------------------------------------------------------------

def page_query():
    st.markdown("# Ask AeroGraph")
    st.markdown(
        "Enter a natural language question about aviation safety incidents. "
        "AeroGraph retrieves evidence from the knowledge graph and ASRS reports, "
        "then generates a grounded answer with source citations."
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the most common contributing factors to runway incursions involving B737 aircraft?",
        label_visibility="collapsed",
    )

    col_k, col_run = st.columns([1, 4])
    with col_k:
        top_k = st.slider("Sources", min_value=3, max_value=20, value=8, help="Number of evidence chunks to retrieve")
    with col_run:
        run_pressed = st.button("Run Query", type="primary", use_container_width=True)

    if run_pressed and query.strip():
        with st.status("Processing query...", expanded=True) as status:
            st.write("Retrieving evidence (vector + graph)...")
            try:
                from aerograph.retrieve import GraphRAGRetriever
                retriever = GraphRAGRetriever()
                retrieval_result = retriever.retrieve(query, top_k=top_k)
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                return

            st.write("Generating answer...")
            try:
                from aerograph.generate import generate_answer
                gen_result = generate_answer(query, retrieval_result)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

            status.update(label="Complete", state="complete", expanded=False)

        # -- Answer ----------------------------------------------------------
        st.markdown("---")
        st.markdown("### Answer")
        st.markdown(gen_result.text)

        # -- Metadata bar ----------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval", f"{retrieval_result.latency_ms:.0f} ms")
        m2.metric("Generation", f"{gen_result.latency_ms:.0f} ms")
        m3.metric("Sources", len(gen_result.source_acns))
        m4.metric("Model", gen_result.model)

        # -- Sources ---------------------------------------------------------
        if retrieval_result.chunks:
            st.markdown("### Retrieved Evidence")
            for i, chunk in enumerate(retrieval_result.chunks, 1):
                prov_cls = _provenance_class(chunk.provenance)
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="acn">ACN {chunk.report_id}</span> '
                    f'<span class="prov {prov_cls}">{chunk.provenance}</span> '
                    f'&middot; score {chunk.score:.4f}'
                    f'<br/>{chunk.text[:300]}{"..." if len(chunk.text) > 300 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # -- Retrieval trace -------------------------------------------------
        with st.expander("Retrieval trace"):
            st.json(retrieval_result.retrieval_trace)

    elif run_pressed:
        st.warning("Please enter a question.")


# ---------------------------------------------------------------------------
# PAGE: Graph Explorer
# ---------------------------------------------------------------------------

def page_graph_explorer():
    st.markdown("# Knowledge Graph Explorer")
    st.markdown(
        "Search for an entity in the aviation safety knowledge graph and "
        "visualize its neighborhood. Nodes are colored by type."
    )

    backend = get_backend()
    if backend is None:
        return

    col_search, col_depth = st.columns([3, 1])
    with col_search:
        entity_query = st.text_input(
            "Entity name",
            placeholder="e.g. bird strike, b737, engine failure",
            label_visibility="collapsed",
        )
    with col_depth:
        depth = st.selectbox("Hops", [1, 2, 3], index=1, help="Neighborhood depth")

    if entity_query.strip():
        canonical = entity_query.lower().strip()
        node = backend.get_node(canonical)

        if node is None:
            # Attempt fuzzy match across graph nodes
            st.warning(f'No exact match for "{entity_query}". Searching for close matches...')
            _show_fuzzy_suggestions(backend, canonical)
            return

        # Node detail card
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Entity", node.name)
        c2.metric("Type", node.type)
        c3.metric("Reports", len(node.report_ids))

        # Neighborhood
        subgraph = backend.get_neighbors(canonical, depth=depth)
        st.markdown(f"**Neighborhood:** {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")

        _render_subgraph(
            subgraph,
            center_entity=canonical,
            title=f"{node.name} -- {depth}-hop neighborhood",
        )

        # Neighbor table
        if subgraph.nodes:
            with st.expander(f"Neighbor details ({len(subgraph.nodes)} nodes)"):
                rows = [
                    {
                        "Name": n.name,
                        "Type": n.type,
                        "Reports": len(n.report_ids),
                    }
                    for n in sorted(subgraph.nodes, key=lambda n: -len(n.report_ids))
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # Edge table
        if subgraph.edges:
            with st.expander(f"Edge details ({len(subgraph.edges)} edges)"):
                edge_rows = [
                    {
                        "Source": e.source,
                        "Relation": e.type,
                        "Target": e.target,
                        "Weight": e.weight,
                    }
                    for e in sorted(subgraph.edges, key=lambda e: -e.weight)
                ]
                st.dataframe(edge_rows, use_container_width=True, hide_index=True)


def _show_fuzzy_suggestions(backend, query: str):
    """Show fuzzy-matched entity suggestions from the graph."""
    from difflib import SequenceMatcher

    try:
        # For NetworkX backend, iterate nodes directly
        if hasattr(backend, "graph"):
            candidates = list(backend.graph.nodes())
        else:
            # Fallback: use high centrality nodes as candidates
            candidates = [name for name, _ in backend.get_high_centrality_nodes(top_n=200)]
    except Exception:
        st.info("Could not search for similar entities.")
        return

    scored = []
    for cand in candidates:
        ratio = SequenceMatcher(None, query, cand).ratio()
        if ratio > 0.4 or query in cand:
            scored.append((cand, ratio))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:10]

    if top:
        st.markdown("**Did you mean:**")
        for name, score in top:
            node = backend.get_node(name)
            ntype = node.type if node else "unknown"
            st.markdown(f"- `{name}` ({ntype})")
    else:
        st.info("No similar entities found. Try a different search term.")


# ---------------------------------------------------------------------------
# PAGE: Statistics
# ---------------------------------------------------------------------------

def page_statistics():
    st.markdown("# Graph Statistics")
    st.markdown("Overview of the AeroGraph knowledge graph structure and contents.")

    backend = get_backend()
    if backend is None:
        return

    # Top-level metrics
    total_nodes = backend.node_count()
    total_edges = backend.edge_count()

    # Count reports
    report_count = 0
    try:
        if hasattr(backend, "graph"):
            all_rids: set[str] = set()
            for _, data in backend.graph.nodes(data=True):
                all_rids.update(data.get("report_ids", []))
            report_count = len(all_rids)
    except Exception:
        pass

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Nodes", f"{total_nodes:,}")
    m2.metric("Total Edges", f"{total_edges:,}")
    m3.metric("Total Reports", f"{report_count:,}")

    st.markdown("---")

    # -- Node type distribution -----------------------------------------------
    col_node, col_edge = st.columns(2)

    type_counts = backend.get_type_counts()
    if type_counts:
        with col_node:
            st.markdown("### Nodes by Type")
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [t for t, _ in sorted_types]
            values = [c for _, c in sorted_types]
            bar_colors = [_node_color(t) for t in labels]

            fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.45)), facecolor="white")
            y_pos = range(len(labels))
            bars = ax.barh(y_pos, values, color=bar_colors, height=0.65, edgecolor="white", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=10, fontweight="500")
            ax.invert_yaxis()
            ax.set_xlabel("Count", fontsize=10, color="#475569")
            ax.tick_params(axis="x", colors="#64748b", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#e2e8f0")
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#f1f5f9", linewidth=0.8)

            # Value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color="#334155", fontweight="600",
                )

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # -- Edge type distribution -----------------------------------------------
    edge_type_counts = backend.get_edge_type_counts()
    if edge_type_counts:
        with col_edge:
            st.markdown("### Edges by Type")
            sorted_edges = sorted(edge_type_counts.items(), key=lambda x: x[1], reverse=True)
            e_labels = [EDGE_TYPE_LABELS.get(t, t.lower().replace("_", " ")) for t, _ in sorted_edges]
            e_values = [c for _, c in sorted_edges]

            palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
            e_colors = [palette[i % len(palette)] for i in range(len(e_labels))]

            fig2, ax2 = plt.subplots(figsize=(6, max(3, len(e_labels) * 0.45)), facecolor="white")
            y_pos2 = range(len(e_labels))
            bars2 = ax2.barh(y_pos2, e_values, color=e_colors, height=0.65, edgecolor="white", linewidth=0.5)
            ax2.set_yticks(y_pos2)
            ax2.set_yticklabels(e_labels, fontsize=10, fontweight="500")
            ax2.invert_yaxis()
            ax2.set_xlabel("Count", fontsize=10, color="#475569")
            ax2.tick_params(axis="x", colors="#64748b", labelsize=9)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["bottom"].set_color("#e2e8f0")
            ax2.set_axisbelow(True)
            ax2.grid(axis="x", color="#f1f5f9", linewidth=0.8)

            for bar, val in zip(bars2, e_values):
                ax2.text(
                    bar.get_width() + max(e_values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color="#334155", fontweight="600",
                )

            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # -- High-centrality nodes ------------------------------------------------
    st.markdown("---")
    st.markdown("### Highest Centrality Entities")
    st.caption("Ranked by PageRank score -- the most influential nodes in the knowledge graph.")

    centrality = backend.get_high_centrality_nodes(top_n=15)
    if centrality:
        rows = []
        for rank, (name, score) in enumerate(centrality, 1):
            node = backend.get_node(name)
            ntype = node.type if node else "unknown"
            reports = len(node.report_ids) if node else 0
            rows.append({
                "Rank": rank,
                "Entity": name,
                "Type": ntype,
                "PageRank": f"{score:.6f}",
                "Reports": reports,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No centrality data available. Build the graph first with `make build`.")


# ---------------------------------------------------------------------------
# PAGE: Causal Chains
# ---------------------------------------------------------------------------

def page_causal_chains():
    st.markdown("# Causal Chain Tracer")
    st.markdown(
        "Trace causal paths between two entities in the knowledge graph. "
        "Paths follow CAUSED_BY, CONTRIBUTED_TO, and PRECEDED_BY edges."
    )

    backend = get_backend()
    if backend is None:
        return

    col_start, col_end = st.columns(2)
    with col_start:
        start_entity = st.text_input(
            "Start entity",
            placeholder="e.g. wind shear",
            key="chain_start",
        )
    with col_end:
        end_entity = st.text_input(
            "End entity",
            placeholder="e.g. go-around",
            key="chain_end",
        )

    trace_pressed = st.button("Trace Causal Paths", type="primary")

    if trace_pressed and start_entity.strip() and end_entity.strip():
        start_canonical = start_entity.lower().strip()
        end_canonical = end_entity.lower().strip()

        # Validate both entities exist
        start_node = backend.get_node(start_canonical)
        end_node = backend.get_node(end_canonical)

        if start_node is None:
            st.error(f'Entity "{start_entity}" not found in the graph.')
            _show_fuzzy_suggestions(backend, start_canonical)
            return
        if end_node is None:
            st.error(f'Entity "{end_entity}" not found in the graph.')
            _show_fuzzy_suggestions(backend, end_canonical)
            return

        with st.spinner("Tracing causal chains..."):
            chains = backend.get_causal_chain(start_canonical, end_canonical)

        st.markdown("---")

        if not chains:
            st.warning(
                f"No causal path found between **{start_node.name}** and **{end_node.name}**. "
                "The entities may not be connected via CAUSED_BY, CONTRIBUTED_TO, or PRECEDED_BY edges."
            )
            # Still show both neighborhoods for context
            st.markdown("### Entity Neighborhoods")
            tab_s, tab_e = st.tabs([start_node.name, end_node.name])
            with tab_s:
                sub = backend.get_neighbors(start_canonical, depth=1)
                _render_subgraph(sub, center_entity=start_canonical, title=f"{start_node.name} -- 1-hop")
            with tab_e:
                sub = backend.get_neighbors(end_canonical, depth=1)
                _render_subgraph(sub, center_entity=end_canonical, title=f"{end_node.name} -- 1-hop")
            return

        st.success(f"Found {len(chains)} causal path{'s' if len(chains) != 1 else ''}")

        for i, chain in enumerate(chains, 1):
            st.markdown(f"**Path {i}** ({len(chain)} steps)")
            # Render chain as styled inline blocks
            parts = []
            for j, step in enumerate(chain):
                parts.append(f'<span class="chain-step">{step}</span>')
                if j < len(chain) - 1:
                    parts.append('<span class="chain-arrow">\u2192</span>')
            st.markdown(" ".join(parts), unsafe_allow_html=True)

        # Visualize the union of all chain paths
        st.markdown("### Path Visualization")

        # Build a subgraph from all chain nodes
        from aerograph.graph import SubgraphResult, GraphNode, GraphEdge

        all_chain_nodes: dict[str, GraphNode] = {}
        all_chain_edges: list[GraphEdge] = []
        for chain in chains:
            for name in chain:
                if name not in all_chain_nodes:
                    node = backend.get_node(name)
                    if node:
                        all_chain_nodes[name] = node
            # Build edges between consecutive chain nodes
            for a, b in zip(chain, chain[1:]):
                all_chain_edges.append(GraphEdge(
                    source=a, target=b,
                    type="CAUSAL_PATH", weight=1,
                ))

        chain_subgraph = SubgraphResult(
            nodes=list(all_chain_nodes.values()),
            edges=all_chain_edges,
        )
        _render_subgraph(
            chain_subgraph,
            center_entity=start_canonical,
            title=f"Causal paths: {start_node.name} \u2192 {end_node.name}",
        )

    elif trace_pressed:
        st.warning("Please provide both a start and end entity.")


# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------

PAGE_MAP = {
    "Query":         page_query,
    "Graph Explorer": page_graph_explorer,
    "Statistics":    page_statistics,
    "Causal Chains": page_causal_chains,
}

PAGE_MAP[page]()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "AeroGraph v0.1.0  \n"
    "GraphRAG over NASA ASRS incident reports"
)
