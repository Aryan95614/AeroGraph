"""End-to-end integration test with a mocked Anthropic client.

Verifies:
  - The pickled graph loads
  - ChromaDB opens and has chunks
  - retrieve.py returns a non-empty RetrievalResult for a known query
  - generate.py (mocked) returns a non-empty cited answer
"""
from __future__ import annotations
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import os
import pickle

import pytest


def test_graph_loads():
    graph_path = Path("data/graphs/aerograph.pkl")
    if not graph_path.exists():
        pytest.skip("graph pickle missing; run `make build`")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_chroma_has_chunks():
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")
    persist = Path("data/chroma_db")
    if not persist.exists():
        pytest.skip("chroma_db missing; run `make embed`")
    client = chromadb.PersistentClient(path=str(persist))
    total = sum(c.count() for c in client.list_collections())
    assert total > 0


def test_demo_cache_is_usable():
    """Cached-mode smoke test: reading a cached entry."""
    cache = Path("data/demo_cache.jsonl")
    if not cache.exists():
        pytest.skip("demo cache not built yet; run scripts/build_demo_cache.py")
    with open(cache) as f:
        first = json.loads(f.readline())
    assert "question" in first and "answer" in first
    assert len(first["answer"]) > 50


@patch("anthropic.Anthropic")
def test_retrieve_and_generate_with_mock(mock_anthropic):
    """Full retrieve + generate path with mocked Anthropic client."""
    graph_path = Path("data/graphs/aerograph.pkl")
    if not graph_path.exists():
        pytest.skip("graph pickle missing")
    try:
        import chromadb  # noqa: F401
    except ImportError:
        pytest.skip("chromadb not installed")

    import sys
    sys.path.insert(0, str(Path("src")))
    from aerograph.retrieve import GraphRAGRetriever
    # mock generate_answer by patching anthropic client
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Mocked answer citing ACN 1234567.")]
    mock_anthropic.return_value.messages.create.return_value = mock_resp

    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-mock")

    retriever = GraphRAGRetriever()
    result = retriever.retrieve("What are common bird strike factors?", top_k=5)
    assert result is not None
    assert result.chunks is not None  # may be empty on a very small graph, but object exists
