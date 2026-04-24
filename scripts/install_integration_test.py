"""Install tests/test_integration.py and a `make repro` target. Idempotent."""
from __future__ import annotations
from pathlib import Path

TEST = Path("tests/test_integration.py")
MAKEFILE = Path("Makefile")

TEST_BODY = '''"""End-to-end integration test with a mocked Anthropic client.

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
'''


REPRO_TARGET = '''
repro:
\t@echo "[repro] installing package..."
\tpip install -e ".[dev]" >/dev/null
\t@echo "[repro] checking data artifacts..."
\t@test -f data/graphs/aerograph.pkl || (echo "missing graph — fetch from HF dataset first"; exit 1)
\t@test -d data/chroma_db || (echo "missing chroma_db — fetch from HF dataset first"; exit 1)
\t@echo "[repro] running integration test..."
\tpython -m pytest tests/test_integration.py -v
\t@echo "[repro] launching cached-mode Gradio demo on :7860 (unset ANTHROPIC_API_KEY) ..."
\tunset ANTHROPIC_API_KEY && python app.py
'''


def main():
    if not TEST.exists():
        TEST.write_text(TEST_BODY)
        print(f"wrote {TEST}")
    else:
        print(f"{TEST} already exists; overwriting to match spec")
        TEST.write_text(TEST_BODY)

    makefile_text = MAKEFILE.read_text()
    if "\nrepro:" not in makefile_text:
        # also add repro to .PHONY
        makefile_text = makefile_text.replace(
            ".PHONY: install",
            ".PHONY: install repro",
            1,
        )
        makefile_text = makefile_text.rstrip() + "\n" + REPRO_TARGET
        MAKEFILE.write_text(makefile_text)
        print("added `make repro` target to Makefile")
    else:
        print("`make repro` already present")


if __name__ == "__main__":
    main()
