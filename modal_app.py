"""AeroGraph — Modal serverless deployment.

Deploys the FastAPI server + Gradio app on Modal with persistent
storage for the knowledge graph and vector index.

Usage:
    modal deploy modal_app.py        # deploy to cloud
    modal serve modal_app.py         # local dev with hot reload
"""

from __future__ import annotations

import modal

app = modal.App("aerograph")

# ---------------------------------------------------------------------------
# Image: install all dependencies
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "anthropic>=0.39.0",
        "chromadb>=0.4.22",
        "networkx>=3.2",
        "sentence-transformers>=2.3.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.25.0",
        "tenacity>=8.2.0",
        "httpx>=0.26.0",
        "matplotlib>=3.8.0",
        "numpy>=1.26.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "gradio>=4.0.0",
    )
)

# ---------------------------------------------------------------------------
# Volume: persistent data (graph, embeddings, reports)
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("aerograph-data", create_if_missing=True)
DATA_MOUNT = "/data"

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

# Create a Modal secret named "anthropic-key" with ANTHROPIC_API_KEY
# modal secret create anthropic-key ANTHROPIC_API_KEY=sk-ant-...


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("anthropic-key")],
    gpu=None,
    cpu=2,
    memory=4096,
    timeout=120,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI server on Modal."""
    import sys
    import os

    # Set data directory to the Modal volume
    os.environ["AEROGRAPH_DATA_DIR"] = DATA_MOUNT

    # Add src to path
    sys.path.insert(0, "/root/src")

    from aerograph.api import app as api_app
    return api_app


# ---------------------------------------------------------------------------
# Gradio web app
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("anthropic-key")],
    gpu=None,
    cpu=2,
    memory=4096,
    timeout=300,
    allow_concurrent_inputs=5,
)
@modal.web_endpoint(method="GET")
def gradio_app():
    """Serve the Gradio app on Modal."""
    import sys
    import os

    os.environ["AEROGRAPH_DATA_DIR"] = DATA_MOUNT
    sys.path.insert(0, "/root/src")

    from app import build_app
    demo = build_app()
    return demo


# ---------------------------------------------------------------------------
# Data upload helper — run once to seed the volume
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    timeout=600,
)
def upload_data():
    """Upload local data files to the Modal volume.

    Run with: modal run modal_app.py::upload_data
    """
    import shutil
    from pathlib import Path

    vol_path = Path(DATA_MOUNT)
    print(f"Volume contents before upload:")
    for p in vol_path.rglob("*"):
        if p.is_file():
            print(f"  {p} ({p.stat().st_size:,} bytes)")

    # The data should be uploaded via modal volume put
    print("\nTo upload data to the volume:")
    print("  modal volume put aerograph-data data/graphs/aerograph.pkl graphs/aerograph.pkl")
    print("  modal volume put aerograph-data data/chroma_db/ chroma_db/")
    print("  modal volume put aerograph-data data/processed/reports.jsonl processed/reports.jsonl")
    print("  modal volume put aerograph-data data/processed/extractions.jsonl processed/extractions.jsonl")


# ---------------------------------------------------------------------------
# Extraction job — run entity extraction on Modal
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    secrets=[modal.Secret.from_name("anthropic-key")],
    timeout=7200,  # 2 hours
    cpu=4,
    memory=8192,
)
def run_extraction(max_reports: int = 2000, batch_size: int = 20, max_workers: int = 5):
    """Run entity extraction on Modal infrastructure.

    Usage: modal run modal_app.py::run_extraction --max-reports 2000
    """
    import sys
    import os

    os.environ["AEROGRAPH_DATA_DIR"] = DATA_MOUNT
    sys.path.insert(0, "/root/src")

    from pathlib import Path
    from aerograph.extract import run_extraction as _run

    processed = Path(DATA_MOUNT) / "processed"
    results = _run(
        reports_path=processed / "reports.jsonl",
        output_path=processed / "extractions.jsonl",
        batch_size=batch_size,
        max_workers=max_workers,
        max_reports=max_reports,
    )
    volume.commit()
    print(f"Extraction complete: {len(results)} reports processed")


# ---------------------------------------------------------------------------
# Graph build job
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    timeout=600,
    cpu=2,
    memory=8192,
)
def build_graph():
    """Build the knowledge graph on Modal.

    Usage: modal run modal_app.py::build_graph
    """
    import sys
    import os

    os.environ["AEROGRAPH_DATA_DIR"] = DATA_MOUNT
    sys.path.insert(0, "/root/src")

    from pathlib import Path
    from aerograph.graph import build_graph as _build, NetworkXBackend

    graphs_dir = Path(DATA_MOUNT) / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    backend = NetworkXBackend(path=graphs_dir / "aerograph.pkl")
    _build(
        extractions_path=Path(DATA_MOUNT) / "processed" / "extractions.jsonl",
        backend=backend,
    )
    volume.commit()
    print("Graph build complete")
