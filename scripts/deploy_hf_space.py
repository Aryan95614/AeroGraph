"""Deploy AeroGraph app.py + cached demo data to a HuggingFace Space.

Requires HF_TOKEN environment variable or prior `huggingface-cli login`.
Cached mode is the default on the Space (no secrets needed at runtime).
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

SPACE_REPO_ID = os.environ.get("HF_SPACE_ID", "Aryan95614/aerograph")
LOCAL_STAGE = Path("/tmp/aerograph_space_stage")


def main():
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("huggingface_hub not installed.")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()

    try:
        user = api.whoami()
        print(f"HF auth as: {user.get('name', user)}")
    except Exception as e:
        print(f"HF auth failed: {e}")
        sys.exit(1)

    # Create repo (idempotent — exist_ok)
    try:
        create_repo(
            SPACE_REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            token=token,
        )
        print(f"Space repo ready: {SPACE_REPO_ID}")
    except Exception as e:
        print(f"create_repo warning: {e}")

    # Stage files
    if LOCAL_STAGE.exists():
        shutil.rmtree(LOCAL_STAGE)
    LOCAL_STAGE.mkdir(parents=True)

    # app.py
    shutil.copy("app.py", LOCAL_STAGE / "app.py")

    # requirements.txt — minimal set the Space needs.
    # audioop-lts backports the stdlib module pydub needs on Python 3.13
    # (PEP 594 removed audioop from cpython 3.13).
    # huggingface_hub<0.28 is required because gradio 4.44 imports HfFolder,
    # which was removed from huggingface_hub in 0.28. Do NOT drop the upper bound.
    # Version pins (all three are load-bearing for HF Spaces):
    #   gradio<5                    stay on the 4.x line we've tested against
    #   huggingface_hub<0.28        gradio 4.44 imports HfFolder (removed in 0.28)
    #   jinja2<3.1.5                gradio's template cache hits
    #                                TypeError: unhashable type: 'dict' on 3.1.5+
    #   audioop-lts                 fallback if HF defaults to Python 3.13
    # Load-bearing version pins:
    #   gradio==4.44.1           bug in 4.44.0 passes a dict as jinja2 cache
    #                             key; 4.44.1 fixed it (gradio #9985)
    #   huggingface_hub<0.28     gradio 4.44 imports HfFolder (removed in 0.28)
    #   jinja2==3.1.2            strict-hash behaviour in 3.1.3+ surfaces
    #                             gradio's latent cache-key bug
    #   audioop-lts              fallback if HF defaults to Python 3.13
    requirements = """gradio==4.44.1
huggingface_hub>=0.20,<0.28
jinja2==3.1.2
audioop-lts>=0.2.1; python_version>='3.13'
anthropic>=0.39.0
chromadb>=0.4.22
networkx>=3.2
sentence-transformers>=2.3.0
numpy>=1.26.0
python-dotenv>=1.0.0
"""
    (LOCAL_STAGE / "requirements.txt").write_text(requirements)

    # README for the Space (different from project README — HF expects YAML frontmatter)
    space_readme = """---
title: AeroGraph
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Graph-RAG over NASA ASRS aviation safety reports
---

# AeroGraph

Hybrid retrieval + knowledge graph over NASA ASRS incident reports.
This Space runs in **cached mode** by default — 10 showcase queries return
precomputed answers. For custom queries, clone the repo and set
`ANTHROPIC_API_KEY`.

[GitHub](https://github.com/Aryan95614/AeroGraph) ·
[Dataset](https://huggingface.co/datasets/Aryan95614/aerograph-asrs)
"""
    (LOCAL_STAGE / "README.md").write_text(space_readme)

    # Cached demo data
    data_dir = LOCAL_STAGE / "data"
    data_dir.mkdir()
    demo_files = [
        "data/demo_cache.jsonl",
        "data/cached_demo_answers.json",
    ]
    for f in demo_files:
        p = Path(f)
        if p.exists():
            shutil.copy(p, data_dir / p.name)
            print(f"staged {f}")

    # Upload
    print(f"Uploading {LOCAL_STAGE} -> {SPACE_REPO_ID}")
    try:
        upload_folder(
            folder_path=str(LOCAL_STAGE),
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message="ship v0.1 — cached-mode demo",
            token=token,
        )
        print(f"Space live: https://huggingface.co/spaces/{SPACE_REPO_ID}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
