"""AeroGraph — minimal HF Space demo (cached-only).

The repo's full Gradio app is in app_full.py; it loads NetworkX graphs,
ChromaDB, and sentence-transformers at query time. That import chain
consistently left HF Space's container in a "Running" state with no port
bound. This file is a deliberately-minimal demo that loads zero heavy
dependencies and reads precomputed answers from data/demo_cache.jsonl.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

DATA = Path(__file__).parent / "data"
DEMO_CACHE = DATA / "demo_cache.jsonl"
REPO_URL = "https://github.com/Aryan95614/AeroGraph"


def _load_answers() -> dict[str, dict]:
    """question -> {answer, sources, latency}. Empty dict if file missing."""
    if not DEMO_CACHE.exists():
        return {}
    out: dict[str, dict] = {}
    with open(DEMO_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = e.get("question")
            if q:
                out[q] = e
    return out


ANSWERS = _load_answers()
PRESET_QUERIES = list(ANSWERS.keys()) if ANSWERS else [
    "What are common factors in bird strike incidents?",
]


def answer_query(q: str) -> tuple[str, str, str]:
    q = (q or "").strip()
    if not q:
        return "", "", ""
    entry = ANSWERS.get(q)
    if not entry:
        msg = (
            "This HF Space runs in cached-only mode with 10 precomputed "
            f"showcase queries. Pick one from the buttons, or clone the repo "
            f"({REPO_URL}) and run `python app_full.py` locally with "
            "`ANTHROPIC_API_KEY` set for arbitrary queries."
        )
        return msg, "", ""
    sources = entry.get("sources", [])[:8]
    src_md = "\n".join(f"- ACN **{a}**" for a in sources) or "*no sources*"
    meta = (
        f"**Cached benchmark answer.** "
        f"Retrieval {entry.get('retrieval_latency_ms', 0):.0f}ms + "
        f"generation {entry.get('generation_latency_ms', 0):.0f}ms"
    )
    return entry["answer"], src_md, meta


def build() -> gr.Blocks:
    with gr.Blocks(
        title="AeroGraph — Cached Demo",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    ) as app:
        gr.Markdown(
            f"""# AeroGraph — Cached Demo
### Hybrid graph-augmented retrieval over 2,000 NASA ASRS safety reports

This Space serves **10 precomputed showcase queries** from our benchmark
run. For the full live system (6 retrievers, dual-judge eval,
graph explorer): see **[GitHub]({REPO_URL})** and `app_full.py`.

Paper finding: retrieval-precision rankings do not replicate on a
disjoint held-out corpus (Kendall τ ≈ −0.07) — we name this
**Extractive Oracle Circularity** and propose held-out swap as a
falsifiability test for graph-RAG benchmarks.
"""
        )

        q_box = gr.Textbox(
            label="Question (pick one of the preset buttons)",
            placeholder=PRESET_QUERIES[0] if PRESET_QUERIES else "",
            lines=2,
        )

        gr.Markdown("**Presets:**")
        with gr.Row():
            for pq in PRESET_QUERIES:
                label = (pq[:70] + "…") if len(pq) > 70 else pq
                gr.Button(label, size="sm").click(lambda p=pq: p, outputs=q_box)

        ask = gr.Button("Show cached answer", variant="primary")
        meta_out = gr.Markdown()
        ans_out = gr.Markdown()
        with gr.Accordion("Source ACN citations", open=True):
            src_out = gr.Markdown()

        ask.click(answer_query, [q_box], [ans_out, src_out, meta_out])
        q_box.submit(answer_query, [q_box], [ans_out, src_out, meta_out])

        gr.Markdown(
            f"\n---\nSource: {REPO_URL} · "
            f"Dataset: https://huggingface.co/datasets/Aryan95614/aerograph-asrs"
        )
    return app


if __name__ == "__main__":
    build().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
    )
