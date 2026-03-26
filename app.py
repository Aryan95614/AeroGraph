import json
import numpy as np
import faiss
from openai import OpenAI

import modal
from config import (
    app, image, volume, VOLUME_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM, GENERATION_MODEL, TOP_K,
)


SYSTEM_PROMPT = """You are AeroGraph, an aviation safety research assistant. You answer questions using NASA ASRS (Aviation Safety Reporting System) incident reports provided as context.

Rules:
- Only use information from the provided ASRS reports. Do not fabricate details.
- Cite specific ASRS report numbers (ACN) when referencing incidents.
- If the context doesn't contain relevant information, say so clearly.
- Be precise and factual. This data matters for aviation safety."""


def load_index_and_chunks():
    index = faiss.read_index(f"{VOLUME_PATH}/index.faiss")
    with open(f"{VOLUME_PATH}/chunks.json") as f:
        chunks = json.load(f)
    return index, chunks


def embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def search(index, chunks: list[dict], query_vec: np.ndarray, top_k: int = TOP_K):
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({"chunk": chunks[idx], "score": float(score)})
    return results


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"--- Report {i} (ACN {r['chunk']['acn']}, relevance {r['score']:.3f}) ---")
        parts.append(r["chunk"]["text"])
    return "\n\n".join(parts)


def generate_answer(client: OpenAI, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("openai-secret")],
)
@modal.web_endpoint(method="POST")
def query(request: dict):
    question = request.get("question", "").strip()
    if not question:
        return {"error": "No question provided"}

    client = OpenAI()
    index, chunks = load_index_and_chunks()

    query_vec = embed_query(client, question)
    results = search(index, chunks, query_vec)
    context = build_context(results)
    answer = generate_answer(client, question, context)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"acn": r["chunk"]["acn"], "score": r["score"]}
            for r in results
        ],
    }


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok", "service": "aerograph"}
