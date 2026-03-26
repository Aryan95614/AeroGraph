import json
import numpy as np
import faiss
from openai import OpenAI

import modal
from config import (
    app, image, volume, VOLUME_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_BATCH_SIZE,
)
from data import load_reports, chunk_all_reports


def get_openai_client() -> OpenAI:
    return OpenAI()


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=3600,
)
def build_index():
    print("Loading ASRS reports from HuggingFace...")
    dataset = load_reports()
    print(f"Loaded {len(dataset)} reports")

    print("Chunking reports...")
    chunks = chunk_all_reports(dataset)
    print(f"Created {len(chunks)} chunks")

    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(chunks), CHUNK_BATCH_SIZE):
        batch = chunks[i : i + CHUNK_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(client, texts)
        all_embeddings.extend(embeddings)
        print(f"Embedded {min(i + CHUNK_BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Building FAISS index...")
    vectors = np.array(all_embeddings, dtype="float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    print(f"Index contains {index.ntotal} vectors")

    faiss.write_index(index, f"{VOLUME_PATH}/index.faiss")
    with open(f"{VOLUME_PATH}/chunks.json", "w") as f:
        json.dump(chunks, f)
    volume.commit()

    print("Pipeline complete. Index and chunks saved to volume.")
    return {"chunks": len(chunks), "vectors": index.ntotal}


@app.local_entrypoint()
def main():
    result = build_index.remote()
    print(f"Result: {result}")
