"""ChromaDB chunking and embedding pipeline.

Chunks each report into 256-token overlapping windows (128 overlap),
embeds with sentence-transformers all-MiniLM-L6-v2, and stores in
ChromaDB collection 'aerograph_chunks'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"

COLLECTION_NAME = "aerograph_chunks"
CHUNK_SIZE = 256  # tokens (approx words for rough tokenization)
CHUNK_OVERLAP = 128
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    report_id: str
    chunk_index: int
    entities_mentioned: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _rough_tokenize(text: str) -> list[str]:
    """Split text into rough word tokens."""
    return text.split()


def chunk_report(
    report_id: str,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    entities: Optional[list[str]] = None,
) -> list[Chunk]:
    """Split a report into overlapping chunks."""
    tokens = _rough_tokenize(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])

        # Find entities mentioned in this chunk
        mentioned = []
        if entities:
            chunk_lower = chunk_text.lower()
            for ent in entities:
                if ent.lower() in chunk_lower:
                    mentioned.append(ent)

        chunks.append(Chunk(
            chunk_id=f"{report_id}_chunk_{chunk_idx}",
            text=chunk_text,
            report_id=report_id,
            chunk_index=chunk_idx,
            entities_mentioned=mentioned,
        ))

        chunk_idx += 1
        start += chunk_size - overlap
        if end == len(tokens):
            break

    return chunks


def get_chroma_client(persist_dir: Optional[Path] = None) -> chromadb.ClientAPI:
    """Get or create ChromaDB client with persistence."""
    if persist_dir is None:
        persist_dir = CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(EMBED_MODEL)


def get_collection(persist_dir: Optional[Path] = None):
    """Get the existing ChromaDB collection."""
    client = get_chroma_client(persist_dir)
    return client.get_collection(COLLECTION_NAME)


def build_index(
    reports_path: Optional[Path] = None,
    extractions_path: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
) -> int:
    """Build the ChromaDB index from processed reports.

    Returns the number of chunks indexed.
    """
    if reports_path is None:
        reports_path = PROCESSED_DIR / "reports.jsonl"
    if extractions_path is None:
        extractions_path = PROCESSED_DIR / "extractions.jsonl"

    # Load reports (deduplicate by ID)
    from aerograph.ingest import load_reports
    all_reports = load_reports(reports_path)
    seen_ids: set[str] = set()
    reports = []
    for r in all_reports:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            reports.append(r)
    print(f"Loaded {len(reports)} reports for embedding (deduped from {len(all_reports)})")

    # Load entity extractions for chunk-entity mapping
    entity_map: dict[str, list[str]] = {}
    if extractions_path.exists():
        with open(extractions_path) as f:
            for line in f:
                data = json.loads(line)
                rid = data["report_id"]
                entities = [e["canonical_name"] for e in data.get("entities", [])]
                entity_map[rid] = entities

    # Chunk all reports
    all_chunks: list[Chunk] = []
    for report in reports:
        entities = entity_map.get(report.id, [])
        chunks = chunk_report(report.id, report.text, entities=entities)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks from {len(reports)} reports")

    if not all_chunks:
        return 0

    # Embed
    model = get_embedding_model()
    texts = [c.text for c in all_chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Store in ChromaDB
    client = get_chroma_client(persist_dir)
    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert (ChromaDB has batch size limits)
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size].tolist()

        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=batch_embeddings,
            documents=[c.text for c in batch],
            metadatas=[{
                "report_id": c.report_id,
                "chunk_index": c.chunk_index,
                "entities_mentioned": json.dumps(c.entities_mentioned),
            } for c in batch],
        )

    print(f"Indexed {len(all_chunks)} chunks into ChromaDB")
    return len(all_chunks)


def query_similar(
    query_text: str,
    top_k: int = 10,
    persist_dir: Optional[Path] = None,
    model: Optional[SentenceTransformer] = None,
) -> list[dict]:
    """Query ChromaDB for similar chunks.

    Returns list of dicts with keys: chunk_id, text, report_id, chunk_index,
    entities_mentioned, distance.
    """
    client = get_chroma_client(persist_dir)
    collection = client.get_collection(COLLECTION_NAME)

    if model is None:
        model = get_embedding_model()
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "report_id": meta["report_id"],
            "chunk_index": meta["chunk_index"],
            "entities_mentioned": json.loads(meta.get("entities_mentioned", "[]")),
            "distance": results["distances"][0][i],
        })

    return chunks


if __name__ == "__main__":
    count = build_index()
    print(f"\nEmbedding complete: {count} chunks indexed")
