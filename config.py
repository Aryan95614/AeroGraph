import modal

app = modal.App("aerograph")

volume = modal.Volume.from_name("aerograph-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai>=1.12.0",
        "faiss-cpu>=1.7.4",
        "datasets>=2.18.0",
        "numpy>=1.26.0",
    )
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
GENERATION_MODEL = "gpt-4o-mini"

TOP_K = 10
CHUNK_BATCH_SIZE = 512
DATASET_NAME = "elihoole/asrs-aviation-reports"
