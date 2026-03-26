# Building a RAG System for Aviation Safety Data

## The problem

NASA's Aviation Safety Reporting System has collected hundreds of thousands of voluntary incident reports since 1976. Pilots, controllers, mechanics, and flight attendants file confidential narratives describing what went wrong — near misses, equipment failures, communication breakdowns, procedural errors.

The data is public. The interface is not built for pattern discovery. You search one field at a time, one report at a time. There is no way to ask "what are the most common human factors in approach-phase incidents?" and get an answer that synthesizes across thousands of reports.

After the Reagan National midair collision in January 2025, the NTSB found that the FAA had over 15,000 documented close-proximity events at DCA. The data existed. Nobody queried it at scale.

## What we're building

AeroGraph is a RAG (Retrieval-Augmented Generation) system that:
1. Loads 47,723 ASRS reports from HuggingFace
2. Chunks each report into a retrievable segment combining narrative, synopsis, and metadata
3. Embeds everything with OpenAI's text-embedding-3-small
4. Indexes the vectors in FAISS for fast similarity search
5. Serves a query endpoint on Modal that retrieves relevant reports and generates cited answers via GPT-4o-mini

The entire system runs serverless on Modal. No infrastructure to manage.

## Step 1: Data loading and chunking

We use the `elihoole/asrs-aviation-reports` dataset on HuggingFace, which contains 47,723 parsed ASRS reports.

Each report has structured fields — aircraft type, airport, flight phase, primary problem, human factors — plus free-text narratives written by the reporters themselves and synopses written by NASA analysts.

We combine these into a single text chunk per report. The chunk includes the ACN (report ID), key metadata fields, the narrative, the synopsis, and callback notes if available. This gives the retrieval model a dense, complete representation of each incident.

See `data.py` for the implementation.

## Step 2: Embedding

We embed each chunk using OpenAI's `text-embedding-3-small` model (1536 dimensions). We batch requests in groups of 512 to stay within rate limits and keep costs reasonable.

At 47K reports, the full embedding pass costs roughly $0.50 in API calls and takes about 15 minutes on Modal.

## Step 3: FAISS indexing

We use a flat inner-product index (`IndexFlatIP`) with L2-normalized vectors, which is equivalent to cosine similarity. For 47K vectors at 1536 dimensions, this fits comfortably in memory and search is instant.

For the full 300K+ corpus, we'd switch to an IVF index for sublinear search time.

## Step 4: Query endpoint

The Modal web endpoint accepts a POST request with a natural language question. It:
1. Embeds the question with the same model
2. Searches the FAISS index for the top-k most similar chunks
3. Passes the retrieved chunks as context to GPT-4o-mini
4. Returns the generated answer with source ACN citations

The system prompt instructs the model to cite specific report numbers and to say when the context doesn't contain relevant information rather than hallucinating.

## Step 5: Deployment

Everything deploys with `modal deploy app.py`. Modal handles container images, secrets, scaling, and HTTPS endpoints. The FAISS index and chunks are persisted on a Modal volume so the query endpoint loads them on cold start.

## What's next

- Scale to the full ASRS corpus (300K+ reports)
- Extract entities (airports, aircraft, operators, failure modes) and build a knowledge graph
- Add an interactive web UI for non-technical users
- Fine-tune a domain-specific embedding model on aviation safety text

## Links

- [NASA ASRS](https://asrs.arc.nasa.gov/)
- [HuggingFace dataset](https://huggingface.co/datasets/elihoole/asrs-aviation-reports)
- [Modal](https://modal.com)
- [FAISS](https://github.com/facebookresearch/faiss)
