# NASA Mission RAG

A Flask Retrieval-Augmented Generation chat app that answers questions about 600+
NASA missions, using content scraped from the [NASA A-to-Z missions index](https://www.nasa.gov/a-to-z-of-nasa-missions/).
Inference is **fully local** — Ollama for the LLM and embeddings, ChromaDB for
vector search. No data leaves the server.

## Architecture

```
User query
   │  POST /api/chat
   ▼
rag.retrieve()         embed query (nomic-embed-text, "search_query:" prefix)
   │                   → ChromaDB vector search + keyword-filtered arms
   │                   → Reciprocal Rank Fusion + per-mission diversity cap
   ▼
rag.generate_response()  format context → stream gemma4:12b via Ollama
   │  Server-Sent Events
   ▼
Browser   marked.js → DOMPurify → rendered markdown
```

- **`app/chat/rag.py`** — retrieval (hybrid vector + keyword via RRF) and streamed generation.
- **`app/chat/ollama_client.py`** — single configured Ollama client (honors `OLLAMA_BASE_URL`, has a timeout).
- **`app/chat/routes.py`** — `GET /` UI, `POST /api/chat` (SSE), `GET /api/health` (deep check).
- **`app/ingest/`** — scrape the A-to-Z index → extract/clean text → chunk → embed (`search_document:` prefix) → ChromaDB.
- **`app/extensions.py`** — ChromaDB client + `nasa_reports` collection (HNSW, cosine).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit SECRET_KEY etc.
```

Requires a running [Ollama](https://ollama.com) with the chat and embedding models:

```bash
ollama pull gemma4:12b
ollama pull nomic-embed-text
```

## GPU (NVIDIA)

Ollama auto-detects the GPU, but to keep both models VRAM-resident on a dedicated
card (e.g. RTX 4090) and serve concurrent chats, add a systemd drop-in at
`/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_FLASH_ATTENTION=true"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`. Verify offload
with `ollama ps` (should show `100% GPU`) and `nvidia-smi` (VRAM in use during a query).

## Ingest

```bash
python scripts/ingest.py                  # all missions (~15-20 min)
python scripts/ingest.py --filter voyager # only matching missions
python scripts/ingest.py --limit 10       # first 10 (smoke test)
```

Re-ingest is idempotent per source URL. To rebuild from scratch, delete
`data/chroma_db/` first.

## Run

```bash
python run.py                                                          # dev, port 8100
gunicorn --worker-class gthread --workers 1 --threads 8 -b 0.0.0.0:8000 wsgi:app   # prod
```

A single gunicorn worker owns ChromaDB; threads serve concurrent SSE streams and
GPU concurrency comes from `OLLAMA_NUM_PARALLEL`. To scale horizontally, move
ChromaDB to server mode (`chromadb.HttpClient`) and the rate limiter to Redis
(`RATELIMIT_STORAGE_URI=redis://...`).

## Test

```bash
pytest tests/ -v
```

Ollama and ChromaDB are mocked, so the tests need no model server or vector DB.

## Configuration

See `.env.example`. Key variables: `OLLAMA_CHAT_MODEL` (`gemma4:12b`),
`OLLAMA_NUM_CTX` (`8192`), `RAG_TOP_K` (`6`), `RAG_MAX_PER_SOURCE` (`2`),
`RAG_TEMPERATURE` (`0.3`).
