# NASA Mission RAG

A Flask Retrieval-Augmented Generation chat app that answers questions about 600+
NASA missions, using content scraped from the [NASA A-to-Z missions index](https://www.nasa.gov/a-to-z-of-nasa-missions/).
Inference is **fully local** — Ollama for the LLM and embeddings, ChromaDB for
vector search. No data leaves the server.

## Architecture

```
User query (+ recent turns as history)
   │  POST /api/chat
   ▼
generate_response()   follow-up? → condense to a standalone query (_contextualize_query)
   │                  ("what did it find at Saturn?" → "...did Voyager find at Saturn?")
   ▼
rag.retrieve()         embed query (nomic-embed-text, "search_query:" prefix)
   │   topical  →  vector search + keyword-filtered arms → Reciprocal Rank Fusion
   │   temporal →  gate candidates by launch-year metadata + year-in-text
   │              (so "latest missions in 2026" can't return Apollo 13)
   │        ▼
   │   cross-encoder rerank (bge-reranker-v2-m3), fused with first-stage rank,
   │   then per-mission diversity cap
   ▼
generate_response()  format context → replay history → stream gemma4:26b via Ollama
   │  Server-Sent Events
   ▼
Browser   marked.js → DOMPurify → rendered markdown
```

- **`app/chat/rag.py`** — two-stage retrieval (hybrid/temporal candidate pool → cross-encoder rerank fused with the first-stage rank), multi-turn follow-up handling (`_contextualize_query`/`_normalize_history`), and streamed generation.
- **`app/chat/reranker.py`** — lazily-loaded cross-encoder, kept GPU-resident, with graceful fallback to fusion order if unavailable.
- **`app/chat/ollama_client.py`** — single configured Ollama client (honors `OLLAMA_BASE_URL`, has a timeout).
- **`app/chat/routes.py`** — `GET /` UI, `POST /api/chat` (SSE), `GET /api/health` (deep check).
- **`app/ingest/`** — scrape the A-to-Z index → extract/clean text → chunk → embed (`search_document:` prefix) → ChromaDB; `extract_year()` tags a per-page launch year for temporal queries.
- **`app/extensions.py`** — ChromaDB client + `nasa_reports` collection (HNSW, cosine).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit SECRET_KEY etc.
```

Requires a running [Ollama](https://ollama.com) with the chat and embedding models:

```bash
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

The cross-encoder reranker (`bge-reranker-v2-m3`, ~2.3 GB) downloads from Hugging
Face on first use and is cached under `~/.cache/huggingface`.

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

`gemma4:26b` (~17 GB) + `nomic-embed-text` + the reranker (~2.3 GB) co-reside on a
24 GB card, so run the app with a **single** gunicorn worker (one reranker
instance). Set `RAG_RERANK_DEVICE=cpu` to move the reranker off the GPU if VRAM
is tight.

## Ingest

```bash
python scripts/ingest.py                  # all missions (~15-20 min)
python scripts/ingest.py --filter voyager # only matching missions
python scripts/ingest.py --limit 10       # first 10 (smoke test)
```

Re-ingest is idempotent per source URL. To rebuild from scratch, delete
`data/chroma_db/` first.

To refresh launch-year metadata and drop scraper noise **without** re-scraping
(recomputes `year` from stored text in place):

```bash
python scripts/repair_metadata.py            # preview (dry run)
python scripts/repair_metadata.py --apply     # write
```

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

See `.env.example`. Key variables: `OLLAMA_CHAT_MODEL` (`gemma4:26b`),
`OLLAMA_NUM_CTX` (`8192`), `RAG_TOP_K` (`6`), `RAG_MAX_PER_SOURCE` (`2`),
`RAG_TEMPERATURE` (`0.3`), `RAG_CANDIDATE_K` (`40`, rerank pool size),
`RAG_RERANK_MODEL` (`BAAI/bge-reranker-v2-m3`), `RAG_RERANK_WEIGHT` (`0.5`,
cross-encoder weight when fused with first-stage retrieval), `RAG_RERANK_DEVICE`
(`auto`). Conversational memory: `RAG_CONTEXTUALIZE` (`true`, rewrite follow-ups
into standalone queries), `RAG_HISTORY_MAX_TURNS` (`6`), `RAG_HISTORY_MAX_CHARS`
(`1200`).
