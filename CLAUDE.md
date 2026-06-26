# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASA Mission RAG is a Flask-based Retrieval-Augmented Generation chat application that answers questions about NASA missions using content scraped from the NASA A-to-Z missions index (`https://www.nasa.gov/a-to-z-of-nasa-missions/`), covering 619+ missions. It uses Ollama for local LLM inference and ChromaDB for vector storage.

## Commands

```bash
# Development server (port 8100, debug mode)
python run.py

# Production server — single worker owns ChromaDB; threads serve concurrent SSE
# streams; GPU concurrency comes from Ollama's OLLAMA_NUM_PARALLEL.
gunicorn --worker-class gthread --workers 1 --threads 8 -b 0.0.0.0:8000 wsgi:app

# Ingest NASA mission web pages into ChromaDB
python scripts/ingest.py              # all missions
python scripts/ingest.py --limit 10   # first 10 only
python scripts/ingest.py --filter voyager  # matching missions only
python scripts/ingest.py --delay 3    # custom delay between fetches (default: 2s)

# Repair launch-year metadata + drop scraper noise IN PLACE (no re-scrape)
python scripts/repair_metadata.py            # dry run (preview)
python scripts/repair_metadata.py --apply     # write

# Install dependencies
pip install -r requirements.txt
```

Tests use pytest: `pytest tests/ -v`. Ollama and ChromaDB are mocked, so no model
server or vector DB is required to run them.

## Architecture

**Data flow:** User query (+ prior turns) → `/api/chat` route → `rag.generate_response()` → on a follow-up, `_contextualize_query()` rewrites it into a standalone search query → `rag.retrieve()` (embed query → build a candidate pool → cross-encoder rerank) → format context, replay history, stream LLM response via SSE → frontend renders streamed markdown.

**Conversational memory:** The chat is multi-turn. The browser keeps the conversation and posts recent turns as `history` with each question (server stays stateless — no session store). Two things use it: (1) **retrieval** — `_contextualize_query()` folds the conversation back into a standalone query so a bare follow-up ("what did *it* find at Saturn?") keeps its entity ("Voyager"); this is one extra non-streamed LLM call, skipped when there's no history. (2) **generation** — normalized prior turns are replayed to the chat model between the system prompt and the current (context-laden) question. `_normalize_history()` validates/bounds the client input (`RAG_HISTORY_MAX_TURNS`, `RAG_HISTORY_MAX_CHARS`). The rewrite is best-effort: any failure falls back to the raw question.

**Retrieval is two-stage** (`rag.retrieve()`):
1. **Candidate pool.** *Topical* queries fuse a vector arm with keyword-filtered arms via Reciprocal Rank Fusion. *Temporal* queries (explicit year, or recency words like "latest/recent") instead gate the pool to chunks whose `year` metadata is in the target window or whose text names the year — otherwise the topical reranker, blind to recency, floats an old-but-vivid mission to the top (e.g. Apollo 13 for "latest missions in 2026").
2. **Rerank.** A cross-encoder (`bge-reranker-v2-m3`) scores query/passage relevance; its rank is **fused via weighted RRF with the first-stage rank** (`RAG_RERANK_WEIGHT`, default 0.5) so a confident-but-wrong reranker can't bury strong retrieval hits (entity drift). A per-mission diversity cap (`RAG_MAX_PER_SOURCE`) is applied last.

**Key modules:**
- `app/__init__.py` — App factory (`create_app`), registers blueprints/error handlers, optionally preloads the reranker (`RAG_RERANK_PRELOAD`)
- `app/config.py` — Environment-based config classes (Development/Production/Testing)
- `app/extensions.py` — ChromaDB client & collection globals (`nasa_reports` collection)
- `app/chat/routes.py` — Three endpoints: `GET /` (UI), `POST /api/chat` (streaming chat), `GET /api/health`
- `app/chat/rag.py` — Core RAG logic: `retrieve()` (two-stage), `_temporal_targets()` (temporal classification), `_contextualize_query()`/`_normalize_history()` (multi-turn follow-ups), `generate_response(question, history)` streams from Ollama
- `app/chat/reranker.py` — Lazily-loaded cross-encoder (GPU, CPU fallback); `score()` returns per-candidate relevance, degrades gracefully if unavailable
- `app/chat/prompts.py` — System, user, and condense (query-rewrite) prompt templates
- `app/ingest/pipeline.py` — Web text extraction → chunking (800 chars, 200 overlap) → embedding → ChromaDB; `extract_year()` tiered launch-year extraction
- `app/ingest/web_sources.py` — Scrapes NASA A-to-Z missions index for mission page URLs (rejects news-feed cards)

**Frontend:** Vanilla JS with SSE streaming (`static/js/chat.js`), marked.js for markdown rendering, single-page chat UI (`templates/index.html`).

## Environment Setup

Copy `.env.example` to `.env`. Required variables:
- `OLLAMA_BASE_URL` — Ollama server (default: `http://127.0.0.1:11434`)
- `OLLAMA_CHAT_MODEL` — LLM model (default: `gemma4:26b`)
- `OLLAMA_EMBED_MODEL` — Embedding model (default: `nomic-embed-text`)
- `OLLAMA_NUM_CTX` — Chat context window in tokens (default: `8192`)
- `OLLAMA_THINK` — Enable the chat model's chain-of-thought (default: `false`)
- `OLLAMA_TIMEOUT` — Ollama request timeout in seconds (default: `120`)
- `CHROMA_DB_PATH` — Vector DB path (default: `data/chroma_db`)
- `RAG_TOP_K` — Number of chunks to retrieve (default: `6`)
- `RAG_MAX_PER_SOURCE` — Max chunks kept per mission page (default: `2`)
- `RAG_TEMPERATURE` — LLM temperature (default: `0.3`)
- `RAG_CANDIDATE_K` — Candidate pool size fed to the reranker (default: `40`)
- `RAG_RERANK_ENABLED` — Toggle the cross-encoder stage (default: `true`; off in tests)
- `RAG_RERANK_MODEL` — Cross-encoder (default: `BAAI/bge-reranker-v2-m3`)
- `RAG_RERANK_DEVICE` — `auto` | `cuda` | `cpu` (default: `auto`)
- `RAG_RERANK_WEIGHT` — Cross-encoder weight when fused with first-stage rank (default: `0.5`)
- `RAG_RERANK_PRELOAD` — Load the reranker at server start (default: `false`; set `true` in the service env)
- `RAG_CONTEXTUALIZE` — Rewrite follow-up questions into standalone search queries via one extra LLM call (default: `true`; `false` keeps history but skips the rewrite)
- `RAG_HISTORY_MAX_TURNS` — Max prior turns replayed to the model / used for the rewrite (default: `6`)
- `RAG_HISTORY_MAX_CHARS` — Per-turn truncation so a long prior answer can't crowd out retrieved context (default: `1200`)

## Key Technical Details

- Rate limiting: 30 requests/minute via flask-limiter (`RATELIMIT_STORAGE_URI`, default in-memory)
- ChromaDB uses HNSW with cosine similarity; chunk metadata includes mission, year, source URL
- Retrieval is two-stage: hybrid/temporal candidate pool → cross-encoder rerank fused with the first-stage rank → per-mission diversity cap (see Architecture)
- `year` metadata powers temporal queries; `extract_year()` is tiered (launch-adjacent year → earliest full "Month Day, Year" → earliest "Month Year" → 0). `scripts/repair_metadata.py` recomputes it in place
- Embeddings use nomic-embed-text task prefixes (`search_query:` for queries, `search_document:` for passages)
- Multi-turn: history lives client-side and is posted as `history` per request (stateless server). Follow-ups are condensed to a standalone query before retrieval (`_contextualize_query`), and prior turns are replayed to the chat model. Disable the rewrite with `RAG_CONTEXTUALIZE=false`
- Ollama context window: configurable via `OLLAMA_NUM_CTX` (default 8192)
- GPU: Ollama runs on an NVIDIA RTX 4090; a systemd drop-in keeps the chat + embed models VRAM-resident (`OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_MAX_LOADED_MODELS=2`). `gemma4:26b` (~17 GB) + embed + reranker (~2.3 GB) co-reside on 24 GB — run **one** gunicorn worker (one reranker instance); `RAG_RERANK_DEVICE=cpu` frees GPU if tight
- Responses stream as JSON-formatted Server-Sent Events
