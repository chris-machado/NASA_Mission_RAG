# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NASA Mission RAG is a Flask-based Retrieval-Augmented Generation chat application that answers questions about NASA missions using content scraped from the NASA A-to-Z missions index (`https://www.nasa.gov/a-to-z-of-nasa-missions/`), covering 619+ missions. It uses Ollama for local LLM inference and ChromaDB for vector storage.

## Commands

```bash
# Development server (port 8100, debug mode)
python run.py

# Production server
gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app

# Ingest NASA mission web pages into ChromaDB
python scripts/ingest.py              # all missions
python scripts/ingest.py --limit 10   # first 10 only
python scripts/ingest.py --filter voyager  # matching missions only
python scripts/ingest.py --delay 3    # custom delay between fetches (default: 2s)

# Install dependencies
pip install -r requirements.txt
```

No test framework is configured yet. The `tests/` directory exists but is empty.

## Architecture

**Data flow:** User query → `/api/chat` route → `rag.retrieve()` (embed query via Ollama → ChromaDB cosine search) → `rag.generate_response()` (format context + stream LLM response via SSE) → frontend renders streamed markdown.

**Key modules:**
- `app/__init__.py` — App factory (`create_app`), registers blueprints and error handlers
- `app/config.py` — Environment-based config classes (Development/Production/Testing)
- `app/extensions.py` — ChromaDB client & collection globals (`nasa_reports` collection)
- `app/chat/routes.py` — Three endpoints: `GET /` (UI), `POST /api/chat` (streaming chat), `GET /api/health`
- `app/chat/rag.py` — Core RAG logic: `retrieve()` does vector search, `generate_response()` streams from Ollama
- `app/chat/prompts.py` — System and user prompt templates
- `app/ingest/pipeline.py` — Web text extraction → chunking (800 chars, 200 overlap) → embedding → ChromaDB storage
- `app/ingest/web_sources.py` — Scrapes NASA A-to-Z missions index for mission page URLs

**Frontend:** Vanilla JS with SSE streaming (`static/js/chat.js`), marked.js for markdown rendering, single-page chat UI (`templates/index.html`).

## Environment Setup

Copy `.env.example` to `.env`. Required variables:
- `OLLAMA_BASE_URL` — Ollama server (default: `http://127.0.0.1:11434`)
- `OLLAMA_CHAT_MODEL` — LLM model (default: `llama3.2:3b`)
- `OLLAMA_EMBED_MODEL` — Embedding model (default: `nomic-embed-text`)
- `CHROMA_DB_PATH` — Vector DB path (default: `data/chroma_db`)
- `RAG_TOP_K` — Number of chunks to retrieve (default: `5`)
- `RAG_TEMPERATURE` — LLM temperature (default: `0.3`)

## Key Technical Details

- Rate limiting: 30 requests/minute via flask-limiter
- ChromaDB uses HNSW with cosine similarity; chunk metadata includes mission, year, source URL
- Ollama context window: 4096 tokens
- Responses stream as JSON-formatted Server-Sent Events
