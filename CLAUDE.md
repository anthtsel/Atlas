# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**ATLAS Workspace Agent** — a monolithic FastAPI application (~777 lines in `atlas_agent.py`) combining local LLM inference (Ollama), hybrid RAG search, and Google Workspace integration. Single Python backend + single HTML frontend.

## Running the app

```bash
# Activate venv
source .venv/bin/activate

# Start the server (requires Ollama running at http://localhost:11434)
python atlas_agent.py

# UI: http://localhost:8000/ui
# API docs: http://localhost:8000/docs
```

Ollama must be running before starting. Start it with: `ollama serve`

## Environment

All config lives in `.env`. Key variables:
- `RAG_MODEL` / `BRIEFING_MODEL` — Ollama models (default: `gemma3:4b`)
- `EMBED_MODEL` / `RERANK_MODEL` — embedding/reranking (default: `nomic-embed-text`)
- `ENABLE_GOOGLE` — set `false` to disable Google integration entirely
- `CHROMA_COLLECTION` — ChromaDB collection name

Google OAuth requires `credentials.json` (downloaded from Google Cloud Console). First run will open browser for auth; token saved to `token.json`.

## Architecture

All logic lives in `atlas_agent.py`. Classes in order:

1. **`HybridSearch`** — 3-stage retrieval: BM25 keyword → ChromaDB vector → LLM reranker. Returns top results with trace metadata for UI visualization.

2. **`LocalMemory`** — Wraps ChromaDB. Chunks documents (500-word chunks, 50-word overlap), embeds via Ollama, stores/retrieves. Loads from `knowledge/` directory on ingest.

3. **`GoogleConnector`** — OAuth2 wrapper for Drive, Gmail, Docs, Sheets, Calendar, Tasks APIs. Handles token refresh automatically.

4. **`ReasoningEngine`** — RAG orchestrator. `ask()` returns answer + metadata; `ask_stream()` yields SSE tokens. `synthesize_briefing()` uses a 14b model to combine calendar + tasks + memory.

5. **`HybridAgent`** — Top-level orchestrator. Composes LocalMemory + ReasoningEngine + GoogleConnector. Google services are optional (graceful degradation if credentials missing).

6. **FastAPI app** — Lifespan initializes `HybridAgent` once at startup. CORS open. Background tasks for ingestion. SSE for streaming.

### Request flow (RAG query)

```
POST /ask  →  ReasoningEngine.ask()
               └─ LocalMemory.retrieve()
                   └─ HybridSearch.search()
                       ├─ BM25 (top 20)
                       ├─ ChromaDB vector (top 10)
                       ├─ Merge + dedupe
                       └─ LLM rerank (top 3)
               └─ Build prompt + call Ollama
               └─ Return answer + chunks + trace
```

### Document ingestion sources

| Source | Entry point | Processing |
|---|---|---|
| Local files | `POST /ingest/local` | `.txt`/`.md` from `knowledge/` |
| Google Doc/Sheet | `POST /ingest/doc` | Exported as text or Markdown table |
| YouTube video | `POST /ingest/doc` | Transcript via `youtube-transcript-api` |

All sources are chunked then vectorized with `EMBED_MODEL`.

## Frontend

`atlas_workspace.html` is a single-page app (vanilla JS, no framework). Dark terminal aesthetic with green accents. Renders markdown responses, animates SSE streams, and visualizes search trace (BM25 scores → vector candidates → reranker picks). Fetches from same-origin FastAPI.

## Dependencies

Install into the existing venv:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

No build step. No test suite currently.

## Key constraints

- No cloud LLM calls — Ollama is the only inference backend.
- ChromaDB is file-persisted in `chroma_db/` (git-ignored). Deleting this directory clears all memory.
- `credentials.json` and `token.json` are git-ignored and must never be committed.
- Background ingestion tasks log to stdout only — no persistent job state.
