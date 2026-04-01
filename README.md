# ATLAS Workspace Agent

An agentic AI assistant combining local LLM inference (Ollama), hybrid RAG search (BM25 + ChromaDB + reranker), and optional Google Workspace integration (Drive, Gmail, Docs, Sheets, Calendar, Tasks). All inference runs locally — no cloud API keys required.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running
- Required models pulled:
  ```bash
  ollama pull gemma3:4b  # RAG model
  ollama pull gemma3:4b       # Briefing model
  ollama pull nomic-embed-text   # Embeddings
  ollama pull gemma3:4b     # Reranker
  ```
- *(Optional)* Google Cloud project with OAuth 2.0 credentials for Workspace integration

## Setup

```bash
git clone <repo-url>
cd atlas_workspace

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env if you want different model names or search tuning values

# Start Ollama (WSL2: use service, not systemctl)
sudo service ollama start   # or: ollama serve

# Run
python atlas_agent.py
```

Open `http://localhost:8000/ui` in your browser.

## Google Workspace Integration

1. Create a project in [Google Cloud Console](https://console.cloud.google.com) and enable the Drive, Gmail, Docs, Sheets, Calendar, and Tasks APIs.
2. Create an OAuth 2.0 Desktop client and download the credentials as `credentials.json`.
3. Place `credentials.json` in the project root (it is gitignored — never commit it).
4. Set `ENABLE_GOOGLE=true` in `.env` (default).
5. On first run, a browser window will open for OAuth consent. After approval, `token.json` is saved automatically and reused on subsequent runs.

**Scopes requested:** Drive (read), Gmail (read), Docs (read), Sheets (read), Calendar (read), Tasks (read/write).

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + status |
| `/ui` | GET | Browser interface |
| `/ask` | POST | RAG query (JSON response) |
| `/ask/stream` | GET | RAG query (SSE stream) |
| `/ingest/local` | POST | Ingest files from `knowledge/` |
| `/ingest/doc` | POST | Ingest Google Doc, Sheet, or YouTube video |
| `/tools/execute` | POST | Calendar / Tasks / Sheets / Drive operations |
| `/briefing` | GET | Morning briefing synthesis |
| `/drive/files` | GET | List Google Drive files |
| `/gmail/recent` | GET | Recent email subjects |
| `/calendar/events` | GET | Upcoming calendar events |
| `/tasks` | GET | List Google Tasks |

## Adding Knowledge

Drop `.txt` or `.md` files into the `knowledge/` directory, then call `POST /ingest/local`. Documents are chunked, embedded, and stored in ChromaDB for retrieval.
