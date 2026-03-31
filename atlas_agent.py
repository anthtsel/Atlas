"""
ATLAS Workspace Agent — v3.0
  Brain     : Ollama qwen2.5-coder:7b (RAG) + qwen2.5:14b (Briefing)
  Reranker  : qwen3-reranker via Ollama
  Search    : BM25 (rank-bm25) + ChromaDB → Reranker hybrid pipeline
  Memory    : ChromaDB persistent vector store
  Senses    : Google Drive, Gmail, Docs, Sheets, Calendar, Tasks
  Ingest    : MarkItDown (Docs/Sheets/PDFs) + youtube-transcript-api
  API       : FastAPI + SSE streaming + BackgroundTasks
  UI        : /ui route → atlas_workspace.html
"""

import os, re, logging, asyncio, json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncIterator
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from rank_bm25 import BM25Okapi

# Google APIs
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
KNOWLEDGE_DIR    = BASE_DIR / "knowledge"
CHROMA_DIR       = BASE_DIR / "chroma_db"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE       = BASE_DIR / "token.json"
FRONTEND_FILE    = BASE_DIR / "atlas_workspace.html"

# ── Config ─────────────────────────────────────────────────────────────────────
RAG_MODEL        = os.getenv("RAG_MODEL",        "qwen2.5-coder:7b")
BRIEFING_MODEL   = os.getenv("BRIEFING_MODEL",   "qwen2.5:14b")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "nomic-embed-text")
RERANK_MODEL     = os.getenv("RERANK_MODEL",     "qwen3-reranker")
CHROMA_COLLECTION= os.getenv("CHROMA_COLLECTION","workspace_memory")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST",      "http://localhost:11434")
BM25_TOP_K       = int(os.getenv("BM25_TOP_K",   "20"))
RERANK_TOP_K     = int(os.getenv("RERANK_TOP_K", "3"))
VECTOR_TOP_K     = int(os.getenv("VECTOR_TOP_K", "10"))

# ── Google Scopes ──────────────────────────────────────────────────────────────
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/tasks",                  # read + write for tasks
]

KNOWLEDGE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question: str
    n_results: int = RERANK_TOP_K

class AskResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    search_trace: dict        # exposes the hybrid search steps to the UI

class IngestDocRequest(BaseModel):
    doc_id: str
    doc_type: str = "gdoc"   # gdoc | gsheet | youtube

class ToolExecuteRequest(BaseModel):
    tool: str                 # google_calendar | google_tasks | google_sheets | google_docs
    action: str               # create_event | create_task | read_range | append_row ...
    data: dict

class StatusResponse(BaseModel):
    status: str
    rag_model: str
    briefing_model: str
    embed_model: str
    rerank_model: str
    collection: str
    doc_count: int
    google_connected: bool
    google_services: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH (BM25 + ChromaDB → Qwen3-Reranker)
# ══════════════════════════════════════════════════════════════════════════════

class HybridSearch:
    """
    3-stage retrieval pipeline:
      Stage 1 — BM25 keyword scoring over all stored documents
      Stage 2 — ChromaDB vector search
      Stage 3 — Qwen3-Reranker selects final top-k from merged candidates
    """

    def __init__(self, collection):
        self.collection = collection
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._refresh_bm25()

    def _refresh_bm25(self):
        """Rebuild the BM25 index from all documents in ChromaDB."""
        try:
            results = self.collection.get()
            docs = results.get("documents", [])
            ids  = results.get("ids", [])
            if docs:
                self._corpus     = docs
                self._corpus_ids = ids
                tokenized        = [d.lower().split() for d in docs]
                self._bm25       = BM25Okapi(tokenized)
                log.info("BM25 index rebuilt — %d documents", len(docs))
        except Exception as e:
            log.warning("BM25 refresh failed: %s", e)

    def search(self, query: str, ollama_client) -> tuple[list[str], dict]:
        """
        Returns (top_chunks, trace_dict) where trace_dict exposes
        each pipeline step for the frontend thought-trace visualization.
        """
        trace = {
            "step1_bm25":    {"candidates": 0, "top_scores": []},
            "step2_vector":  {"candidates": 0},
            "step3_reranker":{"selected": 0, "model": RERANK_MODEL},
        }

        if not self._corpus:
            self._refresh_bm25()
        if not self._corpus:
            return [], trace

        # ── Stage 1: BM25 ─────────────────────────────────────────────────────
        bm25_chunks = []
        if self._bm25:
            tokens     = query.lower().split()
            scores     = self._bm25.get_scores(tokens)
            top_idxs   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]
            bm25_chunks= [self._corpus[i] for i in top_idxs if scores[i] > 0]
            trace["step1_bm25"]["candidates"] = len(bm25_chunks)
            trace["step1_bm25"]["top_scores"] = [round(float(scores[i]), 3) for i in top_idxs[:5]]

        # ── Stage 2: ChromaDB vector search ───────────────────────────────────
        try:
            vec_results   = self.collection.query(query_texts=[query], n_results=min(VECTOR_TOP_K, len(self._corpus)))
            vector_chunks = vec_results["documents"][0] if vec_results["documents"] else []
            trace["step2_vector"]["candidates"] = len(vector_chunks)
        except Exception:
            vector_chunks = []

        # ── Merge & deduplicate ────────────────────────────────────────────────
        seen   = set()
        merged = []
        for chunk in bm25_chunks + vector_chunks:
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        if not merged:
            return [], trace

        # ── Stage 3: Qwen3-Reranker ───────────────────────────────────────────
        try:
            rerank_prompt = (
                f"Given the query: '{query}'\n\n"
                f"Rank these {len(merged)} passages from most to least relevant. "
                f"Return ONLY a JSON array of the top {RERANK_TOP_K} passage indices "
                f"(0-based), like: [2, 0, 5]\n\n"
                + "\n\n".join([f"[{i}] {c[:300]}" for i, c in enumerate(merged)])
            )
            resp    = ollama_client.chat(
                model=RERANK_MODEL,
                messages=[{"role": "user", "content": rerank_prompt}],
            )
            raw     = resp.message.content.strip()
            indices = [int(x) for x in re.findall(r'\d+', raw) if int(x) < len(merged)][:RERANK_TOP_K]
            final   = [merged[i] for i in indices] if indices else merged[:RERANK_TOP_K]
            trace["step3_reranker"]["selected"] = len(final)
        except Exception as e:
            log.warning("Reranker failed, using merged top-%d: %s", RERANK_TOP_K, e)
            final = merged[:RERANK_TOP_K]
            trace["step3_reranker"]["selected"] = len(final)

        return final, trace


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL MEMORY (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class LocalMemory:
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef              = OllamaEmbeddingFunction(url=OLLAMA_HOST, model_name=EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION, embedding_function=ef)
        self.search     = HybridSearch(self.collection)
        log.info("ChromaDB ready — %d docs in '%s'", self.collection.count(), CHROMA_COLLECTION)

    def store(self, text: str, doc_id: str, metadata: Optional[dict] = None) -> int:
        chunks = self._chunk(text)
        self.collection.upsert(
            documents=chunks,
            ids=[f"{doc_id}_chunk{i}" for i in range(len(chunks))],
            metadatas=[{**(metadata or {}), "source": doc_id, "chunk": i}
                       for i in range(len(chunks))],
        )
        self.search._refresh_bm25()
        log.info("Stored %d chunks from '%s'", len(chunks), doc_id)
        return len(chunks)

    def retrieve(self, query: str, ollama_client) -> tuple[list[str], dict]:
        return self.search.search(query, ollama_client)

    def count(self) -> int:
        return self.collection.count()

    def ingest_local_knowledge(self):
        files = list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md"))
        for fp in files:
            self.store(fp.read_text(encoding="utf-8", errors="ignore"),
                       doc_id=fp.stem, metadata={"file": fp.name, "source_type": "local"})
        if files:
            log.info("Ingested %d local files", len(files))

    @staticmethod
    def _chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = [" ".join(words[i:i+size])
                  for i in range(0, max(1, len(words) - overlap), step)]
        return chunks or [text]


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE CONNECTOR (Drive, Gmail, Docs, Sheets, Calendar, Tasks)
# ══════════════════════════════════════════════════════════════════════════════

class GoogleConnector:
    def __init__(self):
        self.creds    = self._authenticate()
        self.drive    = build("drive",    "v3",  credentials=self.creds)
        self.gmail    = build("gmail",    "v1",  credentials=self.creds)
        self.docs     = build("docs",     "v1",  credentials=self.creds)
        self.sheets   = build("sheets",   "v4",  credentials=self.creds)
        self.calendar = build("calendar", "v3",  credentials=self.creds)
        self.tasks    = build("tasks",    "v1",  credentials=self.creds)
        log.info("Google APIs authenticated (Drive, Gmail, Docs, Sheets, Calendar, Tasks)")

    def _authenticate(self) -> Credentials:
        creds = None
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), GOOGLE_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(f"Missing {CREDENTIALS_FILE}. See setup guide Phase 1.")
                flow  = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), GOOGLE_SCOPES)
                creds = flow.run_local_server(port=0)
            TOKEN_FILE.write_text(creds.to_json())
        return creds

    # ── Drive ──────────────────────────────────────────────────────────────────
    def get_doc_text(self, doc_id: str) -> str:
        raw = self.drive.files().export(fileId=doc_id, mimeType="text/plain").execute()
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw

    def list_files(self, query: str = "", max_results: int = 20) -> list[dict]:
        q    = f"name contains '{query}'" if query else ""
        resp = self.drive.files().list(
            q=q, pageSize=max_results,
            fields="files(id,name,mimeType,modifiedTime)").execute()
        return resp.get("files", [])

    # ── Gmail ──────────────────────────────────────────────────────────────────
    def recent_email_subjects(self, n: int = 5) -> list[str]:
        msgs     = self.gmail.users().messages().list(userId="me", maxResults=n).execute()
        subjects = []
        for m in msgs.get("messages", []):
            detail = self.gmail.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["Subject"]).execute()
            for h in detail["payload"]["headers"]:
                if h["name"] == "Subject":
                    subjects.append(h["value"])
        return subjects

    # ── Docs ───────────────────────────────────────────────────────────────────
    def get_doc_content(self, doc_id: str) -> str:
        """Extract plain text from a Google Doc via the Docs API."""
        doc   = self.docs.documents().get(documentId=doc_id).execute()
        texts = []
        for block in doc.get("body", {}).get("content", []):
            for elem in block.get("paragraph", {}).get("elements", []):
                t = elem.get("textRun", {}).get("content", "")
                if t:
                    texts.append(t)
        return "".join(texts)

    # ── Sheets ─────────────────────────────────────────────────────────────────
    def get_sheet_values(self, spreadsheet_id: str, range_: str = "Sheet1") -> list[list]:
        result = self.sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=range_).execute()
        return result.get("values", [])

    def append_sheet_row(self, spreadsheet_id: str, range_: str, values: list) -> dict:
        body   = {"values": [values]}
        result = self.sheets.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=range_,
            valueInputOption="USER_ENTERED", body=body).execute()
        return result

    # ── Calendar ───────────────────────────────────────────────────────────────
    def get_upcoming_events(self, hours: int = 24, max_results: int = 10) -> list[dict]:
        now   = datetime.now(timezone.utc)
        end   = now + timedelta(hours=hours)
        resp  = self.calendar.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = []
        for e in resp.get("items", []):
            start = e.get("start", {})
            events.append({
                "id":       e.get("id"),
                "summary":  e.get("summary", "No title"),
                "start":    start.get("dateTime", start.get("date", "")),
                "location": e.get("location", ""),
                "description": e.get("description", ""),
            })
        return events

    def create_calendar_event(self, summary: str, start: str, end: str,
                               description: str = "", location: str = "") -> dict:
        event = {
            "summary":     summary,
            "description": description,
            "location":    location,
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end":   {"dateTime": end,   "timeZone": "UTC"},
        }
        return self.calendar.events().insert(calendarId="primary", body=event).execute()

    # ── Tasks ──────────────────────────────────────────────────────────────────
    def get_tasks(self, max_results: int = 20) -> list[dict]:
        lists = self.tasks.tasklists().list().execute().get("items", [])
        if not lists:
            return []
        tasklist_id = lists[0]["id"]
        resp        = self.tasks.tasks().list(
            tasklist=tasklist_id, maxResults=max_results,
            showCompleted=False).execute()
        return [{"id": t["id"], "title": t.get("title",""),
                 "due": t.get("due",""), "notes": t.get("notes","")}
                for t in resp.get("items", [])]

    def create_task(self, title: str, notes: str = "", due: str = "") -> dict:
        lists       = self.tasks.tasklists().list().execute().get("items", [])
        tasklist_id = lists[0]["id"] if lists else "@default"
        body        = {"title": title, "notes": notes}
        if due:
            body["due"] = due
        return self.tasks.tasks().insert(tasklist=tasklist_id, body=body).execute()

    def complete_task(self, task_id: str) -> dict:
        lists       = self.tasks.tasklists().list().execute().get("items", [])
        tasklist_id = lists[0]["id"] if lists else "@default"
        task        = self.tasks.tasks().get(tasklist=tasklist_id, task=task_id).execute()
        task["status"] = "completed"
        return self.tasks.tasks().update(
            tasklist=tasklist_id, task=task_id, body=task).execute()


# ══════════════════════════════════════════════════════════════════════════════
# REASONING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    def __init__(self, memory: LocalMemory):
        self.memory = memory
        self.client = ollama.Client(host=OLLAMA_HOST)

    def ask(self, question: str, n_results: int = RERANK_TOP_K) -> tuple[str, int, dict]:
        chunks, trace = self.memory.retrieve(question, self.client)
        context       = "\n\n---\n\n".join(chunks) if chunks else "No context found."
        prompt        = (
            "You are a helpful AI assistant. Answer using ONLY the context below. "
            "Format your response using Markdown — use headers, lists, and tables where appropriate. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        try:
            resp = self.client.chat(
                model=RAG_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            log.error("Ollama RAG call failed: %s", e)
            raise HTTPException(status_code=503, detail="LLM unavailable. Ensure Ollama is running.")
        return resp.message.content.strip(), len(chunks), trace

    async def ask_stream(self, question: str) -> AsyncIterator[str]:
        """Streaming version — yields SSE-formatted chunks."""
        chunks, trace = self.memory.retrieve(question, self.client)
        context       = "\n\n---\n\n".join(chunks) if chunks else "No context found."
        prompt        = (
            "You are a helpful AI assistant. Answer using ONLY the context below. "
            "Use Markdown formatting — headers, lists, and tables where helpful. "
            "If the answer is not in the context, say so.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        )
        # Yield trace first so UI can show thought-trace immediately
        yield f"data: {json.dumps({'type': 'trace', 'trace': trace, 'chunks': len(chunks)})}\n\n"

        try:
            stream = self.client.chat(
                model=RAG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for part in stream:
                token = part.message.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            log.error("Ollama streaming failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'LLM unavailable.'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    def synthesize_briefing(self, events: list, tasks: list, notes: list) -> str:
        """Use the more powerful 14b model to synthesize the morning briefing."""
        events_txt = "\n".join([f"- {e['start'][:16]} | {e['summary']}" for e in events]) or "No events."
        tasks_txt  = "\n".join([f"- {t['title']}" + (f" (due: {t['due'][:10]})" if t.get('due') else "") for t in tasks]) or "No tasks."
        notes_txt  = "\n\n".join(notes[:3]) if notes else "No recent project notes."

        prompt = f"""You are a personal AI assistant preparing a morning briefing.
Synthesize the following into a structured, actionable Plan of Action for today.
Use Markdown with clear sections: ## Today's Schedule, ## Open Tasks, ## Project Context, ## Recommended Actions.

CALENDAR (next 24h):
{events_txt}

OPEN TASKS:
{tasks_txt}

RECENT PROJECT NOTES FROM MEMORY:
{notes_txt}

Provide a concise, prioritized briefing with specific recommended actions."""

        resp = self.client.chat(
            model=BRIEFING_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ingest_youtube_transcript(video_id: str, memory: LocalMemory) -> int:
    """Fetch YouTube transcript and store in ChromaDB."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text       = " ".join([t["text"] for t in transcript])
        max_chars  = int(os.getenv("YOUTUBE_MAX_TRANSCRIPT_CHARS", "50000"))
        text       = text[:max_chars]
        return memory.store(text, doc_id=f"youtube_{video_id}",
                            metadata={"source_type": "youtube", "video_id": video_id})
    except Exception as e:
        raise RuntimeError(f"YouTube transcript failed: {e}")

def ingest_google_sheet(spreadsheet_id: str, google: "GoogleConnector",
                         memory: LocalMemory) -> int:
    """Fetch sheet values, convert to markdown table, store in ChromaDB."""
    rows = google.get_sheet_values(spreadsheet_id)
    if not rows:
        return 0
    # Convert to markdown table
    header = "| " + " | ".join(str(c) for c in rows[0]) + " |"
    divider= "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body   = "\n".join(["| " + " | ".join(str(c) for c in row) + " |"
                        for row in rows[1:]])
    text   = f"{header}\n{divider}\n{body}"
    return memory.store(text, doc_id=f"gsheet_{spreadsheet_id}",
                        metadata={"source_type": "gsheet", "spreadsheet_id": spreadsheet_id})


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID AGENT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class HybridAgent:
    def __init__(self, use_google: bool = True):
        self.memory  = LocalMemory()
        self.memory.ingest_local_knowledge()
        self.engine  = ReasoningEngine(self.memory)
        self.google: Optional[GoogleConnector] = None
        if use_google:
            try:
                self.google = GoogleConnector()
            except FileNotFoundError as e:
                log.warning("Google disabled — %s", e)

    @property
    def google_connected(self) -> bool:
        return self.google is not None

    @property
    def google_services(self) -> list[str]:
        if not self.google_connected:
            return []
        return ["drive", "gmail", "docs", "sheets", "calendar", "tasks"]


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

agent: Optional[HybridAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = HybridAgent(use_google=os.getenv("ENABLE_GOOGLE", "true").lower() == "true")
    yield

app = FastAPI(
    title="ATLAS Workspace Agent",
    description="Agentic AI workspace: Hybrid RAG + Google Workspace + Streaming",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── UI ─────────────────────────────────────────────────────────────────────────
@app.get("/ui")
async def frontend():
    return FileResponse(str(FRONTEND_FILE))


# ── Status ─────────────────────────────────────────────────────────────────────
@app.get("/", response_model=StatusResponse)
async def status():
    return StatusResponse(
        status="running",
        rag_model=RAG_MODEL,
        briefing_model=BRIEFING_MODEL,
        embed_model=EMBED_MODEL,
        rerank_model=RERANK_MODEL,
        collection=CHROMA_COLLECTION,
        doc_count=agent.memory.count(),
        google_connected=agent.google_connected,
        google_services=agent.google_services,
    )


# ── RAG Ask (standard) ─────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    answer, n, trace = agent.engine.ask(req.question, req.n_results)
    return AskResponse(question=req.question, answer=answer,
                       chunks_used=n, search_trace=trace)


# ── RAG Ask (streaming SSE) ────────────────────────────────────────────────────
@app.get("/ask/stream")
async def ask_stream(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    return StreamingResponse(
        agent.engine.ask_stream(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Ingest Local ───────────────────────────────────────────────────────────────
@app.post("/ingest/local")
async def ingest_local(background_tasks: BackgroundTasks):
    background_tasks.add_task(agent.memory.ingest_local_knowledge)
    return {"status": "ingestion started", "doc_count": agent.memory.count()}


# ── Ingest Google Doc / Sheet / YouTube ───────────────────────────────────────
@app.post("/ingest/doc")
async def ingest_doc(req: IngestDocRequest, background_tasks: BackgroundTasks):
    if not agent.google_connected and req.doc_type != "youtube":
        raise HTTPException(status_code=503, detail="Google not connected.")

    def _ingest():
        if req.doc_type == "gdoc":
            text = agent.google.get_doc_text(req.doc_id)
            agent.memory.store(text, doc_id=f"gdoc_{req.doc_id}",
                               metadata={"source_type": "gdoc", "doc_id": req.doc_id})
        elif req.doc_type == "gsheet":
            ingest_google_sheet(req.doc_id, agent.google, agent.memory)
        elif req.doc_type == "youtube":
            ingest_youtube_transcript(req.doc_id, agent.memory)

    background_tasks.add_task(_ingest)
    return {"status": "ingestion started", "doc_id": req.doc_id, "type": req.doc_type}


# ── Tool Execute Router ────────────────────────────────────────────────────────
@app.post("/tools/execute")
async def execute_tool(req: ToolExecuteRequest):
    """
    Unified tool execution endpoint.
    Returns a 'proposed_action' payload for confirmation cards in the UI,
    or executes directly for read operations.
    """
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")

    g = agent.google

    # ── Google Calendar ──────────────────────────────────────────────────────
    if req.tool == "google_calendar":
        if req.action == "list_events":
            hours  = req.data.get("hours", 24)
            events = g.get_upcoming_events(hours=hours)
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": events}
        elif req.action == "create_event":
            result = g.create_calendar_event(
                summary     = req.data.get("summary", "New Event"),
                start       = req.data["start"],
                end         = req.data["end"],
                description = req.data.get("description", ""),
                location    = req.data.get("location", ""),
            )
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": result}

    # ── Google Tasks ─────────────────────────────────────────────────────────
    elif req.tool == "google_tasks":
        if req.action == "list_tasks":
            return {"status": "ok", "tool": req.tool, "action": req.action,
                    "result": g.get_tasks(req.data.get("max_results", 20))}
        elif req.action == "create_task":
            result = g.create_task(
                title = req.data.get("title", "New Task"),
                notes = req.data.get("notes", ""),
                due   = req.data.get("due", ""),
            )
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": result}
        elif req.action == "complete_task":
            result = g.complete_task(req.data["task_id"])
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": result}

    # ── Google Sheets ────────────────────────────────────────────────────────
    elif req.tool == "google_sheets":
        if req.action == "read_range":
            rows = g.get_sheet_values(req.data["spreadsheet_id"],
                                       req.data.get("range", "Sheet1"))
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": rows}
        elif req.action == "append_row":
            result = g.append_sheet_row(
                req.data["spreadsheet_id"],
                req.data.get("range", "Sheet1"),
                req.data.get("values", []),
            )
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": result}

    # ── Google Drive ─────────────────────────────────────────────────────────
    elif req.tool == "google_drive":
        if req.action == "list_files":
            files = g.list_files(req.data.get("query",""), req.data.get("max_results", 20))
            return {"status": "ok", "tool": req.tool, "action": req.action, "result": files}

    raise HTTPException(status_code=400, detail=f"Unknown tool/action: {req.tool}/{req.action}")


# ── Morning Briefing ───────────────────────────────────────────────────────────
@app.get("/briefing")
async def morning_briefing():
    """
    Orchestrates: Calendar (24h) + Tasks + ChromaDB project notes
    → qwen2.5:14b synthesizes into a structured Plan of Action.
    """
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")

    # Fetch in parallel using asyncio.gather with thread pool
    loop = asyncio.get_event_loop()
    events, tasks = await asyncio.gather(
        loop.run_in_executor(None, lambda: agent.google.get_upcoming_events(24)),
        loop.run_in_executor(None, agent.google.get_tasks),
    )

    # Query ChromaDB for recent project notes
    notes, trace = agent.memory.retrieve("project notes tasks priorities", agent.engine.client)

    # Synthesize with 14b model
    briefing = await loop.run_in_executor(
        None, agent.engine.synthesize_briefing, events, tasks, notes)

    return {
        "status":   "ok",
        "briefing": briefing,
        "data": {
            "events": events,
            "tasks":  tasks,
            "notes_retrieved": len(notes),
        },
    }


# ── Google Drive file listing ──────────────────────────────────────────────────
@app.get("/drive/files")
async def list_drive_files(query: str = "", max_results: int = 20):
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")
    return {"files": agent.google.list_files(query, max_results)}


# ── Gmail ──────────────────────────────────────────────────────────────────────
@app.get("/gmail/recent")
async def recent_emails(n: int = 5):
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")
    return {"subjects": agent.google.recent_email_subjects(n)}


# ── Calendar ───────────────────────────────────────────────────────────────────
@app.get("/calendar/events")
async def calendar_events(hours: int = 24):
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")
    return {"events": agent.google.get_upcoming_events(hours)}


# ── Tasks ──────────────────────────────────────────────────────────────────────
@app.get("/tasks")
async def get_tasks(max_results: int = 20):
    if not agent.google_connected:
        raise HTTPException(status_code=503, detail="Google not connected.")
    return {"tasks": agent.google.get_tasks(max_results)}


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("atlas_agent:app", host="0.0.0.0", port=8001, reload=True)
