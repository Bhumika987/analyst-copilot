"""FastAPI app: bulk filing upload/indexing + cross-filing chat over indexed filings."""

import json
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import ingest_filing_stream, ingest_bulk_filings_stream, hydrate_from_postgres
from llm import answer_question, get_embedding, stream_answer
from retrieval import get_index, list_indexed_docs, cross_filing_hybrid_search
from config import get_embedding_model_name
import postgres_store as pg

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "indexes").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Analyst Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    doc_name: Optional[str] = "all"
    # Widened alongside llm.py's _context_chunk_limit ceiling (8 -> 10 for
    # comparison/calculation questions) -- retrieval has to hand over a pool
    # wider than that ceiling for it to mean anything; narrowing here would
    # silently make that widening a no-op.
    top_k: int = 12


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.on_event("startup")
async def startup_load_indexes():
    local_docs = set(list_indexed_docs())
    for doc_name in local_docs:
        get_index(doc_name)

    # Hydrate any filing Postgres knows about but this container's local
    # (possibly empty/ephemeral) disk doesn't -- see
    # ingest.hydrate_from_postgres for why this is what makes a freshly
    # deployed container see previously-indexed filings without a seeded
    # volume. No-op, not an error, when DATABASE_URL isn't set.
    if pg.is_configured():
        try:
            pg_docs = await pg.list_indexed_filings()
        except Exception as exc:
            print(f"Warning: could not reach Postgres on startup (local-only for this run): {exc}")
            pg_docs = []
        missing = [d for d in pg_docs if d not in local_docs]
        for doc_name in missing:
            try:
                await hydrate_from_postgres(doc_name)
            except Exception as exc:
                print(f"Warning: failed to hydrate '{doc_name}' from Postgres: {exc}")
        if missing:
            print(f"Hydrated {len(missing)} filing(s) from Postgres: {missing}")


@app.get("/api/health")
async def health():
    return {"status": "ok", "indexed_docs": len(list_indexed_docs()), "embedding_model": get_embedding_model_name()}


@app.get("/api/filings")
async def list_filings():
    result = []
    for doc_name in list_indexed_docs():
        idx = get_index(doc_name)
        if idx is None:
            continue
        pages = sorted({c.get("page_num") for c in idx.chunks if c.get("page_num") is not None})
        result.append({
            "doc_name": doc_name,
            "chunk_count": len(idx.chunks),
            "pages": {"min": pages[0], "max": pages[-1]} if pages else None,
            "metadata": idx.metadata,
            "embedding_model": get_embedding_model_name(),
            "vector_index_path": str(idx.vector_index_dir()),
        })
    return result


@app.get("/api/filings/{doc_name}/page/{page_num}")
async def get_filing_page(doc_name: str, page_num: int):
    """
    Every chunk from one page of one indexed filing, in original document
    order -- what the UI's clickable page citations fetch so an analyst can
    verify a cited page in context instead of taking a single quoted
    sentence on faith. Not a raw-.htm viewer (the parsed chunk text is what
    the LLM actually saw, which is the more useful thing to audit here).
    """
    idx = get_index(doc_name)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Filing '{doc_name}' is not indexed.")
    page_chunks = [c for c in idx.chunks if c.get("page_num") == page_num]
    if not page_chunks:
        raise HTTPException(status_code=404, detail=f"No content found for page {page_num} of '{doc_name}'.")
    return {
        "doc_name": doc_name,
        "page_num": page_num,
        "chunks": [
            {
                "section": c.get("section"),
                "subsection": c.get("subsection"),
                "table_title": c.get("table_title"),
                "chunk_type": c.get("chunk_type"),
                "text": c.get("text"),
            }
            for c in page_chunks
        ],
    }


@app.post("/api/filings/upload")
async def upload_filing(file: UploadFile = File(...), doc_name: Optional[str] = Form(None)):
    filename = file.filename or ""
    if not (filename.lower().endswith(".htm") or filename.lower().endswith(".html")):
        raise HTTPException(status_code=400, detail="Only .htm and .html files are accepted.")

    resolved_doc_name = doc_name or Path(filename).stem
    dest_path = UPLOAD_DIR / filename
    contents = await file.read()
    dest_path.write_bytes(contents)

    async def event_stream():
        async for event in ingest_filing_stream(str(dest_path), resolved_doc_name, use_embeddings=True):
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/filings/upload_bulk")
async def upload_bulk(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for bulk upload.")

    file_specs = []
    for file in files:
        filename = file.filename or ""
        if not (filename.lower().endswith(".htm") or filename.lower().endswith(".html")):
            continue
        dest_path = UPLOAD_DIR / filename
        contents = await file.read()
        dest_path.write_bytes(contents)
        doc_name = Path(filename).stem
        file_specs.append((str(dest_path), doc_name))

    if not file_specs:
        raise HTTPException(status_code=400, detail="None of the uploaded files were valid .htm or .html files.")

    async def event_stream():
        async for event in ingest_bulk_filings_stream(file_specs, use_embeddings=True):
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    target_doc = req.doc_name or "all"
    query_vector = get_embedding(req.question)
    results = cross_filing_hybrid_search(req.question, query_vector, doc_name=target_doc, top_k=req.top_k)

    async def event_stream():
        chunk_meta = [
            {
                "chunk_idx": c.get("chunk_idx"),
                "doc_name": c.get("doc_name"),
                "company": c.get("company"),
                "filing_type": c.get("filing_type"),
                "fiscal_year": c.get("fiscal_year"),
                "page_num": c.get("page_num"),
                "statement_type": c.get("statement_type"),
                "chunk_type": c.get("chunk_type"),
                "content_evidence_score": c.get("content_evidence_score"),
                "rerank_score": c.get("rerank_score"),
                "concept_matched": c.get("concept_matched"),
                "text_preview": (c.get("text") or "")[:200],
            }
            for c in results
        ]
        yield _sse({"type": "chunks", "chunks": chunk_meta})
        yield _sse({"type": "embedding_model", "embedding_model": get_embedding_model_name()})

        async for event in stream_answer(req.question, target_doc, results):
            if event.get("type") == "result":
                print("\n==================================================")
                print("[FINAL RESPONSE] (Endpoint: /api/chat)")
                print(f"found: {event.get('found')}")
                print(f"answer: {event.get('answer')}")
                print(f"confidence: {event.get('confidence')}")
                print(f"sources: {event.get('sources')}")
                dbg = event.get("debug_info", {})
                print(f"retrieval_status: {dbg.get('retrieval_status')}")
                if event.get("error"):
                    print(f"error: {event.get('error')}")
                print("==================================================\n")
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    target_doc = req.doc_name or "all"
    query_vector = get_embedding(req.question)
    results = cross_filing_hybrid_search(req.question, query_vector, doc_name=target_doc, top_k=req.top_k)

    result = await answer_question(req.question, target_doc, results)
    response = result.to_dict()
    response["embedding_model"] = get_embedding_model_name()
    response["chunks_used"] = [
        {
            "chunk_idx": c.get("chunk_idx"),
            "doc_name": c.get("doc_name"),
            "company": c.get("company"),
            "filing_type": c.get("filing_type"),
            "fiscal_year": c.get("fiscal_year"),
            "page_num": c.get("page_num"),
            "statement_type": c.get("statement_type"),
            "chunk_type": c.get("chunk_type"),
            "content_evidence_score": c.get("content_evidence_score"),
            "rerank_score": c.get("rerank_score"),
            "concept_matched": c.get("concept_matched"),
            "text_preview": (c.get("text") or "")[:200],
        }
        for c in results
    ]

    print("\n==================================================")
    print("[FINAL RESPONSE] (Endpoint: /api/chat/sync)")
    print(f"found: {response.get('found')}")
    print(f"answer: {response.get('answer')}")
    print(f"confidence: {response.get('confidence')}")
    print(f"sources: {response.get('sources')}")
    dbg = response.get("debug_info", {})
    print(f"retrieval_status: {dbg.get('retrieval_status')}")
    if response.get("error"):
        print(f"error: {response.get('error')}")
    print("==================================================\n")

    return response


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

