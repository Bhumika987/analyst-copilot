"""FastAPI app: filing upload/indexing + chat over indexed filings."""

import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import ingest_filing_stream
from llm import answer_question, get_embedding, stream_answer
from retrieval import get_index, list_indexed_docs

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
    doc_name: str
    top_k: int = 5


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.on_event("startup")
async def startup_load_indexes():
    for doc_name in list_indexed_docs():
        get_index(doc_name)


@app.get("/api/health")
async def health():
    return {"status": "ok", "indexed_docs": len(list_indexed_docs())}


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
        })
    return result


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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    index = get_index(req.doc_name)
    if index is None or not index.is_indexed():
        raise HTTPException(status_code=404, detail=f"Filing '{req.doc_name}' is not indexed.")

    query_vector = get_embedding(req.question)
    results = index.hybrid_search(req.question, query_vector, top_k=req.top_k)

    async def event_stream():
        chunk_meta = [
            {
                "page_num": c.get("page_num"),
                "chunk_type": c.get("chunk_type"),
                "retrieval_score": c.get("retrieval_score"),
                "text_preview": (c.get("text") or "")[:200],
            }
            for c in results
        ]
        yield _sse({"type": "chunks", "chunks": chunk_meta})

        async for event in stream_answer(req.question, req.doc_name, results):
            yield _sse(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    index = get_index(req.doc_name)
    if index is None or not index.is_indexed():
        raise HTTPException(status_code=404, detail=f"Filing '{req.doc_name}' is not indexed.")

    query_vector = get_embedding(req.question)
    results = index.hybrid_search(req.question, query_vector, top_k=req.top_k)

    result = await answer_question(req.question, req.doc_name, results)
    response = result.to_dict()
    response["chunks_used"] = [
        {"page_num": c.get("page_num"), "retrieval_score": c.get("retrieval_score")}
        for c in results
    ]
    return response


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
