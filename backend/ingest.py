"""
Orchestrates: raw .htm filing -> parsed page-aware chunks -> saved hybrid index.

Exposes both a streaming version (for the /api/filings/upload SSE endpoint,
so the UI can show live progress on filings that can run for minutes) and a
plain async version (for the eval script, which doesn't need progress events).
"""

import asyncio
from typing import AsyncGenerator, Dict, List, Tuple

import numpy as np

from llm import get_embedding
from config import get_embedding_model_name
from filing_parser import parse_filing_to_window_chunks, extract_filing_metadata
from retrieval import FilingIndex, register_index

EMBED_BATCH_SIZE = 10
EMBED_BATCH_DELAY_SECONDS = 0.5


async def run_blocking(func, *args):
    """asyncio.to_thread compatibility for Python 3.8."""
    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(func, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


async def ingest_filing_stream(filepath: str, doc_name: str, use_embeddings: bool = True) -> AsyncGenerator[Dict, None]:
    try:
        yield {"type": "progress", "doc_name": doc_name, "message": "Parsing filing into page-aware chunks...", "progress": 0.0}
        chunks = await run_blocking(parse_filing_to_window_chunks, filepath)

        if not chunks:
            yield {"type": "error", "doc_name": doc_name, "message": "No extractable text found in this filing."}
            return

        yield {"type": "progress", "doc_name": doc_name, "message": f"Parsed {len(chunks)} chunks.", "progress": 0.15}

        metadata = extract_filing_metadata(filepath, doc_name, chunks)
        index = FilingIndex(doc_name, chunks, metadata=metadata)

        yield {"type": "progress", "doc_name": doc_name, "message": "Building BM25 index...", "progress": 0.20}
        await run_blocking(index.build_bm25)
        yield {"type": "progress", "doc_name": doc_name, "message": "BM25 index built.", "progress": 0.25}

        if use_embeddings:
            embedding_model = get_embedding_model_name()
            yield {"type": "progress", "doc_name": doc_name, "message": f"Generating {embedding_model} FAISS semantic embeddings...", "progress": 0.40}
            await run_blocking(index.build_bge_faiss)
            yield {"type": "progress", "doc_name": doc_name, "message": f"{embedding_model} FAISS semantic vector index built.", "progress": 0.90}
        else:
            yield {"type": "progress", "doc_name": doc_name, "message": "Skipping embeddings (BM25-only mode).", "progress": 0.90}

        yield {"type": "progress", "doc_name": doc_name, "message": "Saving index to disk...", "progress": 0.95}
        await run_blocking(index.save)
        register_index(doc_name, index)

        yield {"type": "progress", "doc_name": doc_name, "message": "Done.", "progress": 1.0}
        yield {"type": "complete", "doc_name": doc_name, "message": f"Indexed '{doc_name}' with {len(chunks)} chunks.", "metadata": metadata}

    except Exception as exc:
        yield {"type": "error", "doc_name": doc_name, "message": f"Ingestion failed: {exc}"}


async def ingest_bulk_filings_stream(file_specs: List[Tuple[str, str]], use_embeddings: bool = True) -> AsyncGenerator[Dict, None]:
    """
    Ingest multiple filings sequentially while streaming progress events.
    file_specs: List of (filepath, doc_name) tuples.
    """
    total_files = len(file_specs)
    completed_files = 0

    yield {"type": "bulk_start", "total_files": total_files, "message": f"Starting bulk ingestion of {total_files} filings."}

    for idx, (filepath, doc_name) in enumerate(file_specs, start=1):
        yield {"type": "file_start", "doc_name": doc_name, "file_num": idx, "total_files": total_files}
        async for event in ingest_filing_stream(filepath, doc_name, use_embeddings=use_embeddings):
            if event["type"] == "progress":
                overall_progress = ((idx - 1) + event.get("progress", 0.0)) / total_files
                event["overall_progress"] = overall_progress
            yield event
            if event["type"] == "complete":
                completed_files += 1

    yield {
        "type": "bulk_complete",
        "total_files": total_files,
        "completed_files": completed_files,
        "message": f"Bulk ingestion finished. Ingested {completed_files}/{total_files} filings.",
    }


async def ingest_filing(filepath: str, doc_name: str, use_embeddings: bool = True) -> FilingIndex:
    """Non-streaming ingest, used by the eval script."""
    chunks = await run_blocking(parse_filing_to_window_chunks, filepath)
    metadata = extract_filing_metadata(filepath, doc_name, chunks)

    index = FilingIndex(doc_name, chunks, metadata=metadata)
    await run_blocking(index.build_bm25)

    if use_embeddings and chunks:
        await run_blocking(index.build_bge_faiss)

    await run_blocking(index.save)
    register_index(doc_name, index)
    return index


