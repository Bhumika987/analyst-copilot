"""
Orchestrates: raw .htm filing -> parsed page-aware chunks -> saved hybrid index.

Exposes both a streaming version (for the /api/filings/upload SSE endpoint,
so the UI can show live progress on filings that can run for minutes) and a
plain async version (for the eval script, which doesn't need progress events).
"""

import asyncio
from typing import AsyncGenerator, Dict

import numpy as np

from llm import get_embedding
from parser import parse_filing_to_window_chunks
from retrieval import FilingIndex, register_index

EMBED_BATCH_SIZE = 10
EMBED_BATCH_DELAY_SECONDS = 0.5


async def ingest_filing_stream(filepath: str, doc_name: str, use_embeddings: bool = True) -> AsyncGenerator[Dict, None]:
    try:
        yield {"type": "progress", "message": "Parsing filing into page-aware chunks...", "progress": 0.0}
        chunks = await asyncio.to_thread(parse_filing_to_window_chunks, filepath)

        if not chunks:
            yield {"type": "error", "message": "No extractable text found in this filing."}
            return

        yield {"type": "progress", "message": f"Parsed {len(chunks)} chunks.", "progress": 0.15}

        index = FilingIndex(doc_name, chunks)

        yield {"type": "progress", "message": "Building BM25 index...", "progress": 0.20}
        await asyncio.to_thread(index.build_bm25)
        yield {"type": "progress", "message": "BM25 index built.", "progress": 0.25}

        if use_embeddings:
            vectors = []
            total = len(chunks)
            for start in range(0, total, EMBED_BATCH_SIZE):
                batch = chunks[start:start + EMBED_BATCH_SIZE]
                batch_vecs = [get_embedding(c["text"]) for c in batch]
                vectors.extend(batch_vecs)

                frac_done = min(1.0, (start + len(batch)) / total)
                progress = 0.25 + frac_done * (0.90 - 0.25)
                yield {
                    "type": "progress",
                    "message": f"Embedding chunks ({start + len(batch)}/{total})...",
                    "progress": progress,
                }
                if start + EMBED_BATCH_SIZE < total:
                    await asyncio.sleep(EMBED_BATCH_DELAY_SECONDS)

            vectors_arr = np.array(vectors, dtype="float32")
            index.set_vectors(vectors_arr)
            yield {"type": "progress", "message": "Vectors embedded.", "progress": 0.90}
        else:
            yield {"type": "progress", "message": "Skipping embeddings (BM25-only mode).", "progress": 0.90}

        yield {"type": "progress", "message": "Saving index to disk...", "progress": 0.95}
        await asyncio.to_thread(index.save)
        register_index(doc_name, index)

        yield {"type": "progress", "message": "Done.", "progress": 1.0}
        yield {"type": "complete", "message": f"Indexed '{doc_name}' with {len(chunks)} chunks."}

    except Exception as exc:
        yield {"type": "error", "message": f"Ingestion failed: {exc}"}


async def ingest_filing(filepath: str, doc_name: str, use_embeddings: bool = True) -> FilingIndex:
    """Non-streaming ingest, used by the eval script."""
    chunks = await asyncio.to_thread(parse_filing_to_window_chunks, filepath)

    index = FilingIndex(doc_name, chunks)
    await asyncio.to_thread(index.build_bm25)

    if use_embeddings and chunks:
        vectors = [get_embedding(c["text"]) for c in chunks]
        index.set_vectors(np.array(vectors, dtype="float32"))

    await asyncio.to_thread(index.save)
    register_index(doc_name, index)
    return index
