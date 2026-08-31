"""
Optional networked persistence backend: Postgres + pgvector.

Additive, not a replacement -- the local file-based store
(`data/indexes/<doc_name>/{chunks.json,bm25.pkl,<model>/vectors.npy,...}`,
see FilingIndex.save/.load in retrieval.py) keeps working exactly as
before and stays the default. This module only activates when the
`DATABASE_URL` environment variable is set, which is the case for a
deployed instance whose container disk is ephemeral -- Postgres is what
lets a redeploy or restart keep every indexed filing without needing a
mounted volume seeded ahead of time.

Schema (created lazily, once, via ensure_schema()):
  filings(doc_name PK, embedding_model, embedding_dim, metadata JSONB, indexed_at)
  chunks(id, doc_name FK, chunk_index, page_num, section, subsection,
         statement_type, table_title, chunk_type, text, embedding vector(384),
         text_search tsvector GENERATED)

Embedding dimension is fixed at 384 (BAAI/bge-small-en-v1.5, this
project's "normal" EMBEDDING_MODEL default and the only one every retrieval
fix in this codebase has actually been verified against). A filing indexed
under a different embedding model can still be migrated -- see
scripts/backfill_postgres.py -- but pgvector columns need one fixed
dimension per table, so only 384-dim vectors can be stored here. Filings
indexed with a different-dimension model fall back to local-only storage;
that's a deliberate, documented limitation, not a bug.

Full-text search substitutes for the local BM25 index. This is a valid
substitution specifically because Reciprocal Rank Fusion (used both here
and in retrieval.py's local hybrid_search) only needs each retriever's
RANK ORDER to be reasonable, not an identical scoring formula between the
two -- Postgres's `ts_rank_cd` over a GIN-indexed tsvector and rank_bm25's
BM25 score are different formulas that still each produce a defensible
relevance ordering, which is all RRF fusion actually consumes.
"""

import json
import os
from typing import Dict, List, Optional

EMBEDDING_DIM = 384

_POOL = None


def is_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


async def _get_pool():
    global _POOL
    if _POOL is not None:
        return _POOL
    try:
        import asyncpg
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set but the 'asyncpg' package isn't installed: pip install asyncpg"
        ) from exc
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    _POOL = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    return _POOL


_SCHEMA_READY = False


async def ensure_schema():
    """Idempotent DDL -- safe to call on every startup."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filings (
                doc_name TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                metadata JSONB,
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                doc_name TEXT NOT NULL REFERENCES filings(doc_name) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_num INTEGER,
                section TEXT,
                subsection TEXT,
                statement_type TEXT,
                table_title TEXT,
                chunk_type TEXT,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
                UNIQUE (doc_name, chunk_index)
            );
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_name);")
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING GIN (text_search);")
        # ivfflat needs at least a handful of rows to build meaningfully;
        # harmless (just slower, not wrong) to create it early and let it
        # fill in as filings are backfilled.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_vec_idx ON chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )
    _SCHEMA_READY = True


def _vec_literal(vector) -> str:
    """Format a vector as the string pgvector's input parser accepts, cast in SQL with ::vector."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


async def save_filing(
    doc_name: str,
    chunks: List[Dict],
    vectors,
    embedding_model: str,
    metadata: Optional[Dict] = None,
) -> bool:
    """
    Upsert one filing's chunks + embeddings into Postgres.

    `vectors` is the same (N, 384) float array FilingIndex keeps in memory
    (see vectors.npy in the local store) -- row i's vector goes with
    chunks[i]. Returns False (no-op, not an error) if embedding_model isn't
    384-dim, since the vector column can't hold anything else.
    """
    if embedding_model != "normal":
        return False
    if vectors is None or len(vectors) != len(chunks):
        return False

    await ensure_schema()
    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO filings (doc_name, embedding_model, embedding_dim, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (doc_name) DO UPDATE
                    SET embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        metadata = EXCLUDED.metadata,
                        indexed_at = now();
                """,
                doc_name, embedding_model, EMBEDDING_DIM, json.dumps(metadata or {}),
            )
            # Replace this filing's chunks wholesale -- simpler and safer
            # than diffing when re-indexing the same doc_name, and this
            # only runs on upload/backfill, not per-query.
            await conn.execute("DELETE FROM chunks WHERE doc_name = $1;", doc_name)
            rows = []
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                rows.append((
                    doc_name,
                    i,
                    chunk.get("page_num"),
                    chunk.get("section"),
                    chunk.get("subsection"),
                    chunk.get("statement_type"),
                    chunk.get("table_title"),
                    chunk.get("chunk_type"),
                    chunk.get("text", ""),
                    _vec_literal(vec),
                ))
            await conn.executemany(
                """
                INSERT INTO chunks
                    (doc_name, chunk_index, page_num, section, subsection,
                     statement_type, table_title, chunk_type, text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector);
                """,
                rows,
            )
    return True


async def is_filing_indexed(doc_name: str) -> bool:
    if not is_configured():
        return False
    await ensure_schema()
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM filings WHERE doc_name = $1;", doc_name)
        return row is not None


async def list_indexed_filings() -> List[str]:
    if not is_configured():
        return []
    await ensure_schema()
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT doc_name FROM filings ORDER BY doc_name;")
        return [r["doc_name"] for r in rows]


async def load_filing(doc_name: str):
    """
    Fetch one filing's full chunk list + embeddings + metadata back out of
    Postgres, in chunk_index order -- everything FilingIndex needs to be
    reconstructed in memory. Used to hydrate a fresh container (empty local
    data/indexes/, ephemeral disk) from Postgres on startup, see
    ingest.hydrate_from_postgres(). Returns (chunks, vectors, metadata) or
    (None, None, None) if this filing isn't in Postgres.
    """
    if not is_configured():
        return None, None, None
    await ensure_schema()
    pool = await _get_pool()
    async with pool.acquire() as conn:
        filing_row = await conn.fetchrow("SELECT metadata FROM filings WHERE doc_name = $1;", doc_name)
        if filing_row is None:
            return None, None, None
        rows = await conn.fetch(
            """
            SELECT chunk_index, page_num, section, subsection, statement_type,
                   table_title, chunk_type, text, embedding
            FROM chunks WHERE doc_name = $1 ORDER BY chunk_index;
            """,
            doc_name,
        )
    import numpy as np
    chunks = []
    vectors = []
    for r in rows:
        chunks.append({
            "page_num": r["page_num"],
            "section": r["section"],
            "subsection": r["subsection"],
            "statement_type": r["statement_type"],
            "table_title": r["table_title"],
            "chunk_type": r["chunk_type"],
            "text": r["text"],
        })
        # asyncpg + pgvector returns the embedding as a string like
        # "[0.1,0.2,...]" without the `pgvector` client codec registered
        # (deliberately not added as a dependency just for this) -- parse
        # it back into floats by hand.
        emb_str = r["embedding"]
        if isinstance(emb_str, str):
            vectors.append([float(x) for x in emb_str.strip("[]").split(",")])
        else:
            vectors.append(list(emb_str))
    metadata = json.loads(filing_row["metadata"]) if filing_row["metadata"] else {}
    return chunks, (np.array(vectors, dtype=np.float32) if vectors else None), metadata


async def hybrid_search(doc_name: str, query_text: str, query_vector, top_k: int = 30) -> List[Dict]:
    """
    RRF fusion of Postgres full-text search (substitutes for local BM25)
    and pgvector cosine similarity (substitutes for local FAISS), scoped to
    one filing. Returns chunk dicts shaped like the local store's
    chunks.json entries so they slot into retrieval.py's existing
    deterministic_rerank/cross_encoder_rerank pipeline unchanged.
    """
    if not is_configured():
        return []
    await ensure_schema()
    pool = await _get_pool()
    vec_literal = _vec_literal(query_vector) if query_vector is not None else None

    async with pool.acquire() as conn:
        if vec_literal is not None:
            rows = await conn.fetch(
                """
                WITH fts AS (
                    SELECT id, row_number() OVER (
                        ORDER BY ts_rank_cd(text_search, plainto_tsquery('english', $2)) DESC
                    ) AS rnk
                    FROM chunks
                    WHERE doc_name = $1 AND text_search @@ plainto_tsquery('english', $2)
                    LIMIT 50
                ),
                vec AS (
                    SELECT id, row_number() OVER (ORDER BY embedding <=> $3::vector) AS rnk
                    FROM chunks
                    WHERE doc_name = $1
                    ORDER BY embedding <=> $3::vector
                    LIMIT 50
                )
                SELECT c.*,
                    COALESCE(1.0 / (60 + fts.rnk), 0) + COALESCE(1.0 / (60 + vec.rnk), 0) AS rrf_score
                FROM chunks c
                LEFT JOIN fts ON fts.id = c.id
                LEFT JOIN vec ON vec.id = c.id
                WHERE fts.id IS NOT NULL OR vec.id IS NOT NULL
                ORDER BY rrf_score DESC
                LIMIT $4;
                """,
                doc_name, query_text, vec_literal, top_k,
            )
        else:
            # No embeddable query vector (e.g. embedding service unavailable) --
            # fall back to full-text-only, same graceful degradation the
            # local hybrid_search does when dense retrieval is unavailable.
            rows = await conn.fetch(
                """
                SELECT c.*, ts_rank_cd(text_search, plainto_tsquery('english', $2)) AS rrf_score
                FROM chunks c
                WHERE doc_name = $1 AND text_search @@ plainto_tsquery('english', $2)
                ORDER BY rrf_score DESC
                LIMIT $3;
                """,
                doc_name, query_text, top_k,
            )

    results = []
    for r in rows:
        results.append({
            "doc_name": doc_name,
            "chunk_index": r["chunk_index"],
            "page_num": r["page_num"],
            "section": r["section"],
            "subsection": r["subsection"],
            "statement_type": r["statement_type"],
            "table_title": r["table_title"],
            "chunk_type": r["chunk_type"],
            "text": r["text"],
            "retrieval_score": float(r["rrf_score"]),
        })
    return results
