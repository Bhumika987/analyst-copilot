"""
One-time (re-runnable) migration: push every already-indexed local filing
into Postgres via backend/postgres_store.py.

Additive and non-destructive: reads from data/indexes/<doc_name>/ (the
local file-based store, untouched by this script) and upserts into
Postgres. Safe to re-run -- each filing's chunks are replaced wholesale on
re-upsert, so running this twice just re-syncs, it doesn't duplicate.

Requires:
  - DATABASE_URL set (see backend/postgres_store.py for the schema this
    creates on first run).
  - Each filing to have been indexed locally with EMBEDDING_MODEL=normal
    (the only dimension the pgvector column supports -- see
    postgres_store.EMBEDDING_DIM). Filings indexed under a different
    embedding model are skipped and reported, not silently dropped.

Usage:
  DATABASE_URL=postgresql://... python scripts/backfill_postgres.py
  DATABASE_URL=postgresql://... python scripts/backfill_postgres.py --doc AMD_2022_10K
"""

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import postgres_store as pg  # noqa: E402
from retrieval import INDEX_DIR, FilingIndex, get_embedding_model_name  # noqa: E402


async def backfill_one(doc_name: str) -> str:
    idx = FilingIndex.load(doc_name)
    if idx is None or not idx.is_indexed():
        return f"SKIP  {doc_name}  (no local BM25/chunks index found)"
    if idx.vectors is None or len(idx.vectors) != len(idx.chunks):
        return f"SKIP  {doc_name}  (no matching local embedding vectors -- run indexing with EMBEDDING_MODEL=normal first)"

    embedding_model = get_embedding_model_name()
    ok = await pg.save_filing(doc_name, idx.chunks, idx.vectors, embedding_model, metadata=idx.metadata)
    if not ok:
        return f"SKIP  {doc_name}  (embedding_model={embedding_model!r}, Postgres column is fixed at 384-dim/'normal' only)"
    return f"OK    {doc_name}  ({len(idx.chunks)} chunks)"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=str, default=None, help="Only backfill this doc_name.")
    args = parser.parse_args()

    if not pg.is_configured():
        print("DATABASE_URL is not set -- nothing to do. Set it to your Postgres connection string first.")
        return

    # Fail fast and loud on a bad connection string / unreachable server /
    # missing firewall rule, instead of silently SKIPping every filing (none
    # of which ever touch the connection -- see backfill_one) and only
    # discovering the real problem at the very end. Confirmed as a real
    # failure mode: a local machine connecting to an Azure Postgres server
    # configured for "Azure services only" access gets a DNS/connection
    # error, but only on the FIRST actual connection attempt -- which,
    # without this check, was the final summary query after the whole
    # (falsely clean-looking) "0/78 migrated" loop had already printed.
    try:
        await pg.ensure_schema()
    except Exception as exc:
        print(f"Could not connect to Postgres -- nothing was migrated.\n  {exc}\n")
        print(
            "Common causes: DATABASE_URL host/password typo'd rather than copied from the "
            "Azure portal's Connection strings blade, or the server's firewall only allows "
            "'Azure services' (which does NOT include your own machine) -- add your current "
            "client IP under the server's Networking > Firewall rules."
        )
        return

    embedding_model = get_embedding_model_name()
    if embedding_model != "normal":
        print(
            f"Warning: EMBEDDING_MODEL is currently {embedding_model!r}, but Postgres only "
            "stores 384-dim ('normal') vectors -- every filing below will be skipped unless "
            "you also have local 'normal' vectors cached (set EMBEDDING_MODEL=normal to use them).\n"
        )

    if args.doc:
        doc_names = [args.doc]
    else:
        if not INDEX_DIR.exists():
            print(f"No local index directory at {INDEX_DIR}")
            return
        doc_names = sorted(p.name for p in INDEX_DIR.iterdir() if p.is_dir())

    print(f"Backfilling {len(doc_names)} filing(s) into Postgres...\n")
    ok_count = 0
    for doc_name in doc_names:
        result = await backfill_one(doc_name)
        print(f"  {result}")
        if result.startswith("OK"):
            ok_count += 1

    print(f"\nDone. {ok_count}/{len(doc_names)} filings migrated.")
    remaining = await pg.list_indexed_filings()
    print(f"Postgres now has {len(remaining)} filing(s) indexed.")


if __name__ == "__main__":
    asyncio.run(main())
