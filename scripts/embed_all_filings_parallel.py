"""
Parallel version of embed_all_filings.py -- same behavior (force re-parse +
re-embed every filing with the current filing_parser.py/embedding model),
but spread across a process pool instead of one filing at a time.

Filings are fully independent of each other (separate chunks.json/FAISS
index per doc_name), so this parallelizes cleanly: each worker process
loads its own copy of the embedding model once and processes a share of
the filing list. On a CPU-only machine this is the realistic way to speed
up a from-scratch re-embed -- there's no GPU here to hand the work to
instead (confirmed: torch reports cuda unavailable).

Run from the project root:
    python scripts/embed_all_filings_parallel.py --workers 6
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\Sakshi Sinha\Downloads\analyst-copilot-data 1\analyst-copilot-data\filings"
)
DEFAULT_UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
FILING_EXTENSIONS = {".htm", ".html"}


def _filing_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in FILING_EXTENSIONS)


def _copy_to_uploads(source_path: Path, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / source_path.name
    if not dest.exists() or dest.stat().st_size != source_path.stat().st_size:
        shutil.copy2(source_path, dest)
    return dest


def _worker_embed_one(args):
    """Runs in a separate process -- imports happen here, not at module
    level, so each worker gets its own model instance (sentence-transformers
    models aren't fork/pickle-safe to share across processes)."""
    filepath_str, doc_name, embedding_model = args
    sys.path.insert(0, str(BACKEND_DIR))
    import os
    os.environ["EMBEDDING_MODEL"] = embedding_model
    from ingest import ingest_filing  # noqa: E402

    t0 = time.time()
    try:
        index = asyncio.run(ingest_filing(filepath_str, doc_name, use_embeddings=True))
        vector_count = 0
        if index.vector_store is not None and index.vector_store.index is not None:
            vector_count = int(index.vector_store.index.ntotal)
        if vector_count <= 0:
            raise RuntimeError(f"No vectors written for EMBEDDING_MODEL={embedding_model}")
        return {"doc_name": doc_name, "ok": True, "chunks": len(index.chunks), "vectors": vector_count, "elapsed": time.time() - t0}
    except Exception as exc:
        return {"doc_name": doc_name, "ok": False, "error": str(exc), "elapsed": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    parser.add_argument("--embedding-model", choices=["normal", "finlang", "financesmall"], default="financesmall")
    parser.add_argument("--workers", type=int, default=6, help="Process pool size. Leave some cores free for the OS.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated doc_names to process.")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    source_files = _filing_files(args.source_dir)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        source_files = [p for p in source_files if p.stem in wanted]

    jobs = []
    for p in source_files:
        filepath = p if args.no_copy else _copy_to_uploads(p, args.uploads_dir)
        jobs.append((str(filepath), p.stem, args.embedding_model))

    print(f"Re-embedding {len(jobs)} filings with {args.workers} worker processes (embedding_model={args.embedding_model})\n")

    started = time.time()
    completed = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker_embed_one, job): job[1] for job in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r["ok"]:
                completed += 1
                print(f"[{i}/{len(jobs)}] OK {r['doc_name']}: chunks={r['chunks']} vectors={r['vectors']} elapsed={r['elapsed']:.1f}s", flush=True)
            else:
                failed.append(r)
                print(f"[{i}/{len(jobs)}] FAILED {r['doc_name']}: {r['error']} elapsed={r['elapsed']:.1f}s", flush=True)

    total_elapsed = time.time() - started
    print(f"\nDone. Completed {completed}/{len(jobs)} in {total_elapsed:.1f}s wall time.")
    if failed:
        report_path = PROJECT_ROOT / "embedding_failures.json"
        report_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"Failure report: {report_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
