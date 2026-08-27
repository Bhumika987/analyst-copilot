"""
Upload and embed every filing from the external FinanceBench filings folder.

This script is resumable:
- copies missing source filings into data/uploads unless --no-copy is used
- skips documents that already have a non-empty FAISS embedding index
- rebuilds BM25 + BGE/FAISS indexes for missing/unembedded documents

Run from the project root:
    python scripts/embed_all_filings.py
    python scripts/embed_all_filings.py --dry-run
    python scripts/embed_all_filings.py --force
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ingest import ingest_filing  # noqa: E402
from config import get_embedding_model_name  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\Sakshi Sinha\Downloads\analyst-copilot-data 1\analyst-copilot-data\filings"
)
DEFAULT_UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DEFAULT_INDEXES_DIR = PROJECT_ROOT / "data" / "indexes"
FILING_EXTENSIONS = {".htm", ".html"}


def _filing_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in FILING_EXTENSIONS
    )


def _vector_index_dir(indexes_dir: Path, doc_name: str, embedding_model: str) -> Path:
    return indexes_dir / doc_name / embedding_model


def _read_faiss_meta(index_dir: Path) -> Dict:
    meta_path = index_dir / "faiss_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _faiss_ntotal(index_dir: Path) -> Optional[int]:
    index_path = index_dir / "faiss.index"
    if not index_path.exists():
        return None
    try:
        import faiss
        return int(faiss.read_index(str(index_path)).ntotal)
    except Exception:
        return None


def is_embedded(indexes_dir: Path, doc_name: str, embedding_model: str) -> bool:
    index_dir = indexes_dir / doc_name
    if not (index_dir / "chunks.json").exists() or not (index_dir / "bm25.pkl").exists():
        return False

    vector_dir = _vector_index_dir(indexes_dir, doc_name, embedding_model)
    if embedding_model == "normal" and not (vector_dir / "faiss.index").exists():
        vector_dir = index_dir

    meta = _read_faiss_meta(vector_dir)
    if meta.get("embedding_model") and meta.get("embedding_model") != embedding_model:
        return False
    chunk_ids = meta.get("chunk_ids") or []
    ntotal = _faiss_ntotal(vector_dir)
    if ntotal is not None:
        return ntotal > 0 and len(chunk_ids) > 0

    # Fallback for environments where faiss cannot be imported during checks.
    return (vector_dir / "faiss.index").exists() and len(chunk_ids) > 0


def copy_to_uploads(source_path: Path, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / source_path.name
    if not dest.exists() or dest.stat().st_size != source_path.stat().st_size:
        shutil.copy2(source_path, dest)
    return dest


async def embed_all(args) -> int:
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    embedding_model = get_embedding_model_name()
    source_files = _filing_files(args.source_dir)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        source_files = [p for p in source_files if p.stem in wanted]
    if args.limit and args.limit > 0:
        source_files = source_files[:args.limit]

    args.uploads_dir.mkdir(parents=True, exist_ok=True)
    args.indexes_dir.mkdir(parents=True, exist_ok=True)

    total = len(source_files)
    embedded_before = sum(1 for p in source_files if is_embedded(args.indexes_dir, p.stem, embedding_model))
    to_process = [
        p for p in source_files
        if args.force or not is_embedded(args.indexes_dir, p.stem, embedding_model)
    ]

    print("Embed All Filings")
    print("=================")
    print(f"Source dir: {args.source_dir}")
    print(f"Embedding model: {embedding_model}")
    print(f"Source filings selected: {total}")
    print(f"Already embedded: {embedded_before}")
    print(f"Will process: {len(to_process)}")
    print(f"Copy to uploads: {not args.no_copy}")
    print(f"Force rebuild: {args.force}")
    print()

    if args.dry_run:
        for path in to_process:
            print(f"WOULD EMBED: {path.stem}  |  {path}")
        return 0

    completed = 0
    failed = []
    started = time.time()

    for idx, source_path in enumerate(to_process, start=1):
        doc_name = source_path.stem
        filepath = source_path
        if not args.no_copy:
            filepath = copy_to_uploads(source_path, args.uploads_dir)

        print(f"[{idx}/{len(to_process)}] Embedding {doc_name} ...", flush=True)
        item_start = time.time()
        try:
            index = await ingest_filing(str(filepath), doc_name, use_embeddings=True)
            vector_count = 0
            if index.vector_store is not None and index.vector_store.index is not None:
                vector_count = int(index.vector_store.index.ntotal)
            if vector_count <= 0:
                raise RuntimeError(f"No vectors written for EMBEDDING_MODEL={embedding_model}")
            elapsed = time.time() - item_start
            print(
                f"  OK {doc_name}: chunks={len(index.chunks)} vectors={vector_count} elapsed={elapsed:.1f}s",
                flush=True,
            )
            completed += 1
        except Exception as exc:
            elapsed = time.time() - item_start
            failed.append({"doc_name": doc_name, "path": str(source_path), "error": str(exc)})
            print(f"  FAILED {doc_name}: {exc} elapsed={elapsed:.1f}s", flush=True)

    final_embedded = sum(1 for p in _filing_files(args.source_dir) if is_embedded(args.indexes_dir, p.stem, embedding_model))
    total_elapsed = time.time() - started

    print()
    print("Embedding Summary")
    print("=================")
    print(f"Completed this run: {completed}")
    print(f"Failed this run: {len(failed)}")
    print(f"Embedded after run: {final_embedded}")
    print(f"Elapsed seconds: {total_elapsed:.1f}")

    if failed:
        report_path = PROJECT_ROOT / "embedding_failures.json"
        report_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
        print(f"Failure report: {report_path}")
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed all filings from the source filings folder.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild even if a FAISS index already exists.")
    parser.add_argument("--embedding-model", choices=["normal", "finlang"], default=None, help="Embedding model override. Defaults to EMBEDDING_MODEL or normal.")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy source filings into data/uploads.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be embedded without doing work.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N selected filings.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated doc_names to process.")
    raise SystemExit(asyncio.run(embed_all(parser.parse_args())))


if __name__ == "__main__":
    main()
