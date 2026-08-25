"""
Find filing documents that exist in a source filings folder but have not been
uploaded and/or indexed by the local Analyst Copilot app.

Run from the project root:
    python scripts/find_missing_filings.py
    python scripts/find_missing_filings.py --mode indexed
    python scripts/find_missing_filings.py --write-json missing_filings.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = Path(
    r"C:\Users\Sakshi Sinha\Downloads\analyst-copilot-data 1\analyst-copilot-data\filings"
)
DEFAULT_UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
DEFAULT_INDEXES_DIR = PROJECT_ROOT / "data" / "indexes"

FILING_EXTENSIONS = {".htm", ".html"}


def _doc_name(path: Path) -> str:
    return path.stem


def _filing_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in FILING_EXTENSIONS
    )


def _indexed_doc_names(indexes_dir: Path) -> Set[str]:
    if not indexes_dir.exists():
        return set()
    return {
        p.name for p in indexes_dir.iterdir()
        if p.is_dir() and (p / "chunks.json").exists()
    }


def find_missing(
    source_dir: Path,
    uploads_dir: Path,
    indexes_dir: Path,
) -> Dict:
    source_files = _filing_files(source_dir)
    uploaded_files = _filing_files(uploads_dir)

    source_by_doc = {_doc_name(p): p for p in source_files}
    uploaded_docs = {_doc_name(p) for p in uploaded_files}
    indexed_docs = _indexed_doc_names(indexes_dir)

    missing_uploads = sorted(set(source_by_doc) - uploaded_docs)
    missing_indexes = sorted(set(source_by_doc) - indexed_docs)

    return {
        "source_dir": str(source_dir),
        "uploads_dir": str(uploads_dir),
        "indexes_dir": str(indexes_dir),
        "source_count": len(source_by_doc),
        "uploaded_count": len(uploaded_docs),
        "indexed_count": len(indexed_docs),
        "missing_upload_count": len(missing_uploads),
        "missing_index_count": len(missing_indexes),
        "missing_uploads": [
            {"doc_name": doc, "path": str(source_by_doc[doc])}
            for doc in missing_uploads
        ],
        "missing_indexes": [
            {"doc_name": doc, "path": str(source_by_doc[doc])}
            for doc in missing_indexes
        ],
    }


def _print_section(title: str, rows: List[Dict], limit: int) -> None:
    print(title)
    print("-" * len(title))
    if not rows:
        print("None")
        print()
        return

    shown = rows if limit <= 0 else rows[:limit]
    for row in shown:
        print(f"{row['doc_name']}  |  {row['path']}")
    if limit > 0 and len(rows) > limit:
        print(f"... {len(rows) - limit} more")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find source filings that have not been uploaded and/or indexed."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    parser.add_argument("--indexes-dir", type=Path, default=DEFAULT_INDEXES_DIR)
    parser.add_argument(
        "--mode",
        choices=("all", "uploaded", "indexed"),
        default="all",
        help="Which missing-file list to print.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit printed rows per section. Use 0 for all rows.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON.",
    )
    args = parser.parse_args()

    report = find_missing(args.source_dir, args.uploads_dir, args.indexes_dir)

    print("Filing Upload/Index Gap Report")
    print("==============================")
    print(f"Source filings: {report['source_count']}")
    print(f"Uploaded filings: {report['uploaded_count']}")
    print(f"Indexed filings: {report['indexed_count']}")
    print(f"Missing uploads: {report['missing_upload_count']}")
    print(f"Missing indexes: {report['missing_index_count']}")
    print()

    if args.mode in ("all", "uploaded"):
        _print_section("Missing From Uploads", report["missing_uploads"], args.limit)
    if args.mode in ("all", "indexed"):
        _print_section("Missing From Indexes", report["missing_indexes"], args.limit)

    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.write_json}")


if __name__ == "__main__":
    main()
