"""
Run every practice question through the full retrieval -> LLM pipeline and
write one consolidated trace log.

The log is JSONL: one run_start record, one question_trace record per question,
and one run_end record. Each question_trace includes the dataset row, retrieval
stdout, LLM stdout, timings, retrieved chunks, final prompt context, raw LLM
output, parsed answer, and benchmark scoring fields.

Examples:
  python scripts/trace_practice_questions.py
  python scripts/trace_practice_questions.py --limit 5
  python scripts/trace_practice_questions.py --doc 3M_2018_10K
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ingest import ingest_filing  # noqa: E402
from llm import answer_question, get_embedding  # noqa: E402
from retrieval import cross_filing_hybrid_search, get_index  # noqa: E402
from config import get_embedding_model_name  # noqa: E402


DATA_ALT = Path(r"C:\Users\Sakshi Sinha\Downloads\analyst-copilot-data 1\analyst-copilot-data")
DEFAULT_QUESTIONS_PATH = DATA_ALT / "practice-questions.jsonl"
if not DEFAULT_QUESTIONS_PATH.exists():
    DEFAULT_QUESTIONS_PATH = PROJECT_DIR / "data" / "practice-questions.jsonl"

DEFAULT_FILINGS_DIR = DATA_ALT / "filings"
if not DEFAULT_FILINGS_DIR.exists():
    DEFAULT_FILINGS_DIR = PROJECT_DIR / "data" / "filings"

DEFAULT_OUTPUT_PATH = PROJECT_DIR / "logs" / "practice_question_trace.jsonl"
PAGE_TOLERANCE = 5
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


class TeeCapture(io.StringIO):
    """Capture stdout while still showing it in the terminal."""

    def __init__(self, passthrough):
        super().__init__()
        self.passthrough = passthrough

    def write(self, s: str) -> int:
        self.passthrough.write(s)
        self.passthrough.flush()
        return super().write(s)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_questions(path: Path, limit: Optional[int], doc_filter: Optional[str]) -> List[Dict[str, Any]]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_jsonl_line"] = line_num
            if doc_filter and row.get("doc_name") != doc_filter:
                continue
            questions.append(row)
            if limit and len(questions) >= limit:
                break
    return questions


async def ensure_indexed(doc_name: str, filings_dir: Path, use_embeddings: bool):
    idx = get_index(doc_name)
    if idx is not None and idx.is_indexed() and (not use_embeddings or idx.has_vector_index()):
        return idx, "loaded_existing_index"

    for suffix in (".htm", ".html"):
        filing_path = filings_dir / f"{doc_name}{suffix}"
        if filing_path.exists():
            idx = await ingest_filing(str(filing_path), doc_name, use_embeddings=use_embeddings)
            return idx, "ingested_from_filing"

    return None, "missing_filing"


def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s.%$-]", "", s)
    return re.sub(r"\s+", " ", s)


def extract_numbers(s: str) -> set:
    if not s:
        return set()
    cleaned = set()
    for value in NUMBER_RE.findall(s):
        value = value.replace(",", "").replace("$", "")
        if value not in ("", "-", "."):
            cleaned.add(value)
    return cleaned


def numbers_equivalent(pred_nums: Iterable[str], gold_nums: Iterable[str]) -> bool:
    for pred in pred_nums:
        for gold in gold_nums:
            try:
                pred_val = float(pred.rstrip("%"))
                gold_val = float(gold.rstrip("%"))
            except ValueError:
                continue
            tolerance = max(0.01, abs(gold_val) * 0.01)
            if abs(pred_val - gold_val) <= tolerance:
                return True
    return False


def leading_yesno(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    if re.match(r"^(yes)\b", s):
        return "yes"
    if re.match(r"^(no)\b", s):
        return "no"
    return None


def answers_match(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False

    pred_norm = normalize_str(predicted)
    gold_norm = normalize_str(gold)
    if pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    gold_yn = leading_yesno(gold)
    pred_yn = leading_yesno(predicted)
    if gold_yn and pred_yn and gold_yn != pred_yn:
        return False

    pred_nums = extract_numbers(predicted)
    gold_nums = extract_numbers(gold)
    if gold_nums and (pred_nums & gold_nums or numbers_equivalent(pred_nums, gold_nums)):
        return True

    gold_words = set(w for w in gold_norm.split() if len(w) > 3)
    pred_words = set(w for w in pred_norm.split() if len(w) > 3)
    if not gold_words:
        return False
    overlap_ratio = len(gold_words & pred_words) / len(gold_words)
    if overlap_ratio >= 0.6:
        return True
    return bool(gold_yn and pred_yn and gold_yn == pred_yn and overlap_ratio >= 0.3 and len(gold_words & pred_words) >= 2)


def page_matches(predicted_page, gold_page) -> bool:
    if predicted_page is None or gold_page is None:
        return False
    try:
        return abs(int(predicted_page) - int(gold_page)) <= PAGE_TOLERANCE
    except (TypeError, ValueError):
        return False


def score_result(result, gold_answer: str, gold_page) -> Dict[str, Any]:
    if not result.found:
        return {"reason": "0_not_found", "points": 0, "answer_match": False, "page_match": False}

    answer_ok = answers_match(result.answer or "", gold_answer or "")
    page_ok = page_matches(result.page_num, gold_page)
    if answer_ok and page_ok:
        reason, points = "+1", 1
    elif answer_ok:
        reason, points = "0_wrong_page", 0
    else:
        reason, points = "-1", -1
    return {"reason": reason, "points": points, "answer_match": answer_ok, "page_match": page_ok}


def chunk_trace(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    traced = []
    for rank, c in enumerate(chunks, start=1):
        traced.append({
            "rank": rank,
            "chunk_idx": c.get("chunk_idx"),
            "doc_name": c.get("doc_name"),
            "page_num": c.get("page_num"),
            "section": c.get("section"),
            "subsection": c.get("subsection"),
            "statement_title": c.get("statement_title"),
            "statement_type": c.get("statement_type"),
            "chunk_type": c.get("chunk_type"),
            "table_title": c.get("table_title"),
            "table_context": c.get("table_context"),
            "units": c.get("units"),
            "retrieval_score": c.get("retrieval_score"),
            "bm25_score": c.get("bm25_score"),
            "metadata_score": c.get("metadata_score"),
            "semantic_score": c.get("semantic_score"),
            "rerank_score": c.get("rerank_score"),
            "content_evidence_score": c.get("content_evidence_score"),
            "bm25_rank": c.get("bm25_rank"),
            "metadata_rank": c.get("metadata_rank"),
            "semantic_rank": c.get("semantic_rank"),
            "concept_matched": c.get("concept_matched"),
            "has_period_match": c.get("has_period_match"),
            "has_numeric_value": c.get("has_numeric_value"),
            "matched_grouping_terms": c.get("matched_grouping_terms"),
            "statement_type_scores": c.get("statement_type_scores"),
            "metadata_diagnostic": c.get("metadata_diagnostic"),
            "text": c.get("text"),
        })
    return traced


def evidence_terms(evidence_text: str) -> List[str]:
    text = normalize_str(evidence_text or "")
    terms = [w for w in text.split() if len(w) > 4 and not w.isdigit()]
    return list(dict.fromkeys(terms))


def evidence_matches_chunk(evidence_text: str, chunk_text: str) -> bool:
    if not evidence_text or not chunk_text:
        return False
    evidence_norm = normalize_str(evidence_text)
    chunk_norm = normalize_str(chunk_text)
    if evidence_norm and evidence_norm[:120] in chunk_norm:
        return True
    terms = evidence_terms(evidence_text)
    if not terms:
        return False
    overlap = sum(1 for term in terms if term in chunk_norm)
    return overlap >= min(8, max(3, int(len(terms) * 0.25)))


def retrieval_diagnostics(q: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidences = q.get("evidence") or []
    gold_pages = [ev.get("evidence_page_num") for ev in evidences if ev.get("evidence_page_num") is not None]
    retrieved_pages = [c.get("page_num") for c in chunks]
    gold_page_rank = None
    for rank, page in enumerate(retrieved_pages, start=1):
        if any(page_matches(page, gold_page) for gold_page in gold_pages):
            gold_page_rank = rank
            break

    evidence_group_results = []
    for idx, ev in enumerate(evidences, start=1):
        text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
        first_rank = None
        for rank, chunk in enumerate(chunks, start=1):
            if evidence_matches_chunk(text, chunk.get("text") or ""):
                first_rank = rank
                break
        evidence_group_results.append({
            "group": idx,
            "gold_page": ev.get("evidence_page_num"),
            "found": first_rank is not None,
            "rank": first_rank,
            "in_top_1": first_rank == 1,
            "in_top_5": first_rank is not None and first_rank <= 5,
            "in_top_10": first_rank is not None and first_rank <= 10,
        })

    gold_evidence_rank = None
    found_ranks = [g["rank"] for g in evidence_group_results if g["rank"] is not None]
    if found_ranks:
        gold_evidence_rank = min(found_ranks)

    def rank_at(limit: int, rank: Optional[int]) -> bool:
        return rank is not None and rank <= limit

    return {
        "question_id": q.get("financebench_id"),
        "gold_pages": gold_pages,
        "retrieved_pages": retrieved_pages,
        "gold_page_rank": gold_page_rank,
        "gold_evidence_rank": gold_evidence_rank,
        "gold_evidence_found": gold_evidence_rank is not None,
        "gold_evidence_in_top_1": rank_at(1, gold_evidence_rank),
        "gold_evidence_in_top_5": rank_at(5, gold_evidence_rank),
        "gold_evidence_in_top_10": rank_at(10, gold_evidence_rank),
        "gold_page_in_top_1": rank_at(1, gold_page_rank),
        "gold_page_in_top_5": rank_at(5, gold_page_rank),
        "gold_page_in_top_10": rank_at(10, gold_page_rank),
        "evidence_groups": evidence_group_results,
        "top_candidates": [
            {
                "rank": rank,
                "chunk_id": c.get("chunk_idx"),
                "page": c.get("page_num"),
                "statement_type": c.get("statement_type"),
                "statement_title": c.get("statement_title"),
                "table_title": c.get("table_title"),
                "table_context": c.get("table_context"),
                "bm25_score": c.get("bm25_score"),
                "vector_score": c.get("semantic_score"),
                "rrf_score": c.get("retrieval_score"),
                "structural_score": c.get("content_evidence_score"),
                "reranker_score": c.get("rerank_score"),
            }
            for rank, c in enumerate(chunks[:10], start=1)
        ],
    }


def classify_failure(score: Dict[str, Any], result_dict: Optional[Dict[str, Any]], diag: Dict[str, Any]) -> str:
    error = (result_dict or {}).get("error") or ""
    if re.search(r"groq|api|rate|429|timeout|connect|network|ssl", error, re.IGNORECASE):
        return "API_FAILURE"
    if diag.get("gold_evidence_in_top_5") or diag.get("gold_page_in_top_5"):
        if score.get("reason") == "+1":
            return "RETRIEVAL_SUCCESS_ANSWER_CORRECT"
        return "RETRIEVAL_SUCCESS_REASONING_FAILED"
    return "RETRIEVAL_FAILED"


async def trace_one_question(
    q: Dict[str, Any],
    ordinal: int,
    total: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    doc_name = q.get("doc_name")
    question_text = q.get("question") or ""
    evidence = q.get("evidence") or [{}]
    gold_page = evidence[0].get("evidence_page_num")
    timings: Dict[str, float] = {}

    print(f"\n[{ordinal}/{total}] Tracing {doc_name}: {question_text[:100]}")
    embedding_model = get_embedding_model_name()
    started = time.perf_counter()

    t0 = time.perf_counter()
    index, index_status = await ensure_indexed(doc_name, args.filings_dir, use_embeddings=not args.no_embed)
    timings["ensure_indexed_seconds"] = round(time.perf_counter() - t0, 4)

    if index is None or not index.is_indexed():
        return {
            "type": "question_trace",
            "timestamp": utc_now(),
            "index": ordinal,
            "status": "skipped",
            "skip_reason": index_status,
            "dataset_row": q,
            "timings": timings,
        }

    t0 = time.perf_counter()
    query_vector = None if args.no_embed else get_embedding(question_text)
    timings["embedding_seconds"] = round(time.perf_counter() - t0, 4)

    retrieval_capture = TeeCapture(sys.stdout)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(retrieval_capture):
        chunks = cross_filing_hybrid_search(question_text, query_vector, doc_name=doc_name, top_k=args.top_k)
    timings["retrieval_seconds"] = round(time.perf_counter() - t0, 4)

    if args.skip_llm:
        result = None
        llm_stdout = ""
        score = {"reason": "retrieval_only", "points": None, "answer_match": None, "page_match": None}
    else:
        llm_capture = TeeCapture(sys.stdout)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(llm_capture):
            result = await answer_question(question_text, doc_name, chunks)
        timings["llm_and_parse_seconds"] = round(time.perf_counter() - t0, 4)
        llm_stdout = llm_capture.getvalue()
        score = score_result(result, q.get("answer"), gold_page)
        if args.sleep_seconds and result.error is None:
            await asyncio.sleep(args.sleep_seconds)

    timings["total_seconds"] = round(time.perf_counter() - started, 4)

    result_dict = None
    if result is not None:
        result_dict = result.to_dict()
        result_dict["raw_response"] = result.raw_response
    diag = retrieval_diagnostics(q, chunks)
    failure_classification = classify_failure(score, result_dict, diag)

    trace = {
        "type": "question_trace",
        "timestamp": utc_now(),
        "index": ordinal,
        "status": "ok",
        "dataset_row": q,
        "doc_name": doc_name,
        "question": question_text,
        "gold_answer": q.get("answer"),
        "gold_page": gold_page,
        "index_status": index_status,
        "top_k": args.top_k,
        "use_embeddings": not args.no_embed,
        "embedding_model": embedding_model,
        "vector_index_path": str(index.vector_index_dir()) if not args.no_embed else None,
        "timings": timings,
        "retrieval_stdout": retrieval_capture.getvalue(),
        "llm_stdout": llm_stdout,
        "retrieved_chunks": chunk_trace(chunks),
        "retrieval_diagnostics": diag,
        "failure_classification": failure_classification,
        "result": result_dict,
        "score": score,
    }
    print(f"[{ordinal}/{total}] Done: {score['reason']} in {timings['total_seconds']}s")
    return trace


async def run(args: argparse.Namespace) -> None:
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    embedding_model = get_embedding_model_name()
    if args.output.exists() and not args.append:
        args.output.unlink()

    questions = load_questions(args.input, limit=args.limit, doc_filter=args.doc)
    run_started = time.perf_counter()
    summary = {
        "+1": 0,
        "0_not_found": 0,
        "0_wrong_page": 0,
        "-1": 0,
        "retrieval_only": 0,
        "skipped": 0,
        "errors": 0,
        "gold_page_top_5": 0,
        "gold_page_top_10": 0,
        "gold_evidence_top_5": 0,
        "gold_evidence_top_10": 0,
    }
    total_score = 0

    write_record(args.output, {
        "type": "run_start",
        "timestamp": utc_now(),
        "input": str(args.input),
        "filings_dir": str(args.filings_dir),
        "total_questions": len(questions),
        "doc_filter": args.doc,
        "limit": args.limit,
        "top_k": args.top_k,
        "use_embeddings": not args.no_embed,
        "embedding_model": embedding_model,
        "skip_llm": args.skip_llm,
    })

    for ordinal, q in enumerate(questions, start=1):
        try:
            trace = await trace_one_question(q, ordinal, len(questions), args)
        except Exception as exc:
            summary["errors"] += 1
            trace = {
                "type": "question_trace",
                "timestamp": utc_now(),
                "index": ordinal,
                "status": "error",
                "dataset_row": q,
                "error": repr(exc),
            }
            print(f"[{ordinal}/{len(questions)}] ERROR: {exc}")
            if not args.continue_on_error:
                write_record(args.output, trace)
                raise

        write_record(args.output, trace)
        if trace.get("status") == "skipped":
            summary["skipped"] += 1
            continue
        reason = (trace.get("score") or {}).get("reason")
        if reason in summary:
            summary[reason] += 1
        points = (trace.get("score") or {}).get("points")
        if isinstance(points, int):
            total_score += points
        diag = trace.get("retrieval_diagnostics") or {}
        if diag.get("gold_page_in_top_5"):
            summary["gold_page_top_5"] += 1
        if diag.get("gold_page_in_top_10"):
            summary["gold_page_top_10"] += 1
        if diag.get("gold_evidence_in_top_5"):
            summary["gold_evidence_top_5"] += 1
        if diag.get("gold_evidence_in_top_10"):
            summary["gold_evidence_top_10"] += 1

    elapsed = round(time.perf_counter() - run_started, 4)
    retrieved_count = len(questions) - summary["skipped"] - summary["errors"]
    write_record(args.output, {
        "type": "run_end",
        "timestamp": utc_now(),
        "elapsed_seconds": elapsed,
        "embedding_model": embedding_model,
        "summary": summary,
        "model_summary": {
            "embedding_model": embedding_model,
            "questions": retrieved_count,
            "correct": summary["+1"],
            "wrong_page": summary["0_wrong_page"],
            "wrong_answer": summary["-1"],
            "not_found": summary["0_not_found"],
            "retrieval_recall_page_at_5": round(summary["gold_page_top_5"] / retrieved_count, 4) if retrieved_count else 0.0,
            "retrieval_recall_page_at_10": round(summary["gold_page_top_10"] / retrieved_count, 4) if retrieved_count else 0.0,
            "retrieval_recall_evidence_at_5": round(summary["gold_evidence_top_5"] / retrieved_count, 4) if retrieved_count else 0.0,
            "retrieval_recall_evidence_at_10": round(summary["gold_evidence_top_10"] / retrieved_count, 4) if retrieved_count else 0.0,
        },
        "total_score": total_score,
    })

    print("\nTrace complete.")
    print(f"Log file: {args.output}")
    print(f"Summary: {summary}")
    print(f"Total score: {total_score}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace practice questions through retrieval and LLM generation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_QUESTIONS_PATH, help="Path to practice-questions.jsonl.")
    parser.add_argument("--filings-dir", type=Path, default=DEFAULT_FILINGS_DIR, help="Directory containing filing .htm/.html files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Single JSONL trace log output path.")
    parser.add_argument("--append", action="store_true", help="Append to an existing log instead of replacing it.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N matching questions.")
    parser.add_argument("--doc", type=str, default=None, help="Only run questions for this doc_name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final retrieved chunks to pass to the LLM.")
    parser.add_argument("--embedding-model", choices=["normal", "finlang", "financesmall"], default=None, help="Embedding model override. Defaults to EMBEDDING_MODEL or normal.")
    parser.add_argument("--no-embed", action="store_true", help="Skip dense embedding retrieval.")
    parser.add_argument("--skip-llm", action="store_true", help="Trace retrieval only, without calling Groq.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Delay between successful LLM calls to avoid rate limits.")
    parser.add_argument("--continue-on-error", action="store_true", help="Log per-question errors and continue.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
