"""
Scoring harness against practice-questions.jsonl.

Mirrors the competition rubric exactly:
  +1  correct answer, correct page (within +/- PAGE_TOLERANCE)
   0  NOT_FOUND (abstained)
   0  correct answer, wrong/missing page
  -1  wrong answer (confidently stated, doesn't match the key)

Run from the project root:
  python scripts/evaluate.py --limit 20
  python scripts/evaluate.py --doc 3M_2018_10K
  python scripts/evaluate.py --no-embed
"""

import argparse
import asyncio
import json
import re
import sys
from typing import Optional
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ingest import ingest_filing  # noqa: E402
from llm import answer_question, get_embedding  # noqa: E402
from retrieval import get_index  # noqa: E402

# ---- path constants: edit these if your data lives elsewhere ----
PRACTICE_QUESTIONS_PATH = SCRIPT_DIR.parent / "data" / "practice-questions.jsonl"
FILINGS_DIR = SCRIPT_DIR.parent / "data" / "filings"
RESULTS_PATH = SCRIPT_DIR.parent / "eval_results.json"

PAGE_TOLERANCE = 5
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s.%$-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(s: str):
    if not s:
        return set()
    found = NUMBER_RE.findall(s)
    cleaned = set()
    for f in found:
        f2 = f.replace(",", "").replace("$", "")
        if f2 not in ("", "-", "."):
            cleaned.add(f2)
    return cleaned


def _leading_yesno(s: str) -> Optional[str]:
    s = s.strip().lower()
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

    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    # Many practice questions are "Is X true?" style with a Yes/No verdict
    # plus reasoning. Directional agreement/disagreement is checked first
    # and explicitly: a shared verdict should lower the bar for what counts
    # as "the same answer" (the reasoning is rarely phrased identically),
    # while a shared vocabulary that still disagrees on Yes/No must never
    # count as a match no matter how much text overlaps.
    gold_yn = _leading_yesno(gold)
    pred_yn = _leading_yesno(predicted)
    if gold_yn and pred_yn and gold_yn != pred_yn:
        return False

    pred_nums = extract_numbers(predicted)
    gold_nums = extract_numbers(gold)
    if gold_nums and pred_nums & gold_nums:
        return True

    gold_words = set(w for w in gold_norm.split() if len(w) > 3)
    pred_words = set(w for w in pred_norm.split() if len(w) > 3)
    overlap = gold_words & pred_words
    if not gold_words:
        return False

    ratio = len(overlap) / len(gold_words)
    if ratio >= 0.6:
        return True
    if gold_yn and pred_yn and gold_yn == pred_yn and ratio >= 0.3 and len(overlap) >= 2:
        return True

    return False


def page_matches(predicted_page, gold_page) -> bool:
    if predicted_page is None or gold_page is None:
        return False
    try:
        return abs(int(predicted_page) - int(gold_page)) <= PAGE_TOLERANCE
    except (ValueError, TypeError):
        return False


def load_questions(limit=None, doc_filter=None):
    questions = []
    with open(PRACTICE_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if doc_filter and q.get("doc_name") != doc_filter:
                continue
            questions.append(q)
    if limit:
        questions = questions[:limit]
    return questions


async def ensure_indexed(doc_name: str, use_embeddings: bool):
    idx = get_index(doc_name)
    if idx is not None and idx.is_indexed():
        return idx

    filepath = FILINGS_DIR / f"{doc_name}.htm"
    if not filepath.exists():
        filepath = FILINGS_DIR / f"{doc_name}.html"
    if not filepath.exists():
        return None

    return await ingest_filing(str(filepath), doc_name, use_embeddings=use_embeddings)


async def run_eval(limit, doc_filter, use_embeddings):
    questions = load_questions(limit=limit, doc_filter=doc_filter)
    print(f"Loaded {len(questions)} practice questions.\n")

    results = []
    score = 0
    counts = {"+1": 0, "0_not_found": 0, "0_wrong_page": 0, "-1": 0, "skipped": 0}

    for i, q in enumerate(questions, start=1):
        doc_name = q.get("doc_name")
        question_text = q.get("question")
        gold_answer = q.get("answer")
        evidence = q.get("evidence") or [{}]
        gold_page = evidence[0].get("evidence_page_num")

        index = await ensure_indexed(doc_name, use_embeddings)
        if index is None or not index.is_indexed():
            print(f"[{i}/{len(questions)}] SKIP (filing not found): {doc_name}")
            counts["skipped"] += 1
            continue

        query_vector = get_embedding(question_text) if use_embeddings else None
        chunks = index.hybrid_search(question_text, query_vector, top_k=5)
        result = await answer_question(question_text, doc_name, chunks)

        # Free-tier Groq keys share an ~8000-token/min budget across every
        # request; pace requests so a normal-sized call doesn't trip 429s
        # that eat into the retry budget before it even starts.
        if result.error is None:
            await asyncio.sleep(2.0)

        reason = None
        if not result.found:
            reason = "0_not_found"
            points = 0
        else:
            correct_answer = answers_match(result.answer, gold_answer)
            correct_page = page_matches(result.page_num, gold_page)
            if correct_answer and correct_page:
                reason = "+1"
                points = 1
            elif correct_answer and not correct_page:
                reason = "0_wrong_page"
                points = 0
            else:
                reason = "-1"
                points = -1

        counts[reason] += 1
        score += points

        status_symbol = {"+1": "PASS", "0_not_found": "ABSTAIN", "0_wrong_page": "WRONG-PAGE", "-1": "FAIL"}[reason]
        print(f"[{i}/{len(questions)}] {status_symbol:10s} ({points:+d})  {doc_name}  Q: {question_text[:70]}")

        results.append({
            "index": i,
            "doc_name": doc_name,
            "question": question_text,
            "gold_answer": gold_answer,
            "gold_page": gold_page,
            "predicted_found": result.found,
            "predicted_answer": result.answer,
            "predicted_page": result.page_num,
            "predicted_evidence": result.evidence_text,
            "confidence": result.confidence,
            "reason": reason,
            "points": points,
            "error": result.error,
        })

    total_scored = len(results)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions evaluated: {total_scored} (skipped: {counts['skipped']})")
    print(f"  +1 (correct answer + page): {counts['+1']}  ({_pct(counts['+1'], total_scored)}%)")
    print(f"   0 (not found / abstained): {counts['0_not_found']}  ({_pct(counts['0_not_found'], total_scored)}%)")
    print(f"   0 (correct answer, wrong page): {counts['0_wrong_page']}  ({_pct(counts['0_wrong_page'], total_scored)}%)")
    print(f"  -1 (wrong answer): {counts['-1']}  ({_pct(counts['-1'], total_scored)}%)")
    print(f"\nTotal score: {score}")
    if total_scored:
        print(f"Average score per question: {score / total_scored:.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_scored": total_scored,
                "skipped": counts["skipped"],
                "plus_one": counts["+1"],
                "not_found": counts["0_not_found"],
                "wrong_page": counts["0_wrong_page"],
                "minus_one": counts["-1"],
                "total_score": score,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


def _pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Analyst Copilot against practice questions.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--doc", type=str, default=None, help="Only evaluate questions for this doc_name.")
    parser.add_argument("--no-embed", action="store_true", help="Skip dense embeddings, BM25-only.")
    args = parser.parse_args()

    asyncio.run(run_eval(limit=args.limit, doc_filter=args.doc, use_embeddings=not args.no_embed))


if __name__ == "__main__":
    main()
