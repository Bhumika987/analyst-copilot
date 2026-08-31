"""
Scoring harness against practice-questions.jsonl.

Mirrors the competition rubric exactly:
  +1  correct answer, correct page (exact match)
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
import os
import re
import sys
from typing import Optional
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ingest import ingest_filing, hydrate_from_postgres  # noqa: E402
from llm import answer_question, get_embedding, call_llm_raw  # noqa: E402
from retrieval import get_index, cross_filing_hybrid_search  # noqa: E402
from config import get_embedding_model_name  # noqa: E402
import postgres_store as pg  # noqa: E402

# Populated only in --from-postgres mode -- deliberately separate from
# retrieval.py's own _INDEX_CACHE/local-disk-backed get_index() so a
# Postgres-sourced run can never silently fall back to (or get shadowed
# by) whatever's already indexed locally on the machine running this
# script. See hydrate_from_postgres(cache_locally=False) in ingest.py.
_PG_INDEX_CACHE = {}

# ---- path constants ----
DATA_ALT = Path(r"c:\Users\Sakshi Sinha\Downloads\analyst-copilot-data 1\analyst-copilot-data")
PRACTICE_QUESTIONS_PATH = SCRIPT_DIR.parent / "data" / "practice-questions.jsonl"
if not PRACTICE_QUESTIONS_PATH.exists() and (DATA_ALT / "practice-questions.jsonl").exists():
    PRACTICE_QUESTIONS_PATH = DATA_ALT / "practice-questions.jsonl"

FILINGS_DIR = SCRIPT_DIR.parent / "data" / "filings"
if not FILINGS_DIR.exists() and (DATA_ALT / "filings").exists():
    FILINGS_DIR = DATA_ALT / "filings"

RESULTS_PATH = SCRIPT_DIR.parent / "eval_results.json"


def _write_results(results, counts, score, embedding_model, in_progress=False):
    """Write eval_results.json. Called after every question (not just once
    at the end of run_eval) so an interrupted run -- Ctrl+C, a crash, a
    dropped connection -- still leaves the questions scored so far on disk
    instead of losing the whole run's progress. "in_progress" flags a
    partial write so a reader can tell it apart from a completed run."""
    total_scored = len(results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "embedding_model": embedding_model,
                "in_progress": in_progress,
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


# The competition rubric scores page citations as exact-or-wrong (there is
# no partial credit for "close"), so this must be 0. It was previously 5,
# which silently absorbed a real off-by-one bug in the parser's page
# attribution (see filing_parser.py) and inflated reported scores.
PAGE_TOLERANCE = 0
# Trailing (?!\w) keeps this from matching the leading digit of an
# alphanumeric token like "3M" (the company name) as if it were the number
# 3 -- confirmed as a real false-positive source: that bogus "3" paired with
# an unrelated "2.0%" elsewhere in an answer, differenced to ~1.0, and
# landed within _delta_matches_gold's tolerance of an unrelated gold value
# purely by coincidence.
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?(?!\w)")


def normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    # Disallowed characters become a SPACE, not empty -- some gold answers
    # spell out a formula with "+"/"/" and no surrounding spaces (e.g.
    # "cash equivalents+Short term investments"). Stripping to empty glued
    # the neighboring words into a single garbage token
    # ("equivalentsshort"), silently destroying word-overlap matching for
    # any such answer.
    s = re.sub(r"[^\w\s.%$-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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


def _numbers_equivalent(pred_nums, gold_nums) -> bool:
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


def _numbers_with_positions(s: str):
    """Like extract_numbers, but keeps every occurrence (not deduped into a
    set) paired with where it appears, for proximity-aware pairing below."""
    out = []
    for m in NUMBER_RE.finditer(s or ""):
        cleaned = m.group(0).replace(",", "").replace("$", "")
        if cleaned in ("", "-", "."):
            continue
        try:
            out.append((float(cleaned.rstrip("%")), m.start()))
        except ValueError:
            continue
    return out


# How close two predicted numbers must appear (in characters) to be treated
# as "the same before/after comparison" for delta-matching. Confirmed as
# necessary, not just theoretical: pairing numbers from anywhere in a
# multi-sentence answer produced a real false positive (an unrelated pair
# ~50 chars apart differenced to a value that coincidentally landed inside
# tolerance of an unrelated gold delta) before the NUMBER_RE fix above
# happened to also remove that specific pair's bogus half. This bounds the
# blast radius of the next coincidence the regex fix doesn't happen to catch.
_DELTA_PROXIMITY_CHARS = 40


def _delta_matches_gold(pred_text: str, gold_nums) -> bool:
    """Gold sometimes states a *change* ("declined by 0.8%") while a correct
    predicted answer states the two raw values it changed between ("from
    19.4% to 18.5%") without ever restating the delta as its own number --
    _numbers_equivalent alone can't see that these agree. Check pairwise
    |a - b| between predicted numbers that appear near each other in the
    text against every gold number. Wider tolerance than
    _numbers_equivalent's: a delta is a second-order quantity built from two
    already-rounded figures, so it inherits both roundings' error, not just
    one.
    """
    try:
        gold_vals = [float(g.rstrip("%")) for g in gold_nums]
    except ValueError:
        return False
    pred_vals = _numbers_with_positions(pred_text)
    if len(pred_vals) < 2:
        return False
    for gold_val in gold_vals:
        tolerance = max(0.15, abs(gold_val) * 0.15)
        for i, (a, pos_a) in enumerate(pred_vals):
            for b, pos_b in pred_vals[i + 1:]:
                if abs(pos_a - pos_b) > _DELTA_PROXIMITY_CHARS:
                    continue
                if abs(abs(a - b) - abs(gold_val)) <= tolerance:
                    return True
    return False


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
    if gold_nums and pred_nums and _numbers_equivalent(pred_nums, gold_nums):
        return True
    if gold_nums and pred_nums and _delta_matches_gold(predicted, gold_nums):
        return True

    gold_words = set(w for w in gold_norm.split() if len(w) > 3)
    pred_words = set(w for w in pred_norm.split() if len(w) > 3)
    overlap = gold_words & pred_words
    if not gold_words:
        return False

    # When gold's answer hinges on a specific number, word overlap must
    # never be the deciding signal for ANY of the branches below (not just
    # the lenient 0.4 one) -- shared topic vocabulary ("restructuring",
    # "liability", "related") doesn't mean the number matched, and the
    # numeric checks above already had their shot. Confirmed against a real
    # false positive: fixing an unrelated normalize_str bug (word-gluing on
    # unicode hyphens) pushed one such case's overlap ratio from 0.50 to
    # 0.67, clearing even the strict >=0.6 bar below on topic words alone,
    # with zero overlap on the actual requested figure.
    if gold_nums:
        return False

    ratio = len(overlap) / len(gold_words)
    if ratio >= 0.6:
        return True
    if gold_yn and pred_yn and gold_yn == pred_yn and ratio >= 0.3 and len(overlap) >= 2:
        return True
    # Short factual/descriptive golds (e.g. "X is a leader in Y production")
    # are often correctly paraphrased below the 0.6 bar -- "packaging
    # industry" vs "packaging production" share the load-bearing noun but
    # not the surrounding words. Requires no yes/no verdict present (that
    # has its own lenient branch above), so this can't flip a wrong
    # directional answer into a match.
    if not gold_yn and not pred_yn and ratio >= 0.4 and len(overlap) >= 2:
        return True

    return False


def numeric_equivalent(predicted: str, gold: str, relative_tolerance: float = 0.01) -> Optional[bool]:
    """Unit/format-agnostic numeric comparison: "$1577.00", "$1,577
    million", "1577", and "USD 1,577 million" all compare equal -- extracts
    numbers from both sides (via the same word-boundary-safe extraction
    answers_match already uses, so "3M's" doesn't yield a bogus "3") and
    compares values, not strings. Returns None, not False, when gold has no
    extractable number at all, so callers know "not applicable" from a real
    mismatch and can fall through to semantic judging instead of silently
    failing a comparison that was never numeric to begin with."""
    gold_nums = extract_numbers(gold)
    if not gold_nums:
        return None
    pred_nums = extract_numbers(predicted)
    if not pred_nums:
        return False
    for g in gold_nums:
        try:
            gv = float(g.rstrip("%"))
        except ValueError:
            continue
        tolerance = max(1e-9, abs(gv) * relative_tolerance)
        for p in pred_nums:
            try:
                pv = float(p.rstrip("%"))
            except ValueError:
                continue
            if abs(pv - gv) <= tolerance:
                return True
    return False


def _gold_is_purely_numeric(gold: str) -> bool:
    """True when gold, with its number(s) removed, has essentially nothing
    left -- gold IS the number ("$1577.00", "24.26", "1.9%"), not a
    sentence that merely mentions one ("...decreased by 1.7% primarily due
    to: ..."). Only these route through numeric_equivalent alone; anything
    with real prose goes through the semantic judge instead, so a
    coincidentally-matching embedded figure (e.g. both sides mentioning the
    same percentage-point change while citing entirely different drivers
    for it) can't stand in for actually checking the answer."""
    stripped = NUMBER_RE.sub("", gold or "")
    stripped = re.sub(r"[^\w]", "", stripped)
    return len(stripped) <= 6


EVALUATE_ANSWER_PROMPT = """Determine whether the predicted answer conveys the same final answer as the reference answer, for the given question.

QUESTION: {question}

REFERENCE ANSWER: {gold}

PREDICTED ANSWER: {predicted}

Ignore:
- wording differences
- formatting
- additional correct explanation
- harmless rounding differences

Do not ignore:
- opposite conclusions
- incorrect reporting periods
- wrong entities or segments
- unsupported additional items

Return exactly one of:
CORRECT
INCORRECT
AMBIGUOUS"""


async def evaluate_answer(question: str, predicted: str, gold: str) -> str:
    """Strict three-way semantic judge for descriptive (non-numeric-only)
    answers -- CORRECT, INCORRECT, or AMBIGUOUS, verbatim per the requested
    design. Falls back to the regex/word-overlap heuristic on judge-call
    failure or an unparseable verdict (collapsed to CORRECT/INCORRECT,
    since the heuristic has no ambiguous state) rather than raising -- one
    question's grading degrading must never abort the run. Checked for
    "INCORRECT" before "CORRECT" since the former contains the latter as a
    substring."""
    if not predicted or not gold:
        return "INCORRECT"
    if normalize_str(predicted) == normalize_str(gold):
        return "CORRECT"

    messages = [{"role": "user", "content": EVALUATE_ANSWER_PROMPT.format(question=question, gold=gold, predicted=predicted)}]
    try:
        verdict = (await call_llm_raw(messages, max_retries=2)).strip().upper()
        if "INCORRECT" in verdict:
            return "INCORRECT"
        if "AMBIGUOUS" in verdict:
            return "AMBIGUOUS"
        if "CORRECT" in verdict:
            return "CORRECT"
        print(f"[LLM JUDGE] unparseable verdict {verdict!r}, falling back to heuristic")
        return "CORRECT" if answers_match(predicted, gold) else "INCORRECT"
    except Exception as exc:
        print(f"[LLM JUDGE] call failed ({exc}), falling back to heuristic")
        return "CORRECT" if answers_match(predicted, gold) else "INCORRECT"


async def is_answer_correct(question: str, predicted: str, gold: str) -> tuple:
    """Single entry point deciding answer_correct for assign_points below.
    Numeric-first for gold answers that ARE a number (exact/tolerant match,
    no LLM call needed); the strict semantic judge for everything else,
    including gold answers that merely contain a number among real prose.
    AMBIGUOUS counts as not-correct -- the marking system has only
    correct/incorrect to assign into, no separate ambiguous bucket -- but
    is returned distinguishably in the method string so it's visible in
    results.json rather than looking identical to a clean INCORRECT.

    Returns (is_correct: bool, method: str) for logging.
    """
    if _gold_is_purely_numeric(gold):
        result = numeric_equivalent(predicted, gold, relative_tolerance=0.01)
        if result is not None:
            return result, f"numeric ({'match' if result else 'no match'})"
        # gold looked like a bare number but nothing extractable from the
        # predicted answer -- fall through to the semantic judge rather
        # than silently defaulting to incorrect.
    verdict = await evaluate_answer(question, predicted, gold)
    return verdict == "CORRECT", f"semantic ({verdict})"


def page_matches(predicted_page, gold_page) -> bool:
    if predicted_page is None or gold_page is None:
        return False
    try:
        return abs(int(predicted_page) - int(gold_page)) <= PAGE_TOLERANCE
    except (ValueError, TypeError):
        return False


def any_source_page_matches(sources, gold_page) -> bool:
    """A calculation answer often cites one SOURCE line per line item it
    used (CAPEX on one page, net sales on another, ...) -- result.page_num
    is only the FIRST one, so scoring against that alone means the exact
    ordering the model happened to list its sources in decides pass/fail,
    not whether it actually had (and cited) the right evidence. Confirmed
    as a real, recurring cost: 4 of 4 wrong-page misses in one 20-question
    run were multi-source answers where a later-listed citation was the
    one gold expected. Check every cited page, not just the first."""
    if not sources or gold_page is None:
        return False
    return any(page_matches(s.get("page_num"), gold_page) for s in sources)


def load_questions(limit=None, doc_filter=None, reverse=False):
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
    if reverse:
        # Reverse before applying limit, not after -- --reverse --limit 20
        # should mean "the last 20 questions, starting from the very last
        # one," not "the first 20, just iterated back-to-front."
        questions = list(reversed(questions))
    if limit:
        questions = questions[:limit]
    return questions


async def ensure_indexed_from_postgres(doc_name: str):
    """
    --from-postgres mode: source this filing's chunks/embeddings/BM25 from
    Postgres exclusively, never from local disk -- see the _PG_INDEX_CACHE
    docstring above for why this doesn't go through retrieval.get_index()
    at all. Returns None (scored as a skip, same as a missing local file)
    if this doc_name was never migrated into Postgres.
    """
    if doc_name in _PG_INDEX_CACHE:
        return _PG_INDEX_CACHE[doc_name]
    idx = await hydrate_from_postgres(doc_name, cache_locally=False)
    _PG_INDEX_CACHE[doc_name] = idx  # cache the None too, so a missing doc isn't re-queried every question
    return idx


async def ensure_indexed(doc_name: str, use_embeddings: bool, from_postgres: bool = False):
    if from_postgres:
        return await ensure_indexed_from_postgres(doc_name)

    idx = get_index(doc_name)
    if idx is not None and idx.is_indexed() and (not use_embeddings or idx.has_vector_index()):
        return idx

    filepath = FILINGS_DIR / f"{doc_name}.htm"
    if not filepath.exists():
        filepath = FILINGS_DIR / f"{doc_name}.html"
    if not filepath.exists():
        return None

    return await ingest_filing(str(filepath), doc_name, use_embeddings=use_embeddings)


async def run_eval(limit, doc_filter, use_embeddings, embedding_model_arg=None, reverse=False, from_postgres=False):
    if embedding_model_arg:
        os.environ["EMBEDDING_MODEL"] = embedding_model_arg
    embedding_model = get_embedding_model_name()
    if from_postgres:
        if not pg.is_configured():
            print("--from-postgres was passed but DATABASE_URL is not set -- nothing to source from Postgres with.")
            return
        # Postgres only ever stores 384-dim ("normal") vectors -- see
        # postgres_store.EMBEDDING_DIM. Silently scoring against a
        # different embedding model here would just produce a wall of
        # skips with no explanation, so fail loud instead.
        if embedding_model != "normal":
            print(f"--from-postgres requires EMBEDDING_MODEL=normal (got {embedding_model!r}).")
            return
    questions = load_questions(limit=limit, doc_filter=doc_filter, reverse=reverse)
    print(f"Loaded {len(questions)} practice questions.\n")
    print(f"Embedding model: {embedding_model}\n")
    if from_postgres:
        print("Source: Postgres (--from-postgres) -- local data/indexes/ and data/filings/ are not used.\n")

    results = []
    score = 0
    counts = {"+1": 0, "0_not_found": 0, "0_wrong_page": 0, "-1": 0, "skipped": 0}

    for i, q in enumerate(questions, start=1):
        doc_name = q.get("doc_name")
        question_text = q.get("question")
        gold_answer = q.get("answer")
        evidence = q.get("evidence") or [{}]
        gold_page = evidence[0].get("evidence_page_num")

        index = await ensure_indexed(doc_name, use_embeddings, from_postgres=from_postgres)
        if index is None or not index.is_indexed():
            print(f"[{i}/{len(questions)}] SKIP (filing not found): {doc_name}")
            counts["skipped"] += 1
            continue

        # Match main.py's own default top_k (12) rather than a narrower
        # ad hoc value -- calculation questions spanning two statements
        # need the same retrieval depth the eval is supposed to grade
        # against, not a tighter one that under-serves them by construction.
        query_vector = get_embedding(question_text) if use_embeddings else None
        chunks = index.hybrid_search(question_text, query_vector, top_k=12)
        result = await answer_question(question_text, doc_name, chunks)

        # Free-tier Groq keys share an ~8000-token/min budget across every
        # request; pace requests so a normal-sized call doesn't trip 429s
        # that eat into the retry budget before it even starts. Pause after
        # EVERY call, not just successes -- a rate-limit error means the
        # per-minute window is already exhausted, so firing the next request
        # immediately just cascades into another 429 (this previously left
        # most of a run reporting NOT_FOUND for "rate limited" rather than
        # a real evidence gap). Failures get a longer pause since the window
        # that caused them needs more time to clear.
        await asyncio.sleep(3.0 if result.error is None else 15.0)

        # Decision order, exactly:
        #   1. Did the model abstain?              -> 0
        #   2. Is the answer factually correct?     No -> -1
        #   3. Is the supporting page correct?      Yes -> +1, No -> 0
        # answer_correct and page_correct are each computed independently
        # (neither reads the other) before assign_points combines them --
        # a page mismatch must never be able to turn a correct answer into
        # a wrong one, and a wrong answer's page is never even checked.
        judge_method = None
        if not result.found:
            reason, points = "0_not_found", 0
        else:
            answer_correct, judge_method = await is_answer_correct(question_text, result.answer, gold_answer)
            # Judge call adds a second LLM round-trip on top of answer_question's
            # own (which itself now includes the mandatory relevance-judge
            # rerank) -- pace it the same way to avoid cascading into the
            # rate-limit window the answer call just used. Skipped entirely
            # when numeric_equivalent decided it (no LLM call was made).
            if judge_method and judge_method.startswith("semantic"):
                await asyncio.sleep(2.0)
            page_correct = any_source_page_matches(result.sources, gold_page)

            if not answer_correct:
                reason, points = "-1", -1
            elif not page_correct:
                reason, points = "0_wrong_page", 0
            else:
                reason, points = "+1", 1

        counts[reason] += 1
        score += points

        # result.page_num is just sources[0]'s page -- when the answer cited
        # several (one per line item it used, commonly), collapsing to only
        # the first hides exactly the information any_source_page_matches
        # above is scoring against. Show every distinct cited page; a single
        # source still renders as a plain int, not a one-item list, so this
        # doesn't change the shape of the common case.
        cited_pages = [s.get("page_num") for s in (result.sources or []) if s.get("page_num") is not None]
        if not cited_pages and result.page_num is not None:
            cited_pages = [result.page_num]
        cited_pages = list(dict.fromkeys(cited_pages))  # de-dupe, preserve order
        if not cited_pages:
            predicted_page_display = None  # NOT_FOUND / abstained: nothing was cited
        elif len(cited_pages) == 1:
            predicted_page_display = cited_pages[0]
        else:
            predicted_page_display = cited_pages

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
            "predicted_page": predicted_page_display,
            "predicted_evidence": result.evidence_text,
            "confidence": result.confidence,
            "embedding_model": embedding_model,
            "vector_index_path": str(index.vector_index_dir()) if use_embeddings else None,
            "reason": reason,
            "points": points,
            # Non-scoring diagnostic only: which path decided answer_correct
            # ("numeric (match)"/"numeric (no match)", or "semantic (CORRECT
            # /INCORRECT/AMBIGUOUS)"). Doesn't affect points -- purely so a
            # human reviewing results.json can see how each verdict was
            # reached, e.g. distinguishing a clean INCORRECT from a judge
            # call that came back AMBIGUOUS (which still scores as -1, since
            # the marking system has no ambiguous bucket, but reads
            # differently on review).
            "judge_method": judge_method,
            "error": result.error,
        })

        # Save after every question, not just at the end -- see
        # _write_results' docstring for why.
        _write_results(results, counts, score, embedding_model, in_progress=True)

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

    _write_results(results, counts, score, embedding_model, in_progress=False)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


def _pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Analyst Copilot against practice questions.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate N questions (the first N, or the last N with --reverse).")
    parser.add_argument("--doc", type=str, default=None, help="Only evaluate questions for this doc_name.")
    parser.add_argument("--no-embed", action="store_true", help="Skip dense embeddings, BM25-only.")
    parser.add_argument("--embedding-model", choices=["normal", "finlang", "financesmall"], default=None, help="Embedding model override. Defaults to EMBEDDING_MODEL or normal.")
    parser.add_argument("--reverse", action="store_true", help="Start from the last question in practice-questions.jsonl and work backward, instead of front-to-back.")
    parser.add_argument("--from-postgres", action="store_true", help="Source every filing's chunks/embeddings from Postgres (DATABASE_URL) instead of local data/indexes/ and data/filings/ -- verifies the migrated data actually works end to end, not just that it exists. Requires EMBEDDING_MODEL=normal.")
    args = parser.parse_args()

    asyncio.run(run_eval(
        limit=args.limit, doc_filter=args.doc, use_embeddings=not args.no_embed,
        embedding_model_arg=args.embedding_model, reverse=args.reverse, from_postgres=args.from_postgres,
    ))


if __name__ == "__main__":
    main()
