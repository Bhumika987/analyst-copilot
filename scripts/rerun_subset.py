"""
Re-run only a specific subset of practice questions (e.g. everything that
scored non-+1 in a previous eval_results.json), in small batches, printing a
banner between batches so progress is visible without waiting for the whole
subset to finish.

Usage:
  python scripts/rerun_subset.py --from-results eval_results.json --reason-not +1 --batch-size 5
  python scripts/rerun_subset.py --from-results eval_results.json --reason-not +1 --batch-size 5 --max-batches 1

Reuses evaluate.py's own scoring functions (ensure_indexed, answer_question,
is_answer_correct, any_source_page_matches) rather than re-implementing
anything, so this scores identically to a normal evaluate.py run -- it's
just a different subset, in smaller steps.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev  # noqa: E402

OUT_PATH = SCRIPT_DIR.parent / "eval_results_rerun.json"


def load_subset(from_results: Path, reason_not: str):
    d = json.loads(from_results.read_text(encoding="utf-8"))
    failed_pairs = {
        (r["doc_name"], r["question"])
        for r in d["results"]
        if r["reason"] != reason_not
    }
    all_questions = ev.load_questions()
    subset = [q for q in all_questions if (q.get("doc_name"), q.get("question")) in failed_pairs]
    return subset


async def score_one(q, use_embeddings):
    doc_name = q.get("doc_name")
    question_text = q.get("question")
    gold_answer = q.get("answer")
    evidence = q.get("evidence") or [{}]
    gold_page = evidence[0].get("evidence_page_num")

    index = await ev.ensure_indexed(doc_name, use_embeddings)
    if index is None or not index.is_indexed():
        return {"doc_name": doc_name, "question": question_text, "reason": "skipped", "points": 0}

    query_vector = ev.get_embedding(question_text) if use_embeddings else None
    chunks = index.hybrid_search(question_text, query_vector, top_k=12)
    result = await ev.answer_question(question_text, doc_name, chunks)

    await asyncio.sleep(3.0 if result.error is None else 15.0)

    judge_method = None
    if not result.found:
        reason, points = "0_not_found", 0
    else:
        answer_correct, judge_method = await ev.is_answer_correct(question_text, result.answer, gold_answer)
        if judge_method and judge_method.startswith("semantic"):
            await asyncio.sleep(2.0)
        page_correct = ev.any_source_page_matches(result.sources, gold_page)
        if not answer_correct:
            reason, points = "-1", -1
        elif not page_correct:
            reason, points = "0_wrong_page", 0
        else:
            reason, points = "+1", 1

    cited_pages = [s.get("page_num") for s in (result.sources or []) if s.get("page_num") is not None]
    if not cited_pages and result.page_num is not None:
        cited_pages = [result.page_num]
    cited_pages = list(dict.fromkeys(cited_pages))
    predicted_page_display = None if not cited_pages else (cited_pages[0] if len(cited_pages) == 1 else cited_pages)

    return {
        "doc_name": doc_name,
        "question": question_text,
        "gold_answer": gold_answer,
        "gold_page": gold_page,
        "predicted_found": result.found,
        "predicted_answer": result.answer,
        "predicted_page": predicted_page_display,
        "reason": reason,
        "points": points,
        "judge_method": judge_method,
        "error": result.error,
    }


def save(results):
    counts = {}
    for r in results:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    OUT_PATH.write_text(json.dumps({
        "summary": {"total": len(results), "counts": counts, "score": sum(r["points"] for r in results)},
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-results", type=str, default="eval_results.json")
    parser.add_argument("--reason-not", type=str, default="+1", help="Rerun everything whose prior reason isn't this.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many batches (omit to run all).")
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    subset = load_subset(Path(args.from_results), args.reason_not)
    print(f"Loaded {len(subset)} questions to rerun (reason != {args.reason_not!r}).\n")

    results = []
    batch_num = 0
    for start in range(0, len(subset), args.batch_size):
        if args.max_batches is not None and batch_num >= args.max_batches:
            print(f"\nStopping after {batch_num} batch(es) (--max-batches).")
            break
        batch = subset[start:start + args.batch_size]
        batch_num += 1
        print(f"\n{'=' * 60}\nBATCH {batch_num}  (questions {start + 1}-{start + len(batch)} of {len(subset)})\n{'=' * 60}")
        for q in batch:
            r = await score_one(q, use_embeddings=not args.no_embed)
            results.append(r)
            save(results)
            symbol = {"+1": "PASS", "0_not_found": "ABSTAIN", "0_wrong_page": "WRONG-PAGE", "-1": "FAIL", "skipped": "SKIP"}[r["reason"]]
            print(f"  [{symbol:10s} ({r['points']:+d})] {r['doc_name']}  {r['question'][:60]}")
        counts = {}
        for r in results:
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
        print(f"\n  Running totals after batch {batch_num}: {counts}  (score so far: {sum(r['points'] for r in results)})")

    print(f"\nDone. {len(results)} questions rerun. Saved to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
