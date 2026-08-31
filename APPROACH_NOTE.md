---
title: "Analyst Copilot — Approach Note"
---

# Analyst Copilot — Approach Note

A chatbot answering analyst questions over SEC filings, built for a rubric
where a confidently wrong answer (**−1**) costs twice what a correct one
(**+1**) earns, and abstaining ("not found") is always **0**. Every design
choice below optimizes for that asymmetry: the system is built to abstain
rather than guess.

## What we tried

Pure dense retrieval first — dropped as the primary signal because even a
real sentence-embedding model (BGE-small, run locally) is too weak on its
own for numeric financial lookups, where exact terms ("capital
expenditure", "$1,577") matter more than semantic similarity. BM25 became
the primary signal, with dense search kept as a secondary Reciprocal Rank
Fusion input to catch paraphrased questions, topped with a cross-encoder
reranker (or a deterministic structural/boost fallback when the reranker
model isn't available locally).

## What we measured

Retrieval score distributions against the practice set — this is what led
us to replace a single numeric confidence threshold with a rule-based
evidence gate (concept match, statement/section match, fiscal-year match,
numeric-value presence) that runs *before* any LLM call, so the system
abstains when the top retrieved evidence doesn't actually satisfy the
question. This is the single biggest lever on the score, since every wrong
answer costs twice what a right one earns.

## What we kept

Page-aware chunking with atomic tables (a split table row is a guaranteed
wrong number), and a strict "quote the source or say NOT_FOUND" system
prompt that forbids the model from using outside knowledge.

## What we threw away

LLM self-reported confidence as a gating signal — it stayed overconfident
even on thin evidence. A single numeric confidence-threshold gate that had
decayed to an inert, unused constant. Three of six LLM providers this
project carried at points (Groq — hit request-size limits once retrieval
moved to page-level chunking; Cerebras — rate limits/disconnects on ~10%
of a real eval run; a direct Anthropic integration — subsumed by serving
the same Claude models through AWS Bedrock instead).

## Bugs found and fixed (evidenced, not guessed)

Each of these was confirmed against a real filing and a real gold answer
before being called a bug:

- **Page-citation drift** — an early cover/TOC page with no page-number
  label caused the running page counter to drift permanently ahead;
  fixed with a resync check that trusts two consecutive real labels over
  a possibly-already-drifted counter. Verified against Nike's 2018 10-K:
  the balance sheet now cites page 45 (gold), not the drifted 48.
- **A parsing bug that silently turned safe abstains into penalized wrong
  answers** — the response parser required an exact string match against
  `"NOT_FOUND"`; a model that wrote it correctly and then kept writing an
  analysis paragraph had the whole blob graded as a real wrong answer.
  Fixed to check the answer's first line, not the whole string.
- **A cross-encoder reranker silently overriding domain-tuned scores** —
  fixed with a sigmoid-blended score instead of a full override.
- **A duplicate-disclosure retrieval trap** — segment data reported twice
  in a 10-Q (a concise MD&A table and a granular footnote copy) had
  nothing distinguishing them; confirmed against two real misses on the
  same filer (JPMorgan 2021Q1 and 2022Q2 10-Qs).
- **Non-calendar fiscal-year column confusion** — a three-column 10-Q
  balance sheet's prior-year-same-quarter column is easy to mistake for
  the fiscal-year-end column beside it; confirmed against a Best Buy
  cash-trend question that got the direction backwards.
- **A "(Gain)/(Loss) on X" sign-convention miss** — the caption word sets
  the true sign in an "Other (income)/deductions, net" rollup, not the
  parenthetical formatting; confirmed against a Pfizer question where an
  $8.1B gain was read as a loss.
- **A dead synonym-expansion code path** — a local financial-term-alias
  function was silently shadowed by an identically-named import, so it
  had never actually run; found by inspecting which function object the
  name was bound to at runtime.

Full architecture, formulas, notation, and the complete evidence trail for
every bug above: see `DOCUMENTATION.md` in the project repository.
