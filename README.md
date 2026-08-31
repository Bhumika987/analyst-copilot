# Analyst Copilot

A chatbot that answers analyst-style questions over SEC annual/quarterly filings
(10-K, 10-Q, 8-K `.htm` files from SEC EDGAR), returning a precise answer with
an exact page citation — or an honest "not found in this filing."

Built for a scoring rubric where a confidently wrong answer is worse than no
answer at all:

| System output | Score |
|---|---|
| Correct answer, correct page | **+1** |
| "Not found in this filing" | **0** |
| Correct answer, wrong page | **0** |
| Confidently wrong answer | **−1** |

A system that always abstains finishes at exactly zero; a system that
guesses finishes below zero. Every design decision here optimizes for that
asymmetry — the system is built to abstain rather than guess. Full
architecture, formulas, and measured results are in
**[DOCUMENTATION.md](DOCUMENTATION.md)** — this README is the quick-start
and submission checklist; that file is the one-page-and-beyond approach
note and technical writeup.

## What's in this submission

| Required | Status |
|---|---|
| Question-answering chatbot over filings, precise answer + exact page or honest "not found" | ✅ [`backend/llm.py`](backend/llm.py) — see [DOCUMENTATION.md §1–2](DOCUMENTATION.md#1-architecture) |
| "Add filing" control, visible progress, completes under 10 min | ✅ SSE progress events, `POST /api/filings/upload` |
| Chat box, plain English questions | ✅ [`frontend/index.html`](frontend/index.html) |
| Evidence (document + page) shown on every reply | ✅ clickable page citations, opens the full source page in-app |
| Ability to decline when evidence is weak/absent | ✅ rule-based gate runs before any LLM call — see [DOCUMENTATION.md §2.4](DOCUMENTATION.md#24-evidence-sufficiency-gate) |
| Git repo + README that runs it from scratch | ✅ this file |
| Running system, live for the session | see [DEPLOYMENT.md](DEPLOYMENT.md) |
| One-page approach note (tried / measured / kept / threw away) | ✅ standalone: [APPROACH_NOTE.pdf](APPROACH_NOTE.pdf) / [APPROACH_NOTE.md](APPROACH_NOTE.md) — condensed version also inline [below](#approach-note) |
| Practice set (136 Qs) as self-eval | ✅ [`scripts/evaluate.py`](scripts/evaluate.py), rubric implemented exactly — see [Running the evaluator](#running-the-evaluator) |

## Quick start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your LLM provider credentials**

`backend/llm.py` dispatches to one of four providers behind a single
`LLM_PROVIDER` env var: `fireworks` (default), `bedrock`, `bedrock_openai`,
`azure`. This list was deliberately narrowed from a wider set this project
carried at points during development — Groq, a direct Anthropic API
integration, and Cerebras were all tried and dropped: Groq started hitting
`413 Payload Too Large` once this codebase moved to page-level chunking
(request bodies grew past the free tier's size limit), Cerebras hit rate
limits/disconnects on roughly 10% of calls in a real eval run, and the
direct-Anthropic integration was subsumed by `bedrock`, which serves the
same Claude models billed through AWS instead of a separate Anthropic key.
Kept:

- **`fireworks`** (default) — OpenAI's open-weight gpt-oss-120b via
  Fireworks AI. Ran cleanly through a full 86-question batch with zero
  transport errors. Needs `FIREWORKS_API_KEY` (optionally `FIREWORKS_MODEL`,
  defaults to `accounts/fireworks/models/gpt-oss-120b`).
- **`bedrock`** — Claude models via Amazon Bedrock's native Messages route,
  authenticated with a Bedrock API key (no boto3/AWS SigV4 needed). Needs
  `AWS_REGION` and `AWS_BEARER_TOKEN_BEDROCK` (optionally `BEDROCK_MODEL`).
- **`bedrock_openai`** — the same gpt-oss-120b model as Fireworks, routed
  through Bedrock's OpenAI-compatible endpoint instead, billed through AWS.
  Same env vars as `bedrock` (optionally `BEDROCK_OPENAI_MODEL`).
- **`azure`** — an OpenAI-compatible model deployed on Azure OpenAI
  Service. Needs `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (e.g.
  `https://your-resource.openai.azure.com`), and `AZURE_OPENAI_DEPLOYMENT`
  (the deployment name you gave the model in Azure AI Foundry — Azure
  selects the model that way, not by a bare model id; optionally
  `AZURE_OPENAI_API_VERSION`, defaults to `2024-10-21`).

PowerShell, sticking with the default:
```powershell
$env:FIREWORKS_API_KEY = "your-fireworks-api-key-here"
```

Or Azure OpenAI:
```powershell
$env:LLM_PROVIDER = "azure"
$env:AZURE_OPENAI_API_KEY = "your-azure-openai-key-here"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "your-deployment-name"
```

**3. Run the server**

```bash
cd backend
python main.py
```

**4. Open the app**

Go to [http://localhost:8000](http://localhost:8000) in your browser.

- Click **Add Filing** to upload a `.htm` SEC filing (indexing completes in
  under 10 minutes; most filings take under a minute).
- Select a filing from the sidebar, then ask a question in the chat box.
- Every evidence citation is clickable — opens the full source page so you
  can verify the answer in context, not just take a quoted sentence on faith.

## Architecture (short version)

![Analyst Copilot architecture diagram](docs/architecture-diagram.png)

```
.htm filing → filing_parser.py (page-aware chunks)
            → retrieval.py (BM25 + FAISS, RRF fusion, deterministic +
              cross-encoder rerank)
            → llm.py (rule-based evidence gate → NOT_FOUND, or → LLM call
              → parsed answer + page citation)
            → main.py (FastAPI: upload/chat endpoints)
```

Full diagram (with the fusion/rerank math, notation, and code snippets for
every stage), the named-ratio formula table, and the scoring-criteria
detail: **[DOCUMENTATION.md §1–2](DOCUMENTATION.md#1-architecture)**.

### Why abstain-first?

Retrieval evidence is the gate, not the LLM's own confidence — an LLM asked
to self-rate its confidence tends to sound sure even when the source
material is thin. `evaluate_retrieval_status()` checks the top chunks
against rule-based requirements (does the requested concept, statement
type, fiscal year, and numeric value actually show up in what was
retrieved) and returns NOT_FOUND before the LLM is ever called if they
don't.

### What makes this different

See **[DOCUMENTATION.md §4](DOCUMENTATION.md#4-what-makes-this-different-outstanding-features)**
for the full case. In short: an abstain gate that's architectural, not
prompt-based; retrieval that understands accounting concepts and statement
types as first-class signals, not just semantic similarity; named-ratio
expansion for questions that never spell out their own formula, calibrated
against this dataset's actual gold answers; page citations verified
byte-for-byte against raw filing HTML; and a deployment story that survives
an ephemeral container disk via additive Postgres persistence.

## Project layout

```
.
├── backend/
│   ├── main.py             FastAPI app + endpoints
│   ├── filing_parser.py    .htm -> page-aware chunks (standalone runnable)
│   ├── retrieval.py        BM25 + FAISS hybrid search, reranking, per-filing index
│   ├── query_analyzer.py   query concept/statement/year extraction, used by
│   │                       both retrieval reranking and the evidence gate
│   ├── numerical_reasoner.py  structured evidence notes for calculation questions
│   ├── embedding_service.py   loads the BGE / FinLang sentence-transformer model
│   ├── vector_store.py     FAISS index wrapper
│   ├── config.py           central tunables (boosts, top-k depths, RRF params)
│   ├── ingest.py           parse -> index -> save pipeline (streaming)
│   ├── postgres_store.py   optional networked persistence (Postgres + pgvector),
│   │                       additive to the local file-based store
│   └── llm.py              multi-provider LLM dispatch, rule-based evidence gate,
│                           answer parsing
├── frontend/
│   └── index.html       single-file dark-themed chat UI
├── scripts/
│   ├── evaluate.py            scoring harness against practice-questions.jsonl
│   │                          (--from-postgres to source from the deployed
│   │                          Postgres store instead of local files)
│   └── backfill_postgres.py   one-time (re-runnable) local -> Postgres migration
├── data/
│   ├── filings/           provided SEC filings (gitignored - not code)
│   ├── practice-questions.jsonl   (gitignored - not code)
│   ├── indexes/           saved per-filing BM25/FAISS indexes (gitignored)
│   └── uploads/           uploaded filings (gitignored)
├── Dockerfile / .dockerignore   generic container build, any Docker-friendly host
├── DEPLOYMENT.md                deployment steps, incl. Postgres/Azure setup
├── DOCUMENTATION.md             full technical writeup — architecture, formulas,
│                                 notation, measured results, bug history
├── requirements.txt
└── .gitignore
```

## Running the evaluator

The evaluator reads `data/practice-questions.jsonl` and `data/filings/`.
Edit the path constants at the top of `scripts/evaluate.py` if your data
lives elsewhere.

```bash
python scripts/evaluate.py --limit 10
```

Useful flags:
- `--limit N` — only run the first N questions
- `--doc DOC_NAME` — only run questions for one filing
- `--no-embed` — BM25-only, skip dense embeddings (faster)
- `--from-postgres` — source every filing from Postgres (`DATABASE_URL`)
  instead of local files, to verify the deployed data path end to end.
  Requires `EMBEDDING_MODEL=normal`.

Results print per-question and as a summary, and are saved to
`eval_results.json`. Scoring criteria implemented exactly as in the table
at the top of this file — see [DOCUMENTATION.md §2.6](DOCUMENTATION.md#26-scoring-criteria-the-rubric-this-whole-system-is-optimized-for)
for the full rubric and [§5](DOCUMENTATION.md#5-measurement--evaluation-methodology-and-example-output)
for measured results and example output.

## Testing the API directly

```bash
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What was FY2018 capital expenditure for 3M?\", \"doc_name\": \"3M_2018_10K\", \"top_k\": 5}"
```

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
