# Analyst Copilot

A chatbot that answers analyst-style questions over SEC annual/quarterly filings
(10-K, 10-Q, 8-K `.htm` files from SEC EDGAR), returning a precise answer with
an exact page citation — or an honest "not found in this filing."

Built for a scoring rubric where a confidently wrong answer is worse than no
answer at all: `+1` correct answer + correct page, `0` abstain, `0` correct
answer but wrong page, `-1` wrong answer. Every design decision here optimizes
for that asymmetry — the system is built to abstain rather than guess.

## Quick start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your LLM provider credentials**

`backend/llm.py` dispatches to one of four providers behind a single
`LLM_PROVIDER` env var: `fireworks` (default), `bedrock`, `bedrock_openai`,
`azure`. This list was deliberately narrowed from a wider set this project
carried at points during development -- Groq, a direct Anthropic API
integration, and Cerebras were all tried and dropped: Groq started hitting
`413 Payload Too Large` once this codebase moved to page-level chunking
(request bodies grew past the free tier's size limit), Cerebras hit rate
limits/disconnects on roughly 10% of calls in a real eval run, and the
direct-Anthropic integration was subsumed by `bedrock`, which serves the
same Claude models billed through AWS instead of a separate Anthropic key.
Kept:

- **`fireworks`** (default) -- OpenAI's open-weight gpt-oss-120b via
  Fireworks AI. Ran cleanly through a full 86-question batch with zero
  transport errors. Needs `FIREWORKS_API_KEY` (optionally `FIREWORKS_MODEL`,
  defaults to `accounts/fireworks/models/gpt-oss-120b`).
- **`bedrock`** -- Claude models via Amazon Bedrock's native Messages route,
  authenticated with a Bedrock API key (no boto3/AWS SigV4 needed). Needs
  `AWS_REGION` and `AWS_BEARER_TOKEN_BEDROCK` (optionally `BEDROCK_MODEL`).
- **`bedrock_openai`** -- the same gpt-oss-120b model as Fireworks, routed
  through Bedrock's OpenAI-compatible endpoint instead, billed through AWS.
  Same env vars as `bedrock` (optionally `BEDROCK_OPENAI_MODEL`).
- **`azure`** -- an OpenAI-compatible model deployed on Azure OpenAI
  Service. Needs `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (e.g.
  `https://your-resource.openai.azure.com`), and `AZURE_OPENAI_DEPLOYMENT`
  (the deployment name you gave the model in Azure AI Foundry -- Azure
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

## Architecture

```
.htm filing
    │
    ▼
filing_parser.py    — splits the filing into page-aware chunks. SEC EDGAR
                       .htm files use <hr/> tags as page breaks, with the
                       printed page number in a small tag right before each
                       break — that number labels the page that STARTS after
                       the break, not the one before it. Tables are kept
                       atomic (never split mid-table); each page/section-
                       scoped block of prose becomes one chunk rather than
                       being sliced into small overlapping windows, so a
                       single page's evidence doesn't get fragmented across
                       multiple retrieval units.
    │
    ▼
retrieval.py         — builds a per-filing hybrid index: BM25 (primary
                       signal, exact financial vocabulary/numbers) + FAISS
                       dense search (secondary, BAAI/bge-small-en-v1.5
                       sentence embeddings for paraphrase recall), fused
                       with Reciprocal Rank Fusion, then reranked (a local
                       cross-encoder when available, else deterministic
                       structural/boost scoring using query_analyzer.py's
                       concept/statement/year extraction).
    │
    ▼
llm.py               — evaluate_retrieval_status() is a rule-based gate
                       (concept, statement/section, period, and numeric-
                       value checks against the top chunks, via
                       query_analyzer.py) that runs before any LLM call. If
                       evidence fails those rules, we return NOT_FOUND
                       immediately and the LLM is never invoked. When it is
                       called, a strict system prompt forbids using outside
                       knowledge and requires NOT_FOUND when the passages
                       don't contain the answer. Provider is configurable
                       via LLM_PROVIDER: Fireworks (openai/gpt-oss-120b,
                       default), AWS Bedrock (Claude or gpt-oss-120b), or
                       Azure OpenAI.
                       Retries with exponential backoff on rate limits.
    │
    ▼
main.py              — FastAPI app: upload/index endpoint (SSE progress),
                       chat endpoint (SSE streaming answer + evidence).
```

`ingest.py` orchestrates parse → BM25 → embed → save, with progress events
for the UI. `scripts/evaluate.py` replays `practice-questions.jsonl` against
the running pipeline and scores it against the exact competition rubric.

### Why abstain-first?

Retrieval evidence is the gate, not the LLM's own confidence — an LLM asked
to self-rate its confidence tends to sound sure even when the source
material is thin. `evaluate_retrieval_status()` checks the top chunks
against rule-based requirements (does the requested concept, statement
type, fiscal year, and numeric value actually show up in what was
retrieved) and returns NOT_FOUND before the LLM is ever called if they
don't. Gating on retrieval evidence this way is a cheaper and more reliable
way to avoid a `-1` than trusting the model's self-reported confidence.

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
│   └── backfill_postgres.py   one-time (re-runnable) local -> Postgres migration
├── data/
│   ├── filings/           provided SEC filings (gitignored - not code)
│   ├── practice-questions.jsonl   (gitignored - not code)
│   ├── indexes/           saved per-filing BM25/FAISS indexes (gitignored)
│   └── uploads/           uploaded filings (gitignored)
├── Dockerfile / .dockerignore   generic container build, any Docker-friendly host
├── DEPLOYMENT.md                deployment steps, incl. Postgres/Azure setup
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

Results print per-question and as a summary, and are saved to
`eval_results.json`.

## Testing the API directly

```bash
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What was FY2018 capital expenditure for 3M?\", \"doc_name\": \"3M_2018_10K\", \"top_k\": 5}"
```

## Approach note

**What we tried:** Pure dense retrieval first — dropped as the primary
signal because even a real sentence-embedding model (BGE-small, run locally
via sentence-transformers) is too weak on its own for numeric financial
lookups, where exact terms ("capital expenditure", "$1,577") matter more
than semantic similarity. BM25 became the primary signal, with dense search
kept as a secondary fusion input via RRF to catch paraphrased questions, and
a cross-encoder reranker (or a deterministic structural/boost fallback when
the reranker model isn't available locally) on top.

**What we measured:** Retrieval score distributions against the practice
set — this is what led us to replace a single numeric confidence threshold
with `evaluate_retrieval_status()`'s rule-based checks (concept, statement
type, fiscal year, numeric value) in `llm.py`, so we abstain before ever
calling the LLM when the top chunks don't actually satisfy the question.
This is the single biggest lever on the score, since every wrong answer
costs twice what a right answer earns.

**What we kept:** Page-aware chunking with atomic tables (a split table row
is a guaranteed wrong number), and a strict "quote the source or say
NOT_FOUND" system prompt.

**What we threw away:** LLM self-reported confidence as a gating signal — it
was overconfident even on thin evidence. We also removed a single numeric
`CONFIDENCE_THRESHOLD` score-gate that had decayed to an inert 0.001 and was
no longer doing anything; `evaluate_retrieval_status()`'s rule-based checks
are the actual, live gate and a more honest one than either.

**Bugs found and fixed, evidenced against real filings and gold answers —
not guessed:**

- **Page-citation drift** (`filing_parser.py`). A printed page-number label
  in a SEC `.htm` filing belongs to the page that STARTS after the `<hr/>`
  it precedes, not the one that just ended — off by one in the obvious
  direction was the first bug found here. The second, subtler one: once an
  early cover/TOC page had no digit label, the parser advanced blindly, and
  every following *correct* sequential label then failed a too-strict
  "labels only move forward" bounds check and got rejected in turn — a
  one-way, uncorrectable drift for the rest of the document. Traced against
  raw HTML byte-by-byte (Nike 2018 10-K: the balance sheet table now lands
  on page 45, matching gold, not the drifted 48).
- **A parser bug that turned safe abstains into penalized wrong answers**
  (`llm.py`). The response parser required an *exact* string match against
  `"NOT_FOUND"` to recognize an abstain. A model that wrote
  `ANSWER: NOT_FOUND` correctly, then kept writing an analysis paragraph
  afterward (format drift, not a wrong guess), had that whole blob treated
  as a real answer — silently converting a `0` into a `-1` under the
  scoring rubric. This is the one failure mode the abstain-first design
  exists specifically to prevent, and it was happening at the parsing
  layer, invisibly. Fixed to check the answer's first line instead of the
  whole string.
- **Cross-encoder reranker silently overriding domain-tuned scores**
  (`retrieval.py`). The reranker's raw logit was replacing, not blending
  with, the deterministic concept/statement-type score — a lay-phrased
  MD&A mention could outrank the actual financial statement it was asking
  about. Fixed with a sigmoid-blended score instead of a full override.
- **A duplicate-disclosure retrieval trap.** Segment revenue/income is
  reported twice in a 10-Q — once as a concise MD&A summary table (what
  gold answers cite), again in far more granular form inside a numbered
  Note in the financial-statement footnotes. Nothing distinguished them, so
  "which segment had the lowest X" questions kept retrieving the Notes
  copy. Confirmed against two real misses on the same filer (JPMorgan
  2021Q1 and 2022Q2 10-Qs).
- **Non-calendar fiscal-year column confusion.** Several filers in this
  corpus (Best Buy, Nike, Target) name a fiscal year for the calendar year
  most of it falls in, ending in late Jan/early Feb — a three-column 10-Q
  balance sheet's "prior-year same quarter" column is easy to mistake for
  the fiscal-year-end column sitting right next to it. Confirmed against a
  Best Buy cash-trend question that got the direction backwards as a
  result.
- **Sign convention on "(Gain)/(Loss) on X" line items.** In an "Other
  (income)/deductions, net" rollup, the caption word sets the true sign,
  not the parenthetical formatting of the adjacent number — these sections
  total net *expense*, so a real gain is entered as a subtraction. Confirmed
  against a Pfizer question where an $8.1B gain (Consumer Healthcare JV
  transaction) was read as an $8.1B loss.
- **A dead code path that silently disabled financial-term synonym
  expansion** (`retrieval.py`). A local `expand_query()` (dividends/debt/
  equity/buyback aliases) was shadowed by an identically-named import from
  `query_analyzer` a few lines later — Python's own name resolution meant
  the synonym expansion had never actually been running. Found by
  introspecting which function object the name was actually bound to at
  runtime, not by reading the code alone.

**Net effect, measured on the hardest available slice** (the ~86 practice
questions that were not already scoring `+1`, re-run after these fixes): 5
questions flipped to fully correct that weren't before, and the aggregate
score on that subset moved from -16 to -11. Real, but modest relative to
the full 136-question set — this was always going to be true of a
deliberately hardest-case subset, and it's the reason a full clean run
(`EMBEDDING_MODEL=normal`, a provider without active rate-limit or payload
issues) against all 136 questions is the right next step for an honest
overall accuracy number, rather than treating this session's targeted fixes
as the final word on the score.
