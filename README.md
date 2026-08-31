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

Defaults to Groq (free tier, no cost, but a shared per-minute rate limit).

PowerShell:
```powershell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```

Get a free key at [console.groq.com](https://console.groq.com).

To use Claude instead (paid, metered, no free tier -- but no shared rate-limit
ceiling and generally stronger reasoning on the calculation/judgment
questions), set `LLM_PROVIDER=claude` plus an Anthropic API key, either as
env vars or in `.env`:

```powershell
$env:LLM_PROVIDER = "claude"
$env:ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

Get a key at [console.anthropic.com](https://console.anthropic.com). Defaults
to `claude-haiku-4-5` (cheapest/fastest tier, no shared rate limit like
Groq's free tier); override with `CLAUDE_MODEL` (e.g. `claude-sonnet-5` or
`claude-opus-5`) if answer quality on harder calculation/judgment questions
matters more than cost per question.

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
                       via LLM_PROVIDER: Groq (openai/gpt-oss-120b, free
                       tier, default) or Claude (claude-haiku-4-5 by
                       default).
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
│   └── llm.py              Groq calls, rule-based evidence gate, answer parsing
├── frontend/
│   └── index.html       single-file dark-themed chat UI
├── scripts/
│   └── evaluate.py       scoring harness against practice-questions.jsonl
├── data/
│   ├── filings/           provided SEC filings (gitignored - not code)
│   ├── practice-questions.jsonl   (gitignored - not code)
│   ├── indexes/           saved per-filing BM25/FAISS indexes (gitignored)
│   └── uploads/           uploaded filings (gitignored)
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
