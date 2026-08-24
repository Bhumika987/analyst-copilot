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

**2. Set your Groq API key**

PowerShell:
```powershell
$env:GROQ_API_KEY = "your-groq-api-key-here"
```

Get a free key at [console.groq.com](https://console.groq.com).

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
parser.py      — splits the filing into page-aware chunks. SEC EDGAR .htm
                  files use <hr/> tags as page breaks with the page number
                  in a small tag just before each break. Tables are kept
                  atomic (never split mid-table); long text sections become
                  350-word overlapping windows.
    │
    ▼
retrieval.py   — builds a per-filing hybrid index: BM25 (primary signal,
                  exact financial vocabulary/numbers) + FAISS dense search
                  (secondary, cheap hashed embeddings for paraphrase
                  recall), fused with Reciprocal Rank Fusion.
    │
    ▼
llm.py         — confidence-gated Groq call (openai/gpt-oss-120b).
                  If the top retrieved chunk's score is below
                  CONFIDENCE_THRESHOLD, the LLM is never even called —
                  we return NOT_FOUND immediately. When it is called, a
                  strict system prompt forbids using outside knowledge and
                  requires NOT_FOUND when the passages don't contain the
                  answer. Retries with exponential backoff on Groq 429s.
    │
    ▼
main.py        — FastAPI app: upload/index endpoint (SSE progress),
                  chat endpoint (SSE streaming answer + evidence).
```

`ingest.py` orchestrates parse → BM25 → embed → save, with progress events
for the UI. `scripts/evaluate.py` replays `practice-questions.jsonl` against
the running pipeline and scores it against the exact competition rubric.

### Why abstain-first?

Retrieval score is the gate, not the LLM's own confidence — an LLM asked to
self-rate its confidence tends to sound sure even when the source material is
thin. Gating on retrieval score means a weak match never reaches the model at
all, which is a cheaper and more reliable way to avoid a `-1`.

## Project layout

```
.
├── backend/
│   ├── main.py        FastAPI app + endpoints
│   ├── parser.py       .htm -> page-aware chunks (standalone runnable)
│   ├── retrieval.py    BM25 + FAISS hybrid search, per-filing index
│   ├── ingest.py        parse -> index -> save pipeline (streaming)
│   └── llm.py           Groq calls, confidence gating, embeddings
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

**What we tried:** Pure dense retrieval first — abandoned because our only
embedding option without a GPU or paid API is a hashed bag-of-words vector,
which is too weak to be a primary signal for numeric financial lookups where
exact terms ("capital expenditure", "$1,577") matter more than semantic
similarity. BM25 became the primary signal, with dense search kept only as a
secondary fusion input via RRF to catch paraphrased questions.

**What we measured:** Retrieval score distributions against the practice set
to set `CONFIDENCE_THRESHOLD` — the point below which the top chunk is too
weak to trust, so we abstain before ever calling the LLM. This is the single
biggest lever on the score, since every wrong answer costs twice what a right
answer earns.

**What we kept:** Page-aware chunking with atomic tables (a split table row
is a guaranteed wrong number), and a strict "quote the source or say
NOT_FOUND" system prompt.

**What we threw away:** LLM self-reported confidence as a gating signal — it
was overconfident even on thin evidence. Retrieval score is a more honest
gate than asking the model how sure it is.
