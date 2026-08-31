"""
Multi-provider LLM integration (Fireworks / AWS Bedrock / Azure OpenAI,
selected via LLM_PROVIDER) and answer generation.

Two safety mechanisms matter more than anything else here, because the
scoring rubric punishes a wrong answer (-1) far harder than it rewards a
right one (+1), while abstaining is always 0:

  1. CONFIDENCE GATING - if retrieval didn't find anything convincing, we
     never even call the LLM. No context worth reading in means no answer
     worth trusting out.
  2. STRICT PROMPTING - the model is told, repeatedly, to answer only from
     the provided passages and to say NOT_FOUND rather than guess.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional
from pathlib import Path
import httpx
import numpy as np

# This module prints extensive debug output (retrieval scope, context
# verification, raw LLM output, ...) straight to stdout. On Windows, stdout
# defaults to the console's codepage (cp1252) rather than UTF-8, and SEC
# filings routinely contain characters outside it -- private-use-area
# bullet glyphs (U+F0B7, a Wingdings-style bullet) being a confirmed real
# case. Without this, a single such character in any printed passage
# crashes the whole process with UnicodeEncodeError, killing an entire eval
# run over one cosmetic debug line. errors="replace" swaps the
# unencodable character for "?" instead of raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from embedding_service import get_embedding_provider

FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

# .env keys this module reads on startup, mirroring config.py's pattern for
# EMBEDDING_MODEL: first environment variable already set wins, .env only
# fills in what isn't.
_ENV_KEYS = (
    "LLM_PROVIDER",
    # Bedrock (both "bedrock" and "bedrock_openai"): AWS_REGION is required
    # (no fallback), and both providers call Bedrock's native HTTP endpoints
    # directly with an Amazon Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) --
    # no AWS access key pair, boto3, or IAM role needed.
    "AWS_REGION", "BEDROCK_MODEL", "AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_OPENAI_MODEL",
    "FIREWORKS_API_KEY", "FIREWORKS_MODEL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
)


def _load_env():
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent / ".env"]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    for key in _ENV_KEYS:
                        if line.startswith(f"{key}=") and not os.environ.get(key):
                            val = line.split("=", 1)[1].strip(" '\"")
                            if val:
                                os.environ[key] = val
            except Exception:
                pass

_load_env()
AWS_REGION = os.environ.get("AWS_REGION", "")
AWS_BEARER_TOKEN_BEDROCK = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

# Which LLM backend answer_question()/stream_answer() call. Three supported
# providers, chosen deliberately (see README's LLM provider section for the
# evidence behind this list -- other providers this codebase has carried at
# points, Groq/Claude-direct/Cerebras, were dropped: Groq hit request-size
# limits once this project moved to page-level chunking, Cerebras hit
# rate limits/disconnects on ~10% of a real eval run, and Claude-direct was
# subsumed by "bedrock", which serves the same Claude models billed through
# AWS instead of a separate Anthropic API key):
#   "fireworks"  -- OpenAI's open-weight gpt-oss-120b via Fireworks AI.
#   "bedrock"    -- Claude models via Amazon Bedrock's native Messages route.
#   "bedrock_openai" -- gpt-oss-120b via Bedrock's OpenAI-compatible route
#                    (same model as "fireworks", billed through AWS instead).
#   "azure"      -- an OpenAI-compatible model deployed on Azure OpenAI
#                    Service (deployment name, not a bare model id, selects
#                    the model -- see AZURE_OPENAI_DEPLOYMENT below).
# Both Bedrock providers authenticate with a Bedrock API key
# (AWS_BEARER_TOKEN_BEDROCK) over plain HTTPS -- no boto3/AWS SigV4
# credentials required.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "fireworks").strip().lower()
_SUPPORTED_PROVIDERS = ("fireworks", "bedrock", "bedrock_openai", "azure")
if LLM_PROVIDER not in _SUPPORTED_PROVIDERS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Supported values: {', '.join(_SUPPORTED_PROVIDERS)}"
    )

# Bedrock model IDs for Claude on the bedrock-runtime endpoint are the full
# dated snapshot ID, not the bare friendly name -- and in us-east-1
# specifically, only the "us."/"global."-prefixed cross-region inference
# profile works, not the bare in-region ID (confirmed against AWS's model
# card: verified this exact string against a live 400 "invalid model
# identifier" error before landing on it). Override via BEDROCK_MODEL in
# .env; check the Bedrock console's model catalog (Programmatic Access
# section on the model's page) if your region/account needs a different
# exact string.
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# gpt-oss-120b via Bedrock's OpenAI-compatible endpoint. AWS's actual model
# ID carries a version suffix ("-1:0") that the bare "openai.gpt-oss-120b"
# name lacks -- confirmed against a live 400 "invalid model identifier"
# error before landing on this exact string. Override via
# BEDROCK_OPENAI_MODEL in .env, e.g. "openai.gpt-oss-20b-1:0" for the
# smaller/cheaper/faster tier.
BEDROCK_OPENAI_MODEL = os.environ.get("BEDROCK_OPENAI_MODEL", "openai.gpt-oss-120b-1:0")

# gpt-oss-120b via Fireworks AI -- same model, second host. Override via
# FIREWORKS_MODEL in .env, e.g. "accounts/fireworks/models/gpt-oss-20b".
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/gpt-oss-120b")

# bedrock-runtime endpoints -- region-scoped, like every Bedrock endpoint.
# Built from AWS_REGION rather than hardcoded so both track whichever region
# the account's model access is enabled in. The Messages route mirrors
# Anthropic's own request/response shape (unlike Invoke/Converse); the Chat
# Completions route mirrors OpenAI's.
BEDROCK_MESSAGES_API_URL = (
    f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/anthropic/v1/messages" if AWS_REGION else ""
)
BEDROCK_OPENAI_API_URL = (
    f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1/chat/completions" if AWS_REGION else ""
)

# Azure OpenAI selects the model via a deployment name you create in the
# Azure AI Foundry / Azure OpenAI resource, not a bare model id -- that
# deployment name goes in the URL path, and the API version is a separate,
# explicit query param (Azure versions its REST surface independently of
# whatever model is behind the deployment). Override AZURE_OPENAI_API_VERSION
# in .env if your resource requires a different one.
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")


# "low" cuts the model's internal reasoning-channel output roughly in half
# without materially hurting instruction-following on a task this
# constrained (quote-and-cite) -- keeps responses fast and requests small
# across every provider above.
REASONING_EFFORT = "low"

# Hard cap per context passage sent to the LLM. Confirmed as a real, costly
# bug at 1400: a full Consolidated Balance Sheet table ran 2441 chars, and
# "Total current liabilities" -- the specific line a quick-ratio/current-
# ratio/working-capital question needed -- sat at char 1484, past the old
# cutoff. The model didn't hallucinate or misread; it never saw the line,
# because Assets always lists before Liabilities in these tables and the
# cut landed mid-statement. Any calculation spanning both sides of a
# balance sheet (which is most balance-sheet ratios) was exposed to this
# whenever the table ran long. Raised well past that table's length, with
# margin for larger ones (income statements with segment breakdowns, debt
# schedules) -- 8 chunks x ~3000 chars is still comfortably inside a
# 128K-token context window.
MAX_PASSAGE_CHARS = 3000

SYSTEM_PROMPT = """You are a financial analyst answering questions using evidence retrieved from SEC 10-K and 10-Q filings.

You will receive:

QUESTION:
{question}

EVIDENCE:
{evidence}

The supplied evidence is the only factual source you may use.

## Primary objective

Answer the question accurately, directly, and completely using the strongest available evidence.

Before answering, verify:

* the company, subsidiary, segment, or other entity
* the requested financial metric or concept
* the fiscal year, quarter, comparison period, or reporting date
* the currency and unit
* the table title, row label, and column heading when applicable
* whether the question requests a reported value, calculation, comparison, trend, or analytical conclusion

Do not select evidence based on passage order.

Many filers (retailers especially -- Best Buy, Nike, Target, and others in
this corpus) use a non-calendar fiscal year that ends in late January or
early February and is named for the calendar year in which MOST of it
falls, not the year it ends in. A balance sheet in a Q2 10-Q typically
carries three date columns: the current quarter-end, the PRIOR FISCAL
YEAR-END (a date in late Jan/early Feb, often 3-4 months before the
current quarter-end -- this is what "FYxxxx" means when a question asks
for a fiscal-year figure), and the same quarter one year earlier (an
interim date, not a fiscal year-end, even though it is also roughly a
year before the current column). Do not treat the prior-year same-quarter
column as "FYxxxx" merely because it is the most recent comparison
column reported -- verify which column's date is actually a fiscal
year-end before treating it as the requested fiscal year's figure.
Confirmed against a real miss: a Best Buy Q2 FY2024 10-Q reported cash as
of May 4 2024, February 3 2024, and April 29 2023 -- February 3 2024 is
the FY2023 year-end figure; April 29 2023 is an interim Q1 FY2024 figure
that happens to also be about a year earlier, and using it as "FY2023"
produced the wrong comparison and the wrong directional conclusion.

## Evidence priority

Prefer evidence in this order:

1. A financial statement or table directly containing the requested value
2. A footnote or direct filing disclosure
3. A narrative disclosure explicitly reporting the value or conclusion
4. Other context necessary to interpret the primary evidence

Use one passage when it independently answers the question.

Combine passages only when necessary, such as when:

* a calculation requires values from different statements
* a comparison requires multiple periods or segments
* an analytical question requires several financial indicators

Only combine values that concern the correct entity, metric, period, currency, and unit.

## Directly reported answers

When the requested answer is directly reported:

* reproduce the value accurately
* include its currency and scale when applicable
* identify the fiscal period or reporting date
* preserve negative signs, parentheses, percentages, and other meaningful notation
* use the exact table row and correct period column

Do not calculate a different metric when the requested metric is already directly reported unless the question explicitly requests recalculation.

A line item captioned "(Gain) on ..." or "(Loss) on ..." inside an
"Other income/(expense)" or "Other (income)/deductions, net" section has
its true sign set by the CAPTION WORD, not by whether the adjacent dollar
figure itself is shown in parentheses -- these sections roll up net
EXPENSE, so a real gain is entered as a subtraction (a negative expense)
and a real loss is entered as an addition. A caption of "(Gain) on
completion of X" paired with a parenthesized dollar amount is a gain that
INCREASES net income by that amount, not a loss that reduces it, even
though the figure is formatted exactly like a negative number. Read the
caption word first; use the parenthetical formatting only to confirm the
sign implied by that word, never to override it.

## Tables

Read financial tables using all of the following:

* table title
* row label
* column heading
* period
* currency
* scale or unit
* footnotes when necessary

Do not take a number from a neighboring row, column, or fiscal period.

A table may contain several related but distinct columns. For example, a sales-growth table may contain Organic, Acquisitions, Divestitures, Translation, and Total change. These figures are not interchangeable.

When a question asks for organic performance or excludes acquisitions, divestitures, or other M&A effects, use the Organic column. Do not select the Total column merely because it has the largest change.

## Calculations

When a calculation is required:

1. State the formula.
2. Identify the values used.
3. Substitute the values.
4. Perform the calculation.
5. Report the result with an appropriate unit and reasonable rounding.

Use only values present in the supplied evidence.

Do not:

* invent missing inputs
* silently mix thousands, millions, and billions
* mix values from incompatible periods
* mix consolidated and segment-level values
* substitute a related but different financial metric without explaining it
* treat percentages and percentage-point changes as equivalent

When values have different scales, convert them to a consistent scale before calculating.

Unless the question or evidence specifies otherwise:

* Ratio = numerator / denominator
* Percentage ratio = numerator / denominator × 100
* Percentage change = (new value - old value) / old value × 100
* Percentage-point change = new percentage - old percentage

When an average-balance formula is explicitly required, use the average of the beginning and ending balances. Otherwise, do not introduce an average balance unless the supplied formula or question requires it.

When the question spells out its own formula in words (e.g. "ROA is defined as: net income / average total assets"), report the result in that literal form and do not add a step the question didn't ask for. A metric that is conventionally expressed as a percentage (ROA, margins, growth rates) is NOT an exception: if the question's own formula has no "× 100" or "%" step, report the plain decimal it produces, rounded exactly as instructed -- do not convert it to a percentage on your own initiative. Confirmed against a real miss: a question defined ROA as a plain ratio and asked for it "rounded to two decimal places" -- the calculation itself was correct (0.01465) but got reported as "1.47%" instead of "0.01", which a grader checking the literal requested format marks wrong even though the arithmetic was right.

## Required formulas and fallback inputs

If the evidence contains `REQUIRED_RATIO_FORMULA`, use that formula.

If the formula specifies:

* preferred line items, and
* an allowed fallback when the preferred items are unavailable,

use the preferred calculation when all necessary inputs are present. Otherwise, use the stated fallback when its required inputs are present.

Do not return `NOT_FOUND` merely because the preferred breakdown is unavailable when the permitted fallback can be calculated.

Clearly identify when a fallback calculation was used.

## Comparisons and trends

For comparison questions:

* use comparable values for all periods, segments, or companies
* calculate both the absolute change and percentage change when useful
* distinguish increase/decrease from positive/negative values
* distinguish year-over-year change from sequential change
* do not call a change material, significant, strong, or weak unless the filing states it or the conclusion is clearly framed as an interpretation

For questions asking “why” a value changed, use only causes explicitly disclosed in the evidence. Do not infer an operational cause solely from the numerical change.

## Yes/no factual questions

For a factual yes/no question, answer “Yes” or “No” only when the evidence directly establishes the fact.

Briefly explain the supporting evidence.

If the evidence does not establish the fact, return `NOT_FOUND`.

## Analytical and classification questions

Some questions ask for an analytical assessment rather than a directly reported fact. Examples include:

* Is the company capital-intensive?
* Is liquidity improving?
* Is leverage high?
* Did profitability weaken?
* Is the company efficiently using its assets?
* Does the company appear financially healthy?

For these questions:

1. Calculate or identify the relevant financial indicators.
2. Use an explicit threshold or decision rule when one is supplied.
3. If no threshold is supplied, determine whether the evidence still supports a reasonable directional assessment.
4. If it does, provide a qualified conclusion and clearly state that it is an analytical interpretation, not a classification based on an explicit threshold.
5. If the evidence is insufficient even for a reasonable directional assessment, return `NOT_FOUND`.

Do not claim that a universal financial threshold exists when none was supplied. Financial classifications may depend on industry, business model, accounting treatment, and comparison period.

Use wording such as:

* “Based on the supplied FY2022 metrics, the company does not appear highly capital-intensive.”
* “The evidence suggests that liquidity improved.”
* “This is a directional assessment because no explicit classification threshold was supplied.”

Do not refuse to answer an analytical question solely because no numerical threshold was provided when the evidence contains the relevant metrics and supports a cautious conclusion.

Do not produce an absolute or universal conclusion when the evidence supports only a qualified interpretation.

## Capital-intensity questions

When evaluating capital intensity, use the indicators available in the evidence, which may include:

* capital expenditures divided by revenue
* net PP&E divided by total assets
* depreciation and amortization
* asset turnover
* return on assets
* capital expenditures relative to operating cash flow
* management’s description of capital requirements
* comparisons with prior periods or industry benchmarks when supplied

Do not classify a company using one raw dollar amount alone.

If the evidence supplies multiple relevant ratios but no threshold, provide a qualified assessment based on their combined direction.

For example, relatively modest CAPEX as a percentage of revenue, a limited proportion of assets held in net PP&E, and a positive ROA may support the qualified conclusion that the company does not appear highly capital-intensive. State that this is an analytical interpretation when no explicit benchmark is supplied.

## Accounting terminology

Treat related accounting terms carefully.

Do not automatically assume that:

* sales always equals total revenue
* net income always equals net income attributable to the company
* operating income equals adjusted operating income
* debt equals total liabilities
* cash equals cash and cash equivalents plus marketable securities
* purchases of PP&E always equals every possible definition of CAPEX
* book value equals market value
* basic EPS equals diluted EPS
* gross PP&E equals net PP&E

Use the term and value that best match the question. If a reasonable proxy is necessary and supported by the evidence, identify it explicitly.

## Negative values and cash-flow presentation

Parentheses in financial statements commonly indicate negative values or cash outflows.

Preserve the financial meaning while presenting calculations clearly. For example, when calculating CAPEX as a positive spending amount from a cash-flow line displayed as `(1,749)`, use $1,749 million as the magnitude and explain that it was reported as a cash outflow.

Do not accidentally produce a negative spending ratio solely because the statement displays the cash outflow in parentheses.

## Rounding and numerical equivalence

Use reasonable rounding based on the question and evidence.

Small rounding differences are acceptable when they arise from the same underlying values. For example:

* 5.11% may be reported as 5.1%
* 19.76% may be reported as 19.8% or approximately 20%
* 12.47% may be reported as 12.5%

Do not create false precision beyond what the evidence supports.

## Exhibit lists are not substantive evidence

An “Exhibit Index” or “Item 15 – Exhibits” table lists documents filed alongside the filing. It is not evidence of what those documents contain.

For example, an exhibit titled “Description of Registrant’s Securities” does not itself establish that a security is registered on a national exchange.

For questions about securities registered under Section 12(b), use:

* the filing cover-page table listing the security title, ticker symbol, and exchange, or
* an equivalent direct disclosure

If that evidence is absent, return `NOT_FOUND`. If the relevant table explicitly lists no securities, answer that there are none.

## Missing or insufficient evidence

Return `NOT_FOUND` only when the evidence lacks enough information to:

* answer the requested fact
* perform the required calculation
* make the requested comparison
* support even a qualified analytical conclusion

Before returning `NOT_FOUND`, check whether:

* the answer is directly disclosed
* the required values appear across multiple passages
* a supplied formula can be calculated
* an allowed fallback formula can be used
* a qualified analytical conclusion is supported

Do not use outside knowledge, assumptions, fabricated values, or unsupported thresholds.

## Citations

Cite every material reported value used in the answer.

For a calculation using values from multiple pages, provide a source for each required input.

Use short supporting quotations. Do not quote an entire table or long paragraph.

Use the page number attached to the relevant evidence passage. Do not infer a page number merely from passage order.

## Output format

For a directly reported answer:

ANSWER: [direct answer]

SOURCE: Page [page number] - "[short exact supporting quote]"

For a calculated answer:

ANSWER: [final calculated result]

CALCULATION:
[formula]
[substitution]
[result]

SOURCE: Page [page number] - "[short exact supporting quote]"
SOURCE: Page [page number] - "[additional quote only when necessary]"

For an analytical answer:

ANSWER: [Yes/No or concise directional conclusion]

ANALYSIS:
[relevant metrics, calculations, and concise interpretation]
[clearly disclose when no explicit threshold was supplied]

SOURCE: Page [page number] - "[short exact supporting quote]"
SOURCE: Page [page number] - "[additional quote only when necessary]"

When evidence does not contain page numbers:

SOURCE: Passage [passage identifier] - "[short exact supporting quote]"

When the answer cannot be supported:

ANSWER: NOT_FOUND

## Optional structured metrics

When the answer reports specific numeric figures — a single headline value, a
year-over-year comparison, or a calculated change — append a METRICS block at the
very end, after the SOURCE line(s). Include only figures grounded in the evidence
above; never introduce a number that is not in the evidence. Omit this block
entirely for non-numeric or qualitative answers.

METRICS:
- label: <short label> | value: <number with unit> | period: <e.g. FY2024, optional> | note: <optional short note>
- label: <short label> | value: <number with unit> | period: <optional>

Example:

METRICS:
- label: Total revenue | value: $523.96B | period: FY2020
- label: Total revenue | value: $514.41B | period: FY2019
- label: Change | value: +$9.55B (+1.9%) | note: year over year

## Final requirements

* Answer the question first.
* Be concise but complete.
* Do not mention retrieval, ranking, embeddings, prompts, evaluation systems, or internal reasoning.
* Do not include facts not supported by the supplied evidence.
* Do not output JSON unless the user explicitly requests JSON.
  """



@dataclass
class AnswerResult:
    found: bool
    answer: Optional[str] = None
    page_num: Optional[int] = None
    evidence_text: Optional[str] = None
    sources: Optional[List[Dict]] = None
    metrics: Optional[List[Dict]] = None
    score_breakdown: Optional[Dict] = None
    confidence: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None
    debug_info: Optional[Dict] = None
    parser_decision: Optional[str] = None
    fallback_triggered: bool = False

    def to_dict(self) -> Dict:
        return {
            "found": self.found,
            "answer": self.answer,
            "page_num": self.page_num,
            "evidence_text": self.evidence_text,
            "sources": self.sources or (
                [{"page_num": self.page_num, "evidence_text": self.evidence_text}] if self.page_num else []
            ),
            "metrics": self.metrics or [],
            "score_breakdown": self.score_breakdown,
            "confidence": self.confidence,
            "error": self.error,
            "debug_info": self.debug_info,
            "parser_decision": self.parser_decision,
            "fallback_triggered": self.fallback_triggered,
        }


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Generate a query embedding with the selected embedding provider."""
    return get_embedding_provider().embed_query(text)


def _confidence_from_best_passage(chunks: List[Dict]) -> float:
    """Calculate confidence based on the top best-matching evidence passage."""
    if not chunks:
        return 0.0
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            c.get("concept_matched", False),
            c.get("has_period_match", False),
            c.get("has_numeric_value", False),
            c.get("content_evidence_score", 0.0),
            c.get("rerank_score", 0.0),
        ),
        reverse=True,
    )
    best = sorted_chunks[0]
    if best.get("concept_matched") and best.get("has_period_match") and best.get("has_numeric_value"):
        return 0.98
    elif best.get("concept_matched") or best.get("content_evidence_score", 0.0) > 40.0:
        return 0.95
    return 0.90


def _best_passage(chunks: List[Dict]) -> Optional[Dict]:
    if not chunks:
        return None
    return sorted(
        chunks,
        key=lambda c: (
            c.get("concept_matched", False),
            c.get("has_period_match", False),
            c.get("has_numeric_value", False),
            c.get("content_evidence_score", 0.0),
            c.get("rerank_score", 0.0),
        ),
        reverse=True,
    )[0]


def _score_breakdown(found: bool, chunks: List[Dict], sources: List[Dict], overall: float) -> Optional[Dict]:
    """Three real retrieval signals behind the headline confidence, so the UI can
    show *why* an answer is trusted rather than a single opaque number. Every value
    here comes from the evidence gate -- nothing is invented.

    - answer_support:   how strongly the top passage matches the asked concept/period/value
    - location_accuracy: whether the citation resolved to a real page AND section
    - source_reliability: whether that passage is structured filing content (a statement
                          or table on a numbered page) rather than loose prose
    """
    if not found:
        return None
    best = _best_passage(chunks) or {}

    if best.get("concept_matched") and best.get("has_period_match") and best.get("has_numeric_value"):
        answer_support = 0.99
    elif best.get("concept_matched") or best.get("content_evidence_score", 0.0) > 40.0:
        answer_support = 0.95
    else:
        answer_support = 0.88

    src = (sources or [{}])[0]
    if src.get("location") and src.get("page_num"):
        location_accuracy = 0.98
    elif src.get("page_num"):
        location_accuracy = 0.80
    else:
        location_accuracy = 0.45

    ctype = (best.get("chunk_type") or "").lower()
    has_structure = ctype in ("table", "statement", "financial_statement") or bool(best.get("statement_type"))
    if best.get("page_num") and has_structure:
        source_reliability = 1.0
    elif best.get("page_num"):
        source_reliability = 0.90
    else:
        source_reliability = 0.70

    # Overall leans on answer support but is genuinely dragged down by a weak
    # location -- half the rubric is "correct location", so the headline number
    # should reflect that rather than parrot the coarse gate score.
    blended = 0.5 * answer_support + 0.3 * location_accuracy + 0.2 * source_reliability
    return {
        "answer_support": round(answer_support, 2),
        "location_accuracy": round(location_accuracy, 2),
        "source_reliability": round(source_reliability, 2),
        "overall": round(blended, 2),
    }


def _context_chunk_limit(query_info) -> int:
    # Raised from 8 to 10 alongside widening the retrieval top_k callers
    # request (8 -> 12, see main.py's ChatRequest default and evaluate.py) --
    # moving one without the other is a no-op: a wider candidate pool that
    # still gets sliced back down to 8 here never reaches the LLM, and a
    # higher ceiling here with no wider pool to draw from has nothing extra
    # to select. Two traced misses this session (a segment-comparison
    # question, a multi-line-item liability breakdown) needed evidence
    # ranked outside the old top-8; this doesn't guarantee either specific
    # case now succeeds (the evidence still has to rank in the top 10, and
    # for one of them it may not exist in any retrieved candidate at all),
    # but it's a genuine widening of the room comparison/calculation
    # questions have to work with, not just a page relabeled.
    if not query_info:
        return 4
    if getattr(query_info, "requires_multiple_evidence_chunks", False):
        return 10
    if getattr(query_info, "query_type", "") in ("COMPARISON", "TREND", "CALCULATION"):
        return 10
    return 5


def _select_top_candidates(chunks: List[Dict], max_chunks: int = 4) -> List[Dict]:
    """Take the top max_chunks from an already-ranked candidate list --
    this is the candidate POOL, not the final presentation order (a later
    step, the LLM relevance judge, reorders it; _render_context then trusts
    that order without re-sorting -- see its docstring for why re-sorting
    downstream is a confirmed bug in its own right).

    A plain slice, not a re-sort. It used to re-sort by
    (concept_matched, has_period_match, has_numeric_value,
    content_evidence_score, rerank_score) -- a second, independent, and
    CRUDER ranking than the one `chunks` already arrives in. chunks comes
    straight from hybrid_search(), whose own ranking already folds in
    cross-encoder scoring on top of everything in that tuple; re-deriving
    priority from a subset of the same fields, with content_evidence_score
    outweighing the cross-encoder-informed rerank_score in the sort order,
    could silently drop a chunk hybrid_search had ranked inside the
    candidate pool. Confirmed against a real miss: hybrid_search correctly
    placed a segment-breakdown table at position 11 of a 12-chunk pool
    (after a targeted rerank fix), but this re-sort dropped it back out
    before it ever reached the LLM -- evaluate_retrieval_status's
    equivalent selection (`chunks[:validation_limit]`) already used a
    plain slice and never had this problem."""
    if not chunks:
        return []
    return chunks[:max_chunks]


def _render_context(top_candidates: List[Dict]) -> str:
    """Format already-ordered candidate passages for LLM generation. Pure
    rendering -- does NOT re-sort, so whatever order top_candidates arrives
    in (heuristic pool selection, then LLM-relevance-judge reordering) is
    exactly what the model sees, including which passage gets tagged
    PRIMARY ANSWER-BEARING EVIDENCE."""
    if not top_candidates:
        return ""

    parts = []
    for i, c in enumerate(top_candidates, start=1):
        doc_name = c.get("doc_name", "Filing")
        company = c.get("company", "")
        filing_type = c.get("filing_type", "")
        fiscal_year = c.get("fiscal_year", "")
        page = c.get("page_num", "?")

        tag = "PRIMARY ANSWER-BEARING EVIDENCE" if i == 1 else "SUPPORTING CONTEXT"
        section = c.get("section") or ""
        subsection = c.get("subsection") or ""
        statement_title = c.get("statement_title") or ""
        statement_type = c.get("statement_type") or ""
        chunk_type = c.get("chunk_type") or ""
        table_title = c.get("table_title") or ""
        table_context = c.get("table_context") or ""
        units = c.get("units") or ""

        meta_parts = [tag, f"DOCUMENT: {doc_name}"]
        if company and company != "Unknown Company":
            meta_parts.append(f"COMPANY: {company}")
        if filing_type and filing_type != "Filing":
            meta_parts.append(f"FILING_TYPE: {filing_type}")
        if fiscal_year and fiscal_year != "Unknown":
            meta_parts.append(f"FILING_PERIOD: {fiscal_year}")
        if section:
            meta_parts.append(f"SECTION: {section}")
        if subsection:
            meta_parts.append(f"SUBSECTION: {subsection}")
        if statement_title:
            meta_parts.append(f"STATEMENT_TITLE: {statement_title}")
        if statement_type:
            meta_parts.append(f"STATEMENT: {statement_type}")
        if chunk_type:
            meta_parts.append(f"TYPE: {chunk_type}")
        if table_title:
            meta_parts.append(f"TABLE: {table_title}")
        if table_context:
            meta_parts.append(f"TABLE_CONTEXT: {table_context}")
        if units:
            meta_parts.append(f"UNIT: {units}")
        meta_parts.append(f"PAGE: {page}")

        meta_header = " | ".join(meta_parts)
        text = c.get("text", "")
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS] + " ...[truncated]"
        parts.append(f"[Passage {i} | {meta_header}]\n{text}")

    return "\n\n".join(parts)


def _parse_rate_limit_duration(s: str) -> Optional[float]:
    """Parse an x-ratelimit-reset-* duration string shared across the
    OpenAI-Chat-Completions-shaped backends this module talks to, e.g.
    '12.112s', '1m26.4s', '547ms'."""
    m = re.match(r"^(?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)$", s.strip())
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3) == "ms" else value
    return minutes * 60 + seconds


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        reset = response.headers.get("x-ratelimit-reset-tokens") or response.headers.get(
            "x-ratelimit-reset-requests"
        )
        parsed = _parse_rate_limit_duration(reset) if reset else None
        if parsed is not None:
            return min(parsed + 0.5, 60.0)
    return min(2 ** attempt, 30.0)


_VALIDATION_STOPWORDS = frozenset("""
a an the of in on for to by with from at as is are was were be been being
this that these those it its if then than or and but not no so such
do does did doing have has had will would can could may might must about into
what which who whom all any both each more most other some own same also
q1 q2 q3 q4 fy fiscal year quarter quarters period periods ended ending
highest lowest largest smallest greatest least higher lower rank ranked compare
compared versus vs amount value total figure give response relying details shown
name company firm corporation corp inc plc llc ltd co jpm jpmorgan
key agenda purpose filing filed dated date form document report reports reported
sec 8k 8 10k 10q 10 1st 2nd 3rd
january february march april may june july august september october november december
""".split())


def _validation_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9$.,%]+", (text or "").lower())
    result = []
    for token in tokens:
        clean = token.strip("$.,%")
        if len(clean) >= 3 and clean not in _VALIDATION_STOPWORDS and not clean.isdigit():
            result.append(clean)
    return list(dict.fromkeys(result))


def _metric_tokens_from_query(query: str, query_info) -> set:
    q_lower = (query or "").lower()
    metric_tokens = set()
    for phrase in getattr(query_info, "accounting_terms", []) or []:
        if phrase in q_lower:
            metric_tokens.update(_validation_tokens(phrase))
    return metric_tokens


def _comparison_grouping_terms(query: str, query_info) -> List[str]:
    metric_tokens = _metric_tokens_from_query(query, query_info)
    return [t for t in _validation_tokens(query) if t not in metric_tokens]


# "Which segment/product/region ... has/drove/dragged down ..." names a
# candidate-set noun without the superlative wording ("highest", "which of")
# the plain substring markers below catch -- it still asks the model to pick
# ONE answer out of several named entities, which needs the same
# wider/multi-entity retrieval a comparison gets, not the narrower default.
# Confirmed against a real miss: retrieval for "which segment has dragged
# down 3M's growth" surfaced only the top-matching segment (Health Care) and
# never the correct one (Consumer), because this phrasing wasn't recognized
# as a comparison across segments at all.
_WHICH_ENTITY_RE = re.compile(r"\bwhich\s+(segment|product|business|division|region|unit|category|line|subsidiary|market)\b")


def _is_comparison_query(query: str, query_info) -> bool:
    q = f" {(query or '').lower()} "
    markers = ("highest", "lowest", "largest", "smallest", "greatest", "least", "rank", "which of", "compare", " versus ", " vs ")
    return bool(
        getattr(query_info, "is_comparison", False)
        or getattr(query_info, "query_type", "") in ("COMPARISON", "TREND")
        or any(m in q for m in markers)
        or _WHICH_ENTITY_RE.search(q)
    )


def _chunk_search_text(c: Dict) -> str:
    return " ".join([
        c.get("section") or "",
        c.get("subsection") or "",
        c.get("statement_title") or "",
        c.get("table_title") or "",
        c.get("table_context") or "",
        c.get("statement_type") or "",
        c.get("chunk_type") or "",
        c.get("units") or "",
        c.get("text") or "",
    ]).lower()


def _concept_present_in_chunks(concept_id: str, chunks: List[Dict]) -> bool:
    concept_info = ACCOUNTING_CONCEPTS.get(concept_id) or {}
    terms = concept_info.get("keywords") or []
    for c in chunks:
        searchable = _chunk_search_text(c)
        if any(term in searchable for term in terms):
            return True
    return False


_DOCUMENT_PURPOSE_MARKERS = (
    "item information",
    "form type",
    "document description",
    "filed as of date",
    "conformed submission type",
    "event",
    "exhibit",
    "financial statements and exhibits",
)


async def _call_openai_compatible(
    url: str, api_key: str, model: Optional[str], missing_key_msg: str, messages: List[Dict],
    max_retries: int = 4, auth_header: str = "Authorization", auth_prefix: str = "Bearer ",
):
    """Shared POST logic for every OpenAI-Chat-Completions-shaped backend
    this module talks to (Bedrock's OpenAI-compatible route, Fireworks,
    Azure OpenAI) -- same JSON request/response shape, same rate-limit-aware
    retry. `auth_header`/`auth_prefix` exist because Azure authenticates
    with a plain 'api-key' header instead of 'Authorization: Bearer' like
    the other two; `model=None` omits the "model" field entirely, since
    Azure selects the model via the deployment name baked into the URL, not
    a payload field. Provider-specific wrappers below just supply what
    differs."""
    if not api_key:
        raise RuntimeError(missing_key_msg)

    headers = {
        auth_header: f"{auth_prefix}{api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "reasoning_effort": REASONING_EFFORT,
    }
    if model is not None:
        payload["model"] = model

    last_exc = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_retry_after_seconds(exc, attempt))
            continue
    raise last_exc


async def _call_fireworks(messages: List[Dict], max_retries: int = 4):
    return await _call_openai_compatible(
        FIREWORKS_API_URL, FIREWORKS_API_KEY, FIREWORKS_MODEL,
        "FIREWORKS_API_KEY environment variable is not set", messages, max_retries,
    )


def _split_system(messages: List[Dict]):
    """Claude (used here via Bedrock's native Messages route) takes 'system'
    as its own top-level request parameter, not a message with role='system'
    the way an OpenAI-Chat-Completions-shaped payload does."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    claude_messages = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), claude_messages


# adaptive thinking + output_config.effort only exist on this model tier;
# older/simpler models (Haiku 4.5, Sonnet 4.5, ...) reject both with a 400.
# Haiku 4.5 doesn't need either for a task this constrained (quote-and-cite,
# no multi-step reasoning) -- it's the cheap/fast tier by design.
_ADAPTIVE_THINKING_MODELS = frozenset({
    "claude-fable-5", "claude-mythos-5",
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
})


def _claude_extra_kwargs(model: str) -> Dict:
    # Bedrock model IDs carry an "anthropic." prefix that isn't part of the
    # bare model name the tier list below is keyed on.
    bare_model = model.split("anthropic.", 1)[-1] if model.startswith("anthropic.") else model
    if bare_model in _ADAPTIVE_THINKING_MODELS:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    return {}


def _require_bedrock_config(provider_name: str, api_url: str):
    if not AWS_BEARER_TOKEN_BEDROCK:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK environment variable is not set")
    if not api_url:
        raise RuntimeError(
            f"AWS_REGION environment variable is not set (required for LLM_PROVIDER={provider_name}; "
            "there is no default region fallback)"
        )


async def _call_bedrock(messages: List[Dict], max_retries: int = 4):
    """Call Claude via Amazon Bedrock's native Messages route, authenticated
    with a Bedrock API key (x-api-key bearer token) rather than AWS SigV4 --
    no boto3, no AWS access key pair, no IAM role. Same request/response
    shape as the first-party Anthropic API (the route mirrors it directly),
    reshaped into the same {"choices": [...]} shape every other provider in
    this module returns, so _parse_llm_output stays provider-agnostic."""
    _require_bedrock_config("bedrock", BEDROCK_MESSAGES_API_URL)
    system_text, claude_messages = _split_system(messages)

    headers = {
        "x-api-key": AWS_BEARER_TOKEN_BEDROCK,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": BEDROCK_MODEL,
        "max_tokens": 4096,
        "system": system_text,
        "messages": claude_messages,
        "stream": False,
        **_claude_extra_kwargs(BEDROCK_MODEL),
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(BEDROCK_MESSAGES_API_URL, headers=headers, json=payload)
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                resp.raise_for_status()
                data = resp.json()
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                return {"choices": [{"message": {"content": text}}]}
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(_retry_after_seconds(exc, attempt))
            continue
    raise last_exc


async def _call_bedrock_openai(messages: List[Dict], max_retries: int = 4):
    """POST to Bedrock's OpenAI-compatible chat completions endpoint -- same
    request/response shape as Fireworks, different bearer token and a
    region check up front."""
    _require_bedrock_config("bedrock_openai", BEDROCK_OPENAI_API_URL)
    return await _call_openai_compatible(
        BEDROCK_OPENAI_API_URL, AWS_BEARER_TOKEN_BEDROCK, BEDROCK_OPENAI_MODEL,
        "AWS_BEARER_TOKEN_BEDROCK environment variable is not set", messages, max_retries,
    )


def _require_azure_config():
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY environment variable is not set")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT environment variables "
            "are both required for LLM_PROVIDER=azure"
        )


def _azure_url() -> str:
    return (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )


async def _call_azure(messages: List[Dict], max_retries: int = 4):
    """POST to Azure OpenAI's Chat Completions endpoint. Doesn't reuse the
    default auth shape _call_openai_compatible assumes: Azure authenticates
    with a plain 'api-key' header (not 'Authorization: Bearer'), and the
    deployment name in the URL selects the model, so no "model" field
    belongs in the payload."""
    _require_azure_config()
    return await _call_openai_compatible(
        _azure_url(), AZURE_OPENAI_API_KEY, None,
        "AZURE_OPENAI_API_KEY environment variable is not set", messages, max_retries,
        auth_header="api-key", auth_prefix="",
    )


async def _call_llm(messages: List[Dict], max_retries: int = 4):
    """Dispatch to whichever backend LLM_PROVIDER selects."""
    if LLM_PROVIDER == "bedrock":
        return await _call_bedrock(messages, max_retries=max_retries)
    if LLM_PROVIDER == "bedrock_openai":
        return await _call_bedrock_openai(messages, max_retries=max_retries)
    if LLM_PROVIDER == "azure":
        return await _call_azure(messages, max_retries=max_retries)
    return await _call_fireworks(messages, max_retries=max_retries)


async def call_llm_raw(messages: List[Dict], max_retries: int = 4) -> str:
    """Public entry point for callers outside this module (currently
    scripts/evaluate.py's LLM-as-judge scorer) that just want a plain-text
    completion through whichever LLM_PROVIDER is configured, without
    reaching into _call_llm/the provider-specific response shape
    themselves."""
    data = await _call_llm(messages, max_retries=max_retries)
    choices = data.get("choices") or []
    return choices[0]["message"]["content"] if choices else ""


async def _stream_openai_compatible_deltas(
    url: str, api_key: str, model: Optional[str], missing_key_msg: str, messages: List[Dict],
    auth_header: str = "Authorization", auth_prefix: str = "Bearer ",
) -> AsyncGenerator[str, None]:
    """Shared streaming logic for every OpenAI-Chat-Completions-shaped
    backend. Raises (rather than retrying internally) on any failure --
    stream_answer's outer retry loop owns retries, since a mid-stream
    failure needs a fresh request anyway. See _call_openai_compatible for
    why `auth_header`/`auth_prefix`/`model=None` exist (Azure)."""
    if not api_key:
        raise RuntimeError(missing_key_msg)

    headers = {
        auth_header: f"{auth_prefix}{api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "stream": True,
        "reasoning_effort": REASONING_EFFORT,
    }
    if model is not None:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code == 429:
                raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # "choices" can be present but empty -- e.g. a trailing
                # usage-only frame some OpenAI-compatible hosts (Fireworks
                # included) send before [DONE] -- so `.get("choices", [{}])`
                # alone isn't enough; a present-but-empty list still needs
                # the fallback.
                choices = obj.get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta


def _stream_fireworks_deltas(messages: List[Dict]) -> AsyncGenerator[str, None]:
    return _stream_openai_compatible_deltas(
        FIREWORKS_API_URL, FIREWORKS_API_KEY, FIREWORKS_MODEL,
        "FIREWORKS_API_KEY environment variable is not set", messages,
    )


async def _stream_bedrock_deltas(messages: List[Dict]) -> AsyncGenerator[str, None]:
    """Yield text deltas from Claude via Bedrock's native Messages route.
    Parses Anthropic's own streaming SSE protocol by hand (content_block_delta
    events carrying text_delta pieces) -- the same events the Anthropic SDK's
    stream.text_stream parses internally, since this route mirrors the
    first-party API directly."""
    _require_bedrock_config("bedrock", BEDROCK_MESSAGES_API_URL)
    system_text, claude_messages = _split_system(messages)

    headers = {
        "x-api-key": AWS_BEARER_TOKEN_BEDROCK,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": BEDROCK_MODEL,
        "max_tokens": 4096,
        "system": system_text,
        "messages": claude_messages,
        "stream": True,
        **_claude_extra_kwargs(BEDROCK_MODEL),
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", BEDROCK_MESSAGES_API_URL, headers=headers, json=payload) as resp:
            if resp.status_code == 429:
                raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text


def _stream_bedrock_openai_deltas(messages: List[Dict]) -> AsyncGenerator[str, None]:
    """Yield text deltas from Bedrock's OpenAI-compatible endpoint -- same
    SSE shape as Fireworks, different bearer token. The region check runs
    eagerly (not lazily inside the generator) so a missing AWS_REGION
    surfaces before any request is attempted."""
    _require_bedrock_config("bedrock_openai", BEDROCK_OPENAI_API_URL)
    return _stream_openai_compatible_deltas(
        BEDROCK_OPENAI_API_URL, AWS_BEARER_TOKEN_BEDROCK, BEDROCK_OPENAI_MODEL,
        "AWS_BEARER_TOKEN_BEDROCK environment variable is not set", messages,
    )


def _stream_azure_deltas(messages: List[Dict]) -> AsyncGenerator[str, None]:
    """Yield text deltas from Azure OpenAI -- same SSE shape as Fireworks,
    'api-key' auth instead of 'Authorization: Bearer', no "model" field
    (the deployment name in the URL selects it). Config check runs eagerly,
    same reasoning as the Bedrock variant above."""
    _require_azure_config()
    return _stream_openai_compatible_deltas(
        _azure_url(), AZURE_OPENAI_API_KEY, None,
        "AZURE_OPENAI_API_KEY environment variable is not set", messages,
        auth_header="api-key", auth_prefix="",
    )


def _stream_llm_deltas(messages: List[Dict]) -> AsyncGenerator[str, None]:
    """Dispatch to whichever backend LLM_PROVIDER selects."""
    if LLM_PROVIDER == "bedrock_openai":
        return _stream_bedrock_openai_deltas(messages)
    if LLM_PROVIDER == "bedrock":
        return _stream_bedrock_deltas(messages)
    if LLM_PROVIDER == "azure":
        return _stream_azure_deltas(messages)
    return _stream_fireworks_deltas(messages)


_NOT_FOUND = {"found": False, "answer": None, "page_num": None, "evidence_text": None, "sources": []}


def _safe_print(value=""):
    """Print diagnostics without crashing on Windows console encodings."""
    text = str(value)
    encoding = getattr(getattr(__import__("sys"), "stdout"), "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _parse_metrics(text: str) -> List[Dict]:
    """Pull the optional trailing METRICS block into structured rows the UI can
    render as tiles. Purely a re-presentation of figures the model already put in
    the answer -- no numbers are computed or inferred here."""
    m = re.search(
        r"METRICS:\s*(.*?)(?:\n\s*(?:SOURCE:|ANSWER:|CALCULATION:|ANALYSIS:)|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    rows: List[Dict] = []
    for raw in m.group(1).splitlines():
        line = raw.strip().lstrip("-•*").strip()
        if not line or ":" not in line:
            continue
        fields: Dict[str, str] = {}
        for part in line.split("|"):
            key, sep, val = part.partition(":")
            if not sep:
                continue
            key = key.strip().lower()
            val = val.strip()
            if key in ("label", "value", "period", "note") and val:
                fields[key] = val[:80]
        if fields.get("label") and fields.get("value"):
            rows.append(fields)
        if len(rows) >= 6:
            break
    return rows


def _parse_llm_output(text: str, top_chunks: Optional[List[Dict]] = None, ret_status: str = "") -> Dict:
    text = (text or "").strip()
    top_chunk = top_chunks[0] if top_chunks else None

    # 1. Standard ANSWER: regex extraction
    # Stops only at a "SOURCE:" line or end of string -- NOT at the first
    # blank line. A blank-line stop truncates any answer the model formats
    # with a paragraph break before a bulleted breakdown (e.g. "verdict +
    # supporting ratios" style answers), silently dropping the actual
    # figures needed to score the answer as correct even when the model got
    # it right.
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n(?:SOURCE|METRICS):|$)", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        metrics = _parse_metrics(text)
        # A model that puts "ANSWER: NOT_FOUND" first and then keeps
        # writing (an ANALYSIS paragraph explaining the gap, say) instead
        # of stopping there violates the prompt's requested order, but the
        # signal is still an honest abstain -- checking the answer's FIRST
        # LINE against "NOT_FOUND" catches that; the old exact-full-string
        # check didn't, and treated "NOT_FOUND\n\nANALYSIS: ..." as a real
        # answer whose text happened to be that whole blob. Confirmed
        # against a real miss: this turned a genuine abstain (correctly
        # worth 0 under the scoring rubric) into a graded WRONG answer
        # (worth -1) purely because of response-formatting drift, not
        # because the model actually guessed wrong -- the one failure mode
        # the abstain-first design exists specifically to avoid.
        answer_head = answer.split("\n", 1)[0].strip().rstrip(".:").upper()
        if answer and answer_head != "NOT_FOUND":
            sources = []
            source_matches = re.findall(
                r"SOURCE:\s*(?:([^\n]*?)\s+)?Page\s*(\d+)\s*[-–—:]\s*[\"“”'']?(.+?)[\"“”'']?(?:\n|$)",
                text,
                re.IGNORECASE,
            )
            if source_matches:
                for doc, pnum, quote in source_matches:
                    sources.append({
                        "doc_name": doc.strip() if doc else None,
                        "page_num": int(pnum),
                        "evidence_text": quote.strip(),
                    })
            # The prompt's own fallback format ("SOURCE: Passage [N] - ...")
            # is meant for when a passage has no page number at all -- but
            # the model sometimes uses it even when the passage DOES have
            # one, citing the right passage by its list position instead of
            # its page. Confirmed as a real miss: "SOURCE: Passage 5 - ..."
            # correctly named the actual Cash Flow Statement passage, but
            # the page-only regex above found nothing, and the result fell
            # through to top_chunks[0]'s page -- an unrelated passage two
            # ranks and eleven pages away. Resolve Passage N back to that
            # passage's real page instead of losing the citation.
            if not sources and top_chunks:
                passage_matches = re.findall(
                    r"SOURCE:\s*Passage\s*(\d+)\s*[-–—:]\s*[\"“”'']?(.+?)[\"“”'']?(?:\n|$)",
                    text,
                    re.IGNORECASE,
                )
                for pnum_str, quote in passage_matches:
                    idx = int(pnum_str) - 1
                    if 0 <= idx < len(top_chunks):
                        c = top_chunks[idx]
                        sources.append({
                            "doc_name": c.get("doc_name"),
                            "page_num": c.get("page_num", 1),
                            "evidence_text": quote.strip(),
                        })
            if not sources and top_chunk:
                sources.append({
                    "doc_name": top_chunk.get("doc_name"),
                    "page_num": top_chunk.get("page_num", 1),
                    "evidence_text": (top_chunk.get("text") or "")[:150],
                })
            elif not sources:
                sources.append({"doc_name": None, "page_num": 1, "evidence_text": answer[:100]})

            res = {
                "found": True,
                "answer": answer,
                "page_num": sources[0]["page_num"],
                "evidence_text": sources[0]["evidence_text"],
                "sources": sources,
                "metrics": metrics,
                "parser_decision": "Stage 1 (Standard ANSWER: Regex Match)",
                "fallback_triggered": False,
            }
            print("\n==================================================")
            print("[PARSED LLM OUTPUT]")
            print(f"found: {res['found']}")
            _safe_print(f"answer: {res['answer']}")
            _safe_print(f"sources: {res['sources']}")
            print(f"parser_decision: {res['parser_decision']}")
            print(f"fallback_triggered: {res['fallback_triggered']}")
            print("==================================================\n")
            return res

    res = dict(_NOT_FOUND)
    res["parser_decision"] = "Failed (NOT_FOUND or unparseable LLM output)"
    res["fallback_triggered"] = False
    print("\n==================================================")
    print("[PARSED LLM OUTPUT]")
    print(f"found: {res['found']}")
    print(f"parser_decision: {res['parser_decision']}")
    print("==================================================\n")
    return res


from numerical_reasoner import extract_evidence_notes
from query_analyzer import analyze_query, ACCOUNTING_CONCEPTS, UNRELATED_TOPIC_KEYWORDS


def evaluate_retrieval_status(chunks: List[Dict], query: str = "") -> tuple:
    """
    Pure Rule-Based Evidence Validation Step.
    SUFFICIENT_EVIDENCE is not a score-threshold decision; evidence is sufficient ONLY
    when top retrieved context satisfies all applicable structural and content rules:
      1. Concept Rule: Contains requested normalized concept or accounting aliases
      2. Statement/Section Rule: Matches requested statement or section when applicable
      3. Period/Year Rule: Contains requested year/period when applicable
      4. Value/Numeric Rule: Contains numeric values/digits required for lookups or calculations
    """
    if not chunks:
        return "NO_EVIDENCE", "No candidate chunks were retrieved."

    query_info = analyze_query(query) if query else None

    # Evaluate enough final candidates to avoid rejecting valid multi-hop,
    # comparison, and multi-year evidence that is present just below rank 3.
    validation_limit = _context_chunk_limit(query_info) if query_info else 3
    top_chunks = chunks[:validation_limit]

    # Rule 1: Concept Rule (Primary Semantic Requirement for Concept-Specific Queries)
    if query_info and (query_info.accounting_terms or query_info.normalized_concepts):
        concept_found = False
        for c in top_chunks:
            text_lower = c.get("text", "").lower()
            tbl_title = (c.get("table_title") or "").lower()
            if any(term in text_lower or term in tbl_title for term in query_info.accounting_terms):
                concept_found = True
                break
        if not concept_found:
            concepts_str = ", ".join(query_info.normalized_concepts) if query_info.normalized_concepts else "requested concept"
            return "WEAK_EVIDENCE", f"Rule Failed [Concept Rule]: Top passages do not contain requested concept ({concepts_str}) or accounting aliases."

        if query_info.requires_calculation and len(query_info.normalized_concepts) > 1:
            missing = [
                concept_id
                for concept_id in query_info.normalized_concepts
                if not _concept_present_in_chunks(concept_id, top_chunks)
            ]
            if missing:
                return (
                    "WEAK_EVIDENCE",
                    f"Rule Failed [Calculation Inputs]: Missing required formula input evidence for {', '.join(missing)}.",
                )

    # Rule 2: Statement / Section Rule (ONLY enforced when user explicitly requested a specific statement/section)
    if query_info and query_info.explicitly_requested_statement != "ANY":
        statement_found = False
        req_st = query_info.explicitly_requested_statement
        for c in top_chunks:
            st_type = c.get("statement_type", "OTHER")
            section = (c.get("section") or "").upper()
            if st_type == req_st or req_st in section:
                statement_found = True
                break
            if c.get("concept_matched"):
                statement_found = True
                break
        if not statement_found:
            return "WEAK_EVIDENCE", f"Rule Failed [Statement Rule]: Top passages do not match explicitly requested statement type ({req_st})."

    if query_info and query_info.requires_calculation and getattr(query_info, "target_statement_types", None):
        missing_statements = []
        for statement_type in query_info.target_statement_types:
            if not any(c.get("statement_type") == statement_type for c in top_chunks):
                missing_statements.append(statement_type)
        if missing_statements:
            return (
                "WEAK_EVIDENCE",
                f"Rule Failed [Calculation Statements]: Missing required statement evidence for {', '.join(missing_statements)}.",
            )

    # Rule 2b: Comparison / Grouping Dimension Rule
    if query_info and _is_comparison_query(query, query_info):
        grouping_terms = _comparison_grouping_terms(query, query_info)
        if grouping_terms:
            matched_terms = set()
            for c in top_chunks:
                searchable = _chunk_search_text(c)
                matched_terms.update(t for t in grouping_terms if t in searchable)
            required = 1 if len(grouping_terms) <= 2 else 2
            if len(matched_terms) < required:
                return (
                    "WEAK_EVIDENCE",
                    "Rule Failed [Comparison Rule]: Top passages do not contain the requested comparison/grouping dimension.",
                )

    # Rule 2c: Document / Filing Purpose Rule
    if query_info and query_info.query_type == "DOCUMENT_PURPOSE":
        marker_found = False
        for c in top_chunks:
            searchable = _chunk_search_text(c)
            if any(marker in searchable for marker in _DOCUMENT_PURPOSE_MARKERS):
                marker_found = True
                break
        if not marker_found:
            return (
                "WEAK_EVIDENCE",
                "Rule Failed [Document Purpose Rule]: Top passages do not contain filing metadata or event/section evidence.",
            )

    # Rule 3: Period / Year Rule (when specific fiscal year requested)
    if query_info and query_info.target_years:
        year_found = False
        for c in top_chunks:
            searchable = _chunk_search_text(c)
            if any(yr in searchable for yr in query_info.target_years):
                year_found = True
                break
        if not year_found:
            return "WEAK_EVIDENCE", f"Rule Failed [Period Rule]: Top passages do not contain requested fiscal year ({', '.join(query_info.target_years)})."

    # Rule 4: Value / Numeric Rule (for numeric lookups and calculations)
    if query_info and query_info.query_type in ("NUMERIC_LOOKUP", "CALCULATION"):
        numeric_found = False
        target_yrs = set(query_info.target_years)
        for c in top_chunks:
            text = c.get("text", "")
            clean_text = re.sub(r"\b(?:note|item|page|part|section)\s+\d+\b", "", text, flags=re.IGNORECASE)
            num_matches = re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", clean_text)
            if any(n not in target_yrs for n in num_matches) or "$" in clean_text or "(" in clean_text:
                numeric_found = True
                break
        if not numeric_found:
            return "WEAK_EVIDENCE", "Rule Failed [Value Rule]: Top passages do not contain numerical values/digits required to answer the query."

    # If all applicable rules pass, evidence is sufficient!
    if query_info and query_info.query_type == "DOCUMENT_PURPOSE":
        return "SUFFICIENT_EVIDENCE", "Document-purpose evidence requirements passed."
    if query_info and _is_comparison_query(query, query_info):
        return "SUFFICIENT_EVIDENCE", "Comparison evidence requirements passed."
    return "SUFFICIENT_EVIDENCE", "All applicable rule-based evidence requirements passed."


_RELEVANCE_JUDGE_PROMPT = """You are evaluating retrieved passages from SEC filings for their relevance to a financial question.

Your task is to identify passages that contain evidence needed to answer the question accurately. Relevance means that a passage directly provides a requested fact, a required calculation input, a decision rule, or necessary supporting context.

QUESTION:
{question}

PASSAGES:
{passages}

## Relevance criteria

Evaluate each passage using all applicable criteria:

1. Entity

   * The passage must concern the requested company, subsidiary, segment, security, or other entity.
   * A value for a different entity is not relevant unless the question explicitly requests a comparison.

2. Metric

   * The passage must contain the requested metric, a valid synonym, or an input needed to calculate it.
   * Similar financial terms are not automatically interchangeable.
   * For example, revenue is not necessarily net income, operating cash flow is not free cash flow, and gross PP&E is not net PP&E.

3. Period

   * The passage must correspond to the requested fiscal year, quarter, reporting date, or comparison period.
   * A passage for another period should rank low unless it provides a required comparison value.

4. Row, column, and unit

   * For tables, verify the row label, column heading, period, currency, and unit.
   * Do not treat a nearby number from another row or column as the requested value.

5. Direct usefulness

   * Rank evidence containing the exact requested value above general discussion of the same topic.
   * Rank a primary financial statement or table above a vague narrative reference when both report the same information.

## Multi-passage questions

A question may require multiple values from different passages.

For example, calculating CAPEX divided by revenue may require:

* CAPEX from a cash-flow statement or capital-expenditure disclosure
* Revenue from an income statement

A passage should be considered directly relevant if it provides any necessary calculation input, even if it cannot independently answer the entire question.

Analytical questions may require several metrics. For example, assessing capital intensity may require CAPEX, revenue, PP&E, total assets, net income, or a stated decision rule. Do not reject a passage merely because it provides only one of the required inputs.

## Evidence priority

When passages are otherwise equally relevant, prefer:

1. Audited financial statements and financial tables
2. Direct filing disclosures
3. Accounting-policy or footnote disclosures
4. Management discussion explicitly reporting the requested fact
5. Other necessary supporting context

## Irrelevant or weak evidence

Rank a passage low or exclude it when it:

* merely shares vocabulary with the question
* concerns the wrong company, segment, security, or period
* contains the requested number in the wrong row or column
* mentions a metric without reporting the value or necessary context
* is only a table of contents
* is an Exhibit Index or list of exhibit filenames
* contains an exhibit title but not the contents of the exhibit
* provides general business commentary that does not support the requested conclusion
* duplicates stronger evidence without adding useful information

## Output

Return only valid JSON in this format:

{
"ranked_passages": [
{
"passage_number": 1,
"relevance": "high",
"reason": "Provides FY2022 capital expenditures used in the required calculation."
},
{
"passage_number": 4,
"relevance": "medium",
"reason": "Provides supporting context but not the primary requested value."
}
]
}

Requirements:

* Order passages from most relevant to least relevant.
* Use only "high", "medium", or "low" for relevance.
* Include only passages with some meaningful relevance.
* Do not answer the financial question.
* Do not calculate the final answer.
* Do not use information outside the supplied passages.
  """



async def _llm_relevance_rerank(question: str, chunks: List[Dict]) -> List[Dict]:
    """Mandatory relevance-judging pass over the already-retrieved
    candidates (BM25/FAISS/cross-encoder have already run by the time
    chunks reaches this point) -- an extra LLM call that asks "does this
    passage actually answer the question, not just share its vocabulary."
    Always runs; there is no flag to skip it. Confirmed against a real
    miss: an exhibit-index table ranked #1 by the existing lexical/semantic
    scorers purely because it mentioned "securities" and "exhibit," and the
    model then misread an exhibit filename as evidence of what it names
    (see the AMEX debt-securities case) -- a relevance judge that reasons
    about what a passage actually contains, not just what it mentions, is
    positioned to catch that a lexical reranker structurally cannot.

    Not making this optional means it always executes, not that it may
    never fail: on any call/parse failure it falls back to the untouched
    input order rather than raising, since a broken judge call must never
    take down answer generation.
    """
    if not chunks or len(chunks) < 2:
        return chunks

    passage_lines = []
    for i, c in enumerate(chunks, start=1):
        # 300 chars was confirmed too short to be useful: a real segment-
        # comparison passage's only discriminating figure (the specific
        # organic-growth percentage) sat past that cutoff, so the judge saw
        # two segments' passages as equally generic "segment performance"
        # text and declined to reorder them at all. 700 costs more per call
        # but stays well under the ~1400-char full passages sent to the
        # actual answer-generation step.
        text = (c.get("text") or "")[:700]
        label_parts = [f"[{i}]"]
        if c.get("table_title"):
            label_parts.append(f"TABLE: {c['table_title']}")
        if c.get("section"):
            label_parts.append(f"SECTION: {c['section'][:60]}")
        passage_lines.append(f"{' | '.join(label_parts)}\n{text}")
    passages_block = "\n\n".join(passage_lines)

    # Plain .replace(), not .format() -- the prompt's JSON output example
    # below (a literal {"ranked_passages": [...]}) contains unescaped
    # braces that .format() parses as placeholders and crashes on
    # (KeyError: '\n"ranked_passages"'). This substitutes only the two
    # named slots and is immune to any braces elsewhere in the prompt.
    prompt_text = _RELEVANCE_JUDGE_PROMPT.replace("{question}", question).replace("{passages}", passages_block)
    messages = [{"role": "user", "content": prompt_text}]

    try:
        data = await _call_llm(messages, max_retries=2)
        choices = data.get("choices") or []
        content = choices[0]["message"]["content"] if choices else ""
        order = [int(n) for n in re.findall(r"\d+", content or "")]
        seen = set()
        ranked = []
        for n in order:
            if 1 <= n <= len(chunks) and n not in seen:
                seen.add(n)
                ranked.append(chunks[n - 1])
        # Anything the judge's output didn't mention (truncated response,
        # skipped a number) still needs to reach context -- append it in
        # its original position rather than silently dropping it.
        for i, c in enumerate(chunks, start=1):
            if i not in seen:
                ranked.append(c)
        print(f"[LLM RELEVANCE JUDGE] reordered {len(chunks)} passages: {order}")
        return ranked
    except Exception as exc:
        print(f"[LLM RELEVANCE JUDGE] call failed, keeping original order: {exc}")
        return chunks


async def answer_question(question: str, doc_name: str, chunks: List[Dict]) -> AnswerResult:
    ret_status, ret_reason = evaluate_retrieval_status(chunks, query=question)
    print(f"RETRIEVAL_STATUS: {ret_status} ({ret_reason})")

    if ret_status in ("NO_EVIDENCE", "WEAK_EVIDENCE"):
        return AnswerResult(
            found=False,
            confidence=0.0,
            error=None,
            debug_info={
                "question": question,
                "retrieval_status": ret_status,
                "retrieval_reason": ret_reason,
                "num_chunks": len(chunks),
                "llm_called": False,
            }
        )

    query_info = analyze_query(question)
    top_chunks = _select_top_candidates(chunks, max_chunks=_context_chunk_limit(query_info))
    top_chunks = await _llm_relevance_rerank(question, top_chunks)
    context = _render_context(top_chunks)
    if not top_chunks or not context.strip():
        return AnswerResult(
            found=False,
            confidence=0.0,
            error="INSUFFICIENT_EVIDENCE: Context passages are empty or missing.",
            debug_info={
                "question": question,
                "retrieval_status": "INSUFFICIENT_EVIDENCE",
                "retrieval_reason": "Context passages provided to answer generator were empty.",
                "num_chunks": 0,
                "llm_called": False,
            }
        )

    chunk_ids = [c.get("chunk_idx") for c in top_chunks if c.get("chunk_idx") is not None]
    print("\n==================================================")
    print("[LLM CONTEXT VERIFICATION]")
    print(f"Retrieval Status: {ret_status}")
    print(f"Chunk IDs: {chunk_ids}")
    print(f"Number of Passages: {len(top_chunks)}")
    print(f"Total Context Character Length: {len(context)}")
    for idx_p, p_c in enumerate(top_chunks, start=1):
        print(f"--- [Passage {idx_p}] Chunk ID: {p_c.get('chunk_idx')} | Page: {p_c.get('page_num')} | Length: {len(p_c.get('text', ''))}")
        print(f"Text Snippet: {p_c.get('text', '')[:200]}...")
    print("==================================================\n")

    reasoning_notes = extract_evidence_notes(question, query_info, top_chunks)
    user_context = context
    if reasoning_notes:
        user_context = f"{reasoning_notes}\n\nSTRUCTURED CONTEXT PASSAGES\n\n{context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{user_context}\n\nQuestion: {question}"},
    ]

    source_filings = list(set(c.get("doc_name") for c in top_chunks if c.get("doc_name")))
    debug_info = {
        "question": question,
        "retrieval_status": ret_status,
        "retrieval_reason": ret_reason,
        "num_filings_searched": len(source_filings),
        "source_filings": source_filings,
        "llm_called": True,
        "validated_evidence_chunks": [
            {
                "chunk_idx": c.get("chunk_idx"),
                "doc_name": c.get("doc_name"),
                "section": c.get("section"),
                "subsection": c.get("subsection"),
                "statement_title": c.get("statement_title"),
                "table_title": c.get("table_title"),
                "table_context": c.get("table_context"),
                "page_num": c.get("page_num"),
                "statement_type": c.get("statement_type"),
                "chunk_type": c.get("chunk_type"),
                "content_evidence_score": c.get("content_evidence_score"),
                "rerank_score": c.get("rerank_score"),
                "concept_matched": c.get("concept_matched"),
                "text_snippet": (c.get("text") or "")[:200],
            }
            for c in top_chunks
        ],
        "final_context_sent": context,
        "deterministic_evidence_notes": reasoning_notes,
    }

    try:
        data = await _call_llm(messages)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response had no choices: {data}")
        content = choices[0]["message"]["content"]

        print("\n==================================================")
        print(f"[RAW {LLM_PROVIDER.upper()} OUTPUT]")
        print(f"Response Length: {len(content)}")
        print(f"Contains 'ANSWER:': {'ANSWER:' in content.upper()}")
        print(f"Contains 'NOT_FOUND': {'NOT_FOUND' in content.upper()}")
        print("Raw Text:")
        print("--------------------------------------------------")
        _safe_print(content)
        print("--------------------------------------------------")
        print(f"[END RAW {LLM_PROVIDER.upper()} OUTPUT]")
        print("==================================================\n")
        parsed = _parse_llm_output(content, top_chunks=top_chunks, ret_status=ret_status)
        parsed["raw_response"] = content
        parsed["confidence"] = _confidence_from_best_passage(top_chunks) if parsed["found"] else 0.0
        parsed["debug_info"] = debug_info
        parsed["sources"] = _enrich_sources(parsed.get("sources", []), top_chunks, doc_name)
        parsed["score_breakdown"] = _score_breakdown(
            parsed["found"], top_chunks, parsed["sources"], parsed["confidence"]
        )
        return AnswerResult(**parsed)
    except Exception as exc:
        return AnswerResult(
            found=False,
            confidence=0.0,
            error=f"{LLM_PROVIDER} API error: {exc}",
            debug_info=debug_info,
            parser_decision="LLM error; abstained",
            fallback_triggered=False,
        )


async def stream_answer(question: str, doc_name: str, chunks: List[Dict]) -> AsyncGenerator[Dict, None]:
    ret_status, ret_reason = evaluate_retrieval_status(chunks, query=question)
    print(f"\n[RETRIEVAL_STATUS]\nStatus: {ret_status}\nReason: {ret_reason}\n")

    if ret_status in ("NO_EVIDENCE", "WEAK_EVIDENCE"):
        yield {
            "type": "result",
            "found": False,
            "answer": None,
            "page_num": None,
            "evidence_text": None,
            "confidence": 0.0,
            "sources": [],
            "debug_info": {
                "question": question,
                "retrieval_status": ret_status,
                "retrieval_reason": ret_reason,
                "num_chunks": len(chunks),
                "llm_called": False,
            }
        }
        return

    query_info = analyze_query(question)
    top_chunks = _select_top_candidates(chunks, max_chunks=_context_chunk_limit(query_info))
    top_chunks = await _llm_relevance_rerank(question, top_chunks)
    context = _render_context(top_chunks)
    if not top_chunks or not context.strip():
        yield {
            "type": "result",
            "found": False,
            "answer": None,
            "page_num": None,
            "evidence_text": None,
            "confidence": 0.0,
            "sources": [],
            "debug_info": {
                "question": question,
                "retrieval_status": "INSUFFICIENT_EVIDENCE",
                "retrieval_reason": "Context passages provided to answer generator were empty.",
                "num_chunks": 0,
                "llm_called": False,
            }
        }
        return

    chunk_ids = [c.get("chunk_idx") for c in top_chunks if c.get("chunk_idx") is not None]
    print("\n==================================================")
    print("[LLM CONTEXT VERIFICATION]")
    print(f"Retrieval Status: {ret_status}")
    print(f"Chunk IDs: {chunk_ids}")
    print(f"Number of Passages: {len(top_chunks)}")
    print(f"Total Context Character Length: {len(context)}")
    for idx_p, p_c in enumerate(top_chunks, start=1):
        print(f"--- [Passage {idx_p}] Chunk ID: {p_c.get('chunk_idx')} | Page: {p_c.get('page_num')} | Length: {len(p_c.get('text', ''))}")
        print(f"Text Snippet: {p_c.get('text', '')[:200]}...")
    print("==================================================\n")

    reasoning_notes = extract_evidence_notes(question, query_info, top_chunks)
    user_context = context
    if reasoning_notes:
        user_context = f"{reasoning_notes}\n\nSTRUCTURED CONTEXT PASSAGES\n\n{context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{user_context}\n\nQuestion: {question}"},
    ]

    source_filings = list(set(c.get("doc_name") for c in chunks if c.get("doc_name")))
    debug_info = {
        "question": question,
        "retrieval_status": ret_status,
        "retrieval_reason": ret_reason,
        "num_filings_searched": len(source_filings),
        "source_filings": source_filings,
        "llm_called": True,
        "validated_evidence_chunks": [
            {
                "chunk_idx": c.get("chunk_idx"),
                "doc_name": c.get("doc_name"),
                "section": c.get("section"),
                "subsection": c.get("subsection"),
                "statement_title": c.get("statement_title"),
                "table_title": c.get("table_title"),
                "table_context": c.get("table_context"),
                "page_num": c.get("page_num"),
                "statement_type": c.get("statement_type"),
                "chunk_type": c.get("chunk_type"),
                "content_evidence_score": c.get("content_evidence_score"),
                "rerank_score": c.get("rerank_score"),
                "concept_matched": c.get("concept_matched"),
                "text_snippet": (c.get("text") or "")[:200],
            }
            for c in chunks
        ],
        "final_context_sent": context,
        "deterministic_evidence_notes": reasoning_notes,
    }

    full_text = ""
    max_retries = 4

    for attempt in range(max_retries):
        try:
            async for delta in _stream_llm_deltas(messages):
                full_text += delta
                yield {"type": "delta", "content": delta}
            print("\n==================================================")
            print(f"[RAW {LLM_PROVIDER.upper()} OUTPUT]")
            print(f"Response Length: {len(full_text)}")
            print(f"Contains 'ANSWER:': {'ANSWER:' in full_text.upper()}")
            print(f"Contains 'NOT_FOUND': {'NOT_FOUND' in full_text.upper()}")
            print("Raw Text:")
            print("--------------------------------------------------")
            _safe_print(full_text)
            print("--------------------------------------------------")
            print(f"[END RAW {LLM_PROVIDER.upper()} OUTPUT]")
            print("==================================================\n")
            break
        except Exception as exc:
            full_text = ""
            if attempt < max_retries - 1:
                await asyncio.sleep(_retry_after_seconds(exc, attempt))
                continue
            yield {"type": "result", "found": False, "answer": None, "page_num": None,
                   "evidence_text": None, "confidence": 0.0, "sources": [], "error": f"llm_error: {exc}", "debug_info": debug_info}
            return

    parsed = _parse_llm_output(full_text, top_chunks=top_chunks, ret_status=ret_status)
    confidence = _confidence_from_best_passage(top_chunks) if parsed["found"] else 0.0
    debug_info["verification_result"] = "found" if parsed["found"] else "not_found"

    enriched_sources = _enrich_sources(parsed.get("sources", []), top_chunks, doc_name)

    yield {
        "type": "result",
        "found": parsed["found"],
        "answer": parsed["answer"],
        "page_num": parsed["page_num"],
        "evidence_text": parsed["evidence_text"],
        "sources": enriched_sources,
        "metrics": parsed.get("metrics", []),
        "score_breakdown": _score_breakdown(parsed["found"], top_chunks, enriched_sources, confidence),
        "confidence": confidence,
        "debug_info": debug_info,
    }


_GENERIC_SECTIONS = {"", "general", "unknown", "n/a", "none"}


def _norm_words(text: str) -> List[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split() if len(w) > 2]


def _build_location(chunk: Dict) -> Optional[str]:
    """A readable "where in the filing" label, preferring the specific statement or
    table title over a generic chapter heading."""
    title = chunk.get("statement_title") or chunk.get("table_title")
    section = chunk.get("section") or ""
    subsection = chunk.get("subsection") or ""
    parts: List[str] = []
    if section.strip().lower() not in _GENERIC_SECTIONS:
        parts.append(section.strip())
    if subsection and subsection.strip().lower() not in _GENERIC_SECTIONS and subsection not in parts:
        parts.append(subsection.strip())
    if title and title not in parts:
        parts.append(title.strip())
    if not parts:
        return None
    loc = " — ".join(parts)
    return loc[:110] + "…" if len(loc) > 110 else loc


def _enrich_sources(sources: List[Dict], top_chunks: List[Dict], requested_doc: str) -> List[Dict]:
    """Attach the filing name and a human-readable location to each cited source by
    matching it back to the passage it came from.

    A *wrong* location scores the same as a wrong answer here, so the match must be
    earned: the location is only attached when the best-matching passage genuinely
    contains the cited quote (strong word overlap or a verbatim fragment). Otherwise
    we fill in the filing/page but stay silent on the section rather than guess."""
    if not sources:
        return sources

    def _best(src: Dict):
        page = src.get("page_num")
        want_doc = (src.get("doc_name") or "").strip().lower()
        qwords = _norm_words(src.get("evidence_text"))
        qfrag = re.sub(r"[^a-z0-9]", "", (src.get("evidence_text") or "").lower())[:18]
        best, best_score, best_overlap = None, 0.0, 0.0
        for c in top_chunks:
            score = 0.0
            if page is not None and c.get("page_num") == page:
                score += 5
            c_doc = (c.get("doc_name") or "").strip().lower()
            if want_doc and c_doc and (want_doc in c_doc or c_doc in want_doc):
                score += 4
            ctext = c.get("text") or ""
            ctext_norm = ctext.lower()
            overlap = 0.0
            if qwords:
                overlap = sum(1 for w in set(qwords) if w in ctext_norm) / len(set(qwords))
            score += 6 * overlap
            if qfrag and len(qfrag) >= 10 and qfrag in re.sub(r"[^a-z0-9]", "", ctext_norm):
                score += 3
                overlap = max(overlap, 0.5)
            if (c.get("chunk_type") or "").lower() in ("table", "statement", "financial_statement"):
                score += 0.5
            if score > best_score:
                best, best_score, best_overlap = c, score, overlap
        return best, best_overlap

    enriched = []
    for src in sources:
        out = dict(src)
        c, overlap = _best(src)
        if c and not out.get("doc_name"):
            out["doc_name"] = c.get("doc_name")
        if not out.get("doc_name") and requested_doc and requested_doc != "all":
            out["doc_name"] = requested_doc
        # Only claim a section when the passage really is the one quoted.
        if c and overlap >= 0.34:
            loc = _build_location(c)
            if loc:
                out["location"] = loc
        enriched.append(out)
    return enriched


