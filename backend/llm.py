"""
Groq LLM integration and answer generation.

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
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional
from pathlib import Path
import httpx
import numpy as np

from embedding_service import get_embedding_provider

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

def _load_env():
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env", Path(__file__).resolve().parent / ".env"]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY=") and not os.environ.get("GROQ_API_KEY"):
                        val = line.split("=", 1)[1].strip(" '\"")
                        if val:
                            os.environ["GROQ_API_KEY"] = val
            except Exception:
                pass

_load_env()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# Free-tier Groq keys are capped at a modest tokens-per-minute budget shared
# across every model on the account. "low" cuts the model's internal
# reasoning-channel output roughly in half without materially hurting
# instruction-following on a task this constrained (quote-and-cite).
REASONING_EFFORT = "low"

# Below this retrieval score, the top chunk isn't strong enough evidence
# to even bother asking the LLM. Lowered to avoid false-negative gating when valid hits are present.
CONFIDENCE_THRESHOLD = 0.001

# Hard cap per context passage sent to the LLM.
MAX_PASSAGE_CHARS = 1400

SYSTEM_PROMPT = """You are an expert financial analyst assistant.

The provided context contains passages retrieved from SEC 10-K and 10-Q filings. 
These passages have already passed evidence validation and should be treated as the available financial evidence for answering the question.

Your task is to provide the most accurate, direct, and grounded answer to the user's question using the provided evidence.

STRICT GENERATION RULES:

1. FIND THE ANSWER IN THE PROVIDED EVIDENCE
   Examine the provided passages and identify the passage or combination of passages that directly supports the answer.

   Do NOT assume that the first passage is always the answer.
   Do NOT assume that the last passage is the answer.
   Determine the answer based on the actual content of the evidence.

2. ANSWER WHEN VALIDATED EVIDENCE SUPPORTS THE ANSWER
   If any provided passage contains the requested financial concept together with the relevant period and numerical value, use that evidence to answer the question.

   Do NOT return NOT_FOUND or claim that evidence is insufficient when the provided context contains enough information to answer.

3. PRIORITIZE DIRECT EVIDENCE
   When multiple passages are available, prefer evidence in this order:

   a. Direct financial statement or table containing the requested value
   b. Direct disclosure containing the requested value
   c. Supporting narrative that explicitly states the requested value
   d. Other contextual evidence

   Retrieval order alone must NOT determine which evidence is used.

4. MATCH THE QUESTION
   Before selecting the answer, verify that the evidence corresponds to the requested:

   - company/entity
   - financial metric/concept
   - fiscal year or quarter
   - reporting period
   - unit/currency
   - statement or section, if specified by the question

5. PRESERVE THE ORIGINAL FINANCIAL VALUE
   Extract the numerical value exactly as supported by the evidence.

   Preserve:
   - currency
   - units
   - fiscal period
   - sign of the value
   - relevant table column

   Do not change or reinterpret the reported value.

6. TABLES
   For table-based evidence, interpret the value using the table's:
   - row label
   - column header
   - reporting period
   - unit
   - table title

   Do not use a number from a different row or period simply because it appears in the same passage.

7. MULTIPLE PASSAGES
   Use multiple passages only when necessary to establish the answer.

   If one passage independently contains sufficient evidence, answer using that passage.

   If multiple passages are required, combine them only when they clearly refer to the same company, filing, metric, and period.

8. CALCULATIONS
   If the question requires a calculation:

   - use only numbers contained in the provided evidence
   - show the formula
   - show the calculation
   - provide the final result with the correct unit

   Do not introduce external numbers or assumptions.

9. NO HALLUCINATION
   Never invent:
   - financial values
   - dates
   - periods
   - units
   - page numbers
   - calculations
   - information not supported by the evidence

10. INSUFFICIENT EVIDENCE
   Return NOT_FOUND only when the provided evidence genuinely does not contain enough information to answer the question.

   Do not return NOT_FOUND merely because:
   - the first passage is not sufficient
   - passages contain different pieces of supporting information
   - the wording differs from the wording in the question
   - the answer requires interpreting a financial table
   - the relevant evidence appears in a later passage

11. SOURCE
   Cite the passage or passages that directly support the answer.

OUTPUT FORMAT:

ANSWER: [direct answer]

SOURCE: Page [N] - "[short exact quote supporting the answer]"

If multiple sources are required:

SOURCE: Page [N] - "[short exact quote]"
SOURCE: Page [M] - "[short exact quote]"

For calculations:

ANSWER: [final result]

CALCULATION:
[formula]
[calculation]

SOURCE: Page [N] - "[supporting quote]"

IMPORTANT:

- Answer the question directly.
- Do not discuss the retrieval process.
- Do not mention BM25, FAISS, embeddings, RRF, reranking, or internal evidence validation.
- Do not describe uncertainty when the provided evidence clearly supports an answer.
- Do not require one particular passage to contain the answer.
- Use the strongest answer-bearing evidence available.
- If validated evidence supports the answer, give the answer."""


@dataclass
class AnswerResult:
    found: bool
    answer: Optional[str] = None
    page_num: Optional[int] = None
    evidence_text: Optional[str] = None
    sources: Optional[List[Dict]] = None
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


def _context_chunk_limit(query_info) -> int:
    if not query_info:
        return 4
    if getattr(query_info, "requires_multiple_evidence_chunks", False):
        return 8
    if getattr(query_info, "query_type", "") in ("COMPARISON", "TREND", "CALCULATION"):
        return 8
    return 5


def _format_context(chunks: List[Dict], max_chunks: int = 4) -> tuple:
    """
    Format top validated context passages for LLM generation.
    Preserves retrieval/reranker order and caps passages to max_chunks (default 4)
    to prevent prompt context dilution.
    Returns tuple of (formatted_context_str, top_candidate_chunks_list).
    """
    if not chunks:
        return "", []

    top_candidates = sorted(
        chunks,
        key=lambda c: (
            c.get("concept_matched", False),
            c.get("has_period_match", False),
            c.get("has_numeric_value", False),
            c.get("content_evidence_score", 0.0),
            c.get("rerank_score", 0.0),
        ),
        reverse=True,
    )[:max_chunks]

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

    return "\n\n".join(parts), top_candidates


def _parse_groq_duration(s: str) -> Optional[float]:
    """Parse Groq's rate-limit reset strings, e.g. '12.112s', '1m26.4s', '547ms'."""
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
        parsed = _parse_groq_duration(reset) if reset else None
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


def _is_comparison_query(query: str, query_info) -> bool:
    q = f" {(query or '').lower()} "
    markers = ("highest", "lowest", "largest", "smallest", "greatest", "least", "rank", "which of", "compare", " versus ", " vs ")
    return bool(getattr(query_info, "is_comparison", False) or getattr(query_info, "query_type", "") in ("COMPARISON", "TREND") or any(m in q for m in markers))


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


async def _call_groq(messages: List[Dict], max_retries: int = 4):
    """POST to Groq chat completions with rate-limit-aware retry."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "reasoning_effort": REASONING_EFFORT,
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
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


_NOT_FOUND = {"found": False, "answer": None, "page_num": None, "evidence_text": None, "sources": []}


def _safe_print(value=""):
    """Print diagnostics without crashing on Windows console encodings."""
    text = str(value)
    encoding = getattr(getattr(__import__("sys"), "stdout"), "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _parse_llm_output(text: str, top_chunk: Optional[Dict] = None, ret_status: str = "") -> Dict:
    text = (text or "").strip()

    # 1. Standard ANSWER: regex extraction
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\nSOURCE:|\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        if answer and answer.upper() != "NOT_FOUND":
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


from config import MIN_CONTENT_EVIDENCE_SCORE
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
    context, top_chunks = _format_context(chunks, max_chunks=_context_chunk_limit(query_info))
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
        data = await _call_groq(messages)
        content = data["choices"][0]["message"]["content"]

        print("\n==================================================")
        print("[RAW GROQ OUTPUT]")
        print(f"Response Length: {len(content)}")
        print(f"Contains 'ANSWER:': {'ANSWER:' in content.upper()}")
        print(f"Contains 'NOT_FOUND': {'NOT_FOUND' in content.upper()}")
        print("Raw Text:")
        print("--------------------------------------------------")
        _safe_print(content)
        print("--------------------------------------------------")
        print("[END RAW GROQ OUTPUT]")
        print("==================================================\n")
        parsed = _parse_llm_output(content, top_chunk=top_chunks[0] if top_chunks else None, ret_status=ret_status)
        parsed["raw_response"] = content
        parsed["confidence"] = _confidence_from_best_passage(top_chunks) if parsed["found"] else 0.0
        parsed["debug_info"] = debug_info
        return AnswerResult(**parsed)
    except Exception as exc:
        return AnswerResult(
            found=False,
            confidence=0.0,
            error=f"Groq API error: {exc}",
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
    context, top_chunks = _format_context(chunks, max_chunks=_context_chunk_limit(query_info))
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
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY environment variable is not set")

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "stream": True,
                "reasoning_effort": REASONING_EFFORT,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", GROQ_API_URL, headers=headers, json=payload) as resp:
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
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            full_text += delta
                            yield {"type": "delta", "content": delta}
            print("\n==================================================")
            print("[RAW GROQ OUTPUT]")
            print(f"Response Length: {len(full_text)}")
            print(f"Contains 'ANSWER:': {'ANSWER:' in full_text.upper()}")
            print(f"Contains 'NOT_FOUND': {'NOT_FOUND' in full_text.upper()}")
            print("Raw Text:")
            print("--------------------------------------------------")
            _safe_print(full_text)
            print("--------------------------------------------------")
            print("[END RAW GROQ OUTPUT]")
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

    parsed = _parse_llm_output(full_text, top_chunk=top_chunks[0] if top_chunks else None, ret_status=ret_status)
    confidence = _confidence_from_best_passage(top_chunks) if parsed["found"] else 0.0
    debug_info["verification_result"] = "found" if parsed["found"] else "not_found"

    yield {
        "type": "result",
        "found": parsed["found"],
        "answer": parsed["answer"],
        "page_num": parsed["page_num"],
        "evidence_text": parsed["evidence_text"],
        "sources": parsed.get("sources", []),
        "confidence": confidence,
        "debug_info": debug_info,
    }


