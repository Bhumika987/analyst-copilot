"""
Per-filing hybrid retrieval: BM25 (primary) + selected FAISS dense embeddings,
fused with Reciprocal Rank Fusion.

BM25 is the signal that actually knows financial vocabulary ("capital
expenditure", "$1,577") token-for-token; the dense vectors catch paraphrases
BM25 misses, so RRF fusion
(rather than a weighted score blend) keeps BM25's ranking dominant without
needing to calibrate two incompatible score scales against each other.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"

_TOKEN_RE = re.compile(r"[a-z0-9$.,%]+")

_INDEX_CACHE: Dict[Tuple[str, str], "FilingIndex"] = {}
DOC_ROUTING_THRESHOLD = 3.0

# SEC filings use specific line-item wording ("Purchases of property, plant
# and equipment") that shares zero tokens with the analyst term for the same
# concept ("capital expenditure"). Plain BM25 can't bridge that gap on its
# own, so query-side synonym expansion adds the filing's likely wording
# before tokenizing. This only touches queries, never indexed chunk text.
FIN_SYNONYM_GROUPS = [
    ["capital expenditure", "capital expenditures", "capex",
     "purchases of property plant and equipment", "purchases of property"],
    ["revenue", "net sales", "total revenue", "net revenue", "total net sales"],
    ["cost of goods sold", "cost of sales", "cost of revenue", "cogs"],
    ["selling general and administrative", "sg&a", "sga expenses"],
    ["research and development", "r&d expense", "research development"],
    ["depreciation and amortization", "d&a"],
    ["net income", "net earnings", "net profit", "profit attributable"],
    ["operating income", "income from operations", "operating profit"],
    ["gross profit", "gross margin"],
    ["cash and cash equivalents", "cash equivalents"],
    ["stockholders equity", "shareholders equity", "shareholders' equity", "stockholders' equity"],
    ["long-term debt", "long term debt", "long-term borrowings"],
    ["dividends paid", "dividend payments", "cash dividends"],
    ["share repurchase", "stock repurchase", "buyback", "repurchase of common stock"],
    ["employees", "headcount", "number of employees"],
    ["free cash flow", "fcf"],
    ["total debt", "total borrowings"],
    ["interest expense", "interest paid"],
    ["income tax", "provision for income taxes", "income tax expense"],
]


def _expand_financial_synonyms(query: str) -> str:
    """Append likely filing-wording synonyms for any financial term the query mentions.

    Named with a leading underscore (was `expand_query`) because the
    unprefixed name collides with `query_analyzer.expand_query`, imported
    below -- that import silently rebinds `expand_query` in this module's
    namespace once the module finishes loading, so this function was never
    actually being called by `tokenize_query`/`search_bge_faiss` despite
    looking like it was. Both expansions are real and complementary (this
    one covers generic filing-wording synonyms; query_analyzer's covers
    query-type-specific retrieval terms), so both are composed explicitly
    at each call site now instead of relying on whichever name won the
    import order.
    """
    q_lower = query.lower()
    extra_terms = []
    for group in FIN_SYNONYM_GROUPS:
        if any(phrase in q_lower for phrase in group):
            for phrase in group:
                if phrase not in q_lower:
                    extra_terms.append(phrase)
    if not extra_terms:
        return query
    return query + " " + " ".join(extra_terms)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    return [t.strip(".,") if not t.startswith("$") and "%" not in t else t for t in tokens if t.strip(".,%$")]


# Analyst questions are wordy ("Give a response to the question by relying
# on the details shown in..."), and rank_bm25's negative-IDF fallback still
# assigns stopwords a small positive score. Summed over a dozen+ stopword
# hits, that reliably outweighs the handful of exact content-word matches a
# short, precise table chunk has - long prose chunks win on stopword volume
# alone. Corpus tokenization is untouched (document-length normalization
# should still see real chunk lengths); only the query is filtered.
_STOPWORDS = frozenset("""
a an the of in on for to by with from at as is are was were be been being
this that these those it its it's if then than or and but not no so such
do does did doing have has had having will would shall should can could may
might must about into over under again further here there when where why how
all any both each few more most other some own same too very just also
you your yours he him his she her hers they them their we our us i me my
what which who whom
""".split())

_RELEVANCE_STOPWORDS = _STOPWORDS | frozenset("""
q1 q2 q3 q4 fy fiscal year quarter quarters period periods ended ending
highest lowest largest smallest greatest least higher lower high low rank ranked
compare compared versus vs amount value total figure give response relying details
shown name company firm corporation corp inc plc llc ltd co jpm jpmorgan
key agenda purpose filing filed dated date form document report reports reported
sec 8k 8 10k 10q 10 1st 2nd 3rd
january february march april may june july august september october november december
""".split())

_COMPARISON_MARKERS = (
    "highest", "lowest", "largest", "smallest", "greatest", "least", "rank",
    "which of", "which segment", "which category", "which region", "which product",
    "compare", "versus", " vs ", "more than", "less than",
)


def tokenize_query(query: str) -> List[str]:
    tokens = tokenize(expand_query(_expand_financial_synonyms(query)))
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return filtered or tokens


def _is_comparison_intent(query: str, query_info: QueryAnalysis) -> bool:
    q = f" {query.lower()} "
    return query_info.is_comparison or query_info.query_type in ("COMPARISON", "TREND") or any(m in q for m in _COMPARISON_MARKERS)


def _distinctive_query_terms(query: str) -> List[str]:
    terms = []
    for token in tokenize(query):
        clean = token.strip("$.,%").lower()
        if len(clean) < 3 or clean in _RELEVANCE_STOPWORDS or re.fullmatch(r"\d+", clean):
            continue
        terms.append(clean)
    return list(dict.fromkeys(terms))


def _metric_terms_in_query(query: str, query_info: QueryAnalysis) -> set:
    q = query.lower()
    metric_terms = set()
    for phrase in query_info.accounting_terms:
        if phrase in q:
            metric_terms.update(_distinctive_query_terms(phrase))
    return metric_terms


def _candidate_metadata_text(c: Dict) -> str:
    parts = [
        c.get("section") or "",
        c.get("subsection") or "",
        c.get("statement_title") or "",
        c.get("table_title") or "",
        c.get("table_context") or "",
        c.get("statement_type") or "",
        c.get("chunk_type") or "",
        c.get("units") or "",
    ]
    text = c.get("text") or ""
    if "[TABLE]" in text:
        parts.extend(text.splitlines()[:6])
    return " ".join(parts).lower()


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


def _doc_identity_terms(c: Dict) -> set:
    parts = [
        c.get("doc_name") or "",
        c.get("company") or "",
        c.get("filing_type") or "",
        c.get("source_filename") or "",
        c.get("fiscal_year") or "",
    ]
    return {
        t
        for part in parts
        for t in _identity_tokens(part, min_len=2)
        if t and not t.isdigit()
    }


def _query_structure_features(query: str, query_info: QueryAnalysis, c: Dict) -> Dict:
    text_lower = (c.get("text") or "").lower()
    metadata_text = _candidate_metadata_text(c)
    searchable = f"{metadata_text} {text_lower}"

    query_terms = _distinctive_query_terms(query)
    metric_terms = _metric_terms_in_query(query, query_info)
    identity_terms = _doc_identity_terms(c)
    grouping_terms = [t for t in query_terms if t not in metric_terms and t not in identity_terms]

    metadata_matches = [t for t in grouping_terms if t in metadata_text]
    body_matches = [t for t in grouping_terms if t in text_lower]
    metric_matches = [t for t in metric_terms if t in searchable]

    grouping_match_count = len(set(metadata_matches + body_matches))
    grouping_coverage = grouping_match_count / max(1, len(grouping_terms))
    metric_matched = bool(metric_matches) or bool(c.get("concept_matched"))
    comparison_intent = _is_comparison_intent(query, query_info)
    is_table = c.get("chunk_type") == "table"
    table_header_matched = is_table and bool(set(metadata_matches + metric_matches))

    query_term_boost = min(20.0, len(set(metadata_matches)) * 6.0 + len(set(body_matches)) * 2.0)
    table_header_boost = 0.0
    if table_header_matched:
        table_header_boost += 15.0
        if grouping_match_count:
            table_header_boost += 10.0
    comparison_dimension_boost = 0.0
    if comparison_intent:
        if metric_matched and grouping_coverage >= 0.25:
            comparison_dimension_boost += 35.0
        if is_table and metric_matched and grouping_match_count:
            comparison_dimension_boost += 20.0
        elif not grouping_match_count:
            comparison_dimension_boost -= 10.0

    filing_purpose_matches = []
    filing_purpose_boost = 0.0
    if query_info.query_type == "DOCUMENT_PURPOSE":
        filing_purpose_matches = [m for m in _DOCUMENT_PURPOSE_MARKERS if m in searchable]
        if filing_purpose_matches:
            filing_purpose_boost = min(35.0, 12.0 + len(filing_purpose_matches) * 5.0)

    return {
        "comparison_intent": comparison_intent,
        "query_terms": query_terms,
        "doc_identity_terms": sorted(identity_terms),
        "metric_terms": sorted(metric_terms),
        "grouping_terms": grouping_terms,
        "matched_grouping_terms": sorted(set(metadata_matches + body_matches)),
        "metadata_grouping_matches": sorted(set(metadata_matches)),
        "metric_matches": sorted(set(metric_matches)),
        "grouping_coverage": grouping_coverage,
        "metric_matched": metric_matched,
        "table_header_matched": table_header_matched,
        "query_term_boost": query_term_boost,
        "table_header_boost": table_header_boost,
        "comparison_dimension_boost": comparison_dimension_boost,
        "filing_purpose_matches": filing_purpose_matches,
        "filing_purpose_boost": filing_purpose_boost,
    }


def _identity_tokens(text: str, min_len: int = 2) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) >= min_len]


def _years_in_text(text: str) -> set:
    return set(re.findall(r"(?<!\d)(20\d\d)", (text or "").lower()))


def _clean_company_identity(company: str) -> str:
    """Keep only the company name when SEC header text was flattened into metadata."""
    company = re.sub(r"\s+", " ", company or "").strip()
    if not company:
        return ""

    markers = [
        " central index key:",
        " standard industrial classification:",
        " irs number:",
        " state of incorporation:",
        " fiscal year end:",
        " filing values:",
        " form type:",
        " sec act:",
        " sec file number:",
        " film number:",
        " business address:",
        " mail address:",
        " former company:",
        " </sec-header>",
        " <document>",
    ]
    lower = company.lower()
    cuts = [lower.find(marker) for marker in markers if lower.find(marker) > 0]
    if cuts:
        company = company[:min(cuts)].strip()
    return company


def _rrf_fuse(weighted_ranked_lists: List[tuple], k: int = 60) -> Dict[int, float]:
    """Weighted Reciprocal Rank Fusion over multiple (ranked_list, weight) pairs."""
    scores: Dict[int, float] = {}
    for ranked, weight in weighted_ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
    return scores


from config import (
    BM25_TOP_K,
    SEMANTIC_TOP_K,
    RRF_K,
    RRF_TOP_K,
    RERANK_TOP_K,
    STATEMENT_TYPE_BOOST,
    TABLE_CHUNK_BOOST,
    CONCEPT_MATCH_BOOST,
    YEAR_MATCH_BOOST,
    DUAL_AGREEMENT_BOOST,
    NEIGHBOR_EXPANSION_ENABLED,
    NEIGHBOR_WINDOW_SIZE,
    ENABLE_CROSS_ENCODER_RERANKER,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_CANDIDATE_K,
    CROSS_ENCODER_LOCAL_ONLY,
    CROSS_ENCODER_BLEND_WEIGHT,
    get_embedding_model_name,
)
from embedding_service import get_embedding_service
from vector_store import FAISSVectorStore
from query_analyzer import QueryAnalysis, analyze_query, expand_query, UNRELATED_TOPIC_KEYWORDS

_CROSS_ENCODER = None
_CROSS_ENCODER_LOAD_FAILED = False


def _get_cross_encoder():
    """Lazy-load optional cross-encoder reranker."""
    global _CROSS_ENCODER, _CROSS_ENCODER_LOAD_FAILED
    if not ENABLE_CROSS_ENCODER_RERANKER or _CROSS_ENCODER_LOAD_FAILED:
        return None
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
        if CROSS_ENCODER_LOCAL_ONLY:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        _CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        return _CROSS_ENCODER
    except Exception as exc:
        print(f"Warning: Cross-encoder reranker unavailable ({exc}); using deterministic reranker only.")
        _CROSS_ENCODER_LOAD_FAILED = True
        return None


def _cross_encoder_text(c: Dict) -> str:
    return "\n".join(
        part for part in [
            f"Section: {c.get('section')}" if c.get("section") else "",
            f"Subsection: {c.get('subsection')}" if c.get("subsection") else "",
            f"Statement Title: {c.get('statement_title')}" if c.get("statement_title") else "",
            f"Statement: {c.get('statement_type')}" if c.get("statement_type") else "",
            f"Table: {c.get('table_title')}" if c.get("table_title") else "",
            f"Table Context: {c.get('table_context')}" if c.get("table_context") else "",
            f"Unit: {c.get('units')}" if c.get("units") else "",
            c.get("text") or "",
        ] if part
    )[:1800]


def cross_encoder_rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Optional semantic relevance refinement over the candidate evidence set.
    Falls back gracefully when the model is unavailable.

    The cross-encoder is a generic passage-relevance model with no notion of
    "this is the actual audited financial statement" vs. "this is a
    narrative table that happens to share vocabulary with the query" -- that
    distinction is exactly what deterministic_rerank's concept/statement-type
    scoring exists to make. So the cross-encoder's score REFINES the
    deterministic ranking (added on top, after squashing its unbounded logit
    into a bounded contribution via sigmoid) rather than replacing it as the
    sole sort key -- otherwise a lay-phrased MD&A mention can outrank the
    correct financial-statement table purely on generic semantic similarity.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    if model is None:
        return candidates

    rerank_pool = candidates[:CROSS_ENCODER_CANDIDATE_K]
    tail = candidates[CROSS_ENCODER_CANDIDATE_K:]
    pairs = [(query, _cross_encoder_text(c)) for c in rerank_pool]

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except TypeError:
        scores = model.predict(pairs)
    except Exception as exc:
        print(f"Warning: Cross-encoder rerank failed ({exc}); using deterministic order.")
        return candidates

    scored = []
    for c, score in zip(rerank_pool, scores):
        c_copy = dict(c)
        ce_score = float(score)
        ce_sigmoid = 1.0 / (1.0 + math.exp(-ce_score))
        c_copy["cross_encoder_score"] = ce_score
        c_copy["final_rerank_score"] = c_copy.get("rerank_score", 0.0) + ce_sigmoid * CROSS_ENCODER_BLEND_WEIGHT
        scored.append(c_copy)

    scored.sort(key=lambda c: c["final_rerank_score"], reverse=True)
    return scored + tail


def deterministic_rerank(query: str, candidates: List[Dict], query_info: Optional[QueryAnalysis] = None) -> List[Dict]:
    """
    Second-stage concept-primary evidence reranker.
    Makes normalized financial concepts the primary semantic signal for ranking:
    1. Concept Match Reward (+60.0): awarded when chunk explicitly matches requested concept or accounting aliases.
    2. Missing Concept Penalty (-40.0): penalizes chunks that match broad attributes (statement type, section, year)
       without matching the requested financial concept.
    3. Numeric & Structure Priorities (+20.0 each): rewards period match, table structure, and numeric value presence.
    4. Statement Metadata Supporting Signal (+15.0): metadata acts strictly as a supporting signal, never proof of relevance.
    """
    if not candidates:
        return []

    if query_info is None:
        query_info = analyze_query(query)

    concept_requested = bool(query_info.normalized_concepts or query_info.accounting_terms)

    reranked = []
    for c in candidates:
        text = c.get("text", "")
        text_lower = text.lower()
        tbl_title = (c.get("table_title") or "").lower()
        statement_text = f"{tbl_title} {text_lower}".replace("shee t", "sheet").replace("flow s", "flows")
        st_type = c.get("statement_type", "OTHER")

        content_evidence_score = c.get("retrieval_score", 0.0) * 100.0
        metadata_boost = 0.0
        concept_penalty = 0.0
        agreement_boost = 0.0

        # 1. Primary Semantic Signal: Accounting Concept Match (+60.0)
        concept_matched = False
        if query_info.accounting_terms:
            for term in query_info.accounting_terms:
                if term in text_lower or term in tbl_title:
                    concept_matched = True
                    content_evidence_score += 60.0
                    break
        elif query_info.normalized_concepts:
            concept_matched = True
            content_evidence_score += 60.0

        # 2. Down-rank Chunks Lacking Requested Concept (-40.0 Penalty)
        # Prevents generic balance-sheet / income-statement chunks from outranking concept-relevant hits
        if concept_requested and not concept_matched:
            concept_penalty = 40.0
            content_evidence_score -= concept_penalty

        # 3. Target Fiscal Year / Period Alignment (+20.0)
        has_period_match = False
        if query_info.target_years:
            for yr in query_info.target_years:
                if yr in text_lower:
                    has_period_match = True
                    content_evidence_score += 20.0
                    break

        # 4. Tabular Numeric Value Presence (+20.0)
        has_numeric_value = bool(re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text))
        if has_numeric_value:
            content_evidence_score += 20.0

        # 5. Dual-Retriever Agreement Bonus (+20.0)
        if c.get("bm25_rank") is not None and c.get("semantic_rank") is not None:
            agreement_boost = DUAL_AGREEMENT_BOOST
            content_evidence_score += agreement_boost
        elif c.get("chunk_type") == "table" and c.get("bm25_rank") is not None and c.get("bm25_rank") <= 10:
            # Dense numeric tables ("Consumer | (0.9) | ...") systematically
            # embed as semantically dissimilar from a natural-language
            # question, even when they ARE the exact right evidence --
            # confirmed against a real miss where a segment-breakdown table
            # ranked #9 in BM25 (clearly lexically relevant, and the actual
            # answer-bearing chunk) never entered FAISS's top 50 at all,
            # purely because it's mostly numbers with almost no narrative
            # English to embed -- not because it's actually unrelated to
            # the question. That meant it lost the full +20 agreement bonus
            # a narrative chunk with identical BM25/table/query-term scores
            # got, on a signal (semantic similarity) that structurally
            # can't represent this kind of evidence well in the first
            # place. Partial (not full) compensation, and only for chunks
            # BM25 already independently ranks as strong -- this corrects a
            # specific, identified blind spot in one of the two aggregated
            # retrieval signals, not a general table preference.
            agreement_boost = DUAL_AGREEMENT_BOOST * 0.75
            content_evidence_score += agreement_boost

        # 6. Explicit vs Inferred Statement Supporting Boosts
        if query_info.explicitly_requested_statement != "ANY":
            sec_upper = (c.get("section") or "").upper()
            if st_type == query_info.explicitly_requested_statement or query_info.explicitly_requested_statement in sec_upper:
                metadata_boost += 25.0
            if query_info.explicitly_requested_statement == "BALANCE_SHEET" and (
                "consolidated balance sheet" in statement_text or
                ("current assets" in statement_text and "total assets" in statement_text and "total liabilities" in statement_text)
            ):
                metadata_boost += 60.0
            elif query_info.explicitly_requested_statement == "CASH_FLOW_STATEMENT" and (
                "consolidated statement of cash flows" in statement_text or
                ("cash flows from operating activities" in statement_text and "cash flows from investing activities" in statement_text)
            ):
                metadata_boost += 60.0
            elif query_info.explicitly_requested_statement == "INCOME_STATEMENT" and (
                "consolidated statement of income" in statement_text or
                "consolidated statement of operations" in statement_text
            ):
                metadata_boost += 60.0
        elif query_info.inferred_statement != "ANY" and st_type == query_info.inferred_statement:
            metadata_boost += 15.0

        target_statements = set(getattr(query_info, "target_statement_types", []) or [])
        if target_statements:
            if st_type in target_statements:
                metadata_boost += 35.0
            elif st_type in {"EQUITY_STATEMENT", "RISK_FACTORS", "BUSINESS"}:
                metadata_boost -= 45.0
            elif st_type == "FOOTNOTES" and query_info.requires_calculation:
                metadata_boost -= 20.0

        if query_info.requires_calculation and any(topic in statement_text for topic in UNRELATED_TOPIC_KEYWORDS):
            metadata_boost -= 25.0

        # 7. Supporting Signal: Table Chunk Type (+15.0)
        if query_info.query_type in ("NUMERIC_LOOKUP", "CALCULATION") and c.get("chunk_type") == "table":
            metadata_boost += 15.0

        # 7b. Superlative/"which segment" comparison tie-break: MD&A summary
        # tables vs Notes/footnote duplicates. Confirmed against a real miss
        # (JPM 2021Q1 10-Q, 2022Q2 10-Q): segment revenue/income is
        # disclosed twice -- once in a concise MD&A "Business Segment
        # Results" summary table (the page gold answers actually cite), and
        # again, in far more granular form, inside a numbered Note ("Note
        # 25 - Business segments") deep in the financial-statement
        # footnotes. Both match segment/revenue keywords equally well, so
        # nothing previously distinguished them, and retrieval kept picking
        # the Notes copy -- much longer and looser as an argmax source for
        # "which segment had the lowest/highest X" questions. Break the tie
        # toward the MD&A summary table explicitly.
        if any(k in query.lower() for k in ("which segment", "which category", "which region", "which product", "which business", "which division")):
            subsection_lower = (c.get("subsection") or "").lower()
            if re.match(r"^\s*note\s+\d+\b", subsection_lower):
                metadata_boost -= 30.0
            if "segment result" in tbl_title or "segment result" in statement_text:
                metadata_boost += 30.0

        structure_features = _query_structure_features(query, query_info, {**c, "concept_matched": concept_matched})
        structure_boost = (
            structure_features["query_term_boost"] +
            structure_features["table_header_boost"] +
            structure_features["comparison_dimension_boost"] +
            structure_features["filing_purpose_boost"]
        )
        metadata_boost += structure_boost

        total_score = content_evidence_score + metadata_boost

        c_copy = dict(c)
        c_copy["concept_matched"] = concept_matched
        c_copy["has_period_match"] = has_period_match
        c_copy["has_numeric_value"] = has_numeric_value
        c_copy["concept_penalty"] = concept_penalty
        c_copy["agreement_boost"] = agreement_boost
        c_copy["content_evidence_score"] = max(0.0, content_evidence_score)
        c_copy["metadata_boost"] = metadata_boost
        c_copy["query_term_boost"] = structure_features["query_term_boost"]
        c_copy["metric_match_boost"] = 60.0 if concept_matched else 0.0
        c_copy["grouping_match_boost"] = structure_features["comparison_dimension_boost"]
        c_copy["table_header_boost"] = structure_features["table_header_boost"]
        c_copy["filing_purpose_boost"] = structure_features["filing_purpose_boost"]
        c_copy["filing_purpose_matches"] = structure_features["filing_purpose_matches"]
        c_copy["comparison_intent"] = structure_features["comparison_intent"]
        c_copy["matched_grouping_terms"] = structure_features["matched_grouping_terms"]
        c_copy["metric_matches"] = structure_features["metric_matches"]
        c_copy["grouping_coverage"] = structure_features["grouping_coverage"]
        c_copy["rerank_score"] = total_score
        reranked.append(c_copy)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def _ensure_statement_coverage(
    top_chunks: List[Dict],
    ranked_pool: List[Dict],
    query_info: Optional[QueryAnalysis],
    top_k: int,
) -> List[Dict]:
    """
    Guarantee every statement type a calculation question needs is present in
    the final slice, not just the single best-matching one.

    A ratio question spanning two statements (e.g. fixed-asset-turnover =
    revenue / average PP&E, needing both the income statement and the
    balance sheet) tends to have one side dominate BM25/rerank scoring --
    "revenue" terms are common and match many chunks, so every slot in a
    small top-k can fill with income-statement chunks while the balance-sheet
    PP&E evidence ranks just outside the window. evaluate_retrieval_status()
    then reports that evidence as "missing" even though it exists lower in
    ranked_pool, and the system abstains on a question it could answer.
    Backfill the highest-ranked candidate of each still-missing required
    statement type from the full reranked pool (not just the top_k slice)
    rather than expanding raw retrieval depth, which doesn't help when
    ranking itself is what's burying the evidence.
    """
    if not query_info or not getattr(query_info, "requires_calculation", False):
        return top_chunks

    required = list(dict.fromkeys(getattr(query_info, "target_statement_types", []) or []))
    if len(required) < 2:
        return top_chunks

    covered = {c.get("statement_type") for c in top_chunks}
    missing = [st for st in required if st not in covered]
    if not missing:
        return top_chunks

    result = list(top_chunks)
    for st in missing:
        candidate = next((c for c in ranked_pool if c.get("statement_type") == st), None)
        if candidate is None or any(c.get("chunk_idx") == candidate.get("chunk_idx") for c in result):
            continue
        if len(result) >= top_k:
            result[-1] = candidate
        else:
            result.append(candidate)
    return result


def expand_chunk_context(chunks: List[Dict], index: "FilingIndex", window: int = NEIGHBOR_WINDOW_SIZE) -> List[Dict]:
    """
    Expand top retrieved chunks with neighboring adjacent chunks (N-1, N, N+1)
    belonging to the same section/filing to prevent cutting off context.
    """
    if not NEIGHBOR_EXPANSION_ENABLED or not chunks or index is None or not index.chunks:
        return chunks

    expanded_results = []
    seen_indices = set()

    for c in chunks:
        idx = c.get("chunk_idx")
        if idx is None or idx in seen_indices:
            expanded_results.append(c)
            continue

        seen_indices.add(idx)

        # Collect neighboring chunks from same page/section
        prev_idx = max(0, idx - window)
        next_idx = min(len(index.chunks) - 1, idx + window)

        neighbor_texts = []
        for i in range(prev_idx, next_idx + 1):
            n_chunk = index.chunks[i]
            if n_chunk.get("section") == c.get("section") or n_chunk.get("page_num") == c.get("page_num"):
                neighbor_texts.append(n_chunk.get("text", ""))

        if neighbor_texts:
            merged_text = "\n\n".join(dict.fromkeys(neighbor_texts))
            c_expanded = dict(c)
            c_expanded["text"] = merged_text
            c_expanded["is_expanded"] = True
            expanded_results.append(c_expanded)
        else:
            expanded_results.append(c)

    return expanded_results


class FilingIndex:

    def __init__(self, doc_name: str, chunks: Optional[List[Dict]] = None, metadata: Optional[Dict] = None):
        self.doc_name = doc_name
        self.chunks: List[Dict] = chunks or []
        self.metadata: Dict = metadata or {}
        self.bm25: Optional[BM25Okapi] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index = None

    # ---------------- building ----------------

    def build_bm25(self):
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def build_bge_faiss(self):
        """Generate selected-model embeddings for chunks and build FAISS vector store."""
        if not self.chunks:
            return
        try:
            texts = [c["text"] for c in self.chunks]
            embed_svc = get_embedding_service()
            embeddings = embed_svc.embed_documents(texts)
            if embeddings is not None:
                store = FAISSVectorStore(
                    dim=embeddings.shape[1],
                    embedding_model=embed_svc.key,
                    model_name=embed_svc.model_name,
                    similarity_metric=embed_svc.similarity_metric,
                    filing_id=self.doc_name,
                )
                store.build_index(self.chunks, embeddings)
                self.vector_store = store
                self.set_vectors(embeddings)
            else:
                raise RuntimeError(f"No embeddings returned for EMBEDDING_MODEL={embed_svc.key}")
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"Warning: Failed to build selected embedding FAISS vector store: {exc}")

    def set_vectors(self, vectors: np.ndarray):
        """vectors: (n_chunks, dim) float32, normalized for cosine sim."""
        if vectors is None or len(vectors) == 0:
            self.vectors = None
            self.faiss_index = None
            return
        vecs = vectors.astype("float32").copy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self.vectors = vecs
        if faiss is not None:
            dim = vecs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(vecs)
            if self.vector_store is None:
                store = FAISSVectorStore(dim=dim)
                store.index = self.faiss_index
                store.chunk_ids = [c.get("chunk_index", i) for i, c in enumerate(self.chunks)]
                self.vector_store = store

    # ---------------- searching ----------------

    def search_bm25(self, query: str, top_k: int = BM25_TOP_K) -> List[int]:
        if self.bm25 is None or not self.chunks:
            return []
        tokens = tokenize_query(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked if scores[i] > 0]

    def bm25_score(self, query: str, chunk_idx: int) -> float:
        if self.bm25 is None:
            return 0.0
        tokens = tokenize_query(query)
        if not tokens:
            return 0.0
        scores = self.bm25.get_scores(tokens)
        if chunk_idx < 0 or chunk_idx >= len(scores):
            return 0.0
        return float(scores[chunk_idx])

    def search_metadata(self, query: str, top_k: int = BM25_TOP_K) -> List[Tuple[int, float]]:
        """Field-style recall for filing-purpose questions where SEC header terms are the evidence."""
        query_info = analyze_query(query)
        if query_info.query_type != "DOCUMENT_PURPOSE":
            return []

        hits = []
        target_years = set(query_info.target_years)
        for idx, chunk in enumerate(self.chunks):
            searchable = _chunk_search_text(chunk)
            matches = [marker for marker in _DOCUMENT_PURPOSE_MARKERS if marker in searchable]
            if not matches:
                continue
            score = float(len(matches) * 10)
            if "item information" in matches:
                score += 15.0
            if "conformed submission type" in matches or "form type" in matches:
                score += 8.0
            if "filed as of date" in matches:
                score += 8.0
            if target_years and any(year in searchable for year in target_years):
                score += 5.0
            hits.append((idx, score))

        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:top_k]

    def search_bge_faiss(self, query: str, top_k: int = SEMANTIC_TOP_K) -> List[Tuple[int, float]]:
        """Search FAISS vector store using BGE query embedding."""
        if self.vector_store is None:
            return []
        try:
            expanded_q = expand_query(_expand_financial_synonyms(query))
            embed_svc = get_embedding_service()
            qv = embed_svc.embed_query(expanded_q)
            if qv is None:
                return []
            return self.vector_store.search(qv, top_k=top_k)
        except Exception as exc:
            print(f"Warning: selected embedding FAISS query search failed: {exc}")
            return []

    def search_dense(self, query_vector: Optional[np.ndarray], top_k: int = SEMANTIC_TOP_K) -> List[int]:
        if query_vector is None or self.vectors is None or len(self.vectors) == 0:
            return []
        qv = query_vector.astype("float32").copy()
        norm = np.linalg.norm(qv)
        if norm == 0:
            return []
        qv = qv / norm

        if self.faiss_index is not None:
            qv2 = qv.reshape(1, -1)
            k = min(top_k, len(self.chunks))
            _, idxs = self.faiss_index.search(qv2, k)
            return [int(i) for i in idxs[0] if i >= 0]

        sims = self.vectors @ qv
        ranked = np.argsort(sims)[::-1][:top_k]
        return [int(i) for i in ranked]

    def _faiss_vector_count(self) -> int:
        if self.vector_store is not None and self.vector_store.index is not None:
            return int(self.vector_store.index.ntotal)
        if self.faiss_index is not None:
            return int(self.faiss_index.ntotal)
        return 0

    def vector_index_dir(self) -> Path:
        return INDEX_DIR / self.doc_name / get_embedding_model_name()

    def hybrid_search(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = RERANK_TOP_K,
        debug: bool = True,
    ) -> List[Dict]:
        query_info = analyze_query(query)

        bm25_top_indices = self.search_bm25(query, top_k=BM25_TOP_K)
        metadata_hits = self.search_metadata(query, top_k=BM25_TOP_K)
        metadata_top_indices = [chunk_id for chunk_id, _ in metadata_hits]
        metadata_score_map = {chunk_id: score for chunk_id, score in metadata_hits}

        faiss_hits = self.search_bge_faiss(query, top_k=SEMANTIC_TOP_K)
        semantic_top_indices = [chunk_id for chunk_id, _ in faiss_hits]
        semantic_score_map = {chunk_id: score for chunk_id, score in faiss_hits}

        if not semantic_top_indices and query_vector is not None and self.vectors is not None:
            semantic_top_indices = self.search_dense(query_vector, top_k=SEMANTIC_TOP_K)

        weighted_lists = []
        if bm25_top_indices:
            weighted_lists.append((bm25_top_indices, 1.0))
        if metadata_top_indices:
            weighted_lists.append((metadata_top_indices, 0.9))
        if semantic_top_indices:
            weighted_lists.append((semantic_top_indices, 1.0))

        if not weighted_lists:
            if debug:
                available_docs = list_indexed_docs()
                faiss_dir = self.vector_index_dir()
                print("\n==================================================")
                print("RETRIEVAL DOCUMENT SCOPE")
                print("==================================================")
                print(f"Query: {query}")
                print(f"Target document: {self.doc_name}")
                print(f"Available documents: {available_docs}")
                print(f"BM25 scope: {self.doc_name}")
                print(f"FAISS scope: {self.doc_name}")
                print(f"Embedding model: {get_embedding_model_name()}")
                print(f"FAISS index: {faiss_dir}")
                print(f"FAISS vectors: {self._faiss_vector_count()}")
                print(f"Chunk count: {len(self.chunks)}")
                print("No BM25 or FAISS candidates returned.")
                print("==================================================\n")
            return []

        fused = _rrf_fuse(weighted_lists, k=RRF_K)
        ranked_idxs = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:RRF_TOP_K]

        candidates = []
        for i in ranked_idxs:
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = dict(self.chunks[i])
            chunk["doc_name"] = self.doc_name
            chunk["company"] = self.metadata.get("company", self.doc_name.split("_")[0])
            chunk["filing_type"] = self.metadata.get("filing_type", "Filing")
            chunk["fiscal_year"] = self.metadata.get("fiscal_year", "")
            chunk["source_filename"] = self.metadata.get("source_filename", f"{self.doc_name}.htm")
            chunk["chunk_idx"] = i
            chunk["retrieval_score"] = fused[i]
            chunk["bm25_score"] = self.bm25_score(query, i)
            chunk["metadata_score"] = metadata_score_map.get(i, 0.0)
            chunk["semantic_score"] = semantic_score_map.get(i, 0.0)
            chunk["bm25_rank"] = (bm25_top_indices.index(i) + 1) if i in bm25_top_indices else None
            chunk["metadata_rank"] = (metadata_top_indices.index(i) + 1) if i in metadata_top_indices else None
            chunk["dense_rank"] = (semantic_top_indices.index(i) + 1) if i in semantic_top_indices else None
            chunk["semantic_rank"] = chunk["dense_rank"]
            candidates.append(chunk)

        deterministic_ranked = deterministic_rerank(query, candidates, query_info=query_info)
        reranked_top = cross_encoder_rerank(query, deterministic_ranked)[:top_k]
        reranked_top = _ensure_statement_coverage(reranked_top, deterministic_ranked, query_info, top_k)
        results = expand_chunk_context(reranked_top, self)

        if debug:
            table_cnt = sum(1 for c in self.chunks if c.get("chunk_type") == "table")
            text_cnt = len(self.chunks) - table_cnt
            available_docs = list_indexed_docs()
            faiss_dir = self.vector_index_dir()
            def _clean(t):
                return t.encode("ascii", errors="replace").decode("ascii") if t else ""

            print(f"\n==================================================")
            print(f"RETRIEVAL DOCUMENT SCOPE")
            print(f"==================================================")
            print(f"Query: {_clean(query)}")
            print(f"Target document: {self.doc_name}")
            print(f"Available documents: {available_docs}")
            print(f"BM25 scope: {self.doc_name}")
            print(f"FAISS scope: {self.doc_name}")
            print(f"Embedding model: {get_embedding_model_name()}")
            print(f"FAISS index: {faiss_dir}")
            print(f"FAISS vectors: {self._faiss_vector_count()}")
            print(f"Chunk count: {len(self.chunks)} (Text: {text_cnt}, Tables: {table_cnt})")
            print(f"Query Type: {query_info.query_type} | Requested Statement: {query_info.requested_statement} | Years: {query_info.target_years} | Concepts: {query_info.detected_concepts}")

            print("\n--------------------------------------------------")
            print("BM25 RESULTS")
            print("--------------------------------------------------")
            if bm25_top_indices:
                for r, idx_i in enumerate(bm25_top_indices[:5], 1):
                    if idx_i < len(self.chunks):
                        c = self.chunks[idx_i]
                        b_score = self.bm25_score(query, idx_i)
                        sec_str = _clean(c.get("section", "N/A"))
                        st_str = c.get("statement_type", "OTHER")
                        prev_str = _clean(c.get("text", "")[:160])
                        print(f"[{r}] doc={self.doc_name} chunk={idx_i} score={b_score:.4f}")
                        print(f"    BM25 Score: {b_score:.4f}")
                        print(f"    Page: {c.get('page_num')}")
                        print(f"    Section: {sec_str}")
                        print(f"    Statement Type: {st_str} | Chunk Type: {c.get('chunk_type')}")
                        print(f"    Preview: {prev_str}...\n")
            else:
                print("No BM25 results.\n")

            print("--------------------------------------------------")
            print("METADATA RESULTS")
            print("--------------------------------------------------")
            if metadata_top_indices:
                for r, idx_i in enumerate(metadata_top_indices[:5], 1):
                    if idx_i < len(self.chunks):
                        c = self.chunks[idx_i]
                        m_score = metadata_score_map.get(idx_i, 0.0)
                        sec_str = _clean(c.get("section", "N/A"))
                        prev_str = _clean(c.get("text", "")[:160])
                        print(f"[{r}] doc={self.doc_name} chunk={idx_i} score={m_score:.4f}")
                        print(f"    Metadata Score: {m_score:.4f}")
                        print(f"    Page: {c.get('page_num')}")
                        print(f"    Section: {sec_str}")
                        print(f"    Statement Type: {c.get('statement_type', 'OTHER')} | Chunk Type: {c.get('chunk_type')}")
                        print(f"    Preview: {prev_str}...\n")
            else:
                print("No metadata results.\n")

            print("--------------------------------------------------")
            print("FAISS RESULTS")
            print("--------------------------------------------------")
            if semantic_top_indices:
                for r, idx_i in enumerate(semantic_top_indices[:5], 1):
                    if idx_i < len(self.chunks):
                        c = self.chunks[idx_i]
                        s_score = semantic_score_map.get(idx_i, 0.0)
                        sec_str = _clean(c.get("section", "N/A"))
                        st_str = c.get("statement_type", "OTHER")
                        prev_str = _clean(c.get("text", "")[:160])
                        print(f"[{r}] doc={self.doc_name} chunk={idx_i} score={s_score:.4f}")
                        print(f"    Semantic Score: {s_score:.4f}")
                        print(f"    Page: {c.get('page_num')}")
                        print(f"    Section: {sec_str}")
                        print(f"    Statement Type: {st_str} | Chunk Type: {c.get('chunk_type')}")
                        print(f"    Preview: {prev_str}...\n")
            else:
                print("No FAISS semantic results.\n")

            print("--------------------------------------------------")
            print("RRF RESULTS")
            print("--------------------------------------------------")
            for r, c in enumerate(candidates[:5], 1):
                b_rank_str = f"{c['bm25_rank']}" if c['bm25_rank'] else "Unranked"
                m_rank_str = f"{c.get('metadata_rank')}" if c.get('metadata_rank') else "Unranked"
                s_rank_str = f"{c['semantic_rank']}" if c['semantic_rank'] else "Unranked"
                print(f"[{r}] doc={c['doc_name']} chunk={c['chunk_idx']} RRF={c['retrieval_score']:.5f}")
                print(f"    RRF Score: {c['retrieval_score']:.5f}")
                print(f"    BM25 Rank: {b_rank_str}")
                print(f"    BM25 Score: {c['bm25_score']:.4f}")
                print(f"    Metadata Rank: {m_rank_str}")
                print(f"    Metadata Score: {c.get('metadata_score', 0.0):.4f}")
                print(f"    Semantic Rank: {s_rank_str}\n")

            print("--------------------------------------------------")
            print("RERANKED TOP RESULTS")
            print("--------------------------------------------------")
            for r, c in enumerate(results[:top_k], 1):
                b_rank_str = f"{c['bm25_rank']}" if c['bm25_rank'] else "Unranked"
                s_rank_str = f"{c['semantic_rank']}" if c['semantic_rank'] else "Unranked"
                sec_str = _clean(c.get("section", "N/A"))
                st_str = c.get("statement_type", "OTHER")
                prev_str = _clean(c.get("text", "")[:160])
                print(f"[{r}] doc={c['doc_name']} chunk={c['chunk_idx']}")
                print(f"    Rerank Score: {c.get('rerank_score', 0):.2f}")
                if c.get("cross_encoder_score") is not None:
                    print(f"    Cross-Encoder Score: {c.get('cross_encoder_score', 0):.4f}")
                print(f"    RRF Score: {c['retrieval_score']:.5f}")
                print(f"    BM25 Rank: {b_rank_str}")
                print(f"    Semantic Rank: {s_rank_str}")
                print(f"    Query Term Boost: {c.get('query_term_boost', 0):.2f}")
                print(f"    Metric Match Boost: {c.get('metric_match_boost', 0):.2f}")
                print(f"    Grouping/Comparison Boost: {c.get('grouping_match_boost', 0):.2f}")
                print(f"    Table/Header Boost: {c.get('table_header_boost', 0):.2f}")
                print(f"    Filing/Purpose Boost: {c.get('filing_purpose_boost', 0):.2f}")
                print(f"    Retriever Agreement Boost: {c.get('agreement_boost', 0):.2f}")
                if c.get("matched_grouping_terms"):
                    print(f"    Matched Grouping Terms: {c.get('matched_grouping_terms')}")
                if c.get("filing_purpose_matches"):
                    print(f"    Filing/Purpose Matches: {c.get('filing_purpose_matches')}")
                print(f"    Page: {c.get('page_num')}")
                print(f"    Section: {sec_str}")
                print(f"    Statement Type: {st_str} | Chunk Type: {c.get('chunk_type')}")
                print(f"    Preview: {prev_str}...\n")

        return results

    def is_indexed(self) -> bool:
        return self.bm25 is not None and len(self.chunks) > 0

    def has_vector_index(self) -> bool:
        return self._faiss_vector_count() > 0

    # ---------------- persistence ----------------

    def save(self):
        out_dir = INDEX_DIR / self.doc_name
        vector_dir = self.vector_index_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        if self.vector_store is not None:
            self.vector_store.save(vector_dir)

        if self.vectors is not None:
            vector_dir.mkdir(parents=True, exist_ok=True)
            np.save(vector_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, doc_name: str) -> Optional["FilingIndex"]:
        in_dir = INDEX_DIR / doc_name
        chunks_path = in_dir / "chunks.json"
        bm25_path = in_dir / "bm25.pkl"
        if not chunks_path.exists() or not bm25_path.exists():
            return None

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        metadata = {}
        meta_path = in_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        idx = cls(doc_name, chunks, metadata=metadata)

        with open(bm25_path, "rb") as f:
            idx.bm25 = pickle.load(f)

        embedding_model = get_embedding_model_name()
        vector_dir = in_dir / embedding_model
        idx.vector_store = FAISSVectorStore.load(
            vector_dir,
            expected_embedding_model=embedding_model,
            expected_filing_id=doc_name,
        )

        if idx.vector_store is None and embedding_model == "normal":
            idx.vector_store = FAISSVectorStore.load(
                in_dir,
                expected_embedding_model=embedding_model,
                expected_filing_id=doc_name,
            )

        vectors_path = vector_dir / "vectors.npy"
        if not vectors_path.exists() and embedding_model == "normal":
            vectors_path = in_dir / "vectors.npy"
        if vectors_path.exists():
            try:
                vectors = np.load(vectors_path)
                idx.set_vectors(vectors)
            except Exception:
                pass

        return idx


def get_index(doc_name: str) -> Optional[FilingIndex]:
    cache_key = (get_embedding_model_name(), doc_name)
    if cache_key in _INDEX_CACHE:
        return _INDEX_CACHE[cache_key]
    idx = FilingIndex.load(doc_name)
    if idx is not None:
        _INDEX_CACHE[cache_key] = idx
    return idx


def register_index(doc_name: str, index: FilingIndex):
    _INDEX_CACHE[(get_embedding_model_name(), doc_name)] = index


def list_indexed_docs() -> List[str]:
    names = {doc_name for _, doc_name in _INDEX_CACHE.keys()}
    if INDEX_DIR.exists():
        for p in INDEX_DIR.iterdir():
            if p.is_dir() and (p / "chunks.json").exists():
                names.add(p.name)
    return sorted(names)


def _query_doc_match_score(query: str, index: FilingIndex) -> float:
    """Score whether the query explicitly points at a filing's metadata."""
    q = query.lower()
    q_tokens = set(_identity_tokens(query, min_len=2))
    q_years = _years_in_text(query)
    q_tokens.update(q_years)
    score = 0.0

    doc_name = index.doc_name.lower()
    doc_tokens = _identity_tokens(doc_name, min_len=2)
    metadata = index.metadata or {}

    company = _clean_company_identity(str(metadata.get("company") or "")).lower()
    company_tokens = _identity_tokens(company, min_len=2)
    fiscal_year = str(metadata.get("fiscal_year") or "").lower()
    filing_type = str(metadata.get("filing_type") or "").lower()
    source_filename = str(metadata.get("source_filename") or "").lower()

    if doc_name and doc_name in q:
        score += 10.0
    if source_filename and Path(source_filename).stem.lower() in q:
        score += 8.0
    if company and company != "unknown company" and " " in company and company in q:
        score += 6.0

    for token in company_tokens:
        if token in q_tokens:
            score += 3.0

    for token in doc_tokens:
        if token in q_tokens:
            score += 1.0

    years = _years_in_text(fiscal_year)
    if not years and fiscal_year:
        years.update(_years_in_text(doc_name))
    for year in years:
        if year in q_years:
            score += 1.5

    if filing_type and filing_type in q:
        score += 1.0
    compact_type = filing_type.replace("-", "")
    if compact_type and compact_type in q:
        score += 1.0

    return score


def cross_filing_hybrid_search(
    query: str,
    query_vector: Optional[np.ndarray] = None,
    doc_name: Optional[str] = None,
    top_k: int = RERANK_TOP_K,
) -> List[Dict]:
    """
    Perform hybrid retrieval across filings.
    If doc_name is specified and not 'all', search only that filing.
    Otherwise, search all indexed filings and rank top chunks globally.
    """
    available_docs = list_indexed_docs()
    requested_doc = doc_name or "all"
    search_scope = requested_doc if requested_doc.lower() not in ("all", "*", "") else "all indexed filings"
    print("\n==================================================")
    print("RETRIEVAL REQUEST SCOPE")
    print("==================================================")
    print(f"Query: {query}")
    print(f"Requested document: {requested_doc}")
    print(f"Search scope: {search_scope}")
    print(f"Available documents: {available_docs}")
    print("==================================================\n")

    if doc_name and doc_name.lower() not in ("all", "*", ""):
        idx = get_index(doc_name)
        if idx is None or not idx.is_indexed():
            return []
        return idx.hybrid_search(query, query_vector, top_k=top_k)

    all_docs = available_docs
    if not all_docs:
        return []

    doc_indexes = []
    for d in all_docs:
        idx = get_index(d)
        if idx is not None and idx.is_indexed():
            doc_indexes.append(idx)

    scoped_indexes = doc_indexes
    doc_match_scores = {idx.doc_name: _query_doc_match_score(query, idx) for idx in doc_indexes}
    positive_matches = [idx for idx in doc_indexes if doc_match_scores.get(idx.doc_name, 0.0) > 0.0]
    if positive_matches and max(doc_match_scores[idx.doc_name] for idx in positive_matches) >= DOC_ROUTING_THRESHOLD:
        max_score = max(doc_match_scores[idx.doc_name] for idx in positive_matches)
        scoped_indexes = [idx for idx in positive_matches if doc_match_scores[idx.doc_name] >= max_score]
        print("==================================================")
        print("AUTO FILING ROUTING")
        print("==================================================")
        print(f"Inferred documents: {[idx.doc_name for idx in scoped_indexes]}")
        print(f"Document match scores: {doc_match_scores}")
        print("==================================================\n")
    elif positive_matches:
        print("==================================================")
        print("CROSS-FILING METADATA BOOSTS")
        print("==================================================")
        print("No strong single-filing route inferred; searching all indexed filings.")
        print(f"Document match scores: {doc_match_scores}")
        print("==================================================\n")

    all_results = []
    for idx in scoped_indexes:
        res = idx.hybrid_search(query, query_vector, top_k=max(10, top_k * 2))
        if doc_match_scores.get(idx.doc_name, 0.0) > 0:
            for chunk in res:
                chunk["doc_match_score"] = doc_match_scores[idx.doc_name]
                chunk["rerank_score"] = chunk.get("rerank_score", 0.0) + doc_match_scores[idx.doc_name]
        all_results.extend(res)

    if not all_results:
        return []

    all_results.sort(key=lambda c: c.get("rerank_score", c.get("retrieval_score", 0.0)), reverse=True)
    return all_results[:top_k]


