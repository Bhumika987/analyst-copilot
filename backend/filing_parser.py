"""
Parses SEC EDGAR .htm filings into page-aware, retrieval-ready chunks.

SEC .htm filings (10-K / 10-Q / 8-K) share a consistent visual structure:
  - A page break is rendered as an <hr/> tag (~130 per 10-K).
  - The printed page number sits as a lone number (1-999) in a centered
    <p>/<font>/<span> a few elements before that <hr/>.
  - Financial statements are real <table> tags; these must stay atomic
    (never split mid-table) so a ratio/margin question always sees whole rows.
  - Inline XBRL tags (ix:nonFraction, ix:nonNumeric, ...) wrap numbers/text
    inside normal elements; BeautifulSoup's default parser already exposes
    their text content via .get_text(), so no special-casing is needed
    beyond leaving them alone.
"""

import re
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

WINDOW_WORDS = 350
OVERLAP_WORDS = 80
LOOKBACK_ELEMENTS = 20
PAGE_NUM_RE = re.compile(r"^\s*(\d{1,3})\s*$")

_UNICODE_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "﻿": "",
}


def normalize_text(text: str) -> str:
    """Normalize unicode punctuation/whitespace to plain ASCII-friendly text."""
    if not text:
        return ""
    for src, dst in _UNICODE_MAP.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\bShee\s+t\b", "Sheet", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFlow\s+s\b", "Flows", text, flags=re.IGNORECASE)
    text = re.sub(r"\bStatement\s+s\b", "Statements", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOperation\s+s\b", "Operations", text, flags=re.IGNORECASE)
    text = re.sub(r"\bActivit\s+y\b", "Activity", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _extract_sec_header_value(text: str, label: str) -> Optional[str]:
    pattern = rf"{re.escape(label)}:\s*(.+?)(?=\s+[A-Z][A-Z /-]+:|\s+</SEC-HEADER>|\s+<DOCUMENT>|\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


F_PAGE_NUM_RE = re.compile(r"^\s*F-\s?(\d{1,3})\s*$", re.IGNORECASE)


def _find_page_number(elements: List, hr_index: int) -> Optional[int]:
    """Look backwards from the <hr/> for a standalone digit run = page number.

    Many 10-Ks restart pagination inside the financial statements section
    with "F-1", "F-2", ... labels instead of plain digits. Those don't match
    PAGE_NUM_RE and are deliberately NOT treated as a plain-digit page here
    (returning a bare int for "F-3" would silently mean page 3, which is
    wrong). Returning None for them is intentional: the caller advances the
    running page count by one physical page instead, rather than freezing on
    the last plain-digit page it saw -- which previously collapsed the
    entire financial statements section (balance sheet, income statement,
    cash flow statement, every footnote) onto a single wrong page number.
    """
    start = max(0, hr_index - LOOKBACK_ELEMENTS)
    for el in reversed(elements[start:hr_index]):
        try:
            text = el.get_text() if hasattr(el, "get_text") else str(el)
        except Exception:
            continue
        text = normalize_text(text).strip()
        if not text:
            continue
        if F_PAGE_NUM_RE.match(text):
            return None
        m = PAGE_NUM_RE.match(text)
        if m:
            num = int(m.group(1))
            if 0 < num < 1000:
                return num
        # Stop scanning further back once we hit substantial prose,
        # the page-number label sits close to the <hr/>.
        if len(text) > 20:
            break
    return None


def _table_to_pipe_text(table) -> str:
    """Render an HTML <table> as pipe-separated rows, one row per line."""
    rows_out = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        cell_texts = [normalize_text(c.get_text(" ", strip=True)) for c in cells]
        cell_texts = [c for c in cell_texts if c != ""]
        if cell_texts:
            rows_out.append(" | ".join(cell_texts))
    return "\n".join(rows_out)


ITEM_RE = re.compile(
    r"^\s*(?:PART\s+[IVX]+\s*[-–—]?\s*)?ITEM\s+([0-9]{1,2}[A-Z]?)\s*[\.\:\-\–\—]?\s*(.*)$",
    re.IGNORECASE
)

SUBSECTION_KEYWORDS = (
    "note ", "results of operations", "performance by", "executive summary",
    "overview", "liquidity", "capital resources", "risk factors",
    "segment information", "critical accounting", "consolidated statements",
    "balance sheets", "statements of income", "statements of cash flows"
)


JUNK_HEADING_RE = re.compile(r"^[\W_]+$")
PERIOD_TITLE_RE = re.compile(
    r"\b(at|as of|as at|for the|years? ended|quarters? ended|three months ended|"
    r"six months ended|nine months ended|twelve months ended)\b",
    re.IGNORECASE,
)
STATEMENT_HEADING_RE = re.compile(
    r"\b("
    r"consolidated\s+(?:balance sheets?|statements? of (?:operations|income|earnings|"
    r"comprehensive income|cash flows?|financial position|stockholders'? equity|"
    r"shareholders'? equity|changes in equity))|"
    r"balance sheets?|statements? of (?:operations|income|earnings|cash flows?|"
    r"financial position|stockholders'? equity|shareholders'? equity|changes in equity)|"
    r"income statements?|cash flow statements?"
    r")\b",
    re.IGNORECASE,
)


def _heading_text(el) -> str:
    try:
        text = el.get_text(" ", strip=True) if hasattr(el, "get_text") else str(el).strip()
    except Exception:
        return ""
    return normalize_text(text)


def _is_junk_heading(text: str) -> bool:
    cleaned = normalize_text(text)
    lower = cleaned.lower().strip()
    if not lower:
        return True
    if lower in {"table of contents", "contents", "index", "[table]"}:
        return True
    if PAGE_NUM_RE.match(lower) or re.fullmatch(r"[\d.,$()% -]+", lower):
        return True
    if JUNK_HEADING_RE.match(lower):
        return True
    if len(lower) > 180:
        return True
    return False


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = re.sub(r"\s+", " ", value.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _extract_table_context(elements: List, table_idx: int) -> Dict[str, Optional[str]]:
    """Collect local heading stack without losing the primary statement title."""
    start = max(0, table_idx - 24)
    candidates = []
    for el in elements[start:table_idx]:
        if getattr(el, "name", None) in {"script", "style", "table", "tr", "td", "th"}:
            continue
        text = _heading_text(el)
        if _is_junk_heading(text):
            continue
        candidates.append(text)

    candidates = _dedupe_preserve_order(candidates)
    statement_title = None
    table_title = None

    for text in reversed(candidates):
        if STATEMENT_HEADING_RE.search(text):
            statement_title = text
            break

    for text in reversed(candidates):
        if text == statement_title:
            continue
        if PERIOD_TITLE_RE.search(text):
            table_title = text
            break

    if table_title is None:
        for text in reversed(candidates):
            if text != statement_title:
                table_title = text
                break

    table_context = " | ".join(candidates[-4:]) if candidates else None
    return {
        "statement_title": statement_title,
        "table_title": table_title,
        "table_context": table_context,
    }


def _extract_table_title(elements: List, table_idx: int) -> Optional[str]:
    """Backward-compatible wrapper for callers that only need table_title."""
    return _extract_table_context(elements, table_idx).get("table_title")


def _statement_structure_scores(
    section: str = "",
    subsection: str = "",
    statement_title: Optional[str] = "",
    table_title: Optional[str] = "",
    table_context: Optional[str] = "",
    text: str = "",
) -> Dict[str, int]:
    """Weighted, deterministic evidence for primary financial statement type."""
    header_text = f"{section} {subsection} {statement_title or ''} {table_title or ''} {table_context or ''}".lower()
    body_text = (text or "").lower()

    scores = {
        "BALANCE_SHEET": 0,
        "INCOME_STATEMENT": 0,
        "CASH_FLOW_STATEMENT": 0,
        "EQUITY_STATEMENT": 0,
    }

    header_indicators = {
        "BALANCE_SHEET": ("balance sheet", "financial position"),
        "INCOME_STATEMENT": ("statement of income", "statements of income", "statement of operations", "statements of operations", "statement of earnings"),
        "CASH_FLOW_STATEMENT": ("statement of cash flow", "statements of cash flow", "cash flows"),
        "EQUITY_STATEMENT": ("statement of equity", "statements of equity", "changes in equity", "stockholders' equity", "shareholders' equity"),
    }
    for stmt, indicators in header_indicators.items():
        if any(ind in header_text for ind in indicators):
            scores[stmt] += 8

    body_indicators = {
        "BALANCE_SHEET": (
            "assets", "current assets", "total assets", "liabilities",
            "current liabilities", "total liabilities", "total equity",
            "property, plant and equipment",
        ),
        "INCOME_STATEMENT": (
            "revenue", "net sales", "sales", "cost of sales", "gross profit",
            "operating income", "income before", "net income", "net earnings",
            "earnings per share", "eps",
        ),
        "CASH_FLOW_STATEMENT": (
            "operating activities", "investing activities", "financing activities",
            "net cash provided", "net cash used", "cash flows from operating",
            "cash flows from investing", "cash flows from financing",
        ),
        "EQUITY_STATEMENT": (
            "retained earnings", "treasury stock", "common stock",
            "additional paid-in capital", "accumulated other comprehensive",
            "stockholders' equity", "shareholders' equity",
        ),
    }
    for stmt, indicators in body_indicators.items():
        for indicator in indicators:
            if indicator in body_text:
                scores[stmt] += 2 if len(indicator.split()) > 1 else 1

    if "total assets" in body_text and ("total liabilities" in body_text or "current liabilities" in body_text):
        scores["BALANCE_SHEET"] += 8
    if "operating activities" in body_text and "investing activities" in body_text:
        scores["CASH_FLOW_STATEMENT"] += 8
    if ("net income" in body_text or "net earnings" in body_text) and ("earnings per share" in body_text or "cost of sales" in body_text):
        scores["INCOME_STATEMENT"] += 6
    if ("retained earnings" in body_text or "treasury stock" in body_text) and ("common stock" in body_text or "additional paid-in capital" in body_text):
        scores["EQUITY_STATEMENT"] += 6

    note_context = subsection.lower().startswith("note") or "notes to consolidated" in header_text
    if note_context and not (statement_title and STATEMENT_HEADING_RE.search(statement_title)):
        for stmt in scores:
            scores[stmt] = max(0, scores[stmt] - 4)

    return scores


def _detect_statement_type(
    section: str = "",
    subsection: str = "",
    table_title: Optional[str] = "",
    text: str = "",
    statement_title: Optional[str] = "",
    table_context: Optional[str] = "",
    is_table: bool = False,
) -> str:
    """Classify generic SEC financial document section type."""
    text_window = text[:1200] if is_table else text[:250]
    combined = f"{section} {subsection} {statement_title or ''} {table_title or ''} {table_context or ''} {text_window}".lower()
    scores = _statement_structure_scores(
        section=section,
        subsection=subsection,
        statement_title=statement_title,
        table_title=table_title,
        table_context=table_context,
        text=text,
    )
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    if (is_table or statement_title or table_context) and best_score >= 8:
        return best_type

    if any(k in combined for k in ("cash flows from investing", "cash flows from operating", "statements of cash flows", "statement of cash flows", "cash flows activities")):
        return "CASH_FLOW_STATEMENT"
    if any(k in combined for k in ("statements of income", "statement of income", "statements of operations", "statement of operations", "statements of earnings", "results of operations")):
        return "INCOME_STATEMENT"
    if any(k in combined for k in ("balance sheets", "balance sheet", "financial position")):
        return "BALANCE_SHEET"
    if any(k in combined for k in ("stockholders' equity", "shareholders' equity", "statements of equity")):
        return "EQUITY_STATEMENT"
    if any(k in combined for k in ("item 1a", "risk factors")):
        return "RISK_FACTORS"
    if any(k in combined for k in ("item 1 - financial statements", "item 1. financial statements")):
        return "OTHER"
    if any(k in combined for k in ("item 1", "business description", "business overview")):
        return "BUSINESS"
    if any(k in combined for k in ("item 7", "management's discussion")):
        return "MD_AND_A"
    if any(k in combined for k in ("note ", "notes to consolidated")):
        return "FOOTNOTES"

    return "OTHER"


def _extract_units(text: str = "", table_title: Optional[str] = "") -> Optional[str]:
    """Extract common financial table units from title/body text."""
    combined = f"{table_title or ''}\n{text or ''}".lower()
    if "in millions" in combined or "millions of" in combined or "$ in millions" in combined:
        return "millions"
    if "in thousands" in combined or "thousands of" in combined or "$ in thousands" in combined:
        return "thousands"
    if "in billions" in combined or "billions of" in combined or "$ in billions" in combined:
        return "billions"
    if "percent" in combined or "%" in combined:
        return "percent"
    return None


def _metadata_diagnostic(statement_type: str, structural_scores: Dict[str, int]) -> Optional[Dict]:
    """Flag table metadata that looks structurally stronger than its label."""
    predicted_type, predicted_score = max(structural_scores.items(), key=lambda item: item[1])
    current_score = structural_scores.get(statement_type, 0)
    if predicted_score < 10:
        return None
    if statement_type in {"OTHER", "BUSINESS", "FOOTNOTES"} or predicted_score >= current_score + 8:
        return {
            "predicted_statement_type": predicted_type,
            "predicted_score": predicted_score,
            "assigned_statement_type": statement_type,
            "assigned_score": current_score,
            "structural_scores": structural_scores,
        }
    return None


def _windowize(text: str, page_num: Optional[int], section: str = "General", subsection: str = "", chunk_index_counter: Optional[List[int]] = None) -> List[Dict]:
    """Split long text into overlapping word-count windows while preserving section metadata."""
    if chunk_index_counter is None:
        chunk_index_counter = [0]

    st_type = _detect_statement_type(section=section, subsection=subsection, text=text)

    words = text.split()
    if len(words) <= WINDOW_WORDS:
        idx = chunk_index_counter[0]
        chunk_index_counter[0] += 1
        return [{
            "page_num": page_num,
            "section": section or "General",
            "subsection": subsection or "",
            "table_title": None,
            "statement_type": st_type,
            "chunk_type": "text",
            "units": _extract_units(text=text),
            "chunk_index": idx,
            "text": text,
        }]

    chunks = []
    step = WINDOW_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        window_words = words[start:start + WINDOW_WORDS]
        if not window_words:
            continue
        idx = chunk_index_counter[0]
        chunk_index_counter[0] += 1
        chunks.append({
            "page_num": page_num,
            "section": section or "General",
            "subsection": subsection or "",
            "table_title": None,
            "statement_type": st_type,
            "chunk_type": "text",
            "units": _extract_units(text=" ".join(window_words)),
            "chunk_index": idx,
            "text": " ".join(window_words),
        })
        if start + WINDOW_WORDS >= len(words):
            break
    return chunks


def parse_filing_to_window_chunks(filepath: str) -> List[Dict]:
    """
    Parse a SEC .htm filing into structure-aware, retrieval-ready chunks.

    Returns a list of dicts:
      {page_num, section, subsection, table_title, statement_type, chunk_type, chunk_index, text}
    """
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.body or soup

    all_elements = body.find_all(True, recursive=True)

    page_after_hr: Dict[int, Optional[int]] = {}
    for i, el in enumerate(all_elements):
        if el.name == "hr":
            page_after_hr[i] = _find_page_number(all_elements, i)

    BLOCK_TAGS = {"p", "div", "table", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "section", "article", "header", "footer", "tr"}

    processed_ids = set()

    chunks: List[Dict] = []
    current_page = 1
    current_section = "General"
    current_subsection = ""
    text_buffer: List[str] = []
    chunk_index_counter = [0]

    def flush_text_buffer(page_num, sec, sub):
        if not text_buffer:
            return
        combined = normalize_text(" ".join(text_buffer))
        text_buffer.clear()
        if combined:
            chunks.extend(_windowize(combined, page_num, section=sec, subsection=sub, chunk_index_counter=chunk_index_counter))

    for i, el in enumerate(all_elements):
        if id(el) in processed_ids:
            continue

        if el.name == "hr":
            pnum = page_after_hr.get(i)
            flush_text_buffer(current_page, current_section, current_subsection)
            # A real page-number label always advances forward and rarely by
            # more than a page or two -- a match that jumps backwards or
            # wildly forward (e.g. 20+ pages) is almost always a stray digit
            # in nearby table/paragraph content, not an actual page label
            # (this is how page_num previously jumped to nonsense values like
            # 619/759 deep in a filing's notes). Reject those as if no label
            # were found at all.
            if pnum is not None and -1 <= (pnum - current_page) <= 20:
                # The printed page-number label found just before this <hr/>
                # belongs to the page that STARTS right after the break, not
                # the one that just ended (SEC .htm filings render each
                # page's number as a running head directly above the rule
                # that opens it, followed by nav chrome like "Table of
                # Contents" and then the page's real content). Verified by
                # cross-referencing gold-labeled practice questions against
                # the raw HTML: content immediately after the "59" label was
                # gold-annotated as page 59, not 60. Do not add +1 here.
                current_page = pnum
            else:
                # No plausible label for this break (e.g. an "F-N" financial
                # statements label, or nothing at all) -- the <hr/> still
                # marks one physical page turn, so advance by one instead of
                # freezing and silently collapsing every subsequent page onto
                # the last successfully-read page number.
                current_page = current_page + 1
            processed_ids.add(id(el))
            continue

        if el.name == "table":
            flush_text_buffer(current_page, current_section, current_subsection)
            table_text = _table_to_pipe_text(el)
            if table_text.strip():
                table_ctx = _extract_table_context(all_elements, i)
                statement_title = table_ctx.get("statement_title")
                tbl_title = table_ctx.get("table_title")
                table_context = table_ctx.get("table_context")
                st_type = _detect_statement_type(
                    section=current_section,
                    subsection=current_subsection,
                    table_title=tbl_title,
                    text=table_text,
                    statement_title=statement_title,
                    table_context=table_context,
                    is_table=True,
                )
                structural_scores = _statement_structure_scores(
                    section=current_section,
                    subsection=current_subsection,
                    statement_title=statement_title,
                    table_title=tbl_title,
                    table_context=table_context,
                    text=table_text,
                )
                meta_hdrs = []
                if current_section and current_section != "General":
                    meta_hdrs.append(f"Section: {current_section}")
                if current_subsection:
                    meta_hdrs.append(f"Subsection: {current_subsection}")
                if statement_title:
                    meta_hdrs.append(f"Statement: {statement_title}")
                if tbl_title:
                    meta_hdrs.append(f"Table: {tbl_title}")
                if table_context:
                    meta_hdrs.append(f"Context: {table_context}")
                meta_hdrs.append(f"Page: {current_page}")
                units = _extract_units(text=table_text, table_title=tbl_title)
                if units:
                    meta_hdrs.append(f"Unit: {units}")

                hdr_line = " | ".join(meta_hdrs)
                full_tbl_str = f"[{hdr_line}]\n[TABLE]\n{table_text}\n[/TABLE]"

                idx = chunk_index_counter[0]
                chunk_index_counter[0] += 1
                chunks.append({
                    "page_num": current_page,
                    "section": current_section,
                    "subsection": current_subsection,
                    "statement_title": statement_title,
                    "table_title": tbl_title,
                    "table_context": table_context,
                    "statement_type": st_type,
                    "statement_type_scores": structural_scores,
                    "metadata_diagnostic": _metadata_diagnostic(st_type, structural_scores),
                    "chunk_type": "table",
                    "units": units,
                    "chunk_index": idx,
                    "text": full_tbl_str,
                })

            processed_ids.add(id(el))
            for desc in el.find_all(True, recursive=True):
                processed_ids.add(id(desc))
            continue


        has_child_block = any(child.name in BLOCK_TAGS for child in el.find_all(True, recursive=True))

        if not has_child_block:
            text = el.get_text(" ", strip=True)
            text_norm = normalize_text(text)

            if text_norm:
                # Check for SEC Item header match
                item_match = ITEM_RE.match(text_norm)
                if item_match:
                    flush_text_buffer(current_page, current_section, current_subsection)
                    item_num = item_match.group(1).upper()
                    item_title = item_match.group(2).strip()
                    current_section = f"Item {item_num}" + (f" - {item_title}" if item_title else "")
                    current_subsection = ""

                # Check for subsection header match
                elif len(text_norm) < 90 and any(kw in text_norm.lower() for kw in SUBSECTION_KEYWORDS):
                    flush_text_buffer(current_page, current_section, current_subsection)
                    current_subsection = text_norm

                text_buffer.append(text_norm)

            processed_ids.add(id(el))
            for desc in el.find_all(True, recursive=True):
                processed_ids.add(id(desc))

    flush_text_buffer(current_page, current_section, current_subsection)

    chunks = [c for c in chunks if c["text"].strip()]

    if not any(c["page_num"] is not None for c in chunks):
        for c in chunks:
            c["page_num"] = c.get("page_num") or 1

    return chunks


def extract_filing_metadata(filepath: str, doc_name: str, chunks: List[Dict]) -> Dict:
    """
    Extract filing metadata: company, filing_type, fiscal_year, filing_date, accession_number, source_filename.
    Combines filename pattern matching with header text parsing.
    """
    path = Path(filepath)
    source_filename = path.name

    stem = doc_name or path.stem
    stem_clean = stem.replace("-", "_").replace(" ", "_")
    parts = [p for p in stem_clean.split("_") if p]

    company = parts[0] if parts else "Unknown Company"
    filing_type = "10-K"
    fiscal_year = "Unknown"
    filing_date = None
    accession_number = None

    for part in parts:
        p_upper = part.upper()
        if p_upper in ("10K", "10-K"):
            filing_type = "10-K"
        elif p_upper in ("10Q", "10-Q"):
            filing_type = "10-Q"
        elif p_upper in ("8K", "8-K"):
            filing_type = "8-K"

    yr_match = re.search(r"\b(20\d\d(?:Q[1-4])?)\b", stem, re.IGNORECASE)
    if yr_match:
        fiscal_year = yr_match.group(1).upper()

    sample_text = "\n".join(c.get("text", "") for c in chunks[:10])

    comp_value = _extract_sec_header_value(sample_text, "COMPANY CONFORMED NAME")
    if not comp_value:
        comp_value = _extract_sec_header_value(sample_text, "COMPANY NAME")
    if comp_value:
        company = comp_value

    form_match = re.search(r"FORM\s+(10-K|10-Q|8-K)", sample_text, re.IGNORECASE)
    if form_match:
        filing_type = form_match.group(1).upper()

    period_match = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8}|\d{4}-\d{2}-\d{2})", sample_text, re.IGNORECASE)
    if period_match:
        raw_p = period_match.group(1)
        if len(raw_p) == 8:
            fiscal_year = f"{raw_p[:4]}-{raw_p[4:6]}-{raw_p[6:]}"
        else:
            fiscal_year = raw_p

    date_match = re.search(r"FILED AS OF DATE:\s*(\d{8}|\d{4}-\d{2}-\d{2})", sample_text, re.IGNORECASE)
    if date_match:
        raw_d = date_match.group(1)
        if len(raw_d) == 8:
            filing_date = f"{raw_d[:4]}-{raw_d[4:6]}-{raw_d[6:]}"
        else:
            filing_date = raw_d

    acc_match = re.search(r"ACCESSION NUMBER:\s*([\d-]+)", sample_text, re.IGNORECASE)
    if acc_match:
        accession_number = acc_match.group(1).strip()

    return {
        "doc_name": doc_name,
        "company": company,
        "filing_type": filing_type,
        "fiscal_year": fiscal_year,
        "filing_date": filing_date,
        "accession_number": accession_number,
        "source_filename": source_filename,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <filepath>")
        sys.exit(1)

    fp = sys.argv[1]
    result = parse_filing_to_window_chunks(fp)
    pages = [c["page_num"] for c in result if c["page_num"] is not None]
    print(f"Chunks: {len(result)}")
    if pages:
        print(f"Page range: {min(pages)} - {max(pages)}")
    table_count = sum(1 for c in result if c["chunk_type"] == "table")
    print(f"Table chunks: {table_count}")
    print(f"Text chunks: {len(result) - table_count}")
    meta = extract_filing_metadata(fp, Path(fp).stem, result)
    print(f"Extracted Metadata: {meta}")
    if result:
        print("\nSample chunk:")
        print(result[0])


