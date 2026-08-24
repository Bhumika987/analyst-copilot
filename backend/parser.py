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
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _find_page_number(elements: List, hr_index: int) -> Optional[int]:
    """Look backwards from the <hr/> for a standalone digit run = page number."""
    start = max(0, hr_index - LOOKBACK_ELEMENTS)
    for el in reversed(elements[start:hr_index]):
        try:
            text = el.get_text() if hasattr(el, "get_text") else str(el)
        except Exception:
            continue
        text = text.strip()
        if not text:
            continue
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


def _windowize(text: str, page_num: Optional[int]) -> List[Dict]:
    """Split long text into overlapping word-count windows."""
    words = text.split()
    if len(words) <= WINDOW_WORDS:
        return [{"page_num": page_num, "text": text, "chunk_type": "text"}]

    chunks = []
    step = WINDOW_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        window_words = words[start:start + WINDOW_WORDS]
        if not window_words:
            continue
        chunks.append({
            "page_num": page_num,
            "text": " ".join(window_words),
            "chunk_type": "text",
        })
        if start + WINDOW_WORDS >= len(words):
            break
    return chunks


def parse_filing_to_window_chunks(filepath: str) -> List[Dict]:
    """
    Parse a SEC .htm filing into page-aware chunks.

    Returns a list of dicts: {page_num: int|None, text: str, chunk_type: str}
    Tables are kept atomic ("mixed" if surrounding text is folded in, else "table").
    Long text runs are split into overlapping windows.
    """
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.body or soup

    # Single linear pass in document order. Using positional indices (rather
    # than re-locating each <hr> by Tag equality) matters: BeautifulSoup's
    # Tag.__eq__ compares structure, not identity, so distinct <hr/> tags
    # with identical attrs/contents compare equal and `.index()` would
    # always resolve to the first one.
    all_elements = body.find_all(True, recursive=True)

    page_after_hr: Dict[int, Optional[int]] = {}
    for i, el in enumerate(all_elements):
        if el.name == "hr":
            page_after_hr[i] = _find_page_number(all_elements, i)

    # Table contents are extracted once, atomically, when the <table> tag
    # itself is visited; mark their descendants so the linear pass skips
    # re-emitting the same text as loose paragraphs.
    table_descendant_ids = set()
    for table in body.find_all("table"):
        for desc in table.find_all(True, recursive=True):
            table_descendant_ids.add(id(desc))

    chunks: List[Dict] = []
    current_page = 1
    text_buffer: List[str] = []

    def flush_text_buffer(page_num):
        if not text_buffer:
            return
        combined = normalize_text(" ".join(text_buffer))
        text_buffer.clear()
        if combined:
            chunks.extend(_windowize(combined, page_num))

    for i, el in enumerate(all_elements):
        if el.name == "hr":
            pnum = page_after_hr.get(i)
            if pnum is not None:
                flush_text_buffer(current_page)
                current_page = pnum + 1
            continue

        if id(el) in table_descendant_ids:
            continue

        if el.name == "table":
            flush_text_buffer(current_page)
            table_text = _table_to_pipe_text(el)
            if table_text.strip():
                chunks.append({
                    "page_num": current_page,
                    "text": f"[TABLE]\n{table_text}\n[/TABLE]",
                    "chunk_type": "table",
                })
            continue

        # Only leaf elements (no element children) contribute text directly,
        # so nested wrappers (div > p > font > text) aren't counted 3x.
        if el.find_all(True, recursive=False):
            continue

        text = el.get_text(" ", strip=True)
        if text:
            text_buffer.append(text)

    flush_text_buffer(current_page)

    chunks = [c for c in chunks if c["text"].strip()]

    if not any(c["page_num"] is not None for c in chunks):
        for c in chunks:
            c["page_num"] = c.get("page_num") or 1

    return chunks


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
    if result:
        print("\nSample chunk:")
        print(result[0])
