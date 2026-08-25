"""
Lightweight evidence extraction before LLM generation.

This module does not try to replace the LLM. It extracts likely answer-bearing
lines, periods, numeric values, and table/text evidence labels so the final
prompt receives the same structure a financial analyst would inspect.
"""

import re
from typing import Dict, List

from query_analyzer import QueryAnalysis

NUMBER_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
YEAR_RE = re.compile(r"(?<!\d)(20\d\d)(?!\d)")


def _line_score(line: str, query_info: QueryAnalysis) -> int:
    line_lower = line.lower()
    score = 0
    if query_info.target_years and any(year in line_lower for year in query_info.target_years):
        score += 4
    if query_info.accounting_terms and any(term in line_lower for term in query_info.accounting_terms):
        score += 5
    if query_info.normalized_concepts and query_info.metric.lower() in line_lower:
        score += 3
    if NUMBER_RE.search(line):
        score += 2
    if any(word in line_lower for word in ("total", "net", "cash", "revenue", "acquisition", "acquired")):
        score += 1
    return score


def _candidate_lines(text: str) -> List[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line and line not in ("[TABLE]", "[/TABLE]"):
            lines.append(line)
    return lines


def extract_evidence_notes(question: str, query_info: QueryAnalysis, chunks: List[Dict], max_notes: int = 12) -> str:
    """
    Build structured deterministic notes from selected evidence chunks.
    Includes both text and table evidence, with extracted values and years.
    """
    if not chunks:
        return ""

    notes = []
    seen = set()
    for chunk in chunks:
        page = chunk.get("page_num", "?")
        evidence_type = (chunk.get("chunk_type") or "text").upper()
        section = chunk.get("section") or ""
        table = chunk.get("table_title") or ""
        units = chunk.get("units") or ""

        scored_lines = []
        for line in _candidate_lines(chunk.get("text", "")):
            score = _line_score(line, query_info)
            if score > 0:
                scored_lines.append((score, line))

        scored_lines.sort(key=lambda item: item[0], reverse=True)
        for _, line in scored_lines[:3]:
            key = (page, line)
            if key in seen:
                continue
            seen.add(key)
            years = ", ".join(dict.fromkeys(YEAR_RE.findall(line)))
            values = ", ".join(dict.fromkeys(NUMBER_RE.findall(line)))
            meta = [f"TYPE={evidence_type}", f"PAGE={page}"]
            if section:
                meta.append(f"SECTION={section}")
            if table:
                meta.append(f"TABLE={table}")
            if units:
                meta.append(f"UNIT={units}")
            if years:
                meta.append(f"YEARS={years}")
            if values:
                meta.append(f"VALUES={values}")
            notes.append(f"- {' | '.join(meta)}\n  EVIDENCE: {line[:500]}")
            if len(notes) >= max_notes:
                break
        if len(notes) >= max_notes:
            break

    if not notes:
        return ""

    mode = "calculate" if query_info.requires_calculation else "extract"
    return (
        "DETERMINISTIC EVIDENCE NOTES\n"
        f"QUESTION_MODE: {mode}\n"
        f"TARGET_YEARS: {', '.join(query_info.target_years) if query_info.target_years else 'not specified'}\n"
        f"METRIC_OR_CONCEPT: {', '.join(query_info.normalized_concepts) if query_info.normalized_concepts else 'not normalized'}\n"
        + "\n".join(notes)
    )
