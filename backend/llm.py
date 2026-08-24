"""
Groq LLM integration + a lightweight local embedding fallback.

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
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

import httpx
import numpy as np

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile was decommissioned from Groq's catalog; gpt-oss-120b
# is the closest current equivalent (strong instruction-following, 131k
# context, cheap). Its responses carry a separate `reasoning`/"analysis"
# channel alongside `content` in both streaming and non-streaming modes -
# the parsing below only ever reads `content`, so the chain-of-thought never
# leaks into an answer or a citation.
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Free-tier Groq keys are capped at a modest tokens-per-minute budget shared
# across every model on the account. "low" cuts the model's internal
# reasoning-channel output roughly in half without materially hurting
# instruction-following on a task this constrained (quote-and-cite).
REASONING_EFFORT = "low"

# Below this retrieval score, the top chunk isn't strong enough evidence
# to even bother asking the LLM. Tune against practice-questions.jsonl.
CONFIDENCE_THRESHOLD = 0.01

# Hard cap per context passage sent to the LLM. Text chunks are already
# ~350 words by construction, but table chunks are kept atomic and
# unbounded (a wide financial table can run to thousands of characters) -
# without a cap, a single huge table could burn most of the per-minute
# token budget in one request.
MAX_PASSAGE_CHARS = 1400

EMBED_DIM = 256
_embed_cache: Dict[str, np.ndarray] = {}

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer ONLY using the provided context passages.
Every number or fact must come from the passages.
If the answer is not present in the passages, respond with exactly: NOT_FOUND
Do not infer, hallucinate, or use outside knowledge.
If the question asks for a calculation (ratio, margin, growth), show the formula and \
compute step by step using ONLY numbers found in the passages.
When a fiscal year is specified (e.g. FY2022), use only that year's column.
The same figure sometimes appears in both a primary financial statement (the
Consolidated Statement of Cash Flows, Balance Sheet, or Income Statement) and
an MD&A highlights/summary table. When a passage is clearly from a primary
financial statement, cite that one over an MD&A summary restating the same
number.
Output format when answer found:
ANSWER: [precise answer]
SOURCE: Page [N] - "[exact quote from the passage]\""""


@dataclass
class AnswerResult:
    found: bool
    answer: Optional[str] = None
    page_num: Optional[int] = None
    evidence_text: Optional[str] = None
    confidence: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "found": self.found,
            "answer": self.answer,
            "page_num": self.page_num,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "error": self.error,
        }


def get_embedding(text: str) -> np.ndarray:
    """
    Cheap hashed-TF-IDF-ish embedding: no Groq embedding endpoint exists,
    so this is only meant to nudge dense search toward paraphrase matches
    on top of BM25, not to be a strong semantic signal on its own.
    """
    key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]

    vec = np.zeros(EMBED_DIM, dtype="float32")
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    if words:
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            bin_idx = h % EMBED_DIM
            vec[bin_idx] += 1.0
        vec = vec / (np.linalg.norm(vec) + 1e-9)

    _embed_cache[key] = vec
    return vec


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = c.get("page_num", "?")
        text = c.get("text", "")
        if len(text) > MAX_PASSAGE_CHARS:
            text = text[:MAX_PASSAGE_CHARS] + " ...[truncated]"
        parts.append(f"[Passage {i} | Page {page}]\n{text}")
    return "\n\n".join(parts)


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
    """
    On a 429, prefer Groq's own reset hint over a blind guess - a fixed
    1/2/4s exponential backoff is far too short against an 8000-token/min
    budget that can take 10-60s to refill.
    """
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


_NOT_FOUND = {"found": False, "answer": None, "page_num": None, "evidence_text": None}


def _parse_llm_output(text: str) -> Dict:
    text = (text or "").strip()

    if "NOT_FOUND" in text.upper():
        # Covers both the instructed bare "NOT_FOUND" reply and the model
        # instead wrapping it as "ANSWER: NOT_FOUND" / with a fabricated
        # SOURCE line - either way it's declining, not answering, and must
        # never be treated as a confident (and scoreable-wrong) answer.
        return dict(_NOT_FOUND)

    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    source_match = re.search(
        r"SOURCE:\s*Page\s*(\d+)\s*-\s*[\"“](.+?)[\"”]", text, re.IGNORECASE | re.DOTALL
    )

    if not answer_match:
        # Model didn't follow the format and didn't say NOT_FOUND either;
        # treat as unusable rather than risk surfacing an unproven answer.
        return dict(_NOT_FOUND)

    answer = answer_match.group(1).strip()
    page_num = int(source_match.group(1)) if source_match else None
    evidence_text = source_match.group(2).strip() if source_match else None

    if page_num is None or not evidence_text:
        return dict(_NOT_FOUND)

    return {"found": True, "answer": answer, "page_num": page_num, "evidence_text": evidence_text}


def _confidence_from_chunks(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    top_score = chunks[0].get("retrieval_score", 0.0)
    return min(1.0, float(top_score) * 10)


async def answer_question(question: str, doc_name: str, chunks: List[Dict]) -> AnswerResult:
    if not chunks:
        return AnswerResult(found=False, confidence=0.0, error=None)

    top_score = chunks[0].get("retrieval_score", 0.0)
    if top_score < CONFIDENCE_THRESHOLD:
        return AnswerResult(found=False, confidence=_confidence_from_chunks(chunks))

    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"},
    ]

    try:
        data = await _call_groq(messages)
    except Exception as exc:
        return AnswerResult(found=False, confidence=0.0, error=f"llm_error: {exc}")

    try:
        raw_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return AnswerResult(found=False, confidence=0.0, error="malformed_llm_response")

    parsed = _parse_llm_output(raw_text)
    confidence = _confidence_from_chunks(chunks) if parsed["found"] else 0.0

    return AnswerResult(
        found=parsed["found"],
        answer=parsed["answer"],
        page_num=parsed["page_num"],
        evidence_text=parsed["evidence_text"],
        confidence=confidence,
        raw_response=raw_text,
    )


async def stream_answer(question: str, doc_name: str, chunks: List[Dict]) -> AsyncGenerator[Dict, None]:
    if not chunks:
        yield {"type": "result", "found": False, "answer": None, "page_num": None,
               "evidence_text": None, "confidence": 0.0}
        return

    top_score = chunks[0].get("retrieval_score", 0.0)
    if top_score < CONFIDENCE_THRESHOLD:
        yield {"type": "result", "found": False, "answer": None, "page_num": None,
               "evidence_text": None, "confidence": _confidence_from_chunks(chunks)}
        return

    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"},
    ]

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
            break
        except Exception as exc:
            full_text = ""
            if attempt < max_retries - 1:
                await asyncio.sleep(_retry_after_seconds(exc, attempt))
                continue
            yield {"type": "result", "found": False, "answer": None, "page_num": None,
                   "evidence_text": None, "confidence": 0.0, "error": f"llm_error: {exc}"}
            return

    parsed = _parse_llm_output(full_text)
    confidence = _confidence_from_chunks(chunks) if parsed["found"] else 0.0

    yield {
        "type": "result",
        "found": parsed["found"],
        "answer": parsed["answer"],
        "page_num": parsed["page_num"],
        "evidence_text": parsed["evidence_text"],
        "confidence": confidence,
    }
