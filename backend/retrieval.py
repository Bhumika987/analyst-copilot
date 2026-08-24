"""
Per-filing hybrid retrieval: BM25 (primary) + FAISS dense (secondary, weak
TF-IDF-hash embeddings), fused with Reciprocal Rank Fusion.

BM25 is the signal that actually knows financial vocabulary ("capital
expenditure", "$1,577") token-for-token; the dense vectors here are cheap
hash embeddings meant only to catch paraphrases BM25 misses, so RRF fusion
(rather than a weighted score blend) keeps BM25's ranking dominant without
needing to calibrate two incompatible score scales against each other.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"

_TOKEN_RE = re.compile(r"[a-z0-9$.,%]+")

_INDEX_CACHE: Dict[str, "FilingIndex"] = {}

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


def expand_query(query: str) -> str:
    """Append likely filing-wording synonyms for any financial term the query mentions."""
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


def tokenize_query(query: str) -> List[str]:
    tokens = tokenize(expand_query(query))
    filtered = [t for t in tokens if t not in _STOPWORDS]
    return filtered or tokens


def _rrf_fuse(weighted_ranked_lists: List[tuple], k: int = 60) -> Dict[int, float]:
    """Weighted Reciprocal Rank Fusion over multiple (ranked_list, weight) pairs."""
    scores: Dict[int, float] = {}
    for ranked, weight in weighted_ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank + 1)
    return scores


class FilingIndex:
    def __init__(self, doc_name: str, chunks: Optional[List[Dict]] = None):
        self.doc_name = doc_name
        self.chunks: List[Dict] = chunks or []
        self.bm25: Optional[BM25Okapi] = None
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index = None

    # ---------------- building ----------------

    def build_bm25(self):
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def set_vectors(self, vectors: np.ndarray):
        """vectors: (n_chunks, dim) float32, will be L2-normalized for cosine sim."""
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

    # ---------------- searching ----------------

    def search_bm25(self, query: str, top_k: int = 20) -> List[int]:
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

    def search_dense(self, query_vector: Optional[np.ndarray], top_k: int = 20) -> List[int]:
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

    def hybrid_search(self, query: str, query_vector: Optional[np.ndarray], top_k: int = 8) -> List[Dict]:
        bm25_ranked = self.search_bm25(query, top_k=max(30, top_k * 3))

        if query_vector is not None and self.vectors is not None:
            dense_ranked = self.search_dense(query_vector, top_k=max(30, top_k * 3))
            # Dense vectors here are a crude hashed bag-of-words, not a real
            # semantic embedding - weighted well below BM25 so it can only
            # nudge ties, not bury BM25's top (exact-vocabulary) match under
            # noise. BM25 is the signal that actually understands "capital
            # expenditure" scores a chunk with those literal words.
            fused = _rrf_fuse([(bm25_ranked, 1.0), (dense_ranked, 0.25)], k=60)
        else:
            # Dense signal unavailable: fall back to BM25-only ranking gracefully.
            fused = _rrf_fuse([(bm25_ranked, 1.0)], k=60)

        if not fused:
            return []

        ranked_idxs = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

        top_bm25_score = max(self.bm25_score(query, i) for i in ranked_idxs) if ranked_idxs else 0.0
        max_possible = max(top_bm25_score, 1e-9)

        results = []
        for i in ranked_idxs:
            if i < 0 or i >= len(self.chunks):
                continue
            chunk = dict(self.chunks[i])
            chunk["chunk_idx"] = i
            chunk["retrieval_score"] = fused[i]
            chunk["bm25_score"] = self.bm25_score(query, i)
            chunk["bm25_score_norm"] = chunk["bm25_score"] / max_possible if max_possible else 0.0
            results.append(chunk)
        return results

    def is_indexed(self) -> bool:
        return self.bm25 is not None and len(self.chunks) > 0

    # ---------------- persistence ----------------

    def save(self):
        out_dir = INDEX_DIR / self.doc_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        if self.vectors is not None:
            np.save(out_dir / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, doc_name: str) -> Optional["FilingIndex"]:
        in_dir = INDEX_DIR / doc_name
        chunks_path = in_dir / "chunks.json"
        bm25_path = in_dir / "bm25.pkl"
        if not chunks_path.exists() or not bm25_path.exists():
            return None

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        idx = cls(doc_name, chunks)

        with open(bm25_path, "rb") as f:
            idx.bm25 = pickle.load(f)

        vectors_path = in_dir / "vectors.npy"
        if vectors_path.exists():
            vectors = np.load(vectors_path)
            idx.set_vectors(vectors)

        return idx


def get_index(doc_name: str) -> Optional[FilingIndex]:
    if doc_name in _INDEX_CACHE:
        return _INDEX_CACHE[doc_name]
    idx = FilingIndex.load(doc_name)
    if idx is not None:
        _INDEX_CACHE[doc_name] = idx
    return idx


def register_index(doc_name: str, index: FilingIndex):
    _INDEX_CACHE[doc_name] = index


def list_indexed_docs() -> List[str]:
    names = set(_INDEX_CACHE.keys())
    if INDEX_DIR.exists():
        for p in INDEX_DIR.iterdir():
            if p.is_dir() and (p / "chunks.json").exists():
                names.add(p.name)
    return sorted(names)
