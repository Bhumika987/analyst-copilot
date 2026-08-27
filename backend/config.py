"""
Central configuration for SEC Filing Analyst Copilot retrieval engine.
"""

import os
from pathlib import Path

# Embedding Model Configuration
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
NORMAL_EMBEDDING_MODEL_NAME = BGE_MODEL_NAME
FINLANG_MODEL_NAME = "FinLang/finance-embeddings-investopedia"
SUPPORTED_EMBEDDING_MODELS = {"normal", "finlang"}
DEFAULT_EMBEDDING_MODEL = "normal"


def _load_embedding_env() -> None:
    for path in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
    ):
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("EMBEDDING_MODEL=") and not os.environ.get("EMBEDDING_MODEL"):
                    value = line.split("=", 1)[1].strip(" '\"")
                    if value:
                        os.environ["EMBEDDING_MODEL"] = value
        except Exception:
            pass


_load_embedding_env()


def get_embedding_model_name() -> str:
    value = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip().lower()
    if value not in SUPPORTED_EMBEDDING_MODELS:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_MODELS))
        raise ValueError(f"Unsupported EMBEDDING_MODEL={value!r}. Supported values: {supported}")
    return value

# Retrieval Depths & Fusion Parameters
# Keep candidate recall high before the inexpensive deterministic reranker.
BM25_TOP_K = 50
SEMANTIC_TOP_K = 50
RRF_K = 60
RRF_TOP_K = 80
RERANK_TOP_K = 12

# Optional cross-encoder reranking. If the model cannot load locally, the
# pipeline falls back to deterministic structural reranking.
ENABLE_CROSS_ENCODER_RERANKER = True
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_CANDIDATE_K = 30
CROSS_ENCODER_LOCAL_ONLY = True

# Generic Relevance Boost Weights
CONCEPT_MATCH_BOOST = 35.0
YEAR_MATCH_BOOST = 15.0
DUAL_AGREEMENT_BOOST = 20.0
TABLE_CHUNK_BOOST = 25.0
STATEMENT_TYPE_BOOST = 25.0

# Evidence Sufficiency Thresholds
MIN_CONTENT_EVIDENCE_SCORE = 25.0  # Content-only score required for SUFFICIENT_EVIDENCE

# Contextual Chunk Expansion Settings
NEIGHBOR_EXPANSION_ENABLED = True
NEIGHBOR_WINDOW_SIZE = 1  # Expands top candidates with (N-1, N, N+1)
