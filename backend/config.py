"""
Central configuration for SEC Filing Analyst Copilot retrieval engine.
"""

import os
from pathlib import Path

# Embedding Model Configuration
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
NORMAL_EMBEDDING_MODEL_NAME = BGE_MODEL_NAME
FINLANG_MODEL_NAME = "FinLang/finance-embeddings-investopedia"
FINANCE_SMALL_MODEL_NAME = "baconnier/Finance_embedding_small_en-V1.5"
SUPPORTED_EMBEDDING_MODELS = {"normal", "finlang", "financesmall"}
DEFAULT_EMBEDDING_MODEL = "normal"


# Keys _load_embedding_env() pulls from .env into os.environ, same
# first-set-wins pattern as llm.py's _load_env(). HF_HUB_OFFLINE forces
# huggingface_hub to skip its online cache-freshness check and serve
# straight from the local cache -- worth setting once a gated model
# (finlang/financesmall) is already downloaded, since that online check
# needs auth and can 401 even though the cached files are fine (see
# embedding_service.py's strict_load). Leave it unset while downloading a
# model for the first time, or it'll fail outright instead of fetching it.
_ENV_KEYS = ("EMBEDDING_MODEL", "HF_HUB_OFFLINE")


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
                for key in _ENV_KEYS:
                    if line.startswith(f"{key}=") and not os.environ.get(key):
                        value = line.split("=", 1)[1].strip(" '\"")
                        if value:
                            os.environ[key] = value
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
# The cross-encoder is a generic passage-relevance model with no notion of
# "audited financial statement" vs. "narrative table that happens to share
# vocabulary" -- it must refine the deterministic domain-aware ranking, not
# replace it outright (that let a lay-phrased MD&A table like "Capital
# Spending" outrank the actual Consolidated Statement of Cash Flows for a
# query that explicitly asked for the cash flow statement). Its sigmoid
# score is added to deterministic_rerank's rerank_score, scaled by this
# weight, rather than used as the sole sort key.
CROSS_ENCODER_BLEND_WEIGHT = 30.0

# Generic Relevance Boost Weights
CONCEPT_MATCH_BOOST = 35.0
YEAR_MATCH_BOOST = 15.0
DUAL_AGREEMENT_BOOST = 20.0
TABLE_CHUNK_BOOST = 25.0
STATEMENT_TYPE_BOOST = 25.0

# Contextual Chunk Expansion Settings
NEIGHBOR_EXPANSION_ENABLED = True
NEIGHBOR_WINDOW_SIZE = 1  # Expands top candidates with (N-1, N, N+1)
