"""
Central configuration for SEC Filing Analyst Copilot retrieval engine.
"""

# Embedding Model Configuration
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"

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
