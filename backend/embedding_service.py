"""
Configurable embedding providers for SEC filing retrieval.

The selected provider is controlled by EMBEDDING_MODEL:
  - normal: existing BAAI/bge-small-en-v1.5 behavior
  - finlang: FinLang/finance-embeddings-investopedia
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import (
    FINLANG_MODEL_NAME,
    NORMAL_EMBEDDING_MODEL_NAME,
    get_embedding_model_name,
)

logger = logging.getLogger(__name__)

BGE_MODEL_NAME = NORMAL_EMBEDDING_MODEL_NAME


class EmbeddingProvider:
    """Small interface shared by all embedding backends."""

    key = "base"
    model_name = ""
    similarity_metric = "cosine"

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        raise NotImplementedError

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    query_prefix = ""
    document_prefix = ""
    strict_load = False

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _load_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading HuggingFace embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("HuggingFace embedding model %s loaded successfully.", self.model_name)
        except Exception as exc:
            if self.strict_load:
                raise RuntimeError(
                    f"Failed to load configured embedding model '{self.model_name}' "
                    f"for EMBEDDING_MODEL={self.key}: {exc}"
                ) from exc
            logger.warning(
                "Failed to load sentence-transformers model '%s': %s. Semantic retrieval will fall back gracefully.",
                self.model_name,
                exc,
            )
            self._load_failed = True
            self._model = None
        return self._model

    def _format_documents(self, texts: List[str]) -> List[str]:
        if not self.document_prefix:
            return texts
        return [f"{self.document_prefix}{text}" for text in texts]

    def _format_query(self, text: str) -> str:
        if not self.query_prefix:
            return text
        return f"{self.query_prefix}{text}"

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts:
            return None
        model = self._load_model()
        if model is None:
            return None
        try:
            embeddings = model.encode(
                self._format_documents(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.astype("float32")
        except Exception as exc:
            logger.warning("Error generating document embeddings: %s", exc)
            return None

    def embed_query(self, text: str) -> Optional[np.ndarray]:
        if not text or not text.strip():
            return None
        model = self._load_model()
        if model is None:
            return None
        try:
            embedding = model.encode(
                self._format_query(text),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.astype("float32")
        except Exception as exc:
            logger.warning("Error generating query embedding: %s", exc)
            return None


class NormalEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    key = "normal"

    def __init__(self):
        super().__init__(NORMAL_EMBEDDING_MODEL_NAME)


class FinLangEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    key = "finlang"
    strict_load = True

    def __init__(self):
        super().__init__(FINLANG_MODEL_NAME)


_PROVIDER_CACHE: Dict[str, EmbeddingProvider] = {}


def get_embedding_provider(embedding_model: Optional[str] = None) -> EmbeddingProvider:
    selected = (embedding_model or get_embedding_model_name()).strip().lower()
    if selected not in _PROVIDER_CACHE:
        if selected == "normal":
            _PROVIDER_CACHE[selected] = NormalEmbeddingProvider()
        elif selected == "finlang":
            _PROVIDER_CACHE[selected] = FinLangEmbeddingProvider()
        else:
            raise ValueError(f"Unsupported EMBEDDING_MODEL={selected!r}")
    return _PROVIDER_CACHE[selected]


class EmbeddingService:
    """Backward-compatible facade for existing retrieval code."""

    @classmethod
    def get_instance(cls) -> EmbeddingProvider:
        return get_embedding_provider()


def get_embedding_service() -> EmbeddingProvider:
    return get_embedding_provider()
