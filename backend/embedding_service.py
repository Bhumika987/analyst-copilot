"""
HuggingFace BGE Embedding Service for SEC Filing Retrieval.

Model: BAAI/bge-small-en-v1.5
Generates L2-normalized 384-dimensional embeddings suitable for cosine similarity.
Loads the model lazily once and reuses it across all chunk and query embedding calls.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: str = BGE_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading HuggingFace embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"HuggingFace embedding model {self.model_name} loaded successfully.")
        except Exception as exc:
            logger.warning(f"Failed to load sentence-transformers model '{self.model_name}': {exc}. Semantic retrieval will fall back gracefully.")
            self._load_failed = True
            self._model = None
        return self._model

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate L2-normalized float32 embeddings for a list of document chunk texts.
        Returns shape (n_chunks, dim) or None if model unavailable.
        """
        if not texts:
            return None
        model = self._load_model()
        if model is None:
            return None
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.astype("float32")
        except Exception as exc:
            logger.warning(f"Error generating document embeddings: {exc}")
            return None

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        Generate L2-normalized float32 embedding for a user query.
        Returns shape (dim,) or None if model unavailable.
        """
        if not query or not query.strip():
            return None
        model = self._load_model()
        if model is None:
            return None
        try:
            embedding = model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.astype("float32")
        except Exception as exc:
            logger.warning(f"Error generating query embedding: {exc}")
            return None


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService.get_instance()
