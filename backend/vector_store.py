"""
FAISS Vector Store for SEC Filing Semantic Retrieval.

Wraps FAISS IndexFlatIP to perform inner-product search over L2-normalized embeddings.
Maintains a 1:1 mapping between vector position index and original chunk ID / chunk_index.
Supports persistent disk saving and loading alongside chunk JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(
        self,
        dim: int = 384,
        embedding_model: Optional[str] = None,
        model_name: Optional[str] = None,
        similarity_metric: str = "cosine",
        filing_id: Optional[str] = None,
    ):
        self.dim = dim
        self.index = None
        self.chunk_ids: List[int] = []
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.similarity_metric = similarity_metric
        self.filing_id = filing_id

    def build_index(self, chunks: List[Dict], embeddings: Optional[np.ndarray]):
        """
        Build FAISS IndexFlatIP with normalized chunk embeddings and store chunk_id mapping.
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings provided to build FAISS index.")
            self.index = None
            self.chunk_ids = []
            return

        if faiss is None:
            logger.warning("faiss package is not installed; semantic vector index disabled.")
            self.index = None
            self.chunk_ids = []
            return

        vecs = embeddings.astype("float32").copy()
        # L2-normalize vectors to ensure inner product equals cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        self.chunk_ids = [c.get("chunk_index", i) for i, c in enumerate(chunks)]
        logger.info(f"Built FAISS vector index with {self.index.ntotal} vectors of dimension {self.dim}.")

    def search(self, query_embedding: Optional[np.ndarray], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search FAISS vector store with normalized query embedding.
        Returns list of (chunk_index, similarity_score) sorted by cosine similarity.
        """
        if self.index is None or self.index.ntotal == 0 or query_embedding is None or faiss is None:
            return []

        qv = query_embedding.astype("float32").copy()
        norm = np.linalg.norm(qv)
        if norm == 0:
            return []
        qv = (qv / norm).reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        try:
            distances, indices = self.index.search(qv, k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.chunk_ids):
                    results.append((self.chunk_ids[idx], float(dist)))
            return results
        except Exception as exc:
            logger.warning(f"Error during FAISS search: {exc}")
            return []

    def save(self, directory: Path):
        """Save FAISS index and metadata mapping to disk directory."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "faiss_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "filing_id": self.filing_id,
                    "embedding_model": self.embedding_model,
                    "model_name": self.model_name,
                    "embedding_dimension": self.dim,
                    "dim": self.dim,
                    "similarity_metric": self.similarity_metric,
                    "chunk_count": len(self.chunk_ids),
                    "chunk_ids": self.chunk_ids,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        directory: Path,
        expected_embedding_model: Optional[str] = None,
        expected_filing_id: Optional[str] = None,
    ) -> Optional["FAISSVectorStore"]:
        """Load FAISS index and metadata mapping from disk directory."""
        index_path = directory / "faiss.index"
        meta_path = directory / "faiss_meta.json"
        if not index_path.exists() or not meta_path.exists() or faiss is None:
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            index_embedding_model = meta.get("embedding_model")
            if expected_embedding_model and index_embedding_model and index_embedding_model != expected_embedding_model:
                raise ValueError(
                    "Embedding model mismatch: "
                    f"configured={expected_embedding_model} index={index_embedding_model}"
                )

            store = cls(
                dim=meta.get("embedding_dimension", meta.get("dim", 384)),
                embedding_model=index_embedding_model or expected_embedding_model,
                model_name=meta.get("model_name"),
                similarity_metric=meta.get("similarity_metric", "cosine"),
                filing_id=meta.get("filing_id") or expected_filing_id,
            )
            store.chunk_ids = meta.get("chunk_ids", [])
            store.index = faiss.read_index(str(index_path))
            logger.info(f"Loaded persistent FAISS index from {directory} with {store.index.ntotal} vectors.")
            return store
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"Failed to load persistent FAISS index from {directory}: {exc}")
            return None
