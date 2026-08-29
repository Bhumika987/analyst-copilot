import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import embedding_service
import retrieval
import trace_practice_questions
from config import (
    FINANCE_SMALL_MODEL_NAME,
    FINLANG_MODEL_NAME,
    NORMAL_EMBEDDING_MODEL_NAME,
    get_embedding_model_name,
)
from retrieval import FilingIndex
from vector_store import FAISSVectorStore, faiss


class EmbeddingModelConfigTests(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.get("EMBEDDING_MODEL")
        os.environ.pop("EMBEDDING_MODEL", None)
        embedding_service._PROVIDER_CACHE.clear()

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("EMBEDDING_MODEL", None)
        else:
            os.environ["EMBEDDING_MODEL"] = self._old_env
        embedding_service._PROVIDER_CACHE.clear()

    def test_default_configuration_selects_normal(self):
        self.assertEqual(get_embedding_model_name(), "normal")
        self.assertIsInstance(embedding_service.get_embedding_provider(), embedding_service.NormalEmbeddingProvider)

    def test_env_configuration_selects_finlang(self):
        os.environ["EMBEDDING_MODEL"] = "finlang"
        self.assertEqual(get_embedding_model_name(), "finlang")
        self.assertIsInstance(embedding_service.get_embedding_provider(), embedding_service.FinLangEmbeddingProvider)

    def test_env_configuration_selects_financesmall(self):
        os.environ["EMBEDDING_MODEL"] = "financesmall"
        self.assertEqual(get_embedding_model_name(), "financesmall")
        self.assertIsInstance(
            embedding_service.get_embedding_provider(), embedding_service.FinanceSmallEmbeddingProvider
        )

    def test_normal_provider_uses_existing_model(self):
        provider = embedding_service.NormalEmbeddingProvider()
        self.assertEqual(provider.key, "normal")
        self.assertEqual(provider.model_name, NORMAL_EMBEDDING_MODEL_NAME)
        self.assertEqual(provider.query_prefix, "")
        self.assertEqual(provider.document_prefix, "")

    def test_finlang_provider_loads_configured_model_without_prefixes(self):
        provider = embedding_service.FinLangEmbeddingProvider()
        self.assertEqual(provider.key, "finlang")
        self.assertEqual(provider.model_name, FINLANG_MODEL_NAME)
        self.assertEqual(provider._format_query("hello"), "hello")
        self.assertEqual(provider._format_documents(["a", "b"]), ["a", "b"])

    def test_financesmall_provider_loads_configured_model_without_prefixes(self):
        provider = embedding_service.FinanceSmallEmbeddingProvider()
        self.assertEqual(provider.key, "financesmall")
        self.assertEqual(provider.model_name, FINANCE_SMALL_MODEL_NAME)
        self.assertEqual(provider._format_query("hello"), "hello")
        self.assertEqual(provider._format_documents(["a", "b"]), ["a", "b"])

    def test_query_and_document_embeddings_use_same_provider(self):
        os.environ["EMBEDDING_MODEL"] = "finlang"

        class FakeProvider:
            key = "finlang"
            model_name = FINLANG_MODEL_NAME
            similarity_metric = "cosine"

            def embed_query(self, text):
                return np.array([1.0, 0.0], dtype="float32")

            def embed_documents(self, texts):
                return np.array([[1.0, 0.0] for _ in texts], dtype="float32")

        with patch("embedding_service.get_embedding_provider", return_value=FakeProvider()):
            provider = embedding_service.get_embedding_service()
            self.assertEqual(provider.key, "finlang")
            self.assertEqual(provider.embed_query("q").shape, (2,))
            self.assertEqual(provider.embed_documents(["d"]).shape, (1, 2))

    def test_model_specific_index_directories_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_index_dir = retrieval.INDEX_DIR
            retrieval.INDEX_DIR = Path(tmp)
            try:
                os.environ["EMBEDDING_MODEL"] = "normal"
                normal = FilingIndex("DOC", chunks=[{"text": "alpha", "chunk_index": 0}])
                self.assertEqual(normal.vector_index_dir(), Path(tmp) / "DOC" / "normal")

                os.environ["EMBEDDING_MODEL"] = "finlang"
                finlang = FilingIndex("DOC", chunks=[{"text": "alpha", "chunk_index": 0}])
                self.assertEqual(finlang.vector_index_dir(), Path(tmp) / "DOC" / "finlang")
            finally:
                retrieval.INDEX_DIR = old_index_dir

    def test_bm25_persists_at_document_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_index_dir = retrieval.INDEX_DIR
            retrieval.INDEX_DIR = Path(tmp)
            try:
                os.environ["EMBEDDING_MODEL"] = "finlang"
                idx = FilingIndex("DOC", chunks=[{"text": "alpha beta", "chunk_index": 0}])
                idx.build_bm25()
                idx.vector_store = FAISSVectorStore(
                    dim=2,
                    embedding_model="finlang",
                    model_name=FINLANG_MODEL_NAME,
                    filing_id="DOC",
                )
                idx.vector_store.chunk_ids = [0]
                idx.save()

                self.assertTrue((Path(tmp) / "DOC" / "bm25.pkl").exists())
                self.assertTrue((Path(tmp) / "DOC" / "chunks.json").exists())
                self.assertTrue((Path(tmp) / "DOC" / "finlang" / "faiss_meta.json").exists())
                self.assertFalse((Path(tmp) / "DOC" / "normal" / "faiss_meta.json").exists())
                with open(Path(tmp) / "DOC" / "bm25.pkl", "rb") as f:
                    self.assertIsNotNone(pickle.load(f))
            finally:
                retrieval.INDEX_DIR = old_index_dir

    @unittest.skipIf(faiss is None, "faiss is not installed")
    def test_index_model_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1.0, 0.0]], dtype="float32"))
            faiss.write_index(index, str(directory / "faiss.index"))
            (directory / "faiss_meta.json").write_text(
                json.dumps({"embedding_model": "normal", "dim": 2, "chunk_ids": [0]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Embedding model mismatch"):
                FAISSVectorStore.load(directory, expected_embedding_model="finlang")

    def test_query_loads_model_specific_index_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_index_dir = retrieval.INDEX_DIR
            retrieval.INDEX_DIR = Path(tmp)
            try:
                root = Path(tmp) / "DOC"
                root.mkdir(parents=True)
                (root / "chunks.json").write_text(json.dumps([{"text": "alpha", "chunk_index": 0}]), encoding="utf-8")
                (root / "metadata.json").write_text("{}", encoding="utf-8")
                with open(root / "bm25.pkl", "wb") as f:
                    pickle.dump(None, f)
                (root / "finlang").mkdir()
                (root / "finlang" / "faiss_meta.json").write_text(
                    json.dumps({"embedding_model": "finlang", "dim": 2, "chunk_ids": [0]}),
                    encoding="utf-8",
                )

                os.environ["EMBEDDING_MODEL"] = "finlang"
                idx = FilingIndex.load("DOC")
                self.assertEqual(idx.vector_index_dir(), root / "finlang")
            finally:
                retrieval.INDEX_DIR = old_index_dir

    def test_trace_records_selected_embedding_model(self):
        os.environ["EMBEDDING_MODEL"] = "finlang"
        diag = trace_practice_questions.retrieval_diagnostics(
            {"financebench_id": "qid", "evidence": []},
            [],
        )
        self.assertEqual(diag["question_id"], "qid")
        self.assertEqual(get_embedding_model_name(), "finlang")


if __name__ == "__main__":
    unittest.main()
