"""Tests for src.retrieval.embedder and src.retrieval.vectorstore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small embedding dimension for tests


def _make_embedder(dim: int = DIM) -> MagicMock:
    """Return a mock Embedder that returns random float32 vectors."""
    embedder = MagicMock()
    embedder.dimension = dim

    def _embed_texts(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        return rng.random((len(texts), dim)).astype(np.float32)

    def _embed_query(query: str) -> np.ndarray:
        rng = np.random.default_rng(seed=0)
        return rng.random((1, dim)).astype(np.float32)

    embedder.embed_texts.side_effect = _embed_texts
    embedder.embed_query.side_effect = _embed_query
    return embedder


def _make_chunks(n: int = 10, chunk_type: str = "prose") -> list[dict]:
    return [
        {
            "text": f"Sample text {i}.",
            "chunk_type": chunk_type,
            "source_file": "docs/test.md",
            "header_path": ["Section"],
            "language": "",
            "chunk_index": 0,
            "total_chunks": 1,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# VectorStore.build + query
# ---------------------------------------------------------------------------

class TestVectorStoreBuildQuery:
    def test_build_and_query_returns_results(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = _make_chunks(10)
        store.build(chunks)

        results = store.query("what is this?", top_k=3)
        assert len(results) == 3

    def test_query_result_has_text_key(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = _make_chunks(5)
        store.build(chunks)

        results = store.query("query", top_k=1)
        assert "text" in results[0]

    def test_query_result_has_score_key(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = _make_chunks(5)
        store.build(chunks)

        results = store.query("query", top_k=1)
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    def test_top_k_respected(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = _make_chunks(20)
        store.build(chunks)

        for k in (1, 3, 5):
            results = store.query("query", top_k=k)
            assert len(results) == k

    def test_top_k_capped_at_index_size(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = _make_chunks(3)
        store.build(chunks)

        results = store.query("query", top_k=100)
        assert len(results) <= 3

    def test_chunk_type_filter_prose(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        prose = _make_chunks(5, chunk_type="prose")
        code = _make_chunks(5, chunk_type="code")
        store.build(prose + code)

        results = store.query("query", top_k=5, chunk_type="prose")
        assert all(r["chunk_type"] == "prose" for r in results)

    def test_chunk_type_filter_code(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        prose = _make_chunks(5, chunk_type="prose")
        code = _make_chunks(5, chunk_type="code")
        store.build(prose + code)

        results = store.query("query", top_k=5, chunk_type="code")
        assert all(r["chunk_type"] == "code" for r in results)

    def test_no_filter_returns_mixed_types(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        prose = _make_chunks(5, chunk_type="prose")
        code = _make_chunks(5, chunk_type="code")
        store.build(prose + code)

        results = store.query("query", top_k=10)
        types = {r["chunk_type"] for r in results}
        assert len(types) == 2

    def test_query_before_build_raises(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        with pytest.raises(RuntimeError):
            store.query("query")

    def test_metadata_preserved_in_results(self):
        embedder = _make_embedder()
        store = VectorStore(embedder)
        chunks = [
            {
                "text": "Special text.",
                "chunk_type": "prose",
                "source_file": "special.md",
                "header_path": ["A", "B"],
                "language": "",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]
        store.build(chunks)

        results = store.query("Special", top_k=1)
        assert results[0]["source_file"] == "special.md"
        assert results[0]["header_path"] == ["A", "B"]


# ---------------------------------------------------------------------------
# VectorStore.save + load
# ---------------------------------------------------------------------------

class TestVectorStorePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        embedder = _make_embedder()
        store = VectorStore(embedder, index_dir=tmp_path)
        chunks = _make_chunks(5)
        store.build(chunks)
        store.save()

        # Load into a fresh store instance.
        store2 = VectorStore(embedder, index_dir=tmp_path)
        store2.load()

        results = store2.query("query", top_k=3)
        assert len(results) == 3

    def test_save_before_build_raises(self, tmp_path):
        embedder = _make_embedder()
        store = VectorStore(embedder, index_dir=tmp_path)
        with pytest.raises(RuntimeError):
            store.save()

    def test_load_missing_files_raises(self, tmp_path):
        embedder = _make_embedder()
        store = VectorStore(embedder, index_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load()

    def test_saved_files_exist(self, tmp_path):
        embedder = _make_embedder()
        store = VectorStore(embedder, index_dir=tmp_path)
        store.build(_make_chunks(3))
        store.save()

        assert (tmp_path / "faiss.index").exists()
        assert (tmp_path / "faiss_meta.json").exists()
