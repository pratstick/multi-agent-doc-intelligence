"""FAISS index build, persist, and query helpers.

Manages a local FAISS flat-L2 index together with a JSON sidecar file that
stores the chunk metadata for each vector row.  Retrieval supports optional
filtering by ``chunk_type`` so that the Router Agent's intent can be used to
narrow the search space before passing context to a downstream agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.config import (
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    INDEX_DIR,
    RETRIEVAL_TOP_K,
)
from src.retrieval.embedder import Embedder


class VectorStore:
    """Wraps a FAISS flat index and its associated metadata store.

    Parameters
    ----------
    embedder:
        :class:`~src.retrieval.embedder.Embedder` instance used to encode
        queries at retrieval time.
    index_dir:
        Directory where the FAISS index and metadata sidecar are persisted.
        Defaults to :data:`src.config.INDEX_DIR`.
    """

    def __init__(
        self,
        embedder: Embedder,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        self._embedder = embedder
        self._index_dir = Path(index_dir)
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Build (or rebuild) the FAISS index from *chunks*.

        Parameters
        ----------
        chunks:
            List of chunk dicts as produced by
            :func:`src.ingestion.chunker.chunk_records`.  Each dict must have
            a ``"text"`` key; all other keys are stored as metadata.
        """
        texts = [c["text"] for c in chunks]
        vectors = self._embedder.embed_texts(texts)
        dim = vectors.shape[1]

        self._index = faiss.IndexFlatL2(dim)
        self._index.add(vectors)
        self._metadata = [
            {k: v for k, v in chunk.items() if k != "text"} | {"text": chunk["text"]}
            for chunk in chunks
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the index and metadata to :attr:`_index_dir`."""
        if self._index is None:
            raise RuntimeError("Index has not been built yet.  Call build() first.")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_dir / FAISS_INDEX_FILE))
        meta_path = self._index_dir / FAISS_META_FILE
        meta_path.write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2))

    def load(self) -> None:
        """Load the index and metadata from :attr:`_index_dir`."""
        index_path = self._index_dir / FAISS_INDEX_FILE
        meta_path = self._index_dir / FAISS_META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {self._index_dir}. "
                "Run the ingestion pipeline first."
            )
        self._index = faiss.read_index(str(index_path))
        self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = RETRIEVAL_TOP_K,
        chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """Return the top-*k* most similar chunks for *query_text*.

        Parameters
        ----------
        query_text:
            The natural-language query to embed and search.
        top_k:
            Maximum number of results to return.
        chunk_type:
            When provided (``"prose"`` or ``"code"``), only chunks of that
            type are returned.  The search still scans the full index for
            efficiency; filtering is applied post-hoc.

        Returns
        -------
        list[dict]
            Chunk dicts ordered by ascending L2 distance (most relevant first).
        """
        if self._index is None:
            raise RuntimeError("Index is not loaded.  Call load() or build() first.")

        query_vec = self._embedder.embed_query(query_text)

        # Over-fetch so that filtering doesn't leave us short.
        fetch_k = top_k * 4 if chunk_type else top_k
        fetch_k = min(fetch_k, self._index.ntotal)
        if fetch_k == 0:
            return []

        distances, indices = self._index.search(query_vec, fetch_k)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self._metadata[idx])
            meta["score"] = float(dist)
            if chunk_type is None or meta.get("chunk_type") == chunk_type:
                results.append(meta)
            if len(results) == top_k:
                break

        return results
