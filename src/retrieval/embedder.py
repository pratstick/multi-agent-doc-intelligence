"""Embedding model wrapper.

Wraps ``sentence-transformers`` to produce dense vector embeddings for both
document chunks (offline ingestion) and user queries (online retrieval).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class Embedder:
    """Thin wrapper around :class:`sentence_transformers.SentenceTransformer`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the value of
        ``EMBEDDING_MODEL`` in :mod:`src.config`.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a float32 numpy array of shape ``(len(texts), dim)``."""
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Return a float32 numpy array of shape ``(1, dim)``."""
        return self.embed_texts([query])

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return self._model.get_sentence_embedding_dimension()
