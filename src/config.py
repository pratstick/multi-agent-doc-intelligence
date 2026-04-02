"""Central configuration for the multi-agent RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Soft maximum number of characters for a prose chunk before it is split at a
# sentence boundary.  Code blocks are never split regardless of this value.
CHUNK_MAX_CHARS: int = int(os.environ.get("CHUNK_MAX_CHARS", "1500"))
# Number of trailing characters from the previous prose chunk to prepend as
# overlap context.
CHUNK_OVERLAP_CHARS: int = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))

# ---------------------------------------------------------------------------
# FAISS index file names (written inside INDEX_DIR)
# ---------------------------------------------------------------------------
FAISS_INDEX_FILE: str = "faiss.index"
FAISS_META_FILE: str = "faiss_meta.json"
