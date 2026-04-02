"""Header- and code-block-preserving chunking logic.

Takes the list of section records produced by :mod:`src.ingestion.parser` and
applies a size-aware splitting strategy:

* **Code records** are emitted as-is — they are never split.
* **Prose records** are split at sentence boundaries when they exceed
  ``CHUNK_MAX_CHARS``.  A configurable overlap prefix taken from the tail of
  the previous sibling prose chunk can be prepended for retrieval context.

All output chunks preserve the full metadata of their source record plus
``chunk_index`` (ordinal within the source record's splits) and
``total_chunks`` (total splits for that record).
"""

from __future__ import annotations

import re
from copy import deepcopy

from src.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_prose(text: str, max_chars: int) -> list[str]:
    """Split *text* into sub-chunks of at most *max_chars* characters.

    Splits are made at sentence boundaries (after ``.``, ``!``, or ``?``)
    whenever possible.  If a single sentence exceeds *max_chars* it is kept
    whole rather than breaking mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # +1 for the space that was consumed by the split
        if current_len + sentence_len + 1 > max_chars and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_len = sentence_len
        else:
            current_parts.append(sentence)
            current_len += sentence_len + 1

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _overlap_prefix(previous_text: str, overlap_chars: int) -> str:
    """Return the last *overlap_chars* characters of *previous_text*.

    The prefix is trimmed to the nearest word boundary so it does not start
    mid-word.
    """
    if not previous_text or overlap_chars <= 0:
        return ""
    tail = previous_text[-overlap_chars:]
    # Advance to the first whitespace boundary so we don't start mid-word.
    first_space = tail.find(" ")
    if first_space != -1:
        tail = tail[first_space + 1 :]
    return tail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_records(
    records: list[dict],
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[dict]:
    """Convert a list of section records into a list of final chunks.

    Parameters
    ----------
    records:
        Output of :func:`src.ingestion.parser.parse_markdown` or similar.
    max_chars:
        Soft upper bound on prose chunk size in characters.
    overlap_chars:
        Number of trailing characters from the previous prose chunk to
        prepend to the next one for retrieval context.  Set to ``0`` to
        disable overlap.

    Returns
    -------
    list[dict]
        Chunks ready for embedding.  Every chunk dict contains all original
        record keys plus ``chunk_index`` and ``total_chunks``.
    """
    chunks: list[dict] = []
    last_prose_text: str = ""

    for record in records:
        if record["chunk_type"] == "code":
            # Code blocks are atomic — emit as a single chunk.
            chunk = deepcopy(record)
            chunk["chunk_index"] = 0
            chunk["total_chunks"] = 1
            chunks.append(chunk)
            # Code blocks do not contribute to prose overlap.
            continue

        # Prose record — may need splitting.
        sub_texts = _split_prose(record["text"], max_chars)
        total = len(sub_texts)

        for idx, sub_text in enumerate(sub_texts):
            chunk = deepcopy(record)

            # Prepend overlap from the last prose chunk on the first sub-chunk
            # of every prose record (skip if there is nothing to prepend).
            if idx == 0 and last_prose_text:
                prefix = _overlap_prefix(last_prose_text, overlap_chars)
                if prefix:
                    sub_text = prefix + " " + sub_text

            chunk["text"] = sub_text
            chunk["chunk_index"] = idx
            chunk["total_chunks"] = total
            chunks.append(chunk)

        # The overlap seed for the next prose record is the *original* last
        # sub-chunk (without any prefix already prepended to it).
        if sub_texts:
            last_prose_text = sub_texts[-1]

    return chunks
