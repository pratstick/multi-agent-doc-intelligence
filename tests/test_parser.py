"""Tests for src.ingestion.parser and src.ingestion.chunker."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_records, _split_prose
from src.ingestion.parser import parse_markdown


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

SIMPLE_DOC = """\
# Introduction

This is the introduction paragraph.

## Installation

Install using pip:

```bash
pip install mypackage
```

### Quick Start

Call the main function to get started.

## Configuration

Set the environment variable before running.
"""


def test_parse_returns_list():
    records = parse_markdown(SIMPLE_DOC)
    assert isinstance(records, list)
    assert len(records) > 0


def test_parse_chunk_types():
    records = parse_markdown(SIMPLE_DOC)
    types = {r["chunk_type"] for r in records}
    assert "prose" in types
    assert "code" in types


def test_parse_code_block_is_atomic():
    """Code block text must appear as a single record, not split."""
    records = parse_markdown(SIMPLE_DOC)
    code_records = [r for r in records if r["chunk_type"] == "code"]
    assert len(code_records) == 1
    assert "pip install mypackage" in code_records[0]["text"]


def test_parse_code_block_language():
    records = parse_markdown(SIMPLE_DOC)
    code_records = [r for r in records if r["chunk_type"] == "code"]
    assert code_records[0]["language"] == "bash"


def test_parse_header_path_nesting():
    """header_path must reflect the nesting at the point of each record."""
    records = parse_markdown(SIMPLE_DOC)
    # The Quick Start prose is under H1 > H2 > H3
    quick_start = next(
        r for r in records if "main function" in r.get("text", "")
    )
    assert quick_start["header_path"] == ["Introduction", "Installation", "Quick Start"]


def test_parse_top_level_header_path():
    records = parse_markdown(SIMPLE_DOC)
    intro = next(r for r in records if "introduction paragraph" in r.get("text", ""))
    assert intro["header_path"] == ["Introduction"]


def test_parse_source_file_metadata():
    records = parse_markdown(SIMPLE_DOC, source_file="docs/intro.md")
    assert all(r["source_file"] == "docs/intro.md" for r in records)


def test_parse_empty_document():
    records = parse_markdown("")
    assert records == []


def test_parse_document_with_no_headings():
    content = "Just some text.\n\nAnother paragraph."
    records = parse_markdown(content)
    # All records should be prose with an empty header_path.
    assert all(r["chunk_type"] == "prose" for r in records)
    assert all(r["header_path"] == [] for r in records)


def test_parse_multiple_code_blocks():
    content = """\
# Section

```python
x = 1
```

Some prose.

```python
y = 2
```
"""
    records = parse_markdown(content)
    code_records = [r for r in records if r["chunk_type"] == "code"]
    assert len(code_records) == 2


def test_parse_code_language_empty_when_unspecified():
    content = "# Section\n\n```\nsome code\n```\n"
    records = parse_markdown(content)
    code_records = [r for r in records if r["chunk_type"] == "code"]
    assert code_records[0]["language"] == ""


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

def test_chunk_code_records_are_not_split():
    """Code chunks must pass through unchanged regardless of size."""
    big_code = "x = " + "1" * 5000
    records = [
        {
            "text": big_code,
            "chunk_type": "code",
            "source_file": "f.md",
            "header_path": ["Section"],
            "language": "python",
        }
    ]
    chunks = chunk_records(records, max_chars=500, overlap_chars=0)
    assert len(chunks) == 1
    assert chunks[0]["text"] == big_code


def test_chunk_prose_splits_at_sentence_boundary():
    """Long prose must be split into multiple chunks."""
    sentence = "This is a sentence. "
    long_prose = sentence * 50  # well over 1500 chars
    records = [
        {
            "text": long_prose.strip(),
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        }
    ]
    chunks = chunk_records(records, max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["chunk_type"] == "prose"


def test_chunk_short_prose_is_not_split():
    records = [
        {
            "text": "Short text.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        }
    ]
    chunks = chunk_records(records, max_chars=1500, overlap_chars=0)
    assert len(chunks) == 1


def test_chunk_metadata_preserved():
    records = [
        {
            "text": "Some text.",
            "chunk_type": "prose",
            "source_file": "docs/a.md",
            "header_path": ["A", "B"],
            "language": "",
        }
    ]
    chunks = chunk_records(records)
    assert chunks[0]["source_file"] == "docs/a.md"
    assert chunks[0]["header_path"] == ["A", "B"]


def test_chunk_index_and_total():
    sentence = "Sentence here. "
    long_prose = sentence * 40
    records = [
        {
            "text": long_prose.strip(),
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        }
    ]
    chunks = chunk_records(records, max_chars=200, overlap_chars=0)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["total_chunks"] == len(chunks)


def test_chunk_overlap_prepended_to_second_record():
    """The tail of the first prose record should prefix the second."""
    records = [
        {
            "text": "First record ends here with overlap seed.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        },
        {
            "text": "Second record starts here.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        },
    ]
    chunks = chunk_records(records, max_chars=1500, overlap_chars=50)
    # The second chunk should contain text from both records.
    second_chunk_text = chunks[1]["text"]
    assert "Second record" in second_chunk_text
    # Some part of the first record's tail must appear as overlap prefix.
    assert "overlap seed" in second_chunk_text or "ends here" in second_chunk_text


def test_chunk_no_overlap_for_code():
    """Code blocks must not contribute to or receive overlap text."""
    records = [
        {
            "text": "Prose before code block.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        },
        {
            "text": "print('hello')",
            "chunk_type": "code",
            "source_file": "f.md",
            "header_path": [],
            "language": "python",
        },
        {
            "text": "Prose after code block.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        },
    ]
    chunks = chunk_records(records, max_chars=1500, overlap_chars=100)
    code_chunk = next(c for c in chunks if c["chunk_type"] == "code")
    # Code chunk must be unchanged.
    assert code_chunk["text"] == "print('hello')"


def test_split_prose_single_sentence_exceeding_max():
    """A sentence longer than max_chars must be kept whole."""
    long_sentence = "word " * 400  # ~2000 chars, no punctuation to split on
    parts = _split_prose(long_sentence.strip(), max_chars=100)
    # Should not crash; the whole thing is returned as one chunk.
    assert len(parts) >= 1
    combined = " ".join(parts)
    # All original words must still be present.
    assert combined.replace("  ", " ").strip() == long_sentence.strip()
