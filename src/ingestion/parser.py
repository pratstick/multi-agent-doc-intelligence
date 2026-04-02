"""Markdown-aware structural document parser.

Parses a Markdown document into a flat list of *section records* that
preserve structural boundaries.  Each record represents either a prose
section or a fenced code block and carries rich metadata.

The parser uses ``markdown-it-py`` to obtain a token stream, which it
walks linearly to collect:

* Heading hierarchy (``header_path``) at every point in the document.
* Fenced code blocks tagged with their language.
* Prose content grouped under their nearest heading.

No chunk-size splitting is done here; that responsibility belongs to
``chunker.py``.

Record schema (dict)
--------------------
``text``        : str   – raw text content of the section
``chunk_type``  : str   – ``"prose"`` | ``"code"``
``source_file`` : str   – path of the originating file (filled by caller)
``header_path`` : list[str] – ordered list of heading texts from root to leaf
``language``    : str   – programming language for code blocks (empty string
                           for prose)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from markdown_it import MarkdownIt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _heading_level(token_markup: str) -> int:
    """Return integer heading level from the ``markup`` field (e.g. ``"##"`` → 2)."""
    return len(token_markup)


def _update_header_path(header_path: list[str], level: int, text: str) -> list[str]:
    """Return a new header_path list reflecting a heading of *level* with *text*.

    The path is truncated to ``level - 1`` ancestors then the new heading is
    appended, so the list always reflects the current nesting.
    """
    new_path = header_path[: level - 1]
    new_path.append(text)
    return new_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_markdown(content: str, source_file: str = "") -> list[dict]:
    """Parse *content* (a Markdown string) into a list of section records.

    Parameters
    ----------
    content:
        Raw Markdown text.
    source_file:
        Originating file path stored in each record's metadata.

    Returns
    -------
    list[dict]
        Ordered list of section records.
    """
    md = MarkdownIt()
    tokens = md.parse(content)

    records: list[dict] = []
    header_path: list[str] = []

    # We accumulate prose lines between structural boundaries.
    prose_lines: list[str] = []

    def _flush_prose() -> None:
        """Emit accumulated prose lines as a record if non-empty."""
        text = "\n".join(prose_lines).strip()
        if text:
            records.append(
                {
                    "text": text,
                    "chunk_type": "prose",
                    "source_file": source_file,
                    "header_path": list(header_path),
                    "language": "",
                }
            )
        prose_lines.clear()

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # ------------------------------------------------------------------
        # Heading tokens: markdown-it emits  heading_open / inline / heading_close
        # ------------------------------------------------------------------
        if token.type == "heading_open":
            _flush_prose()
            level = _heading_level(token.markup)
            # The inline token immediately follows
            inline_token = tokens[i + 1] if i + 1 < len(tokens) else None
            heading_text = inline_token.content if inline_token else ""
            header_path = _update_header_path(header_path, level, heading_text)
            i += 3  # skip heading_open, inline, heading_close
            continue

        # ------------------------------------------------------------------
        # Fenced code blocks: fence token contains the code directly
        # ------------------------------------------------------------------
        if token.type == "fence":
            _flush_prose()
            language = token.info.strip() if token.info else ""
            records.append(
                {
                    "text": token.content.rstrip("\n"),
                    "chunk_type": "code",
                    "source_file": source_file,
                    "header_path": list(header_path),
                    "language": language,
                }
            )
            i += 1
            continue

        # ------------------------------------------------------------------
        # Inline content inside paragraphs / list items
        # ------------------------------------------------------------------
        if token.type == "inline" and token.content:
            prose_lines.append(token.content)
            i += 1
            continue

        i += 1

    _flush_prose()
    return records


def parse_file(path: str | Path) -> list[dict]:
    """Convenience wrapper that reads *path* and calls :func:`parse_markdown`."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return parse_markdown(content, source_file=str(path))


def parse_directory(directory: str | Path) -> Generator[dict, None, None]:
    """Yield section records from every ``*.md`` file in *directory* (recursive)."""
    directory = Path(directory)
    for md_file in sorted(directory.rglob("*.md")):
        yield from parse_file(md_file)
