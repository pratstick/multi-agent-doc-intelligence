"""Entry point for the multi-agent RAG pipeline.

Modes
-----
ingest
    Parse all ``*.md`` files in ``data/raw/``, chunk them, build the FAISS
    index, and save it to ``data/index/``.  Run this once (or whenever the
    documentation changes).

query
    Load the persisted FAISS index and run an interactive query loop.

Usage
-----
::

    # Ingest documentation
    python main.py ingest

    # Interactive query session
    python main.py query

    # Single non-interactive query
    python main.py query --question "What is LangGraph?"
"""

from __future__ import annotations

import argparse
import sys

from src.config import RAW_DIR
from src.ingestion.chunker import chunk_records
from src.ingestion.parser import parse_directory
from src.pipeline.graph import build_graph
from src.retrieval.embedder import Embedder
from src.retrieval.vectorstore import VectorStore


def _ingest() -> None:
    """Parse, chunk, embed, and index all documents in data/raw/."""
    print(f"[ingest] Scanning {RAW_DIR} for Markdown files …")
    records = list(parse_directory(RAW_DIR))
    if not records:
        print(
            "[ingest] No Markdown files found.  "
            f"Place .md files in {RAW_DIR} and try again."
        )
        sys.exit(1)

    print(f"[ingest] Parsed {len(records)} section records.")
    chunks = chunk_records(records)
    print(f"[ingest] Produced {len(chunks)} chunks after splitting.")

    print("[ingest] Loading embedding model …")
    embedder = Embedder()
    store = VectorStore(embedder)

    print("[ingest] Building FAISS index …")
    store.build(chunks)
    store.save()
    print("[ingest] Index saved.  Ingestion complete.")


def _query(question: str | None) -> None:
    """Load index and run a single question or an interactive loop."""
    print("[query] Loading embedding model and FAISS index …")
    embedder = Embedder()
    store = VectorStore(embedder)
    store.load()

    graph = build_graph(store)

    if question:
        _run_query(graph, question)
        return

    print("[query] Interactive mode.  Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[query] Exiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[query] Goodbye.")
            break
        _run_query(graph, user_input)


def _run_query(graph: object, question: str) -> None:
    result = graph.invoke({"query": question, "messages": []})
    intent = result.get("intent", "unknown")
    response = result.get("response", "(no response)")
    print(f"\n[intent: {intent}]\n{response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-agent RAG pipeline for technical documentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Ingest documents and build the FAISS index.")

    query_parser = subparsers.add_parser(
        "query", help="Query the pipeline (interactive or single question)."
    )
    query_parser.add_argument(
        "--question", "-q", type=str, default=None, help="Single question to answer."
    )

    args = parser.parse_args()

    if args.command == "ingest":
        _ingest()
    elif args.command == "query":
        _query(args.question)


if __name__ == "__main__":
    main()
