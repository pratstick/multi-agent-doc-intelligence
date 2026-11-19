"""Entry point for the multi-agent RAG pipeline.

Modes
-----
ingest
    Parse all ``*.md`` files in ``data/raw/`` (or a specified directory), chunk
    them, build the FAISS index, and save it to ``data/index/``. Run this once
    (or whenever the documentation changes).

query
    Load the persisted FAISS index and run an interactive query loop or a single
    question.

Usage
-----
::

    # Ingest documentation
    python main.py ingest

    # Ingest from a custom directory
    python main.py ingest --dir /path/to/docs

    # Interactive query session
    python main.py query

    # Single question
    python main.py query --question "What is LangGraph?"

    # Single question with source citation details
    python main.py query -q "How do I build a StateGraph?" --show-sources
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import GROQ_API_KEY, RAW_DIR
from src.ingestion.chunker import chunk_records
from src.ingestion.parser import parse_directory
from src.pipeline.graph import build_graph
from src.retrieval.embedder import Embedder
from src.retrieval.vectorstore import VectorStore


def _ingest(input_dir: Path | str | None = None) -> None:
    """Parse, chunk, embed, and index all documents in the target directory."""
    target_dir = Path(input_dir) if input_dir else RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Scanning {target_dir} for Markdown files …")
    records = list(parse_directory(target_dir))
    if not records:
        print(
            f"[ingest] Error: No Markdown files found in {target_dir}.\n"
            f"         Place .md files in {target_dir} and try again."
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
    print("[ingest] Index successfully saved. Ingestion complete.")


def _query(question: str | None, show_sources: bool = False) -> None:
    """Load index and run a single question or an interactive loop."""
    if not GROQ_API_KEY:
        print(
            "[warning] GROQ_API_KEY is not set in environment or .env file.\n"
            "          Querying requires a valid Groq API key."
        )

    print("[query] Loading embedding model and FAISS index …")
    embedder = Embedder()
    store = VectorStore(embedder)

    try:
        store.load()
    except FileNotFoundError as err:
        print(f"[error] {err}")
        print("        Run 'python main.py ingest' to build the index first.")
        sys.exit(1)

    graph = build_graph(store)

    if question:
        _run_query(graph, question, show_sources=show_sources)
        return

    print("[query] Interactive mode started. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[query] Exiting session.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("[query] Goodbye.")
            break
        _run_query(graph, user_input, show_sources=show_sources)


def _run_query(graph: object, question: str, show_sources: bool = False) -> None:
    """Execute a query through the LangGraph pipeline and print formatted results."""
    try:
        result = graph.invoke({"query": question, "messages": []})
    except Exception as err:
        print(f"\n[error] Pipeline execution failed: {err}\n")
        return

    intent = result.get("intent", "unknown")
    response = result.get("response", "(no response)")
    retrieved_chunks = result.get("retrieved_chunks", [])

    print("\n" + "=" * 60)
    print(f"  Routing Intent : {intent.upper()}")
    print("=" * 60)

    if show_sources and retrieved_chunks:
        print("\n[Retrieved Context]")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source_file", "unknown")
            header = " > ".join(chunk.get("header_path", [])) or "Root"
            chunk_type = chunk.get("chunk_type", "prose")
            score = chunk.get("score")
            score_str = f" | L2 distance: {score:.4f}" if score is not None else ""
            print(f"  [{i}] ({chunk_type}) {header} (from: {source}{score_str})")

    print("\n[Answer]")
    print(response.strip())
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-agent RAG pipeline for technical documentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents and build the FAISS index."
    )
    ingest_parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=None,
        help="Custom directory containing Markdown documentation files (defaults to data/raw/).",
    )

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query", help="Query the pipeline (interactive or single question)."
    )
    query_parser.add_argument(
        "--question", "-q", type=str, default=None, help="Single question to answer."
    )
    query_parser.add_argument(
        "--show-sources",
        "-s",
        action="store_true",
        help="Display retrieved section headers, file sources, and similarity scores.",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        _ingest(input_dir=args.dir)
    elif args.command == "query":
        _query(args.question, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
