"""Tests for main.py CLI interface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import main


class TestCLIArgumentParsing:
    def test_ingest_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "ingest"])
        with patch("main._ingest") as mock_ingest:
            main.main()
            mock_ingest.assert_called_once_with(input_dir=None)

    def test_ingest_with_custom_dir(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "ingest", "--dir", "custom/docs"])
        with patch("main._ingest") as mock_ingest:
            main.main()
            mock_ingest.assert_called_once_with(input_dir="custom/docs")

    def test_query_interactive_mode(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["main.py", "query"])
        with patch("main._query") as mock_query:
            main.main()
            mock_query.assert_called_once_with(None, show_sources=False)

    def test_query_single_question_with_sources(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["main.py", "query", "-q", "What is state?", "--show-sources"],
        )
        with patch("main._query") as mock_query:
            main.main()
            mock_query.assert_called_once_with("What is state?", show_sources=True)


class TestIngestCommand:
    @patch("main.parse_directory")
    @patch("main.chunk_records")
    @patch("main.Embedder")
    @patch("main.VectorStore")
    def test_ingest_success(
        self, mock_vstore_cls, mock_embedder_cls, mock_chunk, mock_parse, tmp_path
    ):
        mock_parse.return_value = [{"text": "Doc text", "chunk_type": "prose"}]
        mock_chunk.return_value = [{"text": "Doc text", "chunk_type": "prose"}]

        mock_store = MagicMock()
        mock_vstore_cls.return_value = mock_store

        main._ingest(input_dir=tmp_path)

        mock_store.build.assert_called_once()
        mock_store.save.assert_called_once()

    @patch("main.parse_directory")
    def test_ingest_empty_directory_exits(self, mock_parse, tmp_path):
        mock_parse.return_value = []
        with pytest.raises(SystemExit) as excinfo:
            main._ingest(input_dir=tmp_path)
        assert excinfo.value.code == 1


class TestQueryCommand:
    @patch("main.VectorStore")
    @patch("main.Embedder")
    def test_query_missing_index_exits(self, mock_embedder_cls, mock_vstore_cls):
        mock_store = MagicMock()
        mock_store.load.side_effect = FileNotFoundError("Not found")
        mock_vstore_cls.return_value = mock_store

        with pytest.raises(SystemExit) as excinfo:
            main._query("test question")
        assert excinfo.value.code == 1

    @patch("main.VectorStore")
    @patch("main.Embedder")
    @patch("main.build_graph")
    def test_query_single_question_runs(
        self, mock_build_graph, mock_embedder_cls, mock_vstore_cls, capsys
    ):
        mock_store = MagicMock()
        mock_vstore_cls.return_value = mock_store

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "intent": "conceptual",
            "response": "This is the answer.",
            "retrieved_chunks": [
                {
                    "source_file": "doc.md",
                    "header_path": ["Intro"],
                    "chunk_type": "prose",
                    "score": 0.05,
                }
            ],
        }
        mock_build_graph.return_value = mock_graph

        main._query("What is LangGraph?", show_sources=True)

        captured = capsys.readouterr().out
        assert "CONCEPTUAL" in captured
        assert "This is the answer." in captured
        assert "Intro" in captured

    @patch("main.VectorStore")
    @patch("main.Embedder")
    @patch("main.build_graph")
    def test_query_interactive_loop_exit(
        self, mock_build_graph, mock_embedder_cls, mock_vstore_cls, monkeypatch, capsys
    ):
        mock_store = MagicMock()
        mock_vstore_cls.return_value = mock_store
        mock_build_graph.return_value = MagicMock()

        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        main._query(None)
        captured = capsys.readouterr().out
        assert "Goodbye." in captured
