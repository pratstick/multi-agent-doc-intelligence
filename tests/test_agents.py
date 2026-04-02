"""Tests for the three agent node functions.

All tests that would invoke the Groq LLM are mocked at the ChatGroq layer so
that they run without network access or API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents import conceptual_agent, implementation_agent, router_agent
from src.pipeline.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _base_state(**overrides) -> AgentState:
    base: AgentState = {
        "query": "What is LangGraph?",
        "intent": "",
        "retrieved_chunks": [],
        "response": "",
        "messages": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Router Agent
# ---------------------------------------------------------------------------

class TestRouterAgent:
    @patch("src.agents.router_agent.ChatGroq")
    def test_routes_conceptual_query(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("conceptual")
        mock_groq_cls.return_value = mock_llm

        state = _base_state(query="What is LangGraph?")
        result = router_agent.route(state)

        assert result["intent"] == "conceptual"

    @patch("src.agents.router_agent.ChatGroq")
    def test_routes_implementation_query(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("implementation")
        mock_groq_cls.return_value = mock_llm

        state = _base_state(query="Write code to create a LangGraph graph.")
        result = router_agent.route(state)

        assert result["intent"] == "implementation"

    @patch("src.agents.router_agent.ChatGroq")
    def test_defaults_to_conceptual_on_ambiguous_response(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("I'm not sure")
        mock_groq_cls.return_value = mock_llm

        state = _base_state(query="Tell me something.")
        result = router_agent.route(state)

        assert result["intent"] == "conceptual"

    @patch("src.agents.router_agent.ChatGroq")
    def test_messages_are_accumulated(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("conceptual")
        mock_groq_cls.return_value = mock_llm

        state = _base_state()
        result = router_agent.route(state)

        # system + human + ai = at least 3 messages added
        assert len(result["messages"]) >= 3

    @patch("src.agents.router_agent.ChatGroq")
    def test_original_query_preserved(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("conceptual")
        mock_groq_cls.return_value = mock_llm

        state = _base_state(query="Explain transformers.")
        result = router_agent.route(state)

        assert result["query"] == "Explain transformers."


# ---------------------------------------------------------------------------
# Conceptual Agent
# ---------------------------------------------------------------------------

class TestConceptualAgent:
    def _make_store(self, chunks: list[dict]) -> MagicMock:
        store = MagicMock()
        store.query.return_value = chunks
        return store

    @patch("src.agents.conceptual_agent.ChatGroq")
    def test_returns_response(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("LangGraph is a framework.")
        mock_groq_cls.return_value = mock_llm

        store = self._make_store(
            [
                {
                    "text": "LangGraph manages state via TypedDict.",
                    "chunk_type": "prose",
                    "source_file": "docs/intro.md",
                    "header_path": ["Introduction"],
                    "language": "",
                }
            ]
        )
        state = _base_state(query="What is LangGraph?")
        result = conceptual_agent.answer(state, vector_store=store)

        assert result["response"] == "LangGraph is a framework."

    @patch("src.agents.conceptual_agent.ChatGroq")
    def test_retrieved_chunks_populated(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("Answer.")
        mock_groq_cls.return_value = mock_llm

        chunk = {
            "text": "Some prose.",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        }
        store = self._make_store([chunk])
        state = _base_state()
        result = conceptual_agent.answer(state, vector_store=store)

        assert len(result["retrieved_chunks"]) == 1

    @patch("src.agents.conceptual_agent.ChatGroq")
    def test_queries_prose_chunks_only(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("Answer.")
        mock_groq_cls.return_value = mock_llm

        store = self._make_store([])
        state = _base_state()
        conceptual_agent.answer(state, vector_store=store)

        # Must request chunk_type="prose"
        store.query.assert_called_once()
        _, kwargs = store.query.call_args
        assert kwargs.get("chunk_type") == "prose"


# ---------------------------------------------------------------------------
# Implementation Agent
# ---------------------------------------------------------------------------

class TestImplementationAgent:
    def _make_store(self, code_chunks: list[dict], prose_chunks: list[dict]) -> MagicMock:
        store = MagicMock()

        def _query(query_text, top_k, chunk_type=None):
            if chunk_type == "code":
                return code_chunks
            if chunk_type == "prose":
                return prose_chunks
            return code_chunks + prose_chunks

        store.query.side_effect = _query
        return store

    @patch("src.agents.implementation_agent.ChatGroq")
    def test_returns_code_response(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("```python\nx = 1\n```")
        mock_groq_cls.return_value = mock_llm

        store = self._make_store(
            code_chunks=[
                {
                    "text": "x = 1",
                    "chunk_type": "code",
                    "source_file": "f.md",
                    "header_path": [],
                    "language": "python",
                }
            ],
            prose_chunks=[],
        )
        state = _base_state(query="Show me how to assign a variable.")
        result = implementation_agent.answer(state, vector_store=store)

        assert "```python" in result["response"]

    @patch("src.agents.implementation_agent.ChatGroq")
    def test_queries_both_chunk_types(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("Answer.")
        mock_groq_cls.return_value = mock_llm

        store = self._make_store(code_chunks=[], prose_chunks=[])
        state = _base_state()
        implementation_agent.answer(state, vector_store=store)

        call_kwargs = [call.kwargs for call in store.query.call_args_list]
        requested_types = {kw.get("chunk_type") for kw in call_kwargs}
        assert "code" in requested_types
        assert "prose" in requested_types

    @patch("src.agents.implementation_agent.ChatGroq")
    def test_combined_chunks_in_state(self, mock_groq_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_ai_message("Answer.")
        mock_groq_cls.return_value = mock_llm

        code_chunk = {
            "text": "code",
            "chunk_type": "code",
            "source_file": "f.md",
            "header_path": [],
            "language": "python",
        }
        prose_chunk = {
            "text": "prose",
            "chunk_type": "prose",
            "source_file": "f.md",
            "header_path": [],
            "language": "",
        }
        store = self._make_store([code_chunk], [prose_chunk])
        state = _base_state()
        result = implementation_agent.answer(state, vector_store=store)

        types = {c["chunk_type"] for c in result["retrieved_chunks"]}
        assert "code" in types
        assert "prose" in types
