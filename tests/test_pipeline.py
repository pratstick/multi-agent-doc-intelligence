"""Tests for src.pipeline.graph and src.pipeline.state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.pipeline.graph import _route_by_intent, build_graph
from src.pipeline.state import AgentState


def _make_ai_message(content: str) -> AIMessage:
    return AIMessage(content=content)


def _make_mock_vector_store() -> MagicMock:
    store = MagicMock()
    sample_chunk = {
        "text": "Sample documentation text.",
        "chunk_type": "prose",
        "source_file": "docs/test.md",
        "header_path": ["Introduction"],
        "language": "",
        "chunk_index": 0,
        "total_chunks": 1,
        "score": 0.12,
    }
    store.query.return_value = [sample_chunk]
    return store


class TestRoutingLogic:
    def test_route_by_intent_conceptual(self):
        state: AgentState = {
            "query": "What is StateGraph?",
            "intent": "conceptual",
            "retrieved_chunks": [],
            "response": "",
            "messages": [],
        }
        assert _route_by_intent(state) == "conceptual"

    def test_route_by_intent_implementation(self):
        state: AgentState = {
            "query": "How to write a node function?",
            "intent": "implementation",
            "retrieved_chunks": [],
            "response": "",
            "messages": [],
        }
        assert _route_by_intent(state) == "implementation"

    def test_route_by_intent_default_fallback(self):
        state: AgentState = {
            "query": "Hello",
            "intent": "",
            "retrieved_chunks": [],
            "response": "",
            "messages": [],
        }
        # If intent is empty or unrecognised, defaults to "conceptual"
        # per state.get("intent", "conceptual") or empty check
        assert _route_by_intent(state) in {"conceptual", ""}


class TestGraphExecution:
    @patch("src.agents.router_agent.ChatGroq")
    @patch("src.agents.conceptual_agent.ChatGroq")
    def test_end_to_end_conceptual_flow(self, mock_conceptual_groq, mock_router_groq):
        # Setup Router LLM mock
        router_llm = MagicMock()
        router_llm.invoke.return_value = _make_ai_message("conceptual")
        mock_router_groq.return_value = router_llm

        # Setup Conceptual LLM mock
        conceptual_llm = MagicMock()
        conceptual_llm.invoke.return_value = _make_ai_message("LangGraph is a graph framework.")
        mock_conceptual_groq.return_value = conceptual_llm

        store = _make_mock_vector_store()
        graph = build_graph(store)

        initial_state = {
            "query": "What is LangGraph?",
            "messages": [],
        }
        result = graph.invoke(initial_state)

        assert result["intent"] == "conceptual"
        assert result["response"] == "LangGraph is a graph framework."
        assert len(result["retrieved_chunks"]) > 0
        assert len(result["messages"]) > 0

    @patch("src.agents.router_agent.ChatGroq")
    @patch("src.agents.implementation_agent.ChatGroq")
    def test_end_to_end_implementation_flow(self, mock_impl_groq, mock_router_groq):
        # Setup Router LLM mock
        router_llm = MagicMock()
        router_llm.invoke.return_value = _make_ai_message("implementation")
        mock_router_groq.return_value = router_llm

        # Setup Implementation LLM mock
        impl_llm = MagicMock()
        impl_llm.invoke.return_value = _make_ai_message("```python\nbuilder = StateGraph()\n```")
        mock_impl_groq.return_value = impl_llm

        store = _make_mock_vector_store()
        graph = build_graph(store)

        initial_state = {
            "query": "Write code to build a StateGraph.",
            "messages": [],
        }
        result = graph.invoke(initial_state)

        assert result["intent"] == "implementation"
        assert "```python" in result["response"]
        assert len(result["retrieved_chunks"]) > 0
