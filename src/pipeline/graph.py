"""LangGraph state machine for the multi-agent RAG pipeline.

Graph topology
--------------

::

    [START]
       │
       ▼
    router          ← Router Agent: classifies query → sets intent
       │
       ├─ "conceptual"     ──► conceptual_retriever  ──► [END]
       │
       └─ "implementation" ──► implementation_retriever ──► [END]

The two leaf nodes perform retrieval + answer generation in a single step
(retrieval is embedded inside each agent's ``answer()`` function).

Usage
-----
Build the compiled graph once at application start-up and reuse it::

    from src.pipeline.graph import build_graph
    from src.retrieval.embedder import Embedder
    from src.retrieval.vectorstore import VectorStore

    embedder = Embedder()
    store = VectorStore(embedder)
    store.load()

    graph = build_graph(store)
    result = graph.invoke({"query": "How does LangGraph manage state?"})
    print(result["response"])
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.agents import conceptual_agent, implementation_agent, router_agent
from src.pipeline.state import AgentState
from src.retrieval.vectorstore import VectorStore


def _route_by_intent(state: AgentState) -> str:
    """Conditional edge: return the name of the next node based on intent."""
    return state.get("intent", "conceptual")


def build_graph(vector_store: VectorStore) -> "CompiledGraph":  # noqa: F821
    """Construct and compile the LangGraph pipeline.

    Parameters
    ----------
    vector_store:
        A pre-loaded :class:`~src.retrieval.vectorstore.VectorStore` instance.
        It is injected into the agent nodes via :func:`functools.partial` so
        that the graph nodes remain pure functions of :class:`AgentState`.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready for ``.invoke()`` or ``.stream()``.
    """
    builder = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    builder.add_node("router", router_agent.route)
    builder.add_node(
        "conceptual_retriever",
        partial(conceptual_agent.answer, vector_store=vector_store),
    )
    builder.add_node(
        "implementation_retriever",
        partial(implementation_agent.answer, vector_store=vector_store),
    )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "conceptual": "conceptual_retriever",
            "implementation": "implementation_retriever",
        },
    )
    builder.add_edge("conceptual_retriever", END)
    builder.add_edge("implementation_retriever", END)

    return builder.compile()
