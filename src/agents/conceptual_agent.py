"""Conceptual Agent — answers theory and architecture questions.

This node is invoked when the Router Agent classifies the user query as
``"conceptual"``.  It:

1. Queries the FAISS index for the top-k **prose** chunks most similar to
   the user's question.
2. Builds a grounded answer-generation prompt that includes the retrieved
   context and instructs the LLM to cite the document section it is drawing
   from.
3. Calls the Groq LLM and writes the response back into
   :class:`~src.pipeline.state.AgentState`.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL, RETRIEVAL_TOP_K
from src.pipeline.state import AgentState
from src.retrieval.vectorstore import VectorStore

_SYSTEM_PROMPT = """\
You are an expert technical documentation assistant specialising in \
architectural and conceptual explanations.

You will be given a user question and a set of relevant documentation \
excerpts.  Use ONLY the provided excerpts to answer the question.  \
For each key point you make, indicate which documentation section it \
comes from using the format: [Section: <header_path>].

If the excerpts do not contain enough information to answer the question, \
say so explicitly — do not speculate beyond the provided context.

Be concise, precise, and structured.  Use bullet points or numbered lists \
where they improve clarity.
"""


def _build_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def _format_chunks(chunks: list[dict]) -> str:
    """Render retrieved chunks into a readable context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = " > ".join(chunk.get("header_path", []))
        source = chunk.get("source_file", "unknown")
        text = chunk.get("text", "")
        parts.append(
            f"--- Excerpt {i} [Section: {header}] (source: {source}) ---\n{text}"
        )
    return "\n\n".join(parts)


def answer(state: AgentState, vector_store: VectorStore) -> AgentState:
    """LangGraph node: retrieve prose context and generate a conceptual answer.

    Parameters
    ----------
    state:
        Current pipeline state.  Reads ``state['query']``.
    vector_store:
        Pre-loaded :class:`~src.retrieval.vectorstore.VectorStore` instance.

    Returns
    -------
    AgentState
        Updated state with ``retrieved_chunks`` and ``response`` populated.
    """
    chunks = vector_store.query(
        state["query"], top_k=RETRIEVAL_TOP_K, chunk_type="prose"
    )

    context = _format_chunks(chunks)
    user_message_content = (
        f"Documentation context:\n\n{context}\n\n"
        f"Question: {state['query']}"
    )

    llm = _build_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message_content),
    ]
    ai_response = llm.invoke(messages)

    return {
        **state,
        "retrieved_chunks": chunks,
        "response": ai_response.content,
        "messages": state.get("messages", []) + messages + [ai_response],
    }
