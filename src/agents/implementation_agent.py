"""Implementation Agent — generates code snippets from documentation context.

This node is invoked when the Router Agent classifies the user query as
``"implementation"``.  It:

1. Queries the FAISS index for the top-k **code** chunks most similar to the
   user's question, then fetches a smaller set of **prose** siblings to
   provide surrounding explanatory context.
2. Builds a code-generation prompt that grounds the LLM strictly in the
   retrieved examples and asks for runnable, language-consistent output.
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
You are an expert technical documentation assistant specialising in code \
generation and implementation guidance.

You will be given a user question and a set of relevant code examples and \
prose context retrieved from the official documentation.  Generate a \
complete, correct, and runnable code snippet that directly addresses the \
user's question.

Rules:
- Base your answer strictly on the provided documentation context.
- Wrap all code in a fenced code block with the appropriate language tag \
  (e.g. ```python … ```).
- After the code block, provide a brief prose explanation of what the code \
  does and any important caveats.
- If the documentation context does not contain enough information to \
  produce a correct implementation, say so explicitly and explain what is \
  missing.
"""


def _build_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def _format_chunks(chunks: list[dict], label: str) -> str:
    """Render a list of chunks into a labelled context block."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = " > ".join(chunk.get("header_path", []))
        source = chunk.get("source_file", "unknown")
        lang = chunk.get("language", "")
        text = chunk.get("text", "")
        lang_tag = f" [{lang}]" if lang else ""
        parts.append(
            f"--- {label} {i}{lang_tag} [Section: {header}] "
            f"(source: {source}) ---\n{text}"
        )
    return "\n\n".join(parts)


def answer(state: AgentState, vector_store: VectorStore) -> AgentState:
    """LangGraph node: retrieve code + prose context and generate a code answer.

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
    code_chunks = vector_store.query(
        state["query"], top_k=RETRIEVAL_TOP_K, chunk_type="code"
    )
    # Fetch a smaller number of prose chunks as supporting context.
    prose_chunks = vector_store.query(
        state["query"], top_k=max(2, RETRIEVAL_TOP_K // 2), chunk_type="prose"
    )

    all_chunks = code_chunks + prose_chunks

    context_parts: list[str] = []
    if code_chunks:
        context_parts.append(_format_chunks(code_chunks, "Code Example"))
    if prose_chunks:
        context_parts.append(_format_chunks(prose_chunks, "Context"))
    context = "\n\n".join(context_parts)

    user_message_content = (
        f"Documentation context:\n\n{context}\n\n"
        f"Request: {state['query']}"
    )

    llm = _build_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message_content),
    ]
    ai_response = llm.invoke(messages)

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "response": ai_response.content,
        "messages": state.get("messages", []) + messages + [ai_response],
    }
