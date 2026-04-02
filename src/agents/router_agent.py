"""Router Agent — classifies an incoming query as 'conceptual' or 'implementation'.

The router uses a zero-shot classification prompt sent to the Groq LLM.
No retrieval is performed at this stage; the sole output is a routing intent
written back into :class:`~src.pipeline.state.AgentState`.

Intent labels
-------------
``"conceptual"``
    The user wants a conceptual explanation, architectural overview, or
    theoretical description (e.g. "What is X?", "Explain Y", "How does Z
    work?").

``"implementation"``
    The user wants runnable code, a concrete example, or step-by-step
    implementation guidance (e.g. "Write code for X", "Show me how to
    implement Y", "Give me an example of Z").
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.pipeline.state import AgentState

_SYSTEM_PROMPT = """\
You are a query classifier for a technical documentation assistant.
Your task is to classify the user's query into exactly one of two categories:

  conceptual      – The user wants an explanation, overview, or theory.
  implementation  – The user wants code, examples, or step-by-step guidance.

Respond with a single word: either "conceptual" or "implementation".
Do not include any other text.
"""


def _build_llm() -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def route(state: AgentState) -> AgentState:
    """LangGraph node: classify *state['query']* and set *state['intent']*.

    Parameters
    ----------
    state:
        Current pipeline state.  Only ``state['query']`` is read.

    Returns
    -------
    AgentState
        Updated state with ``intent`` set to ``"conceptual"`` or
        ``"implementation"``.
    """
    llm = _build_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["query"]),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip().lower()

    # Normalise the response — default to "conceptual" if unparseable.
    if re.search(r"\bimplementation\b", raw):
        intent = "implementation"
    else:
        intent = "conceptual"

    return {**state, "intent": intent, "messages": messages + [response]}
