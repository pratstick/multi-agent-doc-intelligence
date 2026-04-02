# multi-agent-doc-intelligence

A multi-agent RAG (Retrieval-Augmented Generation) pipeline for parsing and
querying technical documentation.

## Architecture

```
[User Query]
     │
     ▼
┌─────────────────────┐
│   Router Agent      │  ← Groq LLM (zero-shot classification)
└────────┬────────────┘
         │
   ┌─────┴──────┐
   │            │
   ▼            ▼
[conceptual] [implementation]
   │            │
   ▼            ▼
┌──────────────────────────┐
│   FAISS Retrieval        │  ← filtered by chunk_type
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Conceptual / Impl Agent │  ← Groq LLM (answer generation)
└──────────┬───────────────┘
           │
           ▼
     [Final Response]
```

Three agents are orchestrated by [LangGraph](https://github.com/langchain-ai/langgraph):

| Agent | Role |
|---|---|
| **Router** | Zero-shot classifies query as `conceptual` or `implementation` |
| **Conceptual** | Synthesises theory/architecture answers from prose chunks |
| **Implementation** | Generates runnable code grounded in retrieved code + prose chunks |

## Project Structure

```
multi-agent-doc-intelligence/
├── src/
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── conceptual_agent.py
│   │   └── implementation_agent.py
│   ├── pipeline/
│   │   ├── graph.py          # LangGraph state machine
│   │   └── state.py          # AgentState TypedDict
│   ├── ingestion/
│   │   ├── parser.py         # Markdown AST parser
│   │   └── chunker.py        # Header/code-block-preserving chunker
│   ├── retrieval/
│   │   ├── embedder.py       # sentence-transformers wrapper
│   │   └── vectorstore.py    # FAISS build / persist / query
│   └── config.py
├── data/
│   ├── raw/                  # Place your .md documentation files here
│   └── index/                # Auto-generated FAISS index
├── tests/
│   ├── test_parser.py
│   ├── test_agents.py
│   └── test_retrieval.py
├── main.py
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/pratstick/multi-agent-doc-intelligence.git
cd multi-agent-doc-intelligence

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and set GROQ_API_KEY=<your key>
```

## Usage

### 1 — Ingest documentation

Place your Markdown (`.md`) files inside `data/raw/`, then run:

```bash
python main.py ingest
```

This parses, chunks, embeds, and persists a FAISS index to `data/index/`.
You only need to re-run this when the documentation changes.

### 2 — Query the pipeline

**Interactive mode:**

```bash
python main.py query
```

**Single question:**

```bash
python main.py query --question "What is LangGraph?"
```

## Configuration

All tuneable parameters live in `src/config.py` and can be overridden via
environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model ID |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks returned per retrieval |
| `CHUNK_MAX_CHARS` | `1500` | Soft max characters per prose chunk |
| `CHUNK_OVERLAP_CHARS` | `150` | Overlap prefix between adjacent prose chunks |

## Running Tests

```bash
pytest tests/
```

Tests for agents mock the Groq LLM and run without network access.
Tests for retrieval use an in-memory FAISS index and mock embeddings.

## Key Design Decisions

- **Structural Markdown parser** — Uses `markdown-it-py` AST tokens so headings and fenced code blocks are never split across chunks.
- **Metadata-filtered retrieval** — The router's intent (`conceptual` / `implementation`) controls whether prose or code chunks are fetched, reducing noise.
- **Local FAISS** — All vectors and metadata stay on disk; no cloud vector store required.
- **Groq for latency** — All LLM calls use Groq-hosted inference for minimal response time.
- **Stateless nodes, stateful graph** — Agent functions are pure transformations of `AgentState`; LangGraph owns all state transitions.