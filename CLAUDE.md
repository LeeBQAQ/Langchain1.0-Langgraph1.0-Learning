# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a structured learning repository for **LangChain 1.0** and **LangGraph 1.0**, organized in four progressive phases (22 modules + 3 projects). Each module is self-contained under its numbered directory with a `main.py` entry point and a `README.md` explaining the concepts.

## Environment & Setup

```bash
# Create and activate venv (Python 3.10+)
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

**Required**: `GROQ_API_KEY` in `.env` (free, no credit card). The default model provider is Groq + DeepSeek models, configured via `model_init.py` at the project root.

## Running Code

Each module is self-contained — cd into the directory and run its `main.py`:

```bash
python phase1_fundamentals/01_hello_langchain/main.py
```

No build step, no test suite. This is a learning repo where each module's output is printed to stdout.

## Key Import: `model_init.py`

Many modules (phase1 04+, phase2, phase3) import `model` from the root-level `model_init.py` instead of initializing their own model. This file:
- Loads `.env` and reads `GROQ_API_KEY` + `model` from env vars
- Creates a `ChatDeepSeekFixReasoningContent` instance (subclass of `ChatDeepSeek` that patches reasoning content handling)
- Raises if `GROQ_API_KEY` is missing

If you add a new module that needs a model, either import from `model_init` (consistent with other modules) or use `init_chat_model` directly for standalone examples.

## Architecture

### Phase structure (progressive learning)
- **phase1_fundamentals** (01-06): Core LangChain — model invocation, prompts, messages, `@tool`, `create_agent`, agent execution loop
- **phase2_practical** (07-15): Real-world features — memory/checkpointing, middleware, structured output (Pydantic), validation/retry, RAG (basic + advanced with BM25/hybrid search)
- **phase3_advanced** (16-23): LangGraph — StateGraph/nodes/edges, multi-agent supervision, conditional routing, multimodal (image/file/mixed), LangSmith integration, error handling
- **phase4_projects** (01-03): End-to-end projects — RAG system, multi-agent support system, research assistant

### Key LangChain 1.0 APIs (this repo's baseline)
- `langchain.chat_models.init_chat_model` — model initialization
- `langchain.agents.create_agent` — agent creation (**not** the deprecated `langgraph.prebuilt.create_react_agent`)
- `langchain_core.tools.tool` — `@tool` decorator for tool definitions
- `langgraph.graph.StateGraph` + `START`/`END` — custom graph workflows when `create_agent` isn't enough
- `langgraph.checkpoint.memory.MemorySaver` / `InMemorySaver` — conversation memory; SQLite checkpointer for persistence

### Self-documenting code
Each module's `main.py` contains numbered example functions (`example_1_*`, `example_2_*`, etc.) with verbose print output. The code is heavily commented in Chinese. Docstrings explain both "what" and "why" for learners.

## Useful Docs (in `docs/`)
- `FREE_API_GUIDE.md` — how to get free API keys (Groq, Google Gemini, etc.)
- `API_COMPARISON.md` — LangChain 0.x vs 1.0 API differences
- `SIMPLIFIED_LEARNING_PATH.md` — fast-track path through the modules
