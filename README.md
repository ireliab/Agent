# Agent playground

A minimal chat frontend for testing a `deepagents` agent in the browser.

## Setup

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | What it is |
| --- | --- |
| `MODEL_BASE_URL` | OpenAI-compatible endpoint (e.g. your vLLM server's `/v1`) |
| `MODEL_API_KEY` | Key for that endpoint |
| `MODEL_NAME` | Model name the endpoint serves |
| `TAVILY_API_KEY` | For the `internet_search` tool |
| `LANGSMITH_*` | Optional tracing; set `LANGSMITH_TRACING=false` to skip |

## Run

```bash
uv run uvicorn testing.server:app --reload --port 8000
```

Open http://127.0.0.1:8000.

## How it fits together

| File | Role |
| --- | --- |
| `src/testing/agent.py` | The agent: model, tools, system prompt, memory |
| `src/testing/server.py` | FastAPI server; streams the agent over SSE, serves files |
| `src/testing/tools.py` | Tools that write real files to `outputs/` (docx, text) |
| `src/testing/static/index.html` | The chat UI (no build step, no dependencies) |
| `src/testing/test.ipynb` | Scratch notebook |

The browser sends `{message, thread_id}` to `POST /api/chat` and reads back a
stream of events:

| Event | Meaning |
| --- | --- |
| `token` | A piece of the assistant's reply |
| `tool_call` | The agent decided to call a tool, with its arguments |
| `tool_result` | What the tool returned |
| `error` | Something failed; shown in red in the UI |
| `done` | The turn finished |

Each browser tab gets a random `thread_id`. The agent's checkpointer keys
conversation history off it, so follow-up questions keep context, and
**New chat** starts a fresh thread. History lives in memory, so restarting the
server clears it.

`GET /api/health` reports which model and endpoint the server is configured for
— the coloured dot in the UI header.

## Where the agent's files go

This trips everyone up once. `create_deep_agent` gives the agent built-in
`write_file` / `edit_file` tools, but by default they use `StateBackend` — a
**virtual filesystem living in LangGraph state**. When the agent says "I've
written the file", that file is real, but it only exists in the server's memory
for that thread. It is not on your disk and it disappears when the server
restarts.

So the agent has two places to put things:

| Tool | Goes to | User can open it? |
| --- | --- | --- |
| `write_file`, `edit_file` (built in) | Agent state, per thread | No — scratch only |
| `write_docx` (`tools.py`) | `outputs/` on disk | Yes |
| `write_text_file` (`tools.py`) | `outputs/` on disk | Yes |

The **Files panel** above the composer lists both, with download links, and
refreshes after every turn — so nothing the agent produces can go missing.

`agent.py`'s system prompt tells the model which tool is which, so it reaches
for `write_docx` when you ask for a Word document instead of claiming it can
only make `.txt`.

Note that `outputs/` is deliberately a scoped directory rather than full disk
access. deepagents ships a `FilesystemBackend` that gives the agent real
read/write across your filesystem, but its own docs warn against using it behind
a web server — it would let the agent read `.env` and your keys.

## Using the same agent from the notebook

```python
from testing.agent import build_agent

agent = build_agent()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is langgraph?"}]},
    config={"configurable": {"thread_id": "notebook"}},
)
for m in result["messages"]:
    m.pretty_print()
```

## Things to try next

- Add a tool to `tools=[...]` in `agent.py` and watch it appear in the UI.
- Swap `InMemorySaver` for a SQLite checkpointer so history survives restarts.
- Give the agent subagents via `create_deep_agent(subagents=[...])`.
