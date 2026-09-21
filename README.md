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
| `src/testing/tools.py` | Tools for real files: write docx/text, read uploaded pdf/docx/xlsx/csv |
| `src/testing/static/index.html` | The chat UI (no build step, no dependencies) |
| `src/testing/test.ipynb` | Scratch notebook |

The browser sends `{message, thread_id}` to `POST /api/chat` and reads back a
stream of events:

| Event | Meaning |
| --- | --- |
| `token` | A piece of the assistant's reply |
| `todos` | The agent's plan, whenever it changes |
| `tool_call` | The agent decided to call a tool, with its arguments |
| `tool_result` | What the tool returned |
| `notice` | A message from the agent's machinery, not the model (e.g. a limit was hit) |
| `interrupt` | The agent is paused waiting for you to approve a tool |
| `usage` | Token counts for the run |
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

## Guardrails and planning

`agent.py` adds four things on top of the deepagents default stack, all in
`build_middleware()`:

| Middleware | Why |
| --- | --- |
| `TodoListMiddleware` | Gives the agent a `write_todos` tool, so it plans multi-step work before acting. The plan renders live in the chat. |
| `ModelCallLimitMiddleware` | Hard ceiling of `MAX_MODEL_CALLS` per run. A small local model loops more readily than a frontier one. |
| `ToolCallLimitMiddleware` | Caps tool calls at `MAX_TOOL_CALLS`, but with `exit_behavior="continue"` so the model can still write a final answer. |
| `ContextEditingMiddleware` | Past `CONTEXT_EDIT_TRIGGER` tokens, old tool results are replaced with `[cleared]`, keeping the 3 most recent. |

Passing a middleware whose `.name` matches one already in the stack *replaces*
it. That's how `FilesystemMiddleware(tools=FILESYSTEM_TOOLS)` drops the
`execute` tool: `execute` only works on a backend implementing
`SandboxBackendProtocol`, and `StateBackend` is not one, so it could never do
anything but return an error.

Tune the limits in `.env`. `CONTEXT_EDIT_TRIGGER` should sit well below the
context length your model server is configured for.

### Why `RECURSION_LIMIT` is so much larger than `MAX_MODEL_CALLS`

LangGraph's `recursion_limit` counts graph super-steps, not model calls, and
this middleware stack spends roughly **six super-steps per model call**
(measured: 8 model calls exhausted a limit of 50). If the two are set close
together, the recursion limit trips first and raises `GraphRecursionError`
instead of letting `ModelCallLimitMiddleware` stop the run cleanly with an
explanation.

So `RECURSION_LIMIT` defaults to `MAX_MODEL_CALLS * 8 + 20`. **If you raise
`MAX_MODEL_CALLS`, raise this with it** — or just delete it from `.env` and let
it derive itself.

When a run does stop on the model-call limit, the agent posts a `notice` in the
chat saying so. If you see it, scroll up: repeated identical tool calls are the
usual cause, and they are visible as a run of identical tool chips.

## Approving tools before they run

Tools listed in `INTERRUPT_TOOLS` (by default `write_docx` and
`write_text_file`) pause the run and ask first. The chat shows the tool name and
its exact arguments with Approve / Reject buttons; the answer goes to
`POST /api/resume`, which continues the *same* assistant turn.

Rejecting does not just skip the tool — the model is told it was rejected, with
your reason, so it can choose something else.

Set `INTERRUPT_TOOLS=` (empty) to turn the approval step off. Only `approve` and
`reject` are wired up; langchain also supports `edit` (change the arguments
before running) and `respond` (answer on the tool's behalf), which would need a
form in the approval card.

A pausing run re-emits its tool call when it resumes, so the browser de-dupes
tool chips by tool-call id.

## Delegating to subagents

`build_subagents()` defines two, reachable through the built-in `task` tool:

| Subagent | For |
| --- | --- |
| `researcher` | Multi-search web research, returns a summary with sources |
| `report-writer` | Turning notes into a saved .docx or .md |

Each runs with its own context, so a long research detour does not fill the main
conversation. Delegation costs a round trip, so the system prompt tells the
agent to do single-step work itself.

## Attachments

The paperclip uploads files to `uploads/` via `POST /api/upload` (20 MB cap),
and the agent reads them with `read_document`, which handles **.pdf, .docx,
.xlsx, .csv, .txt and .md** and truncates at 20k characters. The message sent to
the model is prefixed with the attached filenames so it knows to go looking.

## Token usage

Each turn shows `N in / N out tokens`, summed across every model call in the
turn — including the part after an approval.

One gotcha: langchain **disables** streaming token counts whenever a custom
`base_url` is set, because many OpenAI-compatible servers do not support it. So
`build_model()` passes `stream_usage=STREAM_USAGE` explicitly. If your server
ignores `stream_options.include_usage`, no usage events arrive and the line
simply does not appear.

## Stopping a run

The Send button becomes a red **Stop** while a run is in flight.

Stopping is more involved than it looks. Aborting the browser's request is not
enough, and neither is closing the agent's async generator server-side —
LangGraph holds the model call in an inner task that keeps running, so the GPU
carries on generating a reply nobody will read. So the browser calls
`POST /api/cancel`, and the server runs the agent in its own task and cancels
it, which propagates down into the HTTP request to the model.

One consequence: a stopped run is not written to the checkpointer, so the
partial reply is not part of the conversation history.

## Things to try next

- Add a tool to `tools=[...]` in `agent.py` and watch it appear in the UI.
- PDF output, alongside `write_docx`.
- Wire up the `edit` decision so you can fix a tool's arguments before approving.
- Swap `InMemorySaver` for a SQLite checkpointer so history survives restarts.
- Give the agent subagents via `create_deep_agent(subagents=[...])`.
