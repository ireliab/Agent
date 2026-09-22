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

See [ROADMAP.md](ROADMAP.md) for what the agent can and cannot do yet, and for
the list of wrong assumptions that cost debugging time.

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

Each conversation has a `thread_id`. The agent's checkpointer keys history off
it, so follow-up questions keep context.

History is stored in **`conversations.sqlite`** via `AsyncSqliteSaver`, so
conversations survive a server restart. The sidebar lists them newest first;
clicking one reopens its transcript, its plan, and any approval it is still
waiting on. Deleting one removes its checkpoints, its index row and its uploads.

The checkpointer owns a database connection, so the agent is built in a FastAPI
lifespan rather than at import time — `get_agent()` returns it, and returns 503
until startup finishes.

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
| `glob`, `grep`, `read_file` (built in) | Search agent state only — **never** the user's uploads | n/a |
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

Every pending call gets its own card, with its arguments in an editable box and
a free-text field. All four of langchain's decisions are available:

| Action | What it sends | Effect |
| --- | --- | --- |
| **Approve** | `approve` | Runs the tool exactly as the agent asked |
| **Run with edits** | `edit` | Runs the tool with *your* arguments. The button relabels itself as soon as you change the JSON |
| **Reject** | `reject` (+ your reason) | Tool does not run; the model is told why, so it can change course |
| **Answer instead** | `respond` | Tool does not run; your text is handed back as if it were the tool's result |

The arguments box is validated before anything is sent: malformed JSON, or JSON
that is not an object, blocks submission and explains itself. "Answer instead"
requires text, since an empty reply would tell the model nothing.

When several tools are pending, each is decided separately and **nothing is
submitted until every one has a decision** — langchain requires exactly one
decision per request, in order. So you can edit one call and reject another in
the same turn.

Set `INTERRUPT_TOOLS=` (empty) to turn the approval step off.

A pending approval survives a page reload and a server restart: reopen the
conversation from the sidebar and the card comes back. That was not true before
durable checkpointing — a restart used to lose the pause silently, streaming a
normal-looking reply while the tool never ran.

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
.xlsx, .csv, .txt and .md** and truncates at 20k characters.

**Uploads are scoped to one conversation.** They live in
`uploads/<thread_id>/`, and `POST /api/upload` requires the `thread_id`. A flat
shared directory meant a brand new chat could see — and answer from — a document
uploaded in a previous one. Starting a New chat starts an empty upload folder.

**Attachments are numbered, and that matters.** A small model cannot reliably
retype a non-ASCII filename: given `RWA_OFC基金代幣化_Brief_2頁_v1.1.pdf` a 9B
kept emitting `Brief_2 頁` with a space, because that is how it re-tokenises the
CJK run. With exact-match lookup it retried the same broken name until the run
died. So:

- the prompt lists attachments as `1. <name>` and tells the agent to call
  `read_document("1")`. The listing comes from `upload_listing()`, the *same*
  function `read_document` resolves numbers against — when the two were
  numbered independently, the prompt's "1" and the tool's "1" were different
  files
- `read_document` also matches leniently — whitespace, zero-width characters,
  Unicode form and case are all ignored, and a unique partial match works
- the not-found message repeats the numbers and says not to retry the name

**Images go to the model directly.** `Qwen3.5-9B` is multimodal
(`Qwen3_5ForConditionalGeneration`, with a vision tower), so an attached image
is sent as an image content block in the user message rather than through a
tool — langchain converts it to the `image_url` data URL that vLLM expects.

Images are sent **exactly as uploaded** — no resizing, no re-encoding, original
format preserved — so the model sees full detail.

Be aware of what that costs: an attached image stays in the conversation history
and is re-sent on *every* subsequent model call in that thread, so a large photo
pays its token price repeatedly. Uploads are capped at 20 MB
(`MAX_UPLOAD_BYTES`); a big image may be rejected by the model server for
exceeding its context or request-body limits, which surfaces as an error in the
chat rather than a silent truncation.

`read_document` on an image tells the model the picture is already attached
visually rather than trying to decode its bytes as text.

Point this at a text-only model and the image blocks will be rejected — that is
the one place this playground assumes a multimodal endpoint.

The lenient match stops short of guessing: a request for a file type that was
never uploaded fails loudly rather than silently returning the one file that
*was*, which would have the agent confidently describing the wrong document.

## Breaking retry loops

`RepeatedCallGuard` refuses a tool call whose name and arguments exactly match
one already made `MAX_IDENTICAL_CALLS` times (default 2) in the same
conversation, returning an error that tells the model to change something. It
counts from the message history rather than instance state, so it is per-thread
and resets with a new chat.

This exists because a failed tool call is, for a small model, an invitation to
try the identical call again — which cannot ever succeed and burns the entire
run budget.

Note the guard does not distinguish a failed repeat from a successful one, so a
genuinely repeated successful call is blocked on the third try too. Raise
`MAX_IDENTICAL_CALLS` if that gets in the way.

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

Tracked in [ROADMAP.md](ROADMAP.md). The cheapest useful next steps are PDF
output alongside `write_docx`, and swapping `InMemorySaver` for a SQLite
checkpointer so conversations survive a restart.
