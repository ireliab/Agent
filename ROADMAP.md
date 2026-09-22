# Capability roadmap

What this agent can do, what it can't yet, and what was deliberately left out.
Working notes live here so the README can stay a reference.

**Status:** ✅ done · 🟡 partial · ⬜ not started

Update this file in the same commit as the change it describes.

---

## Planning & control

| # | Capability | Status | Notes |
| --- | --- | --- | --- |
| 1 | Todo / plan tool | ✅ | `TodoListMiddleware`; the plan renders live in the chat |
| 2 | Subagents | ✅ | `researcher` and `report-writer`, via the built-in `task` tool |
| 3 | Human-in-the-loop | ✅ | All four decisions (approve / edit / reject / respond), per-call when several are pending, reason and answer fields, survives reload and restart |
| 4 | Run limits | ✅ | Model- and tool-call caps, plus a repeated-call guard |
| 5 | Skills | ⬜ | Markdown playbooks loaded on demand |

## Memory & context

| # | Capability | Status | Notes |
| --- | --- | --- | --- |
| 6 | Durable history (SQLite checkpointer) | ✅ | `AsyncSqliteSaver` in `conversations.sqlite`; agent built in a FastAPI lifespan |
| 7 | Cross-thread memory | ⬜ | Needs `MemoryMiddleware` plus a store |
| 8 | Context editing | ✅ | Old tool results become `[cleared]` past `CONTEXT_EDIT_TRIGGER` |

## Documents & output

| # | Capability | Status | Notes |
| --- | --- | --- | --- |
| 9 | PDF writing | ⬜ | Next obvious step after `write_docx` |
| 10 | Excel / CSV output | ⬜ | `openpyxl` is already a dependency (used for reading) |
| 11 | PowerPoint | ⬜ | |
| 12 | Charts in documents | ⬜ | matplotlib → embedded in docx/pdf |
| 13 | Read uploaded files | ✅ | `.pdf .docx .xlsx .csv .txt .md`; images go to the model as vision input |
| 14 | Structured output | ⬜ | `response_format` on `create_deep_agent` |

## Knowledge & data

| # | Capability | Status | Notes |
| --- | --- | --- | --- |
| 15 | Full web page fetch | ⬜ | Tavily returns snippets only |
| 16 | RAG over own documents | ⬜ | Large — becomes a data pipeline, not just an agent change |
| 17 | SQL query tool | ⬜ | |
| 18 | Working code sandbox | ⬜ | The dead `execute` tool was **removed**; making it real needs a `SandboxBackendProtocol` backend |

## Playground & ops

| # | Capability | Status | Notes |
| --- | --- | --- | --- |
| 19 | Stop button | ✅ | Genuinely cancels the model, not just the UI |
| 20 | File upload | ✅ | Scoped per conversation |
| 21 | Token usage display | ✅ | Per turn, summed across model calls |
| 22 | Conversation history list | ✅ | Sidebar lists conversations; reopening restores transcript, plan and pending approval |
| 23 | Eval harness | ⬜ | The way to tell whether a prompt change helped, or you got lucky once |
| 24 | Model fallback / retry | ⬜ | `ModelFallbackMiddleware` / `ModelRetryMiddleware` |

---

## What's actually wired up today

**Agent tools:** `internet_search`, `read_document`, `list_uploads`,
`write_docx`, `write_text_file`, `write_todos`, `task`, plus the deepagents
filesystem tools `ls` / `read_file` / `write_file` / `edit_file` / `glob` /
`grep`.

**Middleware, in order:** `TodoListMiddleware` → `RepeatedCallGuard` →
`FilesystemMiddleware` → `ContextEditingMiddleware` → `ModelCallLimitMiddleware`
→ `ToolCallLimitMiddleware`.

**API:** `/api/chat`, `/api/resume`, `/api/cancel`, `/api/upload`, `/api/threads`,
`/api/threads/{id}` (GET and DELETE), `/api/files`, `/api/files/output`,
`/api/files/workspace`, `/api/health`.

**Knobs** (`.env`): `MAX_MODEL_CALLS`, `MAX_TOOL_CALLS`, `CONTEXT_EDIT_TRIGGER`,
`RECURSION_LIMIT`, `INTERRUPT_TOOLS`, `STREAM_USAGE`, `MAX_IDENTICAL_CALLS`.

---

## Fixes worth remembering

These were not planned features. They are recorded because each one was a wrong
assumption that cost real debugging time, and the same assumption is easy to
make again.

| Problem | What was actually true |
| --- | --- |
| Agent claimed it wrote a file that did not exist | `write_file` writes to a **virtual** filesystem in LangGraph state, never to disk |
| `execute` tool always errored | It only works on a `SandboxBackendProtocol` backend; `StateBackend` is not one |
| Stop button stopped the UI but not the model | Neither aborting the request nor closing the async generator cancels the run — the server must cancel the agent task |
| `GraphRecursionError` before the call limit fired | `recursion_limit` counts graph super-steps (~6 per model call), not model calls |
| Run ended silently at the model-call limit | The limit message is *injected*, not streamed, so it never reached the browser |
| Token counts never appeared | langchain disables `stream_usage` whenever a custom `base_url` is set |
| Agent looped forever on a CJK filename | A small model cannot retype non-ASCII names; it re-tokenises `Brief_2頁` as `Brief_2 頁` |
| New chat answered from an old chat's PDF | Uploads were a flat shared directory, and the prompt's numbering did not match the tool's |
| Agent said it could not see images | The model *is* multimodal (`Qwen3_5ForConditionalGeneration`); the pipeline just never sent them |
| UI edits silently did nothing | Static files were served without `Cache-Control`, and `--reload` was watching `.venv` |

---

## Open questions

- **Nothing here is tuned for this model.** Most verification ran against a stub
  model server, because the vLLM endpoint is not reachable from the dev
  environment. Wiring is proven; behaviour on a 9B is not.
- **Subagents on a small model.** Delegation adds a round trip and a nested
  agent loop. On a model that already struggles to terminate, this may cost more
  than it saves.
- **Images are sent at full resolution** and stay in history, so they are
  re-sent on every model call in that thread.
- **The repeated-call guard cannot tell a failed repeat from a successful one.**
  A legitimately repeated call is blocked on the third try. Raise
  `MAX_IDENTICAL_CALLS` if that bites.
