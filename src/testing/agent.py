"""The agent itself.

Kept separate from the web server so the notebook and the server can share
exactly the same agent definition.
"""

import os
from typing import Literal

import json

from deepagents import FilesystemMiddleware, create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

from testing.tools import list_uploads, read_document, write_docx, write_text_file

load_dotenv()

# Guardrails. A small local model loops more readily than a frontier one, so
# these are deliberately tight; raise them once you trust a given task.
MAX_MODEL_CALLS = int(os.environ.get("MAX_MODEL_CALLS", "25"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "40"))

# Once the conversation passes this many tokens, old tool results are replaced
# with a placeholder. Set it well below your server's context length.
CONTEXT_EDIT_TRIGGER = int(os.environ.get("CONTEXT_EDIT_TRIGGER", "16000"))

# LangGraph counts super-steps, not model calls, and the deepagents middleware
# stack costs roughly six super-steps per model call - so a recursion limit set
# near MAX_MODEL_CALLS fires long before the model-call limit does, raising
# GraphRecursionError instead of stopping cleanly with an explanation. Keep this
# comfortably above MAX_MODEL_CALLS so the middleware is what stops a run.
RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", str(MAX_MODEL_CALLS * 8 + 20)))

# `execute` ships with the filesystem middleware but only works on a backend
# implementing SandboxBackendProtocol. StateBackend is not one, so the tool
# would always fail - leave it out rather than let the model waste turns on it.
# `delete` is left out: the agent reached for it on a hallucinated path while
# flailing at an unrelated problem, and nothing here needs it.
FILESYSTEM_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

# How many times the same tool may be called with identical arguments before
# the call is refused.
MAX_IDENTICAL_CALLS = int(os.environ.get("MAX_IDENTICAL_CALLS", "2"))

# Tools that need the user to approve them before they run. These are the ones
# with an effect outside the conversation. Set INTERRUPT_TOOLS="" to turn the
# approval step off entirely.
INTERRUPT_TOOLS = [
    name.strip()
    for name in os.environ.get("INTERRUPT_TOOLS", "write_docx,write_text_file").split(",")
    if name.strip()
]

# langchain leaves streaming token counts OFF when a custom base_url is set,
# because many OpenAI-compatible servers do not support it. vLLM does, so ask
# for it - and if a server ignores it, the usage events simply never arrive.
STREAM_USAGE = os.environ.get("STREAM_USAGE", "true").lower() not in {"false", "0", "no"}

SYSTEM_PROMPT = """
You are a helpful assistant.

Use the internet_search tool when the question needs fresh or factual
information. Answer clearly and concisely.

Saving files:
- `write_docx` saves a real Word (.docx) file the user can open. Use it whenever
  the user asks for a Word document or a .docx report.
- `write_text_file` saves a real .txt or .md file.
- Your other file tools (`write_file`, `edit_file`, ...) write to a scratch
  workspace that the user CANNOT see or open. Never tell the user a file is
  ready unless you saved it with `write_docx` or `write_text_file`.

Uploaded files:
- When the user attaches a file, read it with `read_document` before answering
  questions about it. Pass the attachment's NUMBER, e.g. read_document("1") -
  never retype a long or non-English filename, you will get it wrong.
- `glob`, `grep` and `read_file` search a scratch workspace, NOT the user's
  uploads. They will never find an attached file. Use `read_document`.
- If a tool call fails, do not repeat it unchanged. Change the arguments or
  say what is blocking you.

Delegating:
- For anything needing several web searches, delegate to the `researcher`
  subagent with the `task` tool rather than searching repeatedly yourself.
- Do the work yourself when it is a single step; delegation costs a round trip.
"""

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set in .env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    try:
        return _get_tavily().search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as exc:  # surface the failure to the model instead of crashing the run
        return {"error": f"search failed: {type(exc).__name__}: {exc}"}


def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.environ.get("MODEL_API_KEY", "not-needed"),
        base_url=os.environ.get("MODEL_BASE_URL") or None,
        stream_usage=STREAM_USAGE,
    )


def build_subagents() -> list[dict]:
    """Focused agents the main agent can delegate to via the `task` tool.

    Each runs with its own context, so a long research detour does not fill up
    the main conversation.
    """
    return [
        {
            "name": "researcher",
            "description": (
                "Researches a topic on the web and returns a factual summary with "
                "source URLs. Use for anything needing current information or "
                "more than one search."
            ),
            "system_prompt": (
                "You research topics using internet_search. Search more than once "
                "if the first results are thin. Return a concise factual summary "
                "with source URLs. Do not speculate beyond what the sources say."
            ),
            "tools": [internet_search],
        },
        {
            "name": "report-writer",
            "description": (
                "Turns notes or research into a finished document saved as .docx "
                "or .md. Give it the full content to write up."
            ),
            "system_prompt": (
                "You turn the notes you are given into a well-structured document "
                "with headings, and save it with write_docx (or write_text_file "
                "for .md/.txt). Do not invent facts that are not in the notes."
            ),
            "tools": [write_docx, write_text_file],
        },
    ]


class RepeatedCallGuard(AgentMiddleware):
    """Refuse a tool call the agent has already made with identical arguments.

    Small models answer a failed tool call by retrying it verbatim, which burns
    the whole run budget without ever changing the input. Blocking the repeat
    and saying so explicitly is what breaks the loop.

    The count comes from the conversation's own history rather than from
    instance state, so it is naturally per-thread and resets with a new chat.
    """

    def __init__(self, limit: int = MAX_IDENTICAL_CALLS) -> None:
        super().__init__()
        self.limit = limit

    @staticmethod
    def _key(call: dict) -> str:
        return json.dumps(
            {"name": call.get("name"), "args": call.get("args")}, sort_keys=True, default=str
        )

    def _blocked(self, request) -> ToolMessage | None:
        key = self._key(request.tool_call)
        seen = 0
        for message in (request.state or {}).get("messages") or []:
            if isinstance(message, AIMessage):
                seen += sum(1 for call in message.tool_calls or [] if self._key(call) == key)
        if seen <= self.limit:
            return None
        name = request.tool_call.get("name")
        return ToolMessage(
            content=(
                f"Blocked: `{name}` has already been called with these exact "
                f"arguments {seen - 1} times and did not work. Repeating it will "
                f"not help. Change the arguments, use a different tool, or tell "
                f"the user what is blocking you."
            ),
            tool_call_id=request.tool_call.get("id", ""),
            name=name,
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        blocked = self._blocked(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        # The server drives the agent asynchronously, so this one is the path
        # that actually runs; the sync version is for notebook use.
        blocked = self._blocked(request)
        if blocked is not None:
            return blocked
        return await handler(request)


def build_middleware() -> list:
    """Middleware added on top of the deepagents default stack.

    Passing a middleware whose `.name` matches one already in the stack replaces
    it, which is how `FilesystemMiddleware` below drops the `execute` tool.
    """
    return [
        # Gives the agent a `write_todos` tool so it plans before acting.
        TodoListMiddleware(),
        RepeatedCallGuard(),
        FilesystemMiddleware(tools=FILESYSTEM_TOOLS),
        # Drop stale tool results once the context gets long, keeping the most
        # recent few so the model still sees what it just did.
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=CONTEXT_EDIT_TRIGGER, keep=3, clear_at_least=1000)]
        ),
        # Hard ceiling on a single run.
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        # Blocks further tool calls but lets the model still write an answer.
        ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="continue"),
    ]


def build_agent(checkpointer=None):
    """Build the agent.

    Args:
        checkpointer: where conversation history lives. Defaults to in-memory,
            which means history is lost when the process restarts.
    """
    return create_deep_agent(
        model=build_model(),
        tools=[internet_search, write_docx, write_text_file, read_document, list_uploads],
        system_prompt=SYSTEM_PROMPT,
        middleware=build_middleware(),
        subagents=build_subagents(),
        interrupt_on={
            # All four decisions langchain supports: run as-is, run with edited
            # arguments, refuse (optionally saying why), or answer on the tool's
            # behalf without running it.
            name: {"allowed_decisions": ["approve", "edit", "reject", "respond"]}
            for name in INTERRUPT_TOOLS
        }
        or None,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
