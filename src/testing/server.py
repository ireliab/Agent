"""A tiny web server that puts a chat UI in front of the agent.

    uv run uvicorn testing.server:app --reload

Then open http://127.0.0.1:8000
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from deepagents.backends.utils import file_data_to_string
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from testing.agent import (
    MAX_MODEL_CALLS,
    RECURSION_LIMIT,
    build_agent,
    build_user_content,
)
from testing.tools import output_root, set_upload_scope, upload_dir

STATIC_DIR = Path(__file__).parent / "static"
MAX_TOOL_RESULT_CHARS = 2000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DB_PATH = Path.cwd() / "conversations.sqlite"

# Filled in by the lifespan below. The agent cannot be built at import time any
# more: its checkpointer owns a database connection with a lifecycle.
_runtime: dict[str, Any] = {"agent": None, "index": None}


def get_agent():
    agent = _runtime["agent"]
    if agent is None:
        raise HTTPException(status_code=503, detail="agent still starting")
    return agent


async def _open_index() -> aiosqlite.Connection:
    """A small table listing conversations, for the history sidebar.

    The checkpointer stores state per thread but has no notion of a title or an
    ordering for humans, so keep our own index alongside it.
    """
    conn = await aiosqlite.connect(str(DB_PATH))
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS threads (
               thread_id  TEXT PRIMARY KEY,
               title      TEXT NOT NULL,
               created_at REAL NOT NULL,
               updated_at REAL NOT NULL
           )"""
    )
    await conn.commit()
    return conn


async def _touch_thread(thread_id: str, first_message: str) -> None:
    """Record the conversation, keeping the title from its opening message."""
    index: aiosqlite.Connection = _runtime["index"]
    if index is None:
        return
    now = time.time()
    title = " ".join(first_message.split())[:80] or "Untitled"
    await index.execute(
        """INSERT INTO threads (thread_id, title, created_at, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at""",
        (thread_id, title, now, now),
    )
    await index.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        await saver.setup()
        _runtime["index"] = await _open_index()
        _runtime["agent"] = build_agent(checkpointer=saver)
        try:
            yield
        finally:
            _runtime["agent"] = None
            index = _runtime.pop("index", None)
            if index is not None:
                await index.close()


app = FastAPI(title="Agent chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    attachments: list[str] = []


class ResumeRequest(BaseModel):
    thread_id: str
    decisions: list[dict[str, Any]]


class CancelRequest(BaseModel):
    thread_id: str


# Set when the browser asks to stop a run. Uvicorn does not reliably deliver
# `http.disconnect` while a streaming response is in flight, so a dropped
# connection alone is not enough to stop the agent - the browser tells us
# explicitly instead.
_cancellations: dict[str, asyncio.Event] = {}


def _cancel_event(thread_id: str) -> asyncio.Event:
    event = _cancellations.get(thread_id)
    if event is None:
        event = _cancellations[thread_id] = asyncio.Event()
    return event


def _sse(event: dict[str, Any]) -> str:
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _text_of(message: Any) -> str:
    """Pull plain text out of a message, whose content may be a string or blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _describe(exc: Exception) -> str:
    """Turn an exception into something a person can act on."""
    if isinstance(exc, GraphRecursionError):
        return (
            f"The agent ran out of steps without finishing (recursion limit "
            f"{RECURSION_LIMIT}). It is most likely stuck repeating itself - "
            f"check the tool calls above to see what it kept retrying. This "
            f"should normally be caught by the {MAX_MODEL_CALLS}-model-call "
            f"limit first; if you raised MAX_MODEL_CALLS, raise RECURSION_LIMIT "
            f"with it."
        )
    return f"{type(exc).__name__}: {exc}"


def _tool_events(update: Any, streamed: set[str]) -> list[dict[str, Any]]:
    """Turn one node's state update into tool-call / tool-result events.

    `streamed` holds the ids of AI messages whose text already reached the
    browser token by token. Anything else with text was injected by middleware
    rather than generated - the "model call limit reached" message, for one -
    and would otherwise never be shown at all.
    """
    events: list[dict[str, Any]] = []
    if not isinstance(update, dict):
        return events
    for message in update.get("messages") or []:
        if isinstance(message, AIMessage):
            text = _text_of(message)
            if text and message.id not in streamed:
                events.append({"type": "notice", "text": text})
            for call in message.tool_calls or []:
                events.append(
                    {
                        "type": "tool_call",
                        "name": call.get("name"),
                        "args": call.get("args"),
                        "id": call.get("id"),
                    }
                )
        elif isinstance(message, ToolMessage):
            result = str(message.content)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            events.append(
                {
                    "type": "tool_result",
                    "name": message.name,
                    "content": result,
                    "id": message.tool_call_id,
                }
            )
    return events


async def _stream(agent_input: Any, thread_id: str) -> AsyncIterator[str]:
    """Stream the agent's work as SSE events.

    Stopping has to genuinely cancel the work. Neither dropping the connection
    nor closing the agent's async generator is enough - LangGraph holds the
    model call in an inner task, which runs happily to completion while nobody
    is reading it. So the agent runs in its own task and Stop cancels that task,
    which propagates `CancelledError` down into the HTTP request to the model.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    # Point the upload tools at this conversation's files for the whole run.
    set_upload_scope(thread_id)
    cancelled = _cancel_event(thread_id)
    cancelled.clear()

    events: asyncio.Queue[Any] = asyncio.Queue()
    finished = object()
    streamed: set[str] = set()
    usage = {"input": 0, "output": 0}

    async def pump() -> None:
        try:
            async for item in get_agent().astream(
                agent_input,
                config=config,
                stream_mode=["messages", "updates"],
            ):
                await events.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await events.put(exc)
        finally:
            events.put_nowait(finished)

    worker = asyncio.create_task(pump())
    stop = asyncio.create_task(cancelled.wait())

    try:
        while True:
            nxt = asyncio.create_task(events.get())
            ready, _ = await asyncio.wait({nxt, stop}, return_when=asyncio.FIRST_COMPLETED)
            # Check the stop flag before the queue: while tokens are streaming
            # there is nearly always one waiting, so both tasks finish together
            # and testing `nxt` first would mean never noticing Stop at all.
            if cancelled.is_set():
                nxt.cancel()
                return
            if nxt not in ready:
                nxt.cancel()
                return
            item = nxt.result()
            if item is finished:
                break
            if isinstance(item, Exception):
                yield _sse({"type": "error", "message": _describe(item)})
                break

            mode, chunk = item
            if mode == "messages":
                msg, _metadata = chunk
                if isinstance(msg, AIMessageChunk):
                    if msg.usage_metadata:
                        usage["input"] += msg.usage_metadata.get("input_tokens", 0)
                        usage["output"] += msg.usage_metadata.get("output_tokens", 0)
                    text = _text_of(msg)
                    if text:
                        if msg.id:
                            streamed.add(msg.id)
                        yield _sse({"type": "token", "text": text})
            elif mode == "updates":
                for _node, update in (chunk or {}).items():
                    if _node == "__interrupt__":
                        # The agent is paused waiting for approval. The browser
                        # answers via /api/resume; nothing more streams here.
                        for interrupt in update or ():
                            value = getattr(interrupt, "value", {}) or {}
                            yield _sse(
                                {
                                    "type": "interrupt",
                                    "requests": value.get("action_requests", []),
                                }
                            )
                        continue
                    if isinstance(update, dict) and update.get("todos"):
                        yield _sse({"type": "todos", "items": update["todos"]})
                    for event in _tool_events(update, streamed):
                        yield _sse(event)
    finally:
        # Covers Stop, a closed tab (the yield above fails), and normal endings.
        worker.cancel()
        stop.cancel()
        _cancellations.pop(thread_id, None)

    if usage["input"] or usage["output"]:
        yield _sse({"type": "usage", **usage})
    yield _sse({"type": "done"})


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    await _touch_thread(payload.thread_id, payload.message)
    content = build_user_content(payload.message, payload.attachments, payload.thread_id)

    return StreamingResponse(
        _stream({"messages": [{"role": "user", "content": content}]}, payload.thread_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/resume")
async def resume(payload: ResumeRequest) -> StreamingResponse:
    """Continue a run that paused for approval, with the user's decisions."""
    return StreamingResponse(
        _stream(Command(resume={"decisions": payload.decisions}), payload.thread_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...), thread_id: str = Form(...)
) -> dict[str, Any]:
    """Store an uploaded file where this conversation's `read_document` finds it."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 20 MB)")
    name = Path(file.filename or "upload").name
    directory = upload_dir(thread_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)
    return {"name": name, "size": len(content)}


@app.post("/api/cancel")
async def cancel(payload: CancelRequest) -> dict[str, bool]:
    """Ask an in-flight run on this thread to stop."""
    _cancel_event(payload.thread_id).set()
    return {"ok": True}


def _replay_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Rebuild the transcript for a conversation being reopened.

    The live stream sends fine-grained events; reloading has only the stored
    messages, so regroup them into the user / assistant turns the UI renders.
    """
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append({"role": "user", "text": _text_of(message)})
            current = None
        elif isinstance(message, AIMessage):
            if current is None:
                current = {"role": "assistant", "text": "", "events": []}
                turns.append(current)
            current["text"] += _text_of(message)
            for call in message.tool_calls or []:
                current["events"].append(
                    {
                        "type": "tool_call",
                        "name": call.get("name"),
                        "args": call.get("args"),
                        "id": call.get("id"),
                    }
                )
        elif isinstance(message, ToolMessage):
            if current is None:
                current = {"role": "assistant", "text": "", "events": []}
                turns.append(current)
            result = str(message.content)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            current["events"].append(
                {
                    "type": "tool_result",
                    "name": message.name,
                    "content": result,
                    "id": message.tool_call_id,
                }
            )
    return turns


def _pending_approval(snapshot: Any) -> list[dict[str, Any]]:
    """Any tool calls this conversation is paused waiting on."""
    requests: list[dict[str, Any]] = []
    for interrupt in getattr(snapshot, "interrupts", ()) or ():
        value = getattr(interrupt, "value", {}) or {}
        requests.extend(value.get("action_requests", []))
    return requests


@app.get("/api/threads")
async def list_threads() -> dict[str, Any]:
    """Conversations for the history sidebar, most recently used first."""
    index: aiosqlite.Connection = _runtime.get("index")
    if index is None:
        return {"threads": []}
    async with index.execute(
        "SELECT thread_id, title, updated_at FROM threads ORDER BY updated_at DESC LIMIT 200"
    ) as cursor:
        rows = await cursor.fetchall()
    return {
        "threads": [
            {"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows
        ]
    }


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    """Everything needed to reopen a conversation, including a pending approval."""
    set_upload_scope(thread_id)
    snapshot = await get_agent().aget_state({"configurable": {"thread_id": thread_id}})
    messages = (snapshot.values or {}).get("messages") or []
    return {
        "thread_id": thread_id,
        "turns": _replay_messages(messages),
        "todos": (snapshot.values or {}).get("todos") or [],
        "pending": _pending_approval(snapshot),
    }


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str) -> dict[str, bool]:
    """Forget a conversation: its checkpoints, its index row and its uploads."""
    await get_agent().checkpointer.adelete_thread(thread_id)
    index: aiosqlite.Connection = _runtime.get("index")
    if index is not None:
        await index.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        await index.commit()
    directory = upload_dir(thread_id)
    if directory.exists():
        for path in directory.iterdir():
            path.unlink(missing_ok=True)
        directory.rmdir()
    return {"ok": True}


@app.get("/api/files")
async def list_files(thread_id: str = Query(...)) -> dict[str, Any]:
    """List what the agent produced: real files on disk, plus its scratch workspace."""
    outputs = []
    directory = output_root()
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if path.is_file():
                stat = path.stat()
                outputs.append({"name": path.name, "size": stat.st_size, "modified": stat.st_mtime})

    set_upload_scope(thread_id)
    workspace = []
    snapshot = await get_agent().aget_state({"configurable": {"thread_id": thread_id}})
    for path_key, file_data in (snapshot.values.get("files") or {}).items():
        workspace.append({"path": path_key, "size": len(file_data_to_string(file_data))})

    return {"outputs": outputs, "workspace": sorted(workspace, key=lambda f: f["path"])}


@app.get("/api/files/output")
async def download_output(name: str = Query(...)) -> FileResponse:
    """Download a real file from the outputs directory."""
    directory = output_root()
    path = (directory / Path(name).name).resolve()
    if not path.is_file() or directory.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/files/workspace")
async def download_workspace_file(
    thread_id: str = Query(...), path: str = Query(...)
) -> PlainTextResponse:
    """Download a file from the agent's in-state scratch workspace."""
    snapshot = await get_agent().aget_state({"configurable": {"thread_id": thread_id}})
    file_data = (snapshot.values.get("files") or {}).get(path)
    if file_data is None:
        raise HTTPException(status_code=404, detail="file not found")
    return PlainTextResponse(
        file_data_to_string(file_data),
        headers={"Content-Disposition": f'attachment; filename="{Path(path).name}"'},
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": os.environ.get("MODEL_NAME"),
        "base_url": os.environ.get("MODEL_BASE_URL"),
        "tavily": bool(os.environ.get("TAVILY_API_KEY")),
    }


class NoCacheStaticFiles(StaticFiles):
    """Serve the UI without caching.

    Browsers happily reuse a cached `index.html`, which means edits to the chat
    UI silently do not show up until a hard refresh. Not what you want while
    building it.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")
