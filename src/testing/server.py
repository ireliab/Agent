"""A tiny web server that puts a chat UI in front of the agent.

    uv run uvicorn testing.server:app --reload

Then open http://127.0.0.1:8000
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from deepagents.backends.utils import file_data_to_string
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from testing.agent import MAX_MODEL_CALLS, RECURSION_LIMIT, build_agent
from testing.tools import OUTPUT_DIR, UPLOAD_DIR

STATIC_DIR = Path(__file__).parent / "static"
MAX_TOOL_RESULT_CHARS = 2000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Agent chat")
agent = build_agent()


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
    cancelled = _cancel_event(thread_id)
    cancelled.clear()

    events: asyncio.Queue[Any] = asyncio.Queue()
    finished = object()
    streamed: set[str] = set()
    usage = {"input": 0, "output": 0}

    async def pump() -> None:
        try:
            async for item in agent.astream(
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
    message = payload.message
    if payload.attachments:
        attached = ", ".join(payload.attachments)
        prefix = f"[The user attached: {attached}. Use read_document to read them.]"
        message = f"{prefix}\n\n{message}"
    return StreamingResponse(
        _stream({"messages": [{"role": "user", "content": message}]}, payload.thread_id),
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
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Store an uploaded file where the `read_document` tool can find it."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 20 MB)")
    name = Path(file.filename or "upload").name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / name).write_bytes(content)
    return {"name": name, "size": len(content)}


@app.post("/api/cancel")
async def cancel(payload: CancelRequest) -> dict[str, bool]:
    """Ask an in-flight run on this thread to stop."""
    _cancel_event(payload.thread_id).set()
    return {"ok": True}


@app.get("/api/files")
async def list_files(thread_id: str = Query(...)) -> dict[str, Any]:
    """List what the agent produced: real files on disk, plus its scratch workspace."""
    outputs = []
    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.iterdir()):
            if path.is_file():
                stat = path.stat()
                outputs.append({"name": path.name, "size": stat.st_size, "modified": stat.st_mtime})

    workspace = []
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    for path_key, file_data in (snapshot.values.get("files") or {}).items():
        workspace.append({"path": path_key, "size": len(file_data_to_string(file_data))})

    return {"outputs": outputs, "workspace": sorted(workspace, key=lambda f: f["path"])}


@app.get("/api/files/output")
async def download_output(name: str = Query(...)) -> FileResponse:
    """Download a real file from the outputs directory."""
    path = (OUTPUT_DIR / Path(name).name).resolve()
    if not path.is_file() or OUTPUT_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/files/workspace")
async def download_workspace_file(
    thread_id: str = Query(...), path: str = Query(...)
) -> PlainTextResponse:
    """Download a file from the agent's in-state scratch workspace."""
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
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
