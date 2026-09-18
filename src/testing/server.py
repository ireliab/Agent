"""A tiny web server that puts a chat UI in front of the agent.

    uv run uvicorn testing.server:app --reload

Then open http://127.0.0.1:8000
"""

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from deepagents.backends.utils import file_data_to_string
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel

from testing.agent import build_agent
from testing.tools import OUTPUT_DIR

STATIC_DIR = Path(__file__).parent / "static"
MAX_TOOL_RESULT_CHARS = 2000

app = FastAPI(title="Agent chat")
agent = build_agent()


class ChatRequest(BaseModel):
    message: str
    thread_id: str


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


def _tool_events(update: Any) -> list[dict[str, Any]]:
    """Turn one node's state update into tool-call / tool-result events."""
    events: list[dict[str, Any]] = []
    if not isinstance(update, dict):
        return events
    for message in update.get("messages") or []:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                events.append(
                    {"type": "tool_call", "name": call.get("name"), "args": call.get("args")}
                )
        elif isinstance(message, ToolMessage):
            result = str(message.content)
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            events.append({"type": "tool_result", "name": message.name, "content": result})
    return events


async def _run(message: str, thread_id: str) -> AsyncIterator[str]:
    """Stream the agent's work as SSE events."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    try:
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg, _metadata = chunk
                if isinstance(msg, AIMessageChunk):
                    text = _text_of(msg)
                    if text:
                        yield _sse({"type": "token", "text": text})
            elif mode == "updates":
                for _node, update in (chunk or {}).items():
                    for event in _tool_events(update):
                        yield _sse(event)
    except Exception as exc:
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    yield _sse({"type": "done"})


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _run(request.message, request.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
