"""A fake OpenAI-compatible model server, for testing the harness itself.

This is a test double, not a model. It exists so the eval harness can be
verified without a GPU: it plays the scripted part of a cooperative model,
emitting the same SSE shapes vLLM does - streamed text, streamed tool calls,
and a usage chunk when `stream_options.include_usage` is set.

It also logs what it was actually sent, which is how the image pipeline was
proved to be sending images at all.

    python -m evals.fakemodel --port 8099

Then point the agent at it:

    MODEL_BASE_URL=http://127.0.0.1:8099/v1 MODEL_NAME=fake python -m evals.run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from evals import assets

app = FastAPI()
VERBOSE = True


# -- SSE plumbing --------------------------------------------------------


def _chunk(model: str, delta: dict, finish: str | None = None) -> str:
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


def _usage_chunk(model: str, prompt_tokens: int, completion_tokens: int) -> str:
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(body)}\n\n"


def _stream_text(model: str, text: str) -> Iterator[str]:
    yield _chunk(model, {"role": "assistant", "content": ""})
    for word in text.split(" "):
        yield _chunk(model, {"content": word + " "})
    yield _chunk(model, {}, finish="stop")


def _stream_tool_call(model: str, name: str, arguments: dict) -> Iterator[str]:
    payload = json.dumps(arguments)
    yield _chunk(model, {"role": "assistant", "content": ""})
    yield _chunk(
        model,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ]
        },
    )
    # Split the arguments so the client has to reassemble them, as a real
    # server's chunking would force it to.
    for piece in [payload[i : i + 24] for i in range(0, len(payload), 24)]:
        yield _chunk(
            model, {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}
        )
    yield _chunk(model, {}, finish="tool_calls")


# -- reading the conversation --------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(parts)


def _describe_images(messages: list[dict]) -> list[str]:
    """Report the images that arrived, by hash, so they can be compared to disk."""
    found = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                head, _, data = url.partition(",")
                raw = data.encode()
                found.append(
                    f"{head[:30]}... {len(raw)} b64 chars sha256={hashlib.sha256(raw).hexdigest()[:16]}"
                )
    return found


def _history(messages: list[dict]) -> str:
    return " ".join(_text_of(m.get("content")) for m in messages if m.get("role") == "user")


def _tool_results(messages: list[dict]) -> list[str]:
    return [str(m.get("content", "")) for m in messages if m.get("role") == "tool"]


# -- the scripted behaviour ----------------------------------------------


def _decide(messages: list[dict]) -> tuple[str, Any]:
    """Return ("text", str) or ("tool", (name, args)) for this conversation state.

    Deliberately keyword-driven rather than clever. Its job is to exercise every
    path the harness measures, not to imitate a model.
    """
    last_user = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user = _text_of(message.get("content"))
            break
    prompt = last_user.casefold()
    results = _tool_results(messages)
    images = _describe_images(messages)

    # Once a tool has answered, say something grounded in what it returned.
    if results:
        joined = " ".join(results)
        if "Blocked:" in joined:
            return "text", "I could not read that file, so I have stopped rather than retrying."
        if "not found" in joined.casefold() or "no files" in joined.casefold():
            return (
                "text",
                "That file is not attached to this conversation, so I cannot read "
                "it. Please upload budget-2027.xlsx and I will total it for you.",
            )
        if assets.PDF_FACT in joined:
            return (
                "text",
                f"It was prepared for the {assets.PDF_FACT} division, and "
                f"{assets.PDF_NUMBER} units shipped in the quarter.",
            )
        if assets.CJK_FACT in joined:
            return "text", f"The primary client this quarter was {assets.CJK_FACT} Holdings."
        if "Units" in joined and "4350" in joined:
            return "text", f"The four regions total {assets.XLSX_TOTAL:,} units."
        if "scratch workspace" in joined or "wrote" in joined.casefold():
            return (
                "text",
                "I put it in my scratch workspace, which you cannot open or "
                "download. Say the word and I will save a real file instead.",
            )
        if "Saved to" in joined or "photosynthesis-final" in joined:
            return "text", "Done - the document has been saved as you asked."
        if "Not this time" in joined or "not save" in joined.casefold():
            return (
                "text",
                "Understood, I have not saved anything. The water cycle is "
                "evaporation, condensation, precipitation and collection.",
            )
        return "text", "Here is what I found: " + joined[:200]

    if images:
        return "text", "The image has three bands: red at the top, green in the middle, blue at the bottom."

    if "2+2" in prompt.replace(" ", ""):
        return "text", "4"
    if "capital of france" in prompt:
        return "text", "Paris"
    if "just acknowledge" in prompt:
        return "text", "Understood."
    if "what is my project called" in prompt:
        history = _history(messages)
        name = "Halcyon" if "Halcyon" in history else "unknown"
        month = "March" if "March" in history else "unknown"
        return "text", f"Your project is called {name} and it ships in {month}."
    if "budget-2027" in prompt:
        return "tool", ("read_document", {"filename": "budget-2027.xlsx"})
    if "write_file" in prompt:
        return "tool", ("write_file", {"file_path": "scratch.txt", "content": "one\ntwo"})
    if "attached" in prompt or "this image" in prompt:
        return "tool", ("read_document", {"filename": "1"})
    if "regions.docx" in prompt:
        content = (
            "# Regional Units\n\n"
            "| Region | Units |\n| --- | --- |\n"
            "| North | 2400 |\n| South | 3100 |\n| East | 1850 |\n| West | 4350 |\n"
        )
        return "tool", ("write_docx", {"filename": "regions.docx", "content": content})
    if "market-brief.docx" in prompt:
        content = (
            "# Renewable Energy Market Brief\n\n"
            "## Solar\n\nCosts continue to fall.\n\n"
            "## Wind\n\nOffshore capacity is growing.\n\n"
            "## Storage\n\nBatteries are the constraint.\n"
        )
        return "tool", ("write_docx", {"filename": "market-brief.docx", "content": content})
    if "draft.docx" in prompt:
        return "tool", (
            "write_docx",
            {"filename": "draft.docx", "content": "Photosynthesis converts light into sugar."},
        )
    if "notes.txt" in prompt:
        return "tool", (
            "write_text_file",
            {"filename": "notes.txt", "content": "Water evaporates, condenses and falls."},
        )
    return "text", "I am a stub model and had no script for that request."


@app.post("/v1/chat/completions")
async def completions(request: Request) -> StreamingResponse:
    payload = await request.json()
    messages = payload.get("messages") or []
    model = payload.get("model", "fake")
    wants_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

    if VERBOSE:
        images = _describe_images(messages)
        cleared = sum(str(m.get("content", "")).count("[cleared]") for m in messages)
        print(
            f"--> {len(messages)} messages, {len(payload.get('tools') or [])} tools"
            f"{f', {len(images)} image(s)' if images else ''}"
            f"{f', {cleared} cleared' if cleared else ''}",
            flush=True,
        )
        for image in images:
            print(f"    image: {image}", flush=True)

    kind, value = _decide(messages)

    def body() -> Iterator[str]:
        if kind == "tool":
            name, arguments = value
            if VERBOSE:
                print(f"<-- tool_call {name}({json.dumps(arguments)[:90]})", flush=True)
            yield from _stream_tool_call(model, name, arguments)
            completion = 20
        else:
            if VERBOSE:
                print(f"<-- text {value[:90]!r}", flush=True)
            yield from _stream_text(model, value)
            completion = max(1, len(value.split()))
        if wants_usage:
            # Roughly proportional to the prompt, so token growth is visible.
            prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
            yield _usage_chunk(model, prompt_tokens, completion)
        yield "data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "fake", "object": "model"}]}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args()
    VERBOSE = not options.quiet
    uvicorn.run(app, host="127.0.0.1", port=options.port, log_level="warning")
