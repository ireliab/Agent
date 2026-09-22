"""Instrumentation that measures a run without changing what the agent does.

Attached as the outermost middleware, so it sees every model call and every
tool call - including the ones a later middleware refuses. It only times and
records; it never alters a request or a response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    duration_s: float
    status: str  # ok | error | blocked | exception
    detail: str = ""


@dataclass
class ModelCallRecord:
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Probe(AgentMiddleware):
    """Records timings and outcomes for one run. Not reusable across runs."""

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_calls: list[ModelCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()

    # -- tool calls ------------------------------------------------------

    @staticmethod
    def _classify(result: Any) -> tuple[str, str]:
        """Work out how a tool call ended, from whatever the handler returned."""
        message = result
        # A middleware may return a Command carrying the ToolMessage instead.
        if not isinstance(message, ToolMessage):
            update = getattr(result, "update", None) or {}
            messages = update.get("messages") if isinstance(update, dict) else None
            message = next(
                (m for m in (messages or []) if isinstance(m, ToolMessage)), None
            )
        if message is None:
            return "ok", ""
        text = str(message.content)
        if getattr(message, "status", None) == "error":
            # The repeated-call guard is the one refusal we want to tell apart
            # from a tool that ran and failed, since it means a loop was broken.
            status = "blocked" if text.startswith("Blocked:") else "error"
            return status, text[:400]
        return "ok", ""

    def _record(self, call: dict, started: float, result: Any) -> None:
        status, detail = self._classify(result)
        self.tool_calls.append(
            ToolCallRecord(
                name=call.get("name", "?"),
                args=call.get("args") or {},
                duration_s=time.perf_counter() - started,
                status=status,
                detail=detail,
            )
        )

    def wrap_tool_call(self, request, handler):
        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception as exc:
            self._fail(request.tool_call, started, exc)
            raise
        self._record(request.tool_call, started, result)
        return result

    async def awrap_tool_call(self, request, handler):
        started = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as exc:
            self._fail(request.tool_call, started, exc)
            raise
        self._record(request.tool_call, started, result)
        return result

    def _fail(self, call: dict, started: float, exc: Exception) -> None:
        self.tool_calls.append(
            ToolCallRecord(
                name=call.get("name", "?"),
                args=call.get("args") or {},
                duration_s=time.perf_counter() - started,
                status="exception",
                detail=f"{type(exc).__name__}: {exc}"[:400],
            )
        )

    # -- model calls -----------------------------------------------------

    def _record_model(self, started: float, response: Any) -> None:
        usage: dict[str, Any] = {}
        for message in getattr(response, "result", None) or []:
            usage = getattr(message, "usage_metadata", None) or usage
        self.model_calls.append(
            ModelCallRecord(
                duration_s=time.perf_counter() - started,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        )

    def wrap_model_call(self, request, handler):
        started = time.perf_counter()
        response = handler(request)
        self._record_model(started, response)
        return response

    async def awrap_model_call(self, request, handler):
        started = time.perf_counter()
        response = await handler(request)
        self._record_model(started, response)
        return response
