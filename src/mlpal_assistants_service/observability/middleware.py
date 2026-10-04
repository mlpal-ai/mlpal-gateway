"""Request observability middleware.

One pure-ASGI middleware (NOT Starlette BaseHTTPMiddleware): BaseHTTPMiddleware
wraps every response in an extra task + memory stream, which adds a scheduling
hop to every streamed chunk — including the first token — and it required a
second middleware that json.loads'd the entire request body just to read the
`model` field. This implementation adds no task, never buffers or re-parses the
body, and emits metrics only after the response has started.

Emits per request: RequestCount and RequestLatency (latency to response start),
dimensioned by operation, model, and status class.

Model dimension resolution order:
1. `request.state.model` if an endpoint set it (authoritative), else
2. a bytes-regex match on the first body chunk of /v1 POSTs (no JSON parse;
   requests with the model field beyond the first chunk fall through), else
3. "unknown".
"""

import re
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mlpal_assistants_service.core.metrics import get_metrics
from mlpal_assistants_service.observability.client import bind_client_context

_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"]{1,128})"')
_SKIP_PATHS = ("/", "/docs", "/redoc", "/openapi.json")


def _extract_operation(path: str) -> str:
    """/v1/chat/completions -> chat_completions."""
    parts = path.strip("/").split("/")
    if parts and parts[0].startswith("v"):
        parts = parts[1:]
    return "_".join(parts) if parts else "unknown"


def _categorize_status(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "success"
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "client_error"
    return "server_error"


class ObservabilityMiddleware:
    """Metrics for every API request, off the first-byte path."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self.metrics = get_metrics()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        bind_client_context(scope)
        path = scope["path"]
        method = scope["method"]
        if method == "OPTIONS" or path.startswith("/health") or path in _SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        sniff_model = method == "POST" and path.startswith("/v1/")
        body_model: str | None = None
        first_chunk_seen = False
        response_started = False

        async def receive_wrapped() -> Message:
            nonlocal body_model, first_chunk_seen
            message = await receive()
            if sniff_model and not first_chunk_seen and message["type"] == "http.request":
                first_chunk_seen = True
                m = _MODEL_RE.search(message.get("body", b"")[:65536])
                if m:
                    body_model = m.group(1).decode("utf-8", errors="replace")
            return message

        async def send_wrapped(message: Message) -> None:
            nonlocal response_started
            await send(message)
            if not response_started and message["type"] == "http.response.start":
                response_started = True
                # AFTER forwarding response.start — metric emission (an EMF log
                # line) never sits between the client and its first byte.
                latency_ms = (time.perf_counter() - start) * 1000
                model = scope.get("state", {}).get("model") or body_model or "unknown"
                dimensions = {
                    "operation": _extract_operation(path),
                    "model": model,
                    "status": _categorize_status(message["status"]),
                }
                self.metrics.put_metric_sync("RequestCount", 1, "Count", dimensions)
                self.metrics.put_metric_sync(
                    "RequestLatency", latency_ms, "Milliseconds", dimensions
                )

        try:
            await self.app(scope, receive_wrapped, send_wrapped)
        except Exception:
            if not response_started:
                dimensions = {
                    "operation": _extract_operation(path),
                    "model": body_model or "unknown",
                    "status": "server_error",
                }
                self.metrics.put_metric_sync("RequestCount", 1, "Count", dimensions)
            raise
