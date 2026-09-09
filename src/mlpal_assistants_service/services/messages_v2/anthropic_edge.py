"""Anthropic provider edge — native Anthropic-Messages passthrough.

The faithful path: forward the (model-rewritten) Anthropic body to the
configured Anthropic backend and pipe the response straight through —
**raw SSE bytes, never parsed-and-reserialized** — so the wire shape is
byte-identical to talking to Anthropic directly. A copy of the stream is teed
into the Anthropic SSE parser only to extract usage for billing (exactly the
v1/messages technique). The core owns heartbeat + telemetry; this edge only
produces bytes and reports usage via ctx.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mlpal_assistants_service.services.bedrock_mantle import (
    parse_non_streaming_usage,
    parse_sse_event,
)
from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
    AnthropicAzureBackend,
    AnthropicBedrockBackend,
    AnthropicFirstPartyBackend,
)
from mlpal_assistants_service.services.messages_v2.edges import EdgeResult, RequestContext
from mlpal_assistants_service.services.messages_v2.errors import error_body
from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage

# Connection pool with the lifetime of the running event loop. A client per
# request would pay a fresh TCP+TLS handshake to the provider on EVERY request —
# connection reuse is the single largest per-request saving on this path.
# Weak-keyed by loop (never reuse a client across loops; ids recycle, objects
# don't) and invalidated when the AsyncClient constructor changes (tests
# monkeypatch it per-test). In production: one loop, one constructor, one
# cached client for the process lifetime.
_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    ctor = httpx.AsyncClient
    cached = _clients.get(loop)
    if cached is not None:
        client, cached_ctor = cached
        if cached_ctor is ctor and not client.is_closed:
            return client
    client = ctor(
        timeout=None,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    _clients[loop] = (client, ctor)
    return client


def _as_stream_error(raw: bytes, status_code: int) -> bytes:
    """Provider error body → the SSE `error` event payload. Anthropic's own
    errors are already `{"type":"error","error":{...}}`; anything else is
    wrapped so the client always sees that shape."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("type") == "error":
            return raw
    except ValueError:
        pass
    return error_body(status_code, raw.decode("utf-8", "replace")[:500])


def _apply_resolved_effort(body: dict[str, Any], ctx: RequestContext) -> None:
    """Native wire: the body is Anthropic-shaped already, so only an EXPLICIT
    `output_config.effort` outside Anthropic's own vocabulary needs rewriting
    (a pure `thinking` budget is left to Anthropic). `none` → thinking
    disabled (the resolver only yields it for models that accept it); a
    clamped rung → the applied rung; no lever → the field is dropped rather
    than bounced by the provider."""
    res = ctx.cc_metadata.get("reasoning_effort")
    if not res or res.get("source") != "explicit":
        return
    oc = dict(body.get("output_config") or {})
    applied = res.get("applied")
    if applied == "none":
        oc.pop("effort", None)
        body["thinking"] = {"type": "disabled"}
    elif applied is None:
        oc.pop("effort", None)
    else:
        oc["effort"] = applied
    if oc:
        body["output_config"] = oc
    else:
        body.pop("output_config", None)


class AnthropicEdge:
    def __init__(
        self,
        backend: AnthropicFirstPartyBackend | AnthropicBedrockBackend | AnthropicAzureBackend,
        timeout: float = 120.0,
    ) -> None:
        self._backend = backend
        self._timeout = timeout

    def _outbound(self, req: ValidatedRequest, ctx: RequestContext) -> tuple[bytes, dict[str, str]]:
        # Rewrite `model` to the provider's id (first-party == the tag, but the
        # router is the source of truth) and re-serialize. The RESPONSE is what
        # must stay byte-faithful, not the request. The backend owns final body
        # adaptation + auth headers (SigV4 backends sign the exact bytes).
        body = dict(req.body)
        body["model"] = ctx.provider_model_id
        _apply_resolved_effort(body, ctx)
        if req.model_kwargs:
            # Native wire: the whole body is provider-native already, so
            # kwargs merge top-level; Anthropic validates unknown fields
            # loudly itself (a 400 naming the field).
            body.update(req.model_kwargs)
        return self._backend.prepare(json.dumps(body).encode(), ctx.headers)

    async def invoke(self, req: ValidatedRequest, ctx: RequestContext) -> EdgeResult:
        content, headers = self._outbound(req, ctx)
        resp = await _shared_client().post(
            self._backend.url, content=content, headers=headers, timeout=self._timeout
        )
        usage_dict, msg_id = parse_non_streaming_usage(resp.content)
        ctx.report(
            CanonicalUsage.from_anthropic(usage_dict) if resp.status_code == 200 else None,
            resp.status_code,
            msg_id,
        )
        return EdgeResult(
            status_code=resp.status_code,
            body=resp.content,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def stream(self, req: ValidatedRequest, ctx: RequestContext) -> AsyncIterator[bytes]:
        content, headers = self._outbound(req, ctx)
        usage_dict: dict[str, Any] = {}
        msg_id: str | None = None
        status_code = 0
        buf = b""
        async with _shared_client().stream(
            "POST", self._backend.url, content=content,
            headers={**headers, "accept": "text/event-stream"},
            timeout=self._timeout,
        ) as resp:
            status_code = resp.status_code
            if status_code != 200:
                # The provider refused the request (a JSON error, not SSE). Our
                # 200 + SSE headers are already on the wire, so surface it as an
                # Anthropic-shaped `error` event instead of a bare JSON body.
                raw = await resp.aread()
                yield b"event: error\ndata: " + _as_stream_error(raw, status_code) + b"\n\n"
                ctx.report(None, status_code, None)
                return
            # aiter_bytes (not aiter_raw): decode the transport Content-
            # Encoding (Anthropic gzips the SSE stream) so the client gets
            # plain SSE. Still event-faithful — we never parse/reserialize
            # the events, only decompress the transport layer. (aiter_raw
            # would forward compressed bytes with no Content-Encoding header
            # -> the client sees garbage.)
            async for chunk in resp.aiter_bytes():
                yield chunk  # event-faithful passthrough
                buf += chunk
                while b"\n\n" in buf:
                    raw_event, buf = buf.split(b"\n\n", 1)
                    event_type, data = parse_sse_event(raw_event)
                    if not data:
                        continue
                    if event_type == "message_start":
                        msg = data.get("message", {}) or {}
                        msg_id = msg.get("id") or msg_id
                        usage_dict.update(msg.get("usage") or {})
                    elif event_type == "message_delta":
                        usage_dict.update(data.get("usage") or {})
        ctx.report(
            CanonicalUsage.from_anthropic(usage_dict) if status_code == 200 else None,
            status_code,
            msg_id,
        )
