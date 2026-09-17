"""Automatic backend failover on both wires.

Contract (worklog 2026-09-17-claude-on-bedrock, "Backend failover"):
- Circuit breakers are PER BACKEND (`family:backend`), so a failing backend
  trips only its own breaker and the model keeps serving from the next one.
- A serving fault (5xx/529, transport error, provider 429, open breaker) on
  the backend that served a model retries the SAME model once on the next
  backend in the family's priority list — before any client model fallback.
- Never on 4xx, never after a stream has emitted, at most one hop, never for
  connection-served requests, off with MLPAL_BACKEND_FAILOVER=false.
- Native stays native on the Anthropic wire (no hop onto the lossy
  translating edge); the failed attempt is metered under its own backend.
- Attribution: chat `metadata.backend_fallback_from`, usage cc_metadata
  `backend_fallback_from`, Anthropic-wire header X-MLPal-Backend-Fallback-From.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter
from mlpal_assistants_service.adapters.circuit_breaker import (
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
)
from mlpal_assistants_service.adapters.factory import AdapterFactory
from mlpal_assistants_service.adapters.serving import BedrockAnthropicAdapter
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    ModelNotAvailableError,
    ProviderError,
)
from mlpal_assistants_service.services.chat import ChatService, _Attempt
from mlpal_assistants_service.services.messages_v2 import anthropic_backend as ab
from mlpal_assistants_service.services.messages_v2 import anthropic_edge as ae
from mlpal_assistants_service.services.messages_v2.anthropic_edge import AnthropicEdge
from mlpal_assistants_service.services.messages_v2.core import (
    MessagesV2Core,
    _backend_failover_headers,
)
from mlpal_assistants_service.services.messages_v2.edges import (
    EdgeResult,
    RequestContext,
    UpstreamRefused,
)
from mlpal_assistants_service.services.messages_v2.schemas import validate
from mlpal_assistants_service.services.router import ModelRouter

MAP = {"claude-opus-5": "global.anthropic.claude-opus-5"}


@pytest.fixture()
def bedrock_first(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "anthropic_backends", "bedrock,first_party")
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(s, "bedrock_anthropic_models", json.dumps(MAP))
    monkeypatch.setattr(s, "bedrock_mantle_models", json.dumps(["claude-opus-5"]))
    monkeypatch.setattr(s, "bedrock_mantle_region", "us-east-2")
    monkeypatch.setattr(s, "azure_openai_endpoint", None)
    monkeypatch.setattr(s, "azure_openai_api_key", None)
    monkeypatch.setattr(s, "backend_failover_enabled", True, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    ab._backends.clear()
    ab._backend_lists.clear()
    f = AdapterFactory()
    f.clear_instances()
    yield s, f
    f.clear_instances()
    ab._backends.clear()
    ab._backend_lists.clear()


# ── factory / router ────────────────────────────────────────────────────────


def test_factory_exclude_walks_to_next_backend(bedrock_first):
    _, f = bedrock_first
    primary, _ = f.resolve("anthropic", "claude-opus-5")
    assert isinstance(primary, BedrockAnthropicAdapter)
    alt, wire = f.resolve("anthropic", "claude-opus-5", frozenset({"bedrock"}))
    assert type(alt) is AnthropicAdapter and wire == "claude-opus-5"
    with pytest.raises(ValueError, match="excluded"):
        f.resolve("anthropic", "claude-opus-5", frozenset({"bedrock", "first_party"}))
    # exclusion is part of the cache key — the primary stays bedrock
    assert f.resolve("anthropic", "claude-opus-5")[0] is primary


@pytest.mark.asyncio
async def test_breakers_are_per_backend():
    router = ModelRouter.__new__(ModelRouter)
    router._circuit_breakers = CircuitBreakerRegistry()
    bedrock = await router.breaker_for("anthropic", "bedrock")
    first_party = await router.breaker_for("anthropic", "first_party")
    assert bedrock is not first_party
    assert bedrock.provider == "anthropic:bedrock"
    for _ in range(bedrock.config.failure_threshold):
        with pytest.raises(RuntimeError):
            async with bedrock:
                raise RuntimeError("down")
    assert bedrock.is_open and first_party.is_closed


def test_retriable_includes_open_breaker_and_statusless_provider_error():
    assert ChatService._retriable(CircuitBreakerOpen("anthropic:bedrock", 30.0))
    # the adapter's wrap of an SDK connection error carries no HTTP status
    assert ChatService._retriable(
        ProviderError("Anthropic API error: Connection error.", provider="anthropic")
    )
    assert not ChatService._retriable(ProviderError("bad", provider="anthropic", status_code=400))


# ── chat wire ───────────────────────────────────────────────────────────────


def _svc(alternate: bool = True) -> ChatService:
    svc = ChatService.__new__(ChatService)
    svc._router = MagicMock()
    if alternate:
        svc._router.resolve_backend.return_value = (object(), "wire")
    else:
        svc._router.resolve_backend.side_effect = ModelNotAvailableError("m", "none")
    return svc


def _once(fail_with: Exception | None, *, backend="bedrock", conn_served=False):
    """A `_chat_once` stand-in: fails on the primary backend, serves on the hop."""
    calls: list[frozenset[str]] = []

    async def fake(user_id, api_key_id, request, tier, model_policy, budgets, *,
                   capture_policy=None, exclude_backends=frozenset(), attempt=None):
        calls.append(exclude_backends)
        if attempt is not None:
            attempt.model, attempt.backend, attempt.conn_served = object(), backend, conn_served
        if not exclude_backends and fail_with is not None:
            raise fail_with
        return MagicMock(metadata={"served": "first_party"})

    return fake, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        ProviderError("boom", provider="anthropic", status_code=503),
        ProviderError("throttled", provider="anthropic", status_code=429),
        CircuitBreakerOpen("anthropic:bedrock", 30.0),
        TimeoutError(),
    ],
)
async def test_chat_hops_once_on_serving_fault(monkeypatch, fault):
    svc = _svc()
    fake, calls = _once(fault)
    monkeypatch.setattr(svc, "_chat_once", fake)
    resp = await svc._chat_with_backend_failover(1, 2, MagicMock(model="claude-opus-5"),
                                                 "standard", None, None, capture_policy=None)
    assert calls == [frozenset(), frozenset({"bedrock"})]
    assert resp.metadata["served"] == "first_party"
    svc._router.resolve_backend.assert_called_once()
    assert svc._router.resolve_backend.call_args.args[1] == frozenset({"bedrock"})


@pytest.mark.asyncio
async def test_chat_no_hop_on_client_error(monkeypatch):
    svc = _svc()
    fake, calls = _once(ProviderError("bad", provider="anthropic", status_code=400))
    monkeypatch.setattr(svc, "_chat_once", fake)
    with pytest.raises(ProviderError):
        await svc._chat_with_backend_failover(1, 2, MagicMock(model="m"), "standard",
                                              None, None, capture_policy=None)
    assert calls == [frozenset()]


@pytest.mark.asyncio
async def test_chat_no_hop_without_alternate_or_for_connections_or_when_off(monkeypatch):
    fault = ProviderError("boom", provider="anthropic", status_code=503)
    # no other backend serves the model
    svc = _svc(alternate=False)
    fake, calls = _once(fault)
    monkeypatch.setattr(svc, "_chat_once", fake)
    with pytest.raises(ProviderError):
        await svc._chat_with_backend_failover(1, 2, MagicMock(model="m"), "standard",
                                              None, None, capture_policy=None)
    assert calls == [frozenset()]
    # tenant connection served it — their outage, never billed onto ours
    svc = _svc()
    fake, calls = _once(fault, conn_served=True)
    monkeypatch.setattr(svc, "_chat_once", fake)
    with pytest.raises(ProviderError):
        await svc._chat_with_backend_failover(1, 2, MagicMock(model="m"), "standard",
                                              None, None, capture_policy=None)
    assert calls == [frozenset()]
    # feature off
    monkeypatch.setattr(get_settings(), "backend_failover_enabled", False, raising=False)
    svc = _svc()
    fake, calls = _once(fault)
    monkeypatch.setattr(svc, "_chat_once", fake)
    with pytest.raises(ProviderError):
        await svc._chat_with_backend_failover(1, 2, MagicMock(model="m"), "standard",
                                              None, None, capture_policy=None)
    assert calls == [frozenset()]


@pytest.mark.asyncio
async def test_chat_hop_failure_propagates_to_model_fallback(monkeypatch):
    """Second backend also fails → the model-fallback loop sees the hop's error."""
    svc = _svc()
    seen = []

    async def fake(*a, exclude_backends=frozenset(), attempt=None, **k):
        seen.append(exclude_backends)
        if attempt is not None:
            attempt.model, attempt.backend = object(), "bedrock"
        raise ProviderError("still down", provider="anthropic", status_code=502)

    monkeypatch.setattr(svc, "_chat_once", fake)
    with pytest.raises(ProviderError, match="still down"):
        await svc._chat_with_backend_failover(1, 2, MagicMock(model="m"), "standard",
                                              None, None, capture_policy=None)
    assert seen == [frozenset(), frozenset({"bedrock"})]


@pytest.mark.asyncio
async def test_chat_stream_hops_only_before_first_chunk(monkeypatch):
    svc = _svc()
    calls = []

    def make(fail_after_emit: bool):
        async def fake(*a, exclude_backends=frozenset(), attempt=None, **k):
            calls.append(exclude_backends)
            if attempt is not None:
                attempt.model, attempt.backend = object(), "bedrock"
            if not exclude_backends:
                if fail_after_emit:
                    yield "partial"
                raise ProviderError("boom", provider="anthropic", status_code=503)
            yield "ok-from-first-party"
        return fake

    monkeypatch.setattr(svc, "_chat_stream_once", make(False))
    out = [c async for c in svc._chat_stream_with_backend_failover(
        1, 2, MagicMock(model="m"), "standard", None, None, capture_policy=None)]
    assert out == ["ok-from-first-party"] and calls == [frozenset(), frozenset({"bedrock"})]

    calls.clear()
    monkeypatch.setattr(svc, "_chat_stream_once", make(True))
    got = []
    with pytest.raises(ProviderError):
        async for c in svc._chat_stream_with_backend_failover(
                1, 2, MagicMock(model="m"), "standard", None, None, capture_policy=None):
            got.append(c)
    assert got == ["partial"] and calls == [frozenset()]  # committed: no hop


def test_attempt_defaults():
    a = _Attempt()
    assert a.model is None and a.backend is None and a.conn_served is False


# ── Anthropic wire ──────────────────────────────────────────────────────────


def _core() -> MessagesV2Core:
    core = MessagesV2Core(router=AsyncMock(), usage_service=AsyncMock(),
                          pricing_service=AsyncMock(), billing_gate=AsyncMock())
    core._meter = AsyncMock(return_value=0)
    return core


def _ctx(backend="bedrock", conn_kind=None) -> RequestContext:
    return RequestContext(model_tag="claude-opus-5", provider="anthropic",
                          provider_model_id="claude-opus-5", backend=backend,
                          trace_id="t", api_key=object(), headers={}, conn_kind=conn_kind)


class _Model:
    provider = "anthropic"
    provider_model_id = "claude-opus-5"
    model_tag = "claude-opus-5"


class _Edge:
    def __init__(self, status=200, *, raise_exc=None, chunks=(b"a", b"b")):
        self.status, self.raise_exc, self.chunks, self.calls = status, raise_exc, chunks, 0

    async def invoke(self, req, ctx):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        ctx.report(None, self.status, None)
        return EdgeResult(status_code=self.status, body=b'{"s":%d}' % self.status)

    async def stream(self, req, ctx):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        if self.status != 200:
            ctx.report(None, self.status, None)
            raise UpstreamRefused(self.status, b'{"type":"error"}')
        for c in self.chunks:
            yield c
        ctx.report(None, 200, "msg")


def test_failover_edge_native_stays_native(bedrock_first):
    s, _ = bedrock_first
    core = _core()
    bedrock = ab.native_backend_for(s, "claude-opus-5")
    assert bedrock.name == "bedrock"
    alt = core._failover_edge(AnthropicEdge(bedrock), _Model(), _ctx("bedrock"))
    assert alt is not None
    edge2, ctx2 = alt
    assert isinstance(edge2, AnthropicEdge) and edge2._backend.name == "first_party"
    assert ctx2.backend == "first_party"
    assert ctx2.cc_metadata["backend_fallback_from"] == "bedrock"
    assert ctx2.status_code == 0 and ctx2.usage is None
    # connection-served: never hop
    assert core._failover_edge(AnthropicEdge(bedrock), _Model(), _ctx("bedrock", "byok")) is None


def test_failover_edge_none_when_no_other_native_backend(bedrock_first, monkeypatch):
    s, _ = bedrock_first
    monkeypatch.setattr(s, "anthropic_backends", "bedrock")
    ab._backends.clear()
    ab._backend_lists.clear()
    core = _core()
    bedrock = ab.native_backend_for(s, "claude-opus-5")
    # first_party is configured for the adapter path, but native→translating
    # would change the response shape mid-conversation: no hop.
    assert core._failover_edge(AnthropicEdge(bedrock), _Model(), _ctx("bedrock")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503, 504, 529, 429])
async def test_v2_invoke_hops_and_meters_failed_attempt(monkeypatch, status):
    core = _core()
    primary, alt = _Edge(status), _Edge(200)
    alt_ctx = _ctx("first_party")
    alt_ctx.cc_metadata["backend_fallback_from"] = "bedrock"
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (alt, alt_ctx))
    ctx = _ctx()
    result, served = await core._invoke(primary, validate(b'{"model":"m","messages":[]}'),
                                        ctx, _Model(), 0.0)
    assert result.status_code == 200 and served is alt_ctx
    assert primary.calls == 1 and alt.calls == 1
    core._meter.assert_awaited_once()  # the failed attempt, under its backend
    assert core._meter.await_args.args[0] is ctx and ctx.status_code == status
    assert _backend_failover_headers(served) == {"X-MLPal-Backend-Fallback-From": "bedrock"}


@pytest.mark.asyncio
async def test_v2_invoke_no_hop_on_4xx_and_at_most_one_hop(monkeypatch):
    core = _core()
    alt = _Edge(200)
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (alt, _ctx("first_party")))
    req = validate(b'{"model":"m","messages":[]}')
    result, served = await core._invoke(_Edge(400), req, _ctx(), _Model(), 0.0)
    assert result.status_code == 400 and alt.calls == 0
    core._meter.assert_not_awaited()
    # hop lands on a backend that also fails: its result is returned, no third try
    second = _Edge(503)
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (second, _ctx("first_party")))
    result, _ = await core._invoke(_Edge(503), req, _ctx(), _Model(), 0.0)
    assert result.status_code == 503 and second.calls == 1


@pytest.mark.asyncio
async def test_v2_invoke_transport_error_hops(monkeypatch):
    core = _core()
    alt = _Edge(200)
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (alt, _ctx("first_party")))
    primary = _Edge(raise_exc=httpx.ConnectError("dns"))
    result, _ = await core._invoke(primary, validate(b'{"model":"m","messages":[]}'),
                                   _ctx(), _Model(), 0.0)
    assert result.status_code == 200 and alt.calls == 1
    # and surfaces a 502 when nowhere to go
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: None)
    result, _ = await core._invoke(_Edge(raise_exc=httpx.ConnectError("dns")),
                                   validate(b'{"model":"m","messages":[]}'), _ctx(), _Model(), 0.0)
    assert result.status_code == 502 and b"transport" in result.body


@pytest.mark.asyncio
async def test_v2_invoke_uses_per_backend_breaker(monkeypatch):
    core = _core()
    registry = CircuitBreakerRegistry()
    core._router.circuit_breakers = registry
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: None)
    req = validate(b'{"model":"m","messages":[]}')
    breaker = registry.get_sync("anthropic:bedrock")
    for _ in range(breaker.config.failure_threshold):
        await core._invoke(_Edge(503), req, _ctx(), _Model(), 0.0)
    assert breaker.is_open and registry.get_sync("anthropic:first_party").is_closed
    # open breaker → 503 without touching the edge
    untouched = _Edge(200)
    result, _ = await core._invoke(untouched, req, _ctx(), _Model(), 0.0)
    assert result.status_code == 503 and untouched.calls == 0


@pytest.mark.asyncio
async def test_v2_stream_hops_before_first_chunk(monkeypatch):
    core = _core()
    alt = _Edge(200, chunks=(b"x", b"y"))
    alt_ctx = _ctx("first_party")
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (alt, alt_ctx))
    queue: asyncio.Queue = asyncio.Queue()
    live = {"ctx": _ctx()}
    await core._pump(_Edge(529), validate(b'{"model":"m","messages":[],"stream":true}'),
                     live["ctx"], _Model(), 0.0, queue, live)
    assert [queue.get_nowait() for _ in range(2)] == [("chunk", b"x"), ("chunk", b"y")]
    assert live["ctx"] is alt_ctx
    core._meter.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_stream_no_hop_on_4xx_or_after_emit(monkeypatch):
    core = _core()
    alt = _Edge(200)
    monkeypatch.setattr(core, "_failover_edge", lambda e, m, c: (alt, _ctx("first_party")))
    req = validate(b'{"model":"m","messages":[],"stream":true}')
    with pytest.raises(UpstreamRefused):
        await core._pump(_Edge(400), req, _ctx(), _Model(), 0.0, asyncio.Queue(), {"ctx": _ctx()})
    assert alt.calls == 0

    class _Mid(_Edge):
        async def stream(self, req, ctx):
            yield b"first"
            raise httpx.ReadError("cut")

    with pytest.raises(httpx.ReadError):
        await core._pump(_Mid(), req, _ctx(), _Model(), 0.0, asyncio.Queue(), {"ctx": _ctx()})
    assert alt.calls == 0


@pytest.mark.asyncio
async def test_anthropic_edge_stream_raises_typed_refusal(monkeypatch, bedrock_first):
    s, _ = bedrock_first

    def handler(request):
        return httpx.Response(529, json={"type": "error",
                                         "error": {"type": "overloaded_error", "message": "busy"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ae, "_shared_client", lambda: client)
    edge = AnthropicEdge(ab.native_backend_for(s, "claude-opus-5"))
    ctx = _ctx()
    req = validate(b'{"model":"claude-opus-5","messages":[],"stream":true,"max_tokens":1}')
    with pytest.raises(UpstreamRefused) as ei:
        async for _ in edge.stream(req, ctx):
            pass
    assert ei.value.status_code == 529 and b"overloaded_error" in ei.value.body
    assert ctx.status_code == 529
