"""Retry policy + abandoned-stream metering (incident 2026-09-29).

Contract:
- One retry layer (adapters/retry.py): a timeout is never retried; a
  pre-response fault (connection, connect-timeout, 408/409/429/5xx, boto
  throttling) is retried once; 4xx never. SDK retries are off.
- A timeout surfaces as ProviderTimeoutError (HTTP 504) and is neither hopped
  to another backend nor retried on a fallback model.
- Anthropic non-streaming completions ride the streaming transport so the
  read timeout is per event, not per generation.
- A stream the client abandons is metered from the provider's first usage
  report (Anthropic message_start, Gemini usage_metadata, v2 edge progressive
  report); with nothing reported it leaves a visible client_disconnect row.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from tests.unit.test_backend_failover import bedrock_first  # noqa: F401 — fixture

from mlpal_assistants_service.adapters import retry
from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter
from mlpal_assistants_service.adapters.base import TokenUsage
from mlpal_assistants_service.adapters.retry import (
    call_provider,
    is_pre_response_fault,
    is_timeout,
)
from mlpal_assistants_service.core.exceptions import (
    ProviderError,
    ProviderTimeoutError,
    http_status_for_provider_error,
)
from mlpal_assistants_service.services.chat import ChatService
from mlpal_assistants_service.services.messages_v2 import anthropic_backend as ab
from mlpal_assistants_service.services.messages_v2 import anthropic_edge as ae
from mlpal_assistants_service.services.messages_v2.anthropic_edge import AnthropicEdge
from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core
from mlpal_assistants_service.services.messages_v2.edges import RequestContext
from mlpal_assistants_service.services.messages_v2.schemas import validate


# -- SDK-shaped exceptions (classified by name / attributes, like the real ones)
class APITimeoutError(Exception): ...
class APIConnectionError(Exception): ...
class ConnectTimeout(Exception): ...
class ReadTimeoutError(Exception): ...
class EndpointConnectionError(Exception): ...


class APIStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


class GoogleAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"google {code}")
        self.code = code


class ClientError(Exception):  # botocore shape
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


@pytest.mark.parametrize(
    "exc,timeout,pre_response",
    [
        (APITimeoutError(), True, False),
        (ReadTimeoutError(), True, False),
        (TimeoutError(), True, False),
        (httpx.ReadTimeout("slow"), True, False),
        (ConnectTimeout(), False, True),
        (httpx.ConnectTimeout("no route"), False, True),
        (APIConnectionError(), False, True),
        (EndpointConnectionError(), False, True),
        (httpx.ConnectError("refused"), False, True),
        (APIStatusError(429), False, True),
        (APIStatusError(529), False, True),
        (APIStatusError(503), False, True),
        (APIStatusError(400), False, False),
        (APIStatusError(401), False, False),
        (GoogleAPIError(503), False, True),
        (GoogleAPIError(400), False, False),
        (ClientError("ThrottlingException", 429), False, True),
        (ClientError("ValidationException", 400), False, False),
        (ValueError("parse"), False, False),
    ],
)
def test_classification(exc, timeout, pre_response):
    assert is_timeout(exc) is timeout
    assert is_pre_response_fault(exc) is pre_response


@pytest.fixture(autouse=True)
def _quiet_retry_metric(monkeypatch):
    monkeypatch.setattr(retry, "_note_retry", lambda *a, **k: None)
    monkeypatch.setattr(retry, "_sleep", AsyncMock())


@pytest.mark.asyncio
async def test_call_provider_retries_pre_response_fault_once():
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) == 1:
            raise APIStatusError(503)
        return "ok"

    assert await call_provider(fn, provider="p") == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_call_provider_gives_up_after_the_one_retry():
    calls = []

    async def fn():
        calls.append(1)
        raise APIStatusError(503)

    with pytest.raises(APIStatusError):
        await call_provider(fn, provider="p")
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [APITimeoutError(), APIStatusError(400), ValueError("x")])
async def test_call_provider_never_retries_timeouts_or_client_errors(exc):
    calls = []

    async def fn():
        calls.append(1)
        raise exc

    with pytest.raises(type(exc)):
        await call_provider(fn, provider="p")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_call_provider_propagates_cancellation_without_retry():
    calls = []

    async def fn():
        calls.append(1)
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await call_provider(fn, provider="p")
    assert len(calls) == 1


def test_timeout_error_is_terminal_on_every_layer():
    e = ProviderTimeoutError("slow", provider="anthropic", original_error="x")
    assert isinstance(e, ProviderError)
    assert e.status_code == 504
    assert http_status_for_provider_error(e) == 504
    assert ChatService._retriable(e) is False
    # A statusless ProviderError (connection failure) still hops.
    assert ChatService._retriable(ProviderError("conn", provider="anthropic")) is True
    assert ChatService._retriable(TimeoutError()) is False
    assert ChatService._retriable(ConnectionError()) is True


# -- Anthropic adapter: non-streaming over the streaming transport ------------
class _StreamCtx:
    def __init__(self, final=None, raise_exc=None):
        self._final, self._raise = final, raise_exc

    async def __aenter__(self):
        if self._raise:
            raise self._raise
        return self

    async def __aexit__(self, *a):
        return False

    async def get_final_message(self):
        return self._final


def _final_message():
    block = SimpleNamespace(type="text", text="hello")
    usage = SimpleNamespace(input_tokens=12, output_tokens=3, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0, cache_creation=None)
    return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn",
                           model="claude-opus-5", model_dump=lambda: {"ok": True})


def _adapter(stream_ctx) -> AnthropicAdapter:
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=stream_ctx)
    client.beta.messages.stream = MagicMock(return_value=stream_ctx)
    a = AnthropicAdapter(api_key="k", client=client)
    return a


@pytest.mark.asyncio
async def test_anthropic_chat_uses_stream_transport_and_never_create():
    a = _adapter(_StreamCtx(final=_final_message()))
    resp = await a.chat(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "hello" and resp.usage.input_tokens == 12
    a._client.messages.stream.assert_called_once()
    a._client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_chat_timeout_is_a_timeout_error_not_retried():
    ctx = _StreamCtx(raise_exc=APITimeoutError("Request timed out"))
    a = _adapter(ctx)
    with pytest.raises(ProviderTimeoutError):
        await a.chat(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    assert a._client.messages.stream.call_count == 1


@pytest.mark.asyncio
async def test_anthropic_chat_retries_overloaded_once():
    seq = [_StreamCtx(raise_exc=APIStatusError(529)), _StreamCtx(final=_final_message())]
    client = MagicMock()
    client.messages.stream = MagicMock(side_effect=seq)
    a = AnthropicAdapter(api_key="k", client=client)
    resp = await a.chat(model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "hello"
    assert client.messages.stream.call_count == 2


# -- Anthropic adapter stream: message_start surfaces input usage -------------
class _Events:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()


@pytest.mark.asyncio
async def test_anthropic_stream_yields_usage_at_message_start():
    usage = SimpleNamespace(input_tokens=250, output_tokens=1, cache_read_input_tokens=100,
                            cache_creation_input_tokens=0, cache_creation=None)
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage)),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="hi")),
    ]
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_Events(events))
    a = AnthropicAdapter(api_key="k", client=client)
    chunks = []
    async for c in a.chat_stream(model="claude-opus-5", messages=[{"role": "user", "content": "x"}]):
        chunks.append(c)
    assert chunks[0].usage is not None and chunks[0].usage.input_tokens == 250
    assert chunks[0].usage.cached_tokens == 100 and chunks[0].done is False
    assert chunks[1].content == "hi" and chunks[1].usage is None


# -- ChatService: abandoned stream metering -----------------------------------
def _svc() -> ChatService:
    svc = ChatService.__new__(ChatService)
    svc._pricing = MagicMock()
    svc._pricing.calculate_compute_units = AsyncMock(return_value=Decimal("0.5"))
    svc._post_request_background = AsyncMock()
    svc._record_failure = AsyncMock()
    return svc


_ABANDON = {
    "resolved_model_tag": "claude-opus-5",
    "byom": False,
    "provider": "anthropic",
    "cached_included": False,
    "conn_kind": None,
    "serving_backend": "bedrock",
    "billing_needs_ensure": False,
}


@pytest.mark.asyncio
async def test_abandoned_stream_with_usage_is_a_success_row():
    svc = _svc()
    usage = TokenUsage(input_tokens=1000, output_tokens=1, cached_tokens=400)
    await svc._record_abandoned_stream(
        user_id=1, api_key_id=2, trace_id="t", request_model="mlpal", abandon=_ABANDON,
        usage=usage, start_time=0.0, budgets=None, served_backend="bedrock",
    )
    svc._record_failure.assert_not_awaited()
    kw = svc._post_request_background.await_args.kwargs
    assert kw["resolved_model_tag"] == "claude-opus-5"
    assert kw["compute_units"] == Decimal("0.5")
    assert kw["extra_metadata"] == {"stream_aborted": True, "output_tokens_known": False}
    assert kw["cache_read_tokens"] == 400
    # input excludes cache reads on the Anthropic adapter: wire input = 1000 + 400
    assert kw["input_tokens"] == 1400


@pytest.mark.asyncio
async def test_abandoned_stream_without_usage_is_a_visible_error_row():
    svc = _svc()
    await svc._record_abandoned_stream(
        user_id=1, api_key_id=2, trace_id="t", request_model="mlpal", abandon=_ABANDON,
        usage=None, start_time=0.0, budgets=None, served_backend="bedrock",
    )
    svc._post_request_background.assert_not_awaited()
    assert svc._record_failure.await_args.kwargs["error_code"] == "client_disconnect"
    assert svc._record_failure.await_args.kwargs["model_tag"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_abandoned_stream_before_routing_names_the_requested_model():
    svc = _svc()
    await svc._record_abandoned_stream(
        user_id=1, api_key_id=2, trace_id="t", request_model="mlpal", abandon=None,
        usage=None, start_time=0.0, budgets=None, served_backend=None,
    )
    assert svc._record_failure.await_args.kwargs["model_tag"] == "mlpal"


# -- v2 Anthropic wire ---------------------------------------------------------
def _ctx() -> RequestContext:
    return RequestContext(model_tag="claude-opus-5", provider="anthropic",
                          provider_model_id="claude-opus-5", backend="bedrock",
                          trace_id="t", api_key=MagicMock(capture_policy=None), headers={})


_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
    b'"usage":{"input_tokens":5000,"output_tokens":1,"cache_read_input_tokens":2000}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"a"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":40}}\n\n',
)


class _StallingBody(httpx.AsyncByteStream):
    """message_start, then the provider keeps the socket open (still generating)."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for c in self._chunks:
            yield c
        await asyncio.sleep(30)  # cancelled by the test, never reached otherwise


@pytest.mark.asyncio
async def test_v2_edge_reports_usage_progressively(monkeypatch, bedrock_first):  # noqa: F811
    s, _ = bedrock_first
    # First-party backend: plain header auth, no SigV4 signing on the loop.
    monkeypatch.setattr(s, "anthropic_backends", "first_party")

    def handler(request):
        return httpx.Response(200, stream=_StallingBody(_SSE[:1]),
                              headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ae, "_shared_client", lambda: client)
    edge = AnthropicEdge(ab.native_backend_for(s, "claude-opus-5"))
    ctx = _ctx()
    req = validate(b'{"model":"claude-opus-5","messages":[],"stream":true,"max_tokens":1}')

    async def consume():  # the core's producer task
        async for _ in edge.stream(req, ctx):
            pass

    task = asyncio.create_task(consume())

    async def _reported():
        while ctx.usage is None:  # first-call setup (signing, factory) can be slow
            await asyncio.sleep(0.02)

    await asyncio.wait_for(_reported(), timeout=60)
    assert ctx.status_code == 200 and ctx.usage is not None
    assert ctx.usage.prompt_total() == 7000  # 5000 input + 2000 cache read
    task.cancel()  # client disconnect → producer cancelled mid-stream
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()
    assert ctx.usage is not None and ctx.usage.output == 1  # input side survives


@pytest.mark.asyncio
async def test_v2_disconnect_meters_what_the_edge_reported():
    core = MessagesV2Core(router=AsyncMock(), usage_service=AsyncMock(),
                          pricing_service=AsyncMock(), billing_gate=AsyncMock())
    core._meter = AsyncMock(return_value=Decimal("0"))
    ctx = _ctx()

    class _Edge:
        async def stream(self, req, c):
            c.report(MagicMock(input=10, output=1), 200, "msg")
            yield b"first"
            await asyncio.sleep(30)  # provider still streaming when the client leaves

    req = validate(b'{"model":"claude-opus-5","messages":[],"stream":true,"max_tokens":1}')
    core._meter_detached = AsyncMock()
    body = core._stream_with_heartbeat(_Edge(), req, ctx, 0.0, model=None)
    assert await body.__anext__() == b"first"
    await asyncio.wait_for(body.aclose(), timeout=2.0)
    await asyncio.sleep(0)  # metering is handed to a detached task
    core._meter.assert_not_awaited()
    core._meter_detached.assert_awaited_once()
    assert ctx.cc_metadata["client_disconnected"] is True
    assert ctx.cc_metadata["output_tokens_known"] is False
    assert ctx.usage is not None  # billed from the progressive report, not zero
