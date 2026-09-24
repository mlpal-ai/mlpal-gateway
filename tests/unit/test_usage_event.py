"""Opt-in per-request usage trailer on the Anthropic wire.

Contract: `X-MLPal-Usage-Event: 1` appends ONE `event: mlpal_usage` after the
provider's stream ends (and puts the same JSON in `X-MLPal-Usage` on
non-streaming responses). Without the header the stream is byte-identical to
the provider's — Claude Code never sees a non-Anthropic event. The trailer
carries the billed CU for this request and Anthropic-named usage fields so a
client can reconcile the meter against the stream it just read. A client that
disconnects mid-stream is still metered exactly once.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.services.messages_v2.core import (
    MessagesV2Core,
    _usage_event_json,
    _wants_usage_event,
)
from mlpal_assistants_service.services.messages_v2.edges import RequestContext
from mlpal_assistants_service.services.messages_v2.schemas import validate
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage


def _core() -> MessagesV2Core:
    core = MessagesV2Core(router=AsyncMock(), usage_service=AsyncMock(),
                          pricing_service=AsyncMock(), billing_gate=AsyncMock())
    core._meter = AsyncMock(return_value=Decimal("0.0000172"))
    return core


def _ctx(headers) -> RequestContext:
    return RequestContext(model_tag="claude-opus-5-5", provider="anthropic",
                          provider_model_id="claude-opus-5-5", backend="bedrock",
                          trace_id="t-1", api_key=object(), headers=headers)


FRAMES = [b"event: message_start\ndata: {}\n\n", b"event: content_block_delta\ndata: {}\n\n",
          b"event: message_stop\ndata: {}\n\n"]


class _Edge:
    async def stream(self, req, ctx):
        for f in FRAMES:
            yield f
        ctx.report(CanonicalUsage(input=13, output=6, cache_read=5407, cache_write=0), 200, "msg")

    async def invoke(self, req, ctx):  # pragma: no cover — not used
        raise NotImplementedError


def test_header_is_opt_in_and_case_insensitive():
    assert not _wants_usage_event({})
    assert not _wants_usage_event({"x-mlpal-usage-event": "0"})
    assert _wants_usage_event({"x-mlpal-usage-event": "1"})
    assert _wants_usage_event({"X-Mlpal-Usage-Event": "true"})


def test_event_json_uses_anthropic_usage_names():
    ctx = _ctx({})
    ctx.report(CanonicalUsage(input=13, output=6, cache_read=5407, cache_write=0, reasoning=2), 200, "m")
    ctx.cc_metadata["backend_fallback_from"] = "bedrock"
    d = json.loads(_usage_event_json(ctx, Decimal("0.0000172")))
    assert d["type"] == "mlpal_usage" and d["compute_units"] == "0.0000172"
    assert d["usage"] == {"input_tokens": 13, "output_tokens": 6, "cache_read_input_tokens": 5407,
                          "cache_creation_input_tokens": 0, "reasoning_tokens": 2}
    assert d["model"] == "claude-opus-5-5" and d["serving_backend"] == "bedrock"
    assert d["trace_id"] == "t-1" and d["backend_fallback_from"] == "bedrock"


@pytest.mark.asyncio
async def test_stream_without_header_is_byte_identical():
    core = _core()
    req = validate(b'{"model":"m","messages":[],"stream":true}')
    out = b"".join([c async for c in core._stream_with_heartbeat(_Edge(), req, _ctx({}), 0.0)])
    assert out == b"".join(FRAMES)
    core._meter.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_with_header_appends_one_trailer_after_message_stop():
    core = _core()
    req = validate(b'{"model":"m","messages":[],"stream":true}')
    chunks = [c async for c in core._stream_with_heartbeat(_Edge(), req, _ctx({"x-mlpal-usage-event": "1"}), 0.0)]
    assert chunks[:3] == FRAMES and len(chunks) == 4
    assert chunks[3].startswith(b"event: mlpal_usage\ndata: ")
    d = json.loads(chunks[3][len(b"event: mlpal_usage\ndata: "):].strip())
    assert d["compute_units"] == "0.0000172" and d["usage"]["cache_read_input_tokens"] == 5407
    core._meter.assert_awaited_once()  # metered before the trailer, not again in finally


@pytest.mark.asyncio
async def test_disconnect_mid_stream_is_metered_once_without_trailer():
    core = _core()
    req = validate(b'{"model":"m","messages":[],"stream":true}')
    gen = core._stream_with_heartbeat(_Edge(), req, _ctx({"x-mlpal-usage-event": "1"}), 0.0)
    first = await gen.__anext__()
    assert first == FRAMES[0]
    await gen.aclose()  # client went away
    core._meter.assert_awaited_once()


@pytest.mark.asyncio
async def test_mid_stream_provider_error_still_ends_with_trailer():
    import httpx

    class _Mid(_Edge):
        async def stream(self, req, ctx):
            yield FRAMES[0]
            ctx.report(None, 502, None)
            raise httpx.ReadError("cut")

    core = _core()
    core._meter = AsyncMock(return_value=Decimal("0"))
    req = validate(b'{"model":"m","messages":[],"stream":true}')
    chunks = [c async for c in core._stream_with_heartbeat(_Mid(), req, _ctx({"x-mlpal-usage-event": "1"}), 0.0)]
    assert chunks[0] == FRAMES[0] and chunks[1].startswith(b"event: error\n")
    d = json.loads(chunks[2][len(b"event: mlpal_usage\ndata: "):].strip())
    assert chunks[2].startswith(b"event: mlpal_usage") and d["status_code"] == 502 and d["compute_units"] == "0"
    assert len(chunks) == 3
    core._meter.assert_awaited_once()
