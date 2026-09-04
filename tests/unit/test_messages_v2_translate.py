"""Conformance tests for the v2 translating edge (OpenAI/Google on the Anthropic
surface).

Covers the three translation seams independently — inbound (Anthropic→common),
outbound non-streaming (AdapterResponse→Anthropic JSON), outbound streaming
(StreamChunk→Anthropic SSE) — and the full path through MessagesV2Core with a
faked adapter, asserting the Anthropic wire shape, the cache-read split in the
usage block, and provider-error mapping to the Anthropic envelope.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import mlpal_assistants_service.services.messages_v2.core as core_mod
from mlpal_assistants_service.adapters.base import (
    AdapterResponse,
    FileAttachment,
    FileSource,
    FileType,
    StreamChunk,
    TokenUsage,
)
from mlpal_assistants_service.core.exceptions import ProviderError
from mlpal_assistants_service.services.messages_v2 import emit
from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core, is_served_chat_model
from mlpal_assistants_service.services.messages_v2.reasoning import effort_from_thinking
from mlpal_assistants_service.services.messages_v2.schemas import validate
from mlpal_assistants_service.services.messages_v2.translate_in import (
    sanitize_google_tool_schema,
    to_common,
)

GPT = "gpt-5.5"


# --- Gemini thought_signature round-trip (via tool_use id) -------------------
def test_emit_encodes_thought_signature_into_tool_use_id():
    tc = {"id": "call_0", "type": "function",
          "function": {"name": "get_time", "arguments": '{"city":"Tokyo"}'},
          "thought_signature": "SIGB64=="}
    block = emit._tool_use_block(tc)
    assert block["id"] == "call_0" + emit.TOOL_SIG_DELIM + "SIGB64=="
    # no signature (e.g. OpenAI) → id untouched
    assert emit._tool_use_block({"id": "call_1", "function": {"name": "f", "arguments": "{}"}})["id"] == "call_1"


def test_translate_in_restores_thought_signature_from_id():
    encoded = "call_0" + emit.TOOL_SIG_DELIM + "SIGB64=="
    body = {"model": "gemini-3.5-flash", "messages": [
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": encoded, "name": "get_time", "input": {"city": "Tokyo"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": encoded, "content": "14:05"}]},
    ]}
    msgs = to_common(body).messages
    asst = next(m for m in msgs if m.get("role") == "assistant")
    assert asst["tool_calls"][0]["id"] == "call_0"  # signature stripped off the id
    assert asst["tool_calls"][0]["thought_signature"] == "SIGB64=="  # restored for the adapter
    tool_msg = next(m for m in msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_0"  # tool_result id normalized to match
    assert tool_msg["name"] == "get_time"  # name carried from tool_use (Google pairs by name)


def test_translate_in_plain_id_has_no_signature():
    body = {"model": "gemini-3.5-flash", "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call_0", "name": "f", "input": {}}]}]}
    tc = to_common(body).messages[0]["tool_calls"][0]
    assert tc["id"] == "call_0" and "thought_signature" not in tc


# --- wildcard admission policy -----------------------------------------------
@pytest.mark.parametrize("provider,operation,served", [
    ("openai", "chat", True),
    ("google", "chat", True),
    ("anthropic", "chat", True),
    ("openai", "image_generation", False),   # non-chat operation
    ("openai", "embedding", False),
    ("bedrock", "chat", True),               # open-weight lineup, translating edge
    ("bedrock", "embedding", False),         # titan embeddings stay excluded
    ("mistral", "chat", False),              # provider with no edge
])
def test_is_served_chat_model(provider, operation, served):
    model = SimpleNamespace(provider=provider, capabilities={"operation": operation})
    assert is_served_chat_model(model) is served


def test_is_served_chat_model_defaults_to_chat_when_unspecified():
    # A model whose capabilities omit "operation" is treated as chat.
    assert is_served_chat_model(SimpleNamespace(provider="openai", capabilities={})) is True


# --- inbound translation -----------------------------------------------------
def test_to_common_text_and_system():
    body = {
        "model": GPT,
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.3,
    }
    c = to_common(body)
    assert c.messages[0] == {"role": "system", "content": "be terse"}
    assert c.messages[1] == {"role": "user", "content": "hi"}
    assert c.max_tokens == 64 and c.temperature == 0.3


def test_to_common_tool_roundtrip():
    body = {
        "model": GPT,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "SF"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "72F"},
            ]},
        ],
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "tool_choice": {"type": "any"},
    }
    c = to_common(body)
    # assistant turn carries OpenAI-shaped tool_calls
    asst = next(m for m in c.messages if m.get("role") == "assistant")
    assert asst["tool_calls"][0]["id"] == "tu_1"
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"city": "SF"}
    # tool result becomes a tool-role message
    tool_msg = next(m for m in c.messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "tu_1" and tool_msg["content"] == "72F"
    assert c.tool_choice == "required"  # any → required
    assert c.tools[0].name == "get_weather"


def test_to_common_image_block_routes_to_files():
    body = {"model": GPT, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "what colors?"},
    ]}]}
    m = to_common(body).messages[0]
    assert m["role"] == "user" and m["content"] == "what colors?"
    # image rides on files as a FileAttachment (NOT a Chat-Completions image_url part)
    assert not isinstance(m["content"], list)
    fa = m["files"][0]
    assert isinstance(fa, FileAttachment)
    assert fa.type == FileType.IMAGE and fa.source == FileSource.BASE64
    assert fa.data == "AAAA" and fa.mime_type == "image/png"


def test_to_common_document_block_routes_to_files_pdf():
    body = {"model": GPT, "messages": [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER"}},
        {"type": "text", "text": "quote the phrase"},
    ]}]}
    m = to_common(body).messages[0]
    fa = m["files"][0]
    assert fa.type == FileType.PDF and fa.mime_type == "application/pdf" and fa.data == "JVBER"
    assert m["content"] == "quote the phrase"  # document not silently dropped


def test_to_common_image_url_block():
    body = {"model": GPT, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
    ]}]}
    fa = to_common(body).messages[0]["files"][0]
    assert fa.source == FileSource.URL and fa.data == "https://x/y.png"


@pytest.mark.parametrize("thinking,expected", [
    (None, None),
    ({"type": "disabled"}, None),
    ({"type": "enabled", "budget_tokens": 1024}, "low"),
    ({"type": "enabled", "budget_tokens": 8000}, "medium"),
    ({"type": "enabled", "budget_tokens": 32000}, "high"),
])
def test_effort_from_thinking(thinking, expected):
    assert effort_from_thinking(thinking) == expected


# --- outbound non-streaming --------------------------------------------------
def test_message_json_text_tooluse_and_cache_split():
    resp = AdapterResponse(
        content="hello",
        model="gpt-5.5",
        provider="openai",
        usage=TokenUsage(input_tokens=100, output_tokens=20, cached_tokens=80),
        finish_reason="tool_calls",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "f", "arguments": '{"x":1}'}}],
    )
    body = json.loads(emit.message_json(resp, "msg_abc", GPT))
    assert body["type"] == "message" and body["role"] == "assistant" and body["model"] == GPT
    assert body["stop_reason"] == "tool_use"  # tool_calls → tool_use
    assert body["content"][0] == {"type": "text", "text": "hello"}
    assert body["content"][1] == {"type": "tool_use", "id": "call_1", "name": "f", "input": {"x": 1}}
    # Anthropic input_tokens excludes the cache-read prefix
    assert body["usage"] == {
        "input_tokens": 20, "output_tokens": 20,
        "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0,
    }


def test_message_json_tooluse_forces_stop_reason_even_if_provider_says_stop():
    # Google reports finish_reason="stop" even when it emits a tool call; the
    # Anthropic-correct stop_reason must be derived from the tool_use block.
    resp = AdapterResponse(
        content="", model="gemini-3.5-flash", provider="google",
        usage=TokenUsage(input_tokens=55, output_tokens=16),
        finish_reason="stop",
        tool_calls=[{"id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}],
    )
    body = json.loads(emit.message_json(resp, "msg_g", "gemini-3.5-flash"))
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["type"] == "tool_use"


# --- outbound streaming ------------------------------------------------------
async def _collect_sse(chunks):
    captured = {}

    def on_final(usage, stop_reason, empty_completion):
        captured["usage"] = usage
        captured["stop_reason"] = stop_reason
        captured["empty_completion"] = empty_completion

    async def _gen():
        yield StreamChunk(content="")  # reasoning keepalive — no block opened
        for c in chunks:
            yield c

    out = bytearray()
    async for b in emit.stream_anthropic_sse(_gen(), "msg_x", GPT, on_final):
        out += b
    return bytes(out), captured


@pytest.mark.asyncio
async def test_stream_text_then_tool_sequence():
    chunks = [
        StreamChunk(content="He"),
        StreamChunk(content="llo"),
        StreamChunk(content="", tool_calls=[{"id": "call_9", "type": "function",
                                             "function": {"name": "f", "arguments": '{"a":2}'}}]),
        StreamChunk(content="", done=True,
                    usage=TokenUsage(input_tokens=50, output_tokens=7, cached_tokens=10),
                    finish_reason="tool_calls"),
    ]
    out, captured = await _collect_sse(chunks)
    text = out.decode()
    # event order: message_start → text block(0) → tool block(1) → message_delta → message_stop
    assert text.index("message_start") < text.index('"index": 0')
    assert text.index('"text_delta"') < text.index('"input_json_delta"')
    assert json.dumps(json.dumps({"a": 2})) in text  # tool args as a partial_json string
    assert text.index("content_block_stop") < text.rindex("message_delta")
    assert text.endswith("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n")
    # message_delta carries final usage with cache split + stop_reason
    last_delta = json.loads(text.split("event: message_delta\ndata: ")[1].split("\n\n")[0])
    assert last_delta["delta"]["stop_reason"] == "tool_use"
    assert last_delta["usage"] == {
        "input_tokens": 40, "output_tokens": 7,
        "cache_read_input_tokens": 10, "cache_creation_input_tokens": 0,
    }
    assert captured["usage"].output_tokens == 7 and captured["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_empty_completion_when_reasoning_exhausts_budget():
    # Reasoning model burns the whole budget: no text/tool chunks, finish=length.
    chunks = [
        StreamChunk(content="", done=True,
                    usage=TokenUsage(input_tokens=19, output_tokens=16, cached_tokens=0),
                    finish_reason="length"),
    ]
    _, captured = await _collect_sse(chunks)
    assert captured["stop_reason"] == "max_tokens"       # faithful signal still emitted
    assert captured["empty_completion"] is True          # ...and flagged for observability


@pytest.mark.asyncio
async def test_stream_not_empty_when_text_present_even_at_length():
    # Truncated-but-non-empty (hit max_tokens after some text) is NOT an empty completion.
    chunks = [
        StreamChunk(content="partial answer"),
        StreamChunk(content="", done=True,
                    usage=TokenUsage(input_tokens=19, output_tokens=64, cached_tokens=0),
                    finish_reason="length"),
    ]
    _, captured = await _collect_sse(chunks)
    assert captured["stop_reason"] == "max_tokens"
    assert captured["empty_completion"] is False


# --- full path through the core with a faked adapter -------------------------
class _FakeAdapter:
    def __init__(self, response=None, chunks=None, error=None):
        self._response, self._chunks, self._error = response, chunks, error

    def validate_model_kwargs(self, model_kwargs):
        return None

    async def chat(self, **kwargs):
        if self._error:
            raise self._error
        return self._response

    async def chat_stream(self, **kwargs):
        if self._error:
            raise self._error
        for c in self._chunks:
            yield c


def _openai_core(monkeypatch, adapter):
    model = SimpleNamespace(
        model_tag=GPT, provider="openai", provider_model_id=GPT, display_name="GPT-5.5",
        capabilities={"tools": True, "vision": True}, max_output_tokens=128000,
    )
    router = MagicMock()
    router.get_model = AsyncMock(return_value=model)
    router.resolve_meta_model = AsyncMock(side_effect=lambda tag, op: (tag, None))
    pricing = MagicMock()
    pricing.get_pricing = AsyncMock(return_value=SimpleNamespace(
        input_cu_rate=Decimal("1.0"), output_cu_rate=Decimal("4.0"), rate_unit="per_1m_tokens",
        markup_multiplier=Decimal("3.0"),
    ))
    usage = MagicMock()
    usage.record_usage = AsyncMock()
    usage.redis = None
    billing = MagicMock()
    billing.can_make_request_cached = AsyncMock(return_value=(True, None, True))
    billing.is_wallet_debit_active = AsyncMock(return_value=True)
    billing.debit_wallet_usage = AsyncMock()
    monkeypatch.setattr(core_mod, "get_adapter_factory",
                        lambda: SimpleNamespace(get=lambda provider: adapter, resolve=lambda provider, pmid: (adapter, pmid)))
    return MessagesV2Core(router, usage, pricing, billing), usage, billing


def _bedrock_core(monkeypatch, adapter):
    """Same fixture shape as _openai_core, but the model row is a bedrock
    open-weight entry — the lineup the translating edge must serve."""
    model = SimpleNamespace(
        model_tag="deepseek.v3.2", provider="bedrock",
        provider_model_id="us.deepseek.v3-2:0", display_name="DeepSeek V3.2",
        capabilities={"tools": True}, max_output_tokens=32000,
    )
    router = MagicMock()
    router.get_model = AsyncMock(return_value=model)
    router.resolve_meta_model = AsyncMock(side_effect=lambda tag, op: (tag, None))
    pricing = MagicMock()
    pricing.get_pricing = AsyncMock(return_value=SimpleNamespace(
        input_cu_rate=Decimal("1.0"), output_cu_rate=Decimal("4.0"), rate_unit="per_1m_tokens",
        markup_multiplier=Decimal("3.0"),
    ))
    usage = MagicMock()
    usage.record_usage = AsyncMock()
    usage.redis = None
    billing = MagicMock()
    billing.can_make_request_cached = AsyncMock(return_value=(True, None, True))
    billing.is_wallet_debit_active = AsyncMock(return_value=True)
    billing.debit_wallet_usage = AsyncMock()
    monkeypatch.setattr(core_mod, "get_adapter_factory",
                        lambda: SimpleNamespace(get=lambda provider: adapter, resolve=lambda provider, pmid: (adapter, pmid)))
    return MessagesV2Core(router, usage, pricing, billing), usage, billing


def _api_key():
    return SimpleNamespace(user_id=1, id=2)


def _stub_post_billing(core):
    core._post_billing = AsyncMock()
    return core


@pytest.mark.asyncio
async def test_core_openai_nonstreaming(monkeypatch):
    adapter = _FakeAdapter(response=AdapterResponse(
        content="hi there", model=GPT, provider="openai",
        usage=TokenUsage(input_tokens=100, output_tokens=20, cached_tokens=80),
        finish_reason="stop",
    ))
    core, usage, billing = _openai_core(monkeypatch, adapter)
    _stub_post_billing(core)
    req = validate(json.dumps({"model": GPT, "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-o1")

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["content"] == [{"type": "text", "text": "hi there"}]
    assert body["usage"]["cache_read_input_tokens"] == 80
    kw = usage.record_usage.await_args.kwargs
    assert kw["provider"] == "openai"
    assert kw["input_tokens"] == 100  # full input recorded; cache surfaced separately
    assert kw["output_tokens"] == 20
    # CU is provider pass-through: per-token cu_rates divided by the ROW's
    # markup (3.0 in this fixture) before multiplying — mirror the impl order.
    _in_rate = Decimal("1.0") / Decimal("1000000") / Decimal("3.0")
    _out_rate = Decimal("4.0") / Decimal("1000000") / Decimal("3.0")
    # Cached tokens (80 of the 100 input) bill at the provider's cache-read
    # rate (openai: 0.10x of input) — the uncached 20 at the full input rate.
    _provider_cu = (
        Decimal("20") * _in_rate
        + Decimal("80") * _in_rate * Decimal("0.10")
        + Decimal("20") * _out_rate
    )
    assert kw["compute_units"] == _provider_cu
    assert kw["status"] == "success"
    assert "empty_completion" not in kw["cc_metadata"]  # normal completion → no flag
    await asyncio.sleep(0.05)
    core._post_billing.assert_awaited_once()


@pytest.mark.asyncio
async def test_core_bedrock_nonstreaming(monkeypatch):
    """Bedrock (open-weight catalog) serves on the Anthropic wire through the
    same translating edge as openai/google — wire universality holds."""
    adapter = _FakeAdapter(response=AdapterResponse(
        content="hello from deepseek", model="us.deepseek.v3-2:0", provider="bedrock",
        usage=TokenUsage(input_tokens=50, output_tokens=10),
        finish_reason="stop",
    ))
    core, usage, _ = _bedrock_core(monkeypatch, adapter)
    _stub_post_billing(core)
    req = validate(json.dumps(
        {"model": "deepseek.v3.2", "messages": [{"role": "user", "content": "hi"}]}
    ).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-b1")

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["content"] == [{"type": "text", "text": "hello from deepseek"}]
    assert body["stop_reason"] == "end_turn"
    kw = usage.record_usage.await_args.kwargs
    assert kw["provider"] == "bedrock"
    assert kw["status"] == "success"


@pytest.mark.asyncio
async def test_core_bedrock_streaming_tool_use(monkeypatch):
    """A bedrock stream that ends in a tool call comes out as Anthropic SSE
    with stop_reason tool_use — the shape agentic clients depend on."""
    adapter = _FakeAdapter(chunks=[
        StreamChunk(content="thinking...", done=False),
        StreamChunk(content="", done=False, tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
        }]),
        StreamChunk(content="", done=True,
                    usage=TokenUsage(input_tokens=30, output_tokens=8),
                    finish_reason="stop"),
    ])
    core, usage, _ = _bedrock_core(monkeypatch, adapter)
    _stub_post_billing(core)
    req = validate(json.dumps({
        "model": "deepseek.v3.2", "stream": True,
        "messages": [{"role": "user", "content": "weather?"}],
    }).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-b2")
    events = b""
    async for part in resp.body_iterator:
        events += part if isinstance(part, bytes) else part.encode()

    text = events.decode()
    assert '"type": "tool_use"' in text or '"type":"tool_use"' in text
    assert "get_weather" in text
    assert "tool_use" in [
        json.loads(line[6:]).get("delta", {}).get("stop_reason")
        for line in text.splitlines() if line.startswith("data: ")
        if "message_delta" in line
    ]


@pytest.mark.asyncio
async def test_core_openai_nonstreaming_empty_completion_flagged(monkeypatch):
    # Reasoning model returned zero visible output, finish=length → flag it in usage_logs.
    adapter = _FakeAdapter(response=AdapterResponse(
        content="", model=GPT, provider="openai",
        usage=TokenUsage(input_tokens=19, output_tokens=16, cached_tokens=0),
        finish_reason="length",
    ))
    core, usage, billing = _openai_core(monkeypatch, adapter)
    _stub_post_billing(core)
    req = validate(json.dumps({"model": GPT, "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 16}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-empty")

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["content"] == []                       # empty, faithfully
    assert body["stop_reason"] == "max_tokens"         # clear signal to retry with more budget
    kw = usage.record_usage.await_args.kwargs
    assert kw["status"] == "success"                   # still a billable turn (tokens were spent)
    assert kw["cc_metadata"]["empty_completion"] is True  # observable


@pytest.mark.asyncio
async def test_core_openai_streaming(monkeypatch):
    adapter = _FakeAdapter(chunks=[
        StreamChunk(content="hel"),
        StreamChunk(content="lo"),
        StreamChunk(content="", done=True,
                    usage=TokenUsage(input_tokens=30, output_tokens=5, cached_tokens=0),
                    finish_reason="stop"),
    ])
    core, usage, _ = _openai_core(monkeypatch, adapter)
    req = validate(json.dumps({"model": GPT, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-o2")
    out = bytearray()
    async for part in resp.body_iterator:
        out += part if isinstance(part, (bytes, bytearray)) else part.encode()
    text = bytes(out).decode()

    assert "message_start" in text and "message_stop" in text
    assert '"text":"hel"' in text.replace(" ", "") or '"text": "hel"' in text
    # We never estimate: message_start.input_tokens is N/A (0) because the
    # provider only reports prompt tokens at stream end; the authoritative count
    # is delivered factually in the terminal message_delta (and usage_logs).
    ms = json.loads(text.split("event: message_start\ndata: ")[1].split("\n\n")[0])
    assert ms["message"]["usage"]["input_tokens"] == 0
    md = json.loads(text.split("event: message_delta\ndata: ")[1].split("\n\n")[0])
    assert md["usage"]["input_tokens"] == 30  # provider-authoritative, exact
    usage.record_usage.assert_awaited_once()
    kw = usage.record_usage.await_args.kwargs
    assert kw["input_tokens"] == 30 and kw["output_tokens"] == 5 and kw["status"] == "success"


# --- message_start carries no fabricated input count -------------------------
@pytest.mark.asyncio
async def test_stream_message_start_input_tokens_is_na_until_end():
    async def _gen():
        yield StreamChunk(content="ok", done=True,
                          usage=TokenUsage(input_tokens=99, output_tokens=2), finish_reason="stop")
    out = bytearray()
    async for b in emit.stream_anthropic_sse(_gen(), "msg_z", GPT, lambda u, s, e: None):
        out += b
    text = bytes(out).decode()
    ms = json.loads(text.split("event: message_start\ndata: ")[1].split("\n\n")[0])
    assert ms["message"]["usage"]["input_tokens"] == 0  # N/A — never estimated
    md = json.loads(text.split("event: message_delta\ndata: ")[1].split("\n\n")[0])
    assert md["usage"]["input_tokens"] == 99  # provider-authoritative, exact


# --- thinking-token streaming ------------------------------------------------
async def _sse_events(chunks):
    """Drain the emitter into a list of (event_type, data_dict), in order."""
    out = bytearray()
    async for b in emit.stream_anthropic_sse(chunks, "msg_t", GPT, lambda u, s, e: None):
        out += b
    events = []
    for block in bytes(out).decode().split("\n\n"):
        if block.startswith("event: "):
            etype = block.splitlines()[0][len("event: "):]
            data = json.loads(block.split("data: ", 1)[1])
            events.append((etype, data))
    return events


@pytest.mark.asyncio
async def test_stream_thinking_block_precedes_text():
    async def _gen():
        yield StreamChunk(content="", thinking="Let me think")
        yield StreamChunk(content="", thinking=" about it.")
        yield StreamChunk(content="Answer: 8")
        yield StreamChunk(content="", done=True,
                          usage=TokenUsage(input_tokens=10, output_tokens=3), finish_reason="stop")
    ev = await _sse_events(_gen())
    starts = [(d["index"], d["content_block"]["type"]) for t, d in ev if t == "content_block_start"]
    # thinking block at index 0, text block at index 1
    assert starts == [(0, "thinking"), (1, "text")]
    # thinking_delta carries the reasoning; text_delta carries the answer; no leakage
    thinking_txt = "".join(d["delta"]["thinking"] for t, d in ev
                           if t == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
    answer_txt = "".join(d["delta"]["text"] for t, d in ev
                         if t == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert thinking_txt == "Let me think about it." and answer_txt == "Answer: 8"
    # thinking block is stopped before the text block starts
    order = [t for t, _ in ev]
    assert order.index("content_block_stop") < order.index("content_block_start", order.index("content_block_start") + 1)


@pytest.mark.asyncio
async def test_stream_thinking_then_tool_use():
    async def _gen():
        yield StreamChunk(content="", thinking="pick a tool")
        yield StreamChunk(content="", tool_calls=[{"id": "c1", "type": "function",
                                                   "function": {"name": "f", "arguments": '{"x":1}'}}])
        yield StreamChunk(content="", done=True,
                          usage=TokenUsage(input_tokens=5, output_tokens=2), finish_reason="tool_calls")
    ev = await _sse_events(_gen())
    starts = [d["content_block"]["type"] for t, d in ev if t == "content_block_start"]
    assert starts == ["thinking", "tool_use"]  # thinking precedes tool_use
    md = next(d for t, d in ev if t == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use"


# --- P2: Google tool-schema sanitization -------------------------------------
def test_sanitize_google_tool_schema_strips_and_inlines():
    schema = {
        "type": "object", "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False, "title": "Args",
        "properties": {
            "city": {"type": "string", "title": "City"},
            "loc": {"$ref": "#/$defs/Loc"},
        },
        "required": ["city"],
        "$defs": {"Loc": {"type": "object", "additionalProperties": False,
                          "properties": {"lat": {"type": "number"}}}},
    }
    clean = sanitize_google_tool_schema(schema)
    # offending keys gone at every level
    assert "additionalProperties" not in clean and "$schema" not in clean
    assert "title" not in clean and "$defs" not in clean
    assert "title" not in clean["properties"]["city"]
    # $ref inlined, nested additionalProperties stripped
    assert clean["properties"]["loc"] == {"type": "object", "properties": {"lat": {"type": "number"}}}
    # kept keys preserved
    assert clean["type"] == "object" and clean["required"] == ["city"]
    assert clean["properties"]["city"] == {"type": "string"}


class _CapturingAdapter:
    def __init__(self):
        self.kwargs = None

    def validate_model_kwargs(self, model_kwargs):
        return None
    async def chat(self, **kwargs):
        self.kwargs = kwargs
        return AdapterResponse(content="ok", model="gemini", provider="google",
                               usage=TokenUsage(input_tokens=5, output_tokens=1), finish_reason="stop")
    async def chat_stream(self, **kwargs):  # pragma: no cover - not used here
        if False:
            yield None


@pytest.mark.asyncio
async def test_google_edge_sanitizes_tool_schema(monkeypatch):
    adapter = _CapturingAdapter()
    model = SimpleNamespace(model_tag="gemini-3.5-flash", provider="google", provider_model_id="gemini-3.5-flash",
                            display_name="Gemini", capabilities={"operation": "chat"}, max_output_tokens=8192)
    router = MagicMock()
    router.get_model = AsyncMock(return_value=model)
    router.resolve_meta_model = AsyncMock(side_effect=lambda tag, op: (tag, None))
    pricing = MagicMock()
    pricing.get_pricing = AsyncMock(return_value=SimpleNamespace(
        input_cu_rate=Decimal("1.0"), output_cu_rate=Decimal("4.0"), rate_unit="per_1m_tokens",
        markup_multiplier=Decimal("3.0")))
    usage = MagicMock()
    usage.record_usage = AsyncMock()
    usage.redis = None
    billing = MagicMock()
    billing.can_make_request_cached = AsyncMock(return_value=(True, None, True))
    billing.is_wallet_debit_active = AsyncMock(return_value=True)
    billing.debit_wallet_usage = AsyncMock()
    monkeypatch.setattr(core_mod, "get_adapter_factory",
                        lambda: SimpleNamespace(get=lambda provider: adapter, resolve=lambda provider, pmid: (adapter, pmid)))
    core = MessagesV2Core(router, usage, pricing, billing)
    req = validate(json.dumps({
        "model": "gemini-3.5-flash",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": {"type": "object", "additionalProperties": False,
                                    "$schema": "http://x", "properties": {"city": {"type": "string"}}}}],
    }).encode())

    await core.handle(req, _api_key(), {}, "trace-g1")

    tool_params = adapter.kwargs["tools"][0].parameters
    assert "additionalProperties" not in tool_params and "$schema" not in tool_params
    assert tool_params["properties"] == {"city": {"type": "string"}}


@pytest.mark.asyncio
async def test_core_resolves_meta_model(monkeypatch):
    # `mlpal` must resolve to a concrete chat model, be admitted, and route —
    # not 404 like it did before v2 ran resolve_meta_model.
    adapter = _FakeAdapter(response=AdapterResponse(
        content="hi", model=GPT, provider="openai",
        usage=TokenUsage(input_tokens=5, output_tokens=2), finish_reason="stop"))
    core, usage, _ = _openai_core(monkeypatch, adapter)
    core._router.resolve_meta_model = AsyncMock(return_value=(GPT, None))  # mlpal → gpt-5.5
    req = validate(json.dumps({"model": "mlpal", "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-meta")

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["model"] == GPT  # response reflects the concrete model that served
    kw = usage.record_usage.await_args.kwargs
    assert kw["model_tag"] == GPT  # billing/usage under the resolved tag
    assert kw["cc_metadata"]["requested_model"] == "mlpal"  # original alias retained


@pytest.mark.asyncio
async def test_core_openai_provider_error_maps_to_envelope(monkeypatch):
    adapter = _FakeAdapter(error=ProviderError(message="bad input", provider="openai", status_code=400))
    core, usage, billing = _openai_core(monkeypatch, adapter)
    _stub_post_billing(core)
    req = validate(json.dumps({"model": GPT, "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-o3")

    assert resp.status_code == 400
    err = json.loads(resp.body)
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error"
    assert usage.record_usage.await_args.kwargs["status"] == "error"
    core._post_billing.assert_not_awaited()
