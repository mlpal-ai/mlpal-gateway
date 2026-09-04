"""Conformance tests for /v2/messages (v2-A, Anthropic native passthrough).

Mocks the Anthropic backend HTTP (httpx.MockTransport) and drives the core +
AnthropicEdge directly, asserting: byte-faithful response passthrough, usage/CU
accounting (incl. cache_read), and the Anthropic error envelope. These fail
loudly on wire drift.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import mlpal_assistants_service.services.messages_v2.anthropic_edge as anthropic_edge
from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core
from mlpal_assistants_service.services.messages_v2.schemas import (
    InvalidMessagesRequest,
    validate,
)

OPUS = "claude-opus-4-8"

# --- canned Anthropic wire payloads -----------------------------------------
NONSTREAM_JSON = json.dumps({
    "id": "msg_01ABC", "type": "message", "role": "assistant", "model": OPUS,
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 8404},
}).encode()

STREAM_SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01X",'
    b'"role":"assistant","model":"' + OPUS.encode() + b'","usage":{"input_tokens":10,'
    b'"output_tokens":0,"cache_read_input_tokens":8404}}}\n\n'
    b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\n\n'
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"OK"}}\n\n'
    b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":4}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _core():
    model = SimpleNamespace(
        model_tag=OPUS, provider="anthropic", provider_model_id=OPUS, display_name="Claude Opus 4.8",
        capabilities={"tools": True, "vision": True}, max_output_tokens=65536,
    )
    router = MagicMock()
    router.get_model = AsyncMock(return_value=model)
    router.resolve_meta_model = AsyncMock(side_effect=lambda tag, op: (tag, None))
    pricing = MagicMock()
    pricing.get_pricing = AsyncMock(return_value=SimpleNamespace(
        input_cu_rate=Decimal("1.5"), output_cu_rate=Decimal("7.5"), rate_unit="per_1m_tokens",
        markup_multiplier=Decimal("3.0"),
    ))
    usage = MagicMock()
    usage.record_usage = AsyncMock()
    usage.redis = None
    billing = MagicMock()
    billing.can_make_request_cached = AsyncMock(return_value=(True, None, True))
    billing.is_wallet_debit_active = AsyncMock(return_value=True)
    billing.debit_wallet_usage = AsyncMock()
    core = MessagesV2Core(router, usage, pricing, billing)
    return core, usage, billing


def _api_key():
    return SimpleNamespace(user_id=1, id=2)


def _mock_backend(monkeypatch, body: bytes, status_code: int = 200,
                  content_type: str = "application/json", streaming: bool = False):
    real_client = httpx.AsyncClient  # capture before patching to avoid recursion

    async def _aiter():
        yield body

    def handler(request: httpx.Request) -> httpx.Response:
        # An async-iterator content makes a streaming response on the async
        # client (aiter_raw works); bytes content is buffered (for .post()).
        content = _aiter() if streaming else body
        return httpx.Response(status_code, content=content, headers={"content-type": content_type})

    def factory(*a, **k):
        return real_client(transport=httpx.MockTransport(handler), timeout=5)

    monkeypatch.setattr(anthropic_edge.httpx, "AsyncClient", factory)


async def _drain(streaming_response) -> bytes:
    out = bytearray()
    async for part in streaming_response.body_iterator:
        out += part if isinstance(part, (bytes, bytearray)) else part.encode()
    return bytes(out)


@pytest.mark.asyncio
async def test_nonstreaming_passthrough_and_usage(monkeypatch):
    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, billing = _core()
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())

    core._post_billing = AsyncMock()
    resp = await core.handle(req, _api_key(), {}, "trace-1")

    assert resp.status_code == 200
    assert resp.body == NONSTREAM_JSON  # byte-faithful
    usage.record_usage.assert_awaited_once()
    kw = usage.record_usage.await_args.kwargs
    assert kw["provider"] == "anthropic"
    assert kw["input_tokens"] == 8414  # 10 + 8404 cache_read
    assert kw["output_tokens"] == 4
    assert kw["compute_units"] > 0
    assert kw["status"] == "success"
    # default surface tag = canonical /v1/messages; the /v2 alias mount passes
    # surface="v2_messages" so alias traffic stays sliceable in usage_logs
    assert kw["cc_metadata"]["api"] == "v1_messages"
    # honest wallet lifecycle: recorded as pending, resolved by _post_billing
    assert kw["wallet_debit_status"] == "pending"
    await asyncio.sleep(0.05)  # let the fire-and-forget task run
    core._post_billing.assert_awaited_once()  # CU>0 -> post-billing fired


@pytest.mark.asyncio
async def test_streaming_byte_faithful_and_cache_usage(monkeypatch):
    _mock_backend(monkeypatch, STREAM_SSE, content_type="text/event-stream", streaming=True)
    core, usage, _ = _core()
    req = validate(json.dumps({"model": OPUS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-2")
    out = await _drain(resp)

    # Every canned SSE event survives byte-faithfully (heartbeats may interleave
    # but never mutate the provider's bytes).
    for event in (b"message_start", b"content_block_delta", b'"text":"OK"', b"message_stop"):
        assert event in out
    usage.record_usage.assert_awaited_once()
    kw = usage.record_usage.await_args.kwargs
    assert kw["input_tokens"] == 8414
    assert kw["output_tokens"] == 4  # from message_delta
    assert kw["compute_units"] > 0


@pytest.mark.asyncio
async def test_non_served_model_returns_anthropic_error(monkeypatch):
    # Under the wildcard allowlist, a resolvable model on an edgeless provider
    # (or a non-chat operation) is still rejected with the Anthropic envelope.
    # (bedrock gained its edge — "mistral" stands in as the edgeless provider.)
    core, usage, _ = _core()
    core._router.get_model = AsyncMock(return_value=SimpleNamespace(
        provider="mistral", provider_model_id="mistral-large", model_tag="mistral-large",
        display_name="x", capabilities={"operation": "chat"}, max_output_tokens=4096,
    ))
    req = validate(json.dumps({"model": "mistral-large", "messages": [{"role": "user", "content": "hi"}]}).encode())
    resp = await core.handle(req, _api_key(), {}, "trace-3")
    assert resp.status_code == 404
    err = json.loads(resp.body)
    assert err["type"] == "error" and err["error"]["type"] == "not_found_error"
    usage.record_usage.assert_not_awaited()  # rejected before any provider call


@pytest.mark.asyncio
async def test_provider_4xx_surfaces_faithfully(monkeypatch):
    err_body = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}).encode()
    _mock_backend(monkeypatch, err_body, status_code=400)
    core, usage, billing = _core()
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())
    core._post_billing = AsyncMock()
    resp = await core.handle(req, _api_key(), {}, "trace-4")
    assert resp.status_code == 400
    assert resp.body == err_body  # provider error forwarded unchanged
    # error -> usage logged as error, no debit
    assert usage.record_usage.await_args.kwargs["status"] == "error"
    await asyncio.sleep(0.05)
    core._post_billing.assert_not_awaited()


def test_validate_rejects_bad_json_and_missing_fields():
    with pytest.raises(InvalidMessagesRequest):
        validate(b"{not json")
    with pytest.raises(InvalidMessagesRequest):
        validate(json.dumps({"messages": []}).encode())  # missing model
    with pytest.raises(InvalidMessagesRequest):
        validate(json.dumps({"model": OPUS}).encode())  # missing messages


# ── admission parity with /v1/chat ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_billing_blocked_key_gets_permission_error(monkeypatch):
    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, billing = _core()
    billing.can_make_request_cached = AsyncMock(return_value=(False, "wallet empty", True))
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())
    resp = await core.handle(req, _api_key(), {}, "trace-adm1")
    assert resp.status_code == 403
    err = json.loads(resp.body)
    assert err["error"]["type"] == "permission_error" and "wallet empty" in err["error"]["message"]
    usage.record_usage.assert_not_awaited()  # rejected before the provider call


@pytest.mark.asyncio
async def test_rate_limited_key_gets_429(monkeypatch):
    from mlpal_assistants_service.core.exceptions import RateLimitExceededError

    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, _ = _core()
    limiter = MagicMock()
    limiter.check_request_limit = AsyncMock(side_effect=RateLimitExceededError("too fast"))
    core._rate_limiter = limiter
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())
    resp = await core.handle(req, _api_key(), {}, "trace-adm2")
    assert resp.status_code == 429
    assert json.loads(resp.body)["error"]["type"] == "rate_limit_error"
    usage.record_usage.assert_not_awaited()


@pytest.mark.asyncio
async def test_key_policy_enforced(monkeypatch):
    from mlpal_assistants_service.core.exceptions import BudgetExceededError, ModelAccessDeniedError

    _mock_backend(monkeypatch, NONSTREAM_JSON)
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())

    core, usage, _ = _core()
    policy = MagicMock()
    policy.check_model_access = MagicMock(side_effect=ModelAccessDeniedError(OPUS, "denied by key policy"))
    policy.check_budgets = AsyncMock()
    core._policy = policy
    resp = await core.handle(req, _api_key(), {}, "trace-adm3")
    assert resp.status_code == 403

    core, usage, _ = _core()
    policy = MagicMock()
    policy.check_model_access = MagicMock()
    policy.check_budgets = AsyncMock(side_effect=BudgetExceededError(
        window="monthly", unit="cu", limit=1.0, spent=1.2, reset_at=None))
    core._policy = policy
    resp = await core.handle(req, _api_key(), {}, "trace-adm4")
    assert resp.status_code == 403
    usage.record_usage.assert_not_awaited()


# ── wallet_debit_status lifecycle in _post_billing ───────────────────────────


def _post_billing_harness(monkeypatch, gate):
    """Run _post_billing with the seams stubbed: fresh-session factory yields a
    mock session, build_billing_gate returns `gate`, UsageService is captured."""
    import contextlib

    import mlpal_assistants_service.db.session as db_session
    import mlpal_assistants_service.services.messages_v2.core as core_mod
    import mlpal_assistants_service.services.usage as usage_mod

    bg_session = MagicMock()
    bg_session.commit = AsyncMock()

    @contextlib.asynccontextmanager
    async def factory():
        yield bg_session

    bg_usage = MagicMock()
    bg_usage.mark_wallet_debit_status = AsyncMock()

    monkeypatch.setattr(db_session, "async_session_factory", factory)
    monkeypatch.setattr(core_mod, "build_billing_gate", lambda s, r: gate)
    monkeypatch.setattr(usage_mod, "UsageService", MagicMock(return_value=bg_usage))
    return bg_usage


def _ctx():
    return SimpleNamespace(api_key=SimpleNamespace(user_id=1, id=2, budgets=None), trace_id="tr-x")


@pytest.mark.asyncio
async def test_post_billing_gating_off_marks_not_applicable(monkeypatch):
    gate = MagicMock()
    gate.is_wallet_debit_active = AsyncMock(return_value=False)
    gate.debit_wallet_usage = AsyncMock()
    core, _, _ = _core()
    bg_usage = _post_billing_harness(monkeypatch, gate)

    await core._post_billing(_ctx(), Decimal("0.001"), 10)

    gate.debit_wallet_usage.assert_not_awaited()  # no double-count with gating off
    bg_usage.mark_wallet_debit_status.assert_awaited_once_with("tr-x", "not_applicable")


@pytest.mark.asyncio
async def test_post_billing_success_marks_debited(monkeypatch):
    gate = MagicMock()
    gate.is_wallet_debit_active = AsyncMock(return_value=True)
    gate.debit_wallet_usage = AsyncMock()
    core, _, _ = _core()
    bg_usage = _post_billing_harness(monkeypatch, gate)

    await core._post_billing(_ctx(), Decimal("0.001"), 10)

    gate.debit_wallet_usage.assert_awaited_once()
    bg_usage.mark_wallet_debit_status.assert_awaited_once_with("tr-x", "debited")


@pytest.mark.asyncio
async def test_post_billing_failed_debit_is_retryable(monkeypatch):
    gate = MagicMock()
    gate.is_wallet_debit_active = AsyncMock(return_value=True)
    gate.debit_wallet_usage = AsyncMock(side_effect=RuntimeError("payments down"))
    core, _, _ = _core()
    bg_usage = _post_billing_harness(monkeypatch, gate)

    await core._post_billing(_ctx(), Decimal("0.001"), 10)

    # failed debits enter the retryable state so DebitRetryWorker can replay them
    args = bg_usage.mark_wallet_debit_status.await_args
    assert args.args[0] == "tr-x" and args.args[1] == "failed_retryable"


# ── backend-audit regression tests (2026-08-11) ──────────────────────────────
# The backend ledger consumes usage_logs.compute_units verbatim and clients
# read X-MLPal-Compute-Units — these pin that the two can never disagree, and
# that the alias-drain tag actually distinguishes the mounts.


@pytest.mark.asyncio
async def test_cu_header_equals_logged_compute_units(monkeypatch):
    """Invariant: the header a client reads IS the figure the ledger records."""
    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, _ = _core()
    core._post_billing = AsyncMock()
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-hdr")

    header = resp.headers.get("x-mlpal-compute-units")
    assert header is not None
    logged = usage.record_usage.await_args.kwargs["compute_units"]
    assert Decimal(header) == logged
    assert logged > 0


@pytest.mark.asyncio
async def test_v2_alias_surface_tag_reaches_usage_row(monkeypatch):
    """The deprecated-alias mount tags rows v2_messages — the drain query's
    entire basis. (The v1_messages default is covered above.)"""
    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, _ = _core()
    core._post_billing = AsyncMock()
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())

    await core.handle(req, _api_key(), {}, "trace-alias", surface="v2_messages")

    assert usage.record_usage.await_args.kwargs["cc_metadata"]["api"] == "v2_messages"


def test_surface_for_path():
    from mlpal_assistants_service.api.v2.messages import surface_for_path

    assert surface_for_path("/v1/messages") == "v1_messages"
    assert surface_for_path("/v2/messages") == "v2_messages"
    # No proxy prefix rewriting in prod (host-based ingress) — a rewritten
    # path would silently mistag; this documents the assumption.
    assert surface_for_path("/gateway/v2/messages") == "v1_messages"


# ── streaming capture + cache observability (yodex gap, 2026-08-11) ──────────


@pytest.mark.asyncio
async def test_streaming_spawns_capture_with_accumulated_sse(monkeypatch):
    """Streaming clients (yodex streams everything) must produce payloads when
    capture is on — previously only the non-streaming path spawned capture."""
    import mlpal_assistants_service.services.messages_v2.core as core_mod

    captured = {}

    async def fake_capture(trace_id, request_body, response_body, redis, **kw):
        captured["trace_id"] = trace_id
        captured["response"] = response_body
        captured["kw"] = kw

    monkeypatch.setattr(core_mod, "_capture_v2", fake_capture)
    _mock_backend(monkeypatch, STREAM_SSE, content_type="text/event-stream", streaming=True)
    core, _, _ = _core()
    core._post_billing = AsyncMock()
    req = validate(json.dumps({"model": OPUS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode())

    resp = await core.handle(req, _api_key(), {}, "trace-cap")
    await _drain(resp)
    await asyncio.sleep(0.05)  # let the fire-and-forget capture task run

    assert captured.get("trace_id") == "trace-cap"
    body = captured["response"]
    assert b"message_start" in body and b"message_stop" in body
    assert b": ping" not in body  # heartbeats are transport, not payload


@pytest.mark.asyncio
async def test_cache_stats_persisted_in_metadata(monkeypatch):
    """cache_read/cache_write survive into the usage row so prompt-cache
    effectiveness is queryable (they were previously computed then dropped)."""
    _mock_backend(monkeypatch, NONSTREAM_JSON)
    core, usage, _ = _core()
    core._post_billing = AsyncMock()
    req = validate(json.dumps({"model": OPUS, "messages": [{"role": "user", "content": "hi"}]}).encode())

    await core.handle(req, _api_key(), {}, "trace-cache")

    meta = usage.record_usage.await_args.kwargs["cc_metadata"]
    assert meta["cache_read_input_tokens"] == 8404  # from the canned usage
    assert meta["cache_creation_input_tokens"] == 0
