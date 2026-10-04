"""Client-controlled model failover (fallback_models) on both wires.

Contract: candidates run the full pipeline and are billed as-served; only
retriable serving failures (5xx/timeout/connection/provider-429, or an
unknown fallback tag) advance the chain; client errors re-raise immediately;
an open stream is committed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.responses import Response, StreamingResponse

from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    ProviderError,
    UnsupportedModelKwargsError,
    WalletEmptyError,
)
from mlpal_assistants_service.services.chat import ChatService
from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core
from mlpal_assistants_service.services.messages_v2.schemas import (
    InvalidMessagesRequest,
    validate,
)

# ── classification ──────────────────────────────────────────────────────────


def test_retriable_classification():
    r = ChatService._retriable
    assert r(ProviderError("boom", provider="openai", status_code=502))
    assert r(ProviderError("rl", provider="openai", status_code=429))
    assert r(ModelNotFoundError("user/nope"))
    # A timeout is terminal since 2026-10-01: the prompt is billed and still
    # generating upstream, so a fallback model would bill it twice.
    assert not r(TimeoutError())
    assert r(ConnectionError())
    assert r(type("APIConnectionError", (Exception,), {})())
    assert not r(ProviderError("bad req", provider="openai", status_code=400))
    assert not r(WalletEmptyError("empty"))
    assert not r(
        UnsupportedModelKwargsError(offending=["x"], allowed=[], provider="openai")
    )


def test_candidates_dedupe_and_cap():
    svc = ChatService.__new__(ChatService)
    # the wire schema caps fallback_models at 3; _candidates dedupes against
    # the primary and re-caps defensively
    req = type(
        "R", (), {"model": "a", "fallback_models": ["b", "a", "c", "d"]}
    )()
    assert svc._candidates(req) == ["a", "b", "c"]


# ── v2 schema extension ─────────────────────────────────────────────────────


def test_v2_validate_pops_fallback_models():
    req = validate(
        b'{"model":"m","messages":[],"fallback_models":["a","b"]}'
    )
    assert req.fallback_models == ["a", "b"]
    assert "fallback_models" not in req.body


def test_v2_validate_rejects_bad_fallback_models():
    with pytest.raises(InvalidMessagesRequest):
        validate(b'{"model":"m","messages":[],"fallback_models":"a"}')
    with pytest.raises(InvalidMessagesRequest):
        validate(b'{"model":"m","messages":[],"fallback_models":["a","b","c","d"]}')


# ── v2 fallback loop ────────────────────────────────────────────────────────


def _core() -> MessagesV2Core:
    return MessagesV2Core(
        router=AsyncMock(), usage_service=AsyncMock(),
        pricing_service=AsyncMock(), billing_gate=AsyncMock(),
    )


def _req(model="primary", fallbacks=("backup",), stream=False):
    fb = "[" + ",".join(f'"{t}"' for t in fallbacks) + "]"
    stream_lit = "true" if stream else "false"
    return validate(
        f'{{"model":"{model}","messages":[],"stream":{stream_lit},"fallback_models":{fb}}}'.encode()
    )


@pytest.mark.asyncio
async def test_v2_fallback_advances_on_5xx_and_marks_response(monkeypatch):
    core = _core()
    attempts = []

    async def fake_handle(req, api_key, headers, trace_id, *, surface):
        attempts.append(req.model)
        assert req.fallback_models is None  # no recursion bombs
        if req.model == "primary":
            return Response(b"overloaded", status_code=529)
        return Response(b"ok", status_code=200)

    monkeypatch.setattr(core, "handle", fake_handle)
    resp = await core._handle_with_fallback(
        _req(), api_key=object(), headers={}, trace_id="t", surface="v1_messages"
    )
    assert attempts == ["primary", "backup"]
    assert resp.status_code == 200
    assert resp.headers["X-MLPal-Fallback-From"] == "primary"


@pytest.mark.asyncio
async def test_v2_fallback_does_not_advance_on_client_error(monkeypatch):
    core = _core()

    async def fake_handle(req, api_key, headers, trace_id, *, surface):
        return Response(b"bad", status_code=400)

    monkeypatch.setattr(core, "handle", fake_handle)
    resp = await core._handle_with_fallback(
        _req(), api_key=object(), headers={}, trace_id="t", surface="v1_messages"
    )
    assert resp.status_code == 400
    assert "X-MLPal-Fallback-From" not in resp.headers  # primary answered


@pytest.mark.asyncio
async def test_v2_fallback_skips_unknown_candidate_and_surfaces_last_failure(monkeypatch):
    core = _core()
    seq = {"primary": 503, "ghost": 404, "backup": 503}
    attempts = []

    async def fake_handle(req, api_key, headers, trace_id, *, surface):
        attempts.append(req.model)
        return Response(b"x", status_code=seq[req.model])

    monkeypatch.setattr(core, "handle", fake_handle)
    resp = await core._handle_with_fallback(
        _req(fallbacks=("ghost", "backup")),
        api_key=object(), headers={}, trace_id="t", surface="v1_messages",
    )
    assert attempts == ["primary", "ghost", "backup"]
    assert resp.status_code == 503  # last real failure, not the ghost 404


@pytest.mark.asyncio
async def test_v2_open_stream_is_committed(monkeypatch):
    core = _core()

    async def _gen():
        yield b"event: message_start\n\n"

    async def fake_handle(req, api_key, headers, trace_id, *, surface):
        return StreamingResponse(_gen(), media_type="text/event-stream")

    monkeypatch.setattr(core, "handle", fake_handle)
    resp = await core._handle_with_fallback(
        _req(stream=True), api_key=object(), headers={}, trace_id="t",
        surface="v1_messages",
    )
    assert isinstance(resp, StreamingResponse)
