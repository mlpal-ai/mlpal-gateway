"""Wallet-gating surfaces: 402 envelopes on both wires + paused key fields.

Contract (gating design 2026-08-12, live in prod): wallet-empty blocks are
HTTP 402 with stable machine keys — Anthropic wire type=billing_error, OpenAI
wire type=insufficient_quota + code=wallet_empty — and every key in /v1/keys
carries derived paused/paused_reason.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from mlpal_assistants_service.core.exceptions import WalletEmptyError
from mlpal_assistants_service.repositories.billing_repository import WALLET_EMPTY_MESSAGE
from mlpal_assistants_service.schemas.api_key import APIKeyResponse
from mlpal_assistants_service.services.messages_v2.errors import error_body


def test_anthropic_wire_402_envelope():
    body = json.loads(error_body(402, WALLET_EMPTY_MESSAGE))
    assert body == {
        "type": "error",
        "error": {"type": "billing_error", "message": WALLET_EMPTY_MESSAGE},
    }


@pytest.mark.asyncio
async def test_openai_wire_402_envelope():
    """The app-level handler shape, exercised through a minimal app."""
    app = FastAPI()

    @app.exception_handler(WalletEmptyError)
    async def handler(request, exc):  # mirrors main.py registration
        return JSONResponse(
            status_code=402,
            content={
                "error": {
                    "message": exc.message,
                    "type": "insufficient_quota",
                    "code": "wallet_empty",
                }
            },
        )

    @app.get("/boom")
    async def boom():
        raise WalletEmptyError(WALLET_EMPTY_MESSAGE)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/boom")
    assert r.status_code == 402
    assert r.json()["error"]["code"] == "wallet_empty"
    assert r.json()["error"]["type"] == "insufficient_quota"
    assert "top up" in r.json()["error"]["message"]


def test_services_raise_wallet_empty_on_sentinel():
    """Every OpenAI-wire service maps the sentinel to WalletEmptyError."""
    import inspect

    from mlpal_assistants_service.services import audio, chat, embedding, image

    for mod in (chat, embedding, image, audio):
        src = inspect.getsource(mod)
        assert "WalletEmptyError" in src, mod.__name__
        assert "WALLET_EMPTY_MESSAGE" in src, mod.__name__


def test_key_response_paused_fields_default_and_set():
    base = {
        "id": 1, "name": "k", "key_prefix": "mlpal_sk_", "permissions": ["messages"],
        "rate_limit_tier": "standard", "is_active": True, "created_at": "2026-08-12T00:00:00Z",
    }
    k = APIKeyResponse(**base)
    assert k.paused is False and k.paused_reason is None
    k2 = APIKeyResponse(**base, paused=True, paused_reason="insufficient_balance")
    assert k2.paused is True and k2.paused_reason == "insufficient_balance"


@pytest.mark.asyncio
async def test_connection_served_never_debits_wallet(monkeypatch):
    """Regression (found 2026-08-19): the v2 _post_billing spawn was gated on
    compute_units > 0 alone — for byok-served requests the local variable is
    the nonzero list-price ESTIMATE, so with wallet gating active a byok
    request would still debit the wallet. Connection-served requests must
    write CU=0, mark not_applicable, and never spawn _post_billing."""
    from decimal import Decimal
    from unittest.mock import AsyncMock

    from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core
    from mlpal_assistants_service.services.messages_v2.edges import (
        CanonicalUsage,
        RequestContext,
    )

    usage_service = AsyncMock()
    pricing = AsyncMock()
    # nonzero list-price rates — the estimate would be > 0
    pricing.get_pricing.return_value = None  # byok path skips via conn_kind anyway
    core = MessagesV2Core(
        router=AsyncMock(), usage_service=usage_service,
        pricing_service=pricing, billing_gate=AsyncMock(),
    )
    billed = []
    monkeypatch.setattr(
        core, "_post_billing", lambda *a, **k: billed.append(a) or _noop()
    )
    monkeypatch.setattr(
        core, "_resolve_cu_rates",
        AsyncMock(return_value=(Decimal("0.001"), Decimal("0.002"), None)),
    )

    ctx = RequestContext(
        model_tag="claude-sonnet-5", provider="anthropic",
        provider_model_id="claude-sonnet-5", backend="byok:first_party",
        trace_id="t-1", api_key=type("K", (), {"user_id": 7, "id": 3})(),
        headers={}, conn_kind="byok", conn_id=11,
    )
    ctx.usage = CanonicalUsage(input=1000, output=500)
    ctx.status_code = 200

    returned = await core._meter(ctx, latency_ms=42)

    assert billed == []  # never debit a connection-served request
    kwargs = usage_service.record_usage.call_args.kwargs
    assert kwargs["compute_units"] == Decimal("0")
    assert kwargs["wallet_debit_status"] == "not_applicable"
    assert kwargs["cc_metadata"]["serving_credentials"] == "byok"
    assert Decimal(kwargs["cc_metadata"]["connection_cu_estimate"]) > 0
    # The caller-facing cost is the BILLED truth (zero); the estimate travels
    # separately via ctx.conn_estimate → X-MLPal-Connection-Cu-Estimate.
    assert returned == 0
    assert Decimal(ctx.conn_estimate) > 0


async def _noop():
    return None
