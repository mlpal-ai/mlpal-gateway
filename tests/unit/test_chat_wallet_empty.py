"""Regression: an empty wallet on the chat wire must surface as WalletEmptyError
(→ 402 wallet_empty), not as an UnboundLocalError-turned-500. The generic
exception handler in _chat_once referenced the tenant-connection variable,
which was only assigned after the billing gate (prod, 2026-09-29)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.core.exceptions import WalletEmptyError
from mlpal_assistants_service.repositories.billing_repository import WALLET_EMPTY_MESSAGE
from mlpal_assistants_service.services.chat import ChatService


def _svc() -> ChatService:
    svc = ChatService.__new__(ChatService)
    svc._rate_limiter = None
    svc._billing = MagicMock(can_make_request_cached=AsyncMock(return_value=(False, WALLET_EMPTY_MESSAGE, True)))
    svc._record_failure = AsyncMock()
    svc._fire_and_forget = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_wallet_empty_before_the_gate_is_a_402_not_a_500():
    req = MagicMock(model="claude-haiku-4-5-20251001")
    with pytest.raises(WalletEmptyError, match="top up"):
        await _svc()._chat_once(1, 2, req, "standard", None, None)


@pytest.mark.asyncio
async def test_wallet_empty_on_the_stream_path_too():
    req = MagicMock(model="claude-haiku-4-5-20251001")
    with pytest.raises(WalletEmptyError):
        async for _ in _svc()._chat_stream_once(1, 2, req, "standard", None, None):
            pass
