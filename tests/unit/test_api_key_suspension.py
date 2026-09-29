"""Suspended keys answer with their reason.

Contract: an operator suspends an account by revoking its keys with
tags.suspended=true (+ suspended_reason). Such a key returns 403 with the
reason on every wire instead of a bare 401 "invalid key", so the holder
knows to contact contact@mlpal.ai. Any other revoked key stays 401.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from mlpal_assistants_service.api.deps import get_current_api_key
from mlpal_assistants_service.core.exceptions import APIKeySuspendedError, InvalidAPIKeyError
from mlpal_assistants_service.services.api_key import APIKeyService


def _service(active_row, inactive_tags):
    svc = APIKeyService.__new__(APIKeyService)
    svc.redis = None
    svc._cache_invalidator = None
    results = [MagicMock(scalar_one_or_none=lambda: active_row), MagicMock(scalar_one_or_none=lambda: inactive_tags)]
    svc.session = MagicMock(execute=AsyncMock(side_effect=results))
    return svc


@pytest.mark.asyncio
async def test_suspended_key_raises_with_reason(monkeypatch):
    monkeypatch.setattr("mlpal_assistants_service.services.api_key.verify_api_key_format", lambda k: True)
    svc = _service(None, {"suspended": True, "suspended_reason": "Account suspended for suspicious activity. Contact contact@mlpal.ai."})
    with pytest.raises(APIKeySuspendedError, match="contact@mlpal.ai"):
        await svc.validate_key("mlpal_sk_" + "a" * 32)


@pytest.mark.asyncio
async def test_plain_revoked_key_stays_invalid(monkeypatch):
    monkeypatch.setattr("mlpal_assistants_service.services.api_key.verify_api_key_format", lambda k: True)
    svc = _service(None, {"source": "console"})
    with pytest.raises(InvalidAPIKeyError):
        await svc.validate_key("mlpal_sk_" + "a" * 32)


@pytest.mark.asyncio
async def test_dependency_maps_suspension_to_403():
    svc = MagicMock(validate_key=AsyncMock(side_effect=APIKeySuspendedError("Account suspended for suspicious activity. Contact contact@mlpal.ai.")))
    with pytest.raises(HTTPException) as ei:
        await get_current_api_key(authorization="Bearer mlpal_sk_x", api_key_service=svc)
    assert ei.value.status_code == 403 and "contact@mlpal.ai" in ei.value.detail
