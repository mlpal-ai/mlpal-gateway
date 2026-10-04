"""Signup-risk hold on gateway key minting: the console must not be the
bypass for the backend's hold, and deployments without the column are
untouched."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mlpal_assistants_service.api.v1 import keys as keys_module
from mlpal_assistants_service.core.exceptions import AccountHeldError
from mlpal_assistants_service.services import api_key as api_key_module
from mlpal_assistants_service.services.api_key import APIKeyService


@pytest.fixture(autouse=True)
def _reset_probe():
    api_key_module._HOLD_COLUMN_PRESENT = None
    yield
    api_key_module._HOLD_COLUMN_PRESENT = None


def _session(probe_hit: bool, hold: str | None):
    session = MagicMock()
    probe = MagicMock()
    probe.first.return_value = (1,) if probe_hit else None
    row = MagicMock()
    row.scalar_one_or_none.return_value = hold
    session.execute = AsyncMock(side_effect=[probe, row])
    return session


@pytest.mark.asyncio
async def test_hold_reason_reads_column_when_present() -> None:
    svc = APIKeyService(_session(True, "signup risk: hosting ASN"), None)
    assert await svc.hold_reason(42) == "signup risk: hosting ASN"
    assert api_key_module._HOLD_COLUMN_PRESENT is True


@pytest.mark.asyncio
async def test_hold_reason_is_none_when_column_absent_and_probe_is_cached() -> None:
    session = _session(False, "ignored")
    svc = APIKeyService(session, None)
    assert await svc.hold_reason(42) is None
    assert api_key_module._HOLD_COLUMN_PRESENT is False
    assert await svc.hold_reason(43) is None
    assert session.execute.await_count == 1  # probe only, never the row query


@pytest.mark.asyncio
async def test_hold_reason_lookup_failure_never_raises() -> None:
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    assert await APIKeyService(session, None).hold_reason(42) is None


@pytest.mark.asyncio
async def test_refuse_if_held_raises_with_stable_code() -> None:
    svc = MagicMock()
    svc.hold_reason = AsyncMock(return_value="held")
    with pytest.raises(AccountHeldError) as exc:
        await keys_module._refuse_if_held(svc, 7)
    assert exc.value.status_code == 403
    assert exc.value.details["code"] == "risk_hold"
    assert exc.value.message == keys_module.HOLD_MESSAGE
    svc.hold_reason = AsyncMock(return_value=None)
    await keys_module._refuse_if_held(svc, 7)  # no hold: passes silently


@pytest.mark.asyncio
async def test_held_handler_envelope_matches_suspension_shape() -> None:
    from mlpal_assistants_service.main import held_handler

    app = FastAPI()
    app.add_exception_handler(AccountHeldError, held_handler)

    @app.post("/mint")
    async def mint():
        raise AccountHeldError(keys_module.HOLD_MESSAGE)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/mint")
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "risk_hold" and body["message"] == keys_module.HOLD_MESSAGE
    assert body["detail"] == keys_module.HOLD_MESSAGE
    assert body["error"] == {"type": "permission_error", "code": "risk_hold", "message": keys_module.HOLD_MESSAGE}
