"""/internal/wallet-cache/invalidate: auth matrix + cache deletion."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from mlpal_assistants_service.api.internal import router
from mlpal_assistants_service.core.config import get_settings


class _FakeRedis:
    def __init__(self):
        self.deleted: list[str] = []
        self.store = {"wallet:7": "x"}

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return sum(1 for k in keys if self.store.pop(k, None) is not None)


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(router, prefix="/internal")
    a.state.redis = _FakeRedis()
    return a


async def _post(app, headers=None, user_id=7):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(
            "/internal/wallet-cache/invalidate",
            json={"platform_user_id": user_id},
            headers=headers or {},
        )


@pytest.mark.asyncio
async def test_404_when_no_secret_configured(app, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "service_identity_token", None)
    monkeypatch.setattr(s, "internal_service_api_key", None)
    assert (await _post(app)).status_code == 404


@pytest.mark.asyncio
async def test_403_on_wrong_secret(app, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "service_identity_token", "mlpal_svc_right")
    monkeypatch.setattr(s, "internal_service_api_key", None)
    r = await _post(app, {"Authorization": "Bearer mlpal_svc_wrong"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_msi_bearer_deletes_snapshot(app, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "service_identity_token", "mlpal_svc_right")
    monkeypatch.setattr(s, "internal_service_api_key", None)
    r = await _post(app, {"Authorization": "Bearer mlpal_svc_right"})
    assert r.status_code == 200
    assert r.json() == {"invalidated": True, "existed": True}
    # one call clears the wallet snapshot, the hold flag and the trust profile
    assert app.state.redis.deleted == ["wallet:7", "hold:7", "trust:7"]


@pytest.mark.asyncio
async def test_legacy_header_accepted_and_missing_snapshot_is_noop(app, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "service_identity_token", None)
    monkeypatch.setattr(s, "internal_service_api_key", "legacy-key")
    r = await _post(app, {"X-Internal-Service-Key": "legacy-key"}, user_id=99)
    assert r.status_code == 200
    assert r.json() == {"invalidated": True, "existed": False}
