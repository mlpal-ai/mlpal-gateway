"""The auth service's identity as a key-management principal (HOP keyring).

Contract (founder, 2026-09-25): accept mlpal_svc_* identities carrying scope
`assistants:hop-keys` on POST /v1/keys (with X-MLPal-Act-As-User), and on
PATCH/DELETE /v1/keys/{id} — ONLY for keys tagged source=hop-keyring. Every
such call is audited. Keys without that tag stay creator-only, and a service
identity cannot even learn they exist (404).
"""

from __future__ import annotations

from datetime import datetime, timezone, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from mlpal_assistants_service.api import deps
from mlpal_assistants_service.api.deps import (
    ServicePrincipal,
    get_key_manager,
    validate_service_identity,
)
from mlpal_assistants_service.api.v1 import keys as keys_api
from mlpal_assistants_service.schemas.api_key import APIKeyCreate, APIKeyUpdate

SETTINGS = SimpleNamespace(auth_service_url="http://auth.test", auth_backend="managed")


def _validate_transport(payload: dict | None, status=200):
    def handler(request):
        assert request.url.path == "/v1/validate"
        assert request.headers["authorization"].startswith("Bearer mlpal_svc_")
        return httpx.Response(status, json=payload or {})
    return httpx.MockTransport(handler)


@pytest.fixture()
def auth_ok(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=_validate_transport(
        {"valid": True, "is_service": True, "service_name": "auth-service",
         "permissions": ["assistants:hop-keys"], "key_id": 9, "user_id": 0})))


@pytest.mark.asyncio
async def test_service_identity_validated_via_auth_service(auth_ok):
    p = await validate_service_identity("mlpal_svc_x", SETTINGS)
    assert isinstance(p, ServicePrincipal) and p.service_name == "auth-service" and p.has_scope("assistants:hop-keys")


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload,expect", [
    (401, {"detail": "nope"}, 401),
    (200, {"valid": True, "is_service": False, "service_name": None, "permissions": ["*"]}, 401),
    (500, {}, 503),
])
async def test_service_identity_rejections(monkeypatch, status, payload, expect):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=_validate_transport(payload, status)))
    with pytest.raises(HTTPException) as ei:
        await validate_service_identity("mlpal_svc_x", SETTINGS)
    assert ei.value.status_code == expect


@pytest.mark.asyncio
async def test_key_manager_requires_scope_and_falls_back_to_users(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=_validate_transport(
        {"valid": True, "is_service": True, "service_name": "auth-service", "permissions": ["other:scope"]})))
    with pytest.raises(HTTPException) as ei:
        await get_key_manager(authorization="Bearer mlpal_svc_x", settings=SETTINGS)
    assert ei.value.status_code == 403
    # a user token goes down the unchanged management path
    called = {}
    async def fake_mgmt(*a):
        called["yes"] = True
        return "user"
    monkeypatch.setattr(deps, "get_management_principal", fake_mgmt)
    assert await get_key_manager(authorization="Bearer mlpal_sk_abc", settings=SETTINGS) == "user" and called


def _svc():
    return ServicePrincipal(service_name="auth-service", scopes=["assistants:hop-keys"], key_id=9)


def _key(id=41, user_id=115, tags=None):
    return SimpleNamespace(id=id, user_id=user_id, tags=tags, name="k", description=None, key_prefix="mlpal_sk_ab…",
                           permissions=["*"], rate_limit_tier="standard", is_active=True, last_used_at=None,
                           expires_at=None, created_at=datetime(2026, 9, 25, tzinfo=UTC), model_policy={"allow": ["claude-opus-5-5"]}, budgets=None,
                           capture_policy=None, revoked_at=None)


@pytest.mark.asyncio
async def test_service_create_requires_hop_tag_and_act_as(monkeypatch):
    svc = MagicMock()
    svc.create_key = AsyncMock(return_value=(_key(tags={"source": "hop-keyring", "hop_id": "h1"}), "secret"))
    body = APIKeyCreate(name="bundle", tags={"source": "hop-keyring", "hop_id": "h1", "hop_key_id": "hk1"})
    with pytest.raises(HTTPException) as ei:
        await keys_api.create_api_key(body, _svc(), svc, act_as_user=None)
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        await keys_api.create_api_key(APIKeyCreate(name="plain"), _svc(), svc, act_as_user=115)
    assert ei.value.status_code == 403 and not svc.create_key.called
    resp = await keys_api.create_api_key(body, _svc(), svc, act_as_user=115)
    assert resp.secret == "secret" and svc.create_key.await_args.kwargs["user_id"] == 115


@pytest.mark.asyncio
async def test_user_cannot_act_as_another_user():
    svc = MagicMock()
    svc.create_key = AsyncMock(return_value=(_key(), "s"))
    user = SimpleNamespace(id=7)
    with pytest.raises(HTTPException) as ei:
        await keys_api.create_api_key(APIKeyCreate(name="x"), user, svc, act_as_user=115)
    assert ei.value.status_code == 403
    await keys_api.create_api_key(APIKeyCreate(name="x"), user, svc, act_as_user=7)  # own id is fine
    assert svc.create_key.await_args.kwargs["user_id"] == 7


@pytest.mark.asyncio
async def test_service_update_and_revoke_confined_to_hop_keys():
    hop = _key(tags={"source": "hop-keyring", "hop_id": "h1"})
    plain = _key(id=42, tags={"source": "console"})
    svc = MagicMock()
    svc.get_hop_keyring_key = AsyncMock(side_effect=lambda kid: hop if kid == 41 else None)
    svc.update_key_policy = AsyncMock(return_value=hop)
    svc.revoke_key = AsyncMock(return_value=hop)
    body = APIKeyUpdate(model_policy={"allow": ["gpt-6-sol"]})
    out = await keys_api.update_api_key_policy(41, body, _svc(), svc)
    assert out.id == 41 and svc.update_key_policy.await_args.args[1] == 115  # owner's id, not the service's
    with pytest.raises(HTTPException) as ei:
        await keys_api.update_api_key_policy(42, body, _svc(), svc)
    assert ei.value.status_code == 404 and svc.update_key_policy.await_count == 1
    await keys_api.revoke_api_key(41, _svc(), svc)
    assert svc.revoke_key.await_args.args == (41, 115)
    with pytest.raises(HTTPException) as ei:
        await keys_api.revoke_api_key(42, _svc(), svc)
    assert ei.value.status_code == 404 and svc.revoke_key.await_count == 1
    assert plain.tags["source"] != "hop-keyring"


@pytest.mark.asyncio
async def test_service_actions_are_audited(monkeypatch):
    events = []
    monkeypatch.setattr(keys_api, "audit", SimpleNamespace(info=lambda ev, **kw: events.append((ev, kw)),
                                                             warning=lambda ev, **kw: events.append((ev, kw))))
    hop = _key(tags={"source": "hop-keyring", "hop_id": "h1", "hop_key_id": "hk1"})
    svc = MagicMock()
    svc.get_hop_keyring_key = AsyncMock(return_value=hop)
    svc.revoke_key = AsyncMock(return_value=hop)
    await keys_api.revoke_api_key(41, _svc(), svc)
    ev, kw = events[-1]
    assert ev == "hop_key.revoke" and kw["actor"] == "auth-service" and kw["owner_user_id"] == 115 and kw["hop_id"] == "h1"
    svc.get_hop_keyring_key = AsyncMock(return_value=None)
    with pytest.raises(HTTPException):
        await keys_api.revoke_api_key(99, _svc(), svc)
    assert events[-1][0] == "hop_key.refused"
