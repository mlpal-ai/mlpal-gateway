"""Account-level suspension enforced by the gateway.

Contract: suspending a USER makes every one of their keys answer 403 with the
reason (keys are not revoked; lifting restores them), purges their keys from
the auth cache on every instance, refuses their management session, and is
callable only by a service identity with `assistants:suspend` (or the local
admin key on an OSS box). Every action is audited.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from mlpal_assistants_service.api import deps
from mlpal_assistants_service.api.deps import ServicePrincipal, get_suspension_principal
from mlpal_assistants_service.api.v1 import internal
from mlpal_assistants_service.core.exceptions import APIKeySuspendedError
from mlpal_assistants_service.services.api_key import APIKeyService


def _svc(key_row, suspension):
    svc = APIKeyService.__new__(APIKeyService)
    svc.redis = None
    svc._cache_invalidator = None
    svc.session = MagicMock(execute=AsyncMock(side_effect=[
        MagicMock(scalar_one_or_none=lambda: key_row),
        MagicMock(scalar_one_or_none=lambda: suspension),
    ]))
    return svc


@pytest.mark.asyncio
async def test_active_key_of_suspended_user_is_refused_with_reason(monkeypatch):
    monkeypatch.setattr("mlpal_assistants_service.services.api_key.verify_api_key_format", lambda k: True)
    key = SimpleNamespace(id=1, user_id=310, expires_at=None)
    susp = SimpleNamespace(reason="Your account has been suspended due to suspicious activity. Contact contact@mlpal.ai.", lifted_at=None)
    with pytest.raises(APIKeySuspendedError, match="contact@mlpal.ai"):
        await _svc(key, susp).validate_key("mlpal_sk_" + "a" * 32)


@pytest.mark.asyncio
async def test_suspend_and_lift_purge_the_users_keys():
    svc = APIKeyService.__new__(APIKeyService)
    svc.redis = MagicMock(delete=AsyncMock())
    svc._cache_invalidator = MagicMock(publish=AsyncMock())
    session = MagicMock(add=MagicMock(), flush=AsyncMock())
    hashes = MagicMock()
    hashes.scalars.return_value.all.return_value = ["h1", "h2"]
    session.execute = AsyncMock(side_effect=[MagicMock(scalar_one_or_none=lambda: None), hashes])
    svc.session = session
    row = await svc.suspend_user(310, "reason text here", by="service:backend", incident="farm")
    assert row.user_id == 310 and row.lifted_at is None and session.add.called
    assert svc.redis.delete.await_count == 2 and svc._cache_invalidator.publish.await_count == 2
    # lift
    row.lifted_at = None
    session.execute = AsyncMock(side_effect=[MagicMock(scalar_one_or_none=lambda: row), hashes])
    lifted = await svc.lift_suspension(310, by="service:backend")
    assert lifted is row and row.lifted_at is not None and row.lifted_by == "service:backend"
    assert svc.redis.delete.await_count == 4


@pytest.mark.asyncio
async def test_principal_requires_suspend_scope_or_local_admin(monkeypatch):
    settings = SimpleNamespace(auth_service_url="http://auth.test", auth_backend="managed")
    monkeypatch.setattr(deps, "validate_service_identity", AsyncMock(return_value=ServicePrincipal("backend", ["assistants:hop-keys"])))
    with pytest.raises(HTTPException) as ei:
        await get_suspension_principal(authorization="Bearer mlpal_svc_x", settings=settings)
    assert ei.value.status_code == 403
    monkeypatch.setattr(deps, "validate_service_identity", AsyncMock(return_value=ServicePrincipal("backend", ["assistants:suspend"])))
    p = await get_suspension_principal(authorization="Bearer mlpal_svc_x", settings=settings)
    assert isinstance(p, ServicePrincipal)
    # a user JWT is never enough in managed mode
    with pytest.raises(HTTPException) as ei:
        await get_suspension_principal(authorization="Bearer eyJ.jwt", settings=settings)
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_routes_audit_and_shape(monkeypatch):
    events = []
    monkeypatch.setattr(internal, "audit", SimpleNamespace(info=lambda ev, **kw: events.append((ev, kw))))
    row = SimpleNamespace(reason="r" * 10, incident="farm", suspended_at=datetime(2026, 9, 29, tzinfo=UTC), lifted_at=None)
    svc = MagicMock(suspend_user=AsyncMock(return_value=row), lift_suspension=AsyncMock(return_value=None), active_suspension=AsyncMock(return_value=None))
    principal = ServicePrincipal("backend", ["assistants:suspend"])
    out = await internal.suspend_user(310, internal.SuspendRequest(reason="r" * 10, incident="farm"), principal, svc)
    assert out.active and out.user_id == 310 and events[-1][0] == "account.suspend" and events[-1][1]["actor"] == "service:backend"
    with pytest.raises(HTTPException) as ei:
        await internal.lift_suspension(310, principal, svc)
    assert ei.value.status_code == 404
    st = await internal.suspension_status(310, principal, svc)
    assert st.active is False and st.reason is None
