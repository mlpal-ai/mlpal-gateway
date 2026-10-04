"""Internal (service-to-service) endpoints.

Account suspension: the backend's admin action calls these with its service
identity (scope assistants:suspend); on a local/OSS box the admin key does.
Every call is audited.
"""

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from mlpal_assistants_service.api.deps import (
    APIKeyServiceDep,
    ServicePrincipal,
    SuspensionPrincipal,
)

router = APIRouter()
audit = structlog.get_logger("audit.suspensions")

DEFAULT_REASON = (
    "Your account has been suspended due to suspicious activity. Contact contact@mlpal.ai."
)


class SuspendRequest(BaseModel):
    reason: str = Field(default=DEFAULT_REASON, min_length=8, max_length=500)
    incident: str | None = Field(default=None, max_length=100)


class SuspensionResponse(BaseModel):
    user_id: int
    active: bool
    reason: str | None = None
    incident: str | None = None
    suspended_at: str | None = None
    lifted_at: str | None = None


def _actor(principal) -> str:
    if isinstance(principal, ServicePrincipal):
        return f"service:{principal.service_name}"
    return f"user:{principal.id}"


def _resp(user_id: int, row) -> SuspensionResponse:
    if row is None:
        return SuspensionResponse(user_id=user_id, active=False)
    return SuspensionResponse(
        user_id=user_id, active=row.lifted_at is None, reason=row.reason, incident=row.incident,
        suspended_at=row.suspended_at.isoformat() if row.suspended_at else None,
        lifted_at=row.lifted_at.isoformat() if row.lifted_at else None,
    )


@router.post("/users/{user_id}/suspension", response_model=SuspensionResponse, summary="Suspend an account")
async def suspend_user(
    user_id: int, body: SuspendRequest, principal: SuspensionPrincipal, api_key_service: APIKeyServiceDep
) -> SuspensionResponse:
    row = await api_key_service.suspend_user(user_id, body.reason, by=_actor(principal), incident=body.incident)
    audit.info("account.suspend", user_id=user_id, actor=_actor(principal), incident=body.incident, reason=body.reason)
    return _resp(user_id, row)


@router.delete("/users/{user_id}/suspension", response_model=SuspensionResponse, summary="Lift a suspension")
async def lift_suspension(
    user_id: int, principal: SuspensionPrincipal, api_key_service: APIKeyServiceDep
) -> SuspensionResponse:
    row = await api_key_service.lift_suspension(user_id, by=_actor(principal))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active suspension for this user")
    audit.info("account.unsuspend", user_id=user_id, actor=_actor(principal))
    return _resp(user_id, row)


@router.get("/users/{user_id}/suspension", response_model=SuspensionResponse, summary="Suspension status")
async def suspension_status(
    user_id: int, principal: SuspensionPrincipal, api_key_service: APIKeyServiceDep
) -> SuspensionResponse:
    return _resp(user_id, await api_key_service.active_suspension(user_id))


class AbuseEventResponse(BaseModel):
    id: int
    created_at: str
    rule: str
    cluster_key: str
    action: str
    evidence: dict


@router.get("/abuse/events", response_model=list[AbuseEventResponse], summary="Recent abuse-detector findings")
async def abuse_events(
    principal: SuspensionPrincipal, api_key_service: APIKeyServiceDep, limit: int = 50
) -> list[AbuseEventResponse]:
    from sqlalchemy import select

    from mlpal_assistants_service.db.models import AbuseEvent

    rows = await api_key_service.session.execute(
        select(AbuseEvent).order_by(AbuseEvent.created_at.desc()).limit(max(1, min(limit, 500)))
    )
    return [
        AbuseEventResponse(
            id=e.id, created_at=e.created_at.isoformat(), rule=e.rule,
            cluster_key=e.cluster_key, action=e.action, evidence=e.evidence,
        )
        for e in rows.scalars().all()
    ]
