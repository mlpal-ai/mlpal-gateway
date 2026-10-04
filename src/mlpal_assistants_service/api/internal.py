"""Internal service-to-service endpoints.

Auth: the SAME secret family the gateway uses outbound — a caller presents
either the MSI bearer (``Authorization: Bearer mlpal_svc_*``) matching
``MLPAL_SERVICE_IDENTITY_TOKEN`` or the legacy ``X-Internal-Service-Key``
matching ``INTERNAL_SERVICE_API_KEY``. When neither secret is configured
(self-hosted boxes: wallet gating is managed-only), every route here 404s —
the surface simply doesn't exist.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel

from mlpal_assistants_service.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_internal(
    authorization: str | None, x_internal_service_key: str | None
) -> None:
    settings = get_settings()
    msi = settings.service_identity_token
    legacy = settings.internal_service_api_key
    if not msi and not legacy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if msi and authorization:
        presented = authorization.removeprefix("Bearer ").strip()
        if hmac.compare_digest(presented, msi):
            return
    if legacy and x_internal_service_key and hmac.compare_digest(x_internal_service_key, legacy):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid service credentials")


class WalletCacheInvalidateRequest(BaseModel):
    platform_user_id: int


@router.post(
    "/wallet-cache/invalidate",
    summary="Drop the cached wallet snapshot for a user",
    description=(
        "Called by payments on wallet credits and by the backend debit worker "
        "when a settled debit exhausts a balance — the admission gate then "
        "reflects the new balance on the next request instead of waiting out "
        "the cache TTL. Idempotent; missing snapshots are a no-op."
    ),
)
async def wallet_cache_invalidate(
    body: WalletCacheInvalidateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_internal_service_key: str | None = Header(default=None),
) -> dict:
    _require_internal(authorization, x_internal_service_key)
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="cache unavailable"
        )
    uid = body.platform_user_id
    # One call clears everything the gate caches about the account: wallet
    # snapshot, signup-risk hold, and the young-account trust profile.
    deleted = await redis.delete(f"wallet:{uid}", f"hold:{uid}", f"trust:{uid}")
    logger.info(
        "wallet cache invalidated",
        extra={"user_id": body.platform_user_id, "existed": bool(deleted)},
    )
    return {"invalidated": True, "existed": bool(deleted)}
