"""API Key management endpoints.

All key management requires JWT authentication from the MLpal platform.
Users must be logged in via Cognito to create/manage API keys.

Two key kinds are issued from this service:
  * `mlpal_sk_*` — user-facing keys (broad permissions, long-lived)
  * `cde_sk_*`   — CDE-pod-scoped keys (narrow permissions, ephemeral)
Both live in the same `assistants.api_keys` table — only the prefix
and default permissions differ. The CLI / dashboard list both via
`GET /v1/keys?kind=cde|user` (default = all).
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from mlpal_assistants_service.api.deps import (
    APIKeyServiceDep,
    BillingRepositoryDep,
    ManagementPrincipal,
    UsageServiceDep,
)
from mlpal_assistants_service.core.exceptions import ValidationError
from mlpal_assistants_service.core.security import CDE_API_KEY_PREFIX
from mlpal_assistants_service.schemas.api_key import (
    APIKeyCreate,
    APIKeyList,
    APIKeyResponse,
    APIKeyUpdate,
    APIKeyWithSecret,
    KeyUsageSummary,
)
from mlpal_assistants_service.schemas.usage import DailyUsageResponse

router = APIRouter()


# Map the user-facing ?kind= filter to internal key-prefix substrings.
# Keeping the mapping in one spot so docs and queries can't drift.
_KIND_TO_PREFIX = {
    "user": "mlpal_sk_",
    "cde": CDE_API_KEY_PREFIX,
}


@router.post(
    "",
    response_model=APIKeyWithSecret,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key. Requires JWT auth from MLpal platform.",
)
async def create_api_key(
    body: APIKeyCreate,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
) -> APIKeyWithSecret:
    """
    Create a new API key.

    Authentication: Bearer token (JWT from Cognito)

    The secret key is only returned once during creation.
    Store it securely - it cannot be retrieved later.
    """
    api_key, secret = await api_key_service.create_key(
        user_id=current_user.id,
        data=body,
    )

    return APIKeyWithSecret(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        rate_limit_tier=api_key.rate_limit_tier,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        model_policy=api_key.model_policy,
        budgets=api_key.budgets,
        capture_policy=api_key.capture_policy,
        secret=secret,
    )


async def _billing_paused(billing_repository, user_id: int) -> bool:
    """User-level billing pause, derived per response. Fails open: a billing
    lookup problem must never break key management."""
    wallet_paused = getattr(billing_repository, "wallet_paused", None)
    if wallet_paused is None:  # OSS local gate: no wallet, never paused
        return False
    try:
        return bool(await wallet_paused(user_id))
    except Exception:  # noqa: BLE001
        return False


@router.get(
    "",
    response_model=APIKeyList,
    summary="List API keys",
    description="List all API keys for the current user.",
)
async def list_api_keys(
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
    billing_repository: BillingRepositoryDep,
    include_revoked: bool = False,
    kind: Literal["user", "cde"] | None = Query(
        default=None,
        description=(
            "Filter by key kind. `user` = standard `mlpal_sk_*` keys; "
            "`cde` = CDE-pod-scoped `cde_sk_*` keys. Omit for all."
        ),
    ),
) -> APIKeyList:
    """List all API keys for the authenticated user, optionally
    filtered to one key kind."""
    prefix_filter = _KIND_TO_PREFIX.get(kind) if kind else None
    keys = await api_key_service.list_keys(
        user_id=current_user.id,
        include_revoked=include_revoked,
        prefix_filter=prefix_filter,
    )
    paused = await _billing_paused(billing_repository, current_user.id)

    return APIKeyList(
        items=[
            APIKeyResponse(
                id=k.id,
                name=k.name,
                description=k.description,
                key_prefix=k.key_prefix,
                permissions=k.permissions,
                rate_limit_tier=k.rate_limit_tier,
                is_active=k.is_active,
                last_used_at=k.last_used_at,
                expires_at=k.expires_at,
                created_at=k.created_at,
                model_policy=k.model_policy,
                budgets=k.budgets,
                capture_policy=k.capture_policy,
                paused=paused,
                paused_reason="insufficient_balance" if paused else None,
            )
            for k in keys
        ],
        total=len(keys),
    )


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Get API key",
    description="Get details of a specific API key.",
)
async def get_api_key(
    key_id: int,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
    billing_repository: BillingRepositoryDep,
) -> APIKeyResponse:
    """Get details of a specific API key."""
    api_key = await api_key_service.get_key_by_id(key_id, current_user.id)

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    paused = await _billing_paused(billing_repository, current_user.id)
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        rate_limit_tier=api_key.rate_limit_tier,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        model_policy=api_key.model_policy,
        budgets=api_key.budgets,
        capture_policy=api_key.capture_policy,
        paused=paused,
        paused_reason="insufficient_balance" if paused else None,
    )


@router.patch(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="Update API key policy",
    description="Replace a key's model access policy and/or spend budgets.",
)
async def update_api_key_policy(
    key_id: int,
    body: APIKeyUpdate,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
) -> APIKeyResponse:
    """Update the model_policy and/or budgets of an existing key.

    Only the fields present in the body are changed; the auth cache is
    invalidated so the new policy takes effect within one request, not one
    cache TTL. Pass `{"model_policy": {"allow": ["*"]}}` to widen, or
    `{"budgets": []}` to clear budgets."""
    fields = body.model_dump(exclude_unset=True)
    updates: dict = {}
    if "model_policy" in fields:
        updates["model_policy"] = (
            body.model_policy.model_dump() if body.model_policy else None
        )
    if "budgets" in fields:
        updates["budgets"] = (
            [b.model_dump() for b in body.budgets] if body.budgets else None
        )
    if "capture_policy" in fields:
        updates["capture_policy"] = (
            body.capture_policy.model_dump(exclude_none=True)
            if body.capture_policy
            else None
        )
    if "rate_limit_tier" in fields and body.rate_limit_tier is not None:
        updates["rate_limit_tier"] = body.rate_limit_tier
    if "is_active" in fields and body.is_active is not None:
        updates["is_active"] = body.is_active

    api_key = await api_key_service.update_key_policy(key_id, current_user.id, updates)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        rate_limit_tier=api_key.rate_limit_tier,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        model_policy=api_key.model_policy,
        budgets=api_key.budgets,
        capture_policy=api_key.capture_policy,
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke (deactivate) an API key.",
)
async def revoke_api_key(
    key_id: int,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
) -> None:
    """
    Revoke an API key.

    This permanently deactivates the key. It cannot be reactivated.
    Works for both `mlpal_sk_*` and `cde_sk_*` keys — they live in the
    same table.
    """
    try:
        result = await api_key_service.revoke_key(key_id, current_user.id)
    except ValidationError as e:
        # Already-revoked is a client-state conflict, not a server error.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )


# ---------------------------------------------------------------------------
# CDE-pod-scoped keys
#
# These are minted programmatically by the CDE service when it
# provisions a pod, not by humans. They carry the `messages` permission
# only — a leaked `cde_sk_*` can call /v1/messages but not /v1/chat,
# /v1/embeddings, etc. By default `expires_at` is null (never expires);
# the owning service revokes via DELETE /v1/keys/{id} on pod terminate.
# Rate limits + scoped permissions remain the defenses against abuse of
# a leaked key during its lifetime.
# ---------------------------------------------------------------------------

_CDE_KEY_PERMISSIONS = ["messages"]


@router.post(
    "/cde",
    response_model=APIKeyWithSecret,
    status_code=status.HTTP_201_CREATED,
    summary="Mint CDE-pod-scoped API key",
    description=(
        "Create a CDE-pod-scoped `cde_sk_*` API key on behalf of the "
        "authenticated user. Issued by the CDE service when it "
        "provisions a pod; the secret is injected into the pod env as "
        "ANTHROPIC_AUTH_TOKEN. Defaults to never-expires (set "
        "`expires_at` to opt into expiry); the owning service should "
        "revoke (DELETE /v1/keys/{id}) when the pod terminates. "
        "Pass `tags` (e.g., `{mcp_server_id, cde_id, source}`) so usage "
        "rolls up by entity."
    ),
)
async def create_cde_api_key(
    body: APIKeyCreate,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
) -> APIKeyWithSecret:
    """
    Mint a `cde_sk_*` API key.

    `permissions` is ignored and forced to `["messages"]`. `expires_at`
    is honored if set; null means never-expires (revocation is the kill
    switch). `tags` is honored verbatim.
    """
    # Always override permissions — CDE keys must be narrow-scope, no
    # matter what the caller asks for.
    body.permissions = list(_CDE_KEY_PERMISSIONS)

    api_key, secret = await api_key_service.create_key(
        user_id=current_user.id,
        data=body,
        prefix=CDE_API_KEY_PREFIX,
    )

    return APIKeyWithSecret(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        rate_limit_tier=api_key.rate_limit_tier,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        model_policy=api_key.model_policy,
        budgets=api_key.budgets,
        capture_policy=api_key.capture_policy,
        secret=secret,
    )


@router.get(
    "/{key_id}/usage",
    response_model=KeyUsageSummary,
    summary="Per-key usage summary",
    description=(
        "Aggregate usage_logs for a single API key. Defaults to the "
        "current calendar month; optional start_date / end_date narrow "
        "the window. Returns 404 if the key isn't owned by the caller "
        "(or doesn't exist) — no information leaks across users."
    ),
)
async def get_api_key_usage(
    key_id: int,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
    usage_service: UsageServiceDep,
    start_date: datetime | None = Query(
        default=None,
        description="Aggregation window start (ISO 8601). Defaults to start of current month.",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Aggregation window end (ISO 8601). Defaults to now.",
    ),
) -> KeyUsageSummary:
    """
    Returns aggregated request, token, and compute-unit usage for a key.

    Authorization model: we first call ``api_key_service.get_key_by_id``
    which already scopes the lookup to ``current_user.id``. If that
    returns ``None`` (key doesn't exist OR isn't owned by caller) we 404
    — same response in both cases so we don't leak key-id existence to
    other users.
    """
    api_key = await api_key_service.get_key_by_id(key_id, current_user.id)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    summary = await usage_service.get_key_usage_summary(
        api_key_id=api_key.id,
        start_date=start_date,
        end_date=end_date,
    )

    return KeyUsageSummary(**summary)


@router.get(
    "/{key_id}/usage/daily",
    response_model=DailyUsageResponse,
    summary="Per-key daily usage",
    description=(
        "Daily compute-unit + token breakdown for a single API key over "
        "the requested window (default 30 days, max 90). Same ownership "
        "check as /usage — 404 if the key isn't owned by the caller."
    ),
)
async def get_api_key_usage_daily(
    key_id: int,
    current_user: ManagementPrincipal,
    api_key_service: APIKeyServiceDep,
    usage_service: UsageServiceDep,
    days: int = Query(default=30, ge=1, le=90, description="Number of days"),
) -> DailyUsageResponse:
    """Per-key daily breakdown. Ownership check then aggregation."""
    api_key = await api_key_service.get_key_by_id(key_id, current_user.id)
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    result = await usage_service.get_usage_daily(api_key_id=api_key.id, days=days)
    return DailyUsageResponse(**result)
