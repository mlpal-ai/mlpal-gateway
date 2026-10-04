"""Tenant connections management: /v1/connections.

BYOK provider keys and BYOM custom endpoints/models, unified (design doc
planning/designs/connections-byom.md). Write-only custody: the raw key is
forwarded to the custody seam and never persisted or returned by the gateway —
list responses carry last4 only. Every write live-verifies the credential with
a ZERO-TOKEN probe (the provider's models-list endpoint) so the console can
show verified/invalid immediately.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from mlpal_assistants_service.api.deps import (
    CacheInvalidatorDep,
    ManagementPrincipal,
    ModelRouterDep,
)
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.db.models.connections import TenantConnection, TenantModel
from mlpal_assistants_service.seams.egress_guard import EndpointRejected, validate_endpoint
from mlpal_assistants_service.services.connections import (
    BYOM_DIALECTS,
    SUPPORTED,
    USER_TAG_PREFIX,
    USER_TAG_RE,
    invalidate_overlay,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ProviderKeyUpsert(BaseModel):
    api_key: str = Field(..., min_length=8, max_length=4000)
    backend: str = Field(default="first_party")
    fallback: str = Field(
        default="mlpal",
        pattern="^(mlpal|none)$",
        description=(
            "If this key fails: 'mlpal' = auto-switch to MLPal keys (billed "
            "to your wallet), 'none' = stop serving this family until fixed."
        ),
    )
    config: dict | None = Field(
        default=None,
        description="Non-secret backend config (azure: {endpoint, deployments?}).",
    )


class EndpointCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    dialect: str = Field(default="openai", description="Wire protocol the endpoint speaks.")
    endpoint: str = Field(..., description="OpenAI-compatible base URL including /v1.")
    api_key: str = Field(default="", max_length=4000, description="Optional bearer key.")


class TenantModelCreate(BaseModel):
    model_tag: str = Field(..., min_length=1, max_length=120)
    provider_model_id: str = Field(..., min_length=1, max_length=256)
    context_length: int = Field(..., gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    input_price_per_m: Decimal = Field(..., ge=0, description="USD per 1M input tokens.")
    output_price_per_m: Decimal = Field(..., ge=0, description="USD per 1M output tokens.")
    capabilities: dict | None = None


class TenantModelPatch(BaseModel):
    context_length: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    input_price_per_m: Decimal | None = Field(default=None, ge=0)
    output_price_per_m: Decimal | None = Field(default=None, ge=0)
    capabilities: dict | None = None
    is_active: bool | None = None


def _conn_out(r: TenantConnection) -> dict:
    return {
        "id": r.id,
        "kind": r.kind,
        "name": r.name,
        "family": r.family,
        "backend": r.backend,
        "last4": r.last4,
        "status": r.status,
        "error": r.error,
        "fallback": r.fallback,
        "config": r.config,
        "updated_at": r.updated_at.isoformat()
        if isinstance(r.updated_at, datetime)
        else r.updated_at,
    }


def _model_out(m: TenantModel, warning: str | None = None) -> dict:
    out = {
        "id": m.id,
        "connection_id": m.connection_id,
        "model_tag": m.model_tag,
        "provider_model_id": m.provider_model_id,
        "operation": m.operation,
        "context_length": m.context_length,
        "max_output_tokens": m.max_output_tokens,
        "input_price_per_m": str(m.input_price_per_m),
        "output_price_per_m": str(m.output_price_per_m),
        "capabilities": m.capabilities,
        "is_active": m.is_active,
        "updated_at": m.updated_at.isoformat()
        if isinstance(m.updated_at, datetime)
        else m.updated_at,
    }
    if warning:
        out["warning"] = warning
    return out


async def _probe_byok(
    family: str, backend: str, api_key: str, config: dict | None
) -> tuple[bool, str | None]:
    """Zero-token key verification against the provider's models surface."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if family == "anthropic":
                r = await client.get(
                    "https://api.anthropic.com/v1/models?limit=1",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                )
            elif family == "openai" and backend == "first_party":
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            elif family == "openai" and backend == "azure":
                endpoint = (config or {}).get("endpoint", "").rstrip("/")
                r = await client.get(
                    f"{endpoint}/openai/v1/models", headers={"api-key": api_key}
                )
            elif family == "google":
                r = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key, "pageSize": 1},
                )
            else:
                return False, f"unsupported target {family}/{backend}"
        if r.status_code == 200:
            return True, None
        if r.status_code in (401, 403):
            return False, f"provider rejected the key ({r.status_code})"
        return False, f"provider returned HTTP {r.status_code}"
    except httpx.HTTPError as e:
        return False, f"could not reach the provider: {type(e).__name__}"


async def _probe_byom(
    endpoint: str, api_key: str
) -> tuple[bool, str | None, list[str]]:
    """Models-list probe of an OpenAI-compatible endpoint. Also returns the
    served model ids (when available) so model registration can warn on
    unknown ids without blocking."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            r = await client.get(f"{endpoint.rstrip('/')}/models", headers=headers)
        if r.status_code == 200:
            try:
                ids = [m.get("id", "") for m in r.json().get("data", [])]
            except Exception:  # noqa: BLE001 — non-JSON 200 still counts as reachable
                ids = []
            return True, None, ids
        if r.status_code in (401, 403):
            return False, f"endpoint rejected the key ({r.status_code})", []
        return False, f"endpoint returned HTTP {r.status_code} on /models", []
    except httpx.HTTPError as e:
        return False, f"could not reach the endpoint: {type(e).__name__}", []


async def _store_secret(user_id: int, name: str, value: str) -> tuple[str, str]:
    from mlpal_assistants_service.seams.custody import build_custody

    ref = await build_custody().store(user_id, name, value)
    return ref, get_settings().custody_driver


async def _publish(cache_invalidator, user_id: int) -> None:
    invalidate_overlay(user_id)
    if cache_invalidator:
        await cache_invalidator.publish("connections")


@router.get("", summary="List your connections (masked)")
async def list_connections(
    principal: ManagementPrincipal, model_router: ModelRouterDep
) -> dict:
    _require_enabled()
    rows = (
        (
            await model_router.session.execute(
                select(TenantConnection).where(TenantConnection.user_id == principal.id)
            )
        )
        .scalars()
        .all()
    )
    return {"data": [_conn_out(r) for r in rows]}


@router.put("/provider/{family}", summary="Add or replace a provider key (BYOK)")
async def upsert_provider_key(
    family: str,
    body: ProviderKeyUpsert,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    backend = body.backend or "first_party"
    if (family, backend) not in SUPPORTED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unsupported target {family}/{backend} — supported: "
            + ", ".join(f"{f}/{b}" for f, b in sorted(SUPPORTED)),
        )
    if backend == "azure":
        endpoint = (body.config or {}).get("endpoint", "")
        if not endpoint.startswith("https://"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="azure backend requires config.endpoint (https://<resource>.services.ai.azure.com)",
            )

    verified, error = await _probe_byok(family, backend, body.api_key, body.config)
    secret_ref, driver = await _store_secret(
        principal.id, f"gateway-conn-{family}-{backend}", body.api_key
    )

    session = model_router.session
    existing = (
        (
            await session.execute(
                select(TenantConnection).where(
                    TenantConnection.user_id == principal.id,
                    TenantConnection.kind == "byok",
                    TenantConnection.family == family,
                    TenantConnection.backend == backend,
                )
            )
        )
        .scalars()
        .first()
    )
    values = {
        "secret_ref": secret_ref,
        "driver": driver,
        "last4": body.api_key[-4:],
        "status": "verified" if verified else "invalid",
        "error": error,
        "fallback": body.fallback,
        "config": body.config,
    }
    if existing:
        old_ref, old_driver = existing.secret_ref, existing.driver
        for k, v in values.items():
            setattr(existing, k, v)
        row = existing
        await _cleanup_replaced_secret(principal.id, old_ref, old_driver, secret_ref)
    else:
        row = TenantConnection(
            user_id=principal.id, kind="byok", family=family, backend=backend, **values
        )
        session.add(row)
    await session.flush()
    await session.refresh(row)
    await session.commit()

    await _publish(cache_invalidator, principal.id)
    logger.info(
        "connection upserted",
        extra={
            "user_id": principal.id,
            "kind": "byok",
            "family": family,
            "backend": backend,
            "status": row.status,
        },
    )
    return _conn_out(row)


@router.post(
    "/endpoints",
    status_code=status.HTTP_201_CREATED,
    summary="Add a custom endpoint (BYOM)",
)
async def create_endpoint(
    body: EndpointCreate,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    if body.dialect not in BYOM_DIALECTS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unsupported dialect {body.dialect!r} — supported: "
            + ", ".join(BYOM_DIALECTS),
        )
    endpoint = body.endpoint.rstrip("/")
    try:
        await validate_endpoint(endpoint)
    except EndpointRejected as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        ) from None

    verified, error, _ = await _probe_byom(endpoint, body.api_key)
    secret_ref, driver = await _store_secret(
        principal.id, f"gateway-conn-byom-{body.name}", body.api_key
    )

    session = model_router.session
    row = TenantConnection(
        user_id=principal.id,
        kind="byom",
        name=body.name,
        family=body.dialect,
        backend="custom",
        secret_ref=secret_ref,
        driver=driver,
        last4=body.api_key[-4:] if body.api_key else "none",
        status="verified" if verified else "invalid",
        error=error,
        fallback="none",  # no catalog equivalent to switch to — always hard-stop
        config={"endpoint": endpoint},
    )
    session.add(row)
    try:
        await session.flush()
    except Exception:  # noqa: BLE001 — unique (user, name) race/duplicate
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"an endpoint named {body.name!r} already exists",
        ) from None
    await session.refresh(row)
    await session.commit()

    await _publish(cache_invalidator, principal.id)
    logger.info(
        "connection created",
        extra={"user_id": principal.id, "kind": "byom", "connection_name": body.name, "status": row.status},
    )
    return _conn_out(row)


@router.post("/{conn_id}/verify", summary="Re-verify a stored credential now")
async def reverify_connection(
    conn_id: int,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    """Re-run the zero-token probe with the stored secret — the console's
    "Check now" action after a credential was auto-flipped invalid or the
    provider had an outage. The secret is revealed internally only, never
    returned."""
    _require_enabled()
    session = model_router.session
    row = await _get_owned(session, principal.id, conn_id)
    from mlpal_assistants_service.seams.custody import build_custody

    api_key = await build_custody().reveal(principal.id, row.secret_ref)
    if row.kind == "byom":
        endpoint = (row.config or {}).get("endpoint", "")
        verified, error, _ = await _probe_byom(endpoint, api_key)
    else:
        verified, error = await _probe_byok(row.family, row.backend, api_key, row.config)
    row.status = "verified" if verified else "invalid"
    row.error = error
    await session.flush()
    await session.refresh(row)
    await session.commit()
    await _publish(cache_invalidator, principal.id)
    return _conn_out(row)


@router.delete("/{conn_id}", summary="Remove a connection")
async def delete_connection(
    conn_id: int,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    session = model_router.session
    row = await _get_owned(session, principal.id, conn_id)
    dependents = (
        (
            await session.execute(
                select(TenantModel.model_tag).where(TenantModel.connection_id == row.id)
            )
        )
        .scalars()
        .all()
    )
    if dependents:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "connection still serves registered models — delete them first: "
                + ", ".join(sorted(dependents))
            ),
        )
    from mlpal_assistants_service.seams.custody import build_custody

    try:
        await build_custody().delete(principal.id, row.secret_ref)
    except Exception:  # noqa: BLE001 — row deletion is the authoritative removal
        logger.warning("connections: custody delete failed", exc_info=True)
    await session.execute(
        sa_delete(TenantConnection).where(TenantConnection.id == row.id)
    )
    await session.commit()
    await _publish(cache_invalidator, principal.id)
    return {"deleted": True, "id": conn_id}


@router.get("/{conn_id}/models", summary="List models registered on a connection")
async def list_tenant_models(
    conn_id: int, principal: ManagementPrincipal, model_router: ModelRouterDep
) -> dict:
    _require_enabled()
    session = model_router.session
    await _get_owned(session, principal.id, conn_id)
    rows = (
        (
            await session.execute(
                select(TenantModel).where(TenantModel.connection_id == conn_id)
            )
        )
        .scalars()
        .all()
    )
    return {"data": [_model_out(m) for m in rows]}


@router.post(
    "/{conn_id}/models",
    status_code=status.HTTP_201_CREATED,
    summary="Register a model on a custom endpoint",
)
async def create_tenant_model(
    conn_id: int,
    body: TenantModelCreate,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    session = model_router.session
    conn = await _get_owned(session, principal.id, conn_id)
    if conn.kind != "byom":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="models can only be registered on custom endpoints (byok keys serve catalog models)",
        )
    tag = body.model_tag
    if not tag.startswith(USER_TAG_PREFIX):
        tag = USER_TAG_PREFIX + tag
    if not USER_TAG_RE.match(tag):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"invalid model tag {tag!r} — must match user/<name> with "
                "lowercase letters, digits, '.', '_' or '-'"
            ),
        )

    # Soft check: warn (never block) when the endpoint's model list is
    # available and doesn't contain the declared wire id.
    warning = None
    from mlpal_assistants_service.seams.custody import build_custody

    try:
        api_key = await build_custody().reveal(principal.id, conn.secret_ref)
        ok, _, ids = await _probe_byom((conn.config or {}).get("endpoint", ""), api_key)
        if ok and ids and body.provider_model_id not in ids:
            warning = (
                f"endpoint's /models does not list {body.provider_model_id!r} — "
                "registered anyway; verify the id if requests fail"
            )
    except Exception:  # noqa: BLE001 — advisory only
        pass

    row = TenantModel(
        user_id=principal.id,
        connection_id=conn.id,
        model_tag=tag,
        provider_model_id=body.provider_model_id,
        context_length=body.context_length,
        max_output_tokens=body.max_output_tokens,
        input_price_per_m=body.input_price_per_m,
        output_price_per_m=body.output_price_per_m,
        capabilities=body.capabilities,
    )
    session.add(row)
    try:
        await session.flush()
    except Exception:  # noqa: BLE001 — unique (user, tag)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"you already have a model tagged {tag!r}",
        ) from None
    await session.refresh(row)
    await session.commit()
    await _publish(cache_invalidator, principal.id)
    logger.info(
        "tenant model registered",
        extra={"user_id": principal.id, "connection_id": conn.id, "model_tag": tag},
    )
    return _model_out(row, warning)


@router.patch("/{conn_id}/models/{tag:path}", summary="Update a registered model")
async def patch_tenant_model(
    conn_id: int,
    tag: str,
    body: TenantModelPatch,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    session = model_router.session
    await _get_owned(session, principal.id, conn_id)
    row = await _get_owned_model(session, principal.id, conn_id, tag)
    for field in (
        "context_length",
        "max_output_tokens",
        "input_price_per_m",
        "output_price_per_m",
        "capabilities",
        "is_active",
    ):
        value = getattr(body, field)
        if value is not None:
            setattr(row, field, value)
    await session.flush()
    await session.refresh(row)
    await session.commit()
    await _publish(cache_invalidator, principal.id)
    return _model_out(row)


@router.delete("/{conn_id}/models/{tag:path}", summary="Remove a registered model")
async def delete_tenant_model(
    conn_id: int,
    tag: str,
    principal: ManagementPrincipal,
    model_router: ModelRouterDep,
    cache_invalidator: CacheInvalidatorDep,
) -> dict:
    _require_enabled()
    session = model_router.session
    await _get_owned(session, principal.id, conn_id)
    row = await _get_owned_model(session, principal.id, conn_id, tag)
    await session.execute(sa_delete(TenantModel).where(TenantModel.id == row.id))
    await session.commit()
    await _publish(cache_invalidator, principal.id)
    return {"deleted": True, "model_tag": row.model_tag}


async def _get_owned(session, user_id: int, conn_id: int) -> TenantConnection:
    row = (
        (
            await session.execute(
                select(TenantConnection).where(
                    TenantConnection.id == conn_id,
                    TenantConnection.user_id == user_id,
                )
            )
        )
        .scalars()
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no such connection"
        )
    return row


async def _get_owned_model(
    session, user_id: int, conn_id: int, tag: str
) -> TenantModel:
    if not tag.startswith(USER_TAG_PREFIX):
        tag = USER_TAG_PREFIX + tag
    row = (
        (
            await session.execute(
                select(TenantModel).where(
                    TenantModel.user_id == user_id,
                    TenantModel.connection_id == conn_id,
                    TenantModel.model_tag == tag,
                )
            )
        )
        .scalars()
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no such model on this connection"
        )
    return row


async def _cleanup_replaced_secret(
    user_id: int, old_ref: str, old_driver: str, new_ref: str
) -> None:
    """Best-effort cleanup of a replaced secret (managed driver only — the
    local driver's ciphertext lives in the row we just overwrote)."""
    if old_driver == "secrets_service" and old_ref != new_ref:
        from mlpal_assistants_service.seams.custody import build_custody

        try:
            await build_custody().delete(user_id, old_ref)
        except Exception:  # noqa: BLE001
            logger.warning("connections: stale secret cleanup failed", exc_info=True)


def _require_enabled() -> None:
    if not get_settings().connections_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connections are not enabled on this deployment",
        )
