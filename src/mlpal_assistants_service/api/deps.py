"""API dependencies for dependency injection."""

from dataclasses import dataclass
from typing import Annotated, Any

import redis.asyncio as redis
import structlog
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.auth import (
    JWTValidationError,
    get_cognito_validator,
)
from mlpal_assistants_service.core.cache import CacheInvalidator
from mlpal_assistants_service.core.config import Settings, get_settings
from mlpal_assistants_service.core.exceptions import InvalidAPIKeyError, RateLimitExceededError
from mlpal_assistants_service.core.security import is_known_api_key_prefix
from mlpal_assistants_service.core.storage import AssetStorageService
from mlpal_assistants_service.db.models import APIKey
from mlpal_assistants_service.db.session import get_session
from mlpal_assistants_service.repositories.meta_routing_repository import MetaRoutingRepository
from mlpal_assistants_service.services.api_key import APIKeyService
from mlpal_assistants_service.services.audio import AudioService
from mlpal_assistants_service.services.chat import ChatService
from mlpal_assistants_service.services.embedding import EmbeddingService
from mlpal_assistants_service.services.image import ImageService
from mlpal_assistants_service.services.pricing import PricingService
from mlpal_assistants_service.services.rate_limiter import RateLimiter
from mlpal_assistants_service.services.router import ModelRouter
from mlpal_assistants_service.services.usage import UsageService

logger = structlog.get_logger(__name__)


@dataclass
class AuthenticatedUser:
    """User authenticated via JWT (from MLpal platform)."""

    id: int
    email: str
    cognito_sub: str

@dataclass
class ServicePrincipal:
    """A platform service (mlpal_svc_* identity validated by the auth
    service) acting on management endpoints. `scopes` are the identity's
    permissions as the auth service reports them."""

    service_name: str
    scopes: list[str]
    key_id: int | None = None

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes


SERVICE_KEY_PREFIX = "mlpal_svc_"
HOP_KEYS_SCOPE = "assistants:hop-keys"
HOP_KEYRING_SOURCE = "hop-keyring"


# Type aliases for cleaner annotations
SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_redis_client(request: Request) -> redis.Redis | None:
    """Get Redis client from app state."""
    return getattr(request.app.state, "redis", None)


async def get_asset_storage(request: Request) -> AssetStorageService | None:
    """Get asset storage service from app state."""
    return getattr(request.app.state, "asset_storage", None)


async def get_sqs_client(request: Request) -> Any:
    """Get SQS client from app state."""
    return getattr(request.app.state, "sqs_client", None)


async def get_cache_invalidator(request: Request) -> CacheInvalidator | None:
    """Get cache invalidator from app state."""
    return getattr(request.app.state, "cache_invalidator", None)


async def get_shared_caches(request: Request) -> dict:
    """Get shared caches from app state for hot-path optimization."""
    return {
        "model_cache": getattr(request.app.state, "model_cache", None),
        "routing_cache": getattr(request.app.state, "routing_cache", None),
        "pricing_cache": getattr(request.app.state, "pricing_cache", None),
    }


RedisDep = Annotated[redis.Redis | None, Depends(get_redis_client)]
AssetStorageDep = Annotated[AssetStorageService | None, Depends(get_asset_storage)]
SQSClientDep = Annotated[Any, Depends(get_sqs_client)]
CacheInvalidatorDep = Annotated[CacheInvalidator | None, Depends(get_cache_invalidator)]
SharedCachesDep = Annotated[dict, Depends(get_shared_caches)]


def get_api_key_service(
    session: SessionDep,
    redis_client: RedisDep,
    cache_invalidator: CacheInvalidatorDep = None,
) -> APIKeyService:
    """Get API key service."""
    return APIKeyService(session, redis_client, cache_invalidator)


def get_pricing_service(
    session: SessionDep,
    redis_client: RedisDep,
    shared_caches: SharedCachesDep = None,
) -> PricingService:
    """Get pricing service."""
    service = PricingService(session, redis_client)
    if shared_caches and shared_caches.get("pricing_cache"):
        service._local_cache = shared_caches["pricing_cache"]
    return service


def get_usage_service(session: SessionDep, redis_client: RedisDep, sqs_client: SQSClientDep = None) -> UsageService:
    """Get usage service."""
    return UsageService(session, redis_client, sqs_client)


def get_model_router(
    session: SessionDep,
    redis_client: RedisDep,
    shared_caches: SharedCachesDep = None,
) -> ModelRouter:
    """Get model router. Shares background-refreshed caches from app state."""
    router = ModelRouter(session, redis_client)
    # Reuse the shared caches for hot-path optimization
    if shared_caches:
        if shared_caches.get("routing_cache"):
            router._routing_cache = shared_caches["routing_cache"]
        if shared_caches.get("model_cache"):
            router._local_cache = shared_caches["model_cache"]
    return router


def get_chat_service(
    session: SessionDep,
    redis_client: RedisDep,
    sqs_client: SQSClientDep = None,
    shared_caches: SharedCachesDep = None,
) -> ChatService:
    """Get chat service - main orchestrator for chat completions."""
    return ChatService(session, redis_client, sqs_client, shared_caches)


def get_embedding_service(
    session: SessionDep,
    redis_client: RedisDep,
    sqs_client: SQSClientDep = None,
    shared_caches: SharedCachesDep = None,
) -> EmbeddingService:
    """Get embedding service for text embeddings."""
    return EmbeddingService(session, redis_client, sqs_client, shared_caches)


def get_image_service(
    session: SessionDep,
    redis_client: RedisDep,
    asset_storage: AssetStorageDep,
    sqs_client: SQSClientDep = None,
    shared_caches: SharedCachesDep = None,
) -> ImageService:
    """Get image service for image generation."""
    return ImageService(session, redis_client, asset_storage, sqs_client, shared_caches)


def get_audio_service(
    session: SessionDep,
    redis_client: RedisDep,
    asset_storage: AssetStorageDep,
    sqs_client: SQSClientDep = None,
    shared_caches: SharedCachesDep = None,
) -> AudioService:
    """Get audio service for TTS and transcription."""
    return AudioService(session, redis_client, asset_storage, sqs_client, shared_caches)


def get_billing_repository(session: SessionDep, redis_client: RedisDep = None):
    """Billing gate via the seam: managed = the wallet-backed BillingRepository,
    local (OSS) = the no-callout gate. Endpoints already tolerate the local
    gate's missing wallet surface (getattr checks) — constructing the managed
    repository unconditionally made self-hosted boxes issue doomed HTTP
    callouts to the MLPal backend with a 2s timeout per key listing."""
    from mlpal_assistants_service.seams.billing import build_billing_gate

    return build_billing_gate(session, redis_client)


def get_meta_routing_repository(session: SessionDep) -> MetaRoutingRepository:
    """Get meta routing repository for model alias lookups."""
    return MetaRoutingRepository(session)


async def get_rate_limiter(redis_client: RedisDep) -> RateLimiter | None:
    """Get rate limiter (None if Redis not available)."""
    if redis_client is None:
        return None
    return RateLimiter(redis_client)


# Service dependencies
APIKeyServiceDep = Annotated[APIKeyService, Depends(get_api_key_service)]
PricingServiceDep = Annotated[PricingService, Depends(get_pricing_service)]
UsageServiceDep = Annotated[UsageService, Depends(get_usage_service)]
ModelRouterDep = Annotated[ModelRouter, Depends(get_model_router)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
AudioServiceDep = Annotated[AudioService, Depends(get_audio_service)]
BillingRepositoryDep = Annotated[Any, Depends(get_billing_repository)]
MetaRoutingRepositoryDep = Annotated[MetaRoutingRepository, Depends(get_meta_routing_repository)]
RateLimiterDep = Annotated[RateLimiter | None, Depends(get_rate_limiter)]


async def get_current_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    api_key_service: APIKeyServiceDep = None,  # type: ignore
) -> APIKey:
    """
    Dependency to validate and return the current API key.

    Accepts `Authorization: Bearer <key>` (OpenAI-style) or `x-api-key: <key>`
    (Anthropic-style). The Anthropic SDK and Claude Code send x-api-key by
    default — without this, the "drop-in for Anthropic SDK users" surface
    401'd unless the caller knew to switch their SDK to bearer auth.
    """
    if authorization is None and x_api_key:
        authorization = f"Bearer {x_api_key}"
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header (or x-api-key)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = parts[1]

    try:
        return await api_key_service.validate_key(api_key)
    except InvalidAPIKeyError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


CurrentAPIKey = Annotated[APIKey, Depends(get_current_api_key)]


async def check_rate_limit(
    api_key: CurrentAPIKey,
    rate_limiter: RateLimiterDep,
) -> None:
    """
    Dependency to check rate limits.

    Should be added to endpoints that need rate limiting.
    """
    if rate_limiter is None:
        return  # Skip rate limiting if Redis not available

    try:
        await rate_limiter.check_request_limit(
            user_id=api_key.user_id,
            tier=api_key.rate_limit_tier,
        )
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.message,
            headers={"Retry-After": str(e.retry_after or 60)},
        )


RateLimitCheck = Annotated[None, Depends(check_rate_limit)]


async def get_current_user_from_jwt(
    authorization: Annotated[str | None, Header()] = None,
    session: SessionDep = None,  # type: ignore
    settings: SettingsDep = None,  # type: ignore
) -> AuthenticatedUser:
    """
    Authenticate user via JWT token (from MLpal platform).

    This is used for endpoints where users authenticate via the web frontend
    (e.g., creating their first API key).

    The JWT is issued by Cognito after user logs in to MLpal platform.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    # Check if it's an API key (any known prefix: mlpal_sk_ or cde_sk_)
    if is_known_api_key_prefix(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This endpoint requires JWT authentication, not API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Validate JWT with Cognito
        validator = get_cognito_validator()
        claims = validator.validate_token(token)
        jwt_user = validator.extract_user(claims)
    except JWTValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up user in database by cognito_sub. A DB failure here is an
    # outage on our side, not an auth verdict — log it loudly and return
    # 503 so it can never be mistaken for a bad token (or hide as an
    # unlogged 500).
    user_schema = settings.user_schema
    query = text(
        f"SELECT id, email FROM {user_schema}.users WHERE cognito_sub = :cognito_sub"
    )
    try:
        result = await session.execute(query, {"cognito_sub": jwt_user.cognito_sub})
        user = result.fetchone()
    except SQLAlchemyError as e:
        logger.error(
            "User lookup failed during JWT auth",
            cognito_sub=jwt_user.cognito_sub,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication backend temporarily unavailable. Please retry.",
        )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found. Please sign up on the MLpal platform first.",
        )

    return AuthenticatedUser(
        id=user.id,
        email=user.email,
        cognito_sub=jwt_user.cognito_sub,
    )


CurrentUserFromJWT = Annotated[AuthenticatedUser, Depends(get_current_user_from_jwt)]


def _extract_bearer_token(authorization: str) -> str:
    """Extract token from 'Bearer <token>' format."""
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return parts[1]


async def get_current_user_flexible(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    api_key_service: APIKeyServiceDep = None,  # type: ignore
    session: SessionDep = None,  # type: ignore
    settings: SettingsDep = None,  # type: ignore
) -> int:
    """
    Flexible auth that accepts either API key or JWT.
    Returns user_id only (no auth type leakage to business logic).

    Performance optimization:
    - Check token format FIRST (O(1) string prefix check)
    - Only validate as API key if starts with a known issued prefix
      (mlpal_sk_ or cde_sk_)
    - Only validate as JWT otherwise

    Use for read-only catalog endpoints (models) that don't consume compute.
    """
    if authorization is None and x_api_key:
        # Anthropic-style header — same normalization as get_current_api_key.
        authorization = f"Bearer {x_api_key}"
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = _extract_bearer_token(authorization)

    # Fast path: API key detection by prefix
    if is_known_api_key_prefix(token):
        try:
            api_key = await api_key_service.validate_key(token)
            return api_key.user_id
        except InvalidAPIKeyError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Slow path: JWT validation (only for non-API-key tokens)
    try:
        validator = get_cognito_validator()
        claims = validator.validate_token(token)
        jwt_user = validator.extract_user(claims)

        # Look up user in database by cognito_sub
        user_schema = settings.user_schema
        query = text(
            f"SELECT id FROM {user_schema}.users WHERE cognito_sub = :cognito_sub"
        )
        result = await session.execute(query, {"cognito_sub": jwt_user.cognito_sub})
        user = result.fetchone()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found. Please sign up on the MLpal platform first.",
            )

        return user.id

    except JWTValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


CurrentUserFlexible = Annotated[int, Depends(get_current_user_flexible)]


async def get_api_key_model_policy(
    authorization: Annotated[str | None, Header()] = None,
    api_key_service: APIKeyServiceDep = None,  # type: ignore
) -> dict | None:
    """The model_policy of the authenticating API key, or None for JWT/dashboard
    callers (who see the unfiltered catalog). Read-only and never raises — the
    route's CurrentUserFlexible already enforces auth; this only reads policy for
    per-key model-list filtering. validate_key is a Redis cache hit here."""
    if not authorization:
        return None
    token = _extract_bearer_token(authorization)
    if not is_known_api_key_prefix(token):
        return None
    try:
        api_key = await api_key_service.validate_key(token)
    except InvalidAPIKeyError:
        return None
    return api_key.model_policy


APIKeyModelPolicy = Annotated[dict | None, Depends(get_api_key_model_policy)]


async def get_management_principal(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    api_key_service: APIKeyServiceDep = None,  # type: ignore
    session: SessionDep = None,  # type: ignore
    settings: SettingsDep = None,  # type: ignore
) -> AuthenticatedUser:
    """Resolve the caller for MANAGEMENT endpoints (create/manage keys, policies).

    Auth seam:
      * "managed" (default) — Cognito JWT + user-schema lookup (unchanged prod).
      * "local" (OSS) — an API key carrying the 'admin' permission manages the
        box; created keys are owned by that admin key's user_id. No Cognito, no
        users table, so a self-hosted box can manage itself.

    Returns AuthenticatedUser so downstream endpoints (which use `.id`) are
    identical across both modes.
    """
    if authorization is None and x_api_key:
        # Anthropic-style header — same normalization as get_current_api_key.
        authorization = f"Bearer {x_api_key}"
    if getattr(settings, "auth_backend", "managed") == "local":
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = _extract_bearer_token(authorization)
        if not is_known_api_key_prefix(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Management operations require an admin API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            api_key = await api_key_service.validate_key(token)
        except InvalidAPIKeyError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not api_key.has_permission("admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin permission required for management operations",
            )
        return AuthenticatedUser(
            id=api_key.user_id, email="admin@localhost", cognito_sub="local-admin"
        )
    # managed: delegate to the existing Cognito JWT path (behavior unchanged).
    return await get_current_user_from_jwt(authorization, session, settings)


ManagementPrincipal = Annotated[AuthenticatedUser, Depends(get_management_principal)]


async def validate_service_identity(token: str, settings: Settings) -> ServicePrincipal:
    """Validate an inbound mlpal_svc_* token with the auth service
    (POST /v1/validate). Raises HTTPException 401 for an invalid identity and
    503 when the auth service cannot be reached — never a silent pass."""
    import httpx

    if not settings.auth_service_url:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service identities are not configured on this deployment",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            resp = await client.post(
                f"{settings.auth_service_url.rstrip('/')}/v1/validate",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as e:
        logger.error("auth service unreachable for service-identity validation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service identity validation unavailable",
        )
    if resp.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service identity",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if resp.status_code != 200:
        logger.error("auth service validate failed", status=resp.status_code, body=resp.text[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service identity validation failed",
        )
    data = resp.json()
    if not data.get("valid") or not data.get("is_service") or not data.get("service_name"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a service identity",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ServicePrincipal(
        service_name=data["service_name"],
        scopes=list(data.get("permissions") or []),
        key_id=data.get("key_id"),
    )


async def get_key_manager(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    api_key_service: APIKeyServiceDep = None,  # type: ignore
    session: SessionDep = None,  # type: ignore
    settings: SettingsDep = None,  # type: ignore
) -> "AuthenticatedUser | ServicePrincipal":
    """Principal for the key-management routes: the usual management user
    (JWT, or the local admin key), OR a platform service identity
    (mlpal_svc_*) carrying the `assistants:hop-keys` scope. The routes
    themselves confine a service principal to HOP-keyring keys — this
    dependency only establishes who is calling."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(None, 1)[1].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token and token.startswith(SERVICE_KEY_PREFIX):
        principal = await validate_service_identity(token, settings)
        if not principal.has_scope(HOP_KEYS_SCOPE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service identity lacks the {HOP_KEYS_SCOPE} scope",
            )
        return principal
    return await get_management_principal(authorization, x_api_key, api_key_service, session, settings)


KeyManager = Annotated["AuthenticatedUser | ServicePrincipal", Depends(get_key_manager)]
