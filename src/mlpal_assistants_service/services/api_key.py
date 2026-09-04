"""API Key service for key management."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

if TYPE_CHECKING:
    from mlpal_assistants_service.core.cache import CacheInvalidator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    InvalidAPIKeyError,
    ValidationError,
)
from mlpal_assistants_service.core.security import (
    generate_api_key,
    hash_api_key,
    verify_api_key_format,
)
from mlpal_assistants_service.db.models import APIKey
from mlpal_assistants_service.schemas.api_key import APIKeyCreate

logger = logging.getLogger(__name__)

# Strong refs for fire-and-forget tasks (create_task alone is GC-collectable).
_BACKGROUND_TASKS: set = set()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

# Redis key prefixes
AUTH_CACHE_PREFIX = "auth:"
LAST_USED_PREFIX = "auth_last_used:"


class APIKeyService:
    """Service for API key management."""

    def __init__(
        self,
        session: AsyncSession,
        redis_client: aioredis.Redis | None = None,
        cache_invalidator: CacheInvalidator | None = None,
    ) -> None:
        self.session = session
        self.redis = redis_client
        self._cache_invalidator = cache_invalidator
        settings = get_settings()
        self._cache_ttl = settings.api_key_cache_ttl

    async def create_key(
        self,
        user_id: str,
        data: APIKeyCreate,
        prefix: str | None = None,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.

        Args:
            user_id: The user's ID
            data: Key creation data
            prefix: Optional key prefix. Defaults to settings.api_key_prefix
                (`mlpal_sk_`). Pass `CDE_API_KEY_PREFIX` for CDE-pod-scoped
                keys (ops clarity + en-masse revocation).

        Returns:
            Tuple of (APIKey model, plaintext secret)
            The secret is only returned once - save it!
        """
        # Generate the key (prefix controls both display + the stored hash)
        secret, key_hash, key_prefix = generate_api_key(prefix=prefix)

        # Create the key record
        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=data.name,
            description=data.description,
            permissions=data.permissions,
            rate_limit_tier=getattr(data, "rate_limit_tier", None) or "standard",
            expires_at=data.expires_at,
            tags=data.tags or {},
            model_policy=data.model_policy.model_dump() if data.model_policy else None,
            budgets=[b.model_dump() for b in data.budgets] if data.budgets else None,
            capture_policy=(
                data.capture_policy.model_dump(exclude_none=True)
                if getattr(data, "capture_policy", None)
                else None
            ),
        )

        self.session.add(api_key)
        await self.session.flush()
        await self.session.refresh(api_key)

        return api_key, secret

    async def validate_key(self, api_key: str) -> APIKey:
        """
        Validate an API key and return the key record.

        Uses Redis cache to avoid DB lookup on every request.
        last_used_at is tracked in Redis and flushed to DB periodically.

        Args:
            api_key: The plaintext API key

        Returns:
            The APIKey model if valid

        Raises:
            InvalidAPIKeyError: If key is invalid, revoked, or expired
        """
        # Check format first (fast check, no I/O)
        if not verify_api_key_format(api_key):
            raise InvalidAPIKeyError("Invalid API key format")

        # Hash the key
        key_hash = hash_api_key(api_key)

        # Try Redis cache first
        if self.redis:
            cached = await self._get_cached_key(key_hash)
            if cached is not None:
                # Track last_used_at in Redis — genuinely fire-and-forget:
                # off the hot path, one fewer awaited round trip per request.
                _spawn(self._track_last_used(cached.id))
                return cached

        # Cache miss - query database
        result = await self.session.execute(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True,  # noqa: E712
                APIKey.revoked_at.is_(None),
            )
        )
        key_record = result.scalar_one_or_none()

        if key_record is None:
            raise InvalidAPIKeyError()

        # Check expiration. expires_at is TIMESTAMP WITH TIME ZONE in Postgres,
        # but some drivers/backends (and tests) hand back naive datetimes —
        # treat those as UTC so the comparison never raises
        # `can't compare offset-naive and offset-aware datetimes`.
        expires_at = key_record.expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at < datetime.now(UTC):
                raise InvalidAPIKeyError("API key has expired")

        # Cache in Redis
        if self.redis:
            await self._cache_key(key_hash, key_record)

        # Track last_used_at in Redis instead of DB UPDATE
        if self.redis:
            _spawn(self._track_last_used(key_record.id))
        else:
            # Fallback: direct DB update if no Redis
            await self.session.execute(
                update(APIKey)
                .where(APIKey.id == key_record.id)
                .values(last_used_at=datetime.utcnow())
            )

        return key_record

    async def get_key_by_id(self, key_id: str, user_id: str) -> APIKey | None:
        """Get an API key by ID (must belong to user)."""
        result = await self.session.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_keys(
        self,
        user_id: str,
        include_revoked: bool = False,
        prefix_filter: str | None = None,
    ) -> Sequence[APIKey]:
        """List all API keys for a user.

        Args:
            user_id: The user's ID.
            include_revoked: Also return revoked keys when True.
            prefix_filter: If set, only return keys whose `key_prefix`
                starts with this string. Use `mlpal_sk_` to fetch only
                user-facing keys, `cde_sk_` for CDE pod keys.
        """
        query = select(APIKey).where(APIKey.user_id == user_id)

        if not include_revoked:
            query = query.where(
                APIKey.is_active == True,  # noqa: E712
                APIKey.revoked_at.is_(None),
            )

        if prefix_filter:
            query = query.where(APIKey.key_prefix.startswith(prefix_filter))

        query = query.order_by(APIKey.created_at.desc())
        result = await self.session.execute(query)
        return result.scalars().all()

    async def revoke_key(self, key_id: str, user_id: str) -> APIKey | None:
        """
        Revoke an API key.

        Also invalidates the Redis cache for this key.

        Args:
            key_id: The key ID to revoke
            user_id: The user ID (for authorization)

        Returns:
            The revoked key, or None if not found
        """
        key_record = await self.get_key_by_id(key_id, user_id)

        if key_record is None:
            return None

        if key_record.revoked_at is not None:
            raise ValidationError("Key is already revoked")

        key_record.is_active = False
        key_record.revoked_at = datetime.utcnow()

        await self.session.flush()

        # Invalidate Redis cache
        if self.redis and key_record.key_hash:
            try:
                await self.redis.delete(f"{AUTH_CACHE_PREFIX}{key_record.key_hash}")
            except Exception as e:
                logger.warning(f"Failed to invalidate auth cache: {e}")

        # Notify other instances via Pub/Sub
        if self._cache_invalidator and key_record.key_hash:
            await self._cache_invalidator.publish(f"api_key:{key_record.key_hash}")

        return key_record

    async def update_key_policy(
        self, key_id: str, user_id: str, updates: dict
    ) -> APIKey | None:
        """Replace a key's model_policy and/or budgets (only the keys present in
        `updates`). Invalidates the auth cache so the change is effective within
        one request, mirroring revoke_key's invalidation path."""
        key_record = await self.get_key_by_id(key_id, user_id)
        if key_record is None:
            return None

        if "model_policy" in updates:
            key_record.model_policy = updates["model_policy"]
        if "budgets" in updates:
            key_record.budgets = updates["budgets"]
        if "capture_policy" in updates:
            key_record.capture_policy = updates["capture_policy"]
        if "rate_limit_tier" in updates:
            key_record.rate_limit_tier = updates["rate_limit_tier"]
        if "is_active" in updates:
            key_record.is_active = updates["is_active"]

        await self.session.flush()
        await self.session.refresh(key_record)

        # Invalidate the auth cache (local Redis entry + cross-instance) so the
        # policy that rides the cached key doesn't stay stale for a full TTL.
        if self.redis and key_record.key_hash:
            try:
                await self.redis.delete(f"{AUTH_CACHE_PREFIX}{key_record.key_hash}")
            except Exception as e:
                logger.warning(f"Failed to invalidate auth cache: {e}")
        if self._cache_invalidator and key_record.key_hash:
            await self._cache_invalidator.publish(f"api_key:{key_record.key_hash}")

        return key_record

    # =========================================================================
    # Flush last_used_at from Redis to DB (called periodically)
    # =========================================================================

    async def flush_last_used(self) -> int:
        """
        Flush last_used_at timestamps from Redis to database.

        Should be called periodically (e.g., every 60s) by a background task.

        Returns:
            Number of keys updated
        """
        if not self.redis:
            return 0

        count = 0
        cursor = 0
        now = datetime.utcnow()

        try:
            while True:
                cursor, keys = await self.redis.scan(
                    cursor, match=f"{LAST_USED_PREFIX}*", count=100
                )

                if keys:
                    # Get all key IDs and delete the Redis keys
                    pipe = self.redis.pipeline()
                    for key in keys:
                        pipe.get(key)
                        pipe.delete(key)
                    results = await pipe.execute()

                    # Update DB in batch
                    for i in range(0, len(results), 2):
                        timestamp = results[i]
                        if timestamp:
                            key_name = keys[i // 2]
                            # Extract key_id from "auth_last_used:{key_id}"
                            key_id_str = key_name.decode() if isinstance(key_name, bytes) else key_name
                            key_id = int(key_id_str.replace(LAST_USED_PREFIX, ""))
                            await self.session.execute(
                                update(APIKey)
                                .where(APIKey.id == key_id)
                                .values(last_used_at=now)
                            )
                            count += 1

                if cursor == 0:
                    break

            if count > 0:
                await self.session.commit()
                logger.info(f"Flushed last_used_at for {count} API keys")

        except Exception as e:
            logger.error(f"Error flushing last_used_at: {e}")

        return count

    # =========================================================================
    # Redis Cache Helpers
    # =========================================================================

    async def _get_cached_key(self, key_hash: str) -> APIKey | None:
        """Get API key record from Redis cache."""
        try:
            data = await self.redis.get(f"{AUTH_CACHE_PREFIX}{key_hash}")
            if data is None:
                return None

            obj = json.loads(data)

            # Check expiration (cached value might have expired since caching).
            # The serialized value is timezone-aware; comparing against naive
            # utcnow() raised TypeError on every hit, which the broad except
            # below swallowed — silently disabling the auth cache for every
            # key that had an expiry.
            if obj.get("expires_at"):
                expires = datetime.fromisoformat(obj["expires_at"])
                now = datetime.now(UTC) if expires.tzinfo else datetime.utcnow()
                if expires < now:
                    # Expired - remove from cache
                    await self.redis.delete(f"{AUTH_CACHE_PREFIX}{key_hash}")
                    raise InvalidAPIKeyError("API key has expired")

            # Reconstruct APIKey object
            api_key = APIKey()
            api_key.id = obj["id"]
            api_key.user_id = obj["user_id"]
            api_key.key_hash = obj["key_hash"]
            api_key.key_prefix = obj["key_prefix"]
            api_key.name = obj["name"]
            api_key.permissions = obj["permissions"]
            api_key.rate_limit_tier = obj["rate_limit_tier"]
            api_key.is_active = obj["is_active"]
            # Policy fields ride the auth cache so enforcement needs no extra
            # query on the hot path. .get() keeps old cache entries readable.
            api_key.model_policy = obj.get("model_policy")
            api_key.budgets = obj.get("budgets")

            return api_key

        except InvalidAPIKeyError:
            raise
        except Exception as e:
            logger.warning(f"Redis auth cache get error: {e}")
            return None

    async def _cache_key(self, key_hash: str, key_record: APIKey) -> None:
        """Cache API key record in Redis."""
        try:
            data = {
                "id": key_record.id,
                "user_id": key_record.user_id,
                "key_hash": key_record.key_hash,
                "key_prefix": key_record.key_prefix,
                "name": key_record.name,
                "permissions": key_record.permissions,
                "rate_limit_tier": key_record.rate_limit_tier,
                "is_active": key_record.is_active,
                "expires_at": key_record.expires_at.isoformat() if key_record.expires_at else None,
                "model_policy": key_record.model_policy,
                "budgets": key_record.budgets,
            }
            await self.redis.setex(
                f"{AUTH_CACHE_PREFIX}{key_hash}",
                self._cache_ttl,
                json.dumps(data),
            )
        except Exception as e:
            logger.warning(f"Redis auth cache set error: {e}")

    async def _track_last_used(self, key_id: int) -> None:
        """Track last_used_at in Redis (fire-and-forget)."""
        try:
            await self.redis.set(
                f"{LAST_USED_PREFIX}{key_id}",
                datetime.utcnow().isoformat(),
                ex=3600,  # Expire after 1 hour if not flushed
            )
        except Exception as e:
            logger.warning(f"Redis last_used tracking error: {e}")

    # Note: User management is handled externally in MLPAL_USER_SCHEMA.
    # This service only manages API keys which reference user_id as a foreign key.
