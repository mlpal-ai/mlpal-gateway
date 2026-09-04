"""Model router for selecting providers and adapters.

The ModelRouter is responsible for:
- Looking up models from the registry
- Selecting the appropriate provider adapter
- Handling failover to fallback models
- Integrating circuit breakers for resilience
- Caching model information for performance
"""

import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.adapters.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from mlpal_assistants_service.adapters.factory import AdapterFactory, get_adapter_factory
from mlpal_assistants_service.core.cache import TTLCache
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    ModelNotAvailableError,
    ModelNotFoundError,
)
from mlpal_assistants_service.db.models import MetaModelRouting, ModelRegistry
from mlpal_assistants_service.repositories import MetaRoutingRepository, ModelRepository
from mlpal_assistants_service.schemas.routing import (
    RoutingMetadata,
    get_strategy,
    is_meta_model,
)

if TYPE_CHECKING:
    from mlpal_assistants_service.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes requests to the appropriate provider adapter.

    Handles:
    - Model lookup and validation
    - Provider selection via AdapterFactory
    - Circuit breaker integration for resilience
    - Fallback routing when primary provider fails
    - Redis caching for performance

    Usage:
        router = ModelRouter(session, redis_client)

        # Get adapter for a model
        adapter, model_id = await router.get_adapter("gpt-5.2")

        # Get adapter with circuit breaker protection
        adapter, model_id, breaker = await router.get_adapter_with_breaker("gpt-5.2")
        async with breaker:
            response = await adapter.chat(model_id, messages)
    """

    CACHE_PREFIX = "model:"

    def __init__(
        self,
        session: AsyncSession,
        redis_client: redis.Redis | None = None,
        adapter_factory: AdapterFactory | None = None,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
    ) -> None:
        """
        Initialize the ModelRouter.

        Args:
            session: Database session for model lookups
            redis_client: Optional Redis client for caching
            adapter_factory: Optional adapter factory (uses singleton if not provided)
            circuit_breaker_registry: Optional circuit breaker registry
        """
        self.session = session
        self.redis = redis_client
        self._settings = get_settings()
        self._model_repo = ModelRepository(session)
        self._meta_routing_repo = MetaRoutingRepository(session)
        self._adapter_factory = adapter_factory or get_adapter_factory()
        self._circuit_breakers = circuit_breaker_registry or CircuitBreakerRegistry()
        self._local_cache: TTLCache = TTLCache(
            default_ttl=self._settings.local_cache_ttl,
            maxsize=self._settings.local_cache_maxsize,
        )
        self._routing_cache: TTLCache = TTLCache(
            default_ttl=self._settings.local_cache_ttl,
            maxsize=self._settings.local_cache_maxsize,
        )

    async def get_model(self, model_tag: str) -> ModelRegistry:
        """
        Get model information by tag.

        Checks caches (Redis, local) before hitting database.

        Args:
            model_tag: Model identifier (e.g., "gpt-5.2", "claude-opus-4.5")

        Returns:
            ModelRegistry record

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotAvailableError: If model is inactive or deprecated
        """
        # Check local TTL cache first
        cached_model = self._local_cache.get(model_tag)
        if cached_model is not None:
            self._raise_if_paused(model_tag, cached_model)
            if cached_model.is_active and not cached_model.is_deprecated:
                return cached_model

        # Check Redis cache
        if self.redis:
            cached = await self._get_from_redis(model_tag)
            if cached:
                self._raise_if_paused(model_tag, cached)
                # Same gate as the local cache: a retired/inactive row must
                # fall through to the DB path, which raises properly — without
                # this, a retirement only takes effect after the Redis TTL.
                if cached.is_active and not cached.is_deprecated:
                    return cached

        # Query database via repository
        model = await self._model_repo.get_by_tag(model_tag)

        if model is None:
            raise ModelNotFoundError(model_tag)

        if not model.is_active:
            if model.is_deprecated:
                # Retired for good (provider killed it): a permanent 404, not a
                # retryable 503. The deprecation_message/fallback_model_tag are
                # in the registry for surfaces that want to render guidance.
                raise ModelNotFoundError(model_tag)
            raise ModelNotAvailableError(model_tag, "Model is currently unavailable")

        self._raise_if_paused(model_tag, model)

        if model.is_deprecated:
            logger.warning(
                f"Model {model_tag} is deprecated: {model.deprecation_message or 'No message'}"
            )

        # Cache a session-independent copy — never the live ORM instance. Once
        # the loading session closes, a cached live instance is detached and
        # reading an attribute that needs a refresh raises DetachedInstanceError
        # (this took down the OpenAI/meta-model chat paths — see ERRORS.md).
        detached = self._detached_model(model)
        await self._local_cache.set(model_tag, detached)
        if self.redis:
            await self._set_to_redis(model_tag, detached)

        return detached

    async def get_adapter(self, model_tag: str) -> tuple["BaseAdapter", str]:
        """
        Get the appropriate adapter for a model.

        Args:
            model_tag: Model identifier

        Returns:
            Tuple of (adapter, provider_model_id)

        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelNotAvailableError: If no adapter available
        """
        model = await self.get_model(model_tag)
        return self._resolve_backend(model)

    def _resolve_backend(self, model: "ModelRegistry") -> tuple["BaseAdapter", str]:
        """Backend-aware adapter selection: the factory walks the family's
        MLPAL_<FAMILY>_BACKENDS priority list (cached — dict lookup on the
        hot path) and returns the adapter plus the wire model ID for it."""
        try:
            return self._adapter_factory.resolve(model.provider, model.provider_model_id)
        except (ValueError, RuntimeError) as e:
            raise ModelNotAvailableError(
                model.model_tag,
                f"No serving backend available for {model.provider}: {e}",
            )

    async def get_adapter_with_breaker(
        self,
        model_tag: str,
    ) -> tuple["BaseAdapter", str, CircuitBreaker]:
        """
        Get adapter with circuit breaker for resilience.

        Args:
            model_tag: Model identifier

        Returns:
            Tuple of (adapter, provider_model_id, circuit_breaker)

        Usage:
            adapter, model_id, breaker = await router.get_adapter_with_breaker("gpt-5.2")
            async with breaker:
                response = await adapter.chat(model_id, messages)
        """
        model = await self.get_model(model_tag)
        adapter, wire_model_id = self._resolve_backend(model)
        breaker = await self._circuit_breakers.get(model.provider)

        return adapter, wire_model_id, breaker

    async def get_adapter_with_fallback(
        self,
        model_tag: str,
    ) -> tuple["BaseAdapter", str, ModelRegistry]:
        """
        Get adapter with automatic fallback on circuit breaker open.

        If the primary provider's circuit is open, automatically
        routes to the fallback model if configured.

        Args:
            model_tag: Model identifier

        Returns:
            Tuple of (adapter, provider_model_id, model_used)

        Raises:
            ModelNotAvailableError: If no adapter available (including fallback)
        """
        model = await self.get_model(model_tag)
        breaker = await self._circuit_breakers.get(model.provider)

        # Check if primary provider is available
        if not breaker.is_open:
            try:
                adapter, wire_model_id = self._resolve_backend(model)
                return adapter, wire_model_id, model
            except ModelNotAvailableError:
                pass  # Try fallback

        # Primary is unavailable, try fallback
        if model.fallback_model_tag:
            try:
                fallback_model = await self.get_model(model.fallback_model_tag)
                fallback_breaker = await self._circuit_breakers.get(fallback_model.provider)

                if not fallback_breaker.is_open:
                    adapter, wire_model_id = self._resolve_backend(fallback_model)
                    logger.info(
                        f"Using fallback model {fallback_model.model_tag} "
                        f"for {model_tag} (primary circuit open)"
                    )
                    return adapter, wire_model_id, fallback_model

            except (ModelNotFoundError, ModelNotAvailableError, ValueError):
                pass  # Fallback also unavailable

        # No adapter available
        raise ModelNotAvailableError(
            model_tag,
            f"Provider {model.provider} circuit is open and no fallback available",
        )

    async def list_models(
        self,
        provider: str | None = None,
        operation: str | None = None,
        include_deprecated: bool = False,
    ) -> list[ModelRegistry]:
        """
        List available models with optional filtering.

        Uses Redis cache (5 min TTL) to avoid repeated DB queries.

        Args:
            provider: Filter by provider
            operation: Filter by operation (chat, embedding, etc.)
            include_deprecated: Include deprecated models

        Returns:
            List of ModelRegistry records
        """
        # Build cache key based on filter params
        cache_key = f"models_list:{provider or 'all'}:{operation or 'all'}:{include_deprecated}"

        # Check Redis cache
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    items = json.loads(cached)
                    return self._filter_enabled([self._dict_to_model(item) for item in items])
            except Exception as e:
                logger.warning(f"Redis cache get error for model list: {e}")

        # Query database
        if include_deprecated:
            if provider:
                models = await self._model_repo.get_by_provider(provider)
            else:
                models = await self._model_repo.get_all(is_active=True)
            # This branch's repo calls don't take an operation filter — apply
            # it here, or ?capability= silently returns everything (and the
            # wrong result then sits in the Redis cache for the full TTL).
            if operation:
                models = [
                    m
                    for m in models
                    if (m.capabilities or {}).get("operation", "chat") == operation
                ]
        else:
            models = await self._model_repo.get_active_models(
                provider=provider,
                operation=operation,
            )

        # Cache in Redis
        if self.redis and models:
            try:
                data = [self._model_to_dict(m) for m in models]
                await self.redis.setex(cache_key, self._settings.model_cache_ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis cache set error for model list: {e}")

        return self._filter_enabled(models)

    async def get_fallback_model(self, model_tag: str) -> ModelRegistry | None:
        """Get the fallback model for a given model."""
        model = await self.get_model(model_tag)

        if model.fallback_model_tag:
            try:
                return await self.get_model(model.fallback_model_tag)
            except (ModelNotFoundError, ModelNotAvailableError):
                return None

        return None

    def get_circuit_breaker_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        return self._circuit_breakers.get_all_status()

    async def clear_cache(self) -> None:
        """Clear the local model and routing caches."""
        await self._local_cache.clear()
        await self._routing_cache.clear()

    # =========================================================================
    # Meta-Model Resolution
    # =========================================================================

    async def resolve_meta_model(
        self,
        model_tag: str,
        operation: str,
    ) -> tuple[str, RoutingMetadata | None]:
        """
        Resolve a meta-model (mlpal, mlpal-flash, mlpal-lite) to actual model.

        If the model_tag is not a meta-model, returns it unchanged with no metadata.

        Args:
            model_tag: Model identifier (may be meta-model or real model)
            operation: Operation type (chat, image_generation, tts, etc.)

        Returns:
            Tuple of (resolved_model_tag, routing_metadata)
            - For meta-models: (actual_model_tag, RoutingMetadata)
            - For regular models: (model_tag, None)

        Raises:
            ModelNotFoundError: If no routing rule exists for meta-model/operation

        Example:
            >>> resolved, metadata = await router.resolve_meta_model("mlpal", "image_generation")
            >>> print(resolved)  # "gemini-3-pro-image"
            >>> print(metadata.strategy)  # "quality"
        """
        # Check if it's a meta-model
        if not is_meta_model(model_tag):
            return model_tag, None

        # Candidate list from cache (lock-free read via TTLCache) or DB. The
        # SERVED check runs per call, not at cache time — pause state and the
        # model cache change independently of the routing rows.
        cache_key = f"{model_tag}:{operation}"
        candidates = self._routing_cache.get(cache_key)
        if candidates is None:
            rows = await self._meta_routing_repo.get_candidates(model_tag, operation)
            if not rows:
                raise ModelNotFoundError(
                    f"No routing rule for {model_tag}/{operation}. "
                    f"Meta-model '{model_tag}' does not support operation '{operation}'."
                )
            candidates = [self._detached_routing(r) for r in rows]
            await self._routing_cache.set(cache_key, candidates)

        # Availability-aware resolution: first candidate this deployment can
        # actually serve wins (model registered + active + not paused, provider
        # configured). This is what makes router tags work on a one-provider
        # box: the list deliberately spans providers.
        factory = self._adapter_factory
        tried: list[str] = []
        for routing in candidates:
            tag = routing.resolved_model_tag
            try:
                model = await self.get_model(tag)
            except (ModelNotFoundError, ModelNotAvailableError):
                tried.append(f"{tag} (not served)")
                continue
            if getattr(model, "is_paused", False):
                tried.append(f"{tag} (paused)")
                continue
            if not factory.is_enabled(model.provider):
                tried.append(f"{tag} (provider '{model.provider}' not configured)")
                continue
            return self._build_routing_result(model_tag, operation, routing)

        raise ModelNotAvailableError(
            model_tag,
            f"no served model for '{model_tag}'/{operation} on this deployment — "
            f"tried: {', '.join(tried)}. Configure a provider key that serves one "
            f"of these, or request a concrete model tag.",
        )

    def _build_routing_result(
        self,
        meta_model_tag: str,
        operation: str,
        routing: MetaModelRouting,
    ) -> tuple[str, RoutingMetadata]:
        """Build the routing result tuple."""
        # We need to get the provider from the resolved model
        # For now, we'll set it when we look up the model later
        metadata = RoutingMetadata(
            requested_model=meta_model_tag,
            resolved_model=routing.resolved_model_tag,
            resolved_provider="pending",  # Will be updated when model is fetched
            operation=operation,
            strategy=get_strategy(meta_model_tag),
        )
        return routing.resolved_model_tag, metadata

    async def get_adapter_for_operation(
        self,
        model_tag: str,
        operation: str,
    ) -> tuple["BaseAdapter", str, ModelRegistry, RoutingMetadata | None]:
        """
        Get adapter for a model, resolving meta-models based on operation.

        This is the preferred method when the operation is known, as it
        properly handles mlpal meta-model routing.

        Args:
            model_tag: Model identifier (may be meta-model or real model)
            operation: Operation type (chat, image_generation, tts, etc.)

        Returns:
            Tuple of (adapter, provider_model_id, model_info, routing_metadata)
            - routing_metadata is None for non-meta-models

        Example:
            >>> adapter, model_id, model, routing = await router.get_adapter_for_operation(
            ...     "mlpal", "image_generation"
            ... )
            >>> # adapter is GoogleAdapter, model_id is "gemini-3-pro-image"
            >>> # routing.strategy is "quality"
        """
        # Resolve meta-model if applicable
        resolved_tag, routing_metadata = await self.resolve_meta_model(model_tag, operation)

        # Get the actual model
        model = await self.get_model(resolved_tag)

        # Update routing metadata with provider info
        if routing_metadata:
            routing_metadata.resolved_provider = model.provider

        # Get adapter
        adapter, wire_model_id = self._resolve_backend(model)

        return adapter, wire_model_id, model, routing_metadata

    async def get_adapter_with_breaker_for_operation(
        self,
        model_tag: str,
        operation: str,
    ) -> tuple["BaseAdapter", str, CircuitBreaker, ModelRegistry, RoutingMetadata | None]:
        """
        Get adapter with circuit breaker, resolving meta-models based on operation.

        Args:
            model_tag: Model identifier (may be meta-model or real model)
            operation: Operation type (chat, image_generation, tts, etc.)

        Returns:
            Tuple of (adapter, provider_model_id, circuit_breaker, model_info, routing_metadata)
        """
        # Resolve meta-model if applicable
        resolved_tag, routing_metadata = await self.resolve_meta_model(model_tag, operation)

        # Get the actual model
        model = await self.get_model(resolved_tag)

        # Update routing metadata with provider info
        if routing_metadata:
            routing_metadata.resolved_provider = model.provider

        # Get adapter and circuit breaker
        adapter, wire_model_id = self._resolve_backend(model)
        breaker = await self._circuit_breakers.get(model.provider)

        return adapter, wire_model_id, breaker, model, routing_metadata

    async def load_routing_table(self) -> int:
        """Load all active meta-model routing rules into the local cache.

        Called on startup and periodically by the background refresh task.

        Returns:
            Number of routing rules loaded.
        """
        try:
            meta_models = await self._meta_routing_repo.get_all_meta_models()
            count = 0
            for meta_model_tag in meta_models:
                routings = await self._meta_routing_repo.get_all_routings_for_model(meta_model_tag)
                # Group into priority-ordered candidate LISTS — the same shape
                # resolve_meta_model caches and consumes.
                by_op: dict[str, list] = {}
                for routing in sorted(routings, key=lambda x: x.priority):
                    by_op.setdefault(routing.operation, []).append(self._detached_routing(routing))
                for operation, candidates in by_op.items():
                    await self._routing_cache.set(f"{meta_model_tag}:{operation}", candidates)
                    count += len(candidates)
            logger.info(f"Loaded {count} meta-model routing rules into cache")
            return count
        except Exception as e:
            logger.error(f"Failed to load routing table: {e}")
            return 0

    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "model_cache": self._local_cache.stats(),
            "routing_cache": self._routing_cache.stats(),
        }

    async def clear_all_caches(self) -> None:
        """Clear both local and Redis caches."""
        await self._local_cache.clear()
        await self._routing_cache.clear()
        if self.redis:
            # Clear all model keys from Redis (model:* and models_list:*)
            for pattern in [f"{self.CACHE_PREFIX}*", "models_list:*"]:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self.redis.delete(*keys)
                    if cursor == 0:
                        break

    # =========================================================================
    # Redis Cache Helpers
    # =========================================================================

    async def _get_from_redis(self, model_tag: str) -> ModelRegistry | None:
        """Get model from Redis cache."""
        if not self.redis:
            return None

        try:
            key = f"{self.CACHE_PREFIX}{model_tag}"
            data = await self.redis.get(key)
            if data:
                # Reconstruct ModelRegistry from cached JSON
                obj = json.loads(data)
                return self._dict_to_model(obj)
        except Exception as e:
            logger.warning(f"Redis cache get error: {e}")

        return None

    async def _set_to_redis(self, model_tag: str, model: ModelRegistry) -> None:
        """Set model in Redis cache."""
        if not self.redis:
            return

        try:
            key = f"{self.CACHE_PREFIX}{model_tag}"
            data = self._model_to_dict(model)
            await self.redis.setex(key, self._settings.model_cache_ttl, json.dumps(data))
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")

    def _model_to_dict(self, model: ModelRegistry) -> dict:
        """Convert ModelRegistry to dict for caching."""
        return {
            "id": model.id,
            "model_tag": model.model_tag,
            "provider": model.provider,
            "provider_model_id": model.provider_model_id,
            "display_name": model.display_name,
            "description": model.description,
            "capabilities": model.capabilities,
            "context_length": model.context_length,
            "max_output_tokens": model.max_output_tokens,
            "pricing_tier": model.pricing_tier,
            "fallback_model_tag": model.fallback_model_tag,
            "priority": model.priority,
            "is_active": model.is_active,
            "is_deprecated": model.is_deprecated,
            "deprecation_message": model.deprecation_message,
            # Carry pause state through the cache so a paused model can't be
            # served from a cached/transient copy.
            "is_paused": getattr(model, "is_paused", False),
            "pause_reason": getattr(model, "pause_reason", None),
        }

    @staticmethod
    def _raise_if_paused(model_tag: str, model: ModelRegistry) -> None:
        """Raise if the model is paused OR its provider adapter isn't configured.

        Provider gate: a self-hosted box only holds keys for some providers, so a
        model whose provider isn't enabled is not serviceable — surface a clean
        404 rather than a provider-init 5xx. In managed/prod every provider is
        enabled, so the provider gate never fires there."""
        if getattr(model, "is_paused", False):
            reason = getattr(model, "pause_reason", None) or "Model is temporarily paused"
            raise ModelNotAvailableError(model_tag, reason)
        provider = getattr(model, "provider", None)
        if not ModelRouter._provider_served(provider):
            raise ModelNotAvailableError(model_tag, f"provider '{provider}' is not configured")

    @staticmethod
    def _provider_served(provider: str | None) -> bool:
        """True if a model with this provider can be served. Meta-models
        (provider 'mlpal' — routing aliases) are always kept: their servability
        is decided when they resolve to a concrete model. Real providers are
        gated by the adapter factory (key present). No-op filter in prod, where
        every real provider is enabled."""
        if not provider or provider == "mlpal":
            return True
        from mlpal_assistants_service.adapters.factory import get_adapter_factory

        return get_adapter_factory().is_enabled(provider)

    @staticmethod
    def _filter_enabled(models: list[ModelRegistry]) -> list[ModelRegistry]:
        """Drop models whose provider adapter isn't enabled (no key configured).
        Keeps meta-models. No-op in managed/prod where every provider is enabled."""
        return [m for m in models if ModelRouter._provider_served(getattr(m, "provider", None))]

    def _dict_to_model(self, data: dict) -> ModelRegistry:
        """Convert dict back to ModelRegistry (for cache hits)."""
        model = ModelRegistry()
        for key, value in data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        return model

    def _detached_model(self, model: ModelRegistry) -> ModelRegistry:
        """Return a session-independent copy of a model, safe to cache.

        A transient instance rebuilt from a plain dict carries no session, so
        reading its attributes never triggers a refresh — unlike a detached
        live instance, which raises DetachedInstanceError. Mirrors the Redis
        cache path, keeping local- and Redis-hit values behaviourally identical.
        """
        return self._dict_to_model(self._model_to_dict(model))

    def _routing_to_dict(self, routing: MetaModelRouting) -> dict:
        """Convert MetaModelRouting to a plain dict for caching."""
        return {
            "id": routing.id,
            "meta_model_tag": routing.meta_model_tag,
            "operation": routing.operation,
            "resolved_model_tag": routing.resolved_model_tag,
            "priority": routing.priority,
            "is_active": routing.is_active,
            "reason": routing.reason,
        }

    def _detached_routing(self, routing: MetaModelRouting) -> MetaModelRouting:
        """Return a session-independent copy of a routing rule, safe to cache.

        Same rationale as :meth:`_detached_model` — avoids DetachedInstanceError
        on cached routing rules read across requests.
        """
        obj = MetaModelRouting()
        for key, value in self._routing_to_dict(routing).items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj
