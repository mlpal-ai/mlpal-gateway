"""Pricing service for compute unit calculation.

The PricingService provides:
- Compute unit calculation for usage tracking
- Pricing lookup with caching (Redis + in-memory)
- Cost estimation for quota pre-checks
- Provider cost vs customer cost breakdown
"""

import json
import logging
from datetime import date
from decimal import Decimal

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.cache import TTLCache
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.db.models import ModelPricing
from mlpal_assistants_service.repositories import PricingRepository

logger = logging.getLogger(__name__)

# Providers whose standard cache-read price is NOT the global 0.10x of input.
# Verified against provider pricing pages 2026-09-01: Google bills cache hits
# (implicit and explicit, excluding explicit-cache storage fees) at 25% of the
# input rate; OpenAI and Anthropic at 10%. Per-model exceptions live in
# ModelPricing.cache_read_rate (e.g. claude-fable-5-1 at $0.25/MTok).
_PROVIDER_CACHE_READ_MULTIPLIER = {"google": Decimal("0.25")}


def provider_cache_read_multiplier(provider: str | None) -> Decimal:
    """The provider's standard cache-read multiple of the input rate."""
    from mlpal_assistants_service.core.config import get_settings

    if provider in _PROVIDER_CACHE_READ_MULTIPLIER:
        return _PROVIDER_CACHE_READ_MULTIPLIER[provider]
    return get_settings().cache_read_multiplier


class PricingService:
    """
    Service for compute unit pricing and calculation.

    Uses PricingRepository for data access with Redis caching
    for high-performance pricing lookups.

    Usage:
        service = PricingService(session, redis_client)

        # Calculate CU for a request
        cu = await service.calculate_compute_units(
            model_tag="gpt-5.2",
            input_tokens=1000,
            output_tokens=500,
        )

        # Get cost breakdown
        breakdown = await service.get_cost_breakdown(
            model_tag="gpt-5.2",
            input_tokens=1000,
            output_tokens=500,
        )
    """

    CACHE_TTL = 3600  # 1 hour
    # Version the key whenever the serialized row shape changes: entries cached
    # under an older prefix are simply never read again (v2: + cache_read_rate).
    CACHE_PREFIX = "pricing:v2:"

    def __init__(
        self,
        session: AsyncSession,
        redis_client: redis.Redis | None = None,
    ) -> None:
        self.session = session
        self.redis = redis_client
        self.settings = get_settings()
        self._repo = PricingRepository(session)
        self._local_cache: TTLCache = TTLCache(
            default_ttl=self.settings.local_cache_ttl,
            maxsize=self.settings.local_cache_maxsize,
        )

    async def get_pricing(
        self,
        model_tag: str,
        operation: str = "chat",
    ) -> ModelPricing | None:
        """
        Get pricing for a model and operation.

        Checks caches before hitting database.

        Args:
            model_tag: Model identifier
            operation: Operation type (chat, embedding, image_generation, etc.)

        Returns:
            ModelPricing record or None if not found
        """
        cache_key = f"{model_tag}:{operation}"

        # Check in-memory TTL cache first
        cached_pricing = self._local_cache.get(cache_key)
        if cached_pricing is not None:
            return cached_pricing

        # Check Redis cache
        if self.redis:
            cached = await self._get_from_redis(cache_key)
            if cached:
                await self._local_cache.set(cache_key, cached)
                return cached

        # Query database via repository
        pricing = await self._repo.get_current_pricing(model_tag, operation)

        if pricing:
            # Cache it
            await self._local_cache.set(cache_key, pricing)
            if self.redis:
                await self._set_to_redis(cache_key, pricing)

        return pricing

    async def calculate_compute_units(
        self,
        model_tag: str,
        input_units: int | Decimal,
        output_units: int | Decimal = 0,
        operation: str = "chat",
        cached_units: int | Decimal = 0,
        cached_included: bool = True,
        provider: str | None = None,
        cache_write_5m_units: int | Decimal = 0,
        cache_write_1h_units: int | Decimal = 0,
    ) -> Decimal:
        """The billed CU for a request: the model's real cost, pass-through.

        Locked pricing (planning/pricing-DECISION.md): tokens are billed at
        provider cost with NO markup — the flat tier fee is the revenue. Stored
        rates still carry the legacy markup_multiplier column, so this divides
        it back out (ModelPricing.calculate_provider_cost).

        Cache reads are billed at the provider's discounted rate, matching what
        the provider charges us: the row's per-model cache_read_rate when set
        (claude-fable-5-1), else input_rate x the provider's standard multiple
        (OpenAI/Anthropic 0.10x, Google 0.25x — verified against provider
        pricing pages 2026-09-01). `cached_included` says whether
        `input_units` already contains the cached subset (OpenAI/Google usage
        semantics: yes; Anthropic's v1-wire adapter: no).
        """
        pricing = await self.get_pricing(model_tag, operation)

        if pricing is None:
            logger.warning(f"No pricing found for {model_tag}:{operation}, using default")
            return self._calculate_default_cu(input_units, output_units)

        cached = Decimal(cached_units or 0)
        write_5m = Decimal(cache_write_5m_units or 0)
        write_1h = Decimal(cache_write_1h_units or 0)
        if cached <= 0 and write_5m <= 0 and write_1h <= 0:
            return pricing.calculate_provider_cost(input_units, output_units)

        # Cache WRITES (Anthropic only) bill at input_rate x 1.25 (5m) / x 2
        # (1h); `cached_included` governs them exactly like reads.
        uncached_input = Decimal(input_units)
        if cached_included:
            uncached_input = max(Decimal("0"), uncached_input - cached - write_5m - write_1h)

        base = pricing.calculate_provider_cost(uncached_input, output_units)

        cache_rate = getattr(pricing, "cache_read_rate", None)
        if cache_rate is None:
            cache_rate = pricing.input_rate * provider_cache_read_multiplier(provider)
        divisor = (
            Decimal("1000")
            if pricing.rate_unit == "per_1k_tokens"
            else Decimal("1000000")
        )
        cu_to_dollar = pricing.cu_to_dollar or Decimal("10")
        settings = get_settings()
        write_dollars = pricing.input_rate * (
            write_5m * settings.cache_5m_write_multiplier
            + write_1h * settings.cache_1h_write_multiplier
        )
        return base + (cached * cache_rate + write_dollars) / divisor / cu_to_dollar


    async def get_cost_breakdown(
        self,
        model_tag: str,
        input_units: int | Decimal,
        output_units: int | Decimal = 0,
        operation: str = "chat",
    ) -> dict:
        """
        Get a detailed cost breakdown for a request.

        Returns:
            Dict with provider_cost, compute_units, markup, rates, etc.
        """
        pricing = await self.get_pricing(model_tag, operation)

        if pricing is None:
            cu = self._calculate_default_cu(input_units, output_units)
            provider_cost = cu
            return {
                "model_tag": model_tag,
                "operation": operation,
                "input_units": int(input_units),
                "output_units": int(output_units),
                "input_rate": None,
                "output_rate": None,
                "rate_unit": "unknown",
                "markup_multiplier": 1.0,
                "provider_cost": float(provider_cost),
                "compute_units": float(cu),
                "pricing_found": False,
            }

        provider_cost = pricing.calculate_provider_cost(input_units, output_units)
        cu = provider_cost  # billed CU IS the pass-through cost

        return {
            "model_tag": model_tag,
            "operation": operation,
            "input_units": int(input_units),
            "output_units": int(output_units),
            "input_rate": float(pricing.input_rate),
            "output_rate": float(pricing.output_rate),
            "rate_unit": pricing.rate_unit,
            "markup_multiplier": float(pricing.markup_multiplier),
            "provider_cost": float(provider_cost),
            "compute_units": float(cu),
            "pricing_found": True,
            "tier": pricing.tier,
            "effective_date": pricing.effective_date.isoformat(),
        }

    async def estimate_compute_units(
        self,
        model_tag: str,
        estimated_tokens: int,
        operation: str = "chat",
    ) -> Decimal:
        """
        Estimate compute units before a request.

        Used for quota pre-checks. Assumes a typical input/output ratio.

        Args:
            model_tag: Model identifier
            estimated_tokens: Estimated total tokens
            operation: Operation type

        Returns:
            Estimated compute units
        """
        # Assume 70% input, 30% output for typical chat
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = estimated_tokens - input_tokens

        return await self.calculate_compute_units(
            model_tag=model_tag,
            input_units=input_tokens,
            output_units=output_tokens,
            operation=operation,
        )

    async def calculate_image_cost(
        self,
        model_tag: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> Decimal:
        """
        Calculate compute units for image generation.

        Args:
            model_tag: Model identifier
            size: Image size
            quality: Image quality (standard, hd)
            n: Number of images

        Returns:
            Compute units as Decimal
        """
        # Image pricing is typically per image, with size/quality multipliers
        return await self.calculate_compute_units(
            model_tag=model_tag,
            input_units=n,  # Number of images as input units
            output_units=0,
            operation="image_generation",
        )

    async def calculate_tts_cost(
        self,
        model_tag: str,
        characters: int,
    ) -> Decimal:
        """
        Calculate compute units for text-to-speech.

        Args:
            model_tag: Model identifier
            characters: Number of characters

        Returns:
            Compute units as Decimal
        """
        return await self.calculate_compute_units(
            model_tag=model_tag,
            input_units=characters,
            output_units=0,
            operation="tts",
        )

    async def calculate_transcription_cost(
        self,
        model_tag: str,
        duration_seconds: float,
    ) -> Decimal:
        """
        Calculate compute units for audio transcription.

        Args:
            model_tag: Model identifier
            duration_seconds: Audio duration in seconds

        Returns:
            Compute units as Decimal
        """
        # Transcription is typically priced per minute
        duration_minutes = Decimal(str(duration_seconds)) / Decimal("60")
        return await self.calculate_compute_units(
            model_tag=model_tag,
            input_units=duration_minutes,
            output_units=0,
            operation="transcription",
        )

    def _calculate_default_cu(
        self,
        input_units: int | Decimal,
        output_units: int | Decimal,
    ) -> Decimal:
        """Fallback when pricing is missing — pass-through, no markup. Rates
        are DOLLARS per 1M; CU = dollars / cu_to_usd (1 CU = $10), matching
        every pricing-backed path — returning raw dollars here would bill a
        priceless model 10x."""
        default_input_rate = Decimal("1.0")   # $1 / 1M tokens
        default_output_rate = Decimal("2.0")  # output typically more expensive

        input_units = Decimal(str(input_units))
        output_units = Decimal(str(output_units))

        input_cost = input_units * default_input_rate / Decimal("1000000")
        output_cost = output_units * default_output_rate / Decimal("1000000")

        return (input_cost + output_cost) / Decimal(str(self.settings.cu_to_usd))

    async def clear_cache(self) -> None:
        """Clear the in-memory pricing cache."""
        await self._local_cache.clear()

    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {"pricing_cache": self._local_cache.stats()}

    async def clear_all_caches(self) -> None:
        """Clear both local and Redis caches."""
        await self._local_cache.clear()
        if self.redis:
            pattern = f"{self.CACHE_PREFIX}*"
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

    async def _get_from_redis(self, cache_key: str) -> ModelPricing | None:
        """Get pricing from Redis cache."""
        if not self.redis:
            return None

        try:
            key = f"{self.CACHE_PREFIX}{cache_key}"
            data = await self.redis.get(key)
            if data:
                obj = json.loads(data)
                return self._dict_to_pricing(obj)
        except Exception as e:
            logger.warning(f"Redis cache get error: {e}")

        return None

    async def _set_to_redis(self, cache_key: str, pricing: ModelPricing) -> None:
        """Set pricing in Redis cache."""
        if not self.redis:
            return

        try:
            key = f"{self.CACHE_PREFIX}{cache_key}"
            data = self._pricing_to_dict(pricing)
            await self.redis.setex(key, self.CACHE_TTL, json.dumps(data))
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")

    def _pricing_to_dict(self, pricing: ModelPricing) -> dict:
        """Convert ModelPricing to dict for caching."""
        return {
            "id": pricing.id,
            "model_tag": pricing.model_tag,
            "operation": pricing.operation,
            "tier": pricing.tier,
            "input_rate": str(pricing.input_rate),
            "output_rate": str(pricing.output_rate),
            "rate_unit": pricing.rate_unit,
            "markup_multiplier": str(pricing.markup_multiplier),
            "effective_date": pricing.effective_date.isoformat(),
            "is_active": pricing.is_active,
            "cu_to_dollar": str(pricing.cu_to_dollar),
            "input_cu_rate": str(pricing.input_cu_rate),
            "output_cu_rate": str(pricing.output_cu_rate),
            # Per-model cache-read list price. MUST round-trip — a missing key
            # here silently drops the model's own cache rate on any cache hit,
            # falling back to the global multiplier (10x over-bill on
            # claude-fable-5-1: $0.25/MTok billed as $1.00/MTok).
            "cache_read_rate": (
                str(pricing.cache_read_rate)
                if pricing.cache_read_rate is not None
                else None
            ),
        }

    def _dict_to_pricing(self, data: dict) -> ModelPricing:
        """Convert dict back to ModelPricing."""
        pricing = ModelPricing()
        pricing.id = data["id"]
        pricing.model_tag = data["model_tag"]
        pricing.operation = data["operation"]
        pricing.tier = data["tier"]
        pricing.input_rate = Decimal(data["input_rate"])
        pricing.output_rate = Decimal(data["output_rate"])
        pricing.rate_unit = data["rate_unit"]
        pricing.markup_multiplier = Decimal(data["markup_multiplier"])
        pricing.effective_date = date.fromisoformat(data["effective_date"])
        pricing.is_active = data["is_active"]
        pricing.cu_to_dollar = Decimal(data["cu_to_dollar"])
        pricing.input_cu_rate = Decimal(data["input_cu_rate"])
        pricing.output_cu_rate = Decimal(data["output_cu_rate"])
        # .get for forward-compat with entries cached before this field existed;
        # those are also flushed by the CACHE_PREFIX version bump below.
        crr = data.get("cache_read_rate")
        pricing.cache_read_rate = Decimal(crr) if crr is not None else None
        return pricing
