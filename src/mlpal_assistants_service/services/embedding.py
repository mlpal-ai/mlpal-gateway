"""Embedding service - generates text embeddings.

Orchestrates embedding generation across providers:
- ModelRouter: Route to appropriate provider adapter
- PricingService: Calculate compute units
- BillingRepository: Check billing status (Redis-cached)
- UsageService: Track and record usage
- RateLimiter: Enforce rate limits
"""

import asyncio
import logging
import time
import uuid
from decimal import Decimal
from typing import Any

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    QuotaExceededError,
    WalletEmptyError,
)
from mlpal_assistants_service.repositories.usage_repository import UsageRepository
from mlpal_assistants_service.schemas.embeddings import (
    EmbeddingCost,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from mlpal_assistants_service.seams.billing import build_billing_gate, is_insufficient_wallet_error
from mlpal_assistants_service.services.account_trust import AccountTrustService
from mlpal_assistants_service.services.policy import PolicyService
from mlpal_assistants_service.services.pricing import PricingService
from mlpal_assistants_service.services.rate_limiter import RateLimiter
from mlpal_assistants_service.services.router import ModelRouter
from mlpal_assistants_service.services.usage import UsageService

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.

    Handles the full request lifecycle:
    1. Rate limiting check (pipelined Redis)
    2. Billing status check (Redis-cached)
    3. Model routing and adapter selection
    4. Embedding generation
    5. Cost calculation
    6. Fire-and-forget: billing increment, usage recording, token recording

    Usage:
        service = EmbeddingService(session, redis_client)
        response = await service.embed(
            user_id=123,
            api_key_id=456,
            request=EmbeddingRequest(
                input=["Hello world", "How are you?"],
                model="text-embedding-3-large",
            ),
        )
    """

    def __init__(
        self,
        session: AsyncSession,
        redis_client: redis.Redis | None = None,
        sqs_client: Any | None = None,
        shared_caches: dict | None = None,
    ) -> None:
        self.session = session
        self.redis = redis_client
        self._sqs_client = sqs_client

        # Initialize component services
        self._router = ModelRouter(session, redis_client)
        self._pricing = PricingService(session, redis_client)
        self._billing = build_billing_gate(session, redis_client)
        self._usage = UsageService(session, redis_client, sqs_client)
        # Per-key policy engine. Pre-check reconciles budget spend from the
        # request session's usage repo; accrual (post-request) is Redis-only.
        self._policy = PolicyService(
            redis_client, UsageRepository(session), trust=AccountTrustService(session, redis_client)
        )
        self._rate_limiter = RateLimiter(redis_client) if redis_client else None

        # Inject shared caches for hot-path optimization
        if shared_caches:
            if shared_caches.get("model_cache"):
                self._router._local_cache = shared_caches["model_cache"]
            if shared_caches.get("routing_cache"):
                self._router._routing_cache = shared_caches["routing_cache"]
            if shared_caches.get("pricing_cache"):
                self._pricing._local_cache = shared_caches["pricing_cache"]

        # Track background tasks to prevent GC
        self._background_tasks: set[asyncio.Task] = set()

    def _fire_and_forget(self, coro: Any) -> None:
        """Schedule a coroutine as a fire-and-forget background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _post_request_background(
        self,
        user_id: int,
        api_key_id: int,
        trace_id: str,
        resolved_model_tag: str,
        provider: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        compute_units: Decimal,
        latency_ms: int,
        billing_needs_ensure: bool,
        status: str = "success",
        error_code: str | None = None,
        budgets: list | None = None,
    ) -> None:
        """Background task for post-provider steps.

        Uses its own DB session since the request session is already closed.
        """
        from mlpal_assistants_service.db.session import async_session_factory

        try:
            async with async_session_factory() as bg_session:
                bg_billing = build_billing_gate(bg_session, self.redis)
                bg_usage = UsageService(bg_session, self.redis, self._sqs_client)

                # Ensure billing status if needed (new user)
                if billing_needs_ensure:
                    await bg_billing.ensure_billing_status(user_id)

                # Record usage (SQS or DB)
                await bg_usage.record_usage(
                    user_id=str(user_id),
                    api_key_id=str(api_key_id),
                    trace_id=trace_id,
                    model_tag=resolved_model_tag,
                    provider=provider,
                    operation=operation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    compute_units=compute_units,
                    latency_ms=latency_ms,
                    status=status,
                    error_code=error_code,
                    wallet_debit_status=("pending" if status == "success" else "not_applicable"),
                )

                if status == "success":
                    # CU v3: skip wallet debit when gating is off (or the debit
                    # kill switch is on). Usage is still logged regardless.
                    if not await bg_billing.is_wallet_debit_active():
                        await bg_usage.mark_wallet_debit_status(
                            trace_id, "not_applicable"
                        )
                    else:
                        try:
                            await bg_billing.debit_wallet_usage(
                                user_id=user_id,
                                compute_units=compute_units,
                                usage_ref=trace_id,
                            )
                        except Exception as debit_error:
                            await bg_usage.mark_wallet_debit_status(
                                trace_id,
                                (
                                    "failed_permanent"
                                    if is_insufficient_wallet_error(debit_error)
                                    else "failed_retryable"
                                ),
                                error=str(debit_error),
                            )
                        else:
                            await bg_usage.mark_wallet_debit_status(trace_id, "debited")

                await bg_session.commit()

            # Record tokens for rate limiting (Redis only, no DB session needed)
            if self._rate_limiter and status == "success":
                total_tokens = input_tokens + output_tokens
                if total_tokens > 0:
                    await self._rate_limiter.record_tokens(str(user_id), total_tokens)

            # Accrue this call's CU onto the key's spend budgets (Redis only).
            await self._policy.record_key_usage(api_key_id, budgets, compute_units)

        except Exception as e:
            logger.error("Background post-request failed", exc_info=e)

    async def embed(
        self,
        user_id: int,
        api_key_id: int,
        request: EmbeddingRequest,
        tier: str = "standard",
        model_policy: dict | None = None,
        budgets: list | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for input text(s).

        Args:
            user_id: User ID
            api_key_id: API key ID
            request: Embedding request
            tier: Rate limit tier

        Returns:
            EmbeddingResponse with vectors and cost info

        Raises:
            ModelNotFoundError: If model doesn't exist
            QuotaExceededError: If user is over quota
            RateLimitExceededError: If rate limited
            ProviderError: If provider API fails
        """
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Normalize input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        try:
            # 1. Check rate limits
            if self._rate_limiter:
                await self._rate_limiter.check_request_limit(str(user_id), tier)

            # 2. Check billing status (Redis-cached)
            (
                can_request,
                block_reason,
                billing_existed,
            ) = await self._billing.can_make_request_cached(user_id)
            if not can_request:
                from mlpal_assistants_service.repositories.billing_repository import (
                    WALLET_EMPTY_MESSAGE,
                )

                if block_reason == WALLET_EMPTY_MESSAGE:
                    raise WalletEmptyError(block_reason)
                raise QuotaExceededError(
                    message=block_reason or "API access blocked",
                    limit=0.0,
                    current_usage=0.0,
                )

            # 3. Get adapter with circuit breaker, resolving meta-models
            (
                adapter,
                provider_model_id,
                breaker,
                model_info,
                routing_metadata,
            ) = await self._router.get_adapter_with_breaker_for_operation(
                request.model,
                operation="embedding",
            )

            # For pricing/usage, use the resolved model if this was a meta-model
            resolved_model_tag = (
                routing_metadata.resolved_model if routing_metadata else request.model
            )

            # 3b. Per-key policy gate (no-op when the key has no policy). Model
            # access is checked against the requested tag AND the resolved model
            # so an alias can't reach a denied model; budgets deny (402) once any
            # window is exhausted. Both run before the paid provider call.
            self._policy.check_model_access(
                model_policy, requested=request.model, resolved=resolved_model_tag
            )
            await self._policy.check_budgets(api_key_id, budgets)

            # 4. Execute request with circuit breaker
            async with breaker:
                response = await adapter.embed(
                    texts=texts,
                    model=provider_model_id,
                    dimensions=request.dimensions,
                )

            # 5. Calculate latency and compute units
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            cu = await self._pricing.calculate_compute_units(
                model_tag=resolved_model_tag,
                input_units=response.usage.input_tokens,
                output_units=0,
                operation="embedding",
            )
            compute_units = cu

            # 6. Fire-and-forget: billing increment, usage record, token recording
            self._fire_and_forget(
                self._post_request_background(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    trace_id=trace_id,
                    resolved_model_tag=resolved_model_tag,
                    provider=model_info.provider,
                    operation="embedding",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=0,
                    compute_units=compute_units,
                    latency_ms=latency_ms,
                    billing_needs_ensure=not billing_existed,
                    budgets=budgets,
                )
            )

            # 7. Build response (returned immediately)
            return EmbeddingResponse(
                data=[
                    EmbeddingData(index=i, embedding=emb)
                    for i, emb in enumerate(response.embeddings)
                ],
                model=request.model,
                usage=EmbeddingUsage(
                    prompt_tokens=response.usage.input_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                cost=EmbeddingCost(
                    model_name=resolved_model_tag,
                    provider=model_info.provider,
                    tokens=response.usage.input_tokens,
                    latency_ms=latency_ms,
                    compute_units=float(compute_units),
                ),
                routing=routing_metadata,
            )

        except Exception as e:
            # Record failed request via fire-and-forget
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_code = type(e).__name__

            try:
                _, resolved_metadata = await self._router.resolve_meta_model(
                    request.model, "embedding"
                )
                resolved_tag = (
                    resolved_metadata.resolved_model if resolved_metadata else request.model
                )
                model_info = await self._router.get_model(resolved_tag)
                provider = model_info.provider
            except ModelNotFoundError:
                resolved_tag = request.model
                provider = "unknown"

            self._fire_and_forget(
                self._post_request_background(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    trace_id=trace_id,
                    resolved_model_tag=resolved_tag,
                    provider=provider,
                    operation="embedding",
                    input_tokens=0,
                    output_tokens=0,
                    compute_units=Decimal("0"),
                    latency_ms=latency_ms,
                    billing_needs_ensure=False,
                    status="error",
                    error_code=error_code,
                    budgets=budgets,
                )
            )
            raise
