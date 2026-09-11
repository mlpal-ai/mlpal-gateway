"""Image generation service.

Orchestrates image generation across providers:
- ModelRouter: Route to appropriate provider adapter
- PricingService: Calculate compute units
- BillingRepository: Check billing status (Redis-cached)
- UsageService: Track and record usage
- AssetStorageService: Store generated images in S3
"""

import asyncio
import base64
import logging
import time
import uuid
from datetime import UTC
from decimal import Decimal
from typing import Any

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.adapters.base import (
    FileAttachment,
    FileSource,
    FileType,
    ImageQuality,
    ImageSize,
)
from mlpal_assistants_service.adapters.base import GeneratedImage as AdapterGeneratedImage
from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    QuotaExceededError,
    WalletEmptyError,
)
from mlpal_assistants_service.core.storage import AssetStorageService
from mlpal_assistants_service.repositories.usage_repository import UsageRepository
from mlpal_assistants_service.schemas.images import (
    GeneratedImage,
    ImageGenerationCost,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ReferenceImage,
)
from mlpal_assistants_service.seams.billing import build_billing_gate, is_insufficient_wallet_error
from mlpal_assistants_service.services.policy import PolicyService
from mlpal_assistants_service.services.pricing import PricingService
from mlpal_assistants_service.services.router import ModelRouter
from mlpal_assistants_service.services.usage import UsageService

logger = logging.getLogger(__name__)


def _extract_image_bytes(img: AdapterGeneratedImage) -> bytes:
    """
    Extract image bytes from adapter response.

    Adapters may return images as:
    - data: Raw bytes
    - base64: Base64-encoded string

    Args:
        img: Generated image from adapter

    Returns:
        Raw image bytes

    Raises:
        ValueError: If no image data available
    """
    if img.data is not None:
        return img.data
    if img.base64 is not None:
        return base64.b64decode(img.base64)
    raise ValueError("No image data available in adapter response")


def _get_image_content_type(img: AdapterGeneratedImage) -> str:
    """Get content type from image format."""
    format_to_mime = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return format_to_mime.get(img.format, "image/png")


def _convert_reference_image(ref: ReferenceImage) -> FileAttachment:
    """
    Convert a ReferenceImage schema to FileAttachment for adapter.

    Handles both URL and base64 inputs.
    """
    if ref.url:
        return FileAttachment(
            type=FileType.IMAGE,
            source=FileSource.URL,
            data=ref.url,
            mime_type=ref.mime_type,
        )
    elif ref.base64:
        return FileAttachment(
            type=FileType.IMAGE,
            source=FileSource.BASE64,
            data=ref.base64,
            mime_type=ref.mime_type or "image/png",
        )
    else:
        raise ValueError("ReferenceImage must have either url or base64")


class ImageService:
    """
    Service for generating images from text prompts.

    Handles the full request lifecycle:
    1. Billing status check (Redis-cached)
    2. Model routing and adapter selection
    3. Image generation via provider
    4. Upload to S3 for temporary storage (parallel)
    5. Cost calculation
    6. Fire-and-forget: billing increment, usage recording

    Generated images are stored temporarily in S3 and returned as
    presigned URLs that expire after 1 hour.

    Usage:
        service = ImageService(session, redis_client, asset_storage)
        response = await service.generate(
            user_id=123,
            api_key_id=456,
            request=ImageGenerationRequest(
                prompt="A sunset over mountains",
                model="gpt-image-2",
            ),
        )
        # response.data[0].url is a presigned S3 URL
        # response.data[0].expires_at indicates when URL expires
    """

    def __init__(
        self,
        session: AsyncSession,
        redis_client: redis.Redis | None = None,
        asset_storage: AssetStorageService | None = None,
        sqs_client: Any | None = None,
        shared_caches: dict | None = None,
    ) -> None:
        self.session = session
        self.redis = redis_client
        self._asset_storage = asset_storage
        self._sqs_client = sqs_client

        # Initialize component services
        self._router = ModelRouter(session, redis_client)
        self._pricing = PricingService(session, redis_client)
        self._billing = build_billing_gate(session, redis_client)
        self._usage = UsageService(session, redis_client, sqs_client)
        # Per-key policy engine. Pre-check reconciles budget spend from the
        # request session's usage repo; accrual (post-request) is Redis-only.
        self._policy = PolicyService(redis_client, UsageRepository(session))

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
        input_tokens: int = 0,
        output_tokens: int = 0,
        compute_units: Decimal = Decimal("0"),
        latency_ms: int = 0,
        billing_needs_ensure: bool = False,
        status: str = "success",
        error_code: str | None = None,
        operation: str = "image_generation",
        budgets: list | None = None,
    ) -> None:
        """Background task for post-provider steps."""
        from mlpal_assistants_service.db.session import async_session_factory

        try:
            async with async_session_factory() as bg_session:
                bg_billing = build_billing_gate(bg_session, self.redis)
                bg_usage = UsageService(bg_session, self.redis, self._sqs_client)

                if billing_needs_ensure:
                    await bg_billing.ensure_billing_status(user_id)

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

            # Accrue this call's CU onto the key's spend budgets (Redis only).
            await self._policy.record_key_usage(api_key_id, budgets, compute_units)

        except Exception as e:
            logger.error("Background post-request failed", exc_info=e)

    async def _upload_single_image(
        self,
        img: AdapterGeneratedImage,
        index: int,
        user_id: str,
        trace_id: str,
    ) -> GeneratedImage:
        """Upload a single image to S3 and return GeneratedImage."""
        image_bytes = _extract_image_bytes(img)
        content_type = _get_image_content_type(img)
        extension = img.format or "png"

        if self._asset_storage is not None:
            asset = await self._asset_storage.upload_asset(
                data=image_bytes,
                filename=f"image_{index}.{extension}",
                content_type=content_type,
                user_id=user_id,
                trace_id=trace_id,
            )
            return GeneratedImage(
                url=asset.url,
                expires_at=asset.expires_at,
                revised_prompt=img.revised_prompt,
                content_type=asset.content_type,
                size_bytes=asset.size_bytes,
            )
        else:
            logger.warning(
                "AssetStorageService not configured, returning base64 data",
                trace_id=trace_id,
            )
            from datetime import datetime, timedelta

            b64_data = img.base64 or base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:{content_type};base64,{b64_data}"
            return GeneratedImage(
                url=data_url,
                expires_at=datetime.now(UTC) + timedelta(hours=24),
                revised_prompt=img.revised_prompt,
                content_type=content_type,
                size_bytes=len(image_bytes),
            )

    async def generate(
        self,
        user_id: int,
        api_key_id: int,
        request: ImageGenerationRequest,
        model_policy: dict | None = None,
        budgets: list | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate images from a text prompt.

        Args:
            user_id: User ID
            api_key_id: API key ID
            request: Image generation request

        Returns:
            ImageGenerationResponse with generated images as presigned URLs

        Raises:
            ModelNotFoundError: If model doesn't exist
            QuotaExceededError: If user is over quota
            ProviderError: If provider API fails
            AssetStorageError: If S3 upload fails
        """
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            # 1. Check billing status (Redis-cached)
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

            # 2. Get adapter with circuit breaker, resolving meta-models
            (
                adapter,
                provider_model_id,
                breaker,
                model_info,
                routing_metadata,
            ) = await self._router.get_adapter_with_breaker_for_operation(
                request.model,
                operation="image_generation",
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

            # Pass the caller's size through UNCHANGED; each adapter's resolver
            # converts it to that provider/model's native format. Do NOT collapse
            # to an aspect ratio here — that would discard explicit pixel
            # dimensions (e.g. "2048x2048") that flexible models like gpt-image-2
            # support, silently downgrading them to a default size.
            size: ImageSize | str = request.size or ImageSize.SQUARE

            # Map quality string to ImageQuality enum
            quality = ImageQuality(request.quality) if request.quality else ImageQuality.STANDARD

            # Convert reference images if provided
            reference_attachments: list[FileAttachment] | None = None
            if request.reference_images:
                reference_attachments = [
                    _convert_reference_image(ref) for ref in request.reference_images
                ]
                logger.info(
                    f"Image generation with {len(reference_attachments)} reference images",
                    extra={"trace_id": trace_id, "reference_count": len(reference_attachments)},
                )

            # 3. Execute request with circuit breaker
            async with breaker:
                response = await adapter.generate_image(
                    prompt=request.prompt,
                    model=provider_model_id,
                    size=size,
                    quality=quality,
                    n=request.n,
                    reference_images=reference_attachments,
                )

            # 4. Calculate latency and compute units
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Token-priced image models (gpt-image-2.x) report usage and carry a
            # per_1m_tokens pricing row: bill text input, image input, and
            # image output exactly. Per-image rows bill n images as before.
            usage = getattr(response, "usage", None)
            pricing_row = await self._pricing.get_pricing(resolved_model_tag, "image_generation")
            token_priced = (
                usage is not None
                and pricing_row is not None
                and pricing_row.rate_unit in ("per_1m_tokens", "per_1k_tokens")
            )
            if token_priced:
                cu = await self._pricing.calculate_compute_units(
                    model_tag=resolved_model_tag,
                    input_units=usage.input_tokens - usage.image_input_tokens,
                    output_units=usage.output_tokens,
                    operation="image_generation",
                    image_input_units=usage.image_input_tokens,
                )
                billed_input_tokens, billed_output_tokens = usage.input_tokens, usage.output_tokens
            else:
                cu = await self._pricing.calculate_compute_units(
                    model_tag=resolved_model_tag,
                    input_units=request.n,
                    output_units=0,
                    operation="image_generation",
                )
                billed_input_tokens, billed_output_tokens = 0, 0
            compute_units = cu

            # 5. Fire-and-forget: billing increment, usage recording
            self._fire_and_forget(
                self._post_request_background(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    trace_id=trace_id,
                    resolved_model_tag=resolved_model_tag,
                    provider=model_info.provider,
                    compute_units=compute_units,
                    input_tokens=billed_input_tokens,
                    output_tokens=billed_output_tokens,
                    latency_ms=latency_ms,
                    billing_needs_ensure=not billing_existed,
                    budgets=budgets,
                )
            )

            # 6. Upload images to S3 in parallel (blocking — response needs URLs)
            images = await asyncio.gather(
                *[
                    self._upload_single_image(img, i, str(user_id), trace_id)
                    for i, img in enumerate(response.images)
                ]
            )

            return ImageGenerationResponse(
                data=list(images),
                model=request.model,
                cost=ImageGenerationCost(
                    model_name=resolved_model_tag,
                    provider=model_info.provider,
                    images_generated=len(images),
                    latency_ms=latency_ms,
                    compute_units=float(compute_units),
                    input_tokens=billed_input_tokens if token_priced else None,
                    output_tokens=billed_output_tokens if token_priced else None,
                ),
                routing=routing_metadata,
            )

        except Exception as e:
            # Record failed request via fire-and-forget
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_code = type(e).__name__

            try:
                _, resolved_metadata = await self._router.resolve_meta_model(
                    request.model, "image_generation"
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
                    compute_units=Decimal("0"),
                    latency_ms=latency_ms,
                    billing_needs_ensure=False,
                    status="error",
                    error_code=error_code,
                    budgets=budgets,
                )
            )
            raise
