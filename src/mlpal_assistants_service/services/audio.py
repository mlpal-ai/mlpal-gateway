"""Audio service - TTS and transcription.

Orchestrates audio operations across providers:
- ModelRouter: Route to appropriate provider adapter
- PricingService: Calculate compute units
- BillingRepository: Check billing status (Redis-cached)
- UsageService: Track and record usage
- AssetStorageService: Store generated audio in S3
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

from mlpal_assistants_service.adapters.base import FileAttachment, FileSource, FileType
from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    QuotaExceededError,
    WalletEmptyError,
)
from mlpal_assistants_service.core.storage import AssetStorageService
from mlpal_assistants_service.repositories.usage_repository import UsageRepository
from mlpal_assistants_service.schemas.audio import (
    TranscriptionCost,
    TranscriptionRequest,
    TranscriptionResponse,
    TTSCost,
    TTSRequest,
    TTSResponse,
)
from mlpal_assistants_service.seams.billing import build_billing_gate, is_insufficient_wallet_error
from mlpal_assistants_service.services.account_trust import AccountTrustService
from mlpal_assistants_service.services.policy import PolicyService
from mlpal_assistants_service.services.pricing import PricingService
from mlpal_assistants_service.services.router import ModelRouter
from mlpal_assistants_service.services.usage import UsageService

logger = logging.getLogger(__name__)


# Content type mapping for audio formats
AUDIO_FORMAT_TO_MIME = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
}


class AudioService:
    """
    Service for audio operations (TTS and transcription).

    Handles:
    - Text-to-Speech (TTS): Convert text to audio, stored in S3
    - Speech-to-Text (STT): Transcribe audio to text

    Generated audio is stored temporarily in S3 and returned as
    presigned URLs that expire after 1 hour.

    Usage:
        service = AudioService(session, redis_client, asset_storage)

        # TTS - returns URL to audio file
        response = await service.text_to_speech(
            user_id=123,
            api_key_id=456,
            request=TTSRequest(input="Hello world", voice="nova"),
        )
        # response.url is a presigned S3 URL

        # Transcription
        response = await service.transcribe(
            user_id=123,
            api_key_id=456,
            audio_data=audio_bytes,
            request=TranscriptionRequest(model="whisper-1"),
        )
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
        self._policy = PolicyService(
            redis_client, UsageRepository(session), trust=AccountTrustService(session, redis_client)
        )

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
                    # kill switch is on). The UI already subtracts CU usage from
                    # wallet balance client-side; usage is still logged.
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
            await self._policy.record_key_usage(api_key_id, budgets, compute_units, user_id=user_id)

        except Exception as e:
            logger.error("Background post-request failed", exc_info=e)

    async def text_to_speech(
        self,
        user_id: int,
        api_key_id: int,
        request: TTSRequest,
        model_policy: dict | None = None,
        budgets: list | None = None,
    ) -> tuple[bytes, TTSResponse]:
        """
        Convert text to speech audio.

        Returns both raw bytes (for streaming) and TTSResponse (with URL).

        Args:
            user_id: User ID
            api_key_id: API key ID
            request: TTS request

        Returns:
            Tuple of (audio_bytes, TTSResponse with URL and metadata)

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
                operation="tts",
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
            await self._policy.check_budgets(api_key_id, budgets, user_id=user_id)

            # 3. Execute request with circuit breaker
            async with breaker:
                response = await adapter.text_to_speech(
                    text=request.input,
                    model=provider_model_id,
                    voice=request.voice,
                    format=request.response_format,
                    speed=request.speed,
                )

            # 4. Calculate latency and compute units
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            cu = await self._pricing.calculate_compute_units(
                model_tag=resolved_model_tag,
                input_units=len(request.input),
                output_units=0,
                operation="tts",
            )
            compute_units = cu

            # 5. Fire-and-forget: billing increment, usage recording
            self._fire_and_forget(
                self._post_request_background(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    trace_id=trace_id,
                    resolved_model_tag=resolved_model_tag,
                    provider=model_info.provider,
                    operation="tts",
                    input_tokens=len(request.input),
                    output_tokens=0,
                    compute_units=compute_units,
                    latency_ms=latency_ms,
                    billing_needs_ensure=not billing_existed,
                    budgets=budgets,
                )
            )

            # 6. Get content type and extract audio bytes
            content_type = AUDIO_FORMAT_TO_MIME.get(request.response_format, "audio/mpeg")
            audio_bytes = response.content if isinstance(response.content, bytes) else b""

            # 7. Upload to S3 (blocking — response needs the URL)
            if self._asset_storage is not None:
                asset = await self._asset_storage.upload_asset(
                    data=audio_bytes,
                    filename=f"speech.{request.response_format}",
                    content_type=content_type,
                    user_id=str(user_id),
                    trace_id=trace_id,
                )

                tts_response = TTSResponse(
                    url=asset.url,
                    expires_at=asset.expires_at,
                    content_type=asset.content_type,
                    size_bytes=asset.size_bytes,
                    cost=TTSCost(
                        model_name=resolved_model_tag,
                        provider=model_info.provider,
                        characters=len(request.input),
                        latency_ms=latency_ms,
                        compute_units=float(compute_units),
                    ),
                    routing=routing_metadata,
                )
            else:
                # Fallback if no storage configured
                logger.warning(
                    "AssetStorageService not configured, returning data URL",
                    trace_id=trace_id,
                )
                from datetime import datetime, timedelta

                b64_data = base64.b64encode(audio_bytes).decode("utf-8")
                data_url = f"data:{content_type};base64,{b64_data}"

                tts_response = TTSResponse(
                    url=data_url,
                    expires_at=datetime.now(UTC) + timedelta(hours=24),
                    content_type=content_type,
                    size_bytes=len(audio_bytes),
                    cost=TTSCost(
                        model_name=resolved_model_tag,
                        provider=model_info.provider,
                        characters=len(request.input),
                        latency_ms=latency_ms,
                        compute_units=float(compute_units),
                    ),
                    routing=routing_metadata,
                )

            return audio_bytes, tts_response

        except Exception as e:
            # Record failed request via fire-and-forget
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_code = type(e).__name__

            try:
                _, resolved_metadata = await self._router.resolve_meta_model(request.model, "tts")
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
                    operation="tts",
                    input_tokens=len(request.input),
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

    async def transcribe(
        self,
        user_id: int,
        api_key_id: int,
        audio_data: bytes,
        filename: str,
        request: TranscriptionRequest,
        model_policy: dict | None = None,
        budgets: list | None = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            user_id: User ID
            api_key_id: API key ID
            audio_data: Audio file bytes
            filename: Original filename (for MIME type detection)
            request: Transcription request

        Returns:
            TranscriptionResponse with transcribed text and cost info

        Raises:
            ModelNotFoundError: If model doesn't exist
            QuotaExceededError: If user is over quota
            ProviderError: If provider API fails
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
                operation="transcription",
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
            await self._policy.check_budgets(api_key_id, budgets, user_id=user_id)

            # Create file attachment from audio data
            audio_attachment = FileAttachment(
                type=FileType.AUDIO,
                source=FileSource.BASE64,
                data=base64.b64encode(audio_data).decode("utf-8"),
                filename=filename,
            )

            # 3. Execute request with circuit breaker
            async with breaker:
                response = await adapter.transcribe(
                    audio=audio_attachment,
                    model=provider_model_id,
                    language=request.language,
                )

            # 4. Calculate latency and compute units
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            estimated_duration = len(audio_data) / 16000

            cu = await self._pricing.calculate_compute_units(
                model_tag=resolved_model_tag,
                input_units=Decimal(str(estimated_duration)) / Decimal("60"),
                output_units=0,
                operation="transcription",
            )
            compute_units = cu

            # 5. Fire-and-forget: billing increment, usage recording
            self._fire_and_forget(
                self._post_request_background(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    trace_id=trace_id,
                    resolved_model_tag=resolved_model_tag,
                    provider=model_info.provider,
                    operation="transcription",
                    input_tokens=0,
                    output_tokens=0,
                    compute_units=compute_units,
                    latency_ms=latency_ms,
                    billing_needs_ensure=not billing_existed,
                    budgets=budgets,
                )
            )

            # 6. Build response (returned immediately)
            return TranscriptionResponse(
                text=response.content if isinstance(response.content, str) else "",
                language=response.language,
                duration=estimated_duration,
                cost=TranscriptionCost(
                    model_name=resolved_model_tag,
                    provider=model_info.provider,
                    duration_seconds=estimated_duration,
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
                    request.model, "transcription"
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
                    operation="transcription",
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
