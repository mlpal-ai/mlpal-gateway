"""Main FastAPI application."""

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlpal_assistants_service.adapters.factory import get_adapter_factory
from mlpal_assistants_service.api.mounting import mount_api
from mlpal_assistants_service.core.cache import CacheInvalidator
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    AccountHeldError,
    APIKeySuspendedError,
    AssistantsServiceError,
    ProviderError,
    WalletEmptyError,
    http_status_for_provider_error,
)
from mlpal_assistants_service.core.logging import configure_stdlib_logging
from mlpal_assistants_service.core.storage import AssetStorageService
from mlpal_assistants_service.core.telemetry import (
    instrument_app,
    setup_tracing,
    shutdown_tracing,
)
from mlpal_assistants_service.db.session import check_database_connection, engine
from mlpal_assistants_service.observability.middleware import (
    ObservabilityMiddleware,
)
from mlpal_assistants_service.schemas.common import ErrorDetail, ErrorResponse, HealthResponse
from mlpal_assistants_service.seams.egress_guard import EndpointRejected

logger = structlog.get_logger()
settings = get_settings()
configure_stdlib_logging(settings.log_level)


def _run_migrations() -> None:
    """Apply pending Alembic migrations on boot.

    Mirrors the pattern in mcp-service / storage-service / secrets-service:
    every fresh pod brings its DB schema to head on startup. Best-effort —
    a migration failure is logged but does NOT crash the pod (so a botched
    migration on one replica during a rolling update can't take the whole
    service down). The DDL will fail loudly on the next call that needs
    the missing column, which is the signal to investigate.
    """
    try:
        from alembic import command as alembic_command
        from alembic.config import Config as AlembicConfig

        alembic_cfg = AlembicConfig(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "alembic.ini",
            )
        )
        alembic_command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied")
    except Exception as e:
        logger.warning("Alembic migration skipped", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    logger.info("Starting application", environment=settings.environment)

    # Bring the schema to head before anything else touches the DB. Sync
    # call (Alembic is sync); takes <1s when there's nothing to apply.
    _run_migrations()

    # Initialize tracing
    tracer_provider = setup_tracing(debug=settings.otel_debug)
    if tracer_provider:
        logger.info("Tracing initialized")

    # Initialize Redis
    try:
        app.state.redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_max_connections,
            health_check_interval=settings.redis_health_check_interval,
        )
        await app.state.redis.ping()
        logger.info(
            "Redis connected",
            host=settings.redis_host,
            max_connections=settings.redis_max_connections,
        )
    except Exception as e:
        logger.warning("Redis not available, rate limiting disabled", error=str(e))
        app.state.redis = None

    # Dedicated pub/sub client with its own small pool, isolated from the
    # command pool above. A stuck or reconnecting cache-invalidation subscriber
    # must never be able to exhaust the connections that serve requests.
    app.state.cache_pubsub_redis = None
    if app.state.redis:
        try:
            app.state.cache_pubsub_redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.redis_pubsub_max_connections,
                health_check_interval=settings.redis_health_check_interval,
            )
            await app.state.cache_pubsub_redis.ping()
        except Exception as e:
            logger.warning("Redis pub/sub client unavailable", error=str(e))
            app.state.cache_pubsub_redis = None

    # Runtime setting overrides (console-set, DB-persisted) load BEFORE the
    # adapters so a stored backend-priority override shapes boot too.
    try:
        from mlpal_assistants_service.db.session import async_session_factory as _asf
        from mlpal_assistants_service.services import runtime_settings

        async with _asf() as _s:
            await runtime_settings.load(_s)
    except Exception as e:  # noqa: BLE001 — a bad row must not block boot
        logger.warning("Runtime settings load failed; using env values", error=str(e))

    # Initialize provider adapters via the factory: each family gets its
    # highest-priority CONFIGURED backend (first-party key, or Azure/Vertex/
    # Bedrock per MLPAL_<FAMILY>_BACKENDS), so a self-hosted box boots with
    # whatever clouds it has. Building them here also warms the factory's
    # backend caches before traffic arrives.
    app.state.adapters = get_adapter_factory().get_enabled()
    logger.info(
        "Provider adapters initialized",
        providers={p: a.backend_name for p, a in app.state.adapters.items()},
    )

    # Catalog-feed subscriber: no-op in bundled mode; in hosted mode it pulls
    # the feed on an interval and reconciles the catalog (runtime-toggleable
    # via /admin/v1/catalog/feed — the loop re-reads the mode every cycle).
    from mlpal_assistants_service.db.session import async_session_factory
    from mlpal_assistants_service.services.catalog_feed import (
        apply_bundled_if_new,
        start_subscriber,
    )

    # Catalog-at-boot: when this build ships a NEWER bundled catalog than the
    # DB has applied, reconcile it now (version-gated + idempotent — racing
    # pods no-op). This is what makes "merge the scout PR → deploy" reach the
    # DB with no manual step, for prod and every bundled-mode box alike.
    try:
        applied = await apply_bundled_if_new(async_session_factory)
        if applied:
            logger.info("Bundled catalog reconciled at boot", **applied)
    except Exception as e:  # noqa: BLE001 — catalog refresh must not block boot
        logger.warning("Bundled catalog reconcile failed; serving existing DB", error=str(e))

    start_subscriber(async_session_factory, app.state.redis)

    # Initialize asset storage (S3)
    try:
        app.state.asset_storage = AssetStorageService(
            bucket=settings.s3_assets_bucket,
            region=settings.s3_assets_region,
            url_expiration=settings.asset_url_expiration,
        )
        # Verify S3 connectivity
        if await app.state.asset_storage.health_check():
            logger.info(
                "Asset storage initialized",
                bucket=settings.s3_assets_bucket,
                region=settings.s3_assets_region,
                url_expiration=settings.asset_url_expiration,
            )
        else:
            logger.warning(
                "Asset storage health check failed, generated files will use fallback",
                bucket=settings.s3_assets_bucket,
            )
    except Exception as e:
        logger.warning(
            "Asset storage initialization failed, generated files will use fallback",
            error=str(e),
        )
        app.state.asset_storage = None

    # Initialize SQS client (aioboto3) for async usage logging
    app.state.sqs_client = None
    if settings.sqs_usage_queue_url:
        try:
            import aioboto3

            sqs_session = aioboto3.Session()
            endpoint_kwargs = {}
            if settings.aws_endpoint_url:
                endpoint_kwargs["endpoint_url"] = settings.aws_endpoint_url
            sqs_client = await sqs_session.client(
                "sqs",
                region_name=settings.aws_region,
                **endpoint_kwargs,
            ).__aenter__()
            # Create queue idempotently (extracts queue name from URL)
            queue_name = settings.sqs_usage_queue_url.rstrip("/").split("/")[-1]
            await sqs_client.create_queue(QueueName=queue_name)
            app.state.sqs_client = sqs_client
            logger.info("SQS client initialized", queue=settings.sqs_usage_queue_url)
        except Exception as e:
            logger.warning("SQS initialization failed, using DB fallback", error=str(e))
            app.state.sqs_client = None

    # Initialize cache invalidator (Redis Pub/Sub for cross-instance cache clearing)
    app.state.cache_invalidator = None
    if app.state.redis:
        app.state.cache_invalidator = CacheInvalidator(
            app.state.redis,
            pubsub_client=app.state.cache_pubsub_redis,
        )

    # Initialize shared in-memory caches (these will be populated on first access
    # and shared across all request handlers to avoid per-request DB hits)
    from mlpal_assistants_service.core.cache import TTLCache

    app.state.model_cache = TTLCache(
        default_ttl=settings.local_cache_ttl,
        maxsize=settings.local_cache_maxsize,
    )
    app.state.routing_cache = TTLCache(
        default_ttl=settings.local_cache_ttl,
        maxsize=settings.local_cache_maxsize,
    )
    app.state.pricing_cache = TTLCache(
        default_ttl=settings.local_cache_ttl,
        maxsize=settings.local_cache_maxsize,
    )
    logger.info(
        "Shared caches initialized",
        local_cache_ttl=settings.local_cache_ttl,
        local_cache_maxsize=settings.local_cache_maxsize,
    )

    # Clear schema-dependent Redis caches on startup
    # This ensures fresh data after deployments with schema changes
    # Only clears: pricing:*, model:*, models_list:*
    # Does NOT clear: auth:* (API keys), credits:*, quota:* (runtime data)
    if app.state.redis:
        cache_patterns = ["pricing:*", "model:*", "models_list:*"]
        total_deleted = 0
        for pattern in cache_patterns:
            cursor = 0
            while True:
                cursor, keys = await app.state.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await app.state.redis.delete(*keys)
                    total_deleted += len(keys)
                if cursor == 0:
                    break
        if total_deleted > 0:
            logger.info(
                "Cleared schema-dependent caches on startup",
                patterns=cache_patterns,
                keys_deleted=total_deleted,
            )

    # Background tasks
    background_tasks: list[asyncio.Task] = []

    # Start background task to flush last_used_at from Redis to DB
    if app.state.redis:

        async def _flush_last_used_loop():
            """Periodically flush API key last_used_at from Redis to DB."""
            from mlpal_assistants_service.db.session import async_session_factory
            from mlpal_assistants_service.services.api_key import APIKeyService

            interval = settings.api_key_last_used_flush_interval
            while True:
                await asyncio.sleep(interval)
                try:
                    async with async_session_factory() as session:
                        svc = APIKeyService(session, app.state.redis)
                        count = await svc.flush_last_used()
                        if count > 0:
                            logger.info("Flushed last_used_at", count=count)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("last_used_at flush failed", error=str(e))

        background_tasks.append(asyncio.create_task(_flush_last_used_loop()))
        logger.info(
            "Background last_used_at flush started",
            interval=settings.api_key_last_used_flush_interval,
        )

    # Start background routing table refresh
    if app.state.redis:

        async def _refresh_routing_loop():
            """Periodically refresh meta-model routing table into in-memory cache."""
            from mlpal_assistants_service.db.session import async_session_factory
            from mlpal_assistants_service.services.router import ModelRouter

            interval = settings.routing_refresh_interval

            # Initial load on startup
            try:
                async with async_session_factory() as session:
                    router = ModelRouter(session, app.state.redis)
                    # Use the shared caches created above
                    router._routing_cache = app.state.routing_cache
                    router._local_cache = app.state.model_cache
                    count = await router.load_routing_table()
                    logger.info("Initial routing table loaded", count=count)
            except Exception as e:
                logger.error("Initial routing table load failed", error=str(e))

            while True:
                await asyncio.sleep(interval)
                try:
                    async with async_session_factory() as session:
                        router = ModelRouter(session, app.state.redis)
                        # Use the shared caches
                        router._routing_cache = app.state.routing_cache
                        router._local_cache = app.state.model_cache
                        count = await router.load_routing_table()
                        if count > 0:
                            logger.debug("Routing table refreshed", count=count)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Routing table refresh failed", error=str(e))

        background_tasks.append(asyncio.create_task(_refresh_routing_loop()))
        logger.info(
            "Background routing refresh started",
            interval=settings.routing_refresh_interval,
        )

    # Start SQS usage consumer (continuous long polling, batch writes to DB)
    if app.state.sqs_client and settings.sqs_usage_queue_url:

        async def _sqs_usage_consumer_loop():
            """Continuous long-poll SQS for usage records and batch write to DB.

            Uses long polling (WaitTimeSeconds=20) which is the AWS best practice:
            - When queue is empty: waits up to 20s before returning (cost-effective)
            - When messages arrive: returns immediately (low latency)
            - ~3 requests/min when idle vs 120/min with short polling
            """
            import json
            from decimal import Decimal

            from mlpal_assistants_service.db.models import UsageLog
            from mlpal_assistants_service.db.session import async_session_factory

            batch_size = min(settings.sqs_usage_batch_size, 10)  # SQS max is 10
            wait_time = min(settings.sqs_long_poll_wait, 20)  # SQS max is 20
            queue_url = settings.sqs_usage_queue_url

            while True:
                try:
                    # Long poll for messages (waits up to wait_time seconds)
                    response = await app.state.sqs_client.receive_message(
                        QueueUrl=queue_url,
                        MaxNumberOfMessages=batch_size,
                        WaitTimeSeconds=wait_time,
                    )

                    messages = response.get("Messages", [])
                    if not messages:
                        continue  # Long poll returned empty, loop again

                    # Parse and write to DB
                    async with async_session_factory() as session:
                        receipt_handles = []
                        for msg in messages:
                            try:
                                record = json.loads(msg["Body"])
                                usage_log = UsageLog(
                                    user_id=int(record["user_id"]),
                                    api_key_id=int(record["api_key_id"]),
                                    trace_id=record["trace_id"],
                                    model_tag=record["model_tag"],
                                    provider=record["provider"],
                                    operation=record["operation"],
                                    input_tokens=record["input_tokens"],
                                    output_tokens=record["output_tokens"],
                                    compute_units=Decimal(record["compute_units"]),
                                    latency_ms=record.get("latency_ms"),
                                    status=record["status"],
                                    error_code=record.get("error_code"),
                                    wallet_debit_status=record.get(
                                        "wallet_debit_status",
                                        "not_applicable",
                                    ),
                                    wallet_debit_attempts=record.get(
                                        "wallet_debit_attempts",
                                        0,
                                    ),
                                    wallet_debit_error=record.get(
                                        "wallet_debit_error",
                                    ),
                                    # Carry Claude-Code attribution + provider
                                    # message id through the SQS path too; the
                                    # producer always sends it but the consumer
                                    # was dropping it (NULL cc_metadata in prod).
                                    cc_metadata=record.get("cc_metadata"),
                                )
                                session.add(usage_log)
                                receipt_handles.append(msg["ReceiptHandle"])
                            except Exception as e:
                                logger.warning(
                                    "Failed to parse SQS message",
                                    error=str(e),
                                    message_id=msg.get("MessageId"),
                                )

                        if receipt_handles:
                            await session.commit()

                            # Delete processed messages from SQS
                            for handle in receipt_handles:
                                await app.state.sqs_client.delete_message(
                                    QueueUrl=queue_url,
                                    ReceiptHandle=handle,
                                )

                            logger.info(
                                "Flushed usage records from SQS",
                                count=len(receipt_handles),
                            )

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("SQS usage consumer error", error=str(e))
                    await asyncio.sleep(5)  # Back off on error

        background_tasks.append(asyncio.create_task(_sqs_usage_consumer_loop()))
        logger.info(
            "Background SQS usage consumer started",
            long_poll_wait=settings.sqs_long_poll_wait,
            batch_size=settings.sqs_usage_batch_size,
        )

    async def _wallet_debit_retry_loop():
        """Replay retryable wallet debit intents in the background."""
        from mlpal_assistants_service.db.session import async_session_factory
        from mlpal_assistants_service.services.debit_retry_worker import DebitRetryWorker

        while True:
            try:
                async with async_session_factory() as session:
                    worker = DebitRetryWorker(session, app.state.redis)
                    processed = await worker.run_once()
                    if processed:
                        logger.info("Wallet debit retry batch processed", count=processed)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Wallet debit retry loop failed", error=str(e))

            await asyncio.sleep(settings.wallet_retry_interval_seconds)

    # Payload-capture retention: purge expired payloads daily. Idempotent
    # delete; cheap no-op while capture has never been enabled.
    async def _payload_retention_loop():
        from mlpal_assistants_service.db.session import async_session_factory
        from mlpal_assistants_service.services.capture import capture_state, purge_expired

        while True:
            try:
                cfg = await capture_state.config(app.state.redis)
                async with async_session_factory() as session:
                    purged = await purge_expired(session, cfg.retention_days)
                if purged:
                    logger.info("Purged expired payloads", count=purged)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — retention must never kill the app
                logger.warning("Payload retention pass failed", error=str(e))
            await asyncio.sleep(24 * 3600)

    background_tasks.append(asyncio.create_task(_payload_retention_loop()))

    # Abuse detector: one tick per interval across the fleet (Redis lock; a
    # single box without Redis just runs it), plus a bounded client_ip scrub
    # once the rows pass retention. Findings are rows + audit log + metric;
    # enforcement is a flag (observe first).
    async def _abuse_detector_loop():
        from datetime import timedelta

        from mlpal_assistants_service.db.session import async_session_factory
        from mlpal_assistants_service.services.abuse_detector import AbuseDetector, scrub_client_ips

        interval = settings.abuse_detector_interval_seconds
        lock_ttl = max(interval - 10, 30)
        while True:
            await asyncio.sleep(interval)
            try:
                if app.state.redis is not None:
                    got = await app.state.redis.set("abuse:detector:lock", "1", nx=True, ex=lock_ttl)
                    if not got:
                        continue
                async with async_session_factory() as session:
                    findings = await AbuseDetector(
                        session, app.state.redis, enforce=settings.abuse_enforce
                    ).run_once()
                    scrubbed = await scrub_client_ips(
                        session, older_than=timedelta(days=settings.client_ip_retention_days)
                    )
                if findings or scrubbed:
                    logger.info(
                        "Abuse detector tick", findings=len(findings), scrubbed_ips=scrubbed,
                        enforce=settings.abuse_enforce,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — detection must never kill the app
                logger.error("Abuse detector tick failed", error=str(e))

    if settings.abuse_detector_enabled:
        background_tasks.append(asyncio.create_task(_abuse_detector_loop()))
        logger.info(
            "Abuse detector started",
            interval=settings.abuse_detector_interval_seconds, enforce=settings.abuse_enforce,
        )

    # Wallet debits are a managed-billing concern; in local (self-hosted) mode
    # there is no payments service to call, so don't spin a retry loop that
    # fails every interval against a blank URL.
    if settings.billing_backend != "local":
        background_tasks.append(asyncio.create_task(_wallet_debit_retry_loop()))
        logger.info(
            "Background wallet debit retry started",
            interval=settings.wallet_retry_interval_seconds,
        )

    # Start cache invalidation Pub/Sub listener
    if app.state.cache_invalidator:
        invalidator = app.state.cache_invalidator

        # Register handlers for each cache target
        async def _clear_model_cache():
            if hasattr(app.state, "model_cache"):
                await app.state.model_cache.clear()
                logger.info("Model cache cleared via invalidation")

        async def _clear_routing_cache():
            if hasattr(app.state, "routing_cache"):
                await app.state.routing_cache.clear()
                logger.info("Routing cache cleared via invalidation")

        async def _clear_pricing_cache():
            if hasattr(app.state, "pricing_cache"):
                await app.state.pricing_cache.clear()
                logger.info("Pricing cache cleared via invalidation")

        async def _clear_api_key_cache(target: str | None = None):
            """Clear a specific API key or log the event."""
            if target and ":" in target:
                key_hash = target.split(":", 1)[1]
                if app.state.redis:
                    try:
                        await app.state.redis.delete(f"auth:{key_hash}")
                        logger.info("API key cache cleared via invalidation", key_hash=key_hash[:8])
                    except Exception as e:
                        logger.warning("Failed to clear API key cache", error=str(e))
            else:
                logger.info("API key cache invalidation received (all keys)")

        async def _reload_runtime_settings():
            from mlpal_assistants_service.db.session import async_session_factory
            from mlpal_assistants_service.services import runtime_settings

            async with async_session_factory() as s:
                await runtime_settings.load(s)
            logger.info("Runtime settings reloaded via invalidation")

        invalidator.register("models", _clear_model_cache)
        invalidator.register("pricing", _clear_pricing_cache)
        invalidator.register("routing", _clear_routing_cache)
        invalidator.register("api_key", _clear_api_key_cache)
        invalidator.register("api_keys", _clear_api_key_cache)
        invalidator.register("settings", _reload_runtime_settings)

        async def _invalidate_connections():
            from mlpal_assistants_service.services.connections import invalidate_overlay

            invalidate_overlay()
            logger.info("BYOK overlay invalidated via pub/sub")

        invalidator.register("connections", _invalidate_connections)

        await invalidator.start_listener()
        logger.info("Cache invalidation listener started")

    yield

    # Cleanup
    logger.info("Shutting down application")

    # Cancel all background tasks
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Stop cache invalidation listener
    if app.state.cache_invalidator:
        await app.state.cache_invalidator.stop()

    if getattr(app.state, "cache_pubsub_redis", None):
        await app.state.cache_pubsub_redis.aclose()

    if getattr(app.state, "sqs_client", None):
        await app.state.sqs_client.__aexit__(None, None, None)

    if app.state.redis:
        await app.state.redis.close()

    if getattr(app.state, "asset_storage", None):
        await app.state.asset_storage.close()

    await engine.dispose()

    # Shutdown tracing (flush pending spans)
    shutdown_tracing()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="MLpal Assistants API - Unified AI Gateway",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Middleware order matters: Starlette runs the LAST-added middleware as the
# OUTERMOST layer. Observability is added first (inner layer); CORS is added
# last so it wraps everything and answers OPTIONS preflights before anything
# else runs.

# Observability (pure ASGI — no task-per-response, no body re-parse; see
# observability/middleware.py for why BaseHTTPMiddleware is banned here)
app.add_middleware(ObservabilityMiddleware)

# CORS LAST = outermost: answers OPTIONS preflights up front and attaches CORS
# headers to every response, including errors raised by inner middleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument app for tracing (FastAPI, SQLAlchemy, Redis, httpx)
instrument_app(app, engine)


# Exception handlers
@app.exception_handler(APIKeySuspendedError)
async def suspended_handler(request: Request, exc: APIKeySuspendedError) -> JSONResponse:
    """Account suspended: one 403 body every client understands — the
    console reads {code, message}, OpenAI-style clients read `detail`,
    Anthropic SDKs read {type: error, error: {type, message}}."""
    return JSONResponse(
        status_code=403,
        content={
            "code": "account_suspended",
            "message": exc.message,
            "detail": exc.message,
            "type": "error",
            "error": {"type": "permission_error", "code": "account_suspended", "message": exc.message},
        },
    )


@app.exception_handler(AccountHeldError)
async def held_handler(request: Request, exc: AccountHeldError) -> JSONResponse:
    """Signup-risk hold: same envelope as suspension so the console has one
    renderer; `code` is the stable field (risk_hold), `message` is shown verbatim."""
    return JSONResponse(
        status_code=403,
        content={
            "code": "risk_hold",
            "message": exc.message,
            "detail": exc.message,
            "type": "error",
            "error": {"type": "permission_error", "code": "risk_hold", "message": exc.message},
        },
    )


@app.exception_handler(EndpointRejected)
async def egress_rejected_handler(request: Request, exc: EndpointRejected) -> JSONResponse:
    """A caller-supplied URL (reference image, attachment, audio) failed the
    egress policy: it is the caller's input, so 400 with the user-safe reason —
    never a silently dropped attachment, never a 5xx."""
    return JSONResponse(
        status_code=400,
        content={"error": {"code": "url_rejected", "message": str(exc), "details": None}},
    )


@app.exception_handler(WalletEmptyError)
async def wallet_empty_handler(request: Request, exc: WalletEmptyError) -> JSONResponse:
    """OpenAI-wire 402: stable machine keys for clients to render a top-up
    prompt (Anthropic-wire surfaces map this in messages_v2 instead)."""
    return JSONResponse(
        status_code=402,
        content={
            "error": {
                "message": exc.message,
                "type": "insufficient_quota",
                "code": "wallet_empty",
            }
        },
    )


@app.exception_handler(AssistantsServiceError)
async def service_exception_handler(
    request: Request,
    exc: AssistantsServiceError,
) -> JSONResponse:
    """Handle service-specific exceptions."""
    logger.error(
        "Service error",
        error=exc.message,
        details=exc.details,
        path=request.url.path,
    )

    # Provider errors carry the upstream HTTP status — map it precisely so a
    # client's bad request (provider 400) isn't reported as a gateway outage.
    if isinstance(exc, ProviderError):
        status_code = http_status_for_provider_error(exc)
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=exc.__class__.__name__,
                    message=exc.message,
                    details=exc.details,
                )
            ).model_dump(),
        )

    # Prefer an explicit status the exception declares (e.g. policy engine's
    # 403/402); fall back to message-substring heuristics for legacy exceptions.
    if getattr(exc, "http_status", None):
        status_code = exc.http_status
    elif "not found" in exc.message.lower():
        status_code = status.HTTP_404_NOT_FOUND
    elif "unauthorized" in exc.message.lower() or "invalid" in exc.message.lower():
        status_code = status.HTTP_401_UNAUTHORIZED
    elif "forbidden" in exc.message.lower():
        status_code = status.HTTP_403_FORBIDDEN
    elif "rate limit" in exc.message.lower() or "quota" in exc.message.lower():
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=exc.__class__.__name__,
                message=exc.message,
                details=exc.details,
            )
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code="ValidationError",
                message="Request validation failed",
                details={"errors": exc.errors()},
            )
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Last-resort handler: every unhandled exception must leave a log line.

    Without this, a 500 shows up in metrics but nowhere in the logs (the
    2026-07-21 /v1/keys incident) — undiagnosable after the fact. The
    traceback is logged here; the client gets a stable opaque error.
    """
    logger.error(
        "Unhandled exception",
        method=request.method,
        path=request.url.path,
        exc_info=exc,
    )
    # This handler runs at ServerErrorMiddleware — OUTSIDE CORSMiddleware — so
    # a 500 would otherwise reach the browser without CORS headers and surface
    # as an opaque "failed to fetch" instead of the actual error. Echo the
    # Origin (we allow all origins) so cross-origin clients can read the 500.
    origin = request.headers.get("origin")
    cors_headers = (
        {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code="InternalServerError",
                message="Internal server error",
            )
        ).model_dump(),
        headers=cors_headers,
    )


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
    )


@app.get("/health/ready", response_model=HealthResponse, tags=["Health"])
async def ready(request: Request) -> HealthResponse:
    """Readiness check - verifies all dependencies."""
    checks = {}

    # Check database
    checks["database"] = await check_database_connection()

    # Check Redis
    if request.app.state.redis:
        try:
            await request.app.state.redis.ping()
            checks["redis"] = True
        except Exception:
            checks["redis"] = False
    else:
        checks["redis"] = False

    # Check asset storage (S3)
    if getattr(request.app.state, "asset_storage", None):
        checks["asset_storage"] = await request.app.state.asset_storage.health_check()
    else:
        checks["asset_storage"] = False

    # Check provider adapters in parallel (with 2s timeout total)
    # Run all provider checks concurrently to avoid sequential timeouts
    adapters = getattr(request.app.state, "adapters", {})
    if adapters:

        async def check_provider(name: str, adapter) -> tuple[str, bool]:
            try:
                result = await asyncio.wait_for(adapter.health_check(), timeout=2.0)
                return (name, result)
            except TimeoutError:
                logger.warning(f"Provider health check timed out: {name}")
                return (name, False)
            except Exception as e:
                logger.warning(f"Provider health check failed: {name}", error=str(e))
                return (name, False)

        provider_results = await asyncio.gather(
            *[check_provider(name, adapter) for name, adapter in adapters.items()]
        )
        for provider_name, result in provider_results:
            checks[f"provider_{provider_name}"] = result

    # Core services must be healthy (db, redis)
    # Providers can be degraded without failing readiness
    core_healthy = checks.get("database", False) and checks.get("redis", False)

    body = HealthResponse(
        status="ready" if core_healthy else "not_ready",
        version=settings.app_version,
        checks=checks,
    )
    if not core_healthy:
        # Probes (k8s readinessProbe, compose healthcheck, LB target groups)
        # look at the STATUS CODE — a 200 "not_ready" keeps a dead-DB pod in
        # rotation serving 500s.
        return JSONResponse(status_code=503, content=body.model_dump())
    return body


@app.get("/health/live", tags=["Health"])
async def live() -> dict:
    """Liveness check - just confirms the service is running."""
    return {"status": "alive"}


# Attach the full route surface at the composition root. One place decides the
# versioning policy and the messages seam (see api/mounting.py).
mount_api(app, settings)


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "disabled",
        "api": settings.api_v1_prefix,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mlpal_assistants_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
