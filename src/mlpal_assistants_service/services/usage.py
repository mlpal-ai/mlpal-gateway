"""Usage tracking service."""

import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import redis.asyncio as redis
from sqlalchemy import Numeric, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.db.models import UsageLog

logger = logging.getLogger(__name__)


class UsageService:
    """
    Service for usage tracking and quota management.

    Usage is logged asynchronously via SQS to avoid blocking requests.
    Quota is tracked in Redis for fast access.
    """

    QUOTA_CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        session: AsyncSession,
        redis_client: redis.Redis | None = None,
        sqs_client: Any | None = None,
    ) -> None:
        self.session = session
        self.redis = redis_client
        self.settings = get_settings()
        self._sqs_client = sqs_client

    async def check_quota(
        self,
        user_id: str,
        estimated_cu: Decimal,
    ) -> bool:
        """
        Check if user has sufficient quota for a request.

        Args:
            user_id: User ID
            estimated_cu: Estimated compute units for the request

        Returns:
            True if quota is available, False otherwise
        """
        # Get current usage from Redis (fast path)
        current_usage = await self._get_monthly_usage_cached(user_id)

        # Get user's quota limit
        quota_limit = await self._get_user_quota_limit(user_id)

        return (current_usage + estimated_cu) <= quota_limit

    async def record_usage(
        self,
        user_id: str,
        api_key_id: str,
        trace_id: str,
        model_tag: str,
        provider: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        compute_units: Decimal,
        latency_ms: int | None = None,
        status: str = "success",
        error_code: str | None = None,
        wallet_debit_status: str = "not_applicable",
        wallet_debit_attempts: int = 0,
        wallet_debit_error: str | None = None,
        cc_metadata: dict | None = None,
    ) -> None:
        """
        Record usage for a request.

        This method updates Redis immediately and queues DB write via SQS.
        It returns immediately - the actual DB write is async.
        """
        # Update Redis quota counter immediately
        if self.redis:
            cache_key = f"quota:{user_id}:monthly"
            await self.redis.incrbyfloat(cache_key, float(compute_units))
            await self.redis.expire(cache_key, self.QUOTA_CACHE_TTL)

        # Queue usage record for async DB write
        usage_record = {
            "user_id": user_id,
            "api_key_id": api_key_id,
            "trace_id": trace_id,
            "model_tag": model_tag,
            "provider": provider,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "compute_units": str(compute_units),
            "latency_ms": latency_ms,
            "status": status,
            "error_code": error_code,
            "wallet_debit_status": wallet_debit_status,
            "wallet_debit_attempts": wallet_debit_attempts,
            "wallet_debit_error": wallet_debit_error,
            "cc_metadata": cc_metadata,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if wallet_debit_status == "pending":
            await self._write_usage_to_db(usage_record)
            await self._accrue_platform_fee(usage_record)
            return

        if self._sqs_client and self.settings.sqs_usage_queue_url:
            # Send to SQS asynchronously (aioboto3)
            try:
                await self._sqs_client.send_message(
                    QueueUrl=self.settings.sqs_usage_queue_url,
                    MessageBody=json.dumps(usage_record),
                )
            except Exception as e:
                logger.warning("SQS send failed, falling back to DB write", error=str(e))
                await self._write_usage_to_db(usage_record)
        else:
            # Fall back to direct DB write (for dev/testing)
            await self._write_usage_to_db(usage_record)

        # AFTER persistence so the DB-fallback month counter sees this row
        # (Redis path is exact either way; SQS path undercounts by one in-flight
        # request at most — negligible against a 300M threshold).
        await self._accrue_platform_fee(usage_record)

    async def _accrue_platform_fee(self, record: dict[str, Any]) -> None:
        """Monthly free-tier accounting (managed only; see services/platform_fee).
        Runs on the recording path — cheap Redis incr, rare insert, never raises."""
        from mlpal_assistants_service.services.platform_fee import (
            FEE_OPERATION,
            maybe_charge_platform_fee,
        )

        if record.get("status") != "success" or record.get("operation") == FEE_OPERATION:
            return
        # Cache READS don't count toward the free tier (founder decision,
        # 2026-08-29): our canonical input_tokens is the full prompt including
        # the cached portion, so subtract the cache-read count recorded in
        # cc_metadata. Cache WRITES stay counted — they're part of input.
        meta = record.get("cc_metadata") or {}
        cache_read = int(meta.get("cache_read_input_tokens") or 0)
        tokens = (
            int(record.get("input_tokens") or 0)
            + int(record.get("output_tokens") or 0)
            - cache_read
        )
        if tokens <= 0:
            return
        await maybe_charge_platform_fee(
            self.session, self.redis, int(record["user_id"]), tokens,
            api_key_id=int(record["api_key_id"]),
        )

    async def _write_usage_to_db(self, record: dict[str, Any]) -> None:
        """Write usage record directly to database."""
        usage_log = UsageLog(
            user_id=int(record["user_id"]),  # Ensure integer
            api_key_id=int(record["api_key_id"]),  # Ensure integer
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
            wallet_debit_status=record.get("wallet_debit_status", "not_applicable"),
            wallet_debit_attempts=record.get("wallet_debit_attempts", 0),
            wallet_debit_error=record.get("wallet_debit_error"),
            cc_metadata=record.get("cc_metadata"),
        )
        self.session.add(usage_log)
        await self.session.flush()

    async def mark_wallet_debit_status(
        self,
        trace_id: str,
        status: str,
        *,
        attempts: int = 1,
        error: str | None = None,
    ) -> None:
        result = await self.session.execute(
            select(UsageLog).where(UsageLog.trace_id == trace_id).limit(1)
        )
        usage_log = result.scalar_one_or_none()
        if usage_log is None:
            return

        usage_log.wallet_debit_status = status
        usage_log.wallet_debit_attempts = attempts
        usage_log.wallet_debit_error = error
        if status == "debited":
            usage_log.wallet_debited_at = datetime.now(UTC)

    async def _get_monthly_usage_cached(self, user_id: str) -> Decimal:
        """Get current month's usage from cache or database."""
        if self.redis:
            cache_key = f"quota:{user_id}:monthly"
            cached = await self.redis.get(cache_key)
            if cached:
                return Decimal(cached)

        # Load from database
        usage = await self._get_monthly_usage_from_db(user_id)

        # Cache it
        if self.redis:
            await self.redis.setex(
                f"quota:{user_id}:monthly",
                self.QUOTA_CACHE_TTL,
                str(usage),
            )

        return usage

    async def _get_monthly_usage_from_db(self, user_id: str) -> Decimal:
        """Load current month's usage from database."""
        start_of_month = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        result = await self.session.execute(
            select(func.coalesce(func.sum(UsageLog.compute_units), 0)).where(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= start_of_month,
            )
        )
        total = result.scalar_one()
        return Decimal(str(total))

    async def _get_user_quota_limit(self, user_id: str) -> Decimal:
        """
        Get user's monthly quota limit.

        Note: User data is stored in external MLPAL_USER_SCHEMA.
        This method returns a default limit. In production, this should
        query the external user table or use a cached value from Redis.
        """
        # TODO: Query external user schema for actual quota limit
        # For now, return default quota
        default_quota = Decimal("100.0")  # 100 compute units

        # Check Redis cache for user quota if available
        if self.redis:
            cache_key = f"user_quota:{user_id}"
            cached = await self.redis.get(cache_key)
            if cached:
                return Decimal(cached)

        return default_quota

    async def get_usage_summary(
        self,
        user_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get usage summary for a user.

        Includes a per-model breakdown via a second indexed query (uses
        ``idx_usage_model`` on ``(model_tag, created_at)``). Both queries
        run against the same window; on a 5K-row table the combined cost
        is under 10ms locally.
        """
        if start_date is None:
            start_date = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = datetime.now(UTC)

        from mlpal_assistants_service.services.platform_fee import month_progress

        progress = await month_progress(self.session, int(user_id))

        # Top-level aggregate.
        result = await self.session.execute(
            select(
                func.count(UsageLog.id).label("total_requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("total_output_tokens"),
                func.coalesce(func.sum(UsageLog.compute_units), 0).label("total_compute_units"),
            ).where(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= start_date,
                UsageLog.created_at <= end_date,
            )
        )
        row = result.one()

        # Per-model breakdown — same filter, group by model_tag.
        by_model_result = await self.session.execute(
            select(
                UsageLog.model_tag,
                func.count(UsageLog.id).label("requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(UsageLog.compute_units), 0).label("compute_units"),
            )
            .where(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= start_date,
                UsageLog.created_at <= end_date,
            )
            .group_by(UsageLog.model_tag)
            .order_by(func.sum(UsageLog.compute_units).desc())
        )
        by_model = {
            r.model_tag: {
                "model_tag": r.model_tag,
                "requests": int(r.requests),
                "input_tokens": int(r.input_tokens),
                "output_tokens": int(r.output_tokens),
                "compute_units": float(Decimal(str(r.compute_units))),
            }
            for r in by_model_result.all()
        }

        # Get quota info
        quota_limit = await self._get_user_quota_limit(user_id)
        total_cu = Decimal(str(row.total_compute_units))

        return {
            "period_start": start_date,
            "period_end": end_date,
            "total_requests": row.total_requests,
            "total_input_tokens": row.total_input_tokens,
            "total_output_tokens": row.total_output_tokens,
            "total_compute_units": float(total_cu),
            "quota_limit": float(quota_limit),
            "quota_remaining": float(quota_limit - total_cu),
            "by_model": by_model,
            **progress,
        }

    async def get_key_usage_summary(
        self,
        api_key_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Aggregate ``usage_logs`` filtered by a single ``api_key_id``.

        Caller is responsible for authz: confirm the key belongs to the
        requesting user before calling this. The query itself is pure
        aggregation and does not re-check ownership (the FK + idx_usage_api_key
        index make this fast, but exposing it without a caller check would
        leak any user's usage to anyone who guesses an id).

        Defaults to the current calendar month, matching get_usage_summary.
        """
        if start_date is None:
            start_date = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = datetime.now(UTC)

        success_filter = UsageLog.status == "success"
        # Observability metadata (nullable — older rows predate these fields):
        cache_read = func.cast(
            func.coalesce(UsageLog.cc_metadata["cache_read_input_tokens"].as_string(), "0"),
            Numeric,
        )
        cache_write = func.cast(
            func.coalesce(UsageLog.cc_metadata["cache_creation_input_tokens"].as_string(), "0"),
            Numeric,
        )
        ttft = func.cast(UsageLog.cc_metadata["ttft_ms"].as_string(), Numeric)
        is_stream = UsageLog.cc_metadata["stream"].as_string() == "true"

        result = await self.session.execute(
            select(
                func.count(UsageLog.id).label("total_requests"),
                func.count(UsageLog.id)
                .filter(success_filter)
                .label("success_requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("total_output_tokens"),
                func.coalesce(func.sum(UsageLog.compute_units), 0).label("total_compute_units"),
                func.max(UsageLog.created_at).label("last_used_at"),
                func.coalesce(func.sum(cache_read), 0).label("cache_read_tokens"),
                func.coalesce(func.sum(cache_write), 0).label("cache_write_tokens"),
                func.count(UsageLog.id).filter(is_stream).label("stream_requests"),
                func.percentile_cont(0.5)
                .within_group(UsageLog.latency_ms)
                .label("latency_p50_ms"),
                func.percentile_cont(0.95)
                .within_group(UsageLog.latency_ms)
                .label("latency_p95_ms"),
                func.percentile_cont(0.5).within_group(ttft).label("ttft_p50_ms"),
            ).where(
                UsageLog.api_key_id == api_key_id,
                UsageLog.created_at >= start_date,
                UsageLog.created_at <= end_date,
            )
        )
        row = result.one()

        total = int(row.total_requests)
        success = int(row.success_requests)

        total_input = int(row.total_input_tokens)
        cache_read_tokens = int(row.cache_read_tokens)
        return {
            "api_key_id": api_key_id,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": int(row.cache_write_tokens),
            # cached share of all input-side tokens (input_tokens CONTAINS the
            # cached portion) — the "is prompt caching working" number.
            "cache_hit_rate": (
                round(cache_read_tokens / total_input, 4) if total_input else 0.0
            ),
            "stream_requests": int(row.stream_requests),
            "latency_p50_ms": int(row.latency_p50_ms) if row.latency_p50_ms is not None else None,
            "latency_p95_ms": int(row.latency_p95_ms) if row.latency_p95_ms is not None else None,
            "ttft_p50_ms": int(row.ttft_p50_ms) if row.ttft_p50_ms is not None else None,
            "period_start": start_date,
            "period_end": end_date,
            "total_requests": total,
            "success_requests": success,
            "error_requests": total - success,
            "total_input_tokens": int(row.total_input_tokens),
            "total_output_tokens": int(row.total_output_tokens),
            "total_compute_units": float(Decimal(str(row.total_compute_units))),
            "last_used_at": row.last_used_at,
        }

    async def get_usage_daily(
        self,
        user_id: str | None = None,
        api_key_id: int | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Aggregate ``usage_logs`` into per-day buckets.

        Exactly one of ``user_id`` or ``api_key_id`` MUST be provided —
        passing both is allowed (extra narrowing); passing neither is a
        usage error and raises ValueError so we never accidentally return
        cross-user data.

        Empty days inside the window are NOT materialised in the result —
        zero-fill in the consumer if a contiguous series is needed. This
        keeps the response payload small for sparse usage patterns.

        ``date_trunc('day', created_at)`` is index-friendly given the
        existing ``idx_usage_user`` / ``idx_usage_api_key`` btree indexes,
        but Postgres still has to scan every row in the window; we keep
        the window capped at 90 days at the route layer to bound cost.
        """
        if user_id is None and api_key_id is None:
            raise ValueError(
                "get_usage_daily requires user_id or api_key_id (refusing to "
                "return cross-user data)"
            )

        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        day_bucket = func.date_trunc("day", UsageLog.created_at).label("day")

        stmt = (
            select(
                day_bucket,
                func.count(UsageLog.id).label("requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(UsageLog.compute_units), 0).label("compute_units"),
            )
            .where(UsageLog.created_at >= start_date, UsageLog.created_at <= end_date)
            .group_by(day_bucket)
            .order_by(day_bucket)
        )
        if user_id is not None:
            stmt = stmt.where(UsageLog.user_id == user_id)
        if api_key_id is not None:
            stmt = stmt.where(UsageLog.api_key_id == api_key_id)

        result = await self.session.execute(stmt)
        rows = result.all()

        daily: list[dict[str, Any]] = [
            {
                "date": row.day.date().isoformat(),
                "requests": int(row.requests),
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "compute_units": float(Decimal(str(row.compute_units))),
            }
            for row in rows
        ]
        total_cu = float(sum(Decimal(str(d["compute_units"])) for d in daily))

        return {
            "period_start": start_date,
            "period_end": end_date,
            "days": days,
            "total_compute_units": total_cu,
            "daily": daily,
        }
