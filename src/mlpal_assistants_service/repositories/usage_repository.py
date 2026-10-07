"""Repository for UsageLog access."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Numeric, cast, func, select

from mlpal_assistants_service.db.models import UsageLog
from mlpal_assistants_service.repositories.base import BaseRepository


class UsageRepository(BaseRepository[UsageLog]):
    """Repository for usage log operations."""

    model = UsageLog

    async def record_usage(
        self,
        user_id: int,
        api_key_id: int,
        trace_id: str,
        model_tag: str,
        provider: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        compute_units: Decimal = Decimal("0"),
        latency_ms: int | None = None,
        status: str = "success",
        error_code: str | None = None,
    ) -> UsageLog:
        """Record a new usage entry."""
        return await self.create(
            user_id=user_id,
            api_key_id=api_key_id,
            trace_id=trace_id,
            model_tag=model_tag,
            provider=provider,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compute_units=compute_units,
            latency_ms=latency_ms,
            status=status,
            error_code=error_code,
        )

    async def get_user_usage_by_attribution(
        self, user_id: int, field: str, value: str, limit: int = 200
    ) -> list[UsageLog]:
        """Rows this user's keys produced under one harness session or run
        (cc_metadata containment, served by the GIN index). `field` is a
        QUERYABLE_KEYS member; tenancy is the caller's user id, never the
        attribution."""
        stmt = (
            select(UsageLog)
            .where(UsageLog.user_id == user_id, UsageLog.cc_metadata.contains({field: value}))
            .order_by(UsageLog.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_trace_id(self, trace_id: str) -> UsageLog | None:
        """Get usage log by trace ID."""
        return await self.get_one(trace_id=trace_id)

    async def get_user_usage(
        self,
        user_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[UsageLog]:
        """Get usage logs for a user within a date range."""
        stmt = select(UsageLog).where(UsageLog.user_id == user_id)

        if start_date:
            stmt = stmt.where(UsageLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(UsageLog.created_at <= end_date)

        stmt = stmt.order_by(UsageLog.created_at.desc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_user_cu_total(
        self,
        user_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Decimal:
        """Get total compute units used by a user in a period."""
        stmt = select(func.coalesce(func.sum(UsageLog.compute_units), 0)).where(
            UsageLog.user_id == user_id,
            UsageLog.status == "success",
        )

        if start_date:
            stmt = stmt.where(UsageLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(UsageLog.created_at <= end_date)

        result = await self.session.execute(stmt)
        return Decimal(str(result.scalar_one()))

    async def get_user_cu_today(self, user_id: int) -> Decimal:
        """Get compute units used by a user today."""
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.get_user_cu_total(user_id, start_date=today)

    async def get_user_cu_this_month(self, user_id: int) -> Decimal:
        """Get compute units used by a user this month."""
        first_of_month = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return await self.get_user_cu_total(user_id, start_date=first_of_month)

    async def get_user_cu_in_window(
        self,
        user_id: int,
        start: datetime | None,
        end: datetime | None,
    ) -> Decimal:
        """Sum of successful compute units for an ACCOUNT in [start, end) —
        the reconcile source for the young-account daily ceiling."""
        stmt = select(func.coalesce(func.sum(UsageLog.compute_units), 0)).where(
            UsageLog.user_id == user_id,
            UsageLog.status == "success",
        )
        if start is not None:
            stmt = stmt.where(UsageLog.created_at >= start)
        if end is not None:
            stmt = stmt.where(UsageLog.created_at < end)
        return Decimal(str((await self.session.execute(stmt)).scalar_one()))

    async def get_api_key_cu_in_window(
        self,
        api_key_id: int,
        start: datetime | None,
        end: datetime | None,
    ) -> Decimal:
        """Sum of successful compute units for a key in [start, end) — the
        authoritative window spend the policy engine reconciles Redis against.
        `[start, end)` half-open so a boundary request isn't double-counted
        across adjacent windows. None bounds are open (lifetime = all time).
        Served by idx_usage_api_key (api_key_id, created_at)."""
        stmt = select(func.coalesce(func.sum(UsageLog.compute_units), 0)).where(
            UsageLog.api_key_id == api_key_id,
            UsageLog.status == "success",
        )
        if start is not None:
            stmt = stmt.where(UsageLog.created_at >= start)
        if end is not None:
            stmt = stmt.where(UsageLog.created_at < end)
        result = await self.session.execute(stmt)
        return Decimal(str(result.scalar_one()))

    async def get_api_key_usage(
        self,
        api_key_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[UsageLog]:
        """Get usage logs for an API key."""
        stmt = select(UsageLog).where(UsageLog.api_key_id == api_key_id)

        if start_date:
            stmt = stmt.where(UsageLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(UsageLog.created_at <= end_date)

        stmt = stmt.order_by(UsageLog.created_at.desc())

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_usage_summary_by_model(
        self,
        user_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Get usage summary grouped by model for a user."""
        stmt = select(
            UsageLog.model_tag,
            UsageLog.operation,
            func.count(UsageLog.id).label("request_count"),
            func.sum(UsageLog.input_tokens).label("total_input_tokens"),
            func.sum(UsageLog.output_tokens).label("total_output_tokens"),
            func.sum(UsageLog.compute_units).label("total_cu"),
            func.avg(UsageLog.latency_ms).label("avg_latency_ms"),
        ).where(
            UsageLog.user_id == user_id,
            UsageLog.status == "success",
        )

        if start_date:
            stmt = stmt.where(UsageLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(UsageLog.created_at <= end_date)

        stmt = stmt.group_by(UsageLog.model_tag, UsageLog.operation)

        result = await self.session.execute(stmt)
        return [
            {
                "model_tag": row.model_tag,
                "operation": row.operation,
                "request_count": row.request_count,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "total_cu": row.total_cu,
                "avg_latency_ms": row.avg_latency_ms,
            }
            for row in result
        ]

    async def get_latency_stats_by_model(
        self,
        days: int = 7,
        min_samples: int = 20,
    ) -> dict[str, dict]:
        """Per-model observed latency over recent successful chat traffic.

        Powers the catalog's latency attribute. Raw `latency_ms` is confounded by
        response length (a model asked for 2k tokens is "slower" than one asked
        for 50), so we also report the length-normalized ms-per-output-token,
        which is what makes models comparable to each other.

        Models with fewer than `min_samples` requests are omitted entirely —
        callers surface latency as unknown rather than publishing a figure drawn
        from a handful of requests.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        pct = func.percentile_cont(0.5).within_group(UsageLog.latency_ms.asc())
        pct95 = func.percentile_cont(0.95).within_group(UsageLog.latency_ms.asc())
        # Cast before dividing: both columns are integers and Postgres would do
        # integer division, truncating (19.397 -> 19) and tying close rankings.
        per_token = func.percentile_cont(0.5).within_group(
            (cast(UsageLog.latency_ms, Numeric) / func.nullif(UsageLog.output_tokens, 0)).asc()
        )
        stmt = (
            select(
                UsageLog.model_tag,
                func.count(UsageLog.id).label("samples"),
                pct.label("p50_ms"),
                pct95.label("p95_ms"),
                per_token.label("ms_per_output_token"),
            )
            .where(
                UsageLog.status == "success",
                UsageLog.operation == "chat",
                UsageLog.created_at >= since,
                UsageLog.latency_ms.isnot(None),
                UsageLog.output_tokens > 0,
            )
            .group_by(UsageLog.model_tag)
            .having(func.count(UsageLog.id) >= min_samples)
        )
        result = await self.session.execute(stmt)
        return {
            row.model_tag: {
                "samples": int(row.samples),
                "p50_ms": int(row.p50_ms) if row.p50_ms is not None else None,
                "p95_ms": int(row.p95_ms) if row.p95_ms is not None else None,
                "ms_per_output_token": (
                    round(float(row.ms_per_output_token), 2)
                    if row.ms_per_output_token is not None
                    else None
                ),
            }
            for row in result
        }

    async def get_recent_requests_per_minute(
        self,
        user_id: int,
        minutes: int = 1,
    ) -> int:
        """Get request count in the last N minutes (for rate limiting)."""
        since = datetime.now(UTC) - timedelta(minutes=minutes)

        stmt = select(func.count(UsageLog.id)).where(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= since,
        )

        result = await self.session.execute(stmt)
        return result.scalar_one()
