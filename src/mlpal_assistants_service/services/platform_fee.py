"""Monthly platform fee: one 5-CU charge when a user crosses the free tier.

Locked pricing (planning/pricing-DECISION.md): the managed service is free up
to 300M tokens/calendar month; past that, a flat fee — implemented as ONE
synthetic usage row per (user, month) so the entire billing pipeline
(aggregates → wallet debit → UI) stays singular. Contract confirmed with the
backend session 2026-08-11: operation='platform_fee', model_tag=
'platform-fee-300m', provider='mlpal', compute_units=5, zero tokens,
deterministic trace_id `fee_{user_id}_{YYYYMM}`.

Idempotency: cheap existence pre-check for the deterministic trace_id, with
the partial unique index (migration 20260811_2200) as the concurrency
backstop — a raced double-insert loses on the constraint and is swallowed.

Self-hosted deployments never charge this (billing_backend == 'local'):
self-hosting is free by design.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import Numeric, func, select
from sqlalchemy.exc import IntegrityError

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.db.models.usage_log import UsageLog

logger = logging.getLogger(__name__)

FEE_OPERATION = "platform_fee"
FEE_MODEL_TAG = "platform-fee-300m"

_COUNTER_TTL = 40 * 24 * 3600  # outlives any month, cheap to rebuild


def _month_id(settings: Any) -> str:
    tz = ZoneInfo(getattr(settings, "budget_timezone", "UTC") or "UTC")
    return datetime.now(tz).strftime("%Y%m")


def fee_trace_id(user_id: int, month_id: str) -> str:
    return f"fee_{user_id}_{month_id}"


async def maybe_charge_platform_fee(
    session: Any,
    redis: Any,
    user_id: int,
    tokens_delta: int,
    api_key_id: int,
) -> bool:
    """Accrue this request's tokens onto the user's monthly counter and charge
    the one-time fee on crossing the threshold. Returns True if a fee row was
    inserted now. Never raises — billing accrual must not fail requests.
    """
    settings = get_settings()
    if getattr(settings, "billing_backend", "managed") == "local":
        return False
    threshold = settings.platform_fee_threshold_tokens
    if threshold <= 0:  # 0 disables the fee entirely
        return False

    try:
        month = _month_id(settings)
        total = await _monthly_tokens(session, redis, user_id, month, tokens_delta)
        if total < threshold:
            return False

        trace_id = fee_trace_id(user_id, month)
        exists = (
            await session.execute(
                select(UsageLog.trace_id).where(UsageLog.trace_id == trace_id).limit(1)
            )
        ).scalar_one_or_none()
        if exists is not None:
            return False

        session.add(
            UsageLog(
                user_id=user_id,
                # NOT NULL column: attribute the fee to the key whose request
                # crossed the threshold. Billing aggregates per-user anyway.
                api_key_id=api_key_id,
                trace_id=trace_id,
                model_tag=FEE_MODEL_TAG,
                provider="mlpal",
                operation=FEE_OPERATION,
                input_tokens=0,
                output_tokens=0,
                compute_units=Decimal(str(settings.platform_fee_cu)),
                latency_ms=None,
                status="success",
                wallet_debit_status="not_applicable",
                cc_metadata={
                    "api": FEE_OPERATION,
                    "threshold_tokens": threshold,
                    "month": f"{month[:4]}-{month[4:]}",
                },
            )
        )
        try:
            await session.commit()
        except IntegrityError:  # raced by another instance — constraint backstop
            await session.rollback()
            return False
        logger.info(
            "platform fee charged", extra={"user_id": user_id, "month": month}
        )
        return True
    except Exception:  # noqa: BLE001 — accrual must never break serving
        logger.exception("platform-fee accrual failed for user %s", user_id)
        return False


async def _monthly_tokens(
    session: Any, redis: Any, user_id: int, month: str, tokens_delta: int
) -> int:
    """Redis counter (re-seeded from usage_logs on miss), same pattern as the
    budget counters. Falls back to a direct DB sum without Redis."""
    key = f"fee_tokens:{user_id}:{month}"
    if redis is not None:
        try:
            total = await redis.incrby(key, int(tokens_delta))
            if total == int(tokens_delta):  # fresh key — re-seed from the ledger
                db_total = await _db_month_tokens(session, user_id)
                if db_total > tokens_delta:
                    await redis.incrby(key, int(db_total - tokens_delta))
                    total = db_total
            await redis.expire(key, _COUNTER_TTL)
            return int(total)
        except Exception:  # noqa: BLE001 — Redis down: fall through to DB
            logger.warning("platform-fee counter failed for %s", key, exc_info=True)
    return await _db_month_tokens(session, user_id)


async def _db_month_tokens(session: Any, user_id: int) -> int:
    settings = get_settings()
    tz = ZoneInfo(getattr(settings, "budget_timezone", "UTC") or "UTC")
    now = datetime.now(tz)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Mirror the accrual path: cache READS are excluded from the free-tier
    # count (input_tokens includes the cached portion; the read count lives in
    # cc_metadata). Rows without the field predate caching metadata — 0.
    cache_read = func.cast(
        func.coalesce(UsageLog.cc_metadata["cache_read_input_tokens"].as_string(), "0"),
        Numeric,
    )
    total = (
        await session.execute(
            select(
                func.coalesce(
                    func.sum(
                        UsageLog.input_tokens + UsageLog.output_tokens - cache_read
                    ),
                    0,
                )
            ).where(
                UsageLog.user_id == user_id,
                UsageLog.created_at >= month_start,
                UsageLog.operation != FEE_OPERATION,
            )
        )
    ).scalar_one()
    return int(total)


async def month_progress(session: Any, user_id: int) -> dict[str, Any]:
    """For /v1/usage/summary: monthly token total, the threshold, and whether
    this month's fee has been charged."""
    settings = get_settings()
    tokens = await _db_month_tokens(session, user_id)
    month = _month_id(settings)
    charged = (
        await session.execute(
            select(UsageLog.trace_id)
            .where(UsageLog.trace_id == fee_trace_id(user_id, month))
            .limit(1)
        )
    ).scalar_one_or_none() is not None
    threshold = settings.platform_fee_threshold_tokens
    return {
        "monthly_tokens": tokens,
        "monthly_token_limit": threshold if threshold > 0 else None,
        "platform_fee_charged": charged,
    }
