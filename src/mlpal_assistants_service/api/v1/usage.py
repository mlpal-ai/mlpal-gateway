"""Usage tracking endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from mlpal_assistants_service.api.deps import CurrentUserFlexible, UsageServiceDep
from mlpal_assistants_service.db.models.usage_log import UsageLog
from mlpal_assistants_service.repositories.usage_repository import UsageRepository
from mlpal_assistants_service.schemas.usage import (
    AttributedUsageRecord,
    AttributedUsageResponse,
    DailyUsageResponse,
    UsageSummary,
)
from mlpal_assistants_service.services.attribution import attribution_fields
from mlpal_assistants_service.services.capture import fetch_payload
from mlpal_assistants_service.services.traces import query_traces, window_from_days

router = APIRouter()


@router.get(
    "",
    response_model=AttributedUsageResponse,
    summary="Usage by harness session or run",
    description=(
        "Rows and summed compute units for one harness session (`session_id`) "
        "or one run (`run_id`), as carried on the request's W3C baggage "
        "(mlpal.session / mlpal.run). Exactly one filter is required. Scoped to "
        "the caller's own keys."
    ),
)
async def get_usage_by_attribution(
    user_id: CurrentUserFlexible,
    usage_service: UsageServiceDep,
    session_id: str | None = Query(default=None, max_length=64),
    run_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=200, ge=1, le=1000),
) -> AttributedUsageResponse:
    if (session_id is None) == (run_id is None):
        raise HTTPException(status_code=400, detail="Pass exactly one of session_id or run_id")
    field, value = ("session_id", session_id) if session_id is not None else ("run_id", run_id)
    repo = UsageRepository(usage_service.session)
    rows = await repo.get_user_usage_by_attribution(user_id, field, value, limit=limit + 1)
    truncated = len(rows) > limit
    rows = rows[:limit]
    items = [
        AttributedUsageRecord(
            trace_id=r.trace_id,
            model_tag=r.model_tag,
            provider=r.provider,
            operation=r.operation,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            compute_units=float(r.compute_units),
            latency_ms=r.latency_ms,
            status=r.status,
            created_at=r.created_at,
            attribution=attribution_fields(r.cc_metadata),
            cache_read_input_tokens=int((r.cc_metadata or {}).get("cache_read_input_tokens") or 0),
        )
        for r in rows
    ]
    return AttributedUsageResponse(
        filter={field: value},
        items=items,
        total_requests=len(items),
        total_compute_units=float(sum((r.compute_units for r in rows), start=0)),
        total_input_tokens=sum(r.input_tokens for r in rows),
        total_output_tokens=sum(r.output_tokens for r in rows),
        truncated=truncated,
    )


@router.get(
    "/summary",
    response_model=UsageSummary,
    summary="Get usage summary",
    description="Get usage summary for the current billing period.",
)
async def get_usage_summary(
    user_id: CurrentUserFlexible,
    usage_service: UsageServiceDep,
    start_date: datetime | None = Query(
        default=None,
        description="Start date for the summary period",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="End date for the summary period",
    ),
) -> UsageSummary:
    """
    Get usage summary for a time period.

    Accepts either API key or JWT authentication.
    Defaults to current month if no dates specified.
    """
    # Default to current month
    if start_date is None:
        start_date = datetime.utcnow().replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
    if end_date is None:
        end_date = datetime.utcnow()

    summary = await usage_service.get_usage_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    # The service dict mirrors the schema — construct with ** so additive
    # fields (free-tier progress etc.) can never be silently dropped here.
    return UsageSummary(**summary)


@router.get(
    "/daily",
    response_model=DailyUsageResponse,
    summary="Get daily usage",
    description=(
        "Daily compute-unit + token breakdown for the calling user over the "
        "requested window (default 30 days, max 90). Days with no logged "
        "requests are omitted from the response; zero-fill in the consumer "
        "if a contiguous series is needed."
    ),
)
async def get_daily_usage(
    user_id: CurrentUserFlexible,
    usage_service: UsageServiceDep,
    days: int = Query(default=30, ge=1, le=90, description="Number of days"),
) -> DailyUsageResponse:
    """Daily breakdown by user. JWT or API key auth."""
    result = await usage_service.get_usage_daily(user_id=user_id, days=days)
    return DailyUsageResponse(**result)


@router.get(
    "/traces",
    summary="My request traces (self-scoped)",
    description=(
        "Per-request trace records for the authenticated principal — the "
        "self-service counterpart of /admin/v1/traces. Works with a user JWT "
        "(dashboard) or an API key; every row belongs to the caller. Newest "
        "first. `api_key_id` outside the caller's keys yields an empty page."
    ),
)
async def list_my_traces(
    user_id: CurrentUserFlexible,
    usage_service: UsageServiceDep,
    status: str | None = Query(default=None, description="success | error"),
    model: str | None = Query(default=None, description="Filter by resolved model tag"),
    operation: str | None = Query(default=None, description="chat | embedding | image_generation | tts | transcription"),
    api: str | None = Query(default=None, description="Surface tag: v1_messages | v2_messages"),
    api_key_id: int | None = Query(default=None, description="Filter by one of YOUR key ids"),
    days: int = Query(default=7, ge=1, le=90, description="Look-back window in days"),
    start_date: datetime | None = Query(default=None, description="Overrides `days` when set"),
    end_date: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    since = start_date or window_from_days(days)
    out = await query_traces(
        usage_service.session,
        user_id=user_id,  # the tenancy boundary — never optional on this surface
        status=status,
        model_tag=model,
        operation=operation,
        api=api,
        api_key_id=api_key_id,
        since=since,
        until=end_date,
        limit=limit,
        offset=offset,
    )
    out["window"] = {
        "since": since.isoformat(),
        "until": end_date.isoformat() if end_date else None,
    }
    return out


@router.get(
    "/traces/{trace_id}/payload",
    summary="My captured request/response bodies for one trace (self-scoped)",
    description=(
        "Returns the captured payload for one of the caller's own traces. "
        "404 = no such trace for this user. 200 with `captured: false` = the "
        "trace exists but no body was stored (payload capture is off by "
        "default in managed — metrics always, bodies only when enabled)."
    ),
)
async def get_my_trace_payload(
    trace_id: str,
    user_id: CurrentUserFlexible,
    usage_service: UsageServiceDep,
) -> dict:
    session = usage_service.session
    owned = (
        await session.execute(
            select(UsageLog.trace_id)
            .where(UsageLog.trace_id == trace_id, UsageLog.user_id == user_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if owned is None:
        # Missing and not-yours are indistinguishable — no enumeration oracle.
        raise HTTPException(status_code=404, detail="trace not found")

    payload = await fetch_payload(session, trace_id)
    if payload is None:
        return {"trace_id": trace_id, "captured": False}
    return {"captured": True, **payload}
