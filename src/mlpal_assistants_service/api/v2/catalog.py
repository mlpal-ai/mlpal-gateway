"""GET /v2/catalog — curated model tiers for agent clients.

Returns:

  - `models`: EVERY served chat model with the attributes a caller routes on —
    cost (absolute CU/1M), context window, caps, observed throughput, strengths.
    This is the source of truth, so onboarding a model makes it selectable with
    no curation edit and callers can express "long context, cheap" instead of
    picking a point on one ordinal scale.
  - `tiers` + `routing_ladder`: the AUTHORITATIVE subtask-routing interface. A
    client maps its complexity estimate onto `routing_ladder` (cheapest -> most
    capable, clamped) and uses that tier's model — no client-side "which model
    is current" logic. Each tier is guaranteed current-generation, cost-ordered,
    provider-scoped (OpenAI/Anthropic/Google), and carries `caps` + real
    `context` so the client can hard-filter a special subtask (vision / long
    context) and fall to an `alternate`. The five guarantees are enforced by
    tests over curated.json (tests/unit/test_catalog.py) so a stale curation
    edit fails CI rather than shipping a superseded ladder.

The ladder depends only on the pricing ledger + model registry (not the
slowly-refreshing latency/feedback stats), so with `ETag` + `Cache-Control:
max-age=3600` the routing decision is deterministic within the cache TTL.
Objective attributes are merged live from the systems that own them; catalog/
curated.json holds judgment only. See services/catalog.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from mlpal_assistants_service.api.deps import (
    CurrentAPIKey,
    ModelRouterDep,
    PricingServiceDep,
    RedisDep,
    SessionDep,
)
from mlpal_assistants_service.api.v2.messages import _require_messages_scope
from mlpal_assistants_service.repositories.feedback_repository import FeedbackRepository
from mlpal_assistants_service.repositories.usage_repository import UsageRepository
from mlpal_assistants_service.services.catalog import available_profiles, build_catalog
from mlpal_assistants_service.services.policy import PolicyService

logger = logging.getLogger(__name__)

router = APIRouter()

# no-cache ≠ don't-cache: clients keep the body but MUST revalidate (cheap 304
# via the ETag). The catalog reflects LIVE availability — a provider key added
# or a model paused must show up on the next fetch, not an hour later (a stale
# max-age misled the console into rendering every tier as dark).
_CACHE_CONTROL = "no-cache"
_LATENCY_CACHE_KEY = "catalog:latency_stats"
_FEEDBACK_CACHE_KEY = "catalog:feedback_quality"
_STATS_CACHE_TTL = 900  # 15 min — these aggregates move slowly; keeps the
                        # percentile/group-by queries off the per-request hot path


async def _cached_stats(redis_client, key: str, compute) -> dict:
    """Redis-cached aggregate, best-effort: on any failure the catalog still
    serves (with that signal absent) rather than 5xx-ing."""
    if redis_client is not None:
        try:
            cached = await redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception:  # noqa: BLE001
            logger.warning(f"catalog: {key} cache read failed", exc_info=True)
    try:
        stats = await compute()
    except Exception:  # noqa: BLE001
        logger.warning(f"catalog: {key} query failed; serving without", exc_info=True)
        return {}
    if redis_client is not None:
        with contextlib.suppress(Exception):
            await redis_client.setex(key, _STATS_CACHE_TTL, json.dumps(stats))
    return stats


@router.get(
    "/catalog",
    summary="Curated model tiers (per profile) with live cost and availability",
)
async def get_catalog(
    request: Request,
    api_key: CurrentAPIKey,
    model_router: ModelRouterDep,
    pricing_service: PricingServiceDep,
    session: SessionDep,
    redis_client: RedisDep,
    profile: str = Query(default="coding", description="Curation profile, e.g. 'coding'"),
) -> Response:
    _require_messages_scope(api_key)

    latency = await _cached_stats(
        redis_client, _LATENCY_CACHE_KEY,
        lambda: UsageRepository(session).get_latency_stats_by_model(),
    )
    feedback = await _cached_stats(
        redis_client, _FEEDBACK_CACHE_KEY,
        lambda: FeedbackRepository(session).get_quality_by_model(),
    )
    # Key-scoped: models outside this key's model_policy are unavailable in the
    # ladder and absent from `models` (same rule as /v1/models).
    policy = getattr(api_key, "model_policy", None)
    allowed = (lambda tag: PolicyService.is_model_allowed(policy, tag)) if policy else None
    catalog = await build_catalog(
        profile, model_router, pricing_service,
        latency_stats=latency, feedback_quality=feedback, allowed=allowed,
    )
    if catalog is None:
        return Response(
            json.dumps({
                "error": f"unknown profile '{profile}'",
                "available_profiles": available_profiles(),
            }),
            status_code=404,
            media_type="application/json",
        )

    body = json.dumps(catalog)
    etag = f'"{hashlib.md5(body.encode()).hexdigest()}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": _CACHE_CONTROL})
    return Response(
        body,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": _CACHE_CONTROL},
    )
