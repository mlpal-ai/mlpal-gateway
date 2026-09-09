"""POST /v2/messages — universal Anthropic-Messages endpoint (v2-A: Anthropic).

Thin API layer: auth + scope, light request validation, then hand to
MessagesV2Core (which owns routing, the provider edge, SSE transport/heartbeat,
telemetry, and the Anthropic error envelope). v1 paths are untouched.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from mlpal_assistants_service.api.deps import (
    CurrentAPIKey,
    ModelRouterDep,
    PricingServiceDep,
    UsageServiceDep,
)
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import ModelNotFoundError
from mlpal_assistants_service.core.security import generate_trace_id
from mlpal_assistants_service.repositories import UsageRepository
from mlpal_assistants_service.seams.billing import build_billing_gate
from mlpal_assistants_service.services.messages_v2.core import (
    ALLOWLIST_WILDCARD,
    SERVED_PROVIDERS,
    MessagesV2Core,
    is_served_chat_model,
)
from mlpal_assistants_service.services.messages_v2.errors import error_body
from mlpal_assistants_service.services.messages_v2.schemas import (
    InvalidMessagesRequest,
    validate,
)
from mlpal_assistants_service.services.policy import PolicyService
from mlpal_assistants_service.services.rate_limiter import RateLimiter
from mlpal_assistants_service.services.reasoning_effort import default_effort, supported_levels

router = APIRouter()


def surface_for_path(path: str) -> str:
    """Which mount served this call: canonical /v1/messages vs the deprecated
    /v2 alias. "v2_messages" matches historical usage_logs rows, so the
    alias-drain query (migration Phase 3) is one WHERE clause. Assumes no
    proxy path-prefix rewriting (prod ingress is host-based; documented in the
    surface tests)."""
    return "v2_messages" if path.startswith("/v2") else "v1_messages"


def _require_messages_scope(api_key) -> None:
    # Same scopes as /v1/messages: route-specific "messages" or grandfathered "chat".
    if not (api_key.has_permission("messages") or api_key.has_permission("chat")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have permission for messages",
        )


@router.post(
    "",
    summary="Universal Anthropic Messages API",
    description=(
        "Anthropic-Messages wire-faithful endpoint with real provider "
        "translation. v2-A serves Anthropic models via the shared core."
    ),
)
async def create_messages_v2(
    request: Request,
    api_key: CurrentAPIKey,
    model_router: ModelRouterDep,
    usage_service: UsageServiceDep,
    pricing_service: PricingServiceDep,
) -> Response:
    _require_messages_scope(api_key)

    raw = await request.body()
    try:
        req = validate(raw)
    except InvalidMessagesRequest as e:
        return Response(error_body(400, str(e)), status_code=400, media_type="application/json")

    # Same admission stack as /v1/chat: billing seam (local = allow-all, no
    # callouts), Redis rate limiter, and the per-key policy engine.
    redis = getattr(usage_service, "redis", None)
    session = model_router.session
    core = MessagesV2Core(
        model_router,
        usage_service,
        pricing_service,
        build_billing_gate(session, redis),
        rate_limiter=RateLimiter(redis) if redis else None,
        policy=PolicyService(redis, UsageRepository(session)),
    )
    surface = surface_for_path(request.url.path)
    return await core.handle(req, api_key, request.headers, generate_trace_id(), surface=surface)


@router.get(
    "/models",
    summary="List models available on /v2/messages with capabilities",
)
async def list_v2_models(api_key: CurrentAPIKey, model_router: ModelRouterDep) -> JSONResponse:
    """Capability advertisement for the models /v2/messages serves: capabilities
    read from the registry, plus reasoning/caching derived per provider. All
    served providers reason (Anthropic extended thinking, OpenAI gpt-5.x,
    Gemini 3 thinking) and surface cache reads in usage, so both are advertised
    for every served provider. Under the wildcard allowlist this lists every
    served chat model; with an explicit allowlist it lists those pinned tags."""
    _require_messages_scope(api_key)
    settings = get_settings()

    if ALLOWLIST_WILDCARD in settings.messages_v2_allowlist:
        candidates = [
            m for m in await model_router.list_models() if is_served_chat_model(m)
        ]
    else:
        candidates = []
        for tag in settings.messages_v2_allowlist:
            try:
                candidates.append(await model_router.get_model(tag))
            except Exception:  # noqa: BLE001 — skip unresolvable tags in the listing
                continue

    # Key-scoped: same rule as /v1/models — a key with a model_policy only
    # sees what it may use.
    policy = getattr(api_key, "model_policy", None)
    if policy:
        candidates = [m for m in candidates if PolicyService.is_model_allowed(policy, m.model_tag)]

    out = []
    for m in candidates:
        caps = m.capabilities if isinstance(m.capabilities, dict) else {}
        levels = list(supported_levels(caps))
        out.append({
            "id": m.model_tag,
            "display_name": m.display_name,
            "provider": m.provider,
            "max_output_tokens": m.max_output_tokens,
            "capabilities": {
                "tools": bool(caps.get("tools", True)),
                "vision": bool(caps.get("vision", True)),
                # A model reasons iff it exposes an effort lever (probe-verified
                # per model in the catalog), not merely because its provider can.
                "reasoning": bool(levels),
                "caching": m.provider in SERVED_PROVIDERS,
            },
            "effort_levels": levels,
            "default_effort": default_effort(caps),
        })
    return JSONResponse({"data": out})


@router.post(
    "/count_tokens",
    summary="Count tokens for a Messages request (Anthropic-wire)",
    description=(
        "Passthrough to the serving backend's count_tokens for Anthropic "
        "models. The Anthropic SDK's client.messages.count_tokens and Claude "
        "Code's context management call this; without it they see a 404. "
        "Not metered (no inference happens)."
    ),
)
async def count_tokens_v2(
    request: Request,
    api_key: CurrentAPIKey,
    model_router: ModelRouterDep,
) -> Response:
    _require_messages_scope(api_key)
    import json as _json

    import httpx

    from mlpal_assistants_service.core.config import get_settings
    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        get_anthropic_backend,
    )

    raw = await request.body()
    try:
        body = _json.loads(raw)
        model_tag = body.get("model") or ""
    except Exception:  # noqa: BLE001 — malformed body is a client error
        return Response(error_body(400, "invalid JSON body"), 400, media_type="application/json")

    try:
        resolved_tag, _ = await model_router.resolve_meta_model(model_tag, "chat")
        model = await model_router.get_model(resolved_tag)
    except ModelNotFoundError:
        return Response(error_body(404, f"model '{model_tag}' not found"), 404, media_type="application/json")

    if model.provider != "anthropic":
        return Response(
            error_body(400, "count_tokens is only available for Anthropic models"),
            400,
            media_type="application/json",
        )
    try:
        backend = get_anthropic_backend(get_settings())
    except ValueError as e:
        return Response(error_body(503, str(e)), 503, media_type="application/json")
    if not backend.url.endswith("/v1/messages") or backend.name == "bedrock":
        # Mantle has no count_tokens surface; adapter-path models likewise.
        return Response(
            error_body(501, f"count_tokens is not supported by the '{backend.name}' backend"),
            501,
            media_type="application/json",
        )

    body["model"] = model.provider_model_id
    content, headers = backend.prepare(_json.dumps(body).encode(), request.headers)
    url = backend.url + "/count_tokens"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=content, headers=headers)
    return Response(
        resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
