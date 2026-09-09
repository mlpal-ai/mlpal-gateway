"""Model catalogue endpoints."""

from fastapi import APIRouter, HTTPException, status

from mlpal_assistants_service.adapters.factory import get_adapter_factory
from mlpal_assistants_service.api.deps import (
    APIKeyModelPolicy,
    CurrentUserFlexible,
    MetaRoutingRepositoryDep,
    ModelRouterDep,
    PricingServiceDep,
)
from mlpal_assistants_service.core.exceptions import ModelNotFoundError
from mlpal_assistants_service.schemas.model import (
    MetaModelInfo,
    MetaModelListResponse,
    MetaModelRouting,
    ModelDetailInfo,
    ModelInfo,
    ModelListResponse,
    ModelPricingInfo,
)
from mlpal_assistants_service.services.policy import PolicyService

router = APIRouter()


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List available models",
    description="Get a list of all available AI models.",
)
async def list_models(
    _user_id: CurrentUserFlexible,
    model_policy: APIKeyModelPolicy,
    model_router: ModelRouterDep,
    provider: str | None = None,
    capability: str | None = None,
    include_deprecated: bool = False,
) -> ModelListResponse:
    """
    List all available models.

    Optionally filter by provider or capability. When the caller authenticates
    with an API key that carries a model_policy, the list is filtered to the
    models that key is permitted to use (JWT/dashboard callers see everything).
    """
    models = await model_router.list_models(
        provider=provider,
        operation=capability,
        include_deprecated=include_deprecated,
    )

    if model_policy:
        models = [m for m in models if PolicyService.is_model_allowed(model_policy, m.model_tag)]

    # Serving truth per model: which configured backend (if any) would take
    # the call. Cached in the factory — no I/O here. null lets the console
    # render unserved models visibly distinct instead of silently listing them.
    factory = get_adapter_factory()
    out = [
        ModelInfo(
            model_tag=m.model_tag,
            display_name=m.display_name,
            provider=m.provider,
            description=m.description,
            capabilities=m.capabilities,
            context_length=m.context_length,
            max_output_tokens=m.max_output_tokens,
            pricing_tier=m.pricing_tier,
            is_deprecated=m.is_deprecated,
            deprecation_message=m.deprecation_message,
            serving_backend=factory.serving_backend_for(m.provider, m.provider_model_id),
        )
        for m in models
    ]

    # The caller's own byom models (`user/…` tags), visible only to them.
    # provider="user" groups them distinctly in the console.
    from mlpal_assistants_service.core.config import get_settings

    if get_settings().connections_enabled and provider in (None, "user"):
        from mlpal_assistants_service.services import connections as conn_svc

        tenant = await conn_svc.get_model_overlay(_user_id, model_router.session)
        for ref in sorted(tenant.values(), key=lambda r: r.model_tag):
            if capability and ref.operation != capability:
                continue
            if model_policy and not PolicyService.is_model_allowed(
                model_policy, ref.model_tag
            ):
                continue
            out.append(
                ModelInfo(
                    model_tag=ref.model_tag,
                    display_name=ref.model_tag.removeprefix("user/"),
                    provider="user",
                    description=None,
                    capabilities=ref.capabilities or {"operation": ref.operation},
                    context_length=ref.context_length,
                    max_output_tokens=ref.max_output_tokens,
                    pricing_tier="user",
                    owned_by="user",
                    serving_backend="byom:custom",
                )
            )

    return ModelListResponse(models=out, total=len(out))


# Meta model strategy descriptions
META_MODEL_DESCRIPTIONS = {
    "mlpal": "Premium tier - routes to the best models for each operation (quality focus)",
    "mlpal-flash": "Fast tier - routes to optimized models for speed",
    "mlpal-lite": "Budget tier - routes to cost-effective models",
}

META_MODEL_STRATEGIES = {
    "mlpal": "quality",
    "mlpal-flash": "speed",
    "mlpal-lite": "cost",
}


@router.get(
    "/aliases",
    response_model=MetaModelListResponse,
    summary="List meta model aliases",
    description="Get all meta model aliases (mlpal, mlpal-flash, mlpal-lite) and their routing mappings.",
)
async def list_meta_models(
    _user_id: CurrentUserFlexible,
    meta_routing_repo: MetaRoutingRepositoryDep,
) -> MetaModelListResponse:
    """
    List all meta model aliases and their operation-to-model mappings.

    Meta models are aliases that automatically route to optimal provider models
    based on the operation type:
    - mlpal: Premium tier, routes to best quality models
    - mlpal-flash: Fast tier, routes to speed-optimized models
    - mlpal-lite: Budget tier, routes to cost-effective models
    """
    # Get all unique meta model tags
    meta_tags = await meta_routing_repo.get_all_meta_models()

    meta_models = []
    for tag in sorted(meta_tags):
        # Get all routings for this meta model
        routings = await meta_routing_repo.get_all_routings_for_model(tag)

        meta_models.append(
            MetaModelInfo(
                meta_model_tag=tag,
                strategy=META_MODEL_STRATEGIES.get(tag, "custom"),
                description=META_MODEL_DESCRIPTIONS.get(tag, f"Meta model alias: {tag}"),
                routings=[
                    MetaModelRouting(
                        operation=r.operation,
                        resolved_model=r.resolved_model_tag,
                    )
                    for r in routings
                ],
            )
        )

    return MetaModelListResponse(
        meta_models=meta_models,
        total=len(meta_models),
    )


@router.get(
    "/{model_tag}",
    response_model=ModelDetailInfo,
    summary="Get model details",
    description="Get detailed information about a specific model including pricing.",
)
async def get_model(
    model_tag: str,
    _user_id: CurrentUserFlexible,
    model_policy: APIKeyModelPolicy,
    model_router: ModelRouterDep,
    pricing_service: PricingServiceDep,
) -> ModelDetailInfo:
    """Get details of a specific model including pricing in compute units."""
    try:
        model = await model_router.get_model(model_tag)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_tag}' not found",
        )
    # Key-scoped, consistent with the listing and with inference (403, not a
    # fake 404: the model exists, this key may not use it).
    if model_policy and not PolicyService.is_model_allowed(model_policy, model.model_tag):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model '{model_tag}' is not permitted by this key's model_policy",
        )

    # Get pricing info - determine operation from capabilities
    operation = "chat"
    caps = model.capabilities
    if isinstance(caps, dict):
        op = caps.get("operation")
        if op:
            operation = op
    elif isinstance(caps, list):
        if "embedding" in caps:
            operation = "embedding"
        elif "image_generation" in caps:
            operation = "image_generation"
        elif "tts" in caps:
            operation = "tts"
        elif "transcription" in caps:
            operation = "transcription"

    pricing_info = None
    pricing = await pricing_service.get_pricing(model_tag, operation)
    if pricing:
        # Use pre-computed CU rates from generated columns
        pricing_info = ModelPricingInfo(
            input_cu_rate=float(pricing.input_cu_rate),
            output_cu_rate=float(pricing.output_cu_rate),
            rate_unit=pricing.rate_unit,
        )

    return ModelDetailInfo(
        model_tag=model.model_tag,
        display_name=model.display_name,
        provider=model.provider,
        description=model.description,
        capabilities=model.capabilities,
        context_length=model.context_length,
        max_output_tokens=model.max_output_tokens,
        pricing_tier=model.pricing_tier,
        is_deprecated=model.is_deprecated,
        deprecation_message=model.deprecation_message,
        pricing=pricing_info,
    )
