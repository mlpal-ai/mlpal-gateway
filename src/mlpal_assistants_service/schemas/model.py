"""Model schemas."""

from pydantic import Field

from mlpal_assistants_service.schemas.common import BaseSchema


class ModelPricingInfo(BaseSchema):
    """Pricing information for a model (in compute units)."""

    input_cu_rate: float = Field(..., description="CU cost per unit (e.g., per 1M tokens)")
    output_cu_rate: float = Field(..., description="CU cost per unit (e.g., per 1M tokens)")
    rate_unit: str = Field(..., description="Unit type (per_1m_tokens, per_image, per_minute)")


class ModelInfo(BaseSchema):
    """Information about an available model."""

    model_tag: str = Field(..., description="Model identifier (e.g., 'gpt-5.2')")
    display_name: str = Field(..., description="Human-readable name")
    provider: str = Field(..., description="Provider (openai, anthropic, google, bedrock)")
    description: str | None = Field(default=None, description="Model description")
    capabilities: list[str] | dict = Field(..., description="Model capabilities")
    context_length: int | None = Field(default=None, description="Max context length")
    max_output_tokens: int | None = Field(default=None, description="Max output tokens")
    pricing_tier: str = Field(..., description="Pricing tier (budget, standard, premium)")
    is_deprecated: bool = Field(default=False, description="Whether model is deprecated")
    deprecation_message: str | None = Field(default=None, description="Deprecation notice")
    owned_by: str | None = Field(
        default=None,
        description="'user' for the caller's own byom models; null for catalog models.",
    )
    serving_backend: str | None = Field(
        default=None,
        description=(
            "Backend that will serve this model (first_party, azure, vertex, "
            "bedrock). null = no configured backend serves it — calls will fail."
        ),
    )


class ModelDetailInfo(ModelInfo):
    """Detailed model information including pricing."""

    pricing: ModelPricingInfo | None = Field(default=None, description="Pricing in compute units")


class ModelListResponse(BaseSchema):
    """Response schema for listing models."""

    models: list[ModelInfo] = Field(..., description="Available models")
    total: int = Field(..., description="Total number of models")
    denied_by_policy: int = Field(
        default=0,
        description="Models hidden from this listing by the caller key's model_policy (0 when unrestricted)",
    )
    model_policy: dict | None = Field(
        default=None,
        description="The caller key's own model_policy (allow/deny globs), null when unrestricted",
    )


class ModelCapabilities(BaseSchema):
    """Detailed model capabilities."""

    chat: bool = Field(default=True, description="Supports chat completions")
    vision: bool = Field(default=False, description="Supports image inputs")
    function_calling: bool = Field(default=False, description="Supports function calling")
    structured_output: bool = Field(default=False, description="Supports structured output")
    streaming: bool = Field(default=True, description="Supports streaming")
    embeddings: bool = Field(default=False, description="Supports embeddings")
    image_generation: bool = Field(default=False, description="Can generate images")
    audio: bool = Field(default=False, description="Supports audio inputs")


class MetaModelRouting(BaseSchema):
    """A single routing rule for a meta model."""

    operation: str = Field(..., description="Operation type (chat, embedding, image_generation, tts, transcription)")
    resolved_model: str = Field(..., description="The actual model this routes to")


class MetaModelInfo(BaseSchema):
    """Information about a meta model alias."""

    meta_model_tag: str = Field(..., description="Meta model identifier (mlpal, mlpal-flash, mlpal-lite)")
    strategy: str = Field(..., description="Routing strategy (quality, speed, cost)")
    description: str = Field(..., description="Description of this meta model tier")
    routings: list[MetaModelRouting] = Field(..., description="Operation-to-model mappings")


class MetaModelListResponse(BaseSchema):
    """Response schema for listing meta model aliases."""

    meta_models: list[MetaModelInfo] = Field(..., description="Available meta model aliases")
    total: int = Field(..., description="Total number of meta models")
