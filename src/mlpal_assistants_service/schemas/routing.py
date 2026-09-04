"""Routing metadata schemas for MLPal meta-models."""

from pydantic import Field

from mlpal_assistants_service.schemas.common import BaseSchema


class RoutingMetadata(BaseSchema):
    """
    Metadata about model routing when using MLPal meta-models.

    Included in responses when the request used a meta-model
    (mlpal, mlpal-flash, mlpal-lite) to show which actual model
    was used underneath.
    """

    requested_model: str = Field(
        ...,
        description="The model tag that was requested (e.g., 'mlpal')",
    )
    resolved_model: str = Field(
        ...,
        description="The actual model used (e.g., 'gemini-3-pro-image')",
    )
    resolved_provider: str = Field(
        ...,
        description="The provider of the resolved model (e.g., 'google')",
    )
    operation: str = Field(
        ...,
        description="The operation type (e.g., 'image_generation', 'chat')",
    )
    strategy: str = Field(
        ...,
        description="The routing strategy: 'quality' (mlpal), 'speed' (mlpal-flash), 'cost' (mlpal-lite)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "requested_model": "mlpal",
                "resolved_model": "gemini-3-pro-image",
                "resolved_provider": "google",
                "operation": "image_generation",
                "strategy": "quality",
            }
        }
    }


# Strategy mapping for meta-models
META_MODEL_STRATEGIES = {
    "mlpal": "quality",
    "mlpal-flash": "speed",
    "mlpal-lite": "cost",
}


def get_strategy(meta_model_tag: str) -> str:
    """Get the strategy for a meta-model tag."""
    return META_MODEL_STRATEGIES.get(meta_model_tag, "unknown")


def is_meta_model(model_tag: str) -> bool:
    """Check if a model tag is a meta-model."""
    return model_tag in META_MODEL_STRATEGIES
