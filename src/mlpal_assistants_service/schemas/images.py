"""Image generation schemas."""

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from mlpal_assistants_service.schemas.common import BaseSchema
from mlpal_assistants_service.schemas.routing import RoutingMetadata


class ReferenceImage(BaseSchema):
    """
    Reference image for image-to-image generation.

    Supports multiple input methods:
    - url: Public URL or presigned S3 URL
    - base64: Base64-encoded image data

    Providers like Google Gemini support up to 14 reference images
    for style transfer, image editing, and guided generation.
    """

    url: str | None = Field(
        default=None,
        description="URL to the reference image (public URL or presigned S3 URL)",
    )
    base64: str | None = Field(
        default=None,
        description="Base64-encoded image data",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type (auto-detected if not provided). E.g., 'image/png', 'image/jpeg'",
    )

    @field_validator("base64", "url", mode="before")
    @classmethod
    def check_at_least_one(cls, v: str | None, info) -> str | None:
        return v

    def model_post_init(self, __context) -> None:
        """Validate that at least one of url or base64 is provided."""
        if self.url is None and self.base64 is None:
            raise ValueError("Either 'url' or 'base64' must be provided for reference image")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/style-image.png",
                    "mime_type": "image/png",
                },
                {
                    "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ...",
                    "mime_type": "image/png",
                },
            ]
        }
    }


class ImageGenerationRequest(BaseSchema):
    """
    Request schema for image generation.

    Supports:
    - Text-to-image: Just provide a prompt
    - Image-to-image: Provide prompt + reference_images for style transfer or guided generation

    Size parameter accepts multiple formats for a unified cross-provider experience:
    - Semantic presets: "square", "landscape", "portrait", "wide", "tall"
    - Aspect ratios: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
    - Pixel dimensions: "1024x1024", "1792x1024", "1536x1024", etc.

    The system automatically converts to the appropriate format for each provider.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text description of the image to generate",
    )
    model: str = Field(
        default="mlpal",
        description=(
            "Image model tag, or a router tag ('mlpal', 'mlpal-flash', 'mlpal-lite') "
            "that resolves to the best served model"
        ),
    )
    n: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate (1-4)",
    )
    size: str = Field(
        default="1024x1024",
        description=(
            "Image size/aspect ratio. Accepts multiple formats: "
            "semantic presets ('square', 'landscape', 'portrait', 'wide', 'tall'), "
            "aspect ratios ('1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3'), "
            "or pixel dimensions ('1024x1024', '1792x1024', '1536x1024'). "
            "Automatically converted to provider-specific format."
        ),
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description=(
            "Image quality level. OpenAI gpt-image: standard=medium, hd=high. "
            "Google Gemini: hd renders at 2K instead of 1K (explicit pixel sizes "
            "select 1K/2K/4K directly)."
        ),
    )
    reference_images: list[ReferenceImage] | None = Field(
        default=None,
        max_length=14,
        description=(
            "Reference images for image-to-image generation. "
            "Supported by Google Gemini image models (up to 14 images) and the "
            "OpenAI gpt-image family. Used for style transfer, image editing, "
            "and guided generation."
        ),
    )
    # Job control. Named `wait`, NOT `background` — OpenAI's Images API already
    # uses `background` for PNG transparency, and this surface may pass that
    # through someday.
    wait: bool = Field(
        default=True,
        description=(
            "When false, the request returns 202 immediately with a job id; "
            "poll GET /v1/images/jobs/{id} for the result. Long renders "
            "(hd/4K edits) should use this — a single HTTP connection is "
            "capped at ~120s by the edge."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional client token for background submits. Retrying a submit "
            "with the same key returns the existing job instead of creating "
            "(and billing) a second one. Reusing a key with a different "
            "request body is rejected with 409."
        ),
    )


class ImageJobSubmitted(BaseSchema):
    """Immediate response to a background submit (`wait: false`)."""

    id: str = Field(..., description="Job id — poll GET /v1/images/jobs/{id}")
    status: Literal["queued", "running", "succeeded", "failed"] = Field(...)
    created_at: datetime = Field(...)


class ImageJobError(BaseSchema):
    """Structured failure for a background job."""

    code: Literal["provider_error", "worker_lost", "invalid_request"] = Field(
        ...,
        description=(
            "provider_error: upstream failed; worker_lost: the serving pod "
            "died mid-render (nothing was billed — safe to resubmit); "
            "invalid_request: validation that only surfaces at execution."
        ),
    )
    message: str = Field(...)


class ImageJobStatusResponse(BaseSchema):
    """Poll response for a background image job."""

    id: str = Field(...)
    status: Literal["queued", "running", "succeeded", "failed"] = Field(...)
    created_at: datetime = Field(...)
    started_at: datetime | None = Field(default=None)
    finished_at: datetime | None = Field(default=None)
    result: "ImageGenerationResponse | None" = Field(
        default=None,
        description="On success: the exact same payload a sync request returns.",
    )
    error: ImageJobError | None = Field(default=None)


class GeneratedImage(BaseSchema):
    """
    Single generated image with temporary URL.

    Like OpenAI, generated images are stored temporarily and made
    available via presigned URLs that expire after 1 hour.

    Users must download images before the URL expires.
    """

    url: str = Field(
        ...,
        description="Presigned URL to download the image. Valid for 1 hour.",
    )
    expires_at: datetime = Field(
        ...,
        description="UTC timestamp when the URL expires. Download before this time.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description="Revised prompt (if model modified it, e.g., DALL-E 3)",
    )
    content_type: str = Field(
        default="image/png",
        description="MIME type of the image",
    )
    size_bytes: int = Field(
        default=0,
        ge=0,
        description="Image file size in bytes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://mlpal-assistants-assets.s3.us-east-2.amazonaws.com/generated/123/abc/uuid_image.png?X-Amz-Signature=...",
                "expires_at": "2026-01-27T11:30:00Z",
                "revised_prompt": "A beautiful sunset over mountains with vibrant orange and purple colors",
                "content_type": "image/png",
                "size_bytes": 524288,
            }
        }
    }


class ImageGenerationCost(BaseSchema):
    """Cost information for image generation."""

    model_name: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    images_generated: int = Field(..., description="Number of images generated")
    latency_ms: int = Field(..., description="Request latency in milliseconds")
    compute_units: float = Field(..., description="Compute units billed (provider pass-through, no markup)")


class ImageGenerationResponse(BaseSchema):
    """
    Response schema for image generation.

    Images are returned as presigned URLs that expire after 1 hour.
    Users should download images immediately after generation.

    When using MLPal meta-models (mlpal, mlpal-flash, mlpal-lite),
    the `routing` field contains information about which actual model
    was used for the request.
    """

    data: list[GeneratedImage] = Field(
        ...,
        description="List of generated images with download URLs",
    )
    model: str = Field(..., description="Model used for generation (requested model tag)")
    cost: ImageGenerationCost = Field(..., description="Cost information")
    routing: RoutingMetadata | None = Field(
        default=None,
        description=(
            "Routing metadata when using MLPal meta-models. "
            "Shows which actual model was used (e.g., mlpal -> gpt-image-2)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {
                        "url": "https://mlpal-assistants-assets.s3.us-east-2.amazonaws.com/...",
                        "expires_at": "2026-01-27T11:30:00Z",
                        "revised_prompt": "A sunset over mountains",
                        "content_type": "image/png",
                        "size_bytes": 524288,
                    }
                ],
                "model": "mlpal",
                "cost": {
                    "model_name": "gpt-image-2",
                    "provider": "openai",
                    "images_generated": 1,
                    "latency_ms": 5200,
                    "compute_units": 0.0159,
                },
                "routing": {
                    "requested_model": "mlpal",
                    "resolved_model": "gpt-image-2",
                    "resolved_provider": "openai",
                    "operation": "image_generation",
                    "strategy": "quality",
                },
            }
        }
    }


# ImageJobStatusResponse.result forward-references ImageGenerationResponse
# (defined above it so the request/job schemas read top-down); resolve it now.
ImageJobStatusResponse.model_rebuild()
