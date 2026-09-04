"""Image generation API endpoints (sync + async jobs)."""

from fastapi import APIRouter, HTTPException, Response, status

from mlpal_assistants_service.api.deps import (
    CurrentAPIKey,
    ImageServiceDep,
    RateLimitCheck,
    SessionDep,
)
from mlpal_assistants_service.schemas.images import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageJobError,
    ImageJobStatusResponse,
    ImageJobSubmitted,
)
from mlpal_assistants_service.services import image_jobs
from mlpal_assistants_service.services.image_jobs import IdempotencyConflictError

router = APIRouter()

# Poll cadence hint for queued/running jobs.
_RETRY_AFTER_SECONDS = "5"


def _require_images_permission(api_key) -> None:
    # Permission gate — same contract as /v1/chat: a key scoped away from this
    # surface must not be able to spend on it.
    # "image" is a legacy singular some existing keys carry — accept both.
    if not (api_key.has_permission("images") or api_key.has_permission("image")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have permission for image generation",
        )


@router.post(
    "/generations",
    response_model=None,
    status_code=status.HTTP_200_OK,
    summary="Generate images",
    description="Generate images from a text prompt (sync, or async with wait:false).",
)
async def generate_images(
    body: ImageGenerationRequest,
    api_key: CurrentAPIKey,
    _rate_limit: RateLimitCheck,
    image_service: ImageServiceDep,
    session: SessionDep,
    response: Response,
) -> ImageGenerationResponse | ImageJobSubmitted:
    """
    Generate images from a text description.

    Supports:
    - Multiple images (1-4; OpenAI text-to-image only — Gemini returns one)
    - Sizes as presets, aspect ratios, or pixels (gpt-image-2: any WxH up to 3840px;
      Gemini: pixels select the 1K/2K/4K tier)
    - Quality levels (standard, hd)
    - Reference images for editing (up to 14)
    - Async jobs: `wait: false` returns 202 with a job id immediately —
      required for renders that may exceed the ~120s edge timeout
      (hd/4K edits). Poll GET /v1/images/jobs/{id}.

    Models: gpt-image-2, gpt-image-1.5, gpt-image-1-mini, gemini-3-pro-image,
    gemini-3.1-flash-image, gemini-3.1-flash-lite-image, or the router tags
    mlpal / mlpal-flash / mlpal-lite.
    """
    _require_images_permission(api_key)

    if body.wait:
        return await image_service.generate(
            user_id=api_key.user_id,
            api_key_id=api_key.id,
            request=body,
            model_policy=api_key.model_policy,
            budgets=api_key.budgets,
        )

    try:
        job, created = await image_jobs.submit_job(
            session,
            api_key=api_key,
            request=body,
            image_service=image_service,
        )
    except IdempotencyConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    # Fresh submit -> 202; idempotent replay -> 200 with the existing job.
    response.status_code = (
        status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    )
    return ImageJobSubmitted(
        id=job.id, status=job.status, created_at=job.created_at
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ImageJobStatusResponse,
    summary="Poll an image job",
    description=(
        "Status of a background image job. Returns the exact sync response "
        "payload in `result` on success. 404 if the job isn't owned by the "
        "caller — no existence leak across users."
    ),
)
async def get_image_job(
    job_id: str,
    api_key: CurrentAPIKey,
    session: SessionDep,
    response: Response,
) -> ImageJobStatusResponse:
    """Poll a background image job (queued|running|succeeded|failed)."""
    _require_images_permission(api_key)

    job = await image_jobs.get_job(session, user_id=api_key.user_id, job_id=job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image job not found",
        )

    if job.status in ("queued", "running"):
        response.headers["Retry-After"] = _RETRY_AFTER_SECONDS

    return ImageJobStatusResponse(
        id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=ImageJobError(**job.error) if job.error else None,
    )
