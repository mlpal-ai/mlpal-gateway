"""Async image generation jobs: submit -> background render -> poll.

Contract (agreed with the image-gen MCP session, 2026-08-29):
  * POST /v1/images/generations with `wait: false` -> 202 {id, status, created_at}
  * GET  /v1/images/jobs/{id} -> status + (on success) the exact sync payload
  * Billing happens inside the render, never at submit. Wallet/quota problems
    surface AT SUBMIT (checked before queueing), so a poll never turns into a
    billing error.
  * Idempotency: same (user, idempotency_key) returns the existing job; the
    same key with a DIFFERENT body is a 409 — never silently the wrong image.
  * Liveness: the worker heartbeats the row while the provider call is in
    flight; a poll that finds `running` with a stale heartbeat marks the job
    failed(worker_lost). No zombie jobs, no background sweeper.

The runner is an in-process asyncio task. That is a deliberate choice at
current volume (single-digit images/day): a queue would add infra for no
reliability gain — worker_lost + client resubmit covers the pod-restart case,
and nothing is billed for a lost job.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.exceptions import (
    ModelNotFoundError,
    QuotaExceededError,
    UnsupportedCapabilityError,
    ValidationError,
    WalletEmptyError,
)
from mlpal_assistants_service.db.models.image_job import (
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    TERMINAL_STATUSES,
    ImageJob,
)
from mlpal_assistants_service.schemas.images import ImageGenerationRequest

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 20
# A poll that finds `running` with a heartbeat older than this declares the
# worker dead. Must be comfortably > HEARTBEAT_INTERVAL_S.
STALE_AFTER = timedelta(seconds=90)

# Strong refs: the event loop keeps only weak references to tasks, so an
# unreferenced fire-and-forget job could be GC'd mid-render.
_JOB_TASKS: set[asyncio.Task] = set()

# Request fields that control the job itself and are not part of the render.
_JOB_CONTROL_FIELDS = ("wait", "idempotency_key")


class IdempotencyConflictError(Exception):
    """Same idempotency_key, different request body."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _render_request_dict(request: ImageGenerationRequest) -> dict[str, Any]:
    """The request as the renderer sees it — job-control fields stripped."""
    body = request.model_dump(mode="json")
    for f in _JOB_CONTROL_FIELDS:
        body.pop(f, None)
    return body


def _request_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def submit_job(
    session: AsyncSession,
    *,
    api_key: Any,
    request: ImageGenerationRequest,
    image_service: Any,
) -> tuple[ImageJob, bool]:
    """Create (or idempotently return) a job and start the render.

    Returns (job, created). Raises IdempotencyConflictError when the key is
    reused with a different body, and the sync path's billing exceptions
    (WalletEmptyError / QuotaExceededError) when the account can't spend.
    """
    body = _render_request_dict(request)
    req_hash = _request_hash(body)

    if request.idempotency_key:
        existing = (
            await session.execute(
                select(ImageJob).where(
                    ImageJob.user_id == api_key.user_id,
                    ImageJob.idempotency_key == request.idempotency_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.request_hash != req_hash:
                raise IdempotencyConflictError(
                    "idempotency_key was already used with a different request body"
                )
            return existing, False

    # Wallet/quota gate at submit, same semantics as the sync path — a job
    # must never be queued for an account that can't pay for it.
    can_request, block_reason, _ = await image_service._billing.can_make_request_cached(
        api_key.user_id
    )
    if not can_request:
        from mlpal_assistants_service.repositories.billing_repository import (
            WALLET_EMPTY_MESSAGE,
        )

        if block_reason == WALLET_EMPTY_MESSAGE:
            raise WalletEmptyError(block_reason)
        raise QuotaExceededError(
            message=block_reason or "API access blocked", limit=0.0, current_usage=0.0
        )

    job = ImageJob(
        id=f"imgjob_{uuid.uuid4().hex}",
        user_id=api_key.user_id,
        api_key_id=api_key.id,
        status=JOB_STATUS_QUEUED,
        request=body,
        request_hash=req_hash,
        idempotency_key=request.idempotency_key,
    )
    session.add(job)
    await session.commit()

    _spawn_runner(
        job_id=job.id,
        user_id=api_key.user_id,
        api_key_id=api_key.id,
        model_policy=api_key.model_policy,
        budgets=api_key.budgets,
        redis_client=image_service.redis,
        asset_storage=image_service._asset_storage,
        sqs_client=image_service._sqs_client,
    )
    return job, True


def _spawn_runner(**kwargs: Any) -> None:
    task = asyncio.create_task(_run_job(**kwargs))
    _JOB_TASKS.add(task)
    task.add_done_callback(_JOB_TASKS.discard)


async def _heartbeat_loop(job_id: str) -> None:
    """Independent timer: touch the row every HEARTBEAT_INTERVAL_S while the
    provider call is in flight. Its own short-lived sessions — the render
    session is busy awaiting the provider."""
    from mlpal_assistants_service.db.session import async_session_factory

    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        try:
            async with async_session_factory() as hb_session:
                await hb_session.execute(
                    update(ImageJob)
                    .where(ImageJob.id == job_id)
                    .values(heartbeat_at=_utcnow())
                )
                await hb_session.commit()
        except Exception as e:  # noqa: BLE001 — heartbeat must never kill the render
            logger.warning(f"image job {job_id}: heartbeat write failed: {e}")


def _classify_failure(exc: Exception) -> str:
    if isinstance(
        exc,
        (ValidationError, ModelNotFoundError, UnsupportedCapabilityError),
    ):
        return "invalid_request"
    return "provider_error"


async def _run_job(
    *,
    job_id: str,
    user_id: int,
    api_key_id: int,
    model_policy: dict | None,
    budgets: list | None,
    redis_client: Any,
    asset_storage: Any,
    sqs_client: Any,
) -> None:
    from mlpal_assistants_service.db.session import async_session_factory
    from mlpal_assistants_service.services.image import ImageService

    hb_task = asyncio.create_task(_heartbeat_loop(job_id))
    try:
        async with async_session_factory() as session:
            job = (
                await session.execute(select(ImageJob).where(ImageJob.id == job_id))
            ).scalar_one()
            job.status = JOB_STATUS_RUNNING
            job.started_at = _utcnow()
            job.heartbeat_at = _utcnow()
            await session.commit()

            request = ImageGenerationRequest.model_validate(job.request)
            service = ImageService(session, redis_client, asset_storage, sqs_client)
            try:
                response = await service.generate(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    request=request,
                    model_policy=model_policy,
                    budgets=budgets,
                )
            except Exception as e:  # noqa: BLE001 — every failure becomes a job error
                job.status = JOB_STATUS_FAILED
                job.error = {"code": _classify_failure(e), "message": str(e)}
                job.finished_at = _utcnow()
                await session.commit()
                logger.warning(f"image job {job_id} failed: {e}")
                return

            job.status = JOB_STATUS_SUCCEEDED
            job.result = response.model_dump(mode="json")
            job.finished_at = _utcnow()
            await session.commit()
            logger.info(f"image job {job_id} succeeded")
    except Exception as e:  # noqa: BLE001 — runner infrastructure failure
        logger.error(f"image job {job_id}: runner crashed: {e}", exc_info=True)
        try:
            async with async_session_factory() as session:
                await session.execute(
                    update(ImageJob)
                    .where(
                        ImageJob.id == job_id,
                        ImageJob.status.in_([JOB_STATUS_QUEUED, JOB_STATUS_RUNNING]),
                    )
                    .values(
                        status=JOB_STATUS_FAILED,
                        error={"code": "provider_error", "message": str(e)},
                        finished_at=_utcnow(),
                    )
                )
                await session.commit()
        except Exception:  # noqa: BLE001
            pass  # the stale-heartbeat path will mark it worker_lost
    finally:
        hb_task.cancel()


async def get_job(
    session: AsyncSession, *, user_id: int, job_id: str
) -> ImageJob | None:
    """Fetch a job scoped to its owner, lazily failing stale `running` rows.

    A pod restart mid-render leaves status=running with a dead worker; the
    first poll after STALE_AFTER converts it to failed(worker_lost) — nothing
    was billed for it, so the client can safely resubmit.
    """
    job = (
        await session.execute(
            select(ImageJob).where(ImageJob.id == job_id, ImageJob.user_id == user_id)
        )
    ).scalar_one_or_none()
    if job is None:
        return None

    if job.status not in TERMINAL_STATUSES:
        # queued jobs are covered too: a runner that never started (crash
        # between commit and spawn) must not leave the job queued forever.
        last_sign_of_life = job.heartbeat_at or job.started_at or job.created_at
        if last_sign_of_life.tzinfo is None:  # sqlite in tests returns naive
            last_sign_of_life = last_sign_of_life.replace(tzinfo=UTC)
        if last_sign_of_life < _utcnow() - STALE_AFTER:
            job.status = JOB_STATUS_FAILED
            job.error = {
                "code": "worker_lost",
                "message": (
                    "The worker serving this render stopped (likely a deploy or "
                    "pod restart). Nothing was billed — resubmit the request."
                ),
            }
            job.finished_at = _utcnow()
            await session.commit()
    return job
