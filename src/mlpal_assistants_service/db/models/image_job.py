"""Async image generation jobs (submit -> poll).

A job row is the source of truth for one background image generation:
the request as submitted, the lifecycle timestamps, and — on completion —
either the exact ImageGenerationResponse payload or a structured error.
Rows outlive the render so clients can poll after network blips; billing
happens inside the render (same path as sync), never at submit.
"""

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"

TERMINAL_STATUSES = frozenset({JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED})


class ImageJob(Base):
    """One background image generation job."""

    __tablename__ = "image_jobs"
    __table_args__ = (
        Index("idx_image_jobs_user", "user_id", "created_at"),
        # Idempotent submits: one live job per (user, idempotency_key).
        Index(
            "uq_image_jobs_user_idem",
            "user_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        {"schema": "assistants"},
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    api_key_id: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default=JOB_STATUS_QUEUED
    )

    # The submitted ImageGenerationRequest (minus job-control fields) and a
    # SHA-256 of its canonical form — an idempotency_key replay with a
    # different body must 409, not silently return the old job's image.
    request: Mapped[dict] = mapped_column(JSONB, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Exact ImageGenerationResponse on success; {"code","message"} on failure.
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Touched by the worker's timer task while the render is in flight; a
    # poll that finds `running` with a stale heartbeat declares worker_lost.
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
