"""image_jobs: async submit -> poll image generation

Job rows for background image renders (contract agreed with the
mlpal-image-gen-mcp session, 2026-08-29). Billing happens inside the
render, never at submit; a partial unique index enforces idempotency.

Revision ID: 20260829_1400
Revises: 20260819_1000
Create Date: 2026-08-29
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260829_1400"
down_revision = "20260819_1000"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.create_table(
        "image_jobs",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("api_key_id", sa.Integer, nullable=False),
        sa.Column("status", sa.String(10), nullable=False, server_default="queued"),
        sa.Column("request", JSONB, nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("error", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_image_jobs_user",
        "image_jobs",
        ["user_id", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "uq_image_jobs_user_idem",
        "image_jobs",
        ["user_id", "idempotency_key"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("image_jobs", schema=SCHEMA)
