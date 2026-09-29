"""user_suspensions: account-level suspension the gateway enforces itself

An operator (or the backend, via the internal endpoint) suspends an ACCOUNT,
not a key: every API key of the user answers 403 with the reason, and the
management surface refuses the user's session. Lifting a suspension restores
all of the user's keys without re-minting. Gateway-owned state so the check
never depends on another service's schema (OSS boxes have no backend).

Revision ID: 20260929_1000
Revises: 20260917_1200
Create Date: 2026-09-29
"""

import sqlalchemy as sa
from alembic import op

revision = "20260929_1000"
down_revision = "20260917_1200"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.create_table(
        "user_suspensions",
        sa.Column("user_id", sa.Integer(), primary_key=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("incident", sa.String(100), nullable=True),
        sa.Column("suspended_by", sa.String(100), nullable=False),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("lifted_by", sa.String(100), nullable=True),
        sa.Column("lifted_at", sa.DateTime(timezone=True), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("user_suspensions", schema=SCHEMA)
