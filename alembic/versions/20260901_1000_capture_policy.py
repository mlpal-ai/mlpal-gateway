"""api_keys.capture_policy: per-key payload-capture control

Tri-state per key: NULL = inherit the deployment default, {"mode": "on"} /
{"mode": "off"} explicit, optional {"models": [...]} filter narrowing capture
to specific tags. "off" is a hard promise — nothing overrides it.

Revision ID: 20260901_1000
Revises: 20260829_1400
Create Date: 2026-09-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260901_1000"
down_revision = "20260829_1400"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("capture_policy", JSONB, nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("api_keys", "capture_policy", schema=SCHEMA)
