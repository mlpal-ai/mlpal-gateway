"""model_pricing.cache_read_rate: per-model cache-read list price

Cache-read pricing stops being a single global multiplier: Claude Fable 5.1
prices cache reads at $0.25/MTok (0.025x of its $10 input rate), while the
standard rate is 0.10x. NULL = fall back to the global multiplier — every
existing row keeps identical billing.

Revision ID: 20260901_1600
Revises: 20260901_1000
Create Date: 2026-09-01
"""

import sqlalchemy as sa
from alembic import op

revision = "20260901_1600"
down_revision = "20260901_1000"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.add_column(
        "model_pricing",
        sa.Column("cache_read_rate", sa.Numeric(12, 8), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("model_pricing", "cache_read_rate", schema=SCHEMA)
