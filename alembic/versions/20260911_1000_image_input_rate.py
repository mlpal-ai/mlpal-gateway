"""model_pricing.image_input_rate: token-priced image models

OpenAI's gpt-image family bills per token with THREE rates (text input,
image input, image output); a flat per-image row cannot reproduce list price
once quality levels span an order of magnitude (gpt-image-2.5 low $0.006 vs
high $0.053 vs max). Token-priced image rows use rate_unit=per_1m_tokens with
input_rate = text input, output_rate = image output, and this column for
image input. NULL = bill image-input tokens at input_rate (or the row is
per_image and the column is unused) — every existing row keeps identical
billing.

Revision ID: 20260911_1000
Revises: 20260901_1600
Create Date: 2026-09-11
"""

import sqlalchemy as sa
from alembic import op

revision = "20260911_1000"
down_revision = "20260901_1600"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.add_column(
        "model_pricing",
        sa.Column("image_input_rate", sa.Numeric(12, 8), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("model_pricing", "image_input_rate", schema=SCHEMA)
