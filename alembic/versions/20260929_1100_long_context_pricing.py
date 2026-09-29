"""model_pricing: long-context pricing tier

Providers price requests whose prompt exceeds a threshold at higher rates
(OpenAI GPT-6 / GPT-5.6: > 272K input tokens at 2x input, 1.5x output;
Gemini Pro: > 200K at 2x input, 1.5x output). One flat rate per model
under-billed those requests by up to 2x (09-28: gpt-6-astra cost AWS $365
against $239 metered). NULL threshold = no tier (every existing row keeps
identical billing). Cache reads above the threshold use long_cache_read_rate
when set, else the provider's standard multiple of long_input_rate; cache
writes use the standard write multiplier of long_input_rate.

Revision ID: 20260929_1100
Revises: 20260929_1000
Create Date: 2026-09-29
"""

import sqlalchemy as sa
from alembic import op

revision = "20260929_1100"
down_revision = "20260929_1000"
branch_labels = None
depends_on = None

SCHEMA = "assistants"
COLUMNS = ("long_input_rate", "long_output_rate", "long_cache_read_rate")


def upgrade() -> None:
    op.add_column("model_pricing", sa.Column("long_context_threshold", sa.Integer(), nullable=True), schema=SCHEMA)
    for col in COLUMNS:
        op.add_column("model_pricing", sa.Column(col, sa.Numeric(12, 8), nullable=True), schema=SCHEMA)


def downgrade() -> None:
    for col in ("long_context_threshold", *COLUMNS):
        op.drop_column("model_pricing", col, schema=SCHEMA)
