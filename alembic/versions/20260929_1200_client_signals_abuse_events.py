"""client signals on usage_logs / api_keys + abuse_events

The 2026-09-28 signup farm was attributed by hand-joining CloudFront logs
against usage timestamps because nothing in the gateway persisted who was
calling. Now every usage row and every key carries the viewer IP and a
user-agent hash, and the abuse detector records its findings.

client_ip is a signal with a 14-day retention (scrubbed by the detector's
daily pass), never a security boundary.

Revision ID: 20260929_1200
Revises: 20260929_1100
Create Date: 2026-09-29
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260929_1200"
down_revision = "20260929_1100"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.add_column("usage_logs", sa.Column("client_ip", sa.String(45), nullable=True), schema=SCHEMA)
    op.add_column("usage_logs", sa.Column("client_ua_hash", sa.String(16), nullable=True), schema=SCHEMA)
    op.create_index("idx_usage_client_ip", "usage_logs", ["client_ip", "created_at"], schema=SCHEMA)
    op.add_column("api_keys", sa.Column("created_ip", sa.String(45), nullable=True), schema=SCHEMA)
    op.add_column("api_keys", sa.Column("created_ua_hash", sa.String(16), nullable=True), schema=SCHEMA)
    op.create_table(
        "abuse_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("rule", sa.String(40), nullable=False),
        sa.Column("cluster_key", sa.String(80), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_abuse_events_cluster", "abuse_events", ["rule", "cluster_key", "created_at"], schema=SCHEMA
    )


def downgrade() -> None:
    op.drop_table("abuse_events", schema=SCHEMA)
    op.drop_column("api_keys", "created_ua_hash", schema=SCHEMA)
    op.drop_column("api_keys", "created_ip", schema=SCHEMA)
    op.drop_index("idx_usage_client_ip", table_name="usage_logs", schema=SCHEMA)
    op.drop_column("usage_logs", "client_ua_hash", schema=SCHEMA)
    op.drop_column("usage_logs", "client_ip", schema=SCHEMA)
