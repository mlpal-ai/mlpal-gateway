"""api_keys: drop the orphaned owner-context columns and trigger

Production carried four columns on assistants.api_keys (owner_type, owner_id,
member_user_id, owner_context_version) plus BEFORE INSERT trigger
trg_api_keys_set_owner_id / function set_api_key_owner_id() that no
migration in this repo created and no code in any MLPal repo reads or
writes (the same shape exists on mlpal-backend's own agent_api_keys table;
here it was applied by hand and never landed in code). owner_id was NOT
NULL, defaulted by the trigger to user_id — so every row's owner_id equals
its user_id and nothing is lost by dropping it. Migrations are the schema's
source of truth; this makes the live table match the model again.

Idempotent: every statement is IF EXISTS, so environments that never had
the drift (rig, OSS installs) apply cleanly.

Revision ID: 20260917_1200
Revises: 20260911_1000
Create Date: 2026-09-17
"""

from alembic import op

revision = "20260917_1200"
down_revision = "20260911_1000"
branch_labels = None
depends_on = None

SCHEMA = "assistants"


def upgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS trg_api_keys_set_owner_id ON {SCHEMA}.api_keys")
    op.execute(f"DROP FUNCTION IF EXISTS {SCHEMA}.set_api_key_owner_id()")
    for column in ("owner_type", "owner_id", "member_user_id", "owner_context_version"):
        op.execute(f"ALTER TABLE {SCHEMA}.api_keys DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    # Restores the shape (not the trigger — it only mirrored user_id).
    op.execute(
        f"ALTER TABLE {SCHEMA}.api_keys "
        "ADD COLUMN IF NOT EXISTS owner_type VARCHAR(16) NOT NULL DEFAULT 'user', "
        "ADD COLUMN IF NOT EXISTS owner_id INTEGER, "
        "ADD COLUMN IF NOT EXISTS member_user_id INTEGER, "
        "ADD COLUMN IF NOT EXISTS owner_context_version INTEGER NOT NULL DEFAULT 1"
    )
    op.execute(f"UPDATE {SCHEMA}.api_keys SET owner_id = user_id WHERE owner_id IS NULL")
    op.execute(f"ALTER TABLE {SCHEMA}.api_keys ALTER COLUMN owner_id SET NOT NULL")
