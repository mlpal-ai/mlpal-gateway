"""API Key model for storing user API keys."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base, TimestampMixin


class APIKey(Base, TimestampMixin):
    """
    API Key model for MLpal Assistants Service.

    Stores hashed API keys with metadata for authentication and rate limiting.
    The actual key is only shown once during creation - we store only the hash.

    Note: user_id references the users table in the configured user schema
    (MLPAL_USER_SCHEMA), not in the assistants schema. The FK is enforced at
    the application level, not database level, for schema flexibility.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("idx_api_keys_hash", "key_hash", unique=True),
        Index("idx_api_keys_user", "user_id"),
        Index("idx_api_keys_user_active", "user_id", postgresql_where="is_active = true"),
        # GIN supports `tags @> '{...}'` containment queries for tag-keyed
        # lookup (e.g., "all keys minted for cde_id=9"). Cheap at our
        # write rate, fast for the dashboard / recovery paths.
        Index("idx_api_keys_tags_gin", "tags", postgresql_using="gin"),
        {"schema": "assistants"},
    )

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Reference to user in the user schema (no FK constraint - cross-schema)
    user_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # Key storage (SHA-256 hash, never plaintext)
    key_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
    )

    # Key prefix for display (e.g., "mlpal_sk_abc12345...")
    key_prefix: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    # User-provided name for the key
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Optional description
    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # Permissions (JSON object of allowed operations)
    # e.g., {"chat": true, "embeddings": true} or {"*": true} for all
    permissions: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False,
    )

    # Rate limit tier
    rate_limit_tier: Mapped[str] = mapped_column(
        String(50),
        default="standard",
        server_default="standard",
        nullable=False,
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
        nullable=False,
    )

    # Revocation timestamp (null if not revoked)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Last used timestamp
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Expiration (optional). NULL means never-expires — used for
    # service-bound keys (e.g., cde_sk_* keys minted by mcp-service for
    # a CDE pod). User-issued keys may also be never-expire by choice.
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Free-form tags for attribution. Used by service callers to mark
    # a key with the entity it belongs to (e.g.,
    # {"mcp_server_id": 18, "cde_id": 9, "source": "mcp_builder_cde"}).
    # Indexed via GIN below for tag-keyed lookup; primary read path is
    # still the FK column on the owning service's table.
    tags: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        server_default="{}",
        nullable=False,
    )

    # Per-key policy engine (see services/policy.py). NULL = unrestricted, so
    # existing keys keep full access with no migration backfill.
    #
    # model_policy: {"allow": ["gpt-5.6-*", "claude-opus-5"], "deny": ["gpt-5.6-sol"]}
    #   glob patterns, "*"/absent allow = all, deny (union) wins. Checked against
    #   the requested tag AND the meta-resolved concrete model.
    model_policy: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    # budgets: list of {"id","unit":"usd"|"cu","amount","window":"daily"|"weekly"
    #   |"monthly"|"lifetime"}. All are enforced (deny if ANY window is exhausted).
    budgets: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    # capture_policy: {"mode": "on"|"off", "models": ["tag", ...]?}. NULL =
    #   inherit the deployment default. "off" is a HARD promise — no default
    #   or operator setting re-enables capture for the key (see
    #   services/capture.py key_allows_capture).
    capture_policy: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"

    @property
    def is_valid(self) -> bool:
        """Check if the key is currently valid (active and not expired)."""
        if not self.is_active or self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at < datetime.now(UTC):
            return False
        return True

    def has_permission(self, operation: str) -> bool:
        """Check if key has permission for an operation."""
        if isinstance(self.permissions, list):
            return "*" in self.permissions or operation in self.permissions
        return self.permissions.get("*", False) or self.permissions.get(operation, False)
