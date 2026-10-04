"""Usage Log model for tracking API usage (metadata only, no message content)."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mlpal_assistants_service.db.models.base import Base


class UsageLog(Base):
    """
    Usage log table for tracking API usage.

    IMPORTANT: We do NOT store message content - only metadata.
    This is for privacy/compliance and matches provider behavior.

    Fields stored:
    - Context: user_id, api_key_id, trace_id
    - What was used: model_tag, provider, operation
    - Metrics: token counts, compute units, latency
    - Status: success/error

    Fields NOT stored:
    - User prompts
    - AI responses
    - Any message content

    Note: user_id references the users table in the configured user schema
    (MLPAL_USER_SCHEMA), not in the assistants schema. The FK is enforced at
    the application level, not database level, for schema flexibility.
    """

    __tablename__ = "usage_logs"
    __table_args__ = (
        Index("idx_usage_user", "user_id", "created_at"),
        Index("idx_usage_api_key", "api_key_id", "created_at"),
        Index("idx_usage_model", "model_tag", "created_at"),
        Index("idx_usage_trace", "trace_id"),
        Index("idx_usage_wallet_debit_status", "wallet_debit_status", "created_at"),
        Index("idx_usage_client_ip", "client_ip", "created_at"),
        {"schema": "assistants"},
    )

    # Composite primary key for potential partitioning
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        primary_key=True,  # Part of composite PK for partitioning
    )

    # Reference to user in the user schema (no FK constraint - cross-schema)
    user_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # Reference to API key (FK constraint to local table)
    api_key_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("assistants.api_keys.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Trace ID for debugging/correlation
    trace_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # What was used
    model_tag: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    operation: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    # Token counts (NOT the actual content)
    input_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )

    # Cost in compute units
    # 24,12: pass-through CU for a cheap-model request can be ~1e-7 — 12,6
    # truncated those to zero at persistence (caught 2026-08-11; the response
    # figure was always exact). Matches payments' wallet-column resolution.
    compute_units: Mapped[Decimal] = mapped_column(
        Numeric(24, 12),
        default=Decimal("0"),
        server_default="0",
        nullable=False,
    )
    # Performance
    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="success",
        nullable=False,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    wallet_debit_status: Mapped[str] = mapped_column(
        String(32),
        default="not_applicable",
        server_default="not_applicable",
        nullable=False,
    )

    wallet_debit_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )

    wallet_debit_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    wallet_debited_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Caller-side metadata: who/what made this call. Used by /v1/messages
    # to persist Claude Code session attribution (cc_session_id,
    # cc_agent_id, cc_parent_agent_id, anthropic_user_id, beta header
    # snapshot). JSONB so the shape can evolve as new attribution
    # signals appear without further migrations. GIN-indexed — see
    # alembic/versions/20260527_2300_cc_metadata.py.
    cc_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Client signals for the abuse detector (observability/client.py). The IP
    # is scrubbed after retention; the UA hash is a 16-hex sha256 prefix.
    client_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    client_ua_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Relationship to API key
    api_key = relationship("APIKey", backref="usage_logs")

    def __repr__(self) -> str:
        return f"<UsageLog(trace={self.trace_id}, model={self.model_tag}, cu={self.compute_units})>"
