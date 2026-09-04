"""Meta Model Routing - maps MLPal meta-models to actual provider models."""

from sqlalchemy import Boolean, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base, TimestampMixin


class MetaModelRouting(Base, TimestampMixin):
    """
    Routing rules for MLPal meta-models.

    Maps (meta_model_tag, operation) -> resolved_model_tag

    Example:
        meta_model_tag = "mlpal"
        operation = "image_generation"
        resolved_model_tag = "gemini-3-pro-image"

    This allows mlpal, mlpal-flash, and mlpal-lite to route to the
    optimal model for each operation (chat, image_generation, tts, etc.)
    """

    __tablename__ = "meta_model_routing"
    __table_args__ = (
        Index(
            "idx_meta_routing_lookup",
            "meta_model_tag",
            "operation",
            "is_active",
        ),
        # One active row per (tag, operation, priority) — candidate LISTS are
        # the model now (priority-ordered, availability-resolved at request time).
        Index(
            "idx_meta_routing_unique_active",
            "meta_model_tag",
            "operation",
            "priority",
            unique=True,
            postgresql_where="is_active = true",
        ),
        {"schema": "assistants"},
    )

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Meta model identification
    meta_model_tag: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Meta model tag: mlpal, mlpal-flash, mlpal-lite",
    )

    # Operation type
    operation: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Operation: chat, image_generation, tts, transcription, embedding, structured_output",
    )

    # Resolved model
    resolved_model_tag: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Actual model tag to use (must exist in model_registry)",
    )

    # Priority for fallback ordering (lower = higher priority)
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Priority for fallback ordering (0 = primary)",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Reason for this routing (for documentation)
    reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Why this model was chosen for this operation",
    )

    def __repr__(self) -> str:
        return (
            f"<MetaModelRouting("
            f"{self.meta_model_tag}/{self.operation} -> {self.resolved_model_tag}"
            f")>"
        )
