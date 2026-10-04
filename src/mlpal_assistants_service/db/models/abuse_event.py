"""Abuse detector findings: one row per (rule, cluster, batch of accounts).

The evidence that justified an automatic hold or suspension, kept so a
false positive is explainable and reversible. `action` is what the detector
did: `observe` (recorded only), `suspend` (accounts suspended via
APIKeyService.suspend_user with incident `auto:<rule>:<cluster_key>`).
"""

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base

ABUSE_ACTIONS = ("observe", "suspend")


class AbuseEvent(Base):
    __tablename__ = "abuse_events"
    __table_args__ = (
        Index("idx_abuse_events_cluster", "rule", "cluster_key", "created_at"),
        {"schema": "assistants"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    rule: Mapped[str] = mapped_column(String(40), nullable=False)
    # The shared attribute the cluster hangs on (an IP, a UA hash).
    cluster_key: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    # {"user_ids": [...], ...rule-specific evidence}
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)

    def __repr__(self) -> str:
        return f"<AbuseEvent(rule={self.rule}, cluster={self.cluster_key}, action={self.action})>"
