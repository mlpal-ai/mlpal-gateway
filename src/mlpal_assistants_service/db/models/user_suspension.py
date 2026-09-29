"""Account-level suspension (migration 20260929_1000)."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from mlpal_assistants_service.db.models.base import Base


class UserSuspension(Base):
    __tablename__ = "user_suspensions"
    __table_args__ = {"schema": "assistants"}

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    incident: Mapped[str | None] = mapped_column(String(100), nullable=True)
    suspended_by: Mapped[str] = mapped_column(String(100), nullable=False)
    suspended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    lifted_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lifted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def active(self) -> bool:
        return self.lifted_at is None
