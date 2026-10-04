"""Account trust: how much a user may spend per day before they have paid.

Why (incident 2026-09-28): farm accounts spent their whole welcome credit in
minutes on long-context frontier models. The founder's constraint is the
other way round — a genuine new user must be able to try the latest model on
day one. So the ramp caps *rate*, never model access:

  * first ``young_account_first_window_hours`` (48 h):
    ``young_account_first_ceiling_cu`` per day (2 CU)
  * until the first paid top-up: ``young_account_unpaid_ceiling_cu`` (10 CU)
  * any paid wallet transaction (``paid_wallet_sources``): no account ceiling —
    the tier and the key budgets govern

"Paid" is read from the backend's wallet ledger; account age from the users
table. Both live in the backend schema (``settings.user_schema``), so on a
deployment without them (OSS, local billing) there is never a ceiling. The
profile is cached in Redis for ``account_trust_cache_ttl_seconds``; the
backend drops it through ``/internal/wallet-cache/invalidate`` when a payment
lands, so a top-up lifts the ceiling at once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from mlpal_assistants_service.core.config import get_settings

logger = logging.getLogger(__name__)

# None = not probed yet; probed once per process (backend schema present?).
_SCHEMA_PRESENT: bool | None = None


def trust_cache_key(user_id: int | str) -> str:
    return f"trust:{user_id}"


@dataclass(frozen=True)
class AccountTrust:
    user_id: int
    age_hours: float
    paid: bool
    held: bool

    def daily_ceiling_cu(self, settings: Any) -> Decimal | None:
        """Account-level daily CU cap, or None when no ceiling applies."""
        if self.paid:
            return None
        if self.age_hours < float(settings.young_account_first_window_hours):
            return Decimal(str(settings.young_account_first_ceiling_cu))
        return Decimal(str(settings.young_account_unpaid_ceiling_cu))


class AccountTrustService:
    """Resolves (and caches) an account's trust profile."""

    def __init__(self, session: Any, redis: Any, *, settings: Any = None) -> None:
        self._session = session
        self._redis = redis
        self._settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        s = self._settings
        return bool(getattr(s, "young_account_ramp_enabled", False)) and (
            getattr(s, "billing_backend", "managed") != "local"
        )

    async def daily_ceiling_cu(self, user_id: int | str) -> Decimal | None:
        """The ceiling to enforce for this request, or None. Never raises:
        an unreadable profile means no ceiling (the wallet gate and the key
        budgets still apply), logged so it is not silent."""
        if not self.enabled:
            return None
        trust = await self.resolve(user_id)
        return trust.daily_ceiling_cu(self._settings) if trust else None

    async def resolve(self, user_id: int | str) -> AccountTrust | None:
        cached = await self._cache_get(trust_cache_key(user_id))
        if cached:
            try:
                data = json.loads(cached)
                return AccountTrust(**data)
            except (ValueError, TypeError):
                pass
        trust = await self._load(int(user_id))
        if trust is not None:
            await self._cache_set(trust_cache_key(user_id), json.dumps(asdict(trust)))
        return trust

    async def _load(self, user_id: int) -> AccountTrust | None:
        global _SCHEMA_PRESENT
        if _SCHEMA_PRESENT is False:
            return None
        schema = self._settings.user_schema
        try:
            if _SCHEMA_PRESENT is None:
                probe = await self._session.execute(
                    text(
                        "SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema = :s AND table_name IN ('users', 'wallets', 'wallet_transactions')"
                    ),
                    {"s": schema},
                )
                _SCHEMA_PRESENT = (probe.scalar_one() or 0) == 3
                if not _SCHEMA_PRESENT:
                    logger.info("account trust: backend schema %s absent; no account ceilings", schema)
                    return None
            sources = [s.strip() for s in self._settings.paid_wallet_sources.split(",") if s.strip()]
            row = (
                await self._session.execute(
                    text(
                        f"SELECT u.created_at, u.hold_reason, "
                        f"EXISTS (SELECT 1 FROM {schema}.wallet_transactions t "
                        f"        JOIN {schema}.wallets w ON w.id = t.wallet_id "
                        f"        WHERE w.owner_id = u.id AND t.source = ANY(:sources)) AS paid "
                        f"FROM {schema}.users u WHERE u.id = :uid"
                    ),
                    {"uid": user_id, "sources": sources},
                )
            ).first()
        except Exception as e:  # noqa: BLE001 — never block traffic on this lookup
            logger.error("account trust lookup failed user_id=%s: %s", user_id, e)
            return None
        if row is None:
            return None
        created_at, hold_reason, paid = row
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_hours = (
            (datetime.now(UTC) - created_at).total_seconds() / 3600 if created_at else float("inf")
        )
        return AccountTrust(user_id=user_id, age_hours=age_hours, paid=bool(paid), held=bool(hold_reason))

    async def _cache_get(self, key: str) -> str | None:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(key)
            return raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # noqa: BLE001 — cache miss on any Redis fault
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.setex(key, int(self._settings.account_trust_cache_ttl_seconds), value)
        except Exception:  # noqa: BLE001 — best effort
            logger.debug("account trust cache write failed", exc_info=True)
