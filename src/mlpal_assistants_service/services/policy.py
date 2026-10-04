"""Per-API-key policy engine: model access control + spend budgets.

Two enforcement primitives, both designed for the hot path:

  * MODEL ACCESS — an allow/deny list (glob patterns, ``*`` = all) attached to the
    key. Deny wins. Checked against the requested tag AND the meta-resolved
    concrete tag, so neither an alias nor a fallback can smuggle a denied model
    past the gate. Pure in-memory: no I/O.

  * SPEND BUDGETS — a list of ``{unit: usd|cu, amount, window}`` rules with
    calendar-aligned windows (daily / weekly / monthly / lifetime) in a
    configured timezone. Spend is tracked in COMPUTE UNITS (what we bill);
    a USD budget is normalized to CU via ``cu_to_usd``. Enforcement is
    "deny when spent >= limit": one already-admitted request may overshoot by
    its own cost (we never token-estimate — see the no-estimate rule), then
    everything after is hard-denied. This matches how OpenRouter/Cloudflare
    treat hard caps (bounded in-flight overage) and keeps the pre-check to a
    single Redis read.

Spend counter model: ``usage_logs`` is the source of truth; Redis holds a
per-(key, window) running counter that is (a) incremented on each call's actual
CU and (b) re-seeded from the DB window sum whenever it is absent (cold key, pod
restart, TTL expiry). The counter's TTL is the time to the window's end, so a
new window starts fresh automatically. On a total store outage the check
fails OPEN (logged), matching the rest of the service's availability posture.
"""

from __future__ import annotations

import fnmatch
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    BudgetExceededError,
    ModelAccessDeniedError,
)

logger = logging.getLogger(__name__)

WINDOWS = ("daily", "weekly", "monthly", "lifetime")
UNITS = ("usd", "cu")
_COUNTER_PREFIX = "kbudget:"
# Seconds a window counter lives past the window's end before Redis evicts it —
# a small grace so a request completing right at the boundary still accrues.
_TTL_GRACE = 3600
# A lifetime counter used to live forever in Redis. It is a cache over
# usage_logs (a miss re-seeds from the DB), so it may expire: 30 days after the
# last accrual, or — when the caller knows the key's expiry (short-lived HOP
# lease keys: two per turn) — shortly after the key itself dies.
_LIFETIME_TTL = 30 * 24 * 3600


# ---------------------------------------------------------------------------
# Window math (calendar-aligned, timezone-aware). Pure functions so they're
# trivially unit-testable without Redis or a clock seam beyond `now`.
# ---------------------------------------------------------------------------

def window_id(window: str, now_local: datetime) -> str:
    """Stable identifier for the window instance containing `now_local`.

    Immutable per window instance, so a request's accrual and a later reconcile
    land on the same counter. Weekly uses ISO week (Monday start)."""
    if window == "daily":
        return now_local.strftime("%Y-%m-%d")
    if window == "weekly":
        iso = now_local.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if window == "monthly":
        return now_local.strftime("%Y-%m")
    return "lifetime"


def window_bounds(window: str, now_local: datetime) -> tuple[datetime | None, datetime | None]:
    """(start, end) of the current window as UTC datetimes, for DB reconcile and
    reset-time reporting. Lifetime is unbounded -> (None, None)."""
    if window == "lifetime":
        return None, None
    if window == "daily":
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif window == "weekly":
        midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start = midnight - timedelta(days=midnight.weekday())  # back to Monday
        end = start + timedelta(days=7)
    elif window == "monthly":
        start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # first of next month
        end = (start + timedelta(days=32)).replace(day=1)
    else:  # defensive; validated upstream
        return None, None
    return start.astimezone(UTC), end.astimezone(UTC)


class PolicyService:
    """Stateless-per-request policy checks over a key's compiled policy.

    Construct cheaply per request (holds only a redis handle + a usage repo for
    DB reconcile). Model-access methods need no I/O; budget methods touch Redis
    and fall back to the usage repo.
    """

    def __init__(
        self, redis: Any, usage_repo: Any, *, settings: Any = None, trust: Any = None
    ) -> None:
        self._redis = redis
        self._usage_repo = usage_repo
        self._settings = settings or get_settings()
        # services/account_trust.AccountTrustService (None = no account ceilings)
        self._trust = trust

    # ----- model access (pure, no I/O) -------------------------------------

    @staticmethod
    def is_model_allowed(model_policy: dict | None, model: str) -> bool:
        """True if `model` passes the policy. No policy / no lists => allowed.
        Deny (union) wins over allow (with `*` or empty allow = allow-all)."""
        if not model_policy:
            return True
        deny = model_policy.get("deny") or []
        if any(fnmatch.fnmatchcase(model, p) for p in deny):
            return False
        allow = model_policy.get("allow") or ["*"]
        if "*" in allow:
            return True
        return any(fnmatch.fnmatchcase(model, p) for p in allow)

    def check_model_access(
        self, model_policy: dict | None, requested: str, resolved: str | None = None
    ) -> None:
        """Raise ModelAccessDeniedError unless the key may use this model.

        Deny is checked against BOTH the requested alias and the resolved
        concrete model (so a denied model can't be reached via an alias/fallback).
        Allow passes if EITHER the requested alias or the resolved model is
        permitted (allowing an alias like `mlpal-lite` grants its routing)."""
        if not model_policy:
            return
        tags = [t for t in (requested, resolved) if t]
        deny = model_policy.get("deny") or []
        for t in tags:
            if any(fnmatch.fnmatchcase(t, p) for p in deny):
                raise ModelAccessDeniedError(t, "explicitly denied by key policy")
        allow = model_policy.get("allow") or ["*"]
        if "*" in allow:
            return
        if any(any(fnmatch.fnmatchcase(t, p) for p in allow) for t in tags):
            return
        raise ModelAccessDeniedError(requested, "not in this key's allowed models")

    @staticmethod
    def is_routing_allowed(model_policy: dict | None, alias: str, resolved: str) -> bool:
        """Listing-side mirror of check_model_access for a meta-model routing:
        deny wins on either the alias or the resolved model; allow passes if
        either is permitted (allowing `mlpal-lite` grants its routing)."""
        if not model_policy:
            return True
        deny = model_policy.get("deny") or []
        if any(fnmatch.fnmatchcase(t, p) for t in (alias, resolved) for p in deny):
            return False
        allow = model_policy.get("allow") or ["*"]
        if "*" in allow:
            return True
        return any(fnmatch.fnmatchcase(t, p) for t in (alias, resolved) for p in allow)

    def filter_models(self, model_policy: dict | None, tags: list[str]) -> list[str]:
        """Subset of `tags` the key may use — for the model-list endpoint."""
        if not model_policy:
            return list(tags)
        return [t for t in tags if self.is_model_allowed(model_policy, t)]

    # ----- unit conversion -------------------------------------------------

    def _to_cu(self, amount: Decimal, unit: str) -> Decimal:
        """Normalize a budget amount to compute units (the recorded unit)."""
        if unit == "usd":
            return amount / Decimal(str(self._settings.cu_to_usd))
        return amount  # already CU

    def _from_cu(self, cu: Decimal, unit: str) -> Decimal:
        """CU spend back into the budget's display unit (for the error body)."""
        if unit == "usd":
            return cu * Decimal(str(self._settings.cu_to_usd))
        return cu

    # ----- budget windows / counters --------------------------------------

    def _now_local(self) -> datetime:
        return datetime.now(ZoneInfo(self._settings.budget_timezone))

    def _counter_key(self, subject: int | str, window: str, wid: str) -> str:
        # Keys are the bare id (existing live counters keep working); accounts
        # are "user:<id>" (young-account daily ceiling).
        return f"{_COUNTER_PREFIX}{subject}:{window}:{wid}"

    def _ttl_seconds(
        self, window: str, now_local: datetime, expires_at: datetime | None = None
    ) -> int | None:
        """Counter TTL: to the window's end (+grace); lifetime windows 30 days
        from the last touch; either capped to the key's own expiry when known."""
        if window == "lifetime":
            secs = _LIFETIME_TTL
        else:
            _, end_utc = window_bounds(window, now_local)
            if end_utc is None:
                secs = _LIFETIME_TTL
            else:
                secs = max(int((end_utc - datetime.now(UTC)).total_seconds()) + _TTL_GRACE, _TTL_GRACE)
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            key_secs = max(int((expires_at - datetime.now(UTC)).total_seconds()) + _TTL_GRACE, _TTL_GRACE)
            secs = min(secs, key_secs)
        return secs

    async def _window_spend_cu(
        self,
        subject: int | str,
        window: str,
        now_local: datetime,
        reconcile: Any,
        expires_at: datetime | None = None,
    ) -> Decimal:
        """Current-window spend in CU. Redis counter, re-seeded from the DB window
        sum (`reconcile(start, end)`) on miss. Fails OPEN (0) if both stores are
        unreachable."""
        wid = window_id(window, now_local)
        ckey = self._counter_key(subject, window, wid)
        try:
            cached = await self._redis.get(ckey)
            if cached is not None:
                return Decimal(cached.decode() if isinstance(cached, bytes) else cached)
        except Exception:  # noqa: BLE001 — Redis down: fall through to DB
            logger.warning("policy: budget counter read failed for %s", ckey, exc_info=True)
        start_utc, end_utc = window_bounds(window, now_local)
        try:
            spent = Decimal((await reconcile(start_utc, end_utc)) or 0)
        except Exception:  # noqa: BLE001 — DB also down: fail open, don't block traffic
            logger.warning("policy: budget DB reconcile failed for %s", subject, exc_info=True)
            return Decimal(0)
        # Seed the counter so subsequent requests are Redis-only for this window.
        try:
            await self._redis.set(ckey, str(spent), ex=self._ttl_seconds(window, now_local, expires_at))
        except Exception:  # noqa: BLE001 — best-effort seed
            logger.debug("policy: budget counter seed failed for %s", ckey, exc_info=True)
        return spent

    async def _check_rules(
        self,
        subject: int | str,
        budgets: list[dict],
        reconcile: Any,
        *,
        label: str = "Spend budget exhausted",
        hint: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        now_local = self._now_local()
        # Hot path: read every window counter in ONE pipeline round trip; the
        # per-window reseed path (_window_spend_cu) runs only on a cache miss.
        windows = list({b["window"] for b in budgets})
        spend: dict[str, Decimal] = {}
        try:
            pipe = self._redis.pipeline(transaction=False)
            for window in windows:
                pipe.get(self._counter_key(subject, window, window_id(window, now_local)))
            for window, raw in zip(windows, await pipe.execute(), strict=True):
                if raw is not None:
                    spend[window] = Decimal(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:  # noqa: BLE001 — Redis down: reseed path handles it
            logger.warning("policy: batched budget read failed for %s", subject, exc_info=True)
        for b in budgets:
            window, unit = b["window"], b["unit"]
            amount = Decimal(str(b["amount"]))
            limit_cu = self._to_cu(amount, unit)
            spent_cu = spend.get(window)
            if spent_cu is None:
                spent_cu = await self._window_spend_cu(subject, window, now_local, reconcile, expires_at)
                spend[window] = spent_cu
            if spent_cu >= limit_cu:
                _, end_utc = window_bounds(window, now_local)
                raise BudgetExceededError(
                    window=window,
                    unit=unit,
                    limit=float(amount),
                    spent=float(round(self._from_cu(spent_cu, unit), 6)),
                    reset_at=end_utc.isoformat() if end_utc else None,
                    budget_id=b.get("id"),
                    label=label,
                    hint=hint,
                )

    async def check_budgets(
        self,
        api_key_id: int,
        budgets: list[dict] | None,
        user_id: int | str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Raise BudgetExceededError if ANY of the key's budget windows is at/over
        its cap (deny when spent >= limit: bounded single-request overage), or,
        when `user_id` is given and the account is still ramping, if the
        account's daily ceiling is reached (services/account_trust.py)."""
        if budgets:
            await self._check_rules(
                api_key_id,
                budgets,
                lambda s, e: self._usage_repo.get_api_key_cu_in_window(api_key_id, s, e),
                expires_at=expires_at,
            )
        if user_id is not None and self._trust is not None:
            ceiling = await self._trust.daily_ceiling_cu(user_id)
            if ceiling is not None:
                await self._check_rules(
                    f"user:{user_id}",
                    [self._account_rule(ceiling)],
                    lambda s, e: self._usage_repo.get_user_cu_in_window(int(user_id), s, e),
                    label="New-account daily limit reached",
                    hint="Add a payment method in Billing to lift it.",
                )

    @staticmethod
    def _account_rule(ceiling_cu: Decimal) -> dict:
        return {"id": "young-account-daily", "unit": "cu", "amount": str(ceiling_cu), "window": "daily"}

    async def record_key_usage(
        self,
        api_key_id: int,
        budgets: list[dict] | None,
        cu: Decimal | float,
        user_id: int | str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Accrue `cu` onto every window this key budgets on, and onto the
        account's daily counter when a ramp applies. Best-effort, pipelined;
        fails open (spend still lands in usage_logs and re-seeds)."""
        cu = Decimal(str(cu))
        if cu <= 0:
            return
        targets: list[tuple[int | str, str]] = [(api_key_id, b["window"]) for b in (budgets or [])]
        if user_id is not None and self._trust is not None and self._trust.enabled:
            targets.append((f"user:{user_id}", "daily"))
            if await self._trust.daily_ceiling_cu(user_id) is not None:
                # Spend by accounts still on the ramp — the "new accounts are
                # burning credit" alarm (incident 2026-09-28 shape).
                try:
                    from mlpal_assistants_service.core.metrics import get_metrics

                    get_metrics().put_metric_sync("YoungAccountComputeUnits", float(cu), "None")
                except Exception:  # noqa: BLE001 — metrics never break accrual
                    logger.debug("YoungAccountComputeUnits emit failed", exc_info=True)
        if not targets:
            return
        now_local = self._now_local()
        try:
            # Increment ONLY when the counter exists. incrbyfloat would CREATE
            # a missing counter at just this request's cu — after a Redis
            # restart/eviction mid-window that tiny counter looks "seeded", so
            # check_budgets never takes its usage_logs reconcile path and the
            # cap is silently lifted for the rest of the window. Leaving the
            # key absent instead routes the next check through the reseed.
            lua = (
                "if redis.call('EXISTS', KEYS[1]) == 1 then "
                "return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1]) "
                "else return false end"
            )
            pipe = self._redis.pipeline(transaction=False)
            for subject, window in dict.fromkeys(targets):
                wid = window_id(window, now_local)
                ckey = self._counter_key(subject, window, wid)
                pipe.eval(lua, 1, ckey, float(cu))
                # the account counter is not bound to any one key's expiry
                pipe.expire(ckey, self._ttl_seconds(window, now_local, None if subject != api_key_id else expires_at))
            await pipe.execute()
        except Exception:  # noqa: BLE001
            logger.warning("policy: budget accrual failed for key %s", api_key_id, exc_info=True)
