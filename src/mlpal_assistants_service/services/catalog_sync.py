"""Declarative reconcile of the model registry + pricing against a curated feed.

The feed (``registry.json`` / ``pricing.json``, or a remote URL) is the DESIRED
STATE; the DB converges to it. Three transitions, not a blunt upsert:

  * present & new     → insert
  * present & changed → update the changed fields
  * feed-owned & gone → **soft-retire** (is_active=False, is_deprecated=True) —
    never a hard DELETE, because ``model_tag`` is referenced by usage, billing,
    per-key policy, and feedback; history and audit must survive.

Provenance guards local state: only ``source='mlpal-feed'`` rows are touched.
Operator-added (``source='local'``) models are never inserted-over, updated, or
retired by a feed refresh — so pinning your own model/adapter is safe.

Pricing is **effective-dated**: a price change inserts a new active row (today's
effective_date) and deactivates the superseded one, so past usage still prices
against the rate that was live then.

The planning functions are pure (no DB) and unit-tested; ``reconcile`` applies a
plan through the ORM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.db.models.meta_routing import MetaModelRouting
from mlpal_assistants_service.db.models.model_pricing import ModelPricing
from mlpal_assistants_service.db.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

FEED_SOURCE = "mlpal-feed"
LOCAL_SOURCE = "local"

# Feed-owned registry fields (compared to detect updates; model_tag is the key).
REGISTRY_FIELDS = (
    "provider", "provider_model_id", "display_name", "description", "capabilities",
    "context_length", "max_output_tokens", "pricing_tier", "fallback_model_tag",
    "priority", "is_active", "is_deprecated", "deprecation_message", "is_paused",
    "pause_reason",
)
# Pricing is keyed by (model_tag, operation); these fields define a "price".
# NOTE: input_cu_rate/output_cu_rate are Postgres GENERATED columns (derived
# from rate × markup ÷ cu_to_dollar) — they must never be written or compared;
# feeds may carry them (the live dump does) but the reconcile ignores them.
PRICING_KEY = ("model_tag", "operation")
PRICING_FIELDS = (
    "tier", "input_rate", "output_rate", "cache_read_rate", "rate_unit",
    "markup_multiplier", "cu_to_dollar",
)
_NUMERIC = {
    "input_rate", "output_rate", "cache_read_rate", "markup_multiplier",
    "cu_to_dollar",
}


def _norm(value: Any) -> Any:
    """Compare-normalize: numeric strings/Decimals collapse (``"3.00" == 3``),
    everything else compares by value (dicts, bools, strings, None)."""
    if isinstance(value, (dict, list, bool)) or value is None:
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return value


def _changed(feed_row: dict, existing: dict, fields) -> dict:
    """Fields present in the feed whose value differs from the existing row."""
    out = {}
    for f in fields:
        if f in feed_row and _norm(feed_row[f]) != _norm(existing.get(f)):
            out[f] = feed_row[f]
    return out


# ── registry plan ────────────────────────────────────────────────────────────


@dataclass
class RegistryPlan:
    insert: list[dict] = field(default_factory=list)        # full feed rows
    update: list[dict] = field(default_factory=list)        # {model_tag, changes}
    retire: list[str] = field(default_factory=list)         # model_tags to soft-retire

    @property
    def is_empty(self) -> bool:
        return not (self.insert or self.update or self.retire)


def plan_registry(existing: list[dict], feed: list[dict]) -> RegistryPlan:
    """Diff the feed against existing rows. `existing` rows must carry `source`,
    `model_tag`, and the REGISTRY_FIELDS (at least is_active/is_deprecated)."""
    ex = {r["model_tag"]: r for r in existing}
    feed_by_tag = {r["model_tag"]: r for r in feed}
    plan = RegistryPlan()

    for tag, frow in feed_by_tag.items():
        cur = ex.get(tag)
        if cur is None:
            plan.insert.append(frow)
        elif cur.get("source", FEED_SOURCE) == FEED_SOURCE:
            changes = _changed(frow, cur, REGISTRY_FIELDS)
            if changes:
                plan.update.append({"model_tag": tag, "changes": changes})
        # else: a local row shadows this tag — the feed never touches it.

    for tag, cur in ex.items():
        already_retired = not cur.get("is_active", True) and cur.get("is_deprecated", False)
        if (
            cur.get("source", FEED_SOURCE) == FEED_SOURCE
            and tag not in feed_by_tag
            and not already_retired
        ):
            plan.retire.append(tag)
    return plan


# ── pricing plan (effective-dated) ───────────────────────────────────────────


@dataclass
class PricingPlan:
    activate: list[dict] = field(default_factory=list)          # new active rows
    deactivate: list[tuple[str, str]] = field(default_factory=list)  # (tag, op) keys

    @property
    def is_empty(self) -> bool:
        return not (self.activate or self.deactivate)


def plan_pricing(existing_active: list[dict], feed: list[dict]) -> PricingPlan:
    """Only NEW or CHANGED prices are (re)activated; unchanged prices are left
    alone. A changed price supersedes the prior active row (deactivate).

    Dirty-feed tolerance: if the feed carries multiple rows for one
    (model_tag, operation) — e.g. a live dump where a superseded row was never
    deactivated — keep the one with the latest effective_date (last wins on a
    tie) instead of inserting both and violating uq_pricing."""
    deduped: dict[tuple[str, str], dict] = {}
    for frow in feed:
        key = (frow["model_tag"], frow["operation"])
        cur = deduped.get(key)
        if cur is None or str(frow.get("effective_date") or "") >= str(cur.get("effective_date") or ""):
            deduped[key] = frow

    ex = {(r["model_tag"], r["operation"]): r for r in existing_active}
    plan = PricingPlan()
    for key, frow in deduped.items():
        cur = ex.get(key)
        if cur is None or _changed(frow, cur, PRICING_FIELDS):
            plan.activate.append(frow)
            if cur is not None:
                plan.deactivate.append(key)
    return plan


# ── routing (router-tag candidate lists) ─────────────────────────────────────


@dataclass
class RoutingPlan:
    """Desired rows per (meta_model_tag, operation, priority); anything active
    in the DB beyond the desired shape is deactivated."""

    upsert: list[dict] = field(default_factory=list)   # {tag, op, priority, resolved, reason}
    deactivate: list[int] = field(default_factory=list)  # row ids


def plan_routing(existing_active: list[dict], feed_routes: list[dict]) -> RoutingPlan:
    """Converge meta_model_routing to the feed's ordered candidate lists.

    existing_active rows: {id, meta_model_tag, operation, resolved_model_tag,
    priority, reason}. Feed routes: {meta_model_tag, operation,
    candidates: [{model_tag, reason}, ...]} — index in the list IS the priority.
    """
    plan = RoutingPlan()
    desired: dict[tuple[str, str, int], dict] = {}
    for route in feed_routes:
        for prio, cand in enumerate(route["candidates"]):
            desired[(route["meta_model_tag"], route["operation"], prio)] = {
                "meta_model_tag": route["meta_model_tag"],
                "operation": route["operation"],
                "priority": prio,
                "resolved_model_tag": cand["model_tag"],
                "reason": cand.get("reason"),
            }

    ex_by_key = {(r["meta_model_tag"], r["operation"], r["priority"]): r for r in existing_active}
    for key, want in desired.items():
        have = ex_by_key.get(key)
        if have is None or have["resolved_model_tag"] != want["resolved_model_tag"] \
                or (have.get("reason") or None) != (want.get("reason") or None):
            plan.upsert.append(want)
    for key, have in ex_by_key.items():
        if key not in desired:
            plan.deactivate.append(have["id"])
    return plan


# ── summary + async apply ────────────────────────────────────────────────────


@dataclass
class ReconcileSummary:
    inserted: int = 0
    updated: int = 0
    retired: int = 0
    prices_activated: int = 0
    prices_deactivated: int = 0
    routes_upserted: int = 0
    routes_deactivated: int = 0

    def __str__(self) -> str:
        return (
            f"registry: +{self.inserted} ~{self.updated} retired {self.retired} | "
            f"pricing: +{self.prices_activated} superseded {self.prices_deactivated} | "
            f"routing: ~{self.routes_upserted} -{self.routes_deactivated}"
        )


def _registry_kwargs(row: dict) -> dict:
    return {k: row[k] for k in ("model_tag", *REGISTRY_FIELDS) if k in row}


def _pricing_kwargs(row: dict) -> dict:
    out: dict[str, Any] = {"model_tag": row["model_tag"], "operation": row["operation"]}
    for f in PRICING_FIELDS:
        if f not in row or row[f] is None:
            continue
        out[f] = Decimal(str(row[f])) if f in _NUMERIC else row[f]
    return out


async def reconcile(
    session: AsyncSession,
    registry_feed: list[dict],
    pricing_feed: list[dict],
    routing_feed: list[dict] | None = None,
    *,
    retire_message: str | None = None,
) -> ReconcileSummary:
    """Apply a full reconcile in one transaction. Idempotent: a second run with
    the same feed is a no-op."""
    retire_message = retire_message or f"Retired from the MLPal catalog feed on {date.today()}"
    summary = ReconcileSummary()

    # `user/` is the reserved tenant-model namespace (tenant_models table) —
    # a feed row can never claim it, so tenant/catalog collision is
    # structurally impossible, not just discouraged.
    blocked = [r["model_tag"] for r in registry_feed if r.get("model_tag", "").startswith("user/")]
    if blocked:
        logger.warning("catalog feed rejected reserved user/ tags: %s", blocked)
        registry_feed = [r for r in registry_feed if r["model_tag"] not in blocked]

    existing = {m.model_tag: m for m in (await session.execute(select(ModelRegistry))).scalars()}
    reg_plan = plan_registry(
        [{"model_tag": t, "source": m.source, **{f: getattr(m, f) for f in REGISTRY_FIELDS}}
         for t, m in existing.items()],
        registry_feed,
    )
    for row in reg_plan.insert:
        session.add(ModelRegistry(source=FEED_SOURCE, **_registry_kwargs(row)))
        summary.inserted += 1
    for upd in reg_plan.update:
        m = existing[upd["model_tag"]]
        for f, v in upd["changes"].items():
            setattr(m, f, v)
        summary.updated += 1
    for tag in reg_plan.retire:
        m = existing[tag]
        m.is_active = False
        m.is_deprecated = True
        m.deprecation_message = retire_message
        summary.retired += 1

    active = (
        await session.execute(select(ModelPricing).where(ModelPricing.is_active.is_(True)))
    ).scalars().all()
    by_key = {(p.model_tag, p.operation): p for p in active}
    price_plan = plan_pricing(
        [{"model_tag": p.model_tag, "operation": p.operation,
          **{f: getattr(p, f) for f in PRICING_FIELDS}} for p in active],
        pricing_feed,
    )
    for key in price_plan.deactivate:
        by_key[key].is_active = False
        summary.prices_deactivated += 1
    for row in price_plan.activate:
        # Same-day re-reprice: a row with (tag, op, today) may already exist
        # (possibly the one just deactivated above). uq_pricing spans ALL
        # rows, not only active ones, so INSERT would violate it and fail the
        # whole reconcile (hit live 2026-08-22: a correction shipped the same
        # day as the price it corrected). Update that row in place instead.
        existing_today = (
            await session.execute(
                select(ModelPricing).where(
                    ModelPricing.model_tag == row["model_tag"],
                    ModelPricing.operation == row["operation"],
                    ModelPricing.effective_date == date.today(),
                )
            )
        ).scalars().first()
        if existing_today is not None:
            for f in PRICING_FIELDS:
                if f in row:
                    setattr(existing_today, f, row[f])
            existing_today.is_active = True
        else:
            session.add(
                ModelPricing(is_active=True, effective_date=date.today(), **_pricing_kwargs(row))
            )
        summary.prices_activated += 1

    if routing_feed is not None:
        active_routes = (
            await session.execute(
                select(MetaModelRouting).where(MetaModelRouting.is_active.is_(True))
            )
        ).scalars().all()
        route_by_key = {(r.meta_model_tag, r.operation, r.priority): r for r in active_routes}
        route_plan = plan_routing(
            [{"id": r.id, "meta_model_tag": r.meta_model_tag, "operation": r.operation,
              "resolved_model_tag": r.resolved_model_tag, "priority": r.priority,
              "reason": r.reason} for r in active_routes],
            routing_feed,
        )
        for want in route_plan.upsert:
            have = route_by_key.get((want["meta_model_tag"], want["operation"], want["priority"]))
            if have is not None:
                have.resolved_model_tag = want["resolved_model_tag"]
                have.reason = want["reason"]
            else:
                session.add(MetaModelRouting(is_active=True, **want))
            summary.routes_upserted += 1
        by_id = {r.id: r for r in active_routes}
        for rid in route_plan.deactivate:
            by_id[rid].is_active = False
            summary.routes_deactivated += 1

    await session.commit()
    return summary
