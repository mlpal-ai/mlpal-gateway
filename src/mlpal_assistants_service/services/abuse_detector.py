"""Abuse detector: cluster young accounts by shared client signals and act.

What it looks for (all over accounts whose first API key is < 48 h old):

- ``lockstep_ip``   ≥ 5 young accounts calling from one client IP in the
                    last hour.
- ``lockstep_ua``   ≥ 5 young accounts sharing a user-agent hash whose calls
                    land in the same one-minute buckets (≥ 3 shared buckets).
- ``key_mint_burst`` ≥ 3 young accounts minting keys from one IP inside ten
                    minutes (the console operator working down a list).
- ``key_mint_burst_ua`` ≥ 8 young accounts minting keys with one user-agent
                    hash inside ten minutes (one scripted client rotating IPs).

Every finding becomes an ``abuse_events`` row carrying the evidence. In
enforce mode the two lockstep rules suspend the cluster (incident
``auto:<rule>:<cluster_key>``, lifted like any other suspension);
the key-mint rules only ever record — holding a signup is the backend's
call. A cluster already recorded in the last 24 h is only re-reported for
accounts that were not in the earlier event, so a tick never spams.

The rules run in Python over three bounded queries (young accounts, their
usage in the last hour, their keys in the last day) rather than in SQL:
portable across the OSS SQLite tests and Postgres, and the volume is small
by construction — accounts under two days old.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.metrics import get_metrics
from mlpal_assistants_service.db.models import AbuseEvent, APIKey, UsageLog, UserSuspension

logger = logging.getLogger(__name__)
audit = structlog.get_logger("audit.abuse")

RULE_LOCKSTEP_IP = "lockstep_ip"
RULE_LOCKSTEP_UA = "lockstep_ua"
RULE_KEY_MINT_BURST = "key_mint_burst"
RULE_KEY_MINT_BURST_UA = "key_mint_burst_ua"
RULE_KEY_FANOUT = "key_fanout"
ENFORCED_RULES = frozenset({RULE_LOCKSTEP_IP, RULE_LOCKSTEP_UA})

YOUNG_ACCOUNT_AGE = timedelta(hours=48)
USAGE_WINDOW = timedelta(minutes=60)
KEY_WINDOW = timedelta(hours=24)
KEY_BURST_WINDOW = timedelta(minutes=10)
DEDUPE_WINDOW = timedelta(hours=24)
MIN_CLUSTER = 5
MIN_SHARED_BUCKETS = 3
MIN_BUCKET_USERS = 3
KEY_BURST_MIN = 3
KEY_BURST_UA_MIN = 8

KEY_FANOUT_WINDOW = timedelta(minutes=60)
KEY_FANOUT_EXEMPT_TIERS = frozenset({"enterprise"})
KEY_FANOUT_EXEMPT_TAG = "fanout_ok"

AUTO_SUSPEND_REASON = (
    "Your account has been suspended due to suspicious activity. Contact contact@mlpal.ai."
)


@dataclass(frozen=True)
class UsageRow:
    user_id: int
    at: datetime
    client_ip: str | None
    ua_hash: str | None


@dataclass(frozen=True)
class KeyRow:
    user_id: int
    at: datetime
    created_ip: str | None
    created_ua_hash: str | None = None


@dataclass(frozen=True)
class KeyUsageRow:
    api_key_id: int
    user_id: int
    client_ip: str | None
    model_tag: str | None = None


@dataclass
class Finding:
    rule: str
    cluster_key: str
    user_ids: list[int]
    evidence: dict[str, Any] = field(default_factory=dict)
    action: str = "observe"


def _minute(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M")


def find_lockstep_ip(usage: list[UsageRow]) -> list[Finding]:
    by_ip: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in usage:
        if row.client_ip:
            by_ip[row.client_ip][row.user_id] += 1
    findings = []
    for ip, calls in by_ip.items():
        if len(calls) >= MIN_CLUSTER:
            findings.append(
                Finding(
                    RULE_LOCKSTEP_IP,
                    ip,
                    sorted(calls),
                    {"calls_per_user": {str(u): n for u, n in sorted(calls.items())}},
                )
            )
    return findings


def find_lockstep_ua(usage: list[UsageRow]) -> list[Finding]:
    by_ua: dict[str, list[UsageRow]] = defaultdict(list)
    for row in usage:
        if row.ua_hash:
            by_ua[row.ua_hash].append(row)
    findings = []
    for ua, rows in by_ua.items():
        if len({r.user_id for r in rows}) < MIN_CLUSTER:
            continue
        buckets: dict[str, set[int]] = defaultdict(set)
        for r in rows:
            buckets[_minute(r.at)].add(r.user_id)
        shared = {b: users for b, users in buckets.items() if len(users) >= MIN_BUCKET_USERS}
        if len(shared) < MIN_SHARED_BUCKETS:
            continue
        users = set().union(*shared.values())
        if len(users) < MIN_CLUSTER:
            continue
        findings.append(
            Finding(
                RULE_LOCKSTEP_UA,
                ua,
                sorted(users),
                {"shared_minutes": sorted(shared), "users_per_minute": {b: len(u) for b, u in sorted(shared.items())}},
            )
        )
    return findings


def _mint_bursts(keys: list[KeyRow], attr: str, rule: str, min_users: int) -> list[Finding]:
    groups: dict[str, list[KeyRow]] = defaultdict(list)
    for k in keys:
        v = getattr(k, attr)
        if v:
            groups[v].append(k)
    findings = []
    for key, rows in groups.items():
        rows.sort(key=lambda k: k.at)
        best: set[int] = set()
        for i, first in enumerate(rows):
            window = {k.user_id for k in rows[i:] if k.at - first.at <= KEY_BURST_WINDOW}
            if len(window) > len(best):
                best = window
        if len(best) >= min_users:
            findings.append(
                Finding(
                    rule,
                    key,
                    sorted(best),
                    {"keys_in_group_24h": len(rows), "window_minutes": int(KEY_BURST_WINDOW.total_seconds() // 60)},
                )
            )
    return findings


def find_key_mint_burst(keys: list[KeyRow]) -> list[Finding]:
    """Two groupings of the same shape. By IP: the console operator working
    down a list of accounts (2026-09-29, 5 keys in 2.5 min). By UA hash: the
    2026-09-30 wave rotated an IP per account but minted 17 keys with one
    client, so the IP grouping never fired. The UA bar is higher because a
    popular browser build is a legitimately shared hash."""
    return _mint_bursts(keys, "created_ip", RULE_KEY_MINT_BURST, KEY_BURST_MIN) + _mint_bursts(
        keys, "created_ua_hash", RULE_KEY_MINT_BURST_UA, KEY_BURST_UA_MIN
    )


def _net16(ip: str) -> str | None:
    """Coarse network of a client IP: /16 for IPv4, /32 for IPv6."""
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    bits = 16 if addr.version == 4 else 32
    return str(ipaddress.ip_network(f"{addr}/{bits}", strict=False))


def find_key_fanout(rows: list[KeyUsageRow], *, min_ips: int, min_nets: int) -> list[Finding]:
    """One key, many places: a key used from >= min_ips distinct IPs spread
    over >= min_nets coarse networks inside the window. Both thresholds must
    hold — a NAT fleet (many IPs, one network) or a travelling laptop (few
    IPs) never qualifies; a resold key (incident B: 110 IPs) does."""
    by_key: dict[int, list[KeyUsageRow]] = defaultdict(list)
    for r in rows:
        if r.client_ip:
            by_key[r.api_key_id].append(r)
    out: list[Finding] = []
    for key_id, krows in sorted(by_key.items()):
        ips = {r.client_ip for r in krows}
        nets = {n for n in (_net16(ip) for ip in ips) if n}
        if len(ips) < min_ips or len(nets) < min_nets:
            continue
        models: dict[str, int] = defaultdict(int)
        for r in krows:
            if r.model_tag:
                models[r.model_tag] += 1
        out.append(
            Finding(
                RULE_KEY_FANOUT,
                f"key:{key_id}",
                [krows[0].user_id],
                evidence={
                    "api_key_id": key_id,
                    "distinct_ips": len(ips),
                    "distinct_nets": len(nets),
                    "sample_ips": sorted(ips)[:5],
                    "models": dict(sorted(models.items(), key=lambda kv: -kv[1])[:5]),
                    "window_minutes": int(KEY_FANOUT_WINDOW.total_seconds() // 60),
                },
            )
        )
    return out


def find_clusters(usage: list[UsageRow], keys: list[KeyRow]) -> list[Finding]:
    return find_lockstep_ip(usage) + find_lockstep_ua(usage) + find_key_mint_burst(keys)


class AbuseDetector:
    def __init__(
        self,
        session: AsyncSession,
        redis_client=None,
        *,
        enforce: bool = False,
        enforce_key_fanout: bool | None = None,
    ) -> None:
        self._session = session
        self._redis = redis_client
        self._enforce = enforce
        settings = get_settings()
        self._enforce_key_fanout = (
            settings.abuse_key_fanout_enforce if enforce_key_fanout is None else enforce_key_fanout
        )
        self._fanout_min_ips = settings.abuse_key_fanout_min_ips
        self._fanout_min_nets = settings.abuse_key_fanout_min_nets

    async def run_once(self, now: datetime | None = None) -> list[Finding]:
        now = now or datetime.now(UTC)
        acted: list[Finding] = list(await self._fanout_pass(now))
        young = await self._young_accounts(now)
        usage, keys = (await self._signals(young, now)) if young else ([], [])
        for finding in find_clusters(usage, keys):
            known = await self._known_users(finding, now)
            new_users = [u for u in finding.user_ids if u not in known]
            if not new_users:
                continue
            finding.evidence["user_ids"] = new_users
            finding.evidence["cluster_size"] = len(finding.user_ids)
            if self._enforce and finding.rule in ENFORCED_RULES:
                finding.action = "suspend"
                await self._suspend(finding, new_users)
            self._session.add(
                AbuseEvent(
                    rule=finding.rule, cluster_key=finding.cluster_key,
                    action=finding.action, evidence=finding.evidence,
                )
            )
            acted.append(finding)
            audit.warning(
                "abuse cluster detected", rule=finding.rule, cluster_key=finding.cluster_key,
                action=finding.action, user_ids=new_users, cluster_size=len(finding.user_ids),
            )
            await get_metrics().put_metric(
                "AbuseClusterDetected", len(new_users), "Count",
                {"rule": finding.rule, "action": finding.action},
            )
        if acted:
            await self._session.commit()
            await publish_finding_metrics(acted)
        return acted

    # ----- key fan-out (every key, not only young accounts) -----------------

    async def _fanout_pass(self, now: datetime) -> list[Finding]:
        """Detect, dedupe, enforce and record key fan-out findings. Never
        raises: a fault here must not stop the cluster pass."""
        try:
            rows = await self._key_usage(now)
        except Exception:  # noqa: BLE001
            logger.error("abuse: key usage query failed", exc_info=True)
            return []
        acted: list[Finding] = []
        for finding in find_key_fanout(
            rows, min_ips=self._fanout_min_ips, min_nets=self._fanout_min_nets
        ):
            if await self._known_key(finding, now):
                continue
            exempt = await self._key_exempt(finding.evidence["api_key_id"])
            if exempt:
                finding.evidence["exempt"] = exempt
            elif self._enforce_key_fanout:
                finding.action = "revoke_key"
                await self._revoke_key(finding)
            self._session.add(
                AbuseEvent(
                    rule=finding.rule, cluster_key=finding.cluster_key,
                    action=finding.action, evidence=finding.evidence,
                )
            )
            acted.append(finding)
            audit.warning(
                "key fan-out detected", rule=finding.rule, cluster_key=finding.cluster_key,
                action=finding.action, user_ids=finding.user_ids,
                distinct_ips=finding.evidence["distinct_ips"], distinct_nets=finding.evidence["distinct_nets"],
            )
            await get_metrics().put_metric(
                "AbuseClusterDetected", 1, "Count", {"rule": finding.rule, "action": finding.action}
            )
        return acted

    async def _key_usage(self, now: datetime) -> list[KeyUsageRow]:
        result = await self._session.execute(
            select(UsageLog.api_key_id, UsageLog.user_id, UsageLog.client_ip, UsageLog.model_tag)
            .where(UsageLog.created_at >= now - KEY_FANOUT_WINDOW, UsageLog.client_ip.is_not(None))
        )
        return [KeyUsageRow(int(r[0]), int(r[1]), r[2], r[3]) for r in result.all()]

    async def _known_key(self, finding: Finding, now: datetime) -> bool:
        result = await self._session.execute(
            select(AbuseEvent.id).where(
                AbuseEvent.rule == finding.rule,
                AbuseEvent.cluster_key == finding.cluster_key,
                AbuseEvent.created_at >= now - DEDUPE_WINDOW,
            ).limit(1)
        )
        return result.first() is not None

    async def _key_exempt(self, api_key_id: int) -> str | None:
        key = (await self._session.execute(select(APIKey).where(APIKey.id == api_key_id))).scalar_one_or_none()
        if key is None:
            return "missing"
        if not key.is_active:
            return "inactive"
        if key.rate_limit_tier in KEY_FANOUT_EXEMPT_TIERS:
            return f"tier:{key.rate_limit_tier}"
        if (key.tags or {}).get(KEY_FANOUT_EXEMPT_TAG) is True:
            return f"tag:{KEY_FANOUT_EXEMPT_TAG}"
        return None

    async def _revoke_key(self, finding: Finding) -> None:
        """Revoke the key (not the account), purge it from the auth cache on
        every instance, and tell the owner through the backend."""
        from mlpal_assistants_service.services.api_key import AUTH_CACHE_PREFIX
        from mlpal_assistants_service.services.notify import KIND_KEY_REVOKED, notify_owner

        key_id = finding.evidence["api_key_id"]
        key = (await self._session.execute(select(APIKey).where(APIKey.id == key_id))).scalar_one_or_none()
        if key is None:
            return
        key.is_active = False
        key.revoked_at = datetime.now(UTC)
        key.tags = {
            **(key.tags or {}),
            "auto_revoked": {
                "rule": finding.rule,
                "at": key.revoked_at.isoformat(),
                "distinct_ips": finding.evidence["distinct_ips"],
                "distinct_nets": finding.evidence["distinct_nets"],
            },
        }
        await self._session.flush()
        if self._redis and key.key_hash:
            try:
                await self._redis.delete(f"{AUTH_CACHE_PREFIX}{key.key_hash}")
            except Exception as e:  # noqa: BLE001 — cache purge must not fail the action
                logger.warning("fan-out: auth cache purge failed: %s", e)
        finding.evidence["notified"] = await notify_owner(
            KIND_KEY_REVOKED,
            {
                "user_id": key.user_id,
                "key_id": key.id,
                "key_prefix": key.key_prefix,
                "key_name": key.name,
                "reason": "used from an unusual number of networks in the last hour",
                "observed": {k: finding.evidence[k] for k in ("distinct_ips", "distinct_nets", "sample_ips", "models", "window_minutes")},
            },
        )

    async def _young_accounts(self, now: datetime) -> set[int]:
        cutoff = now - YOUNG_ACCOUNT_AGE
        result = await self._session.execute(
            select(APIKey.user_id).group_by(APIKey.user_id).having(func.min(APIKey.created_at) >= cutoff)
        )
        return {row[0] for row in result.all()}

    async def _signals(self, young: set[int], now: datetime) -> tuple[list[UsageRow], list[KeyRow]]:
        ids = list(young)
        usage_q = await self._session.execute(
            select(UsageLog.user_id, UsageLog.created_at, UsageLog.client_ip, UsageLog.client_ua_hash)
            .where(UsageLog.user_id.in_(ids), UsageLog.created_at >= now - USAGE_WINDOW)
        )
        keys_q = await self._session.execute(
            select(APIKey.user_id, APIKey.created_at, APIKey.created_ip, APIKey.created_ua_hash)
            .where(APIKey.user_id.in_(ids), APIKey.created_at >= now - KEY_WINDOW)
        )
        usage = [UsageRow(r[0], r[1], r[2], r[3]) for r in usage_q.all()]
        keys = [KeyRow(r[0], r[1], r[2], r[3]) for r in keys_q.all()]
        return usage, keys

    async def _known_users(self, finding: Finding, now: datetime) -> set[int]:
        result = await self._session.execute(
            select(AbuseEvent.evidence).where(
                AbuseEvent.rule == finding.rule,
                AbuseEvent.cluster_key == finding.cluster_key,
                AbuseEvent.created_at >= now - DEDUPE_WINDOW,
            )
        )
        known: set[int] = set()
        for (evidence,) in result.all():
            known.update(int(u) for u in (evidence or {}).get("user_ids", []))
        return known

    async def _suspend(self, finding: Finding, user_ids: list[int]) -> None:
        from mlpal_assistants_service.services.api_key import APIKeyService

        svc = APIKeyService(self._session, self._redis)
        already = await self._session.execute(
            select(UserSuspension.user_id).where(
                UserSuspension.user_id.in_(user_ids), UserSuspension.lifted_at.is_(None)
            )
        )
        skip = {row[0] for row in already.all()}
        incident = f"auto:{finding.rule}:{finding.cluster_key}"
        from mlpal_assistants_service.services.notify import KIND_ACCOUNT_SUSPENDED, notify_owner

        notified: list[int] = []
        for uid in user_ids:
            if uid in skip:
                continue
            await svc.suspend_user(uid, AUTO_SUSPEND_REASON, by="abuse-detector", incident=incident)
            # Backend emails the owner (reply path for false positives).
            if await notify_owner(KIND_ACCOUNT_SUSPENDED, {"user_id": uid, "reason": AUTO_SUSPEND_REASON}):
                notified.append(uid)
        finding.evidence["suspended"] = [u for u in user_ids if u not in skip]
        finding.evidence["notified"] = notified
        finding.evidence["already_suspended"] = sorted(skip)


async def publish_finding_metrics(findings: list[Finding]) -> None:
    """Direct CloudWatch PutMetricData for the alarm path.

    The EMF lines the metrics emitter prints are only metrics once a log
    agent ships them; the gateway pods have none (2026-09-29), so an alarm on
    the EMF stream would never fire. One un-dimensioned datapoint for the
    alarm plus one per (rule, action) for the dashboard. Never raises."""
    settings = get_settings()
    if settings.environment == "local" or not settings.metrics_enabled:
        return
    try:
        import aioboto3

        data: list[dict[str, Any]] = [
            {"MetricName": "AbuseClusterDetected", "Value": float(len(findings)), "Unit": "Count"},
        ]
        per_rule: dict[tuple[str, str], int] = defaultdict(int)
        for f in findings:
            per_rule[(f.rule, f.action)] += 1
        for (rule, action), n in per_rule.items():
            data.append({
                "MetricName": "AbuseClusterDetected", "Value": float(n), "Unit": "Count",
                "Dimensions": [{"Name": "rule", "Value": rule}, {"Name": "action", "Value": action}],
            })
        async with aioboto3.Session().client("cloudwatch", region_name=settings.aws_region) as cw:
            await cw.put_metric_data(Namespace=settings.metrics_namespace, MetricData=data)
    except Exception as e:  # noqa: BLE001 — alerting must never break detection
        logger.warning("abuse metric publish failed: %s", e)


async def scrub_client_ips(session: AsyncSession, *, older_than: timedelta, limit: int = 5000) -> int:
    """Null out client_ip on usage rows past retention. Bounded per call so
    the daily pass never holds a long lock on the busiest table."""
    cutoff = datetime.now(UTC) - older_than
    ids = await session.execute(
        select(UsageLog.id)
        .where(UsageLog.created_at < cutoff, UsageLog.client_ip.is_not(None))
        .limit(limit)
    )
    id_list = [row[0] for row in ids.all()]
    if not id_list:
        return 0
    await session.execute(update(UsageLog).where(UsageLog.id.in_(id_list)).values(client_ip=None))
    await session.commit()
    return len(id_list)
