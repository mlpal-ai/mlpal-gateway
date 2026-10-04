"""Key fan-out rule (incident B, 2026-09-30: one stolen gateway key used from
110 IPs within hours). One key, many networks → revoke the key, tell the owner.
Never the account; enterprise-tier or fanout_ok-tagged keys are reported only."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlpal_assistants_service.services import abuse_detector as ad
from mlpal_assistants_service.services.abuse_detector import (
    RULE_KEY_FANOUT,
    AbuseDetector,
    Finding,
    KeyUsageRow,
    find_key_fanout,
)

T0 = datetime(2026, 9, 30, 10, 30, tzinfo=UTC)


def _rows(key_id: int, ips: list[str], user_id: int = 115, model: str = "claude-opus-4-6"):
    return [KeyUsageRow(key_id, user_id, ip, model) for ip in ips]


def test_resold_key_shape_fires():
    # 30 IPs across 15 /16s — the incident-B shape (110 IPs, dozens of networks)
    ips = [f"{a}.{b}.1.{c}" for a, b, c in [(10 + i, 20 + i, 7) for i in range(15)]] + [
        f"{a}.{b}.2.{c}" for a, b, c in [(10 + i, 20 + i, 9) for i in range(15)]
    ]
    out = find_key_fanout(_rows(39, ips), min_ips=25, min_nets=12)
    assert len(out) == 1
    f = out[0]
    assert f.rule == RULE_KEY_FANOUT and f.cluster_key == "key:39" and f.user_ids == [115]
    assert f.evidence["distinct_ips"] == 30 and f.evidence["distinct_nets"] == 15
    assert f.evidence["models"] == {"claude-opus-4-6": 30} and len(f.evidence["sample_ips"]) == 5


def test_nat_fleet_and_travelling_laptop_do_not_fire():
    one_network = [f"10.20.{i}.{j}" for i in range(6) for j in range(6)]  # 36 IPs, one /16
    assert find_key_fanout(_rows(1, one_network), min_ips=25, min_nets=12) == []
    few_ips = [f"{10 + i}.0.0.1" for i in range(12)]  # 12 networks but only 12 IPs
    assert find_key_fanout(_rows(1, few_ips), min_ips=25, min_nets=12) == []
    assert find_key_fanout(_rows(1, [None] * 50), min_ips=25, min_nets=12) == []  # no IP signal


def test_ipv6_counts_by_32():
    ips = [f"2001:{470 + i:x}:da82::{j}" for i in range(13) for j in range(2)]
    out = find_key_fanout(_rows(2, ips), min_ips=25, min_nets=12)
    assert len(out) == 1 and out[0].evidence["distinct_nets"] == 13


# ---- orchestration -----------------------------------------------------------
def _finding(key_id: int = 39) -> Finding:
    return Finding(RULE_KEY_FANOUT, f"key:{key_id}", [115], evidence={
        "api_key_id": key_id, "distinct_ips": 110, "distinct_nets": 40,
        "sample_ips": ["1.2.3.4"], "models": {"claude-opus-4-6": 90}, "window_minutes": 60,
    })


def _detector(enforce_key_fanout: bool = True) -> AbuseDetector:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    d = AbuseDetector(session, None, enforce=False, enforce_key_fanout=enforce_key_fanout)
    d._key_usage = AsyncMock(return_value=[])
    d._known_key = AsyncMock(return_value=False)
    d._key_exempt = AsyncMock(return_value=None)
    d._revoke_key = AsyncMock()
    d._young_accounts = AsyncMock(return_value=set())
    return d


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(ad, "publish_finding_metrics", AsyncMock())
    gm = MagicMock()
    gm.return_value.put_metric = AsyncMock()
    monkeypatch.setattr(ad, "get_metrics", gm)


@pytest.mark.asyncio
async def test_fanout_runs_without_young_accounts_and_revokes_the_key():
    d = _detector()
    with patch.object(ad, "find_key_fanout", return_value=[_finding()]):
        out = await d.run_once(now=T0)
    assert [f.action for f in out] == ["revoke_key"]
    d._revoke_key.assert_awaited_once()
    d._session.add.assert_called_once()
    d._session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_fanout_observe_only_when_exempt_or_enforce_off():
    d = _detector()
    d._key_exempt = AsyncMock(return_value="tier:enterprise")
    with patch.object(ad, "find_key_fanout", return_value=[_finding()]):
        out = await d.run_once(now=T0)
    assert out[0].action == "observe" and out[0].evidence["exempt"] == "tier:enterprise"
    d._revoke_key.assert_not_awaited()

    d2 = _detector(enforce_key_fanout=False)
    with patch.object(ad, "find_key_fanout", return_value=[_finding()]):
        out = await d2.run_once(now=T0)
    assert out[0].action == "observe"
    d2._revoke_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_fanout_dedupes_a_known_key_for_a_day():
    d = _detector()
    d._known_key = AsyncMock(return_value=True)
    with patch.object(ad, "find_key_fanout", return_value=[_finding()]):
        assert await d.run_once(now=T0) == []
    d._revoke_key.assert_not_awaited()
    d._session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_fanout_query_failure_never_stops_the_cluster_pass():
    d = _detector()
    d._key_usage = AsyncMock(side_effect=RuntimeError("db"))
    assert await d.run_once(now=T0) == []


@pytest.mark.asyncio
async def test_revoke_key_marks_purges_and_notifies():
    key = SimpleNamespace(id=39, user_id=115, key_hash="h", key_prefix="mlpal_sk_abc", name="da-harness",
                          is_active=True, revoked_at=None, tags={"source": "x"})
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=key)))
    session.flush = AsyncMock()
    redis = MagicMock()
    redis.delete = AsyncMock()
    d = AbuseDetector(session, redis, enforce=False, enforce_key_fanout=True)
    f = _finding()
    with patch("mlpal_assistants_service.services.notify.notify_owner", new=AsyncMock(return_value=True)) as n:
        await d._revoke_key(f)
    assert key.is_active is False and key.revoked_at is not None
    assert key.tags["source"] == "x" and key.tags["auto_revoked"]["distinct_ips"] == 110
    redis.delete.assert_awaited_once_with("auth:h")
    payload = n.await_args.args[1]
    assert n.await_args.args[0] == "key-revoked"
    assert payload["user_id"] == 115 and payload["key_id"] == 39 and payload["observed"]["distinct_nets"] == 40
    assert f.evidence["notified"] is True
