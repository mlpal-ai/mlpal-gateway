"""Abuse detector: rules replayed on the 2026-09-28 signup-farm shapes, plus
the run_once contract (dedupe, observe vs enforce, key-mint never suspends)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlpal_assistants_service.services.abuse_detector import (
    AUTO_SUSPEND_REASON,
    RULE_KEY_MINT_BURST,
    RULE_KEY_MINT_BURST_UA,
    RULE_LOCKSTEP_IP,
    RULE_LOCKSTEP_UA,
    AbuseDetector,
    Finding,
    KeyRow,
    UsageRow,
    find_clusters,
    find_key_mint_burst,
    find_lockstep_ip,
    find_lockstep_ua,
    publish_finding_metrics,
    scrub_client_ips,
)

T0 = datetime(2026, 9, 29, 18, 0, tzinfo=UTC)


def _usage(user_id: int, minute: int, ip: str | None = "47.242.138.158", ua: str | None = "bridge22") -> UsageRow:
    return UsageRow(user_id, T0 + timedelta(minutes=minute), ip, ua)


# ---- rules ------------------------------------------------------------------


def test_lockstep_ip_fires_on_wave_two_shape() -> None:
    # 22 accounts, ~10 calls each, one Alibaba IP, one bridge UA.
    usage = [_usage(uid, m) for uid in range(509, 531) for m in range(0, 50, 5)]
    findings = find_lockstep_ip(usage)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule == RULE_LOCKSTEP_IP and f.cluster_key == "47.242.138.158"
    assert f.user_ids == list(range(509, 531))
    assert f.evidence["calls_per_user"]["509"] == 10


def test_lockstep_ip_ignores_small_groups_and_missing_ip() -> None:
    usage = [_usage(uid, 0) for uid in range(1, 5)]  # four users: below threshold
    usage += [_usage(uid, 0, ip=None) for uid in range(10, 30)]  # no IP recorded
    assert find_lockstep_ip(usage) == []


def test_lockstep_ua_needs_shared_minute_buckets() -> None:
    # Same UA, five users, but each calls in its own minute: no lockstep.
    spread = [_usage(uid, uid, ip=None) for uid in range(1, 6)]
    assert find_lockstep_ua(spread) == []
    # Same five users landing in the same three minutes: lockstep.
    sync = [_usage(uid, m, ip=None) for uid in range(1, 6) for m in (0, 1, 2)]
    findings = find_lockstep_ua(sync)
    assert len(findings) == 1
    assert findings[0].rule == RULE_LOCKSTEP_UA
    assert findings[0].user_ids == [1, 2, 3, 4, 5]
    assert len(findings[0].evidence["shared_minutes"]) == 3


def test_lockstep_ua_ignores_two_users_sharing_minutes() -> None:
    # Two heavy users on one UA in the same minutes must not form a cluster.
    sync = [_usage(uid, m, ip=None) for uid in (1, 2) for m in range(0, 10)]
    assert find_lockstep_ua(sync) == []


def test_key_mint_burst_fires_on_console_operator_shape() -> None:
    # Keys "1,2,3,5,7" minted for five accounts in 2.5 minutes from one IP.
    keys = [KeyRow(uid, T0 + timedelta(seconds=s), "38.83.41.46") for uid, s in ((300, 0), (292, 25), (291, 44), (284, 130), (282, 156))]
    keys.append(KeyRow(303, T0 + timedelta(hours=1), "38.83.41.46"))  # later, outside the window
    findings = find_key_mint_burst(keys)
    assert len(findings) == 1
    assert findings[0].rule == RULE_KEY_MINT_BURST
    assert findings[0].user_ids == [282, 284, 291, 292, 300]
    assert findings[0].evidence["keys_in_group_24h"] == 6


def test_key_mint_burst_ignores_slow_minting() -> None:
    keys = [KeyRow(uid, T0 + timedelta(minutes=15 * i), "1.2.3.4") for i, uid in enumerate(range(1, 6))]
    assert find_key_mint_burst(keys) == []


def test_find_clusters_combines_rules() -> None:
    usage = [_usage(uid, m) for uid in range(1, 6) for m in (0, 1, 2)]
    keys = [KeyRow(uid, T0 + timedelta(seconds=10 * uid), "9.9.9.9") for uid in range(1, 4)]
    rules = {f.rule for f in find_clusters(usage, keys)}
    assert rules == {RULE_LOCKSTEP_IP, RULE_LOCKSTEP_UA, RULE_KEY_MINT_BURST}


# ---- run_once orchestration (rules mocked; the SQL is exercised live) --------


def _detector(enforce: bool = False) -> AbuseDetector:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    d = AbuseDetector(session, None, enforce=enforce)
    d._young_accounts = AsyncMock(return_value={509, 510, 511, 512, 513})
    d._signals = AsyncMock(return_value=([], []))
    d._known_users = AsyncMock(return_value=set())
    d._suspend = AsyncMock()
    d._fanout_pass = AsyncMock(return_value=[])
    return d


@pytest.fixture(autouse=True)
def _no_cloudwatch():
    with patch("mlpal_assistants_service.services.abuse_detector.publish_finding_metrics", new=AsyncMock()) as p:
        yield p


def _findings() -> list[Finding]:
    return [
        Finding(RULE_LOCKSTEP_IP, "47.242.138.158", [509, 510, 511, 512, 513]),
        Finding(RULE_KEY_MINT_BURST, "38.83.41.46", [282, 284, 291]),
    ]


@pytest.mark.asyncio
async def test_run_once_observe_records_events_and_never_suspends() -> None:
    d = _detector(enforce=False)
    with (
        patch("mlpal_assistants_service.services.abuse_detector.find_clusters", return_value=_findings()),
        patch("mlpal_assistants_service.services.abuse_detector.get_metrics") as gm,
    ):
        gm.return_value.put_metric = AsyncMock()
        out = await d.run_once(now=T0)
    assert [f.action for f in out] == ["observe", "observe"]
    d._suspend.assert_not_awaited()
    added = [c.args[0] for c in d._session.add.call_args_list]
    assert [(e.rule, e.cluster_key, e.action) for e in added] == [
        (RULE_LOCKSTEP_IP, "47.242.138.158", "observe"),
        (RULE_KEY_MINT_BURST, "38.83.41.46", "observe"),
    ]
    assert added[0].evidence["user_ids"] == [509, 510, 511, 512, 513]
    assert added[0].evidence["cluster_size"] == 5
    d._session.commit.assert_awaited_once()
    gm.return_value.put_metric.assert_any_await(
        "AbuseClusterDetected", 5, "Count", {"rule": RULE_LOCKSTEP_IP, "action": "observe"}
    )


@pytest.mark.asyncio
async def test_run_once_enforce_suspends_lockstep_but_never_key_mint() -> None:
    d = _detector(enforce=True)
    with (
        patch("mlpal_assistants_service.services.abuse_detector.find_clusters", return_value=_findings()),
        patch("mlpal_assistants_service.services.abuse_detector.get_metrics") as gm,
    ):
        gm.return_value.put_metric = AsyncMock()
        out = await d.run_once(now=T0)
    assert {f.rule: f.action for f in out} == {RULE_LOCKSTEP_IP: "suspend", RULE_KEY_MINT_BURST: "observe"}
    d._suspend.assert_awaited_once()
    finding, users = d._suspend.await_args.args
    assert finding.rule == RULE_LOCKSTEP_IP and users == [509, 510, 511, 512, 513]


@pytest.mark.asyncio
async def test_run_once_reports_only_newcomers_of_a_known_cluster() -> None:
    d = _detector(enforce=True)
    d._known_users = AsyncMock(return_value={509, 510, 511, 512})
    with (
        patch("mlpal_assistants_service.services.abuse_detector.find_clusters", return_value=_findings()[:1]),
        patch("mlpal_assistants_service.services.abuse_detector.get_metrics") as gm,
    ):
        gm.return_value.put_metric = AsyncMock()
        out = await d.run_once(now=T0)
    assert len(out) == 1 and out[0].evidence["user_ids"] == [513]
    assert d._suspend.await_args.args[1] == [513]
    d._known_users = AsyncMock(return_value={509, 510, 511, 512, 513})
    with (
        patch("mlpal_assistants_service.services.abuse_detector.find_clusters", return_value=_findings()[:1]),
        patch("mlpal_assistants_service.services.abuse_detector.get_metrics"),
    ):
        assert await d.run_once(now=T0) == []


@pytest.mark.asyncio
async def test_run_once_is_a_no_op_without_young_accounts() -> None:
    d = _detector()
    d._young_accounts = AsyncMock(return_value=set())
    assert await d.run_once(now=T0) == []
    d._signals.assert_not_awaited()
    d._session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_suspend_skips_already_suspended_and_tags_incident() -> None:
    session = MagicMock()
    already = MagicMock()
    already.all.return_value = [(510,)]
    session.execute = AsyncMock(return_value=already)
    d = AbuseDetector(session, None, enforce=True)
    finding = Finding(RULE_LOCKSTEP_IP, "47.242.138.158", [509, 510, 511])
    with (
        patch("mlpal_assistants_service.services.api_key.APIKeyService.suspend_user", new=AsyncMock()) as su,
        patch("mlpal_assistants_service.services.notify.notify_owner", new=AsyncMock(return_value=True)) as no,
    ):
        await d._suspend(finding, [509, 510, 511])
    assert [c.args[0] for c in su.await_args_list] == [509, 511]
    assert all(c.kwargs == {"by": "abuse-detector", "incident": f"auto:{RULE_LOCKSTEP_IP}:47.242.138.158"} for c in su.await_args_list)
    assert finding.evidence == {"suspended": [509, 511], "already_suspended": [510], "notified": [509, 511]}
    assert [c.args for c in no.await_args_list] == [
        ("account-suspended", {"user_id": 509, "reason": AUTO_SUSPEND_REASON}),
        ("account-suspended", {"user_id": 511, "reason": AUTO_SUSPEND_REASON}),
    ]


@pytest.mark.asyncio
async def test_scrub_client_ips_is_bounded_and_stops_when_clean() -> None:
    session = MagicMock()
    ids = MagicMock()
    ids.all.return_value = [(1,), (2,)]
    empty = MagicMock()
    empty.all.return_value = []
    session.execute = AsyncMock(side_effect=[ids, None, empty])
    session.commit = AsyncMock()
    assert await scrub_client_ips(session, older_than=timedelta(days=14), limit=2) == 2
    session.commit.assert_awaited_once()
    assert await scrub_client_ips(session, older_than=timedelta(days=14), limit=2) == 0
    session.commit.assert_awaited_once()  # nothing to scrub: no second commit


@pytest.mark.asyncio
async def test_publish_finding_metrics_puts_total_and_per_rule_datapoints(_no_cloudwatch) -> None:
    cw = MagicMock()
    cw.put_metric_data = AsyncMock()
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=cw)
    client_cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.client = MagicMock(return_value=client_cm)
    findings = _findings() + [Finding(RULE_LOCKSTEP_IP, "9.9.9.9", [1, 2, 3, 4, 5], action="suspend")]
    settings = MagicMock(environment="production", metrics_enabled=True, aws_region="us-east-2", metrics_namespace="MLPal/Assistants")
    with (
        patch("mlpal_assistants_service.services.abuse_detector.get_settings", return_value=settings),
        patch.dict("sys.modules", {"aioboto3": MagicMock(Session=MagicMock(return_value=session))}),
    ):
        await publish_finding_metrics(findings)
    kwargs = cw.put_metric_data.await_args.kwargs
    assert kwargs["Namespace"] == "MLPal/Assistants"
    total = [d for d in kwargs["MetricData"] if "Dimensions" not in d]
    assert total == [{"MetricName": "AbuseClusterDetected", "Value": 3.0, "Unit": "Count"}]
    dims = {(d["Dimensions"][0]["Value"], d["Dimensions"][1]["Value"]): d["Value"] for d in kwargs["MetricData"] if "Dimensions" in d}
    assert dims == {(RULE_LOCKSTEP_IP, "observe"): 1.0, (RULE_KEY_MINT_BURST, "observe"): 1.0, (RULE_LOCKSTEP_IP, "suspend"): 1.0}


@pytest.mark.asyncio
async def test_publish_finding_metrics_is_silent_locally_and_on_failure() -> None:
    local = MagicMock(environment="local", metrics_enabled=True)
    with patch("mlpal_assistants_service.services.abuse_detector.get_settings", return_value=local), patch.dict("sys.modules", {"aioboto3": None}):
        await publish_finding_metrics(_findings())  # local: never touches boto
    prod = MagicMock(environment="production", metrics_enabled=True, aws_region="us-east-2", metrics_namespace="X")
    boom = MagicMock(Session=MagicMock(side_effect=RuntimeError("no creds")))
    with patch("mlpal_assistants_service.services.abuse_detector.get_settings", return_value=prod), patch.dict("sys.modules", {"aioboto3": boom}):
        await publish_finding_metrics(_findings())  # failure is logged, not raised


def test_key_mint_burst_ua_fires_on_rotating_ip_wave() -> None:
    # 2026-09-30 wave: 17 keys in ~11 min, each from a different IP, one client UA hash.
    keys = [KeyRow(600 + i, T0 + timedelta(seconds=35 * i), f"10.0.{i}.1", "ca9b2eafc1cd8634") for i in range(17)]
    findings = find_key_mint_burst(keys)
    assert [f.rule for f in findings] == [RULE_KEY_MINT_BURST_UA]  # no IP repeats, so the IP rule stays silent
    assert findings[0].cluster_key == "ca9b2eafc1cd8634"
    assert len(findings[0].user_ids) == 17


def test_key_mint_burst_ua_tolerates_a_popular_browser() -> None:
    # Seven distinct users sharing a common browser hash within ten minutes is not a finding.
    keys = [KeyRow(700 + i, T0 + timedelta(seconds=60 * i), f"10.1.{i}.1", "chrome-common") for i in range(7)]
    assert find_key_mint_burst(keys) == []


def test_key_mint_burst_ua_never_suspends_in_enforce() -> None:
    d = _detector(enforce=True)
    f = Finding(RULE_KEY_MINT_BURST_UA, "ca9b2eafc1cd8634", list(range(600, 617)))
    with (
        patch("mlpal_assistants_service.services.abuse_detector.find_clusters", return_value=[f]),
        patch("mlpal_assistants_service.services.abuse_detector.get_metrics") as gm,
    ):
        gm.return_value.put_metric = AsyncMock()
        out = await_(d.run_once(now=T0))
    assert out[0].action == "observe"
    d._suspend.assert_not_awaited()


def await_(coro):
    import asyncio

    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)
