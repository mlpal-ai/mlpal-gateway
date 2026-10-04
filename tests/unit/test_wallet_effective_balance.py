"""Admission gate: effective balance = wallet snapshot − CU billed since.

Why (incident 2026-09-28): the gate only saw the payments balance, refreshed
on a TTL and settled by the backend debit worker in 60 s batches. Between
those, a wallet with 5 CU could serve 13–20 CU of long-context requests. Now
every billed success row bumps a per-user counter and the gate subtracts the
delta since the snapshot, so spend between settlements is bounded by what is
in flight, not by the refresh cadence.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.repositories.billing_repository import (
    WALLET_EMPTY_MESSAGE,
    WALLET_LOW_INFLIGHT_MESSAGE,
    BillingRepository,
)
from mlpal_assistants_service.services.usage import UsageService, wallet_spent_key


class _Redis:
    """The subset of redis.asyncio the gate and the recorder use."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}

    async def get(self, k):
        return self.kv.get(k)

    async def setex(self, k, ttl, v):
        self.kv[k] = v

    async def incrbyfloat(self, k, v):
        self.kv[k] = str(float(self.kv.get(k, "0")) + v)
        return float(self.kv[k])

    async def expire(self, k, ttl):
        return True

    async def delete(self, k):
        return int(self.kv.pop(k, None) is not None)

    async def incr(self, k):
        self.kv[k] = str(int(self.kv.get(k, "0")) + 1)
        return int(self.kv[k])

    async def decr(self, k):
        self.kv[k] = str(int(self.kv.get(k, "0")) - 1)
        return int(self.kv[k])


def _repo(redis: _Redis, balance: Decimal) -> BillingRepository:
    repo = BillingRepository(MagicMock(), redis)
    repo._get_wallet_rollout_config = AsyncMock(return_value={"wallet_gating_enabled": True})
    repo._fetch_wallet_balance = AsyncMock(return_value=balance)
    return repo


async def _bill(redis: _Redis, user_id: int, cu: str) -> None:
    svc = UsageService(MagicMock(), redis, None)
    svc._write_usage_to_db = AsyncMock()
    svc._accrue_platform_fee = AsyncMock()
    await svc.record_usage(
        user_id=str(user_id), api_key_id="1", trace_id="t", model_tag="m", provider="p",
        operation="chat", input_tokens=1, output_tokens=1, compute_units=Decimal(cu),
        status="success", wallet_debit_status="pending",
    )


@pytest.mark.asyncio
async def test_spend_since_snapshot_closes_the_gate_before_settlement():
    redis = _Redis()
    repo = _repo(redis, Decimal("5"))
    assert (await repo.can_make_request_cached(7))[0] is True  # snapshot taken, baseline 0
    await _bill(redis, 7, "3")
    assert (await repo.can_make_request_cached(7))[0] is True  # 5 - 3 > 0
    await _bill(redis, 7, "2")
    allowed, reason, _ = await repo.can_make_request_cached(7)
    assert allowed is False and reason == WALLET_EMPTY_MESSAGE  # 5 - 5 = 0, no refetch needed
    repo._fetch_wallet_balance.assert_awaited_once()  # the cached snapshot decided it


@pytest.mark.asyncio
async def test_snapshot_refresh_resets_the_baseline_to_the_settled_balance():
    redis = _Redis()
    repo = _repo(redis, Decimal("5"))
    await _bill(redis, 7, "4")  # billed before any snapshot existed
    assert (await repo.can_make_request_cached(7))[0] is True  # baseline = 4, nothing since
    await _bill(redis, 7, "4.5")
    assert (await repo.can_make_request_cached(7))[0] is True  # 5 - 4.5 > 0
    await _bill(redis, 7, "0.5")
    assert (await repo.can_make_request_cached(7))[0] is False  # 5 - 5
    # Backend settled and invalidated; the refetched balance carries the new baseline.
    await redis.delete("wallet:7")
    repo._fetch_wallet_balance = AsyncMock(return_value=Decimal("0.3"))
    assert (await repo.can_make_request_cached(7))[0] is True
    await _bill(redis, 7, "0.3")
    assert (await repo.can_make_request_cached(7))[0] is False


@pytest.mark.asyncio
async def test_counter_restart_never_goes_negative():
    redis = _Redis()
    repo = _repo(redis, Decimal("2"))
    await _bill(redis, 7, "10")
    assert (await repo.can_make_request_cached(7))[0] is True  # baseline 10
    await redis.delete(wallet_spent_key(7))  # expiry / flush: counter restarts below baseline
    assert await repo.effective_balance_cu(7, await repo._get_wallet_gate_snapshot(7)) == Decimal("2")
    assert (await repo.can_make_request_cached(7))[0] is True


@pytest.mark.asyncio
async def test_error_and_connection_served_rows_do_not_count():
    redis = _Redis()
    svc = UsageService(MagicMock(), redis, None)
    svc._write_usage_to_db = AsyncMock()
    svc._accrue_platform_fee = AsyncMock()
    common = {"user_id": "7", "api_key_id": "1", "trace_id": "t", "model_tag": "m",
              "provider": "p", "operation": "chat", "input_tokens": 1, "output_tokens": 1}
    await svc.record_usage(**common, compute_units=Decimal("0"), status="error", error_code="x")
    await svc.record_usage(**common, compute_units=Decimal("0"), status="success",
                           wallet_debit_status="not_applicable")  # connection-served: CU is zero
    assert wallet_spent_key(7) not in redis.kv


@pytest.mark.asyncio
async def test_gate_without_redis_falls_back_to_the_snapshot_balance():
    repo = _repo(None, Decimal("1"))  # type: ignore[arg-type]
    repo._redis = None
    assert (await repo.can_make_request_cached(7))[0] is True


# -- low-balance in-flight cap ---------------------------------------------------
@pytest.mark.asyncio
async def test_inflight_cap_applies_only_below_the_threshold(monkeypatch):
    redis = _Redis()
    repo = _repo(redis, Decimal("5"))
    assert (await repo.can_make_request_cached(7))[0] is True
    assert await repo.reserve_inflight(7) == (None, None)  # 5 CU effective: no cap
    await _bill(redis, 7, "4.5")  # effective 0.5 < 1.0 threshold
    s1, b1 = await repo.reserve_inflight(7)
    s2, b2 = await repo.reserve_inflight(7)
    s3, b3 = await repo.reserve_inflight(7)
    assert s1 == s2 == "wallet:inflight:7" and b1 is None and b2 is None
    assert s3 is None and b3 == WALLET_LOW_INFLIGHT_MESSAGE
    assert redis.kv["wallet:inflight:7"] == "2"  # the refused one did not leak
    await repo.release_inflight(s1)
    assert (await repo.reserve_inflight(7))[0] == "wallet:inflight:7"


@pytest.mark.asyncio
async def test_inflight_release_never_leaves_a_negative_count():
    redis = _Redis()
    repo = _repo(redis, Decimal("5"))
    await repo.release_inflight("wallet:inflight:7")  # slot expired before release
    assert "wallet:inflight:7" not in redis.kv
    await repo.release_inflight(None)


@pytest.mark.asyncio
async def test_inflight_cap_skipped_when_gating_off_or_no_redis():
    redis = _Redis()
    repo = _repo(redis, Decimal("0.1"))
    repo._get_wallet_rollout_config = AsyncMock(return_value={"wallet_gating_enabled": False})
    assert await repo.reserve_inflight(7) == (None, None)
    repo = _repo(None, Decimal("0.1"))  # type: ignore[arg-type]
    repo._redis = None
    assert await repo.reserve_inflight(7) == (None, None)


# -- fail-open when the backend / payments are unreachable --------------------------
@pytest.mark.asyncio
async def test_unreachable_platform_config_fails_open_and_is_loud(monkeypatch):
    import httpx

    redis = _Redis()
    repo = BillingRepository(MagicMock(), redis)
    repo._settings = MagicMock(wallet_fail_open=True, backend_base_url="http://down", wallet_timeout_seconds=0.01,
                               wallet_cache_ttl_seconds=60, wallet_low_balance_cache_ttl_seconds=10,
                               wallet_low_balance_inflight_threshold_cu=Decimal("1"), wallet_low_balance_inflight_cap=2)
    monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectTimeout("down")))
    noted = []
    repo._note_degraded = lambda what, exc, user_id=None: noted.append(what)
    repo.get_by_user_id = AsyncMock(return_value=None)  # gate off → billing-status path
    config = await repo._get_wallet_rollout_config()
    assert config == {"wallet_gating_enabled": False, "degraded": True} and noted == ["platform_config"]
    assert (await repo.can_make_request_cached(7))[0] is True
    assert "wallet:config" in redis.kv  # reused for the degraded window, not re-fetched per request


@pytest.mark.asyncio
async def test_unreachable_payments_balance_fails_open_per_policy():
    redis = _Redis()
    repo = _repo(redis, Decimal("0"))
    repo._fetch_wallet_balance = AsyncMock(side_effect=RuntimeError("payments down"))
    repo.get_by_user_id = AsyncMock(return_value=None)  # balance unknown → billing-status path
    noted = []
    repo._note_degraded = lambda what, exc, user_id=None: noted.append(what)
    assert (await repo.can_make_request_cached(7))[0] is True
    assert noted == ["wallet_balance"] and await repo.reserve_inflight(7) == (None, None)

    repo2 = _repo(_Redis(), Decimal("0"))
    repo2._fetch_wallet_balance = AsyncMock(side_effect=RuntimeError("payments down"))
    repo2._settings = MagicMock(wallet_fail_open=False, wallet_cache_ttl_seconds=60,
                                wallet_low_balance_cache_ttl_seconds=10)
    with pytest.raises(RuntimeError):
        await repo2.can_make_request_cached(7)


# -- spend-rate metrics -------------------------------------------------------------
@pytest.mark.asyncio
async def test_billed_rows_emit_compute_units_metric(monkeypatch):
    from mlpal_assistants_service.services import usage as usage_mod

    m = MagicMock()
    monkeypatch.setattr("mlpal_assistants_service.core.metrics.get_metrics", lambda: m)
    redis = _Redis()
    await _bill(redis, 7, "0.25")
    calls = [c.args for c in m.put_metric_sync.call_args_list if c.args[0] == "ComputeUnits"]
    assert calls[0][1] == 0.25 and len(calls) == 2  # platform-wide + per provider/model
    assert calls[1][3] == {"provider": "p", "model": "m"}
    m.reset_mock()
    svc = usage_mod.UsageService(MagicMock(), redis, None)
    svc._write_usage_to_db = AsyncMock()
    await svc.record_usage(user_id="7", api_key_id="1", trace_id="t", model_tag="m", provider="p",
                           operation="chat", input_tokens=0, output_tokens=0, compute_units=Decimal("0"),
                           status="error", error_code="x")
    assert not [c for c in m.put_metric_sync.call_args_list if c.args[0] == "ComputeUnits"]
