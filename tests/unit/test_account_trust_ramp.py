"""Young-account ramp, default key budgets, request-time hold.

Founder guidance (2026-10-01): controls must not hinder genuine users — the
ramp caps daily spend for unpaid accounts (2 CU for 48 h, then 10 CU) and
never restricts models; every key gets a daily budget by default; a
signup-risk hold refuses requests everywhere, not just key minting (account
625 was held and still drove the harness on 2026-09-30).
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.core.exceptions import AccountHeldError, BudgetExceededError
from mlpal_assistants_service.services import account_trust as at
from mlpal_assistants_service.services.account_trust import AccountTrust, AccountTrustService
from mlpal_assistants_service.services.api_key import APIKeyService, default_key_budgets
from mlpal_assistants_service.services.policy import PolicyService
from tests.unit.test_policy import FakeRedis

SETTINGS = SimpleNamespace(
    young_account_ramp_enabled=True,
    billing_backend="managed",
    young_account_first_window_hours=48,
    young_account_first_ceiling_cu=Decimal("2"),
    young_account_unpaid_ceiling_cu=Decimal("10"),
    paid_wallet_sources="stripe_checkout,auto_reload",
    account_trust_cache_ttl_seconds=300,
    user_schema="mlpal_test",
    cu_to_usd=10.0,
    budget_timezone="UTC",
)


class _Redis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.setex_calls = 0

    async def setex(self, k, ttl, v):
        self.setex_calls += 1
        await self.set(k, v)


# -- ceiling schedule ------------------------------------------------------------
@pytest.mark.parametrize(
    "age_hours,paid,expected",
    [(1, False, Decimal("2")), (47.9, False, Decimal("2")), (48, False, Decimal("10")),
     (500, False, Decimal("10")), (1, True, None), (500, True, None)],
)
def test_ceiling_schedule(age_hours, paid, expected):
    t = AccountTrust(user_id=1, age_hours=age_hours, paid=paid, held=False)
    assert t.daily_ceiling_cu(SETTINGS) == expected


@pytest.mark.asyncio
async def test_ramp_off_on_local_billing_or_flag():
    for s in (SimpleNamespace(**{**vars(SETTINGS), "billing_backend": "local"}),
              SimpleNamespace(**{**vars(SETTINGS), "young_account_ramp_enabled": False})):
        svc = AccountTrustService(MagicMock(), _Redis(), settings=s)
        assert svc.enabled is False
        assert await svc.daily_ceiling_cu(7) is None  # no DB touch


@pytest.mark.asyncio
async def test_resolve_reads_backend_row_and_caches(monkeypatch):
    monkeypatch.setattr(at, "_SCHEMA_PRESENT", True)
    from datetime import UTC, datetime, timedelta

    session = MagicMock()
    row = (datetime.now(UTC) - timedelta(hours=3), None, False)
    session.execute = AsyncMock(return_value=MagicMock(first=MagicMock(return_value=row)))
    redis = _Redis()
    svc = AccountTrustService(session, redis, settings=SETTINGS)
    assert await svc.daily_ceiling_cu(7) == Decimal("2")
    assert redis.setex_calls == 1
    session.execute.reset_mock()
    assert await svc.daily_ceiling_cu(7) == Decimal("2")  # served from the cache
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_failure_means_no_ceiling(monkeypatch):
    monkeypatch.setattr(at, "_SCHEMA_PRESENT", True)
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    svc = AccountTrustService(session, _Redis(), settings=SETTINGS)
    assert await svc.daily_ceiling_cu(7) is None


# -- enforcement through the policy engine ---------------------------------------
class _Repo:
    def __init__(self, key_cu="0", user_cu="0"):
        self.key_cu, self.user_cu = Decimal(key_cu), Decimal(user_cu)

    async def get_api_key_cu_in_window(self, api_key_id, start, end):
        return self.key_cu

    async def get_user_cu_in_window(self, user_id, start, end):
        return self.user_cu


class _Trust:
    def __init__(self, ceiling):
        self.ceiling, self.enabled = ceiling, True

    async def daily_ceiling_cu(self, user_id):
        return self.ceiling


def _policy(redis, repo, ceiling):
    return PolicyService(redis, repo, settings=SETTINGS, trust=_Trust(ceiling))


@pytest.mark.asyncio
async def test_account_ceiling_denies_from_reconciled_spend():
    redis = _Redis()
    p = _policy(redis, _Repo(user_cu="2.0"), Decimal("2"))
    with pytest.raises(BudgetExceededError) as ei:
        await p.check_budgets(11, None, user_id=7)
    assert ei.value.details["budget_id"] == "young-account-daily"
    assert str(ei.value).startswith("New-account daily limit reached")
    assert "payment method" in str(ei.value)


@pytest.mark.asyncio
async def test_account_ceiling_accrues_and_trips_without_a_key_budget():
    redis = _Redis()
    p = _policy(redis, _Repo(user_cu="1.5"), Decimal("2"))
    await p.check_budgets(11, None, user_id=7)  # 1.5 < 2: allowed, counter seeded
    await p.record_key_usage(11, None, Decimal("0.6"), user_id=7)  # 2.1
    with pytest.raises(BudgetExceededError):
        await p.check_budgets(11, None, user_id=7)
    # the key itself had no budget: no key counter was ever created
    assert not any(k.startswith("kbudget:11:") for k in redis.store)


@pytest.mark.asyncio
async def test_paid_account_has_no_ceiling_and_key_budget_still_applies():
    redis = _Redis()
    p = _policy(redis, _Repo(key_cu="20", user_cu="999"), None)
    await p.check_budgets(11, None, user_id=7)  # no ceiling, no key budget: allowed
    with pytest.raises(BudgetExceededError) as ei:
        await p.check_budgets(11, [{"id": "default-daily", "unit": "cu", "amount": 20.0, "window": "daily"}], user_id=7)
    assert ei.value.details["budget_id"] == "default-daily"


# -- default key budget ------------------------------------------------------------
def test_default_key_budgets(monkeypatch):
    from mlpal_assistants_service.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "default_key_daily_budget_cu", Decimal("20"))
    assert default_key_budgets("standard") == [
        {"id": "default-daily", "unit": "cu", "amount": 20.0, "window": "daily"}
    ]
    assert default_key_budgets("enterprise") is None
    monkeypatch.setattr(s, "default_key_daily_budget_cu", Decimal("0"))
    assert default_key_budgets("standard") is None


# -- request-time hold ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_held_is_cached_and_refuses_every_request(monkeypatch):
    from mlpal_assistants_service.api import deps

    svc = APIKeyService.__new__(APIKeyService)
    svc.redis = _Redis()
    svc.hold_reason = AsyncMock(return_value="verify a card to continue")
    assert await svc.held(5) is True
    assert await svc.held(5) is True
    svc.hold_reason.assert_awaited_once()  # second answer came from the cache

    svc.validate_key = AsyncMock(return_value=SimpleNamespace(user_id=5, id=1))
    with pytest.raises(AccountHeldError) as ei:
        await deps.get_current_api_key(authorization="Bearer k", api_key_service=svc)
    assert ei.value.details["code"] == "risk_hold"

    svc2 = APIKeyService.__new__(APIKeyService)
    svc2.redis = _Redis()
    svc2.hold_reason = AsyncMock(return_value=None)
    svc2.validate_key = AsyncMock(return_value=SimpleNamespace(user_id=6, id=2))
    key = await deps.get_current_api_key(authorization="Bearer k", api_key_service=svc2)
    assert key.user_id == 6
