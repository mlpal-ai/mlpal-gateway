"""Per-key policy engine: model access + spend budgets.

Model-access is pure; budgets are exercised against an in-memory fake Redis and
a mock usage repo so the counter/reconcile/accrual paths run for real."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from mlpal_assistants_service.core.exceptions import (
    BudgetExceededError,
    ModelAccessDeniedError,
)
from mlpal_assistants_service.services.policy import PolicyService, window_bounds, window_id

UTC = ZoneInfo("UTC")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _Pipe:
    def __init__(self, store):
        self._store = store
        self._ops = []

    def incrbyfloat(self, k, amt):
        self._ops.append(("incr", k, amt))
        return self

    def eval(self, script, numkeys, k, amt):
        # Mirrors the exists-gated INCRBYFLOAT the service now uses: a missing
        # counter is NOT created (so check_budgets reseeds from usage_logs).
        self._ops.append(("incr_if_exists", k, float(amt)))
        return self

    def expire(self, k, ttl):
        self._ops.append(("expire", k, ttl))
        return self

    async def execute(self):
        for op in self._ops:
            if op[0] == "incr":
                cur = float(self._store.get(op[1], 0) or 0)
                self._store[op[1]] = str(cur + op[2])
            elif op[0] == "incr_if_exists":
                if op[1] in self._store:
                    cur = float(self._store.get(op[1], 0) or 0)
                    self._store[op[1]] = str(cur + op[2])
        self._ops.clear()


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.fail = False

    async def get(self, k):
        if self.fail:
            raise RuntimeError("redis down")
        v = self.store.get(k)
        return v.encode() if isinstance(v, str) else v

    async def set(self, k, v, ex=None):
        self.store[k] = str(v)

    def pipeline(self, transaction=False):
        return _Pipe(self.store)


class FakeUsageRepo:
    def __init__(self, cu=Decimal("0")):
        self.cu = cu
        self.calls = []
        self.fail = False

    async def get_api_key_cu_in_window(self, api_key_id, start, end):
        if self.fail:
            raise RuntimeError("db down")
        self.calls.append((api_key_id, start, end))
        return self.cu


def _svc(redis=None, repo=None, cu_to_usd=10.0, tz="UTC"):
    settings = SimpleNamespace(cu_to_usd=cu_to_usd, budget_timezone=tz)
    return PolicyService(redis or FakeRedis(), repo or FakeUsageRepo(), settings=settings)


# --------------------------------------------------------------------------- #
# Model access (pure)
# --------------------------------------------------------------------------- #

def test_no_policy_allows_everything():
    s = _svc()
    assert s.is_model_allowed(None, "gpt-5.6-sol") is True
    s.check_model_access(None, "anything")  # no raise
    assert s.filter_models(None, ["a", "b"]) == ["a", "b"]


def test_allowlist_glob_and_exact():
    s = _svc()
    pol_ = {"allow": ["gpt-5.6-*", "claude-opus-5"]}
    assert s.is_model_allowed(pol_, "gpt-5.6-luna") is True
    assert s.is_model_allowed(pol_, "claude-opus-5") is True
    assert s.is_model_allowed(pol_, "gemini-3.5-flash") is False


def test_deny_wins_over_allow():
    s = _svc()
    pol_ = {"allow": ["gpt-5.6-*"], "deny": ["gpt-5.6-sol"]}
    assert s.is_model_allowed(pol_, "gpt-5.6-luna") is True
    assert s.is_model_allowed(pol_, "gpt-5.6-sol") is False
    with pytest.raises(ModelAccessDeniedError):
        s.check_model_access(pol_, "gpt-5.6-sol")


def test_wildcard_allow_with_denylist():
    s = _svc()
    pol_ = {"allow": ["*"], "deny": ["gemini-*"]}
    assert s.is_model_allowed(pol_, "gpt-5.6-sol") is True
    assert s.is_model_allowed(pol_, "gemini-3.5-flash") is False


def test_check_access_meta_requested_or_resolved_allowed():
    s = _svc()
    # key allows the alias; the resolved concrete model isn't separately listed
    pol_ = {"allow": ["mlpal-lite"]}
    s.check_model_access(pol_, requested="mlpal-lite", resolved="gpt-5-nano")  # allowed via alias
    # allowing the concrete model also works when requested directly
    pol2 = {"allow": ["gpt-5-nano"]}
    s.check_model_access(pol2, requested="gpt-5-nano", resolved="gpt-5-nano")


def test_check_access_deny_applies_to_resolved_model():
    # a denied concrete model can't be smuggled in via an allowed alias
    s = _svc()
    pol_ = {"allow": ["mlpal", "mlpal-lite"], "deny": ["gpt-5.6-sol"]}
    with pytest.raises(ModelAccessDeniedError) as ei:
        s.check_model_access(pol_, requested="mlpal", resolved="gpt-5.6-sol")
    assert ei.value.http_status == 403
    assert ei.value.details["code"] == "model_access_denied"


def test_check_access_not_in_allowlist_raises():
    s = _svc()
    with pytest.raises(ModelAccessDeniedError):
        s.check_model_access({"allow": ["claude-*"]}, requested="gpt-5.6-sol", resolved="gpt-5.6-sol")


def test_filter_models_respects_policy():
    s = _svc()
    pol_ = {"allow": ["gpt-*"], "deny": ["gpt-5.6-sol"]}
    got = s.filter_models(pol_, ["gpt-5.6-luna", "gpt-5.6-sol", "claude-opus-5"])
    assert got == ["gpt-5.6-luna"]


# --------------------------------------------------------------------------- #
# Window math
# --------------------------------------------------------------------------- #

def test_window_ids():
    now = datetime(2026, 8, 2, 15, 30, tzinfo=UTC)  # Sunday
    assert window_id("daily", now) == "2026-08-02"
    assert window_id("monthly", now) == "2026-08"
    assert window_id("lifetime", now) == "lifetime"
    # 2026-08-02 is ISO week 31
    assert window_id("weekly", now) == "2026-W31"


def test_window_bounds_daily_weekly_monthly():
    now = datetime(2026, 8, 5, 13, 0, tzinfo=UTC)  # Wednesday
    ds, de = window_bounds("daily", now)
    assert ds == datetime(2026, 8, 5, tzinfo=UTC) and de == datetime(2026, 8, 6, tzinfo=UTC)
    ws, we = window_bounds("weekly", now)
    assert ws == datetime(2026, 8, 3, tzinfo=UTC)  # Monday
    assert we == datetime(2026, 8, 10, tzinfo=UTC)
    ms, me = window_bounds("monthly", now)
    assert ms == datetime(2026, 8, 1, tzinfo=UTC) and me == datetime(2026, 9, 1, tzinfo=UTC)
    assert window_bounds("lifetime", now) == (None, None)


def test_month_end_rolls_to_next_year():
    now = datetime(2026, 12, 20, tzinfo=UTC)
    ms, me = window_bounds("monthly", now)
    assert ms == datetime(2026, 12, 1, tzinfo=UTC) and me == datetime(2027, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #

def test_usd_cu_conversion_uses_config_rate():
    s = _svc(cu_to_usd=10.0)
    # $100 budget == 10 CU at $10/CU
    assert s._to_cu(Decimal("100"), "usd") == Decimal("10")
    assert s._to_cu(Decimal("10"), "cu") == Decimal("10")
    assert s._from_cu(Decimal("10"), "usd") == Decimal("100")


# --------------------------------------------------------------------------- #
# Budget enforcement
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_no_budgets_is_noop():
    s = _svc()
    await s.check_budgets(1, None)
    await s.check_budgets(1, [])


@pytest.mark.asyncio
async def test_budget_under_limit_passes_and_seeds_from_db():
    repo = FakeUsageRepo(cu=Decimal("3"))
    redis = FakeRedis()
    s = _svc(redis=redis, repo=repo)
    budgets = [{"id": "d", "window": "daily", "unit": "cu", "amount": 10}]
    await s.check_budgets(7, budgets)  # 3 < 10 -> ok
    assert repo.calls, "should have reconciled from DB on cold counter"
    # counter got seeded so a second check is Redis-only
    repo.calls.clear()
    await s.check_budgets(7, budgets)
    assert not repo.calls


@pytest.mark.asyncio
async def test_budget_at_or_over_limit_denies_402_with_details():
    repo = FakeUsageRepo(cu=Decimal("10"))  # exactly at the CU limit
    s = _svc(repo=repo)
    budgets = [{"id": "cap", "window": "monthly", "unit": "usd", "amount": 100}]  # 100usd=10cu
    with pytest.raises(BudgetExceededError) as ei:
        await s.check_budgets(7, budgets)
    e = ei.value
    assert e.http_status == 402
    assert e.window == "monthly" and e.unit == "usd"
    assert e.limit == 100.0 and e.spent == 100.0  # 10 CU -> $100
    assert e.reset_at is not None and e.details["code"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_multiple_budgets_any_over_denies():
    # daily ok (2<5) but lifetime over (2>=1) -> deny on lifetime
    repo = FakeUsageRepo(cu=Decimal("2"))
    s = _svc(repo=repo)
    budgets = [
        {"window": "daily", "unit": "cu", "amount": 5},
        {"window": "lifetime", "unit": "cu", "amount": 1},
    ]
    with pytest.raises(BudgetExceededError) as ei:
        await s.check_budgets(7, budgets)
    assert ei.value.window == "lifetime"


@pytest.mark.asyncio
async def test_accrual_increments_each_window_and_check_sees_it():
    redis = FakeRedis()
    repo = FakeUsageRepo(cu=Decimal("0"))  # DB empty; counter driven by accrual
    s = _svc(redis=redis, repo=repo)
    budgets = [
        {"window": "daily", "unit": "cu", "amount": 10},
        {"window": "lifetime", "unit": "cu", "amount": 100},
    ]
    # Real request flow: check_budgets runs FIRST and seeds the counters from
    # usage_logs; accrual then increments only existing counters (a missing
    # counter after Redis eviction must reseed, not restart from this request).
    await s.check_budgets(7, budgets)
    await s.record_key_usage(7, budgets, Decimal("4"))
    await s.record_key_usage(7, budgets, Decimal("4"))
    # daily now 8 (<10 ok); push over
    await s.record_key_usage(7, budgets, Decimal("3"))  # daily 11
    with pytest.raises(BudgetExceededError) as ei:
        await s.check_budgets(7, budgets)
    assert ei.value.window == "daily"


@pytest.mark.asyncio
async def test_accrual_ignored_for_zero_or_no_budgets():
    redis = FakeRedis()
    s = _svc(redis=redis)
    await s.record_key_usage(7, None, Decimal("5"))
    await s.record_key_usage(7, [{"window": "daily", "unit": "cu", "amount": 1}], Decimal("0"))
    assert redis.store == {}


@pytest.mark.asyncio
async def test_fails_open_when_both_stores_down():
    redis = FakeRedis()
    redis.fail = True
    repo = FakeUsageRepo()
    repo.fail = True
    s = _svc(redis=redis, repo=repo)
    # spend unknowable -> treat as 0 -> allow (availability over strictness)
    await s.check_budgets(7, [{"window": "daily", "unit": "cu", "amount": 1}])


# --------------------------------------------------------------------------- #
# Wiring: the gate actually fires inside ChatService BEFORE the provider call,
# and the exceptions map to the right HTTP status through the app handler.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_chat_service_gate_denies_model_before_provider_call():
    from unittest.mock import AsyncMock, MagicMock

    from mlpal_assistants_service.schemas.chat import ChatCompletionRequest
    from mlpal_assistants_service.services.chat import ChatService

    svc = ChatService(session=MagicMock(), redis_client=None)
    svc._billing.can_make_request_cached = AsyncMock(return_value=(True, None, True))
    adapter = MagicMock()
    adapter.chat = AsyncMock()  # MUST NOT be reached
    # meta 'mlpal' resolves to a concrete model the key does not allow
    rmeta = SimpleNamespace(resolved_model="gpt-5.6-sol")
    svc._router.get_adapter_with_breaker_for_operation = AsyncMock(
        return_value=(adapter, "gpt-5.6-sol", MagicMock(), SimpleNamespace(provider="openai"), rmeta)
    )
    req = ChatCompletionRequest(model="mlpal", messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(ModelAccessDeniedError):
        await svc.chat(user_id=1, api_key_id=1, request=req, model_policy={"allow": ["claude-*"]})
    adapter.chat.assert_not_called()  # denied before spending a provider call


@pytest.mark.asyncio
async def test_app_handler_maps_policy_exceptions_to_403_and_402():
    from unittest.mock import MagicMock

    import mlpal_assistants_service.main as m

    req = MagicMock()
    req.url.path = "/v1/chat/completions"
    r403 = await m.service_exception_handler(req, ModelAccessDeniedError("gpt-5.6-sol"))
    assert r403.status_code == 403
    r402 = await m.service_exception_handler(
        req, BudgetExceededError(window="daily", unit="usd", limit=100, spent=100, reset_at=None)
    )
    assert r402.status_code == 402



def test_is_routing_allowed_mirrors_check_model_access():
    from mlpal_assistants_service.services.policy import PolicyService as P

    assert P.is_routing_allowed(None, "mlpal", "claude-opus-5")
    assert P.is_routing_allowed({"allow": ["mlpal"]}, "mlpal", "claude-opus-5")        # alias grants routing
    assert P.is_routing_allowed({"allow": ["claude-*"]}, "mlpal", "claude-opus-5")     # resolved allowed
    assert not P.is_routing_allowed({"allow": ["gpt-*"]}, "mlpal", "claude-opus-5")    # neither allowed
    assert not P.is_routing_allowed({"allow": ["mlpal"], "deny": ["claude-opus-5"]}, "mlpal", "claude-opus-5")  # deny wins
    assert not P.is_routing_allowed({"deny": ["mlpal*"]}, "mlpal-lite", "gpt-5.6-luna")
