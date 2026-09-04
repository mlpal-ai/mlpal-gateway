"""Per-model cache-read pricing (claude-fable-5-1: $0.25/MTok = 0.025x input).

The invariant under test: metering reproduces the provider's list price.
NULL cache_read_rate keeps the global 0.10x multiplier — identical billing
for every pre-existing row.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.services.bedrock_mantle import compute_units_from_usage

# claude-fable-5-1 list prices, per token, in CU (1 CU = $10)
IN_CU = Decimal("10") / Decimal(1_000_000) / Decimal("10")     # $10/MTok
OUT_CU = Decimal("50") / Decimal(1_000_000) / Decimal("10")    # $50/MTok
CACHE_CU = Decimal("0.25") / Decimal(1_000_000) / Decimal("10")  # $0.25/MTok

USAGE = {
    "input_tokens": 1000,
    "output_tokens": 200,
    "cache_read_input_tokens": 50_000,
    "cache_creation": {"ephemeral_5m_input_tokens": 4000},
}


def test_per_model_cache_rate_reproduces_list_price():
    cu = compute_units_from_usage(USAGE, IN_CU, OUT_CU, CACHE_CU)
    # dollars = input 1000*$10/M + write 4000*$12.5/M + read 50k*$0.25/M + out 200*$50/M
    dollars = (
        Decimal(1000) * 10 + Decimal(4000) * 10 * Decimal("1.25")
        + Decimal(50_000) * Decimal("0.25") + Decimal(200) * 50
    ) / Decimal(1_000_000)
    assert cu * 10 == dollars  # exact, not approximate — five-decimals promise


def test_null_cache_rate_falls_back_to_global_multiplier():
    with_default = compute_units_from_usage(USAGE, IN_CU, OUT_CU, None)
    explicit_tenth = compute_units_from_usage(
        USAGE, IN_CU, OUT_CU, IN_CU * Decimal("0.10")
    )
    assert with_default == explicit_tenth


def test_fable_51_cache_reads_bill_4x_below_global():
    """The whole point: 0.025x vs 0.10x on a cache-read-heavy request."""
    heavy = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000}
    global_cu = compute_units_from_usage(heavy, IN_CU, OUT_CU, None)
    fable51_cu = compute_units_from_usage(heavy, IN_CU, OUT_CU, CACHE_CU)
    assert global_cu == fable51_cu * 4


@pytest.mark.asyncio
async def test_v2_resolver_returns_cache_rate():
    from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core

    core = MessagesV2Core.__new__(MessagesV2Core)
    core._pricing = SimpleNamespace(
        get_pricing=AsyncMock(
            return_value=SimpleNamespace(
                rate_unit="per_1m_tokens",
                markup_multiplier=Decimal("3.00"),
                input_cu_rate=Decimal("3.00"),
                output_cu_rate=Decimal("15.00"),
                cache_read_rate=Decimal("0.25"),
                cu_to_dollar=Decimal("10.00"),
            )
        )
    )
    rates = await core._resolve_cu_rates("claude-fable-5-1")
    assert rates == (IN_CU, OUT_CU, CACHE_CU)


@pytest.mark.asyncio
async def test_v2_resolver_none_cache_rate_passes_none():
    from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core

    core = MessagesV2Core.__new__(MessagesV2Core)
    core._pricing = SimpleNamespace(
        get_pricing=AsyncMock(
            return_value=SimpleNamespace(
                rate_unit="per_1m_tokens",
                markup_multiplier=Decimal("3.00"),
                input_cu_rate=Decimal("1.50"),
                output_cu_rate=Decimal("7.50"),
                cache_read_rate=None,
                cu_to_dollar=Decimal("10.00"),
            )
        )
    )
    rates = await core._resolve_cu_rates("claude-opus-5")
    assert rates[2] is None


def test_catalog_rows_present_and_consistent():
    base = Path("src/mlpal_assistants_service/catalog")
    reg = json.load(open(base / "registry.json"))
    models = reg if isinstance(reg, list) else reg["models"]
    row = next(m for m in models if m["model_tag"] == "claude-fable-5-1")
    assert row["is_active"] and row["provider"] == "anthropic"
    assert row["context_length"] == 1_000_000 and row["max_output_tokens"] == 128_000

    pricing = json.load(open(base / "pricing.json"))
    rows = pricing if isinstance(pricing, list) else pricing["pricing"]
    prow = next(r for r in rows if r["model_tag"] == "claude-fable-5-1")
    assert Decimal(prow["input_rate"]) == 10 and Decimal(prow["output_rate"]) == 50
    assert Decimal(prow["cache_read_rate"]) == Decimal("0.25")
    # cu-rate convention: rate x markup / cu_to_dollar
    assert Decimal(prow["input_cu_rate"]) == Decimal("3.00")
    assert Decimal(prow["output_cu_rate"]) == Decimal("15.00")


def test_catalog_sync_carries_cache_read_rate():
    from mlpal_assistants_service.services.catalog_sync import (
        _NUMERIC,
        PRICING_FIELDS,
        _pricing_kwargs,
    )

    assert "cache_read_rate" in PRICING_FIELDS and "cache_read_rate" in _NUMERIC
    kwargs = _pricing_kwargs(
        {"model_tag": "claude-fable-5-1", "operation": "chat", "cache_read_rate": "0.25"}
    )
    assert kwargs["cache_read_rate"] == Decimal("0.25")
    # absent key -> not passed (column default NULL preserved)
    kwargs2 = _pricing_kwargs({"model_tag": "claude-opus-5", "operation": "chat"})
    assert "cache_read_rate" not in kwargs2


# ── v1 chat path: per-provider cached-token semantics ────────────────────────
def _pricing_service(cache_read_rate=None, input_rate=Decimal("10")):
    from mlpal_assistants_service.services.pricing import PricingService

    row = SimpleNamespace(
        input_rate=input_rate,
        rate_unit="per_1m_tokens",
        cu_to_dollar=Decimal("10.00"),
        cache_read_rate=cache_read_rate,
        calculate_provider_cost=lambda input_units=0, output_units=0: (
            Decimal(input_units) * input_rate + Decimal(output_units) * Decimal("50")
        ) / Decimal(1_000_000) / Decimal("10"),
    )
    svc = PricingService.__new__(PricingService)
    svc.get_pricing = AsyncMock(return_value=row)
    return svc


@pytest.mark.asyncio
async def test_v1_openai_semantics_cached_subtracted():
    """OpenAI: input INCLUDES cached — bill (input-cached) full + cached at 0.10x."""
    svc = _pricing_service()
    cu = await svc.calculate_compute_units(
        "gpt-5.2", 1000, 100, cached_units=800, cached_included=True, provider="openai"
    )
    dollars = (Decimal(200) * 10 + Decimal(800) * 1 + Decimal(100) * 50) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_v1_anthropic_semantics_cached_added():
    """Anthropic v1-wire: input EXCLUDES cached — bill input full + cached at 0.10x."""
    svc = _pricing_service()
    cu = await svc.calculate_compute_units(
        "claude-opus-5", 1000, 100, cached_units=800, cached_included=False,
        provider="anthropic",
    )
    dollars = (Decimal(1000) * 10 + Decimal(800) * 1 + Decimal(100) * 50) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_v1_google_provider_multiplier_is_quarter():
    svc = _pricing_service()
    cu = await svc.calculate_compute_units(
        "gemini-3-pro", 1000, 0, cached_units=800, cached_included=True, provider="google"
    )
    dollars = (Decimal(200) * 10 + Decimal(800) * Decimal("2.5")) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_v1_row_cache_rate_beats_provider_default():
    svc = _pricing_service(cache_read_rate=Decimal("0.25"))
    cu = await svc.calculate_compute_units(
        "claude-fable-5-1", 1000, 0, cached_units=800, cached_included=False,
        provider="anthropic",
    )
    dollars = (Decimal(1000) * 10 + Decimal(800) * Decimal("0.25")) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_v1_cached_clamped_to_input():
    """Defensive: cached > input (bad provider data) never bills negative."""
    svc = _pricing_service()
    cu = await svc.calculate_compute_units(
        "gpt-5.2", 100, 0, cached_units=500, cached_included=True, provider="openai"
    )
    dollars = (Decimal(0) * 10 + Decimal(500) * 1) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_v1_no_cache_identical_to_before():
    svc = _pricing_service()
    plain = await svc.calculate_compute_units("gpt-5.2", 1000, 100)
    explicit = await svc.calculate_compute_units(
        "gpt-5.2", 1000, 100, cached_units=0, cached_included=True, provider="openai"
    )
    assert plain == explicit


def test_provider_multiplier_map():
    from mlpal_assistants_service.services.pricing import provider_cache_read_multiplier

    assert provider_cache_read_multiplier("google") == Decimal("0.25")
    assert provider_cache_read_multiplier("openai") == Decimal("0.10")
    assert provider_cache_read_multiplier("anthropic") == Decimal("0.10")
    assert provider_cache_read_multiplier(None) == Decimal("0.10")


# ── v2 flat path (translating edge: OpenAI/Google on the Anthropic wire) ─────
def test_v2_flat_path_discounts_cache_reads():
    from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage

    u = CanonicalUsage(input=1000, output=100, cache_read=800, cache_write=0, raw=None)
    cu = u.compute_units(IN_CU, OUT_CU, IN_CU * Decimal("0.10"))
    dollars = (Decimal(200) * 10 + Decimal(800) * 1 + Decimal(100) * 50) / Decimal(1_000_000)
    assert cu * 10 == dollars


# ── Redis cache round-trip must preserve the per-model cache rate ─────────────
def test_pricing_cache_roundtrip_preserves_cache_read_rate():
    """Regression: the Redis serializer dropped cache_read_rate, so cache HITS
    fell back to the global 0.10x multiplier (claude-fable-5-1 billed at
    $1.00/MTok instead of $0.25 on ~1 in 4 prod requests, 2026-09-04)."""
    from datetime import date

    from mlpal_assistants_service.db.models.model_pricing import ModelPricing
    from mlpal_assistants_service.services.pricing import PricingService

    svc = PricingService.__new__(PricingService)
    row = ModelPricing()
    for k, v in {
        "id": 1, "model_tag": "claude-fable-5-1", "operation": "chat", "tier": "premium",
        "input_rate": Decimal("10"), "output_rate": Decimal("50"), "rate_unit": "per_1m_tokens",
        "markup_multiplier": Decimal("3"), "effective_date": date(2026, 9, 1), "is_active": True,
        "cu_to_dollar": Decimal("10"), "input_cu_rate": Decimal("3"), "output_cu_rate": Decimal("15"),
        "cache_read_rate": Decimal("0.25"),
    }.items():
        setattr(row, k, v)

    back = svc._dict_to_pricing(json.loads(json.dumps(svc._pricing_to_dict(row))))
    assert back.cache_read_rate == Decimal("0.25")

    row.cache_read_rate = None
    back = svc._dict_to_pricing(json.loads(json.dumps(svc._pricing_to_dict(row))))
    assert back.cache_read_rate is None

    # entries cached before the field existed must not crash (prefix bump
    # makes them unreachable, but the loader stays tolerant regardless)
    legacy = svc._pricing_to_dict(row)
    legacy.pop("cache_read_rate")
    assert svc._dict_to_pricing(legacy).cache_read_rate is None


def test_pricing_cache_prefix_is_versioned():
    from mlpal_assistants_service.services.pricing import PricingService

    assert PricingService.CACHE_PREFIX != "pricing:", "bump the prefix when the cached shape changes"
