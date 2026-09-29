"""Long-context pricing tier (migration 20260929_1100).

Contract: when a request's prompt (all tokens sent, cached + written subsets
included) EXCEEDS the row's long_context_threshold, the whole request bills
at the long rates: input, output, cache reads (long_cache_read_rate, else the
provider multiple of long input) and cache writes (standard write multiple of
long input). Rows without a threshold bill exactly as before. Both wires.
Rates verified 2026-09-29 against the OpenAI and Google pricing pages.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.db.models.model_pricing import ModelPricing
from mlpal_assistants_service.services.catalog_sync import _NUMERIC, PRICING_FIELDS
from mlpal_assistants_service.services.messages_v2.core import MessagesV2Core
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage
from mlpal_assistants_service.services.pricing import PricingService, long_tier_applies

M = Decimal(1_000_000)


def _astra(**over) -> ModelPricing:
    base = {"model_tag": "gpt-6-astra", "operation": "chat", "tier": "premium", "rate_unit": "per_1m_tokens",
            "input_rate": Decimal("10"), "output_rate": Decimal("50"), "markup_multiplier": Decimal("3"),
            "cu_to_dollar": Decimal("10"), "input_cu_rate": Decimal("3"), "output_cu_rate": Decimal("15"),
            "long_context_threshold": 272000, "long_input_rate": Decimal("20"), "long_output_rate": Decimal("75"),
            "long_cache_read_rate": Decimal("2")}
    base.update(over)
    return ModelPricing(**base)


def _svc(row) -> PricingService:
    svc = PricingService.__new__(PricingService)
    svc.get_pricing = AsyncMock(return_value=row)
    return svc


def test_tier_boundary_is_strictly_above_threshold():
    row = _astra()
    assert not long_tier_applies(row, 272000)
    assert long_tier_applies(row, 272001)
    assert not long_tier_applies(_astra(long_context_threshold=None), 10**7)
    assert not long_tier_applies(SimpleNamespace(input_rate=1), 10**7)  # rows without the columns


@pytest.mark.asyncio
async def test_openai_long_request_bills_all_four_rates_long():
    # 300K prompt: 250K cached, 20K written, 30K plain; 1K output — OpenAI convention (input includes subsets)
    cu = await _svc(_astra()).calculate_compute_units(
        "gpt-6-astra", 300_000, 1_000, cached_units=250_000, cached_included=True,
        provider="openai", cache_write_5m_units=20_000,
    )
    dollars = (Decimal(30_000) * 20 + Decimal(250_000) * 2 + Decimal(20_000) * 20 * Decimal("1.25") + Decimal(1_000) * 75) / M
    assert cu == dollars / 10


@pytest.mark.asyncio
async def test_short_request_on_tiered_row_is_unchanged():
    row = _astra()
    cu = await _svc(row).calculate_compute_units("gpt-6-astra", 200_000, 1_000, cached_units=150_000, cached_included=True, provider="openai")
    dollars = (Decimal(50_000) * 10 + Decimal(150_000) * 1 + Decimal(1_000) * 50) / M
    assert cu == dollars / 10


@pytest.mark.asyncio
async def test_anthropic_convention_counts_cache_toward_the_prompt():
    # Anthropic-style adapter usage: input EXCLUDES cached; 100K plain + 200K cached = 300K prompt > 272K
    cu = await _svc(_astra()).calculate_compute_units(
        "gpt-6-astra", 100_000, 10, cached_units=200_000, cached_included=False, provider="openai",
    )
    assert cu == (Decimal(100_000) * 20 + Decimal(200_000) * 2 + Decimal(10) * 75) / M / 10


@pytest.mark.asyncio
async def test_gemini_long_cache_read_defaults_to_provider_multiple():
    row = _astra(model_tag="gemini-2.5-pro", input_rate=Decimal("1.25"), output_rate=Decimal("10"),
                 long_context_threshold=200000, long_input_rate=Decimal("2.5"), long_output_rate=Decimal("15"), long_cache_read_rate=None)
    cu = await _svc(row).calculate_compute_units("gemini-2.5-pro", 250_000, 100, cached_units=50_000, cached_included=True, provider="google")
    assert cu == (Decimal(200_000) * Decimal("2.5") + Decimal(50_000) * Decimal("2.5") * Decimal("0.25") + Decimal(100) * 15) / M / 10


@pytest.mark.asyncio
async def test_v2_meter_switches_tier_by_prompt_total():
    core = MessagesV2Core.__new__(MessagesV2Core)
    core._pricing = SimpleNamespace(get_pricing=AsyncMock(return_value=_astra()))
    in_cu, out_cu, cache_cu = await core._resolve_cu_rates("gpt-6-astra")
    assert (in_cu, out_cu) == (Decimal("3") / M / 3, Decimal("15") / M / 3)
    threshold, (lin, lout, lcache) = core.long_tiers["gpt-6-astra"]
    assert threshold == 272000 and (lin, lout, lcache) == (Decimal("20") / M / 10, Decimal("75") / M / 10, Decimal("2") / M / 10)
    # prompt_total: OpenAI convention vs Anthropic raw usage
    assert CanonicalUsage(input=300_000, output=1, cache_read=250_000).prompt_total() == 300_000
    assert CanonicalUsage.from_anthropic({"input_tokens": 100_000, "cache_read_input_tokens": 200_000, "output_tokens": 1}).prompt_total() == 300_000


def test_catalog_rows_and_sync_fields():
    rows = json.load(open("src/mlpal_assistants_service/catalog/pricing.json"))
    by = {r["model_tag"]: r for r in rows if r["operation"] == "chat"}
    assert by["gpt-6-astra"]["long_context_threshold"] == 272000 and Decimal(by["gpt-6-astra"]["long_input_rate"]) == 20
    assert Decimal(by["gpt-6-astra"]["long_output_rate"]) == 75 and Decimal(by["gpt-6-astra"]["long_cache_read_rate"]) == 2
    assert by["gpt-6-luna"]["long_context_threshold"] == 272000 and Decimal(by["gpt-6-luna"]["long_input_rate"]) == Decimal("0.2")
    assert by["gemini-3.1-pro-preview"]["long_context_threshold"] == 200000 and Decimal(by["gemini-3.1-pro-preview"]["cache_read_rate"]) == Decimal("0.2")
    assert by["gemini-2.5-pro"]["long_cache_read_rate"] is None
    for tag in ("claude-opus-5-5", "claude-sonnet-5", "gpt-5.5"):
        assert by[tag].get("long_context_threshold") is None  # Anthropic: 1M at standard; no tier
    for f in ("long_context_threshold", "long_input_rate", "long_output_rate", "long_cache_read_rate"):
        assert f in PRICING_FIELDS
    assert {"long_input_rate", "long_output_rate", "long_cache_read_rate"} <= _NUMERIC


def test_serializer_round_trips_tier_and_prefix_bumped():
    svc = PricingService.__new__(PricingService)
    assert PricingService.CACHE_PREFIX == "pricing:v4:"
    from datetime import date

    row = _astra()
    row.id, row.is_active, row.effective_date = 1, True, date(2026, 9, 29)
    back = svc._dict_to_pricing(svc._pricing_to_dict(row))
    assert back.long_context_threshold == 272000 and back.long_input_rate == Decimal("20") and back.long_cache_read_rate == Decimal("2")
    row2 = _astra(long_context_threshold=None, long_input_rate=None, long_output_rate=None, long_cache_read_rate=None)
    row2.id, row2.is_active, row2.effective_date = 2, True, date(2026, 9, 29)
    assert svc._dict_to_pricing(svc._pricing_to_dict(row2)).long_context_threshold is None
