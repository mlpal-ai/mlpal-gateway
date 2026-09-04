"""Cache WRITES on the OpenAI wire (Anthropic models with cache_control).

Regression for the under-bill found 2026-09-04: Anthropic's usage.input_tokens
excludes cache-creation tokens, so the first (writing) call of a cached
prompt billed only the uncached remainder. Writes bill at 1.25x (5m) / 2x (1h)
and the wire reports one convention for every provider: input_tokens is the
whole prompt; cached_tokens / cache_write_tokens are subsets.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter
from mlpal_assistants_service.adapters.base import TokenUsage as AdapterTokenUsage
from mlpal_assistants_service.db.models.model_pricing import ModelPricing
from mlpal_assistants_service.services.chat import _wire_token_usage
from mlpal_assistants_service.services.pricing import PricingService


def _fable_pricing() -> ModelPricing:
    return ModelPricing(
        model_tag="claude-fable-5-1", operation="chat", input_rate=Decimal("10"),
        output_rate=Decimal("50"), cache_read_rate=Decimal("0.25"),
        rate_unit="per_1m_tokens", markup_multiplier=Decimal("1"),
        cu_to_dollar=Decimal("10"),
        # Generated columns in Postgres: rate x markup / cu_to_dollar.
        input_cu_rate=Decimal("1"), output_cu_rate=Decimal("5"),
    )


@pytest.fixture
def pricing() -> PricingService:
    svc = PricingService.__new__(PricingService)
    svc.get_pricing = AsyncMock(return_value=_fable_pricing())
    return svc


def _dollars(inp, w5, w1h, read, out) -> Decimal:
    return (
        Decimal(inp) * 10 + Decimal(w5) * 10 * Decimal("1.25") + Decimal(w1h) * 10 * 2
        + Decimal(read) * Decimal("0.25") + Decimal(out) * 50
    ) / Decimal(1_000_000)


@pytest.mark.asyncio
async def test_write_leg_bills_the_written_prefix_anthropic_semantics(pricing):
    # Anthropic wire: input_tokens = uncached remainder only.
    cu = await pricing.calculate_compute_units(
        "claude-fable-5-1", 16, 4, cached_units=0, cached_included=False,
        provider="anthropic", cache_write_5m_units=25_214,
    )
    assert cu * 10 == _dollars(16, 25_214, 0, 0, 4)


@pytest.mark.asyncio
async def test_one_hour_tier_bills_double(pricing):
    cu = await pricing.calculate_compute_units(
        "claude-fable-5-1", 0, 0, cached_included=False, provider="anthropic",
        cache_write_1h_units=1_000,
    )
    assert cu * 10 == _dollars(0, 0, 1_000, 0, 0)


@pytest.mark.asyncio
async def test_writes_subtracted_when_input_includes_them(pricing):
    # OpenAI-style semantics: prompt total includes the written subset.
    cu = await pricing.calculate_compute_units(
        "claude-fable-5-1", 1_016, 0, cached_included=True, provider="anthropic",
        cache_write_5m_units=1_000,
    )
    assert cu * 10 == _dollars(16, 1_000, 0, 0, 0)


@pytest.mark.asyncio
async def test_no_cache_tokens_is_byte_identical_to_plain_path(pricing):
    plain = await pricing.calculate_compute_units("claude-fable-5-1", 100, 10)
    assert plain * 10 == _dollars(100, 0, 0, 0, 10)


def test_anthropic_usage_extracts_tiered_writes_and_reads():
    usage = SimpleNamespace(
        input_tokens=16, output_tokens=4, cache_read_input_tokens=0,
        cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=25_214, ephemeral_1h_input_tokens=7),
    )
    tu = AnthropicAdapter._token_usage(usage)
    assert (tu.input_tokens, tu.cached_tokens, tu.cache_write_5m_tokens, tu.cache_write_1h_tokens) == (16, 0, 25_214, 7)


def test_anthropic_usage_legacy_untiered_total_counts_as_5m():
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1, cache_read_input_tokens=3,
        cache_creation=None, cache_creation_input_tokens=500,
    )
    tu = AnthropicAdapter._token_usage(usage)
    assert (tu.cached_tokens, tu.cache_write_5m_tokens, tu.cache_write_1h_tokens) == (3, 500, 0)


def test_wire_usage_sums_prompt_for_anthropic_semantics():
    adapter_usage = AdapterTokenUsage(input_tokens=16, output_tokens=4, cached_tokens=25_214, cache_write_5m_tokens=100)
    wire = _wire_token_usage(adapter_usage, cached_included=False)
    assert wire.input_tokens == 16 + 25_214 + 100
    assert wire.total_tokens == wire.input_tokens + 4
    assert (wire.cached_tokens, wire.cache_write_tokens) == (25_214, 100)


def test_wire_usage_unchanged_for_openai_semantics():
    adapter_usage = AdapterTokenUsage(input_tokens=20_000, output_tokens=4, cached_tokens=16_012)
    wire = _wire_token_usage(adapter_usage, cached_included=True)
    assert (wire.input_tokens, wire.cached_tokens, wire.cache_write_tokens) == (20_000, 16_012, 0)
