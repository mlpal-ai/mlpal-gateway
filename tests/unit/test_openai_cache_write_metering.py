"""OpenAI prompt-cache WRITES (gpt-5.6 / gpt-6 generation).

Verified live 2026-09-07: Responses usage.input_tokens_details carries
cache_write_tokens, and OpenAI's pricing page charges 1.25x input for them
(gpt-6-astra $12.50/M, gpt-5.6-sol $5.00/M). input_tokens already INCLUDES
both the cached and the written subsets. Both wires must bill the write leg.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from mlpal_assistants_service.adapters.openai import (
    _cache_write_tokens,
    _completions_cache_tokens,
    _is_reasoning_model,
)
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage

IN = Decimal("1")      # $10/MTok at 1 CU = $10 → 1 CU per 1M tokens
OUT = Decimal("5")


def test_responses_usage_extracts_cache_writes():
    usage = SimpleNamespace(input_tokens_details=SimpleNamespace(cache_write_tokens=16023, cached_tokens=0))
    assert _cache_write_tokens(usage) == 16023
    assert _cache_write_tokens(SimpleNamespace(input_tokens_details=None)) == 0
    assert _cache_write_tokens(None) == 0


def test_completions_usage_extracts_cached_and_written():
    usage = SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=7, cache_write_tokens=9))
    assert _completions_cache_tokens(usage) == (7, 9)
    assert _completions_cache_tokens(SimpleNamespace()) == (0, 0)


def test_gpt6_is_a_reasoning_model():
    assert _is_reasoning_model("gpt-6-astra")
    assert not _is_reasoning_model("gpt-4o")


def test_from_openai_carries_writes_into_canonical_usage():
    adapter_usage = SimpleNamespace(input_tokens=16026, output_tokens=5, cached_tokens=0,
                                    cache_write_5m_tokens=16023, cache_write_1h_tokens=0)
    cu = CanonicalUsage.from_openai(adapter_usage)
    assert (cu.input, cu.cache_read, cu.cache_write, cu.raw) == (16026, 0, 16023, None)


@pytest.mark.parametrize("multiplier", [Decimal("1.25")])
def test_flat_cu_bills_write_leg_at_write_tier(monkeypatch, multiplier):
    from mlpal_assistants_service.core import config

    monkeypatch.setattr(
        config, "get_settings",
        lambda: SimpleNamespace(cache_5m_write_multiplier=multiplier, cache_1h_write_multiplier=Decimal("2")),
    )
    write_leg = CanonicalUsage(input=16026, output=5, cache_read=0, cache_write=16023, raw=None)
    # 3 plain tokens at input, 16023 at 1.25x input, 5 output
    expected = (Decimal(3) + Decimal(16023) * multiplier) * IN / Decimal(1_000_000) + Decimal(5) * OUT / Decimal(1_000_000)
    assert write_leg.compute_units(IN / Decimal(1_000_000), OUT / Decimal(1_000_000)) == expected


def test_flat_cu_read_leg_unchanged():
    read_leg = CanonicalUsage(input=16026, output=5, cache_read=16023, cache_write=0, raw=None)
    cache = IN / Decimal(10)  # 0.10x
    expected = (Decimal(3) * IN + Decimal(16023) * cache + Decimal(5) * OUT) / Decimal(1_000_000)
    assert read_leg.compute_units(IN / Decimal(1_000_000), OUT / Decimal(1_000_000), cache / Decimal(1_000_000)) == expected
