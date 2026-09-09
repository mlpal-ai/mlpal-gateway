"""Provider-agnostic usage normalization for /v2/messages.

The canonical usage object is the single shape every provider edge reports to
the core. In v2-A only the Anthropic edge exists, so normalization is from the
Anthropic Messages `usage` object — but the shape is deliberately
provider-neutral so the v2-B (OpenAI) and v2-C (Google) edges populate the same
fields (e.g. OpenAI `cached_tokens` → `cache_read`).

CU computation for v2-A delegates to the proven, cache-tier-accurate
`bedrock_mantle.compute_units_from_usage` (Anthropic-format `usage` in →
tier-priced CU out). Generalizing that into a fully provider-agnostic CU
function is a v2-B task (tracked in the feature spec §7); for v2-A we reuse it
verbatim so Anthropic billing stays tier-faithful.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from mlpal_assistants_service.services.bedrock_mantle import (
    billable_input_tokens,
    compute_units_from_usage,
)


@dataclass
class CanonicalUsage:
    """The one usage shape every edge reports to the core.

    `input`/`output` are the billable token totals for the UsageLog int
    columns; `cache_read`/`cache_write` surface caching to the client and to
    analytics. `reasoning` is optional (only where a provider reports it).
    `raw` keeps the provider's native usage dict so CU can be computed with
    full fidelity (e.g. Anthropic's 5m/1h cache-write tiers, which the flat
    canonical fields don't distinguish).
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_anthropic(cls, usage: dict[str, Any]) -> CanonicalUsage:
        """Build canonical usage from an Anthropic Messages `usage` object.

        `cache_creation_input_tokens` is the flat total; the tiered breakdown
        (`cache_creation.ephemeral_5m/1h_input_tokens`) is preserved in `raw`
        for CU. `billable_input_tokens` sums every input-side class.
        """
        usage = usage or {}
        cache_write = usage.get("cache_creation_input_tokens")
        if cache_write is None:
            cc = usage.get("cache_creation") or {}
            cache_write = int(cc.get("ephemeral_5m_input_tokens") or 0) + int(
                cc.get("ephemeral_1h_input_tokens") or 0
            )
        return cls(
            input=billable_input_tokens(usage),
            output=int(usage.get("output_tokens") or 0),
            cache_read=int(usage.get("cache_read_input_tokens") or 0),
            cache_write=int(cache_write or 0),
            raw=usage,
        )

    @classmethod
    def from_openai(cls, usage: Any) -> CanonicalUsage:
        """Build canonical usage from an adapter ``TokenUsage`` (OpenAI/Google).

        ``input_tokens`` from these providers already includes the cached and
        written prefixes (``cached_tokens`` / cache writes are subsets); ``raw``
        is left None to force the flat CU path, which bills the cached portion
        at the provider's cache-read rate, writes at the standard 1.25x write
        tier, and the remainder at the input rate (matching v1 billing).
        """
        writes = int(getattr(usage, "cache_write_5m_tokens", 0) or 0) + int(
            getattr(usage, "cache_write_1h_tokens", 0) or 0
        )
        return cls(
            input=int(getattr(usage, "input_tokens", 0) or 0),
            output=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read=int(getattr(usage, "cached_tokens", 0) or 0),
            cache_write=writes,
            raw=None,
        )

    def compute_units(
        self,
        input_cu_per_token: Decimal,
        output_cu_per_token: Decimal,
        cache_read_cu_per_token: Decimal | None = None,
    ) -> Decimal:
        """Tier-accurate CU. Uses the provider's raw usage when available
        (Anthropic cache tiers); otherwise a flat input/output computation.
        `cache_read_cu_per_token` overrides the global cache-read multiplier
        for models with their own cache-read list price."""
        if self.raw is not None:
            return compute_units_from_usage(
                self.raw, input_cu_per_token, output_cu_per_token,
                cache_read_cu_per_token,
            )
        # Flat path = from_openai semantics: `input` INCLUDES the cached and
        # written subsets. Bill the plain remainder at the input rate, the
        # cached portion at the provider's cache-read rate, and writes at the
        # standard 1.25x write tier (OpenAI charges it since gpt-5.6/gpt-6).
        from mlpal_assistants_service.core.config import get_settings

        cached = min(Decimal(self.cache_read), Decimal(self.input))
        written = min(Decimal(self.cache_write), Decimal(self.input) - cached)
        cache_rate = (
            cache_read_cu_per_token
            if cache_read_cu_per_token is not None
            else input_cu_per_token
        )
        return (
            (Decimal(self.input) - cached - written) * input_cu_per_token
            + cached * cache_rate
            + written * input_cu_per_token * get_settings().cache_5m_write_multiplier
            + Decimal(self.output) * output_cu_per_token
        )
