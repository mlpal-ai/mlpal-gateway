"""
Bedrock-mantle pass-through client.

Why a separate client (not the existing BedrockAdapter):

The existing `BedrockAdapter` calls the Bedrock **Converse** API on
`bedrock-runtime.{region}.amazonaws.com`. That requires Anthropic→internal
→Converse schema translation in both directions, which is brittle for
Claude Code's full feature surface (tools, cache_control, streaming
deltas, etc.).

This client targets the newer **Claude in Amazon Bedrock** endpoint at
`bedrock-mantle.{region}.api.aws/anthropic/v1/messages`, which speaks
the native Anthropic Messages API with native Anthropic SSE streaming.
That lets the `POST /v1/messages` route be pure pass-through — verified
end-to-end with real `claude` CLI invocations against an isolated test
proxy first (see `mlpal/code/cc-bedrock-test/`).

Current limitations:
- Mantle only exposes Opus 4.7 and Haiku 4.5 in us-east-1 as of 2026-05.
  Sonnet must be added when AWS exposes it on mantle.
- us-east-2 has the mantle hostname but no models — must run client
  region as us-east-1.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Body fields Bedrock's /anthropic/v1/messages rejects. Each entry MUST
# come with a clear rationale (when it was added, why we strip, and
# whether the loss has a user-visible impact). Adding to this set is
# load-bearing: it means the feature is silently disabled for every
# Claude Code session that uses our gateway.
#
# Empty by default — start permissive, capture Bedrock's actual
# rejection messages, then make per-field decisions backed by data.
_REJECTED_BODY_FIELDS: frozenset[str] = frozenset()


# `anthropic-beta` header values that bedrock-mantle does not recognize.
# Probed individually against the live endpoint with each beta in
# isolation; only flags that genuinely return "invalid beta flag" go
# here. Per-flag rationale required for every entry; the impact must
# be understood before adding.
#
# Format: { "beta-flag-name": "why we filter it + what's lost" }
INCOMPATIBLE_BETA_FLAGS: dict[str, str] = {
    # Anthropic's "global prompt-caching scope" (2026-01) — broadens the
    # cache hit radius beyond a single request. Bedrock-mantle returns
    # 400 'invalid beta flag' when present. Loss: caching still works
    # per-request (cache_creation/cache_read tokens are billed); only
    # the *cross-request* scope optimization is unavailable. Confirmed
    # by isolating each flag Claude Code sends and probing one at a
    # time on bedrock-mantle us-east-1, 2026-05-27.
    "prompt-caching-scope-2026-01-05":
        "Bedrock-mantle 'invalid beta flag'. Cross-request cache scope "
        "unavailable; per-request prompt caching still works.",
}


def filter_anthropic_beta_header(value: str | None) -> tuple[str | None, list[str]]:
    """Strip incompatible flags from a comma-separated anthropic-beta
    header. Returns (filtered_value_or_None, stripped_list).

    Order is preserved. The remaining flags are forwarded to Bedrock
    intact. If all flags get stripped, returns None so the caller can
    skip sending the header entirely.
    """
    if not value:
        return value, []
    flags = [f.strip() for f in value.split(",")]
    kept: list[str] = []
    stripped: list[str] = []
    for f in flags:
        if f in INCOMPATIBLE_BETA_FLAGS:
            stripped.append(f)
        elif f:
            kept.append(f)
    filtered = ",".join(kept) if kept else None
    return filtered, stripped


class BedrockMantleClient:
    """Thin SigV4-signing helper for bedrock-mantle. The actual HTTP
    transport happens in the route (httpx for streaming support) — this
    class only owns the regional URL, the signer, and body adaptation."""

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        self.url = f"https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages"
        self._session = boto3.Session(region_name=region)

    def sign(
        self, body: bytes, extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return SigV4 headers for a POST of `body` to self.url.

        Extra headers (e.g. `anthropic-beta`, `anthropic-version` from
        the caller) are included in the signed header set so AWS treats
        them as integrity-protected. If we add headers AFTER signing,
        some AWS services accept them but bedrock-mantle's behavior is
        unspecified — safer to include them up front."""
        creds = self._session.get_credentials().get_frozen_credentials()
        base = {"Content-Type": "application/json"}
        if extra_headers:
            base.update(extra_headers)
        req = AWSRequest(
            method="POST", url=self.url, data=body, headers=base,
        )
        SigV4Auth(creds, "bedrock", self.region).add_auth(req)
        return dict(req.prepare().headers)

    @staticmethod
    def adapt_body(raw: bytes) -> tuple[bytes, dict[str, Any], list[str]]:
        """Normalize the incoming Anthropic Messages body for Bedrock.

        Mutations:
          * `anthropic_version` injected if absent (Bedrock requires it)
          * `model` prefixed with "anthropic." if it lacks the prefix
          * known-rejected fields stripped (see _REJECTED_BODY_FIELDS)
          * `output_config` stripped when it lacks `format.schema`
            (real spec mismatch: Anthropic's API accepts schema-less
            output_config for plain text output, but Bedrock's
            structured-outputs beta requires `format.schema`. Stripping
            falls through to Bedrock's default text output, which is
            what Claude Code's probe call actually wanted.)

        Returns (rewritten_body, parsed_dict, removed_fields).
        `removed_fields` lists every field we stripped, so the caller
        can log them and they appear in observability — no silent drops.
        """
        obj = json.loads(raw)
        obj.setdefault("anthropic_version", ANTHROPIC_VERSION)
        model = obj.get("model", "")
        if model and not model.startswith("anthropic."):
            obj["model"] = f"anthropic.{model}"

        removed: list[str] = []
        for field in _REJECTED_BODY_FIELDS:
            if field in obj:
                del obj[field]
                removed.append(field)

        # Bedrock-mantle's structured-outputs beta (as of 2026-05) only
        # accepts `format.type: "text"`. The `json_schema` variant
        # Claude Code uses (e.g. for the session-title generator) is
        # rejected with 'output_config.format: Extra inputs are not
        # permitted'. Verified by direct probes against both Opus 4.7
        # and Haiku 4.5 on bedrock-mantle us-east-1.
        #
        # Translation: drop output_config when format.type isn't
        # something Bedrock accepts. Caller's intent ("give me
        # structured JSON") is partially lost — Claude Code will get
        # free-form text instead and its downstream parser (e.g. title
        # generator) will fall back to defaults. Acceptable for our
        # CDE use case; users rarely look at auto-generated titles.
        #
        # When Bedrock adds json_schema support, remove this filter.
        oc = obj.get("output_config")
        if isinstance(oc, dict):
            fmt = oc.get("format")
            if isinstance(fmt, dict):
                fmt_type = fmt.get("type")
                if fmt_type and fmt_type != "text":
                    del obj["output_config"]
                    removed.append(f"output_config[format.type={fmt_type}]")

        return json.dumps(obj).encode(), obj, removed


# ---------------------------------------------------------------------------
# SSE + usage parsing
# ---------------------------------------------------------------------------

def parse_sse_event(raw: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """Parse one SSE event block (`event: NAME\\ndata: JSON`). Returns
    (event_type, parsed_data) or (event_type, None) if data is missing or
    not JSON. Bytes-in, dict-out — minimal copies."""
    event_type: str | None = None
    data_line: str | None = None
    for line in raw.split(b"\n"):
        if line.startswith(b"event:"):
            event_type = line[6:].strip().decode(errors="replace")
        elif line.startswith(b"data:"):
            data_line = line[5:].strip().decode(errors="replace")
    if data_line is None:
        return event_type, None
    try:
        return event_type, json.loads(data_line)
    except json.JSONDecodeError:
        return event_type, None


def parse_non_streaming_usage(content: bytes) -> tuple[dict[str, Any], str | None]:
    """For non-streaming responses: extract `usage` + Bedrock message
    id from the JSON body. Returns ({}, None) on parse failure."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return {}, None
    return obj.get("usage") or {}, obj.get("id")


# ---------------------------------------------------------------------------
# CU calculation with cache tier pricing
# ---------------------------------------------------------------------------

# Anthropic Bedrock cache pricing multipliers relative to base input rate.
# These match Anthropic's published Claude-on-Bedrock pricing as of 2026-05.
# When the ModelPricing schema is extended with explicit cache columns,
# move these into the DB.
# Cache CU multipliers are env-tunable (see core/config). Read at call time so
# an env change takes effect on the next request without a code change.


def compute_units_from_usage(
    usage: dict[str, Any],
    input_cu_per_unit: Decimal,
    output_cu_per_unit: Decimal,
    cache_read_cu_per_unit: Decimal | None = None,
) -> Decimal:
    """Compute total CU from an Anthropic Messages `usage` object,
    accounting for cache token tiers.

    `input_cu_per_unit` / `output_cu_per_unit` are the per-token CU
    rates (i.e. `ModelPricing.input_cu_rate / divisor`) — caller does
    the divisor math from ModelPricing.

    Cache token sources priced separately:
      input_tokens                    → 1.00x
      ephemeral_5m_input_tokens       → 1.25x (cache write, 5m TTL)
      ephemeral_1h_input_tokens       → 2.00x (cache write, 1h TTL)
      cache_read_input_tokens         → per-model rate when the row sets
                                        cache_read_rate (claude-fable-5-1:
                                        $0.25/MTok = 0.025x), else the
                                        global 0.10x multiplier
      output_tokens                   → output_cu_per_unit
    """
    from mlpal_assistants_service.core.config import get_settings

    settings = get_settings()
    inp = Decimal(usage.get("input_tokens") or 0)
    out = Decimal(usage.get("output_tokens") or 0)
    cache_creation = usage.get("cache_creation") or {}
    cache_5m = Decimal(cache_creation.get("ephemeral_5m_input_tokens") or 0)
    cache_1h = Decimal(cache_creation.get("ephemeral_1h_input_tokens") or 0)
    cache_read = Decimal(usage.get("cache_read_input_tokens") or 0)
    cache_read_rate = (
        cache_read_cu_per_unit
        if cache_read_cu_per_unit is not None
        else input_cu_per_unit * settings.cache_read_multiplier
    )

    return (
        inp * input_cu_per_unit
        + cache_5m * input_cu_per_unit * settings.cache_5m_write_multiplier
        + cache_1h * input_cu_per_unit * settings.cache_1h_write_multiplier
        + cache_read * cache_read_rate
        + out * output_cu_per_unit
    )


def billable_input_tokens(usage: dict[str, Any]) -> int:
    """Sum every input-side token class for UsageLog.input_tokens (which
    is a single int column). For accurate per-class billing use
    compute_units_from_usage; this is the canonical 'total input
    tokens used' number for the simpler column."""
    cache_creation = usage.get("cache_creation") or {}
    return (
        int(usage.get("input_tokens") or 0)
        + int(cache_creation.get("ephemeral_5m_input_tokens") or 0)
        + int(cache_creation.get("ephemeral_1h_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
    )
