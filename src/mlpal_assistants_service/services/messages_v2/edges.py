"""The core↔edge seam for /v2/messages.

A provider edge turns a validated Anthropic-Messages request into Anthropic
wire output (JSON for non-streaming, raw Anthropic SSE bytes for streaming) and
reports normalized usage back to the core. Everything cross-cutting — auth,
routing, usage→CU, wallet debit, error envelope, SSE transport + heartbeat — is
owned by the core and called identically for every edge.

v2-A implements only the Anthropic edge (native passthrough). v2-B adds an
OpenAI edge and v2-C a Google edge implementing this SAME protocol, with a
shared StreamChunk→Anthropic-SSE emitter; the Anthropic edge bypasses that
emitter via native passthrough. The protocol is what keeps this a converging
architecture rather than three snowflakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage


@dataclass
class RequestContext:
    """Per-request state the core hands to an edge, plus the slots the edge
    fills in (`usage`, `status_code`, `provider_message_id`) so the core can
    meter after the response completes."""

    model_tag: str            # registry tag (used for pricing + usage logs)
    provider: str             # "anthropic" | "openai" | "google"
    provider_model_id: str    # what the provider expects (e.g. "claude-opus-4-8")
    backend: str              # "first_party" | "bedrock" (config choice)
    trace_id: str
    api_key: Any
    headers: Mapping[str, str]

    usage: CanonicalUsage | None = None
    status_code: int = 0
    provider_message_id: str | None = None
    # Tenant connection serving (byok key or byom endpoint) — wallet debit is
    # skipped (their tokens, their bill) and provider auth failures get
    # connection attribution instead of a generic error. conn_kind None means
    # deployment-served.
    conn_kind: str | None = None  # "byok" | "byom" | None
    conn_id: int | None = None
    # byom only: declared-price USD estimate inputs (set at planning time).
    byom_prices: tuple[Any, Any] | None = None  # (input_per_m, output_per_m) Decimals
    # Set by _meter for connection-served requests: the informative estimate
    # (byok: CU at list price; byom: USD at declared prices) for response headers.
    conn_estimate: str | None = None
    # Streaming only: ms from request start to the FIRST provider chunk —
    # perceived latency for streaming clients (total latency_ms hides it).
    ttft_ms: int | None = None
    # True when the provider spent the token budget without producing any visible
    # output (reasoning-model empty completion). Metered for observability.
    empty_completion: bool = False
    # Registry `capabilities` of the served model (effort_levels etc.).
    capabilities: Any = None
    cc_metadata: dict[str, Any] = field(default_factory=dict)

    def report(
        self,
        usage: CanonicalUsage | None,
        status_code: int,
        provider_message_id: str | None = None,
    ) -> None:
        self.usage = usage
        self.status_code = status_code
        if provider_message_id:
            self.provider_message_id = provider_message_id


@dataclass
class EdgeResult:
    """Non-streaming result: faithful Anthropic JSON bytes + status."""

    status_code: int
    body: bytes
    media_type: str = "application/json"


@runtime_checkable
class ProviderEdge(Protocol):
    """Implemented per provider. The core calls exactly one of these per request
    depending on `req.stream`."""

    async def invoke(self, req: ValidatedRequest, ctx: RequestContext) -> EdgeResult:
        """Non-streaming. Return faithful Anthropic JSON; call ctx.report(...)."""
        ...

    def stream(self, req: ValidatedRequest, ctx: RequestContext) -> AsyncIterator[bytes]:
        """Streaming. Yield raw Anthropic-SSE byte chunks; call ctx.report(...)
        when usage is known (typically message_start + message_delta)."""
        ...
