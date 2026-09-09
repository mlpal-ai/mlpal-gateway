"""v2 universal core for /v2/messages.

Owns everything cross-cutting so every provider edge behaves identically:
model resolution (ModelRouter + allowlist), SSE transport + heartbeat, usage→CU,
wallet debit, and the Anthropic error envelope. Edges only produce Anthropic
wire bytes and report usage (see edges.py). v2-A wires the Anthropic edge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from decimal import Decimal
from typing import Any

from fastapi.responses import Response, StreamingResponse

from mlpal_assistants_service.adapters.factory import get_adapter_factory
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    BudgetExceededError,
    ModelAccessDeniedError,
    ModelNotAvailableError,
    ModelNotFoundError,
    RateLimitExceededError,
)
from mlpal_assistants_service.core.metrics import get_metrics
from mlpal_assistants_service.seams.billing import build_billing_gate, is_insufficient_wallet_error
from mlpal_assistants_service.services.messages_v2.anthropic_backend import get_anthropic_backend
from mlpal_assistants_service.services.messages_v2.anthropic_edge import AnthropicEdge
from mlpal_assistants_service.services.messages_v2.edges import ProviderEdge, RequestContext
from mlpal_assistants_service.services.messages_v2.errors import error_body
from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest
from mlpal_assistants_service.services.messages_v2.translating_edge import TranslatingEdge

logger = logging.getLogger(__name__)

OPERATION = "chat"  # matches existing ModelPricing rows

# Providers that have a /v2/messages edge: anthropic native passthrough;
# openai, google, and bedrock (open-weight lineup) via the translating edge.
SERVED_PROVIDERS = frozenset({"anthropic", "openai", "google", "bedrock"})

# Sentinel: an allowlist of ["*"] admits any served chat model (GA default),
# instead of an explicit per-tag pin list.
ALLOWLIST_WILDCARD = "*"

# Strong refs for fire-and-forget tasks (debit/capture) — the event loop keeps
# only weak references, so an unreferenced task can be GC'd mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def is_served_chat_model(model: Any) -> bool:
    """True if a resolved model can be served on /v2/messages under the wildcard
    policy: a chat-operation model on a provider that has an edge. Excludes
    image/embedding/tts/transcription models and unsupported providers (which
    would otherwise fail confusingly inside an adapter)."""
    if model.provider not in SERVED_PROVIDERS:
        return False
    caps = model.capabilities if isinstance(model.capabilities, dict) else {}
    return caps.get("operation", "chat") == "chat"


from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class _ByomModel:
    """Stand-in for a ModelRegistry row when a tenant model serves the
    request — carries exactly the fields the request path reads. Tenant
    models never touch the registry or its caches."""

    model_tag: str
    provider: str
    provider_model_id: str


class ModelNotAllowed(Exception):
    """Model isn't on the v2 dev allowlist."""


class MessagesV2Core:
    def __init__(
        self,
        router: Any,
        usage_service: Any,
        pricing_service: Any,
        billing_gate: Any,
        *,
        rate_limiter: Any = None,
        policy: Any = None,
    ) -> None:
        self._router = router
        self._usage = usage_service
        self._pricing = pricing_service
        self._billing = billing_gate  # seams.billing.BillingGate (local or managed)
        self._rate_limiter = rate_limiter
        self._policy = policy
        self._settings = get_settings()

    # -- admission policy ---------------------------------------------------
    def _model_allowed(self, model: Any) -> bool:
        """Wildcard (["*"]) admits any served chat model; otherwise the model
        tag must be explicitly pinned in the allowlist."""
        allow = self._settings.messages_v2_allowlist
        if ALLOWLIST_WILDCARD in allow:
            return is_served_chat_model(model)
        return model.model_tag in allow

    # -- edge selection -----------------------------------------------------
    def _backend_label(self, model) -> str:
        """Observability label: which backend serves this request — the
        native backend's name on the passthrough path, else the resolved
        adapter's backend_name (first_party / azure / vertex / bedrock)."""
        if model.provider == "anthropic":
            try:
                backend = get_anthropic_backend(self._settings)
                if backend.serves(model.provider_model_id):
                    return backend.name
            except ValueError:
                pass
        try:
            adapter, _ = get_adapter_factory().resolve(
                model.provider, model.provider_model_id
            )
            return getattr(adapter, "backend_name", "first_party")
        except (ValueError, RuntimeError):
            return "unresolved"

    def _edge_for(self, model) -> ProviderEdge:
        provider = model.provider
        if provider == "anthropic":
            # Native path (byte-faithful) when a native backend serves this
            # model: first-party serves everything; bedrock-mantle only its
            # allowlist. Otherwise fall back to the adapter path — the factory
            # priority picks the serving backend (bedrock SDK, vertex).
            try:
                backend = get_anthropic_backend(self._settings)
            except ValueError:
                backend = None
            if backend is not None and backend.serves(model.provider_model_id):
                return AnthropicEdge(backend)
            return self._translating_edge(provider, model.provider_model_id)
        if provider in ("openai", "google", "bedrock"):
            # Same translating edge for all three: Anthropic surface ↔
            # OpenAI-common ↔ provider adapter ↔ Anthropic wire (see
            # translating_edge.py). For bedrock this serves the open-weight
            # catalog (Converse API) — the factory resolves the serving
            # backend exactly as on the OpenAI wire.
            return self._translating_edge(provider, model.provider_model_id)
        raise ModelNotAllowed(f"provider '{provider}' not yet served by /v2/messages")

    @staticmethod
    def _translating_edge(provider: str, provider_model_id: str) -> TranslatingEdge:
        try:
            adapter, wire_id = get_adapter_factory().resolve(provider, provider_model_id)
        except (ValueError, RuntimeError) as e:
            raise ModelNotAllowed(str(e))
        return TranslatingEdge(adapter, wire_model_id=wire_id)

    # -- request handling ---------------------------------------------------
    async def handle(
        self,
        req: ValidatedRequest,
        api_key: Any,
        headers: Mapping[str, str],
        trace_id: str,
        *,
        surface: str = "v1_messages",
    ) -> Response:
        self._surface = surface
        if req.fallback_models:
            return await self._handle_with_fallback(
                req, api_key, headers, trace_id, surface=surface
            )
        # byom: `user/…` tags are tenant models — resolved through the
        # per-user overlay, never the shared model caches or the catalog.
        # Unknown / inactive / feature-off reads as model-not-found, exactly
        # like an unknown catalog tag.
        byom = None  # (adapter, wire_model_id, TenantModelRef)
        if req.model.startswith("user/"):
            from mlpal_assistants_service.services import connections as conn_svc

            byom = await conn_svc.resolve_tenant_model(
                api_key.user_id, self._router.session, req.model
            )
            if byom is None:
                return Response(
                    error_body(404, f"model '{req.model}' not found"),
                    404,
                    media_type="application/json",
                )
            model = _ByomModel(
                model_tag=req.model,
                provider=byom[2].conn.family,
                provider_model_id=byom[1],
            )
        else:
            try:
                # Resolve mlpal meta-models (mlpal / mlpal-turbo / mlpal-lite) to the
                # concrete chat model before admission + routing, exactly as /v1/chat
                # does — otherwise get_model(meta) 404s.
                resolved_tag, _routing = await self._router.resolve_meta_model(req.model, OPERATION)
                model = await self._router.get_model(resolved_tag)
            except ModelNotFoundError:
                return Response(error_body(404, f"model '{req.model}' not found"), 404, media_type="application/json")
            except ModelNotAvailableError as e:
                return Response(error_body(503, str(e)), 503, media_type="application/json")

            if not self._model_allowed(model):
                return Response(
                    content=error_body(404, f"model '{req.model}' is not available on /v2/messages"),
                    status_code=404,
                    media_type="application/json",
                )

        # Admission — the SAME pipeline as /v1/chat (rate limit → billing gate →
        # per-key policy), mapped to the Anthropic error envelope. This surface
        # is the primary prod endpoint; a key blocked on /v1/chat must be
        # equally blocked here.
        try:
            # Rate limit and billing gate are independent Redis reads — run them
            # concurrently (the rate limiter never touches the SQL session, so
            # this pair is safe even on the billing gate's DB-miss path).
            # Evaluation order below is unchanged: 429 outranks 403.
            billing_coro = self._billing.can_make_request_cached(api_key.user_id)
            if self._rate_limiter:
                rate_result, billing_result = await asyncio.gather(
                    self._rate_limiter.check_request_limit(
                        str(api_key.user_id), getattr(api_key, "rate_limit_tier", None)
                    ),
                    billing_coro,
                    return_exceptions=True,
                )
                if isinstance(rate_result, BaseException):
                    raise rate_result
            else:
                billing_result = await billing_coro
            if isinstance(billing_result, BaseException):
                raise billing_result
            can_request, block_reason, _billing_existed = billing_result
            if not can_request:
                # Wallet-empty is actionable and distinct: 402 + billing_error,
                # so clients can render a top-up prompt instead of a generic 403.
                from mlpal_assistants_service.repositories.billing_repository import (
                    WALLET_EMPTY_MESSAGE,
                )

                blocked_status = 402 if block_reason == WALLET_EMPTY_MESSAGE else 403
                return Response(
                    error_body(blocked_status, block_reason or "API access blocked"),
                    blocked_status,
                    media_type="application/json",
                )
            if self._policy is not None:
                self._policy.check_model_access(
                    getattr(api_key, "model_policy", None),
                    requested=req.model,
                    resolved=model.model_tag,
                )
                await self._policy.check_budgets(
                    api_key.id, getattr(api_key, "budgets", None)
                )
        except RateLimitExceededError as e:
            return Response(error_body(429, str(e)), 429, media_type="application/json")
        except (ModelAccessDeniedError, BudgetExceededError) as e:
            return Response(error_body(403, str(e)), 403, media_type="application/json")

        cc_metadata = _cc_metadata(headers, req.metadata)
        cc_metadata["stream"] = bool(req.stream)
        if model.model_tag != req.model:  # served via a meta-model alias
            cc_metadata["requested_model"] = req.model
        # byok: tenant credentials (if any) outrank deployment credentials for
        # this model's family. Never raises — a broken tenant credential
        # degrades to the deployment path. Skipped for byom-served requests
        # (already on a tenant connection).
        tenant_plan = None
        if byom is None and getattr(self._settings, "connections_enabled", False):
            from mlpal_assistants_service.services import connections as conn_svc

            try:
                tenant_plan = await conn_svc.plan_tenant_serving(
                    api_key.user_id, self._router.session, model
                )
            except conn_svc.ConnectionBlocked as e:
                # Their key is invalid and they opted out of billed fallback.
                return Response(
                    error_body(502, str(e)), 502, media_type="application/json"
                )

        ctx = RequestContext(
            # Resolved concrete tag drives pricing, usage_logs, and the response
            # `model` field (what actually served the request).
            model_tag=model.model_tag,
            provider=model.provider,
            provider_model_id=model.provider_model_id,
            capabilities=getattr(model, "capabilities", None),
            backend=(
                "byom:custom"
                if byom is not None
                else f"byok:{tenant_plan[3].backend}"
                if tenant_plan
                else self._backend_label(model)
            ),
            trace_id=trace_id,
            api_key=api_key,
            headers=headers,
            cc_metadata=cc_metadata,
            conn_kind=("byom" if byom else "byok" if tenant_plan else None),
            conn_id=(
                byom[2].conn.id if byom else tenant_plan[3].id if tenant_plan else None
            ),
            byom_prices=(
                (byom[2].input_price_per_m, byom[2].output_price_per_m) if byom else None
            ),
        )
        try:
            if byom is not None:
                edge = TranslatingEdge(byom[0], wire_model_id=byom[1])
            elif tenant_plan is not None:
                kind, obj, wire_id, _cred = tenant_plan
                if kind == "native":
                    edge = AnthropicEdge(obj)
                else:
                    edge = TranslatingEdge(obj, wire_model_id=wire_id)
            else:
                edge = self._edge_for(model)
        except ModelNotAllowed as e:
            return Response(error_body(400, str(e)), 400, media_type="application/json")

        t0 = time.perf_counter()
        if req.stream:
            return StreamingResponse(
                self._stream_with_heartbeat(edge, req, ctx, t0),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        result = await edge.invoke(req, ctx)
        compute_units = await self._meter(ctx, int((time.perf_counter() - t0) * 1000))
        # Opt-in payload capture — hard-off keys skip even the task spawn;
        # enabled-check + key resolution + zlib all happen inside the task.
        if _key_capture_possible(ctx):
            _spawn(
                _capture_v2(
                    ctx.trace_id, req.body, result.body,
                    getattr(self._usage, "redis", None),
                    capture_policy=_key_capture_policy(ctx),
                    requested_model=req.model,
                    resolved_model=ctx.model_tag,
                )
            )
        # Surface CU on the Anthropic-wire response via headers (the body stays a
        # faithful Anthropic Messages object). Streaming can't do this — headers
        # are sent before the CU is known — so streamed requests are billing-only.
        if ctx.conn_kind and result.status_code in (401, 403):
            # THEIR credential was rejected — say so (a raw provider 401
            # reads as "my MLPal key is broken") and flip the connection so
            # the console shows it and serving falls back where possible.
            self._flag_conn_rejected(ctx)
            return Response(
                error_body(
                    502,
                    "Your provider credential was rejected by the provider "
                    "(connection_rejected). Update it in Settings → "
                    "Connections; catalog models fall back to MLPal's keys "
                    "meanwhile.",
                ),
                status_code=502,
                media_type="application/json",
            )
        return Response(
            content=result.body,
            status_code=result.status_code,
            media_type=result.media_type,
            headers={**_cu_headers(compute_units), **_conn_headers(ctx), **_effort_headers(ctx)},
        )

    # Serving failures worth advancing to the next fallback candidate for.
    # Deliberately excludes 4xx: client errors (bad request, kwargs rejection,
    # policy/billing denials, our own rate limits) repeat identically on every
    # candidate — retrying them just burns admission checks.
    _FALLBACK_STATUSES = frozenset({500, 502, 503, 504, 529})

    async def _handle_with_fallback(
        self,
        req: ValidatedRequest,
        api_key: Any,
        headers: Mapping[str, str],
        trace_id: str,
        *,
        surface: str,
    ) -> Response:
        """Client-controlled model failover: candidates run the FULL pipeline
        (admission, policy, metering) and are billed as-served. Non-streaming
        failures (and streaming failures that die BEFORE the stream opens —
        those return plain error Responses) advance the chain; an open stream
        is committed. Each attempt is metered under its own model with its
        own error status, so failovers are visible in usage_logs."""
        candidates = [req.model]
        for tag in req.fallback_models or []:
            if tag not in candidates:
                candidates.append(tag)
        last: Response | None = None
        for i, tag in enumerate(candidates):
            attempt = ValidatedRequest(
                raw_body=req.raw_body,
                body={**req.body, "model": tag},
                model=tag,
                stream=req.stream,
                metadata=req.metadata,
                model_kwargs=req.model_kwargs,
                fallback_models=None,
            )
            resp = await self.handle(
                attempt, api_key, headers, trace_id, surface=surface
            )
            if isinstance(resp, StreamingResponse):
                # the stream opened — committed to this candidate
                if i > 0:
                    resp.headers["X-MLPal-Fallback-From"] = req.model
                return resp
            # 404 on a candidate = misspelled/retired fallback tag — skip it
            # (the PRIMARY's 404 is only skippable when fallbacks exist, which
            # is exactly this loop).
            if resp.status_code in self._FALLBACK_STATUSES or (
                resp.status_code == 404 and i < len(candidates) - 1
            ):
                logger.warning(
                    f"[v2.messages] fallback: {tag} -> HTTP {resp.status_code}, "
                    f"trying next (trace={trace_id})"
                )
                last = resp
                continue
            if i > 0:
                resp.headers["X-MLPal-Fallback-From"] = req.model
            return resp
        return last  # every candidate failed — surface the final failure

    # -- streaming transport (heartbeat) ------------------------------------
    async def _stream_with_heartbeat(
        self, edge: ProviderEdge, req: ValidatedRequest, ctx: RequestContext, t0: float
    ) -> AsyncIterator[bytes]:
        """Pipe the edge's raw Anthropic-SSE bytes through, emitting `: ping`
        keepalives during silence so long reasoning/tool phases can't trip the
        client/ALB idle timeouts. Producer/queue decouples the heartbeat from
        the edge generator (same pattern as /v1/chat); meter after completion."""
        interval = self._settings.messages_v2_heartbeat_interval
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        # Accumulate the provider's SSE bytes for opt-in payload capture (the
        # non-streaming path captures result.body; without this, streaming
        # clients — yodex streams everything — never produce payloads). Hard
        # cap bounds memory; capture_payload truncates to max_body_kb anyway.
        capture_buf = bytearray()
        capture_cap = 1024 * 1024

        async def _produce() -> None:
            try:
                async for chunk in edge.stream(req, ctx):
                    await queue.put(("chunk", chunk))
            except Exception as e:  # noqa: BLE001
                await queue.put(("error", e))
            finally:
                await queue.put(("end", None))

        last_chunk = time.monotonic()

        async def _keepalive() -> None:
            # Idle-ping injector: replaces a wait_for timer per chunk (a timer
            # handle + TimeoutError control flow on EVERY event) with one task
            # that only wakes at the heartbeat interval.
            while True:
                await asyncio.sleep(interval)
                if time.monotonic() - last_chunk >= interval:
                    await queue.put(("ping", None))

        producer = asyncio.create_task(_produce())
        keepalive = asyncio.create_task(_keepalive())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "ping":
                    if time.monotonic() - last_chunk >= interval:
                        yield b": ping\n\n"
                    continue
                if kind == "chunk":
                    last_chunk = time.monotonic()
                    if ctx.ttft_ms is None:
                        ctx.ttft_ms = int((time.perf_counter() - t0) * 1000)
                    if len(capture_buf) < capture_cap:
                        capture_buf += payload[: capture_cap - len(capture_buf)]
                    yield payload
                elif kind == "error":
                    logger.error(f"[v2.messages] stream error trace={ctx.trace_id}: {payload}")
                    # Mid-stream provider failure: emit an Anthropic error event.
                    yield b"event: error\ndata: " + error_body(502, "upstream stream error") + b"\n\n"
                    break
                else:  # end
                    break
        finally:
            keepalive.cancel()
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            await self._meter(ctx, int((time.perf_counter() - t0) * 1000))
            if _key_capture_possible(ctx):
                _spawn(
                    _capture_v2(
                        ctx.trace_id, req.body, bytes(capture_buf),
                        getattr(self._usage, "redis", None),
                        capture_policy=_key_capture_policy(ctx),
                        requested_model=req.model,
                        resolved_model=ctx.model_tag,
                    )
                )

    def _flag_conn_rejected(self, ctx: RequestContext) -> None:
        if ctx.conn_id is None:
            return

        async def _mark() -> None:
            from mlpal_assistants_service.db.session import async_session_factory
            from mlpal_assistants_service.services import connections as conn_svc

            async with async_session_factory() as session:
                await conn_svc.mark_invalid(
                    session, ctx.conn_id, "provider rejected the credential (401/403)"
                )

        _spawn(_mark())

    # -- telemetry + billing (never raises) ---------------------------------
    async def _meter(self, ctx: RequestContext, latency_ms: int) -> Decimal:
        is_success = ctx.status_code == 200 and ctx.usage is not None
        if ctx.conn_kind and ctx.status_code in (401, 403):
            self._flag_conn_rejected(ctx)
        input_tokens = ctx.usage.input if is_success else 0
        output_tokens = ctx.usage.output if is_success else 0
        # Billed CU = the model's pass-through cost, no markup (locked pricing:
        # flat tier fee is the revenue; tokens at cost). Stored rates carry the
        # legacy markup column, so divide it back out.
        compute_units = Decimal("0")
        byom_usd = None
        if is_success and ctx.conn_kind == "byom":
            # user/ models have no catalog pricing — estimate at the user's
            # declared per-1M prices (their visibility only, never billed).
            inp, outp = ctx.byom_prices or (Decimal("0"), Decimal("0"))
            byom_usd = (
                Decimal(input_tokens) * inp + Decimal(output_tokens) * outp
            ) / Decimal(1_000_000)
        elif is_success:
            rates = await self._resolve_cu_rates(ctx.model_tag)
            if rates is None:
                logger.warning(f"[v2.messages] no pricing for {ctx.model_tag}; CU=0 trace={ctx.trace_id}")
            else:
                in_cu, out_cu, cache_cu = rates
                if cache_cu is None:
                    # No per-model cache-read list price: the provider's
                    # standard multiple applies (google 0.25x, else 0.10x).
                    from mlpal_assistants_service.services.pricing import (
                        provider_cache_read_multiplier,
                    )

                    cache_cu = in_cu * provider_cache_read_multiplier(ctx.provider)
                compute_units = ctx.usage.compute_units(in_cu, out_cu, cache_cu)

        logger.info(
            f"[v2.messages] trace={ctx.trace_id} model={ctx.model_tag} provider={ctx.provider} "
            f"backend={ctx.backend} status={ctx.status_code} latency_ms={latency_ms} "
            f"usage={json.dumps(ctx.usage.raw) if ctx.usage else None} cu={compute_units}"
        )
        # Reasoning-budget empty completion: the turn "succeeded" (200) but the
        # model produced no visible output because reasoning consumed max_tokens.
        # We surface stop_reason=max_tokens faithfully; here we make it observable
        # (distinct log + metric + a usage_log flag) so the reliability property is
        # monitorable, not just anecdotal. The client sees a clear signal to retry
        # with a larger budget — we never silently inflate it (that would surprise
        # and over-bill the caller).
        if is_success and ctx.empty_completion:
            logger.warning(
                f"[v2.messages] empty_completion (reasoning budget exhausted) "
                f"trace={ctx.trace_id} model={ctx.model_tag} provider={ctx.provider} "
                f"output_tokens={output_tokens}"
            )
            get_metrics().put_metric_sync(
                "EmptyCompletion", 1, dimensions={"provider": ctx.provider, "model": ctx.model_tag}
            )
        try:
            await self._usage.record_usage(
                user_id=str(ctx.api_key.user_id),
                api_key_id=str(ctx.api_key.id),
                trace_id=ctx.trace_id,
                model_tag=ctx.model_tag,
                provider=ctx.provider,
                operation=OPERATION,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                # Connection-served: billable CU is ZERO — the metered
                # estimate goes to metadata only. Every downstream aggregator
                # (cu aggregates, effective balance, invoicing) sums this
                # column, and their tokens are already paid on their bill.
                compute_units=Decimal("0") if ctx.conn_kind else compute_units,
                latency_ms=latency_ms,
                status="success" if is_success else "error",
                error_code=None if is_success else f"http_{ctx.status_code}",
                # "pending" until _post_billing resolves it (debited / failed_* /
                # not_applicable) — same lifecycle as /v1/chat, so DebitRetryWorker
                # and reconciliation see this surface identically.
                wallet_debit_status=(
                    "pending"
                    if (is_success and compute_units > 0 and not ctx.conn_kind)
                    # Connection-served: their tokens, their bill — the wallet
                    # is never touched. Token metering above still runs (tier
                    # thresholds count tokens routed, not tokens paid).
                    else "not_applicable"
                ),
                # `api` tags which mount served the call (v1_messages vs the
                # deprecated v2_messages alias) — drives the alias-drain query.
                cc_metadata={
                    **ctx.cc_metadata,
                    "api": getattr(self, "_surface", "v1_messages"),
                    # Which serving backend took the call (first_party /
                    # bedrock / azure / vertex / adapter) — queryable per
                    # trace and shown in the console trace detail.
                    "serving_backend": ctx.backend,
                    **(
                        {
                            "serving_credentials": ctx.conn_kind,
                            # Informative cost estimate (never billed by
                            # MLPal): byok = catalog list price in CU,
                            # byom = user-declared prices in USD.
                            **(
                                {"connection_usd_estimate": str(byom_usd)}
                                if ctx.conn_kind == "byom"
                                else {"connection_cu_estimate": str(compute_units)}
                            ),
                        }
                        if ctx.conn_kind
                        else {}
                    ),
                    "provider_message_id": ctx.provider_message_id,
                    # Cache observability: input_tokens above CONTAINS the cached
                    # portion; these make prompt-cache effectiveness queryable
                    # (cc_metadata->>'cache_read_input_tokens') per trace.
                    **(
                        {
                            "cache_read_input_tokens": ctx.usage.cache_read,
                            "cache_creation_input_tokens": ctx.usage.cache_write,
                        }
                        if is_success and ctx.usage is not None
                        else {}
                    ),
                    **({"ttft_ms": ctx.ttft_ms} if ctx.ttft_ms is not None else {}),
                    **({"empty_completion": True} if ctx.empty_completion else {}),
                },
            )
        except Exception:  # pragma: no cover - telemetry must never 500 the request
            logger.exception(f"[v2.messages] record_usage failed trace={ctx.trace_id}")

        if is_success and compute_units > 0 and not ctx.conn_kind:
            _spawn(self._post_billing(ctx, compute_units, input_tokens + output_tokens))

        # The caller-facing cost is the BILLED truth: zero for connection-
        # served requests (their key, their bill). The informative estimate
        # travels separately (ctx.conn_estimate → response headers).
        if ctx.conn_kind:
            ctx.conn_estimate = str(byom_usd) if ctx.conn_kind == "byom" else str(compute_units)
            return Decimal("0")
        return compute_units

    async def _resolve_cu_rates(
        self, model_tag: str
    ) -> tuple[Decimal, Decimal, Decimal | None] | None:
        """Per-token CU rates at PASS-THROUGH: the stored cu_rates include the
        row's legacy markup_multiplier (generated columns), so divide the row's
        own markup back out — the same source of truth the v1 chat path uses
        (ModelPricing.calculate_provider_cost). Never a config knob: a setting
        that must mirror per-row DB state is a correctness trap (caught live:
        OSS rows carry markup 1.0 while the old setting said 3.0 → 3× under-
        billing on /v1/messages vs /v1/chat for identical usage)."""
        pricing = await self._pricing.get_pricing(model_tag, OPERATION)
        if pricing is None:
            return None
        divisor = Decimal("1000") if pricing.rate_unit == "per_1k_tokens" else Decimal("1000000")
        markup = pricing.markup_multiplier or Decimal("1")
        # Per-model cache-read rate (list $/unit → CU/token, no markup — it is
        # stored as the raw list price). None = global multiplier applies.
        cache_read = getattr(pricing, "cache_read_rate", None)
        cache_read_cu = (
            cache_read / divisor / (pricing.cu_to_dollar or Decimal("10"))
            if cache_read is not None
            else None
        )
        return (
            pricing.input_cu_rate / divisor / markup,
            pricing.output_cu_rate / divisor / markup,
            cache_read_cu,
        )

    async def _post_billing(self, ctx: RequestContext, compute_units: Decimal, total_tokens: int) -> None:
        """Post-response billing, mirroring /v1/chat's background flow: gated
        wallet debit with an honest wallet_debit_status lifecycle (pending →
        debited / failed_retryable / failed_permanent / not_applicable), token
        rate-limit accrual, and per-key budget accrual. Runs on a fresh session —
        the request session is closed by the time this task executes."""
        from mlpal_assistants_service.db.session import async_session_factory
        from mlpal_assistants_service.services.usage import UsageService

        redis = getattr(self._usage, "redis", None)
        try:
            async with async_session_factory() as bg_session:
                gate = build_billing_gate(bg_session, redis)
                bg_usage = UsageService(bg_session, redis)
                if not await gate.is_wallet_debit_active():
                    # Gating off (or local billing): the ledger derives spend from
                    # usage rows; debiting the wallet too would double-count.
                    await bg_usage.mark_wallet_debit_status(ctx.trace_id, "not_applicable")
                else:
                    try:
                        await gate.debit_wallet_usage(
                            user_id=ctx.api_key.user_id,
                            compute_units=compute_units,
                            usage_ref=ctx.trace_id,
                        )
                    except Exception as debit_error:  # noqa: BLE001 — classified below
                        await bg_usage.mark_wallet_debit_status(
                            ctx.trace_id,
                            (
                                "failed_permanent"
                                if is_insufficient_wallet_error(debit_error)
                                else "failed_retryable"
                            ),
                            error=str(debit_error),
                        )
                    else:
                        await bg_usage.mark_wallet_debit_status(ctx.trace_id, "debited")
                await bg_session.commit()

            if self._rate_limiter:
                await self._rate_limiter.record_tokens(str(ctx.api_key.user_id), total_tokens)
            if self._policy is not None:
                await self._policy.record_key_usage(
                    ctx.api_key.id, getattr(ctx.api_key, "budgets", None), compute_units
                )
        except Exception:  # pragma: no cover — billing telemetry must never crash the loop
            logger.exception(
                f"[v2.messages] post-billing failed user={ctx.api_key.user_id} trace={ctx.trace_id}"
            )


def _key_capture_policy(ctx: RequestContext) -> dict | None:
    return getattr(ctx.api_key, "capture_policy", None)


def _key_capture_possible(ctx: RequestContext) -> bool:
    """Pre-spawn short-circuit: a hard {"mode": "off"} key pays nothing,
    not even the capture task spawn."""
    policy = _key_capture_policy(ctx)
    return not (isinstance(policy, dict) and policy.get("mode") == "off")


async def _capture_v2(
    trace_id: str,
    request_body: Any,
    response_body: bytes | str,
    redis: Any,
    capture_policy: dict | None = None,
    requested_model: str | None = None,
    resolved_model: str | None = None,
) -> None:
    """Background capture for the universal messages core."""
    try:
        from mlpal_assistants_service.services.capture import (
            capture_payload,
            capture_state,
            key_allows_capture,
        )

        cfg = await capture_state.config(redis)
        if not cfg.enabled:
            return
        if not key_allows_capture(
            capture_policy, cfg.key_default, requested_model, resolved_model
        ):
            return
        body = (
            response_body.decode("utf-8", "replace")
            if isinstance(response_body, bytes)
            else response_body
        )
        await capture_payload(trace_id, request_body, body, cfg.max_body_kb)
    except Exception:  # noqa: BLE001 — debug data must never raise
        logger.exception(f"[v2.messages] capture failed trace={trace_id}")


def _cu_headers(compute_units: Decimal) -> dict[str, str]:
    """CU envelope for the Anthropic-wire response (body stays untouched)."""
    return {"X-MLPal-Compute-Units": str(compute_units)}


def _effort_headers(ctx: RequestContext) -> dict[str, str]:
    """Never-silent effort resolution on the translating edge (body stays pure
    Anthropic): `requested->applied`, e.g. `max->high` when clamped."""
    res = ctx.cc_metadata.get("reasoning_effort")
    if not isinstance(res, dict) or not res.get("requested"):
        return {}
    return {"X-MLPal-Reasoning-Effort": f"{res['requested']}->{res.get('applied')}"}


def _conn_headers(ctx: RequestContext) -> dict[str, str]:
    """Connection-served responses: billed CU is 0 (in X-MLPal-Compute-Units);
    these headers say WHY and what the tokens were worth — byok in CU at
    catalog list price, byom in USD at the user's declared prices."""
    if not ctx.conn_kind:
        return {}
    out = {"X-MLPal-Serving-Credentials": ctx.conn_kind}
    if ctx.conn_estimate is not None:
        key = (
            "X-MLPal-Connection-Usd-Estimate"
            if ctx.conn_kind == "byom"
            else "X-MLPal-Connection-Cu-Estimate"
        )
        out[key] = ctx.conn_estimate
    return out


def _cc_metadata(headers: Mapping[str, str], metadata: dict[str, Any]) -> dict[str, Any]:
    """Claude-Code / Anthropic attribution signals for usage logs."""
    return {
        "cc_session_id": headers.get("x-claude-code-session-id"),
        "cc_agent_id": headers.get("x-claude-code-agent-id"),
        "cc_parent_agent_id": headers.get("x-claude-code-parent-agent-id"),
        "anthropic_user_id": metadata.get("user_id") if isinstance(metadata, dict) else None,
        "anthropic_version_header": headers.get("anthropic-version"),
        "anthropic_beta_header": headers.get("anthropic-beta"),
    }
