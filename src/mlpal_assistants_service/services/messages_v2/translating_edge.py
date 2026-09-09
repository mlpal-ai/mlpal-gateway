"""Translating provider edge — non-Anthropic providers on the Anthropic surface.

The single edge that serves OpenAI (v2-B) and Google (v2-C): inbound Anthropic
Messages → OpenAI-common (``translate_in``) → existing provider adapter →
Anthropic wire (``emit``). One translation in, one translation out, the proven
adapters reused unchanged. The Anthropic edge does NOT use this path (it's native
passthrough); this edge is exactly the converging seam the architecture promised.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from mlpal_assistants_service.adapters.base import (
    BaseAdapter,
    ToolDefinition,
    UnsupportedModalityError,
)
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    UnsupportedCapabilityError,
    UnsupportedModelKwargsError,
)
from mlpal_assistants_service.services.messages_v2 import emit
from mlpal_assistants_service.services.messages_v2.edges import EdgeResult, RequestContext
from mlpal_assistants_service.services.messages_v2.errors import error_body
from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest
from mlpal_assistants_service.services.messages_v2.translate_in import (
    CommonRequest,
    cache_control_ttl,
    sanitize_google_tool_schema,
    to_common,
)
from mlpal_assistants_service.services.messages_v2.usage import CanonicalUsage
from mlpal_assistants_service.services.reasoning_effort import (
    check_no_native_conflict,
    resolve_effort,
)

logger = logging.getLogger(__name__)


def _message_id(trace_id: str) -> str:
    return f"msg_{trace_id.replace('-', '')}"


def _error_status(exc: Exception) -> int:
    if isinstance(
        exc,
        (UnsupportedModalityError, UnsupportedCapabilityError, UnsupportedModelKwargsError),
    ):
        return 400
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else 502


class TranslatingEdge:
    def __init__(self, adapter: BaseAdapter, wire_model_id: str | None = None) -> None:
        self._adapter = adapter
        # Serving backends address models by their own IDs (Azure deployment
        # names, Bedrock inference profiles). When set, this overrides
        # ctx.provider_model_id on the adapter call — billing/observability
        # still use the catalog tag from ctx.
        self._wire_model_id = wire_model_id

    def _prepare(self, req: ValidatedRequest, ctx: RequestContext) -> tuple[CommonRequest, dict[str, Any]]:
        """Translate inbound once → (CommonRequest, adapter chat kwargs). Applies
        the Google-only tool-schema sanitization (P2) so a standard JSON-Schema
        `input_schema` (with additionalProperties/$schema) doesn't 400 on Gemini."""
        common = to_common(req.body)
        if ctx.provider == "google" and common.tools:
            common.tools = [
                ToolDefinition(
                    name=t.name,
                    description=t.description,
                    parameters=sanitize_google_tool_schema(t.parameters),
                    strict=t.strict,
                )
                for t in common.tools
            ]
        # Universal effort: explicit `output_config.effort` or the `thinking`
        # budget band, resolved onto the rungs this model accepts (clamped,
        # never silent — the resolution rides cc_metadata and a response header).
        check_no_native_conflict(common.reasoning_effort, req.model_kwargs)
        effort = resolve_effort(
            common.reasoning_effort, ctx.capabilities, model=ctx.model_tag
        )
        if effort.requested:
            ctx.cc_metadata["reasoning_effort"] = effort.as_metadata()
        # Model-specific kwargs: validated against the serving adapter —
        # rejected with a 400 listing the offenders, never silently dropped.
        self._adapter.validate_model_kwargs(req.model_kwargs)

        kwargs: dict[str, Any] = {
            "model": self._wire_model_id or ctx.provider_model_id,
            "messages": common.messages,
            "tools": common.tools,
            "tool_choice": common.tool_choice,
            "top_p": common.top_p,
            "stop": common.stop,
            "model_kwargs": req.model_kwargs,
            "reasoning_effort": effort.applied,
        }
        if common.max_tokens is not None:
            kwargs["max_tokens"] = common.max_tokens
        if common.temperature is not None:
            kwargs["temperature"] = common.temperature

        # Explicit prefix caching is Google-only (Anthropic passes cache_control
        # through; OpenAI caches automatically). Hint the adapter only when the
        # client marked a breakpoint and the flag is on; the adapter still decides
        # whether the prefix is big enough / tool mode allows it.
        settings = get_settings()
        if ctx.provider == "google" and settings.messages_v2_cache_google:
            ttl = cache_control_ttl(req.body, settings.messages_v2_cache_ttl_seconds)
            if ttl is not None:
                kwargs["prefix_cache_ttl"] = ttl
        return common, kwargs

    async def invoke(self, req: ValidatedRequest, ctx: RequestContext) -> EdgeResult:
        message_id = _message_id(ctx.trace_id)
        try:
            _, kwargs = self._prepare(req, ctx)
            resp = await self._adapter.chat(**kwargs)
            ctx.empty_completion = (
                not resp.content and not resp.tool_calls and resp.finish_reason == "length"
            )
        except Exception as e:  # noqa: BLE001 — map any provider failure to Anthropic envelope
            status = _error_status(e)
            ctx.report(None, status, message_id)
            logger.warning(f"[v2.messages] translate invoke error trace={ctx.trace_id} status={status}: {e}")
            return EdgeResult(status_code=status, body=error_body(status, str(e)))
        ctx.report(CanonicalUsage.from_openai(resp.usage), 200, message_id)
        return EdgeResult(
            status_code=200,
            body=emit.message_json(resp, message_id, ctx.model_tag),
        )

    async def stream(self, req: ValidatedRequest, ctx: RequestContext) -> AsyncIterator[bytes]:
        message_id = _message_id(ctx.trace_id)
        reported: dict[str, Any] = {}

        def _on_final(usage: Any, _stop_reason: str, empty_completion: bool) -> None:
            reported["usage"] = usage
            ctx.empty_completion = empty_completion
            ctx.report(CanonicalUsage.from_openai(usage) if usage else None, 200, message_id)

        try:
            _, kwargs = self._prepare(req, ctx)
            chunks = self._adapter.chat_stream(
                **kwargs, stream_thinking=get_settings().messages_v2_stream_thinking
            )
            async for sse_bytes in emit.stream_anthropic_sse(chunks, message_id, ctx.model_tag, _on_final):
                yield sse_bytes
        except Exception as e:  # noqa: BLE001
            status = _error_status(e)
            ctx.report(None, status, message_id)
            logger.warning(f"[v2.messages] translate stream error trace={ctx.trace_id} status={status}: {e}")
            yield b"event: error\ndata: " + error_body(status, str(e)) + b"\n\n"
            return
        # A clean stream that never surfaced usage still needs a billing report.
        if "usage" not in reported:
            ctx.report(None, 200, message_id)
