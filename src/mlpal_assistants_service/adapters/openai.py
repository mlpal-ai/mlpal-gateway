"""OpenAI provider adapter using native SDK.

Production-grade implementation following LangChain patterns:
- Multiple structured output methods (json_schema, function_calling, json_mode)
- Support for images, PDFs, and documents
- Proper error handling with include_raw option
- Retry mechanisms for validation failures
"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from mlpal_assistants_service.adapters.base import (
    AdapterResponse,
    AudioResponse,
    BaseAdapter,
    EmbeddingResponse,
    FileAttachment,
    FileSource,
    FileType,
    GeneratedImage,
    ImageGenerationResponse,
    ImageQuality,
    ImageSize,
    ImageSizeResolver,
    ModelCapabilities,
    StreamChunk,
    TokenUsage,
    ToolDefinition,
    UnsupportedModalityError,
    provider_status_code,
)
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.core.exceptions import (
    ProviderError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)

logger = logging.getLogger(__name__)


def _is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models (gpt-5.x and the o-series). Only these accept the
    Responses API `reasoning` param, so we gate reasoning-summary streaming on it."""
    return model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def _cache_write_tokens(usage: Any) -> int:
    """Prompt-cache WRITE tokens from a Responses usage object. OpenAI started
    reporting (and charging 1.25x for) cache writes on the gpt-5.6 / gpt-6
    generation — verified live 2026-09-07 on sol and astra. Absent → 0."""
    details = getattr(usage, "input_tokens_details", None) if usage else None
    return int(getattr(details, "cache_write_tokens", 0) or 0)


def _completions_cache_tokens(usage: Any) -> tuple[int, int]:
    """(cached, written) from a Chat Completions usage.prompt_tokens_details."""
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return (
        int(getattr(details, "cached_tokens", 0) or 0),
        int(getattr(details, "cache_write_tokens", 0) or 0),
    )


def _cached_tokens(usage: Any) -> int:
    """Cached (prompt-cache hit) input tokens from a Responses API usage object.

    Surfaced on TokenUsage so callers can report cache reads (e.g. the
    Anthropic-shaped /v2/messages usage block). Absent on older usage payloads,
    so this is defensively optional and defaults to 0.
    """
    details = getattr(usage, "input_tokens_details", None) if usage else None
    return int(getattr(details, "cached_tokens", 0) or 0)


# =============================================================================
# OpenAI Adapter
# =============================================================================

# Message keys the gateway understands but OpenAI-compatible servers do not.
_GATEWAY_ONLY_MESSAGE_KEYS = frozenset({"files", "images", "documents", "cache_control"})


class OpenAIAdapter(BaseAdapter):
    # Responses-API params we forward via model_kwargs (curated; the adapter
    # speaks the Responses wire, so chat-completions-only params like
    # logit_bias/seed are deliberately absent).
    SUPPORTED_KWARGS = frozenset(
        {
            "reasoning", "text", "service_tier", "metadata",
            "parallel_tool_calls", "store", "truncation", "include", "user",
            "safety_identifier", "prompt_cache_key", "top_logprobs",
            "max_tool_calls",
        }
    )

    # Wire dialect: "responses" (OpenAI first-party/Azure — the richer API)
    # or "chat_completions" (the de-facto standard served by vLLM, Ollama,
    # TGI, Together, …). byom custom endpoints get "chat_completions" —
    # most OpenAI-compatible servers don't implement /v1/responses.
    wire: str = "responses"

    """
    Production-grade adapter for OpenAI API using native SDK.

    Supports:
    - Chat completion (text)
    - Structured output with multiple methods:
        - json_schema: Native OpenAI structured outputs (strict mode)
        - function_calling: Tool-based extraction (most compatible)
        - json_mode: response_format: json_object (requires prompt engineering)
    - Vision (images via URL or base64)
    - Documents (PDFs via base64)
    - Tool/function calling
    - Streaming

    Models:
    - gpt-4o, gpt-4o-mini, gpt-5.2 (recommended)
    - gpt-4-turbo, gpt-4
    """

    provider_name = "openai"

    # Model capabilities registry
    MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
        # GPT-4o family - full multimodal support
        "gpt-4o": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,  # Use gpt-4o-audio-preview for audio
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=128000,
            max_output_tokens=16384,
        ),
        "gpt-4o-mini": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=128000,
            max_output_tokens=16384,
        ),
        # GPT-4o Audio - supports native audio input/output
        "gpt-4o-audio-preview": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=16384,
        ),
        # GPT-5 family
        # GPT-6 Astra (2026-09-03): 1.05M context, 128K output; tools,
        # structured output, streaming, image input probe-verified 2026-09-07.
        "gpt-6": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=1050000,
            max_output_tokens=128000,
        ),
        "gpt-5": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        "gpt-5.4": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        "gpt-5.2": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        "gpt-5-mini": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        "gpt-5-nano": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        # GPT-5 Pro - reasoning model with timeout issues for heavy capabilities
        "gpt-5-pro": ModelCapabilities(
            supports_images=False,  # Causes timeout due to reasoning overhead
            supports_pdf=False,  # Causes timeout due to reasoning overhead
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=False,  # Causes timeout due to reasoning overhead
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        # GPT-5.1/5.2 Codex - optimized for coding
        "gpt-5.1-codex": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        "gpt-5.2-codex": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=400000,
            max_output_tokens=128000,
        ),
        # O-series reasoning models
        "o3": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=100000,
        ),
        "o3-mini": ModelCapabilities(
            supports_images=False,  # o3-mini doesn't support vision
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=100000,
        ),
        "o4-mini": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=100000,
        ),
        # GPT-4.1 family - 1M context, vision support
        "gpt-4.1": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=1000000,
            max_output_tokens=32768,
        ),
        "gpt-4.1-mini": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=1000000,
            max_output_tokens=32768,
        ),
        "gpt-4.1-nano": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=1000000,
            max_output_tokens=32768,
        ),
        # GPT-4 Turbo
        "gpt-4-turbo": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,  # No native PDF support
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        ),
        # GPT-4 (original)
        "gpt-4": ModelCapabilities(
            supports_images=False,  # No vision
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=False,
            max_context_tokens=8192,
            max_output_tokens=4096,
        ),
        # GPT-3.5
        "gpt-3.5-turbo": ModelCapabilities(
            supports_images=False,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=False,
            max_context_tokens=16384,
            max_output_tokens=4096,
        ),
    }

    # Default capabilities for unknown models (assume GPT-4o-like)
    DEFAULT_CAPABILITIES = ModelCapabilities(
        supports_images=True,
        supports_pdf=True,
        supports_audio=False,
        supports_video=False,
        supports_tools=True,
        supports_structured_output=True,
        supports_mcp=True,
        max_context_tokens=128000,
        max_output_tokens=16384,
    )

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        if api_key:
            self._api_key = api_key
        else:
            settings = get_settings()
            self._api_key = settings.openai_api_key

        # Create client with connection pooling. Ceiling on simultaneous in-flight
        # requests to OpenAI across ALL callers — raised for client concurrency.
        # base_url seam lets serving backends (Azure /openai/v1) reuse this
        # adapter unchanged — the v1 surface is OpenAI-wire-compatible.
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._base_url = base_url
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            http_client=http_client,
        )

    # =========================================================================
    # Model Capabilities
    # =========================================================================

    def extract_files_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[FileAttachment]:
        """
        Extract all file attachments from messages.

        Handles both:
        - FileAttachment objects in `files` key
        - Legacy dict formats in `files`, `images`, and `documents` keys
        """
        files = []
        for msg in messages:
            # Handle files key (new format)
            msg_files = msg.get("files", [])
            for f in msg_files:
                if isinstance(f, FileAttachment):
                    files.append(f)
                elif isinstance(f, dict):
                    file_type_str = f.get("type", "other")
                    try:
                        file_type = FileType(file_type_str)
                    except ValueError:
                        file_type = FileType.OTHER

                    # Determine source
                    if f.get("base64"):
                        source = FileSource.BASE64
                        data = f["base64"]
                    elif f.get("url"):
                        source = FileSource.URL
                        data = f["url"]
                    elif f.get("file_id"):
                        source = FileSource.FILE_ID
                        data = f["file_id"]
                    elif f.get("path"):
                        source = FileSource.PATH
                        data = f["path"]
                    else:
                        data = f.get("data", "")
                        source = FileSource.BASE64

                    files.append(FileAttachment(
                        type=file_type,
                        source=source,
                        data=data,
                        mime_type=f.get("mime_type"),
                        filename=f.get("filename"),
                    ))

            # Handle legacy images key
            for img in msg.get("images", []):
                if img.get("url"):
                    source = FileSource.URL
                    data = img["url"]
                elif img.get("base64"):
                    source = FileSource.BASE64
                    data = img["base64"]
                else:
                    continue

                files.append(FileAttachment(
                    type=FileType.IMAGE,
                    source=source,
                    data=data,
                    mime_type=img.get("mime_type"),
                ))

            # Handle legacy documents key
            for doc in msg.get("documents", []):
                if doc.get("url"):
                    source = FileSource.URL
                    data = doc["url"]
                elif doc.get("base64"):
                    source = FileSource.BASE64
                    data = doc["base64"]
                elif doc.get("file_id"):
                    source = FileSource.FILE_ID
                    data = doc["file_id"]
                else:
                    continue

                files.append(FileAttachment(
                    type=FileType.PDF,
                    source=source,
                    data=data,
                    mime_type=doc.get("mime_type", "application/pdf"),
                    filename=doc.get("filename"),
                ))

        return files

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific OpenAI model."""
        # Check exact match first
        if model in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[model]

        # Check prefix matches (e.g., "gpt-4o-2024-05-13" matches "gpt-4o")
        for model_prefix, capabilities in self.MODEL_CAPABILITIES.items():
            if model.startswith(model_prefix):
                return capabilities

        # Return default capabilities for unknown models
        return self.DEFAULT_CAPABILITIES

    # =========================================================================
    # Chat Completion
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict[str, Any] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AdapterResponse:
        """Execute chat completion via OpenAI Responses API."""
        start_time = time.perf_counter()

        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            # Normalize messages and extract system instruction for Responses API
            if self.wire == "chat_completions":
                return await self._chat_via_completions(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, top_p=top_p, stop=stop, tools=tools,
                    tool_choice=tool_choice, response_format=response_format,
                    model_kwargs=model_kwargs, start_time=start_time,
                )

            instructions, response_input = self._normalize_messages_for_responses(messages)

            # Some models (reasoning, nano) don't support temperature/top_p
            _no_sampling = self._model_skips_sampling_params(model)

            # Build request parameters
            params: dict[str, Any] = {
                "model": model,
                "input": response_input,
            }

            if not _no_sampling:
                params["temperature"] = temperature
            if instructions:
                params["instructions"] = instructions
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens
            if top_p is not None and not _no_sampling:
                params["top_p"] = top_p
            if stop is not None:
                params["stop"] = stop

            # Add tools if provided
            if tools:
                params["tools"] = self.convert_tools(tools)
                if tool_choice:
                    params["tool_choice"] = self._convert_tool_choice(tool_choice)

            # Add response_format if provided
            if response_format:
                params["text"] = {"format": self._convert_response_format_for_responses(response_format)}

            # Add MCP servers as tools
            if mcp_servers:
                if "tools" not in params:
                    params["tools"] = []
                for server in mcp_servers:
                    mcp_tool: dict[str, Any] = {
                        "type": "mcp",
                        "server_label": server.get("name", "mcp"),
                        "server_url": server["url"],
                        "require_approval": server.get("require_approval") or "never",
                    }
                    if server.get("allowed_tools"):
                        mcp_tool["allowed_tools"] = server["allowed_tools"]
                    if server.get("auth_token"):
                        mcp_tool["headers"] = {"Authorization": f"Bearer {server['auth_token']}"}
                    params["tools"].append(mcp_tool)

            # Model-specific kwargs, pre-validated by the caller: merged into
            # the wire body (Responses API) — never silently dropped upstream.
            if model_kwargs:
                params["extra_body"] = {**params.get("extra_body", {}), **model_kwargs}

            # Make API call via Responses API
            response = await self._client.responses.create(**params)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract content
            content = response.output_text or ""

            # Extract tool calls from output items
            tool_calls = None
            for item in response.output:
                if item.type == "function_call":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": item.call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": item.arguments,
                        },
                    })

            # Build token usage
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                cached_tokens=_cached_tokens(response.usage),
                cache_write_5m_tokens=_cache_write_tokens(response.usage),
            )

            # Derive finish reason
            finish_reason = "stop"
            if tool_calls:
                finish_reason = "tool_calls"
            elif response.status == "incomplete":
                finish_reason = "length"

            return AdapterResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                raw_response=response.model_dump(),
            )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"OpenAI chat error after {latency_ms}ms: {error_message}")

            if "rate_limit" in error_message.lower():
                raise ProviderUnavailableError(
                    message="OpenAI rate limit exceeded",
                    provider=self.provider_name,
                    original_error=error_message,
                )
            raise ProviderError(
                message=f"OpenAI API error: {error_message}",
                provider=self.provider_name,
                status_code=provider_status_code(e),
                original_error=error_message,
            )

    # =========================================================================
    # Streaming
    # =========================================================================

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict[str, Any] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        stream_thinking: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion via Responses API."""
        if self.wire == "chat_completions":
            async for chunk in self._chat_stream_via_completions(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, top_p=top_p, stop=stop, tools=tools,
                tool_choice=tool_choice, response_format=response_format,
                model_kwargs=model_kwargs,
            ):
                yield chunk
            return
        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            instructions, response_input = self._normalize_messages_for_responses(messages)

            _no_sampling = self._model_skips_sampling_params(model)

            params: dict[str, Any] = {
                "model": model,
                "input": response_input,
                "stream": True,
            }

            if not _no_sampling:
                params["temperature"] = temperature
            if instructions:
                params["instructions"] = instructions
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens
            if top_p is not None and not _no_sampling:
                params["top_p"] = top_p
            if stop is not None:
                params["stop"] = stop
            # Ask for a streamed reasoning summary so callers can render live
            # "thinking" during the (otherwise silent) reasoning phase. Only
            # reasoning models accept the `reasoning` param.
            if stream_thinking and _is_reasoning_model(model):
                params["reasoning"] = {"summary": "auto"}

            if tools:
                params["tools"] = self.convert_tools(tools)
                if tool_choice:
                    params["tool_choice"] = self._convert_tool_choice(tool_choice)

            if response_format:
                params["text"] = {"format": self._convert_response_format_for_responses(response_format)}

            # Add MCP servers as tools
            if mcp_servers:
                if "tools" not in params:
                    params["tools"] = []
                for server in mcp_servers:
                    mcp_tool: dict[str, Any] = {
                        "type": "mcp",
                        "server_label": server.get("name", "mcp"),
                        "server_url": server["url"],
                        "require_approval": server.get("require_approval") or "never",
                    }
                    if server.get("allowed_tools"):
                        mcp_tool["allowed_tools"] = server["allowed_tools"]
                    if server.get("auth_token"):
                        mcp_tool["headers"] = {"Authorization": f"Bearer {server['auth_token']}"}
                    params["tools"].append(mcp_tool)

            if model_kwargs:
                params["extra_body"] = {**params.get("extra_body", {}), **model_kwargs}
            stream = await self._client.responses.create(**params)

            # The Responses API carries a function call's id+name on the output_item.added
            # event, but its arguments arrive (and complete) on separate
            # function_call_arguments.* events keyed only by item_id. Track id+name here so we
            # can emit a complete tool_call when the arguments finish.
            fn_calls_by_item: dict[str, dict[str, str]] = {}

            async for event in stream:
                # Text content delta
                if event.type == "response.output_text.delta":
                    yield StreamChunk(
                        content=event.delta,
                        done=False,
                    )

                # A function-call output item begins — capture its call_id + name.
                elif event.type == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        fn_calls_by_item[item.id] = {
                            "call_id": item.call_id,
                            "name": item.name,
                        }

                # Function call arguments complete — join with the captured id+name.
                elif event.type == "response.function_call_arguments.done":
                    meta = fn_calls_by_item.get(event.item_id, {})
                    yield StreamChunk(
                        content="",
                        done=False,
                        tool_calls=[{
                            "id": meta.get("call_id", event.item_id),
                            "type": "function",
                            "function": {
                                "name": meta.get("name", ""),
                                "arguments": event.arguments,
                            },
                        }],
                    )

                # Reasoning summary text (only when stream_thinking asked for it
                # via reasoning={"summary":"auto"}): surface as a distinct
                # `thinking` channel, kept out of assistant content.
                elif event.type == "response.reasoning_summary_text.delta":
                    yield StreamChunk(content="", done=False, thinking=event.delta)

                # Other reasoning phase events (gpt-5.x / o-series): the model
                # "thinks" before any output token. Emit an empty keepalive chunk
                # so bytes flow on the wire during long reasoning instead of going
                # byte-silent and tripping idle timeouts.
                elif event.type.startswith("response.reasoning"):
                    yield StreamChunk(content="", done=False)

                # Response completed — final chunk with usage
                elif event.type == "response.completed":
                    resp = event.response
                    finish_reason = "stop"
                    if any(item.type == "function_call" for item in resp.output):
                        finish_reason = "tool_calls"
                    elif resp.status == "incomplete":
                        finish_reason = "length"

                    usage = TokenUsage(
                        input_tokens=resp.usage.input_tokens if resp.usage else 0,
                        output_tokens=resp.usage.output_tokens if resp.usage else 0,
                        cached_tokens=_cached_tokens(resp.usage),
                        cache_write_5m_tokens=_cache_write_tokens(resp.usage),
                    )
                    yield StreamChunk(
                        content="",
                        done=True,
                        usage=usage,
                        finish_reason=finish_reason,
                    )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise ProviderError(
                message=f"OpenAI streaming error: {e}",
                provider=self.provider_name,
                status_code=provider_status_code(e),
                original_error=str(e),
            )

    # =========================================================================
    # Message Normalization
    # =========================================================================

    # =========================================================================
    # chat_completions wire (byom custom endpoints: vLLM / Ollama / TGI / …)
    # =========================================================================

    def _completions_params(
        self, model, messages, temperature, max_tokens, top_p, stop, tools,
        tool_choice, response_format, model_kwargs,
    ) -> dict[str, Any]:
        """Build a /v1/chat/completions body. The gateway's canonical message
        shape IS chat-completions shape (role/content, OpenAI-style tool_calls,
        role=tool results), so messages pass through minus gateway-only keys."""
        wire_messages = []
        for m in messages:
            wm = {k: v for k, v in m.items() if k not in _GATEWAY_ONLY_MESSAGE_KEYS}
            wire_messages.append(wm)
        params: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if stop is not None:
            params["stop"] = stop
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tool_choice:
                params["tool_choice"] = tool_choice
        if response_format:
            params["response_format"] = response_format
        if model_kwargs:
            params["extra_body"] = {**params.get("extra_body", {}), **model_kwargs}
        return params

    @staticmethod
    def _completions_tool_calls(message) -> list[dict[str, Any]] | None:
        tcs = getattr(message, "tool_calls", None)
        if not tcs:
            return None
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ]

    async def _create_completions(self, params: dict[str, Any]):
        """Create with a bounded compat loop for well-known chat-completions
        dialect drift. vLLM/Ollama/TGI speak `max_tokens` + free sampling;
        OpenAI's reasoning models demand `max_completion_tokens` and pin
        temperature/top_p. Each retry is triggered ONLY by the provider's own
        explicit param complaint (its `param` field) — never a heuristic —
        and anything else re-raises untouched."""
        params = dict(params)
        for _ in range(3):
            try:
                return await self._client.chat.completions.create(**params)
            except Exception as e:  # noqa: BLE001 — classified below, else re-raised
                msg = str(e)
                offender = getattr(getattr(e, "body", None), "get", lambda *_: None)("param") \
                    if hasattr(e, "body") and isinstance(e.body, dict) else None
                if offender is None and hasattr(e, "body") and isinstance(e.body, dict):
                    offender = (e.body.get("error") or {}).get("param")
                if (
                    "max_tokens" in params
                    and (offender == "max_tokens" or "max_completion_tokens" in msg)
                ):
                    params["max_completion_tokens"] = params.pop("max_tokens")
                    continue
                if offender in ("temperature", "top_p") and offender in params:
                    params.pop(offender)
                    continue
                raise
        return await self._client.chat.completions.create(**params)

    async def _chat_via_completions(
        self, *, model, messages, temperature, max_tokens, top_p, stop, tools,
        tool_choice, response_format, model_kwargs, start_time,
    ) -> AdapterResponse:
        params = self._completions_params(
            model, messages, temperature, max_tokens, top_p, stop, tools,
            tool_choice, response_format, model_kwargs,
        )
        response = await self._create_completions(params)
        choice = response.choices[0]
        usage = response.usage
        return AdapterResponse(
            content=choice.message.content or "",
            model=model,
            provider=self.provider_name,
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_tokens=_completions_cache_tokens(usage)[0],
                cache_write_5m_tokens=_completions_cache_tokens(usage)[1],
            ),
            finish_reason=choice.finish_reason or "stop",
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            tool_calls=self._completions_tool_calls(choice.message),
        )

    async def _chat_stream_via_completions(
        self, *, model, messages, temperature, max_tokens, top_p, stop, tools,
        tool_choice, response_format, model_kwargs,
    ) -> AsyncIterator[StreamChunk]:
        params = self._completions_params(
            model, messages, temperature, max_tokens, top_p, stop, tools,
            tool_choice, response_format, model_kwargs,
        )
        params["stream"] = True
        # Ask for usage on the final chunk; some servers (older vLLM/Ollama)
        # ignore stream_options — usage then reports 0s, which is factual
        # ("unknown"), never estimated.
        params["stream_options"] = {"include_usage": True}
        stream = await self._create_completions(params)
        final_usage = None
        finish_reason = None
        # tool-call deltas accumulate per index across chunks
        pending_tools: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                final_usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = pending_tools.setdefault(
                    tc.index, {"id": None, "name": "", "arguments": ""}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if delta.content:
                yield StreamChunk(content=delta.content)
        tool_calls = [
            {
                "id": t["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": t["name"], "arguments": t["arguments"]},
            }
            for i, t in sorted(pending_tools.items())
        ] or None
        yield StreamChunk(
            content="",
            done=True,
            finish_reason=finish_reason or "stop",
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=getattr(final_usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(final_usage, "completion_tokens", 0) or 0,
                cached_tokens=_completions_cache_tokens(final_usage)[0],
                cache_write_5m_tokens=_completions_cache_tokens(final_usage)[1],
            ),
        )

    def _normalize_messages_for_responses(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert messages to OpenAI Responses API input format.

        Returns:
            Tuple of (instructions, input_items) where instructions is extracted
            from system messages and input_items is the Responses API input array.

        Key differences from Chat Completions:
        - System messages become the `instructions` parameter
        - Tool results use `function_call_output` type
        - Assistant tool calls use `function_call` type
        """
        instructions = None
        normalized = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages into instructions
            if role == "system":
                if instructions:
                    instructions += "\n\n" + str(content)
                else:
                    instructions = str(content)
                continue

            # Handle tool calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                # First add any text content as a message
                if content:
                    normalized.append({"role": "assistant", "content": str(content)})
                # Then add each tool call as a function_call item
                for tc in msg["tool_calls"]:
                    normalized.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })
                continue

            # Handle tool results — Responses API uses function_call_output
            if role == "tool":
                normalized.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": str(content),
                })
                continue

            # Handle files/images/documents in content
            files = msg.get("files", [])
            images = msg.get("images", [])
            documents = msg.get("documents", [])

            if files or images or documents:
                # Build multimodal content
                content_parts: list[dict[str, Any]] = []

                # Add text
                if content:
                    content_parts.append({"type": "input_text", "text": str(content)})

                # Process files (supports both FileAttachment and dict)
                for f in files:
                    content_part = self._process_file_attachment(f)
                    if content_part:
                        content_parts.append(content_part)

                # Add images directly (legacy format)
                for img in images:
                    content_parts.append(self._build_image_content(img))

                # Add documents directly (legacy format)
                for doc in documents:
                    content_parts.append(self._build_document_content(doc))

                normalized.append({"role": role, "content": content_parts})
            else:
                # Check if content is already multimodal format
                if isinstance(content, list):
                    normalized.append({"role": role, "content": content})
                else:
                    normalized.append({"role": role, "content": str(content)})

        return instructions, normalized

    def _convert_response_format_for_responses(
        self,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert Chat Completions response_format to Responses API text.format."""
        fmt_type = response_format.get("type", "text")
        if fmt_type == "json_schema":
            schema_spec = response_format.get("json_schema", {})
            return {
                "type": "json_schema",
                "name": schema_spec.get("name", "response"),
                "schema": schema_spec.get("schema", {}),
                "strict": schema_spec.get("strict", True),
            }
        elif fmt_type == "json_object":
            return {"type": "json_object"}
        else:
            return {"type": "text"}

    def _process_file_attachment(
        self,
        attachment: FileAttachment | dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Process a file attachment into OpenAI content format.

        Handles both FileAttachment objects and legacy dict format.
        Routes to appropriate handler based on file type.
        """
        # Handle FileAttachment object
        if isinstance(attachment, FileAttachment):
            if attachment.type == FileType.IMAGE:
                return self._build_image_from_attachment(attachment)
            elif attachment.type in (FileType.PDF, FileType.DOCUMENT):
                return self._build_document_from_attachment(attachment)
            elif attachment.type == FileType.AUDIO:
                return self._build_audio_from_attachment(attachment)
            elif attachment.type in (FileType.CODE, FileType.TEXT):
                # Code and text are sent as text content
                return self._build_text_from_attachment(attachment)
            else:
                logger.warning(f"Unsupported file type: {attachment.type}")
                return None

        # Handle dict with source/data keys (from service layer)
        if "source" in attachment and "data" in attachment:
            file_type_str = attachment.get("type", "other")
            try:
                ft = FileType(file_type_str)
            except ValueError:
                ft = FileType.OTHER
            fa = FileAttachment(
                type=ft,
                source=FileSource(attachment["source"]),
                data=attachment["data"],
                mime_type=attachment.get("mime_type"),
                filename=attachment.get("filename"),
            )
            return self._process_file_attachment(fa)

        # Handle legacy dict format (url/base64 keys directly)
        file_type = attachment.get("type", "")
        if file_type == "image":
            return self._build_image_content(attachment)
        elif file_type in ("pdf", "document"):
            return self._build_document_content(attachment)
        elif file_type == "audio":
            return self._build_audio_content(attachment)
        elif file_type in ("code", "text"):
            # Include as text
            text_content = attachment.get("text") or attachment.get("data", "")
            filename = attachment.get("filename", "file")
            return {"type": "text", "text": f"[{filename}]\n{text_content}"}
        else:
            logger.warning(f"Unknown file type in dict: {file_type}")
            return None

    def _build_image_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """Build image content block from FileAttachment (Responses API format)."""
        if attachment.source == FileSource.URL:
            return {
                "type": "input_image",
                "image_url": attachment.data,
                "detail": "auto",
            }
        elif attachment.source == FileSource.BASE64:
            mime = attachment.mime_type or "image/png"
            return {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{attachment.data}",
                "detail": "auto",
            }
        elif attachment.source == FileSource.PATH:
            # Read file and convert to base64
            import base64
            with open(attachment.data, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            mime = attachment.mime_type or "image/png"
            return {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{data}",
                "detail": "auto",
            }
        else:
            raise ValueError(f"Unsupported image source: {attachment.source}")

    def _build_document_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """
        Build document content block from FileAttachment.

        Note: OpenAI requires documents (PDFs) to be uploaded via their Files API
        and referenced by file_id. Inline base64 data is NOT supported for documents.

        For PATH and BASE64 sources, this method synchronously uploads the file
        to OpenAI's Files API and uses the returned file_id.
        """
        if attachment.source == FileSource.FILE_ID:
            return {
                "type": "input_file",
                "file_id": attachment.data,
            }
        elif attachment.source == FileSource.URL:
            return {
                "type": "input_file",
                "url": attachment.data,
            }
        elif attachment.source == FileSource.PATH:
            file_id = self._upload_file_sync(
                file_path=attachment.data,
                purpose="user_data",
            )
            return {
                "type": "input_file",
                "file_id": file_id,
            }
        elif attachment.source == FileSource.BASE64:
            import base64
            import os
            import tempfile

            file_data = base64.b64decode(attachment.data)
            filename = attachment.filename or "document.pdf"

            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                file_id = self._upload_file_sync(
                    file_path=tmp_path,
                    purpose="user_data",
                )
            finally:
                os.unlink(tmp_path)

            return {
                "type": "input_file",
                "file_id": file_id,
            }
        else:
            raise ValueError(f"Unsupported document source: {attachment.source}")

    def _upload_file_sync(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Synchronously upload a file to OpenAI's Files API.

        Args:
            file_path: Local path to the file
            purpose: Upload purpose (assistants, fine-tune, etc.)

        Returns:
            The file_id of the uploaded file
        """
        from pathlib import Path

        # Use synchronous client for file upload within async context
        from openai import OpenAI

        sync_client = OpenAI(api_key=self._api_key, base_url=self._base_url)

        file_path_obj = Path(file_path)
        with open(file_path_obj, "rb") as f:
            response = sync_client.files.create(
                file=f,
                purpose=purpose,
            )

        logger.info(f"Uploaded file {file_path_obj.name} to OpenAI Files API: {response.id}")
        return response.id

    def _build_audio_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """Build audio content block from FileAttachment (for gpt-4o-audio-preview)."""
        if attachment.source == FileSource.BASE64:
            # Determine audio format from mime type
            mime = attachment.mime_type or "audio/wav"
            format_map = {
                "audio/wav": "wav",
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
            }
            audio_format = format_map.get(mime, "wav")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": attachment.data,
                    "format": audio_format,
                },
            }
        elif attachment.source == FileSource.PATH:
            # Read file and convert to base64
            import base64
            with open(attachment.data, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            mime = attachment.mime_type or "audio/wav"
            format_map = {
                "audio/wav": "wav",
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
            }
            audio_format = format_map.get(mime, "wav")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": data,
                    "format": audio_format,
                },
            }
        else:
            raise ValueError(
                f"Unsupported audio source: {attachment.source}. "
                "OpenAI audio input requires base64 or local path."
            )

    def _build_text_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """Build text content block from code/text FileAttachment."""
        # For text/code files, we need to read and include as text
        if attachment.source == FileSource.PATH:
            with open(attachment.data, encoding="utf-8") as f:
                text_content = f.read()
        elif attachment.source == FileSource.BASE64:
            import base64
            text_content = base64.b64decode(attachment.data).decode("utf-8")
        elif attachment.source == FileSource.URL:
            # Can't fetch URL content here - would need async
            text_content = f"[File from URL: {attachment.data}]"
        else:
            text_content = attachment.data

        filename = attachment.filename or "file"
        return {"type": "text", "text": f"[{filename}]\n{text_content}"}

    def _build_image_content(self, image: dict[str, Any]) -> dict[str, Any]:
        """Build image content block for vision (Responses API format)."""
        if image.get("url"):
            return {
                "type": "input_image",
                "image_url": image["url"],
                "detail": image.get("detail", "auto"),
            }
        elif image.get("base64"):
            mime = image.get("mime_type", "image/png")
            return {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{image['base64']}",
                "detail": image.get("detail", "auto"),
            }
        else:
            raise ValueError("Image must have either 'url' or 'base64' field")

    def _build_document_content(self, doc: dict[str, Any]) -> dict[str, Any]:
        """
        Build document content block (PDF, etc.) from legacy dict format.

        Note: OpenAI requires documents to be uploaded via the Files API.
        Base64 data is uploaded automatically.
        """
        if doc.get("file_id"):
            return {
                "type": "input_file",
                "file_id": doc["file_id"],
            }
        elif doc.get("url"):
            return {
                "type": "input_file",
                "url": doc["url"],
            }
        elif doc.get("base64"):
            import base64 as b64_module
            import os
            import tempfile

            file_data = b64_module.b64decode(doc["base64"])
            filename = doc.get("filename", "document.pdf")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                file_id = self._upload_file_sync(tmp_path, purpose="user_data")
            finally:
                os.unlink(tmp_path)

            return {
                "type": "input_file",
                "file_id": file_id,
            }
        elif doc.get("text"):
            # Plain text document - include as text
            return {
                "type": "text",
                "text": f"Document content:\n{doc['text']}",
            }
        else:
            raise ValueError("Document must have 'base64', 'url', 'file_id', or 'text' field")

    def _build_audio_content(self, audio: dict[str, Any]) -> dict[str, Any]:
        """Build audio content block from legacy dict format (for gpt-4o-audio-preview)."""
        if audio.get("base64"):
            mime = audio.get("mime_type", "audio/wav")
            format_map = {
                "audio/wav": "wav",
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
            }
            audio_format = format_map.get(mime, "wav")
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": audio["base64"],
                    "format": audio_format,
                },
            }
        else:
            raise ValueError(
                "Audio must have 'base64' field. "
                "OpenAI audio input requires base64-encoded data."
            )

    # =========================================================================
    # Tool Conversion
    # =========================================================================

    def convert_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI Responses API format (flattened)."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": tool.strict,
            }
            for tool in tools
        ]

    # Models that don't support temperature/top_p
    # O-series: all variants skip sampling (o1-*, o3-*, o4-*)
    _NO_SAMPLING_PREFIXES = ("o1", "o3", "o4")
    # Specific models that skip sampling (exact match required to avoid gpt-5 matching gpt-5.2)
    # gpt-5.5 family rejects `temperature` outright ("not supported with this model"),
    # verified against the live API — unlike gpt-5.4-mini/nano which still accept it.
    _NO_SAMPLING_EXACT = (
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
        "gpt-5.2-pro", "gpt-5.1-codex", "gpt-5.2-codex",
        "gpt-5.5", "gpt-5.5-pro", "gpt-5.4-pro",
        # GPT-5.6 family (Sol/Terra/Luna) rejects `temperature` outright,
        # verified against the live API.
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
        # GPT-6 Astra: `temperature` rejected ("not supported with this
        # model"), verified live 2026-09-07.
        "gpt-6-astra",
    )

    def _model_skips_sampling_params(self, model: str) -> bool:
        """Check if model doesn't support temperature/top_p."""
        # Check exact matches first (for gpt-5 family where prefix matching is problematic)
        if model in self._NO_SAMPLING_EXACT:
            return True
        # Check o-series prefixes
        return any(model.startswith(p) for p in self._NO_SAMPLING_PREFIXES)

    def _convert_tool_choice(self, tool_choice: str | dict) -> str | dict:
        """Convert tool_choice to OpenAI format."""
        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required"):
                return tool_choice
            else:
                # Specific tool name
                return {"type": "function", "function": {"name": tool_choice}}
        return tool_choice

    # =========================================================================
    # Generation Methods
    # =========================================================================

    # Default models for generation
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
    DEFAULT_IMAGE_MODEL = "gpt-image-2"  # DALL-E models are retired from the API
    DEFAULT_TTS_MODEL = "tts-1-hd"
    DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings using OpenAI's embedding models.

        Args:
            texts: List of text strings to embed
            model: Model ID (default: text-embedding-3-large)
            dimensions: Output dimensions (only for text-embedding-3-* models)

        Returns:
            EmbeddingResponse with embedding vectors

        Models:
            - text-embedding-3-large: 3072 dimensions (best quality)
            - text-embedding-3-small: 1536 dimensions (cost-effective)
            - text-embedding-ada-002: 1536 dimensions (legacy)
        """
        model = model or self.DEFAULT_EMBEDDING_MODEL
        start_time = time.perf_counter()

        try:
            params: dict[str, Any] = {
                "model": model,
                "input": texts,
            }

            # Only text-embedding-3-* models support dimensions parameter
            if dimensions and model.startswith("text-embedding-3"):
                params["dimensions"] = dimensions

            response = await self._client.embeddings.create(**params)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            embeddings = [item.embedding for item in response.data]
            actual_dimensions = len(embeddings[0]) if embeddings else 0

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                provider=self.provider_name,
                usage=TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=0,
                ),
                latency_ms=latency_ms,
                dimensions=actual_dimensions,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"OpenAI embedding error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"OpenAI embedding error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: ImageSize | str = ImageSize.SQUARE,
        quality: ImageQuality = ImageQuality.STANDARD,
        n: int = 1,
        reference_images: list[FileAttachment] | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate images using OpenAI's image generation models.

        Args:
            prompt: Text description of the image to generate
            model: Model ID (default: gpt-image-2)
            size: Output size (ImageSize enum or string)
            quality: Quality level (standard or hd)
            n: Number of images (1 for DALL-E 3, 1-10 for DALL-E 2)
            reference_images: Optional reference images (requires gpt-image-1)

        Returns:
            ImageGenerationResponse with generated images

        Models:
            - dall-e-3: Best quality, text-to-image only
            - dall-e-2: Legacy, supports n > 1
            - gpt-image-1: Supports image-to-image with references
        """
        model = model or self.DEFAULT_IMAGE_MODEL

        # Route to appropriate method based on whether we have references
        if reference_images and len(reference_images) > 0:
            return await self._generate_image_with_references(
                prompt=prompt,
                reference_images=reference_images,
                model=model,
                size=size,
                quality=quality,
            )
        else:
            return await self._generate_image_text_only(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                n=n,
            )

    async def _generate_image_text_only(
        self,
        prompt: str,
        model: str,
        size: ImageSize | str,
        quality: ImageQuality,
        n: int,
    ) -> ImageGenerationResponse:
        """Generate images from text prompt only using Images API."""
        start_time = time.perf_counter()

        try:
            # Use ImageSizeResolver to convert any size format to OpenAI pixels
            size_str = ImageSizeResolver.to_pixels_openai(size, model=model)

            # gpt-image models use different quality values than DALL-E
            if model.startswith("gpt-image"):
                # gpt-image: low, medium, high, auto
                quality_map = {
                    ImageQuality.STANDARD: "medium",
                    ImageQuality.HD: "high",
                    ImageQuality.AUTO: "auto",
                }
                quality_str = quality_map.get(quality, "medium")
            else:
                # DALL-E: standard, hd
                quality_str = quality.value if quality != ImageQuality.AUTO else "standard"

            params: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "size": size_str,
                "quality": quality_str,
            }

            # gpt-image models don't support response_format parameter
            # DALL-E models support it
            if not model.startswith("gpt-image"):
                params["response_format"] = "b64_json"

            # DALL-E 3 only supports n=1
            if model != "dall-e-3":
                params["n"] = n
            else:
                params["n"] = 1

            response = await self._client.images.generate(**params)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            images = []
            revised_prompt = None
            for img_data in response.data:
                if hasattr(img_data, "revised_prompt") and img_data.revised_prompt:
                    revised_prompt = img_data.revised_prompt

                images.append(GeneratedImage(
                    base64=img_data.b64_json,
                    format="png",
                    revised_prompt=getattr(img_data, "revised_prompt", None),
                ))

            return ImageGenerationResponse(
                images=images,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                prompt=prompt,
                revised_prompt=revised_prompt,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"OpenAI image generation error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"OpenAI image generation error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    async def _generate_image_with_references(
        self,
        prompt: str,
        reference_images: list[FileAttachment],
        model: str,
        size: ImageSize | str,
        quality: ImageQuality,
    ) -> ImageGenerationResponse:
        """
        Generate images with reference images using GPT-Image models.

        Uses the images.edit endpoint which supports image-to-image generation.
        Pass reference images via the 'image' parameter.
        """
        import base64

        import httpx

        start_time = time.perf_counter()

        try:
            # Convert reference images to file-like objects with proper MIME type
            # OpenAI SDK requires (filename, file_bytes, mime_type) tuple or file-like object
            image_files: list[tuple[str, bytes, str]] = []

            for i, ref in enumerate(reference_images):
                mime_type = ref.mime_type or "image/png"

                # Determine file extension from MIME type
                ext_map = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/jpg": "jpg",
                    "image/webp": "webp",
                    "image/gif": "gif",
                }
                ext = ext_map.get(mime_type, "png")
                filename = f"image_{i}.{ext}"

                if ref.source == FileSource.BASE64:
                    img_bytes = base64.b64decode(ref.data)
                elif ref.source == FileSource.PATH:
                    with open(ref.data, "rb") as f:
                        img_bytes = f.read()
                elif ref.source == FileSource.URL:
                    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                        response = await client.get(ref.data)
                        response.raise_for_status()
                        img_bytes = response.content
                        # Try to get MIME type from response headers
                        content_type = response.headers.get("content-type", "").split(";")[0]
                        if content_type in ext_map:
                            mime_type = content_type
                            ext = ext_map[content_type]
                            filename = f"image_{i}.{ext}"
                else:
                    continue

                image_files.append((filename, img_bytes, mime_type))

            if not image_files:
                raise ProviderError(
                    message="No valid reference images provided",
                    provider=self.provider_name,
                )

            # Only the gpt-image family has an edits endpoint.
            image_model = model if model.startswith("gpt-image") else self.DEFAULT_IMAGE_MODEL

            # Use ImageSizeResolver to convert any size format to OpenAI pixels
            size_str = ImageSizeResolver.to_pixels_openai(size, model=image_model)

            # gpt-image models use different quality values than DALL-E
            if image_model.startswith("gpt-image"):
                quality_map = {
                    ImageQuality.STANDARD: "medium",
                    ImageQuality.HD: "high",
                    ImageQuality.AUTO: "auto",
                }
                quality_str = quality_map.get(quality, "medium")
            else:
                quality_str = quality.value if quality != ImageQuality.AUTO else "standard"

            # Use images.edit endpoint for image-to-image
            # For single image: pass tuple (filename, bytes, mime_type)
            # For multiple images: pass list of tuples
            if len(image_files) == 1:
                image_input = image_files[0]
            else:
                image_input = image_files

            response = await self._client.images.edit(
                model=image_model,
                image=image_input,
                prompt=prompt,
                size=size_str,
                quality=quality_str,
                n=1,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            images = []
            revised_prompt = None
            for img_data in response.data:
                if hasattr(img_data, "revised_prompt") and img_data.revised_prompt:
                    revised_prompt = img_data.revised_prompt

                images.append(GeneratedImage(
                    base64=img_data.b64_json,
                    format="png",
                    revised_prompt=getattr(img_data, "revised_prompt", None),
                ))

            return ImageGenerationResponse(
                images=images,
                model=image_model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                prompt=prompt,
                revised_prompt=revised_prompt,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"OpenAI image generation with references error: {e}")
            raise ProviderError(
                message=f"OpenAI image generation error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    async def transcribe(
        self,
        audio: FileAttachment,
        model: str | None = None,
        language: str | None = None,
    ) -> AudioResponse:
        """
        Transcribe audio to text using OpenAI's transcription models.

        Args:
            audio: Audio file attachment (mp3, wav, m4a, etc.)
            model: Model ID (default: gpt-4o-transcribe)
            language: Optional language code (e.g., "en", "es")

        Returns:
            AudioResponse with transcribed text

        Models:
            - gpt-4o-transcribe: Latest, highest accuracy
            - whisper-1: Stable, multilingual
        """
        import base64

        model = model or self.DEFAULT_TRANSCRIPTION_MODEL
        start_time = time.perf_counter()

        try:
            # Get audio bytes
            if audio.source == FileSource.PATH:
                with open(audio.data, "rb") as f:
                    audio_bytes = f.read()
                filename = audio.filename or audio.data.split("/")[-1]
            elif audio.source == FileSource.BASE64:
                audio_bytes = base64.b64decode(audio.data)
                filename = audio.filename or "audio.mp3"
            else:
                raise ValueError(f"Unsupported audio source for transcription: {audio.source}")

            # Create file tuple for OpenAI
            params: dict[str, Any] = {
                "model": model,
                "file": (filename, audio_bytes),
                "response_format": "json",
            }

            if language:
                params["language"] = language

            response = await self._client.audio.transcriptions.create(**params)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return AudioResponse(
                content=response.text,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                language=language,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"OpenAI transcription error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"OpenAI transcription error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    async def text_to_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str = "alloy",
        format: str = "mp3",
        speed: float = 1.0,
    ) -> AudioResponse:
        """
        Convert text to speech using OpenAI's TTS models.

        Args:
            text: Text to convert to speech
            model: Model ID (default: tts-1-hd)
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            format: Audio format (mp3, opus, aac, flac, wav)
            speed: Playback speed (0.25 to 4.0)

        Returns:
            AudioResponse with audio bytes

        Models:
            - tts-1-hd: High quality
            - tts-1: Standard quality, faster
            - gpt-4o-mini-tts: Compact
        """
        model = model or self.DEFAULT_TTS_MODEL
        start_time = time.perf_counter()

        try:
            response = await self._client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=format,
                speed=speed,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Get audio bytes
            audio_bytes = response.content

            return AudioResponse(
                content=audio_bytes,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                format=format,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"OpenAI TTS error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"OpenAI TTS error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_documents(self) -> bool:
        return True
