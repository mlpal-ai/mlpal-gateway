"""Anthropic provider adapter using native SDK."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from mlpal_assistants_service.adapters.base import (
    AdapterResponse,
    AudioResponse,
    BaseAdapter,
    EmbeddingResponse,
    FileAttachment,
    FileSource,
    FileType,
    ImageGenerationResponse,
    ImageQuality,
    ImageSize,
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


def _apply_effort(params: dict[str, Any], reasoning_effort: str | None) -> None:
    """Universal effort → Anthropic knobs. `none` means what the client said:
    no thinking at all. Other rungs ride `output_config.effort` (the resolver
    only hands us rungs this model accepts)."""
    if not reasoning_effort:
        return
    if reasoning_effort == "none":
        params["thinking"] = {"type": "disabled"}
        return
    params["output_config"] = {**params.get("output_config", {}), "effort": reasoning_effort}


def _append_system_text(system: str | list | None, text: str) -> str | list:
    """Append instruction text to a system prompt that may be a string or a
    block list (block lists appear when a cache_control breakpoint is set)."""
    if isinstance(system, list):
        return [*system, {"type": "text", "text": text}]
    return (system or "") + text


class AnthropicAdapter(BaseAdapter):
    # Messages-API params we forward via model_kwargs.
    SUPPORTED_KWARGS = frozenset({"top_k", "metadata", "service_tier", "thinking", "output_config"})

    """
    Adapter for Anthropic API using native SDK.

    Supports:
    - Chat completion (text)
    - Structured output (JSON schema via tool use)
    - Vision (image input via URL or base64)
    - Documents (PDFs via document content blocks)
    - Tool/function calling
    - Streaming

    Models (Claude 4.5):
    - claude-opus-4-5-20251101 (most capable)
    - claude-sonnet-4-5-20250929 (balanced)
    - claude-haiku-4-5-20251001 (fast)
    """

    # Anthropic-style usage: input_tokens EXCLUDES cache reads (see BaseAdapter).
    cached_tokens_included_in_input = False

    provider_name = "anthropic"

    # Models that reject `temperature` ("deprecated for this model"), verified
    # against the live API. Newer Opus tiers and the Mythos-class models dropped
    # the sampling knob; older Claude (sonnet-4-6, opus-4-6 and earlier) still
    # accept it. Match by prefix so dated snapshots are covered too.
    _NO_TEMPERATURE_PREFIXES = (
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    )

    def _skips_temperature(self, model: str) -> bool:
        """Whether this model rejects the `temperature` parameter."""
        return any(model.startswith(p) for p in self._NO_TEMPERATURE_PREFIXES)

    # Model capabilities registry
    MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
        # Claude 4.5 family - full multimodal support
        "claude-opus-4-5-20251101": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=32768,
        ),
        "claude-sonnet-4-5-20250929": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=16384,
        ),
        "claude-haiku-4-5-20251001": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=8192,
        ),
        # Claude 3.5 family
        "claude-3-5-sonnet-20241022": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=8192,
        ),
        "claude-3-5-haiku-20241022": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=8192,
        ),
        # Claude 3 family
        "claude-3-opus-20240229": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,  # Older models don't support PDF natively
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
        "claude-3-sonnet-20240229": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
        "claude-3-haiku-20240307": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            supports_mcp=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
    }

    # Default capabilities for unknown models (assume Claude 3.5+ capabilities)
    DEFAULT_CAPABILITIES = ModelCapabilities(
        supports_images=True,
        supports_pdf=True,
        supports_audio=False,
        supports_video=False,
        supports_tools=True,
        supports_structured_output=True,
        supports_mcp=True,
        max_context_tokens=200000,
        max_output_tokens=8192,
    )

    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        if api_key:
            self._api_key = api_key
        else:
            settings = get_settings()
            self._api_key = settings.anthropic_api_key

        # Create client with connection pooling. Ceiling on simultaneous in-flight
        # requests to Anthropic across ALL callers — raised for client concurrency.
        # `client` seam lets serving backends supply AsyncAnthropicBedrock /
        # AsyncAnthropicVertex — same .messages surface, different auth/host.
        if client is not None:
            self._client = client
            return
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self._client = AsyncAnthropic(
            api_key=self._api_key,
            http_client=http_client,
        )

    # =========================================================================
    # Model Capabilities
    # =========================================================================

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Anthropic model."""
        # Check exact match first
        if model in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[model]

        # Check prefix matches (e.g., "claude-3-5-sonnet" matches dated versions)
        for model_prefix, capabilities in self.MODEL_CAPABILITIES.items():
            # Extract base name without date
            base_prefix = model_prefix.rsplit("-", 1)[0] if model_prefix.count("-") > 2 else model_prefix
            if model.startswith(base_prefix):
                return capabilities

        # Return default capabilities for unknown models
        return self.DEFAULT_CAPABILITIES

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
        reasoning_effort: str | None = None,
    ) -> AdapterResponse:
        """Execute chat completion via Anthropic API."""
        start_time = time.perf_counter()

        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            # Normalize messages for Anthropic format
            system_message, anthropic_messages = self._normalize_messages(messages)

            # Build request parameters
            params: dict[str, Any] = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Add optional parameters
            if temperature is not None and not self._skips_temperature(model):
                params["temperature"] = temperature
            if top_p is not None:
                params["top_p"] = top_p
            if stop is not None:
                params["stop_sequences"] = stop

            # Add tools if provided
            if tools:
                params["tools"] = self.convert_tools(tools)
                if tool_choice:
                    params["tool_choice"] = self._convert_tool_choice(tool_choice)

            # Add response format via tool use (Anthropic structured output)
            _structured_tool_name = None
            if response_format:
                fmt_type = response_format.get("type")
                if fmt_type == "json_schema":
                    schema_spec = response_format.get("json_schema", {})
                    _structured_tool_name = schema_spec.get("name", "structured_output")
                    structured_tool = {
                        "name": _structured_tool_name,
                        "description": f"Return response in {_structured_tool_name} format. Always use this tool.",
                        "input_schema": schema_spec.get("schema", {}),
                    }
                    if "tools" not in params:
                        params["tools"] = []
                    params["tools"].append(structured_tool)
                    params["tool_choice"] = {"type": "tool", "name": _structured_tool_name}
                elif fmt_type == "json_object":
                    # For json_object, prepend instruction to system message
                    sys_msgs = [m for m in params.get("messages", []) if m.get("role") == "system"]
                    if not sys_msgs:
                        params["system"] = _append_system_text(
                            params.get("system"), "\nYou must respond with valid JSON only."
                        )

            # Add MCP servers if provided — requires beta API
            use_beta = False
            if mcp_servers:
                mcp_server_params = []
                for s in mcp_servers:
                    server_param: dict[str, Any] = {
                        "type": "url",
                        "url": s["url"],
                        "name": s["name"],
                    }
                    if s.get("auth_token"):
                        server_param["authorization_token"] = s["auth_token"]
                    mcp_server_params.append(server_param)
                params["mcp_servers"] = mcp_server_params

                # Add mcp_toolset entries to tools
                if "tools" not in params:
                    params["tools"] = []
                for s in mcp_servers:
                    params["tools"].append({
                        "type": "mcp_toolset",
                        "mcp_server_name": s["name"],
                    })
                use_beta = True

            # Model-specific kwargs, pre-validated by the caller.
            if model_kwargs:
                params["extra_body"] = {**params.get("extra_body", {}), **model_kwargs}
            _apply_effort(params, reasoning_effort)

            # Make API call (use beta API for MCP)
            if use_beta:
                response = await self._client.beta.messages.create(
                    betas=["mcp-client-2025-11-20"],
                    **params,
                )
            else:
                response = await self._client.messages.create(**params)

            # If structured output was requested via tool use, extract the tool result as content
            if _structured_tool_name:
                for block in response.content:
                    if block.type == "tool_use" and block.name == _structured_tool_name:
                        # Replace content with the structured JSON
                        latency_ms = int((time.perf_counter() - start_time) * 1000)
                        usage = self._token_usage(response.usage)
                        return AdapterResponse(
                            content=json.dumps(block.input),
                            model=response.model,
                            provider=self.provider_name,
                            usage=usage,
                            finish_reason="stop",
                            latency_ms=latency_ms,
                            tool_calls=None,
                            raw_response=response.model_dump(),
                        )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract response content and tool calls
            content = ""
            tool_calls = None

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })

            # Build token usage
            usage = self._token_usage(response.usage)

            return AdapterResponse(
                content=content,
                model=response.model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=self._map_stop_reason(response.stop_reason),
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                raw_response=response.model_dump(),
            )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"Anthropic chat error after {latency_ms}ms: {error_message}")

            if "rate_limit" in error_message.lower() or "overloaded" in error_message.lower():
                raise ProviderUnavailableError(
                    message="Anthropic rate limit exceeded or overloaded",
                    provider=self.provider_name,
                    original_error=error_message,
                )
            raise ProviderError(
                message=f"Anthropic API error: {error_message}",
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
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion."""
        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            system_message, anthropic_messages = self._normalize_messages(messages)

            params: dict[str, Any] = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens or 4096,
            }

            if system_message:
                params["system"] = system_message
            if temperature is not None and not self._skips_temperature(model):
                params["temperature"] = temperature
            if top_p is not None:
                params["top_p"] = top_p
            if stop is not None:
                params["stop_sequences"] = stop
            if model_kwargs:
                params["extra_body"] = {**params.get("extra_body", {}), **model_kwargs}
            _apply_effort(params, reasoning_effort)

            if tools:
                params["tools"] = self.convert_tools(tools)
                if tool_choice:
                    params["tool_choice"] = self._convert_tool_choice(tool_choice)

            # Add response format via tool use (Anthropic structured output)
            if response_format:
                fmt_type = response_format.get("type")
                if fmt_type == "json_schema":
                    schema_spec = response_format.get("json_schema", {})
                    structured_tool_name = schema_spec.get("name", "structured_output")
                    structured_tool = {
                        "name": structured_tool_name,
                        "description": f"Return response in {structured_tool_name} format. Always use this tool.",
                        "input_schema": schema_spec.get("schema", {}),
                    }
                    if "tools" not in params:
                        params["tools"] = []
                    params["tools"].append(structured_tool)
                    params["tool_choice"] = {"type": "tool", "name": structured_tool_name}
                elif fmt_type == "json_object":
                    sys_msgs = [m for m in params.get("messages", []) if m.get("role") == "system"]
                    if not sys_msgs:
                        params["system"] = _append_system_text(
                            params.get("system"), "\nYou must respond with valid JSON only."
                        )

            # Add MCP servers if provided — requires beta API
            use_beta = False
            if mcp_servers:
                mcp_server_params = []
                for s in mcp_servers:
                    server_param: dict[str, Any] = {
                        "type": "url",
                        "url": s["url"],
                        "name": s["name"],
                    }
                    if s.get("auth_token"):
                        server_param["authorization_token"] = s["auth_token"]
                    mcp_server_params.append(server_param)
                params["mcp_servers"] = mcp_server_params

                if "tools" not in params:
                    params["tools"] = []
                for s in mcp_servers:
                    params["tools"].append({
                        "type": "mcp_toolset",
                        "mcp_server_name": s["name"],
                    })
                use_beta = True

            # Use streaming context manager (beta API for MCP)
            if use_beta:
                stream_ctx = self._client.beta.messages.stream(
                    betas=["mcp-client-2025-11-20"],
                    **params,
                )
            else:
                stream_ctx = self._client.messages.stream(**params)
            async with stream_ctx as stream:
                current_tool_call: dict[str, Any] | None = None
                tool_input_json = ""

                async for event in stream:
                    # Handle text delta
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield StreamChunk(
                                content=event.delta.text,
                                done=False,
                            )
                        elif hasattr(event.delta, "partial_json"):
                            # Accumulate tool input
                            tool_input_json += event.delta.partial_json

                    # Handle tool use start
                    elif event.type == "content_block_start":
                        if hasattr(event.content_block, "type") and event.content_block.type == "tool_use":
                            current_tool_call = {
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": event.content_block.name,
                                    "arguments": "",
                                },
                            }
                            tool_input_json = ""

                    # Handle content block stop (tool use complete)
                    elif event.type == "content_block_stop":
                        if current_tool_call:
                            current_tool_call["function"]["arguments"] = tool_input_json
                            yield StreamChunk(
                                content="",
                                done=False,
                                tool_calls=[current_tool_call],
                            )
                            current_tool_call = None
                            tool_input_json = ""

                    # Handle message complete
                    elif event.type == "message_stop":
                        # Get final message for usage
                        final_message = await stream.get_final_message()
                        yield StreamChunk(
                            content="",
                            done=True,
                            usage=self._token_usage(final_message.usage),
                            finish_reason=self._map_stop_reason(final_message.stop_reason),
                        )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise ProviderError(
                message=f"Anthropic streaming error: {e}",
                provider=self.provider_name,
                status_code=provider_status_code(e),
                original_error=str(e),
            )

    # =========================================================================
    # Message Normalization
    # =========================================================================

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (system_message, normalized_messages)

        Anthropic format:
        - System message is separate from the messages array
        - Messages must alternate between user and assistant
        - Vision uses "image" content block type
        - Documents use "document" content block type
        - Tool results use "tool_result" content block type

        Supports both:
        - FileAttachment objects in the `files` key
        - Legacy dict format with `images`/`documents` keys
        """
        system_blocks: list[dict[str, Any]] = []
        normalized = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system message. With a cache_control breakpoint the system
            # prompt becomes a block list so the marker lands on the block
            # (Anthropic caches prefixes, and system is the first prefix).
            if role == "system":
                block = {"type": "text", "text": str(content)}
                if msg.get("cache_control"):
                    block["cache_control"] = msg["cache_control"]
                system_blocks.append(block)
                continue

            normalized.append(self._normalize_turn(role, content, msg))
            if msg.get("cache_control"):
                self._mark_cache_breakpoint(normalized[-1], msg["cache_control"])

        # A plain string system prompt keeps the historical wire shape (and
        # byte-identical requests); blocks only when a breakpoint needs one.
        if not system_blocks:
            system_message = None
        elif any("cache_control" in b for b in system_blocks):
            system_message = system_blocks
        else:
            system_message = "\n\n".join(b["text"] for b in system_blocks)
        return system_message, normalized

    @staticmethod
    def _token_usage(usage: Any) -> TokenUsage:
        """Anthropic usage → TokenUsage. input_tokens EXCLUDES cache reads and
        writes on this wire; both are surfaced separately so billing can
        price each tier (reads per-model/0.10x, writes 1.25x/2x)."""
        creation = getattr(usage, "cache_creation", None)
        write_5m = (getattr(creation, "ephemeral_5m_input_tokens", 0) or 0) if creation else 0
        write_1h = (getattr(creation, "ephemeral_1h_input_tokens", 0) or 0) if creation else 0
        if not creation:
            # Older usage shape: a single untiered total (5m is the default TTL).
            write_5m = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Anthropic reports thinking tokens (subset of output) on newer models
        # via output_tokens_details.thinking_tokens; absent → unknown (None).
        details = getattr(usage, "output_tokens_details", None)
        thinking = getattr(details, "thinking_tokens", None) if details is not None else None
        return TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
            reasoning_tokens=int(thinking) if thinking is not None else None,
        )

    def _normalize_turn(
        self, role: str, content: Any, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert one non-system message to an Anthropic message dict."""
        # Handle tool calls in assistant messages
        if role == "assistant" and msg.get("tool_calls"):
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": str(content)})

            for tc in msg["tool_calls"]:
                # Parse arguments if they're a string
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": args,
                })

            return {"role": "assistant", "content": content_blocks}

        # Handle tool results
        if role == "tool":
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": str(content),
                }],
            }

        # Handle files/images/documents in content
        files = msg.get("files", [])
        images = msg.get("images", [])
        documents = msg.get("documents", [])

        if files or images or documents:
            # Build multimodal content
            content_parts: list[dict[str, Any]] = []

            # Add text first (Anthropic prefers text before images)
            if content:
                content_parts.append({"type": "text", "text": str(content)})

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

            return {"role": role, "content": content_parts}

        # Check if content is already multimodal format
        if isinstance(content, list):
            return {"role": role, "content": content}
        return {"role": role, "content": str(content)}

    @staticmethod
    def _mark_cache_breakpoint(message: dict[str, Any], cache_control: dict) -> None:
        """Put the breakpoint on the LAST content block of a normalized
        message — the prefix up to and including it is what gets cached."""
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        elif isinstance(content, list) and content:
            content[-1] = {**content[-1], "cache_control": cache_control}

    def _process_file_attachment(
        self,
        attachment: FileAttachment | dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Process a file attachment into Anthropic content format.

        Handles both FileAttachment objects and legacy dict format.
        Routes to appropriate handler based on file type.
        """
        # Handle FileAttachment object
        if isinstance(attachment, FileAttachment):
            if attachment.type == FileType.IMAGE:
                return self._build_image_from_attachment(attachment)
            elif attachment.type in (FileType.PDF, FileType.DOCUMENT):
                return self._build_document_from_attachment(attachment)
            elif attachment.type in (FileType.CODE, FileType.TEXT):
                # Code and text are sent as text content
                return self._build_text_from_attachment(attachment)
            else:
                logger.warning(f"Unsupported file type for Anthropic: {attachment.type}")
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
        """Build image content block from FileAttachment."""
        if attachment.source == FileSource.URL:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": attachment.data,
                },
            }
        elif attachment.source == FileSource.BASE64:
            mime = attachment.mime_type or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": attachment.data,
                },
            }
        elif attachment.source == FileSource.PATH:
            # Read file and convert to base64
            import base64
            with open(attachment.data, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            mime = attachment.mime_type or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        else:
            raise ValueError(f"Unsupported image source: {attachment.source}")

    def _build_document_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """Build document content block from FileAttachment (for PDFs)."""
        if attachment.source == FileSource.BASE64:
            mime = attachment.mime_type or "application/pdf"
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": attachment.data,
                },
            }
        elif attachment.source == FileSource.URL:
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": attachment.data,
                },
            }
        elif attachment.source == FileSource.PATH:
            # Read file and convert to base64
            import base64
            with open(attachment.data, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            mime = attachment.mime_type or "application/pdf"
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        else:
            raise ValueError(f"Unsupported document source: {attachment.source}")

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
        """Build image content block for vision from legacy dict format."""
        if image.get("url"):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image["url"],
                },
            }
        elif image.get("base64"):
            mime = image.get("mime_type", "image/png")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": image["base64"],
                },
            }
        else:
            raise ValueError("Image must have either 'url' or 'base64' field")

    def _build_document_content(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Build document content block for PDFs from legacy dict format."""
        if doc.get("url"):
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": doc["url"],
                },
            }
        elif doc.get("base64"):
            mime = doc.get("mime_type", "application/pdf")
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": doc["base64"],
                },
            }
        elif doc.get("text"):
            # Plain text document - include as text
            return {
                "type": "text",
                "text": f"Document content:\n{doc['text']}",
            }
        else:
            raise ValueError("Document must have 'url', 'base64', or 'text' field")

    def _prepare_schema_for_anthropic(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a JSON schema for Anthropic's tool input_schema.

        Cleans up Pydantic-specific fields and handles $defs.
        """
        schema = schema.copy()

        # Handle $defs by inlining references (simplified approach)
        defs = schema.pop("$defs", {})

        def resolve_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        ref_name = ref_path.split("/")[-1]
                        if ref_name in defs:
                            return resolve_refs(defs[ref_name].copy())
                return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj

        schema = resolve_refs(schema)

        # Remove title if present (Anthropic doesn't need it)
        if "title" in schema:
            del schema["title"]

        return schema

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict[str, Any]:
        """Convert tool_choice to Anthropic format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            elif tool_choice == "required":
                return {"type": "any"}
            else:
                # Assume it's a tool name
                return {"type": "tool", "name": tool_choice}
        return tool_choice

    def _map_stop_reason(self, stop_reason: str | None) -> str:
        """Map Anthropic stop reasons to standard format."""
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        return mapping.get(stop_reason or "", stop_reason or "stop")

    # =========================================================================
    # Tool Conversion
    # =========================================================================

    def convert_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    # =========================================================================
    # Generation Methods (NOT SUPPORTED)
    # =========================================================================

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """
        NOT SUPPORTED: Anthropic does not provide embedding models.

        Alternatives:
        - OpenAI text-embedding-3-large (best quality)
        - OpenAI text-embedding-3-small (cost-effective)
        - Google text-embedding-004
        - Voyage AI (recommended by Anthropic)
        - Cohere embed-english-v3
        """
        raise NotImplementedError(
            "Anthropic does not support embeddings. "
            "Recommended alternatives:\n"
            "  - OpenAI: text-embedding-3-large (3072 dims, best quality)\n"
            "  - OpenAI: text-embedding-3-small (1536 dims, cost-effective)\n"
            "  - Google: text-embedding-004 (768 dims)\n"
            "  - Voyage AI: voyage-large-2 (recommended by Anthropic)\n"
            "  - Cohere: embed-english-v3 (available on Bedrock)"
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
        NOT SUPPORTED: Anthropic does not provide image generation.

        Alternatives:
        - OpenAI gpt-image-2 - Flagship, arbitrary resolutions, editing
        - Google gemini-3-pro-image - Highest quality, 4K, 14 reference images
        - Google gemini-3.1-flash-image - Fast native generation
        """
        raise NotImplementedError(
            "Anthropic does not support image generation. "
            "Recommended alternatives:\n"
            "  - OpenAI: gpt-image-2 (flagship, arbitrary resolutions, editing)\n"
            "  - Google: gemini-3-pro-image (highest quality, up to 4K)\n"
            "  - Google: gemini-3.1-flash-image (fast)"
        )

    async def transcribe(
        self,
        audio: FileAttachment,
        model: str | None = None,
        language: str | None = None,
    ) -> AudioResponse:
        """
        NOT SUPPORTED: Anthropic does not provide audio transcription.

        Note: Claude models can analyze audio in chat (via file attachments),
        but there is no dedicated transcription API.

        Alternatives:
        - OpenAI gpt-4o-transcribe (highest accuracy)
        - OpenAI whisper-1 (multilingual, stable)
        - Google Gemini (via generate_content with audio)
        - AWS Transcribe (via Bedrock/AWS)
        """
        raise NotImplementedError(
            "Anthropic does not support dedicated audio transcription. "
            "Recommended alternatives:\n"
            "  - OpenAI: gpt-4o-transcribe (highest accuracy)\n"
            "  - OpenAI: whisper-1 (multilingual, stable)\n"
            "  - Google: gemini-3-flash-preview (native audio understanding)\n"
            "  - AWS: Amazon Transcribe (dedicated service)\n\n"
            "Note: Claude can analyze audio content in chat messages, "
            "but this is for understanding, not transcription."
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
        NOT SUPPORTED: Anthropic does not provide text-to-speech.

        Alternatives:
        - OpenAI tts-1-hd (high quality)
        - OpenAI tts-1 (faster, standard quality)
        - Google Cloud Text-to-Speech
        - AWS Polly
        - ElevenLabs (for ultra-realistic voices)
        """
        raise NotImplementedError(
            "Anthropic does not support text-to-speech. "
            "Recommended alternatives:\n"
            "  - OpenAI: tts-1-hd (high quality, voices: alloy, echo, fable, onyx, nova, shimmer)\n"
            "  - OpenAI: tts-1 (faster, standard quality)\n"
            "  - Google: Cloud Text-to-Speech API\n"
            "  - AWS: Amazon Polly\n"
            "  - ElevenLabs: For ultra-realistic voice synthesis"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            # Make a minimal API call to check connectivity
            await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
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
