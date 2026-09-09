"""Google AI (Gemini) provider adapter using native SDK."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
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
from mlpal_assistants_service.services import gemini_cache

logger = logging.getLogger(__name__)


def _is_cache_missing_error(exc: Exception) -> bool:
    """True when a generate failed because the referenced cachedContent is gone
    (expired/deleted before our Redis mapping). Signals a one-shot uncached retry."""
    msg = str(exc).lower()
    return "cachedcontent" in msg or "cached_content" in msg or (
        "not found" in msg and "cache" in msg
    )

# Google's documented placeholder that bypasses strict thought-signature
# validation on echoed function calls. Used only when the real signature isn't
# available (e.g. a client didn't round-trip it) so multi-turn tool calls never
# hard-fail with a 400 — at the cost of reasoning continuity for that call.
_THOUGHT_SIG_SENTINEL = b"skip_thought_signature_validator"


# Per-model thinking_budget constraints, as (can_disable, min_active_budget):
# whether the model accepts thinking_budget=0 (fully disables thinking), and the
# smallest non-zero budget it allows. Budgets in the open interval
# (0, min_active) are rejected by the API — the valid set can be discontinuous
# (e.g. 2.5-flash-lite accepts {0} ∪ {>=512} but rejects 1..511). Verified
# against the live Gemini API 2026-07. Unlisted models default to (False, 1):
# thinking is mandatory on the newest tiers (3.6-flash, 3.5-flash-lite reject 0)
# and every current model accepts budget=1 — 2.5-flash-lite being the sole
# exception, listed explicitly. Pro models are handled separately (128 floor).
_THINKING_LIMITS: dict[str, tuple[bool, int]] = {
    "gemini-2.5-flash-lite": (True, 512),
    # Verified against the live API 2026-08-14: accepts 0 (disable), 1, 128,
    # and -1 (dynamic) — unlike 3.6-flash, thinking is fully optional here.
    "gemini-3.7-flash": (True, 1),
}


def _thinking_limits(model: str) -> tuple[bool, int]:
    if "pro" in model:
        return (False, 128)
    for key, lim in _THINKING_LIMITS.items():
        if key in model:
            return lim
    return (False, 1)


def _apply_thinking_output_reserve(
    config: dict[str, Any], model: str, max_tokens: int | None, has_tools: bool
) -> None:
    """Cap thinking so at least `google_thinking_output_reserve` output tokens
    remain within the caller's max_tokens budget.

    Gemini thinking models can spend a tight max_tokens entirely on reasoning and
    return empty text (stop_reason=max_tokens). Capping thinking_budget to
    (max_tokens - reserve) reserves output room. Strictly within the caller's
    own budget — thinking may still use everything up to the cap, and output can
    use more if thinking uses less. Skipped when tools are set (that path owns
    thinking_config for multi-turn thought signatures) or for non-thinking
    models. The desired cap is clamped into the model's valid budget set (see
    `_THINKING_LIMITS`): when it lands in a forbidden low range we either disable
    thinking (if the model allows it — best, frees the most output room) or lift
    to the model's minimum active budget, so we never emit an API-rejected value.
    """
    reserve = get_settings().google_thinking_output_reserve
    if has_tools or reserve <= 0 or max_tokens is None:
        return
    # A client-chosen effort (thinking_level) is an explicit instruction; the
    # budget cap below would silently replace it (and can disable thinking).
    if getattr(config.get("thinking_config"), "thinking_level", None):
        return
    if "gemini-3" not in model and "gemini-2.5" not in model:
        return
    can_disable, min_active = _thinking_limits(model)
    desired = int(max_tokens) - reserve
    if desired >= min_active:
        budget = desired
    else:
        budget = 0 if can_disable else min_active
    existing = config.get("thinking_config")
    include_thoughts = getattr(existing, "include_thoughts", None) if existing else None
    config["thinking_config"] = types.ThinkingConfig(
        thinking_budget=budget, include_thoughts=include_thoughts
    )


def _gemini_thought_tokens(usage_metadata: Any) -> int:
    """Thinking tokens. Google reports them OUTSIDE candidates_token_count and
    bills them as output — they must be added, not ignored."""
    return int(getattr(usage_metadata, "thoughts_token_count", 0) or 0) if usage_metadata else 0


def _gemini_output_tokens(usage_metadata: Any) -> int:
    if not usage_metadata:
        return 0
    return int(usage_metadata.candidates_token_count or 0) + _gemini_thought_tokens(usage_metadata)


def _gemini_token_usage(usage_metadata: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=(usage_metadata.prompt_token_count or 0) if usage_metadata else 0,
        output_tokens=_gemini_output_tokens(usage_metadata),
        cached_tokens=_gemini_cached_tokens(usage_metadata),
        reasoning_tokens=_gemini_thought_tokens(usage_metadata),
    )


def _gemini_cached_tokens(usage_metadata: Any) -> int:
    """Cached (context-cache hit) input tokens from Gemini usage_metadata.

    Surfaced on TokenUsage so callers (e.g. the /v2/messages usage block) can
    report cache reads as cache_read_input_tokens. Absent/None on responses
    without a cache hit, so defensively defaults to 0.
    """
    return int(getattr(usage_metadata, "cached_content_token_count", 0) or 0) if usage_metadata else 0


# Native Gemini image models share one capability profile: they accept image
# inputs (reference images) but none of the chat-side features.
_IMAGE_MODEL_CAPABILITIES = ModelCapabilities(
    supports_images=True,
    supports_pdf=False,
    supports_audio=False,
    supports_video=False,
    supports_tools=False,
    supports_structured_output=False,
    max_context_tokens=1000000,
    max_output_tokens=65536,
)


class GoogleAdapter(BaseAdapter):
    # GenerateContentConfig fields we forward via model_kwargs (snake_case,
    # as the google-genai SDK expects).
    SUPPORTED_KWARGS = frozenset(
        {
            "top_k", "candidate_count", "seed", "presence_penalty",
            "frequency_penalty", "safety_settings", "response_logprobs",
            "logprobs", "thinking_config", "media_resolution",
        }
    )

    """
    Adapter for Google AI (Gemini) API using native SDK.

    Supports:
    - Chat completion (text)
    - Structured output (JSON schema via response_schema)
    - Vision (image input via URL or base64)
    - Documents (PDFs)
    - Audio (mp3, wav, etc.)
    - Video (mp4, etc.)
    - Tool/function calling
    - Streaming

    Models:
    - gemini-3.1-pro-preview (latest, 4-level thinking)
    - gemini-3-flash-preview (fast, multimodal)
    - gemini-2.5-pro (stable, thinking capabilities)
    - gemini-2.5-flash (fast, efficient)
    """

    provider_name = "google"

    # Model capabilities registry - Gemini has broad multimodal support
    MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
        # Gemini 3.1 family - latest with 4-level thinking (minimal, low, medium, high)
        "gemini-3.1-pro-preview": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=True,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        ),
        # Gemini 3 family
        "gemini-3-flash-preview": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=True,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        ),
        "gemini-3-pro-image-preview": _IMAGE_MODEL_CAPABILITIES,
        "gemini-3-pro-image": _IMAGE_MODEL_CAPABILITIES,
        "gemini-3.1-flash-image": _IMAGE_MODEL_CAPABILITIES,
        "gemini-3.1-flash-lite-image": _IMAGE_MODEL_CAPABILITIES,
        # Gemini 2.5 family
        "gemini-2.5-pro": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=True,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        ),
        "gemini-2.5-flash": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=True,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        ),
        "gemini-2.5-flash-lite": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=True,
            supports_video=True,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=1000000,
            max_output_tokens=65536,
        ),
    }

    # Default capabilities for unknown models (assume Gemini 2.0 capabilities)
    DEFAULT_CAPABILITIES = ModelCapabilities(
        supports_images=True,
        supports_pdf=True,
        supports_audio=True,
        supports_video=True,
        supports_tools=True,
        supports_structured_output=True,
        max_context_tokens=1000000,
        max_output_tokens=8192,
    )

    def __init__(self, api_key: str | None = None) -> None:
        if api_key:
            self._api_key = api_key
        else:
            settings = get_settings()
            self._api_key = settings.google_api_key
        self._client = genai.Client(api_key=self._api_key)

    # =========================================================================
    # Model Capabilities
    # =========================================================================

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Google model."""
        # Check exact match first
        if model in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[model]

        # Check prefix matches (e.g., "gemini-2.0-flash-exp" matches "gemini-2.0-flash")
        for model_prefix, capabilities in self.MODEL_CAPABILITIES.items():
            if model.startswith(model_prefix):
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
        - Legacy dict formats in `files`, `images`, `documents`, `audio`, `video` keys
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
                else:
                    continue

                files.append(FileAttachment(
                    type=FileType.PDF,
                    source=source,
                    data=data,
                    mime_type=doc.get("mime_type", "application/pdf"),
                    filename=doc.get("filename"),
                ))

            # Handle legacy audio key (Google supports audio)
            for audio in msg.get("audio", []):
                if audio.get("url"):
                    source = FileSource.URL
                    data = audio["url"]
                elif audio.get("base64"):
                    source = FileSource.BASE64
                    data = audio["base64"]
                else:
                    continue

                files.append(FileAttachment(
                    type=FileType.AUDIO,
                    source=source,
                    data=data,
                    mime_type=audio.get("mime_type", "audio/mpeg"),
                    filename=audio.get("filename"),
                ))

            # Handle legacy video key (Google supports video)
            for video in msg.get("video", []):
                if video.get("url"):
                    source = FileSource.URL
                    data = video["url"]
                elif video.get("base64"):
                    source = FileSource.BASE64
                    data = video["base64"]
                else:
                    continue

                files.append(FileAttachment(
                    type=FileType.VIDEO,
                    source=source,
                    data=data,
                    mime_type=video.get("mime_type", "video/mp4"),
                    filename=video.get("filename"),
                ))

        return files

    # =========================================================================
    # Chat Completion
    # =========================================================================

    async def _apply_prefix_cache(
        self,
        config: dict[str, Any],
        model: str,
        tools: list[ToolDefinition] | None,
        tool_choice: str | dict | None,
        prefix_cache_ttl: int | None,
    ) -> dict[str, Any] | None:
        """Swap the stable system+tools prefix for a cachedContent reference.

        Gemini forbids `cached_content` alongside request-level tools/tool_config,
        so the tools must live IN the cache and the request runs in AUTO tool mode.
        That means a forced/"none" tool_choice can't use the cache — we skip it
        rather than silently change tool behaviour. On success the stripped
        {system_instruction, tools, tool_config} is returned so the caller can
        restore it for an uncached retry if the cache turns out to be gone.
        Returns None whenever the normal (uncached) path should run.
        """
        if prefix_cache_ttl is None:
            return None
        if tools and tool_choice not in (None, "auto"):
            return None  # cached tools only honour AUTO; don't drop a forced choice
        system_instruction = config.get("system_instruction")
        gtools = config.get("tools")
        name = await gemini_cache.get_or_create_cache(
            self._client, model, system_instruction, gtools, prefix_cache_ttl
        )
        if not name:
            return None
        stripped = {
            k: config.pop(k)
            for k in ("system_instruction", "tools", "tool_config")
            if k in config
        }
        config["cached_content"] = name
        return stripped

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
        prefix_cache_ttl: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> AdapterResponse:
        """Execute chat completion via Google AI API.

        `prefix_cache_ttl` (set only by the v2 edge when the client marks a
        cache breakpoint) opts this request into explicit prefix caching.
        """
        start_time = time.perf_counter()

        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            # Normalize messages
            gemini_contents, system_instruction = self._normalize_messages(messages)

            # Build config
            config: dict[str, Any] = {
                "temperature": temperature,
            }

            if max_tokens is not None:
                config["max_output_tokens"] = max_tokens
            if top_p is not None:
                config["top_p"] = top_p
            if stop is not None:
                config["stop_sequences"] = stop
            if system_instruction:
                config["system_instruction"] = system_instruction

            # Add tools if provided
            if tools:
                config["tools"] = self._convert_tools(tools)
                if tool_choice:
                    config["tool_config"] = self._convert_tool_choice(tool_choice)
                # Enable thinking mode for thought_signature support in multi-turn tool calling
                # Gemini 3.1 uses 4-level thinking (minimal, low, medium, high)
                # Gemini 3 uses 2-level thinking (low, high) - deprecated March 9, 2026
                # Gemini 2.5 uses thinking_budget
                if "gemini-3.1" in model:
                    # Gemini 3.1 has 4 thinking levels: minimal, low, medium, high
                    # Use "low" for tool calling - good balance of latency and signature support
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level="low",
                    )
                elif "gemini-3" in model:
                    # Gemini 3 (deprecated) - 2 levels: low, high
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level="low",
                    )
                elif "gemini-2.5" in model:
                    config["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

            # Add structured output if response_format specified
            if response_format:
                if response_format.get("type") == "json_schema":
                    config["response_mime_type"] = "application/json"
                    schema = response_format.get("json_schema", {}).get("schema", {})
                    if schema:
                        config["response_schema"] = self._prepare_schema_for_google(schema)
                elif response_format.get("type") == "json_object":
                    config["response_mime_type"] = "application/json"

            # Reject MCP servers (Google doesn't have native MCP support yet)
            if mcp_servers:
                raise UnsupportedCapabilityError(
                    capability="mcp",
                    model=model,
                    suggestions=["gpt-5.2", "claude-opus-4.5"],
                )

            # Model-specific kwargs, pre-validated by the caller: merged into
            # GenerateContentConfig (the SDK accepts plain dicts for nested types).
            if model_kwargs:
                config.update(model_kwargs)
            # Universal effort (resolved to a level this model accepts) wins
            # over the tool-calling default level set above.
            if reasoning_effort:
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort,
                    include_thoughts=getattr(config.get("thinking_config"), "include_thoughts", None),
                )

            _apply_thinking_output_reserve(config, model, max_tokens, has_tools=bool(tools))

            cached = await self._apply_prefix_cache(
                config, model, tools, tool_choice, prefix_cache_ttl
            )

            # Make API call. If a referenced cache turned out to be gone, drop the
            # stale mapping, restore the inline prefix, and retry once uncached.
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config),
                )
            except Exception as e:  # noqa: BLE001
                if not (cached and _is_cache_missing_error(e)):
                    raise
                await gemini_cache.invalidate(
                    model, cached.get("system_instruction"), cached.get("tools")
                )
                config.pop("cached_content", None)
                config.update(cached)
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config),
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract content and tool calls
            content = ""
            tool_calls = None

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        content += part.text
                    elif hasattr(part, "function_call") and part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        tool_call_data = {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        }
                        # Include thought_signature for Gemini 3 models (required for multi-turn tool calling)
                        # Try multiple locations where thought_signature might be
                        import base64
                        thought_sig = None
                        # Check on the part itself
                        if hasattr(part, "thought_signature") and part.thought_signature:
                            thought_sig = part.thought_signature
                            logger.debug(f"Found thought_signature on part: {len(thought_sig)} bytes")
                        # Check on the function_call object
                        elif hasattr(part.function_call, "thought_signature") and part.function_call.thought_signature:
                            thought_sig = part.function_call.thought_signature
                            logger.debug(f"Found thought_signature on function_call: {len(thought_sig)} bytes")
                        else:
                            # Log available attributes for debugging
                            logger.debug(f"No thought_signature found. Part attrs: {[a for a in dir(part) if not a.startswith('_')]}")
                            if part.function_call:
                                logger.debug(f"FunctionCall attrs: {[a for a in dir(part.function_call) if not a.startswith('_')]}")

                        if thought_sig:
                            tool_call_data["thought_signature"] = base64.b64encode(thought_sig).decode("utf-8")
                        tool_calls.append(tool_call_data)

            # Get token counts
            usage = _gemini_token_usage(response.usage_metadata)

            return AdapterResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=usage,
                finish_reason=self._map_finish_reason(response),
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                raw_response={"text": content},
            )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"Google AI chat error after {latency_ms}ms: {error_message}")

            if "quota" in error_message.lower() or "rate" in error_message.lower():
                raise ProviderUnavailableError(
                    message="Google AI rate limit exceeded",
                    provider=self.provider_name,
                    original_error=error_message,
                )
            raise ProviderError(
                message=f"Google AI error: {error_message}",
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
        prefix_cache_ttl: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute streaming chat completion.

        `prefix_cache_ttl` opts into explicit prefix caching (see `chat`).
        """
        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            gemini_contents, system_instruction = self._normalize_messages(messages)

            config: dict[str, Any] = {
                "temperature": temperature,
            }

            if max_tokens is not None:
                config["max_output_tokens"] = max_tokens
            if top_p is not None:
                config["top_p"] = top_p
            if stop is not None:
                config["stop_sequences"] = stop
            if system_instruction:
                config["system_instruction"] = system_instruction

            if tools:
                config["tools"] = self._convert_tools(tools)
                if tool_choice:
                    config["tool_config"] = self._convert_tool_choice(tool_choice)
                # Enable thinking mode for thought_signature support in multi-turn tool calling
                # Gemini 3.1 uses 4-level thinking (minimal, low, medium, high)
                # Gemini 3 uses 2-level thinking (low, high) - deprecated March 9, 2026
                # Gemini 2.5 uses thinking_budget
                if "gemini-3.1" in model:
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level="low",
                    )
                elif "gemini-3" in model:
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level="low",
                    )
                elif "gemini-2.5" in model:
                    config["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)

            # Surface the model's thoughts (thinking) as a distinct channel when
            # the caller opts in. Augments any tool-driven thinking_config rather
            # than overriding its level/budget.
            if stream_thinking and ("gemini-3" in model or "gemini-2.5" in model):
                tc = config.get("thinking_config")
                if tc is not None:
                    tc.include_thoughts = True
                else:
                    config["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

            # Add structured output if response_format specified
            if response_format:
                if response_format.get("type") == "json_schema":
                    config["response_mime_type"] = "application/json"
                    schema = response_format.get("json_schema", {}).get("schema", {})
                    if schema:
                        config["response_schema"] = self._prepare_schema_for_google(schema)
                elif response_format.get("type") == "json_object":
                    config["response_mime_type"] = "application/json"

            # Reject MCP servers (Google doesn't have native MCP support yet)
            if mcp_servers:
                raise UnsupportedCapabilityError(
                    capability="mcp",
                    model=model,
                    suggestions=["gpt-5.2", "claude-opus-4.5"],
                )

            # Model-specific kwargs, pre-validated by the caller: merged into
            # GenerateContentConfig (the SDK accepts plain dicts for nested types).
            if model_kwargs:
                config.update(model_kwargs)
            # Universal effort (resolved to a level this model accepts) wins
            # over the tool-calling default level set above.
            if reasoning_effort:
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=reasoning_effort,
                    include_thoughts=getattr(config.get("thinking_config"), "include_thoughts", None),
                )

            _apply_thinking_output_reserve(config, model, max_tokens, has_tools=bool(tools))

            cached = await self._apply_prefix_cache(
                config, model, tools, tool_choice, prefix_cache_ttl
            )

            # Stream is created before any chunk is yielded, so a missing-cache
            # error here is safe to recover: restore the inline prefix and retry
            # uncached without the client ever seeing a partial stream.
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config),
                )
            except Exception as e:  # noqa: BLE001
                if not (cached and _is_cache_missing_error(e)):
                    raise
                await gemini_cache.invalidate(
                    model, cached.get("system_instruction"), cached.get("tools")
                )
                config.pop("cached_content", None)
                config.update(cached)
                stream = await self._client.aio.models.generate_content_stream(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(**config),
                )

            total_input_tokens = 0
            total_output_tokens = 0
            total_reasoning_tokens = 0
            total_cached_tokens = 0
            pending_tool_calls: list[dict[str, Any]] = []

            async for chunk in stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            # Gemini marks reasoning parts with part.thought=True;
                            # route them to the thinking channel, keep the answer
                            # in content.
                            if getattr(part, "thought", False):
                                yield StreamChunk(content="", done=False, thinking=part.text)
                            else:
                                yield StreamChunk(content=part.text, done=False)
                        elif hasattr(part, "function_call") and part.function_call:
                            tool_call_data = {
                                "id": f"call_{len(pending_tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(dict(part.function_call.args)),
                                },
                            }
                            # Include thought_signature for Gemini 3 models
                            import base64
                            thought_sig = None
                            if hasattr(part, "thought_signature") and part.thought_signature:
                                thought_sig = part.thought_signature
                            elif hasattr(part.function_call, "thought_signature") and part.function_call.thought_signature:
                                thought_sig = part.function_call.thought_signature
                            if thought_sig:
                                tool_call_data["thought_signature"] = base64.b64encode(thought_sig).decode("utf-8")
                            pending_tool_calls.append(tool_call_data)

                # Track tokens
                if chunk.usage_metadata:
                    total_input_tokens = chunk.usage_metadata.prompt_token_count or 0
                    total_output_tokens = _gemini_output_tokens(chunk.usage_metadata)
                    total_reasoning_tokens = _gemini_thought_tokens(chunk.usage_metadata)
                    total_cached_tokens = _gemini_cached_tokens(chunk.usage_metadata)

            # Emit tool calls if any
            if pending_tool_calls:
                yield StreamChunk(
                    content="",
                    done=False,
                    tool_calls=pending_tool_calls,
                )

            # Final chunk
            yield StreamChunk(
                content="",
                done=True,
                usage=TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    reasoning_tokens=total_reasoning_tokens,
                    cached_tokens=total_cached_tokens,
                ),
                finish_reason="stop",
            )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            logger.error(f"Google AI streaming error: {e}")
            raise ProviderError(
                message=f"Google AI streaming error: {e}",
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
    ) -> tuple[list[types.Content], str | None]:
        """
        Convert messages to Gemini format.

        Returns:
            Tuple of (contents, system_instruction)

        Supports both:
        - FileAttachment objects in the `files` key
        - Legacy dict format with `images`/`documents`/`audio`/`video` keys
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system message
            if role == "system":
                if system_instruction:
                    system_instruction += "\n\n" + str(content)
                else:
                    system_instruction = str(content)
                continue

            # Map roles
            gemini_role = "model" if role == "assistant" else "user"

            # Handle tool calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                parts: list[types.Part] = []
                if content:
                    parts.append(types.Part.from_text(text=str(content)))

                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    # Create function call part
                    fc_part = types.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=args or {},
                    )
                    # Thought signatures are required on echoed function calls for
                    # Gemini thinking models, or the API 400s. Replay the real one
                    # when the client sent it back; otherwise fall back to Google's
                    # documented sentinel so we never hard-fail (at the cost of
                    # reasoning continuity for that call).
                    if tc.get("thought_signature"):
                        import base64
                        fc_part.thought_signature = base64.b64decode(tc["thought_signature"])
                    else:
                        fc_part.thought_signature = _THOUGHT_SIG_SENTINEL
                    parts.append(fc_part)

                contents.append(types.Content(role=gemini_role, parts=parts))
                continue

            # Handle tool results
            if role == "tool":
                tool_parts = [types.Part.from_function_response(
                    name=msg.get("name", "unknown"),
                    response={"content": str(content)},
                )]
                contents.append(types.Content(role="user", parts=tool_parts))
                continue

            # Build parts for regular messages
            parts = []

            # Add text content
            if content:
                parts.append(types.Part.from_text(text=str(content)))

            # Process files (supports both FileAttachment and dict)
            files = msg.get("files", [])
            for f in files:
                part = self._process_file_attachment(f)
                if part:
                    parts.append(part)

            # Add images directly (legacy format)
            for img in msg.get("images", []):
                part = self._build_image_part(img)
                if part:
                    parts.append(part)

            # Add documents directly (legacy format - PDFs)
            for doc in msg.get("documents", []):
                part = self._build_document_part(doc)
                if part:
                    parts.append(part)

            # Add audio directly (legacy format)
            for audio in msg.get("audio", []):
                part = self._build_audio_part(audio)
                if part:
                    parts.append(part)

            # Add video directly (legacy format)
            for video in msg.get("video", []):
                part = self._build_video_part(video)
                if part:
                    parts.append(part)

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return contents, system_instruction

    def _process_file_attachment(
        self,
        attachment: FileAttachment | dict[str, Any],
    ) -> types.Part | None:
        """
        Process a file attachment into a Gemini Part.

        Handles both FileAttachment objects and legacy dict format.
        Routes to appropriate handler based on file type.
        """
        # Handle FileAttachment object
        if isinstance(attachment, FileAttachment):
            if attachment.type == FileType.IMAGE:
                return self._build_part_from_attachment(attachment)
            elif attachment.type in (FileType.PDF, FileType.DOCUMENT):
                return self._build_part_from_attachment(attachment)
            elif attachment.type == FileType.AUDIO:
                return self._build_part_from_attachment(attachment)
            elif attachment.type == FileType.VIDEO:
                return self._build_part_from_attachment(attachment)
            elif attachment.type in (FileType.CODE, FileType.TEXT):
                # Code and text are sent as text content
                return self._build_text_part_from_attachment(attachment)
            else:
                logger.warning(f"Unsupported file type for Google: {attachment.type}")
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
            return self._build_image_part(attachment)
        elif file_type in ("pdf", "document"):
            return self._build_document_part(attachment)
        elif file_type == "audio":
            return self._build_audio_part(attachment)
        elif file_type == "video":
            return self._build_video_part(attachment)
        elif file_type in ("code", "text"):
            # Include as text
            text_content = attachment.get("text") or attachment.get("data", "")
            filename = attachment.get("filename", "file")
            return types.Part.from_text(text=f"[{filename}]\n{text_content}")
        else:
            logger.warning(f"Unknown file type in dict: {file_type}")
            return None

    def _build_part_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> types.Part | None:
        """Build a Gemini Part from FileAttachment for binary data (image, audio, video, PDF)."""
        import base64 as b64_module

        import httpx

        mime = attachment.mime_type or "application/octet-stream"

        if attachment.source == FileSource.BASE64:
            data = b64_module.b64decode(attachment.data)
            return types.Part.from_bytes(data=data, mime_type=mime)

        elif attachment.source == FileSource.URL:
            url = attachment.data
            # Google supports gs:// URIs directly
            if url.startswith("gs://"):
                return types.Part.from_uri(file_uri=url, mime_type=mime)
            else:
                # Fetch the file and convert to bytes
                try:
                    with httpx.Client(timeout=60.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        data = response.content
                        # Detect mime type from response if not provided
                        if attachment.mime_type is None:
                            content_type = response.headers.get("content-type", mime)
                            mime = content_type.split(";")[0].strip()
                        return types.Part.from_bytes(data=data, mime_type=mime)
                except Exception as e:
                    logger.warning(f"Failed to fetch file from URL: {e}")
                    return None

        elif attachment.source == FileSource.PATH:
            with open(attachment.data, "rb") as f:
                data = f.read()
            return types.Part.from_bytes(data=data, mime_type=mime)

        elif attachment.source == FileSource.FILE_ID:
            # Google file URI
            return types.Part.from_uri(file_uri=attachment.data, mime_type=mime)

        else:
            logger.warning(f"Unsupported file source: {attachment.source}")
            return None

    def _build_text_part_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> types.Part:
        """Build a text Part from code/text FileAttachment."""
        import base64 as b64_module

        if attachment.source == FileSource.PATH:
            with open(attachment.data, encoding="utf-8") as f:
                text_content = f.read()
        elif attachment.source == FileSource.BASE64:
            text_content = b64_module.b64decode(attachment.data).decode("utf-8")
        elif attachment.source == FileSource.URL:
            # Can't fetch URL content here - would need async
            text_content = f"[File from URL: {attachment.data}]"
        else:
            text_content = attachment.data

        filename = attachment.filename or "file"
        return types.Part.from_text(text=f"[{filename}]\n{text_content}")

    def _build_image_part(self, image: dict[str, Any]) -> types.Part | None:
        """Build image part for vision from legacy dict format."""
        import base64 as b64_module
        if image.get("base64"):
            image_data = b64_module.b64decode(image["base64"])
            return types.Part.from_bytes(
                data=image_data,
                mime_type=image.get("mime_type", "image/png"),
            )
        elif image.get("url"):
            url = image["url"]
            # Google only supports gs:// URIs directly
            # For http(s) URLs, we need to fetch and convert to bytes
            if url.startswith("gs://"):
                return types.Part.from_uri(
                    file_uri=url,
                    mime_type=image.get("mime_type", "image/png"),
                )
            else:
                # Fetch the image and convert to bytes
                import httpx
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        image_data = response.content
                        # Detect mime type from response or use provided
                        mime_type = image.get("mime_type")
                        if not mime_type:
                            content_type = response.headers.get("content-type", "image/png")
                            mime_type = content_type.split(";")[0].strip()
                        return types.Part.from_bytes(
                            data=image_data,
                            mime_type=mime_type,
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch image from URL: {e}")
                    return None
        return None

    def _build_document_part(self, doc: dict[str, Any]) -> types.Part | None:
        """Build document part for PDFs from legacy dict format."""
        import base64 as b64_module

        import httpx

        mime_type = doc.get("mime_type", "application/pdf")

        if doc.get("base64"):
            data = b64_module.b64decode(doc["base64"])
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        elif doc.get("url"):
            url = doc["url"]
            if url.startswith("gs://"):
                return types.Part.from_uri(file_uri=url, mime_type=mime_type)
            else:
                try:
                    with httpx.Client(timeout=60.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        data = response.content
                        return types.Part.from_bytes(data=data, mime_type=mime_type)
                except Exception as e:
                    logger.warning(f"Failed to fetch document from URL: {e}")
                    return None
        return None

    def _build_audio_part(self, audio: dict[str, Any]) -> types.Part | None:
        """Build audio part from legacy dict format."""
        import base64 as b64_module

        import httpx

        mime_type = audio.get("mime_type", "audio/mpeg")

        if audio.get("base64"):
            data = b64_module.b64decode(audio["base64"])
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        elif audio.get("url"):
            url = audio["url"]
            if url.startswith("gs://"):
                return types.Part.from_uri(file_uri=url, mime_type=mime_type)
            else:
                try:
                    with httpx.Client(timeout=60.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        data = response.content
                        return types.Part.from_bytes(data=data, mime_type=mime_type)
                except Exception as e:
                    logger.warning(f"Failed to fetch audio from URL: {e}")
                    return None
        return None

    def _build_video_part(self, video: dict[str, Any]) -> types.Part | None:
        """Build video part from legacy dict format."""
        import base64 as b64_module

        import httpx

        mime_type = video.get("mime_type", "video/mp4")

        if video.get("base64"):
            data = b64_module.b64decode(video["base64"])
            return types.Part.from_bytes(data=data, mime_type=mime_type)
        elif video.get("url"):
            url = video["url"]
            if url.startswith("gs://"):
                return types.Part.from_uri(file_uri=url, mime_type=mime_type)
            else:
                try:
                    with httpx.Client(timeout=120.0) as client:
                        response = client.get(url)
                        response.raise_for_status()
                        data = response.content
                        return types.Part.from_bytes(data=data, mime_type=mime_type)
                except Exception as e:
                    logger.warning(f"Failed to fetch video from URL: {e}")
                    return None
        return None

    def _convert_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to Google format."""
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })

        return [{"function_declarations": function_declarations}]

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict[str, Any]:
        """Convert tool_choice to Google format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"function_calling_config": {"mode": "AUTO"}}
            elif tool_choice == "none":
                return {"function_calling_config": {"mode": "NONE"}}
            elif tool_choice == "required":
                return {"function_calling_config": {"mode": "ANY"}}
            else:
                # Specific tool name
                return {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool_choice],
                    }
                }
        return tool_choice

    def _prepare_schema_for_google(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a JSON schema for Google's response_schema.

        Cleans up Pydantic-specific fields and handles $defs.
        """
        schema = schema.copy()

        # Handle $defs by inlining references
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

        # Strip fields Google doesn't support
        def strip_unsupported(obj: Any) -> Any:
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if k in ("title", "additionalProperties", "$schema"):
                        continue
                    cleaned[k] = strip_unsupported(v)
                return cleaned
            elif isinstance(obj, list):
                return [strip_unsupported(item) for item in obj]
            return obj

        return strip_unsupported(schema)

    def _map_finish_reason(self, response: Any) -> str:
        """Map Gemini finish reason to standard format."""
        if response.candidates and response.candidates[0].finish_reason:
            reason = str(response.candidates[0].finish_reason)
            if "STOP" in reason:
                return "stop"
            elif "MAX_TOKENS" in reason:
                return "length"
            elif "SAFETY" in reason:
                return "content_filter"
            elif "FUNCTION" in reason:
                return "tool_calls"
        return "stop"

    # =========================================================================
    # Generation Methods
    # =========================================================================

    # Default models for generation
    DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
    DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"  # Native image generation (Nano Banana Pro)
    DEFAULT_TRANSCRIPTION_MODEL = "gemini-3-flash-preview"  # Native audio understanding
    DEFAULT_TTS_MODEL = "gemini-2.5-pro-preview-tts"  # TTS via generate_content

    # Size conversion now handled by ImageSizeResolver in base.py

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings using Google's embedding model.

        Args:
            texts: List of text strings to embed
            model: Model ID (default: text-embedding-004)
            dimensions: Output dimensions (optional)

        Returns:
            EmbeddingResponse with embedding vectors

        Models:
            - text-embedding-004: Latest Google embedding model (768 dimensions)
            - textembedding-gecko@003: Legacy model
        """
        model = model or self.DEFAULT_EMBEDDING_MODEL
        start_time = time.perf_counter()

        try:
            # Google embedding API
            embeddings = []

            for text in texts:
                result = await self._client.aio.models.embed_content(
                    model=model,
                    contents=text,
                )

                if result.embeddings:
                    embeddings.append(list(result.embeddings[0].values))

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            actual_dimensions = len(embeddings[0]) if embeddings else 0

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                provider=self.provider_name,
                usage=TokenUsage(
                    input_tokens=sum(len(t.split()) for t in texts),  # Approximate
                    output_tokens=0,
                ),
                latency_ms=latency_ms,
                dimensions=actual_dimensions,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Google embedding error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Google embedding error: {e}",
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
        Generate images using Google's image generation models.

        Supports:
        - Text-to-image: Just provide a prompt
        - Image-to-image: Provide prompt + reference_images

        Args:
            prompt: Text description of the image to generate
            model: Model ID (default: gemini-3-pro-image)
            size: Aspect ratio / preset / pixels; pixels also select the
                resolution tier (see ImageSizeResolver.to_image_size_google)
            quality: "hd" renders at 2K when the size does not already imply
                a tier — Gemini has no separate quality knob
            n: Accepted for interface parity; Gemini returns one image per call
            reference_images: Optional reference images for image-to-image

        Returns:
            ImageGenerationResponse with generated images

        Models (native Gemini image generation, all support reference images):
            - gemini-3-pro-image: highest quality, 1K/2K/4K
            - gemini-3.1-flash-image: fast, 512px/1K/2K/4K
            - gemini-3.1-flash-lite-image: cheapest, 1K only
        """
        import base64

        model = model or self.DEFAULT_IMAGE_MODEL
        start_time = time.perf_counter()

        try:
            aspect_ratio = ImageSizeResolver.to_aspect_ratio_google(size)
            image_size = ImageSizeResolver.to_image_size_google(size, quality, model)

            # Build content parts
            content_parts: list[Any] = [prompt]

            # Add reference images if provided
            if reference_images:
                for ref in reference_images:
                    if ref.source == FileSource.BASE64:
                        img_data = base64.b64decode(ref.data)
                    elif ref.source == FileSource.PATH:
                        with open(ref.data, "rb") as f:
                            img_data = f.read()
                    elif ref.source == FileSource.URL:
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                            response = await client.get(ref.data)
                            response.raise_for_status()
                            img_data = response.content
                    else:
                        continue

                    content_parts.append(types.Part.from_bytes(
                        data=img_data,
                        mime_type=ref.mime_type or "image/png",
                    ))

            # Configure for image generation with aspect ratio
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                ),
            )

            response = await self._client.aio.models.generate_content(
                model=model,
                contents=content_parts,
                config=config,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract images from response
            images = []
            for candidate in response.candidates or []:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            # Get image data
                            img_bytes = part.inline_data.data
                            mime_type = part.inline_data.mime_type or "image/png"
                            fmt = mime_type.split("/")[-1] if "/" in mime_type else "png"

                            images.append(GeneratedImage(
                                data=img_bytes,
                                base64=base64.b64encode(img_bytes).decode("utf-8") if img_bytes else None,
                                format=fmt,
                            ))

            return ImageGenerationResponse(
                images=images,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                prompt=prompt,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Google image generation error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Google image generation error: {e}",
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
        Transcribe audio to text using Gemini's native audio understanding.

        Args:
            audio: Audio file attachment (mp3, wav, m4a, etc.)
            model: Model ID (default: gemini-3-flash-preview)
            language: Optional language code

        Returns:
            AudioResponse with transcribed text

        Models:
            - gemini-3-flash-preview: Fast, native audio understanding
            - gemini-3.1-pro-preview: Higher quality transcription
        """
        import base64

        model = model or self.DEFAULT_TRANSCRIPTION_MODEL
        start_time = time.perf_counter()

        try:
            # Get audio bytes
            if audio.source == FileSource.PATH:
                with open(audio.data, "rb") as f:
                    audio_bytes = f.read()
            elif audio.source == FileSource.BASE64:
                audio_bytes = base64.b64decode(audio.data)
            elif audio.source == FileSource.URL:
                import httpx
                with httpx.Client(timeout=60.0) as client:
                    response = client.get(audio.data)
                    response.raise_for_status()
                    audio_bytes = response.content
            else:
                raise ValueError(f"Unsupported audio source: {audio.source}")

            mime_type = audio.mime_type or "audio/mpeg"

            # Build content with audio and transcription prompt
            prompt = "Transcribe this audio accurately. Return only the transcription."
            if language:
                prompt += f" The audio is in {language}."

            content_parts = [
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt,
            ]

            response = await self._client.aio.models.generate_content(
                model=model,
                contents=content_parts,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract transcription
            text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

            return AudioResponse(
                content=text.strip(),
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                language=language,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Google transcription error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Google transcription error: {e}",
                provider=self.provider_name,
                original_error=str(e),
            )

    async def text_to_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str = "default",
        format: str = "mp3",
        speed: float = 1.0,
    ) -> AudioResponse:
        """
        Convert text to speech using Google's TTS capabilities.

        Note: Gemini's TTS is via generate_content with audio output.
        For production TTS, consider using Google Cloud Text-to-Speech API.

        Args:
            text: Text to convert to speech
            model: Model ID (default: gemini-2.5-pro-preview-tts)
            voice: Voice name (limited support in Gemini)
            format: Audio format (mp3, wav)
            speed: Playback speed (may not be supported)

        Returns:
            AudioResponse with audio bytes

        Models:
            - gemini-2.5-pro-preview-tts: Experimental TTS
        """
        model = model or self.DEFAULT_TTS_MODEL
        start_time = time.perf_counter()

        try:
            # Configure for audio output
            config = types.GenerateContentConfig(
                response_mime_type=f"audio/{format}",
            )

            response = await self._client.aio.models.generate_content(
                model=model,
                contents=f"Convert this text to natural speech: {text}",
                config=config,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract audio from response
            audio_bytes = None
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        audio_bytes = part.inline_data.data
                        break

            if not audio_bytes:
                # TTS may not be fully supported yet
                raise ProviderError(
                    message="Google TTS did not return audio. Consider using Google Cloud TTS API.",
                    provider=self.provider_name,
                    original_error="No audio in response",
                )

            return AudioResponse(
                content=audio_bytes,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                format=format,
            )

        except ProviderError:
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Google TTS error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Google TTS error: {e}. Consider using Google Cloud TTS API for production.",
                provider=self.provider_name,
                original_error=str(e),
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if Google AI API is accessible."""
        try:
            # Make a minimal API call to verify connectivity
            # Use list() which returns a pager object
            models_response = await self._client.aio.models.list()
            # Just access the response to verify it worked
            _ = models_response
            return True
        except Exception as e:
            logger.warning(f"Google AI health check failed: {e}")
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

    @property
    def supports_audio(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return True
