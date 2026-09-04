"""Chat completion schemas."""

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from mlpal_assistants_service.schemas.common import BaseSchema
from mlpal_assistants_service.schemas.routing import RoutingMetadata

# =============================================================================
# File Attachments
# =============================================================================


class FileAttachment(BaseSchema):
    """File attachment in a message."""

    type: Literal["image", "audio", "document"] = Field(..., description="File type")
    url: str | None = Field(default=None, description="URL to the file")
    base64: str | None = Field(default=None, description="Base64-encoded file content")
    mime_type: str | None = Field(default=None, description="MIME type")

    @field_validator("url", "base64")
    @classmethod
    def validate_file_source(cls, v: str | None, info: Any) -> str | None:
        """Ensure at least one file source is provided."""
        return v


# =============================================================================
# Tool Definitions and Calls
# =============================================================================


class FunctionDefinition(BaseSchema):
    """Function definition within a tool."""

    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    parameters: dict[str, Any] = Field(..., description="JSON Schema for parameters")
    strict: bool = Field(default=False, description="Strict schema validation")


class ToolDefinitionSchema(BaseSchema):
    """Tool definition in OpenAI format."""

    type: Literal["function"] = Field(default="function", description="Tool type")
    function: FunctionDefinition = Field(..., description="Function definition")


class FunctionCallSchema(BaseSchema):
    """A function call made by the model."""

    name: str = Field(..., description="Function name")
    arguments: str = Field(..., description="JSON string of arguments")


class ToolCallSchema(BaseSchema):
    """A tool call made by the model."""

    id: str = Field(..., description="Tool call ID")
    type: Literal["function"] = Field(default="function", description="Tool type")
    function: FunctionCallSchema = Field(..., description="Function call details")
    thought_signature: str | None = Field(
        default=None,
        description="Thought signature for Gemini 3 multi-turn tool calling"
    )


# =============================================================================
# Structured Output
# =============================================================================


class JsonSchemaSpec(BaseSchema):
    """JSON schema specification for structured output."""

    name: str = Field(..., description="Schema name")
    description: str | None = Field(default=None, description="Schema description")
    schema_: dict[str, Any] = Field(..., alias="schema", description="JSON Schema object")
    strict: bool = Field(default=True, description="Strict schema validation")

    model_config = {"populate_by_name": True}


class ResponseFormat(BaseSchema):
    """Response format specification."""

    type: Literal["json_schema", "json_object", "text"] = Field(
        default="text", description="Response format type"
    )
    json_schema: JsonSchemaSpec | None = Field(
        default=None, description="JSON schema for structured output (type='json_schema')"
    )


# =============================================================================
# MCP Configuration
# =============================================================================


class MCPServerConfig(BaseSchema):
    """Configuration for an MCP server (pass-through to provider)."""

    url: str = Field(..., description="MCP server URL (SSE endpoint)")
    name: str = Field(..., description="Human-readable server name")
    auth_token: str | None = Field(default=None, description="Bearer token for auth")
    allowed_tools: list[str] | None = Field(
        default=None, description="Restrict which tools are available"
    )
    require_approval: str | dict | None = Field(
        default=None, description="'never' | 'always' | per-tool config"
    )


# =============================================================================
# Messages
# =============================================================================


class CacheControl(BaseSchema):
    """Anthropic-style prompt-cache breakpoint, exposed on the OpenAI wire."""

    type: Literal["ephemeral"] = Field(default="ephemeral")
    ttl: Literal["5m", "1h"] | None = Field(
        default=None, description="Cache TTL; provider default (5m) when omitted."
    )


class ChatMessage(BaseSchema):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Message role",
    )
    content: str | None = Field(
        default=None,
        description="Message content (None allowed for assistant messages with only tool_calls)",
    )
    files: list[FileAttachment] | None = Field(
        default=None,
        description="Optional file attachments (for vision/multimodal)",
    )
    tool_calls: list[ToolCallSchema] | None = Field(
        default=None,
        description="Tool calls made by assistant (assistant messages only)",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Tool call ID this message responds to (tool messages only)",
    )
    name: str | None = Field(
        default=None,
        description="Optional name for tool messages",
    )
    cache_control: CacheControl | None = Field(
        default=None,
        description=(
            "Prompt-cache breakpoint after this message (Anthropic models; "
            "others cache automatically and ignore it). Everything up to and "
            "including this message becomes the cached prefix. Max 4 per request."
        ),
    )


class ChatCompletionRequest(BaseSchema):
    """Request schema for chat completion."""

    model: str = Field(
        default="gpt-5.2",
        description="Model to use (e.g., 'gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro')",
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation messages",
    )

    @field_validator("messages")
    @classmethod
    def _max_four_cache_breakpoints(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        # Anthropic accepts at most 4 cache_control breakpoints per request;
        # reject here with a clear 400 instead of a provider error mid-call.
        if sum(1 for m in v if m.cache_control is not None) > 4:
            raise ValueError("at most 4 messages may carry cache_control")
        return v
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    stop: list[str] | None = Field(
        default=None,
        max_length=4,
        description="Stop sequences (max 4)",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    tools: list[ToolDefinitionSchema] | None = Field(
        default=None,
        description="Tool definitions for function calling",
    )
    fallback_models: list[str] | None = Field(
        default=None,
        max_length=3,
        description=(
            "Up to 3 model tags tried in order when the primary model's "
            "serving fails retriably (5xx/timeout/provider rate-limit). "
            "Billed as-served; the response names the serving model and "
            "carries metadata.fallback_from when a fallback answered. "
            "Streaming switches only before the first chunk."
        ),
    )
    model_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provider-native parameters forwarded to the serving backend "
            "(e.g. OpenAI reasoning/service_tier, Anthropic top_k, Google "
            "safety_settings). Keys the serving adapter cannot take are "
            "REJECTED with a 400 listing them — never silently dropped. "
            "Custom (byom) endpoints accept any non-reserved key."
        ),
    )
    tool_choice: str | dict | None = Field(
        default=None,
        description="Tool choice strategy ('auto', 'none', 'required', or specific tool)",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow parallel tool calls",
    )
    response_format: ResponseFormat | None = Field(
        default=None,
        description="Response format for structured output",
    )
    mcp_servers: list[MCPServerConfig] | None = Field(
        default=None,
        description="MCP servers to connect to (pass-through to provider)",
    )


class TokenUsage(BaseSchema):
    """Token usage information."""

    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    total_tokens: int = Field(..., description="Total tokens used")
    cached_tokens: int = Field(
        default=0,
        description="Input tokens served from the provider's prompt cache (subset of input_tokens)",
    )
    cache_write_tokens: int = Field(
        default=0,
        description=(
            "Input tokens written to the provider's prompt cache this request "
            "(subset of input_tokens; Anthropic models with cache_control). "
            "Billed at the provider's cache-write tier."
        ),
    )


class CostInfo(BaseSchema):
    """Cost information for the request. `compute_units` is the model's
    pass-through cost, no markup — the flat tier fee is the revenue."""

    model_name: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens: TokenUsage = Field(..., description="Token usage")
    latency_ms: int = Field(..., description="Request latency in milliseconds")
    compute_units: float = Field(..., description="Compute units billed (provider pass-through, no markup)")


class ChatCompletionResponse(BaseSchema):
    """Response schema for chat completion."""

    content: str | None = Field(
        default=None,
        description="Generated response content (None if only tool_calls)",
    )
    role: Literal["assistant"] = Field(default="assistant", description="Response role")
    finish_reason: str = Field(
        default="stop",
        description="Finish reason: 'stop', 'tool_calls', 'length', 'content_filter'",
    )
    tool_calls: list[ToolCallSchema] | None = Field(
        default=None,
        description="Tool calls requested by the model",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Parsed structured output data (when response_format is used)",
    )
    cost: CostInfo = Field(..., description="Cost and usage information")
    assets: list[dict[str, Any]] | None = Field(
        default=None,
        description="Generated assets (images, etc.)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (trace_id, etc.)",
    )
    routing: RoutingMetadata | None = Field(
        default=None,
        description="Routing metadata when using MLPal meta-models.",
    )


class StreamChunk(BaseSchema):
    """A single chunk in a streamed response."""

    # BaseSchema sets str_strip_whitespace=True, which is fine for a whole message but corrupts a
    # token stream: each delta is its own model, so stripping eats the spaces at every chunk boundary
    # ("Let" + " me" -> "Let" + "me"). Preserve content deltas verbatim. (Configs merge in pydantic v2.)
    model_config = ConfigDict(str_strip_whitespace=False)

    content: str = Field(..., description="Content chunk")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    tool_calls: list[ToolCallSchema] | None = Field(
        default=None,
        description=(
            "Tool calls emitted mid-stream. A tool-use turn yields these on the chunk(s) "
            "before the final chunk; same shape as the non-streaming response."
        ),
    )
    finish_reason: str | None = Field(
        default=None,
        description="Finish reason on the final chunk: 'stop', 'tool_calls', 'length', 'content_filter'.",
    )
    cost: CostInfo | None = Field(default=None, description="Cost info (only in final chunk)")
    routing: RoutingMetadata | None = Field(
        default=None,
        description="Routing metadata when using MLPal meta-models (only in final chunk).",
    )
