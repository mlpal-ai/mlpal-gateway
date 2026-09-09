"""AWS Bedrock provider adapter using boto3 Converse API."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import aioboto3
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


class BedrockAdapter(BaseAdapter):
    # Converse has additionalModelRequestFields — a first-class arbitrary-
    # params channel — so this adapter accepts any non-reserved key.
    accept_all_kwargs = True

    """
    Adapter for AWS Bedrock API using boto3 Converse API.

    Supports:
    - Chat completion (text)
    - Structured output (JSON mode via tool use)
    - Vision (image input for supported models)
    - Documents (PDFs for supported models)
    - Tool/function calling
    - Streaming

    Models:
    - Llama 3: meta.llama3-1-70b-instruct-v1:0, meta.llama3-1-405b-instruct-v1:0
    - Mistral: mistral.mistral-large-2407-v1:0, mistral.mistral-large-3-675b-instruct
    - Claude (via Bedrock): anthropic.claude-3-5-sonnet-20241022-v2:0
    """

    # Anthropic-style usage: input_tokens EXCLUDES cache reads (see BaseAdapter).
    cached_tokens_included_in_input = False

    provider_name = "bedrock"

    # Model capabilities registry
    # Note: Bedrock model IDs vary; using common patterns for matching
    MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
        # Current open-weight set (2026-08). Keys are version-suffix-free
        # prefixes of the Bedrock model ids (get_model_capabilities prefix-matches).
        "zai.glm-5": ModelCapabilities(
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=200000, max_output_tokens=8192,
        ),
        "moonshotai.kimi-k2.5": ModelCapabilities(
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=256000, max_output_tokens=8192,
        ),
        "deepseek.v3.2": ModelCapabilities(
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=128000, max_output_tokens=8192,
        ),
        "qwen.qwen3-coder-480b": ModelCapabilities(
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=262144, max_output_tokens=8192,
        ),
        "us.meta.llama4-maverick": ModelCapabilities(
            supports_images=True,  # Maverick is natively multimodal
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=1000000, max_output_tokens=8192,
        ),
        "openai.gpt-oss-120b": ModelCapabilities(
            supports_tools=True, supports_structured_output=True,
            max_context_tokens=128000, max_output_tokens=8192,
        ),
        # Llama 3.1 models - text only, no vision
        "us.meta.llama3-1-70b-instruct": ModelCapabilities(
            supports_images=False,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        ),
        "us.meta.llama3-1-405b-instruct": ModelCapabilities(
            supports_images=False,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        ),
        # Llama 3.2 models - vision support
        "us.meta.llama3-2-11b-instruct": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        ),
        "us.meta.llama3-2-90b-instruct": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=4096,
        ),
        # Mistral models
        "mistral.mistral-large-2407": ModelCapabilities(
            supports_images=False,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=8192,
        ),
        # Mistral Large 3 - supports vision
        "mistral.mistral-large-3": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,  # Check if PDF supported
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=128000,
            max_output_tokens=8192,
        ),
        # Claude models via Bedrock
        "anthropic.claude-3-5-sonnet": ModelCapabilities(
            supports_images=True,
            supports_pdf=True,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=200000,
            max_output_tokens=8192,
        ),
        "anthropic.claude-3-opus": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
        "anthropic.claude-3-sonnet": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
        "anthropic.claude-3-haiku": ModelCapabilities(
            supports_images=True,
            supports_pdf=False,
            supports_audio=False,
            supports_video=False,
            supports_tools=True,
            supports_structured_output=True,
            max_context_tokens=200000,
            max_output_tokens=4096,
        ),
    }

    # Default capabilities for unknown models (conservative - text only)
    DEFAULT_CAPABILITIES = ModelCapabilities(
        supports_images=False,
        supports_pdf=False,
        supports_audio=False,
        supports_video=False,
        supports_tools=True,
        supports_structured_output=True,
        max_context_tokens=128000,
        max_output_tokens=4096,
    )

    def __init__(
        self,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        if region_name:
            self._region_name = region_name
        else:
            try:
                settings = get_settings()
                self._region_name = settings.aws_region or "us-east-2"
            except Exception:
                self._region_name = "us-east-2"
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._session = aioboto3.Session(
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
        )

    # =========================================================================
    # Model Capabilities
    # =========================================================================

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Bedrock model."""
        # Check exact match first
        if model in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[model]

        # Check prefix matches (model IDs often have version suffixes)
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
        """Execute chat completion via Bedrock Converse API."""
        start_time = time.perf_counter()

        try:
            # Validate file attachments against model capabilities
            files = self.extract_files_from_messages(messages)
            if files:
                self.validate_file_attachments(model, files)

            async with self._session.client("bedrock-runtime") as client:
                # Normalize messages for Bedrock format
                system_messages, bedrock_messages = self._normalize_messages(messages)

                # Build request parameters
                params: dict[str, Any] = {
                    "modelId": model,
                    "messages": bedrock_messages,
                }

                # Build inference config
                inference_config: dict[str, Any] = {}
                if temperature is not None:
                    inference_config["temperature"] = temperature
                if max_tokens is not None:
                    inference_config["maxTokens"] = max_tokens
                if top_p is not None:
                    inference_config["topP"] = top_p
                if stop is not None:
                    inference_config["stopSequences"] = stop

                if inference_config:
                    params["inferenceConfig"] = inference_config

                # Add system messages if present
                if system_messages:
                    params["system"] = system_messages

                # Add tools if provided
                if tools:
                    params["toolConfig"] = {
                        "tools": self._convert_tools(tools),
                    }
                    # Only add toolChoice if explicitly provided and not "required"
                    # Some models (like Llama) don't support toolChoice.any
                    if tool_choice and tool_choice != "required":
                        params["toolConfig"]["toolChoice"] = self._convert_tool_choice(tool_choice)

                # Handle structured output via toolConfig
                _structured_tool_name = None
                if response_format and response_format.get("type") == "json_schema":
                    schema = response_format.get("json_schema", {}).get("schema", {})
                    if schema and not tools:
                        _structured_tool_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        params["toolConfig"] = {
                            "tools": [{
                                "toolSpec": {
                                    "name": _structured_tool_name,
                                    "description": f"Return response in {_structured_tool_name} format. Always use this tool.",
                                    "inputSchema": {"json": self._prepare_schema_for_bedrock(schema)},
                                }
                            }],
                        }

                # Reject MCP servers
                if mcp_servers:
                    raise UnsupportedCapabilityError(
                        capability="mcp",
                        model=model,
                        suggestions=["gpt-5.2", "claude-opus-4.5"],
                    )

                # Model-specific kwargs → Converse's designed escape hatch.
                if model_kwargs:
                    params["additionalModelRequestFields"] = {
                        **params.get("additionalModelRequestFields", {}),
                        **model_kwargs,
                    }

                # Make API call
                response = await client.converse(**params)

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                # Extract response content and tool calls
                content = ""
                tool_calls = None

                output_message = response.get("output", {}).get("message", {})
                for block in output_message.get("content", []):
                    if "text" in block:
                        content += block["text"]
                    elif "toolUse" in block:
                        tool_use = block["toolUse"]
                        # If this is our structured output tool, return its input as content
                        if _structured_tool_name and tool_use.get("name") == _structured_tool_name:
                            content = json.dumps(tool_use.get("input", {}))
                            continue
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append({
                            "id": tool_use.get("toolUseId", f"call_{len(tool_calls or [])}"),
                            "type": "function",
                            "function": {
                                "name": tool_use.get("name"),
                                "arguments": json.dumps(tool_use.get("input", {})),
                            },
                        })

                # Get token counts
                usage_data = response.get("usage", {})
                usage = TokenUsage(
                    input_tokens=usage_data.get("inputTokens", 0),
                    output_tokens=usage_data.get("outputTokens", 0),
                )

                return AdapterResponse(
                    content=content,
                    model=model,
                    provider=self.provider_name,
                    usage=usage,
                    finish_reason=self._map_stop_reason(response.get("stopReason")),
                    latency_ms=latency_ms,
                    tool_calls=tool_calls,
                    raw_response=response,
                )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"Bedrock chat error after {latency_ms}ms: {error_message}")

            if "throttl" in error_message.lower() or "rate" in error_message.lower():
                raise ProviderUnavailableError(
                    message="Bedrock rate limit exceeded",
                    provider=self.provider_name,
                    original_error=error_message,
                )
            raise ProviderError(
                message=f"Bedrock API error: {error_message}",
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

            async with self._session.client("bedrock-runtime") as client:
                system_messages, bedrock_messages = self._normalize_messages(messages)

                params: dict[str, Any] = {
                    "modelId": model,
                    "messages": bedrock_messages,
                }

                inference_config: dict[str, Any] = {}
                if temperature is not None:
                    inference_config["temperature"] = temperature
                if max_tokens is not None:
                    inference_config["maxTokens"] = max_tokens
                if top_p is not None:
                    inference_config["topP"] = top_p
                if stop is not None:
                    inference_config["stopSequences"] = stop

                if inference_config:
                    params["inferenceConfig"] = inference_config

                if system_messages:
                    params["system"] = system_messages

                if tools:
                    params["toolConfig"] = {
                        "tools": self._convert_tools(tools),
                    }
                    # Only add toolChoice if explicitly provided and not "required"
                    if tool_choice and tool_choice != "required":
                        params["toolConfig"]["toolChoice"] = self._convert_tool_choice(tool_choice)

                # Handle structured output via toolConfig
                if response_format and response_format.get("type") == "json_schema":
                    schema = response_format.get("json_schema", {}).get("schema", {})
                    if schema and not tools:
                        # Use tool-based approach for structured output
                        schema_name = response_format.get("json_schema", {}).get("name", "structured_output")
                        params["toolConfig"] = {
                            "tools": [{
                                "toolSpec": {
                                    "name": schema_name,
                                    "description": f"Return response in {schema_name} format",
                                    "inputSchema": {"json": self._prepare_schema_for_bedrock(schema)},
                                }
                            }],
                        }

                # Reject MCP servers
                if mcp_servers:
                    raise UnsupportedCapabilityError(
                        capability="mcp",
                        model=model,
                        suggestions=["gpt-5.2", "claude-opus-4.5"],
                    )

                # Use converseStream for streaming
                if model_kwargs:
                    params["additionalModelRequestFields"] = {
                        **params.get("additionalModelRequestFields", {}),
                        **model_kwargs,
                    }
                response = await client.converse_stream(**params)

                total_input_tokens = 0
                total_output_tokens = 0
                stop_reason: str | None = None
                pending_tool_calls: list[dict[str, Any]] = []
                current_tool_use: dict[str, Any] | None = None
                current_tool_input = ""

                async for event in response["stream"]:
                    # Handle content block delta
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            yield StreamChunk(content=delta["text"], done=False)
                        elif "toolUse" in delta:
                            # Accumulate tool input
                            current_tool_input += delta["toolUse"].get("input", "")

                    # Handle content block start
                    elif "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        if "toolUse" in start:
                            current_tool_use = {
                                "id": start["toolUse"].get("toolUseId", f"call_{len(pending_tool_calls)}"),
                                "type": "function",
                                "function": {
                                    "name": start["toolUse"].get("name"),
                                    "arguments": "",
                                },
                            }
                            current_tool_input = ""

                    # Handle content block stop
                    elif "contentBlockStop" in event:
                        if current_tool_use:
                            current_tool_use["function"]["arguments"] = current_tool_input
                            pending_tool_calls.append(current_tool_use)
                            current_tool_use = None
                            current_tool_input = ""

                    # Handle message stop with usage
                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason")

                    # Handle metadata with usage
                    elif "metadata" in event:
                        usage_data = event["metadata"].get("usage", {})
                        total_input_tokens = usage_data.get("inputTokens", 0)
                        total_output_tokens = usage_data.get("outputTokens", 0)

                # Emit tool calls if any
                if pending_tool_calls:
                    yield StreamChunk(
                        content="",
                        done=False,
                        tool_calls=pending_tool_calls,
                    )

                # Final chunk. Mapping the real stopReason matters on the
                # Anthropic wire: "length" becomes stop_reason=max_tokens and
                # feeds empty-completion detection.
                yield StreamChunk(
                    content="",
                    done=True,
                    usage=TokenUsage(
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    ),
                    finish_reason=self._map_stop_reason(stop_reason),
                )

        except (UnsupportedModalityError, UnsupportedCapabilityError):
            raise
        except Exception as e:
            logger.error(f"Bedrock streaming error: {e}")
            raise ProviderError(
                message=f"Bedrock streaming error: {e}",
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Convert messages to Bedrock Converse format.

        Returns:
            Tuple of (system_messages, normalized_messages)

        Bedrock Converse format:
        - System messages are separate
        - Content is a list of content blocks
        - Images use {"image": {"format": "...", "source": {"bytes": "..."}}}
        - Documents use {"document": {"format": "...", "source": {"bytes": "..."}}}
        - Tool calls use {"toolUse": {...}}
        - Tool results use {"toolResult": {...}}

        Supports both:
        - FileAttachment objects in the `files` key
        - Legacy dict format with `images`/`documents` keys
        """
        system_messages = []
        normalized = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract system messages
            if role == "system":
                system_messages.append({"text": str(content)})
                continue

            # Handle tool calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                if content:
                    content_blocks.append({"text": str(content)})

                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}

                    content_blocks.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", f"call_{len(content_blocks)}"),
                            "name": tc["function"]["name"],
                            "input": args,
                        }
                    })

                normalized.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
                continue

            # Handle tool results
            if role == "tool":
                normalized.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg.get("tool_call_id"),
                            "content": [{"text": str(content)}],
                        }
                    }],
                })
                continue

            # Build content blocks for regular messages
            content_blocks = []

            # Add text content
            if content:
                content_blocks.append({"text": str(content)})

            # Process files (supports both FileAttachment and dict)
            files = msg.get("files", [])
            for f in files:
                block = self._process_file_attachment(f)
                if block:
                    content_blocks.append(block)

            # Add images directly (legacy format)
            for img in msg.get("images", []):
                image_block = self._build_image_block(img)
                if image_block:
                    content_blocks.append(image_block)

            # Add documents directly (legacy format)
            for doc in msg.get("documents", []):
                doc_block = self._build_document_block(doc)
                if doc_block:
                    content_blocks.append(doc_block)

            if content_blocks:
                normalized.append({
                    "role": role,
                    "content": content_blocks,
                })

        return system_messages, normalized

    def _process_file_attachment(
        self,
        attachment: FileAttachment | dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Process a file attachment into a Bedrock content block.

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
                logger.warning(f"Unsupported file type for Bedrock: {attachment.type}")
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
            return self._build_image_block(attachment)
        elif file_type in ("pdf", "document"):
            return self._build_document_block(attachment)
        elif file_type in ("code", "text"):
            # Include as text
            text_content = attachment.get("text") or attachment.get("data", "")
            filename = attachment.get("filename", "file")
            return {"text": f"[{filename}]\n{text_content}"}
        else:
            logger.warning(f"Unknown file type in dict: {file_type}")
            return None

    def _build_image_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any] | None:
        """Build image content block from FileAttachment."""
        import base64 as b64_module

        import httpx

        mime = attachment.mime_type or "image/png"
        format_type = mime.split("/")[-1]  # e.g., "png", "jpeg"

        if attachment.source == FileSource.BASE64:
            image_bytes = b64_module.b64decode(attachment.data)
            return {
                "image": {
                    "format": format_type,
                    "source": {"bytes": image_bytes},
                }
            }
        elif attachment.source == FileSource.URL:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(attachment.data)
                    response.raise_for_status()
                    image_bytes = response.content
                    if attachment.mime_type is None:
                        content_type = response.headers.get("content-type", mime)
                        mime = content_type.split(";")[0].strip()
                        format_type = mime.split("/")[-1]
                    return {
                        "image": {
                            "format": format_type,
                            "source": {"bytes": image_bytes},
                        }
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch image from URL: {e}")
                return None
        elif attachment.source == FileSource.PATH:
            with open(attachment.data, "rb") as f:
                image_bytes = f.read()
            return {
                "image": {
                    "format": format_type,
                    "source": {"bytes": image_bytes},
                }
            }
        else:
            logger.warning(f"Unsupported image source: {attachment.source}")
            return None

    def _build_document_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any] | None:
        """Build document content block from FileAttachment (for PDFs)."""
        import base64 as b64_module

        import httpx

        mime = attachment.mime_type or "application/pdf"
        format_type = "pdf" if "pdf" in mime else mime.split("/")[-1]
        name = attachment.filename or "document"

        if attachment.source == FileSource.BASE64:
            doc_bytes = b64_module.b64decode(attachment.data)
            return {
                "document": {
                    "format": format_type,
                    "name": name,
                    "source": {"bytes": doc_bytes},
                }
            }
        elif attachment.source == FileSource.URL:
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.get(attachment.data)
                    response.raise_for_status()
                    doc_bytes = response.content
                    return {
                        "document": {
                            "format": format_type,
                            "name": name,
                            "source": {"bytes": doc_bytes},
                        }
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch document from URL: {e}")
                return None
        elif attachment.source == FileSource.PATH:
            with open(attachment.data, "rb") as f:
                doc_bytes = f.read()
            return {
                "document": {
                    "format": format_type,
                    "name": name,
                    "source": {"bytes": doc_bytes},
                }
            }
        else:
            logger.warning(f"Unsupported document source: {attachment.source}")
            return None

    def _build_text_from_attachment(
        self,
        attachment: FileAttachment,
    ) -> dict[str, Any]:
        """Build text content block from code/text FileAttachment."""
        import base64 as b64_module

        if attachment.source == FileSource.PATH:
            with open(attachment.data, encoding="utf-8") as f:
                text_content = f.read()
        elif attachment.source == FileSource.BASE64:
            text_content = b64_module.b64decode(attachment.data).decode("utf-8")
        elif attachment.source == FileSource.URL:
            text_content = f"[File from URL: {attachment.data}]"
        else:
            text_content = attachment.data

        filename = attachment.filename or "file"
        return {"text": f"[{filename}]\n{text_content}"}

    def _build_image_block(self, image: dict[str, Any]) -> dict[str, Any] | None:
        """Build image content block for vision from legacy dict format."""
        import base64 as b64_module

        if image.get("base64"):
            image_bytes = b64_module.b64decode(image["base64"])
            mime_type = image.get("mime_type", "image/png")
            format_type = mime_type.split("/")[-1]  # e.g., "png", "jpeg"
            return {
                "image": {
                    "format": format_type,
                    "source": {"bytes": image_bytes},
                }
            }
        elif image.get("url"):
            # Fetch the image for Bedrock
            import httpx
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(image["url"])
                    response.raise_for_status()
                    image_bytes = response.content
                    content_type = response.headers.get("content-type", "image/png")
                    mime_type = content_type.split(";")[0].strip()
                    format_type = mime_type.split("/")[-1]
                    return {
                        "image": {
                            "format": format_type,
                            "source": {"bytes": image_bytes},
                        }
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch image from URL: {e}")
                return None
        return None

    def _build_document_block(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        """Build document content block for PDFs from legacy dict format."""
        import base64 as b64_module

        import httpx

        mime_type = doc.get("mime_type", "application/pdf")
        format_type = "pdf" if "pdf" in mime_type else mime_type.split("/")[-1]
        name = doc.get("filename", "document")

        if doc.get("base64"):
            doc_bytes = b64_module.b64decode(doc["base64"])
            return {
                "document": {
                    "format": format_type,
                    "name": name,
                    "source": {"bytes": doc_bytes},
                }
            }
        elif doc.get("url"):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.get(doc["url"])
                    response.raise_for_status()
                    doc_bytes = response.content
                    return {
                        "document": {
                            "format": format_type,
                            "name": name,
                            "source": {"bytes": doc_bytes},
                        }
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch document from URL: {e}")
                return None
        return None

    def _convert_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert tool definitions to Bedrock format."""
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.parameters,
                    },
                }
            }
            for tool in tools
        ]

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict[str, Any]:
        """Convert tool_choice to Bedrock format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"auto": {}}
            elif tool_choice == "none":
                return {"auto": {}}  # Bedrock doesn't have explicit "none"
            elif tool_choice == "required":
                return {"any": {}}
            else:
                # Specific tool name
                return {"tool": {"name": tool_choice}}
        return tool_choice

    def _prepare_schema_for_bedrock(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a JSON schema for Bedrock's tool inputSchema.
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

        # Remove title if present
        if "title" in schema:
            del schema["title"]

        return schema

    def _map_stop_reason(self, stop_reason: str | None) -> str:
        """Map Bedrock stop reasons to standard format."""
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "content_filtered": "content_filter",
        }
        return mapping.get(stop_reason or "", stop_reason or "stop")

    def _fix_nested_json_strings(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Fix nested JSON strings in model output.

        Some models return arrays or objects as JSON strings instead of
        proper Python objects. This recursively tries to parse them.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Try to parse as JSON if it looks like JSON
                if value.startswith("[") or value.startswith("{"):
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self._fix_nested_json_strings(value)
            else:
                result[key] = value
        return result

    # =========================================================================
    # Generation Methods
    # =========================================================================

    # Default models for generation
    DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
    DEFAULT_IMAGE_MODEL = "amazon.titan-image-generator-v2:0"  # More widely available than Stability

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings using Bedrock embedding models.

        Args:
            texts: List of text strings to embed
            model: Model ID (default: amazon.titan-embed-text-v2:0)
            dimensions: Output dimensions (for Titan: 256, 512, or 1024)

        Returns:
            EmbeddingResponse with embedding vectors

        Models:
            - amazon.titan-embed-text-v2:0: AWS Titan (256/512/1024 dims)
            - cohere.embed-english-v3: Cohere English (1024 dims)
            - cohere.embed-multilingual-v3: Cohere 100+ languages (1024 dims)
        """

        model = model or self.DEFAULT_EMBEDDING_MODEL
        start_time = time.perf_counter()

        try:
            embeddings = []

            async with self._session.client("bedrock-runtime") as client:
                for text in texts:
                    # Build request body based on model
                    if model.startswith("amazon.titan"):
                        body = {
                            "inputText": text,
                        }
                        if dimensions:
                            body["dimensions"] = dimensions

                        response = await client.invoke_model(
                            modelId=model,
                            body=json.dumps(body),
                        )

                        # Read the streaming body
                        response_body = await response["body"].read()
                        result = json.loads(response_body)
                        embeddings.append(result["embedding"])

                    elif model.startswith("cohere"):
                        body = {
                            "texts": [text],
                            "input_type": "search_document",
                        }

                        response = await client.invoke_model(
                            modelId=model,
                            body=json.dumps(body),
                        )

                        response_body = await response["body"].read()
                        result = json.loads(response_body)
                        embeddings.append(result["embeddings"][0])

                    else:
                        raise ProviderError(
                            message=f"Unsupported embedding model: {model}",
                            provider=self.provider_name,
                            original_error=f"Model {model} not recognized as an embedding model",
                        )

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

        except ProviderError:
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Bedrock embedding error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Bedrock embedding error: {e}",
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
        Generate images using Bedrock image generation models.

        Args:
            prompt: Text description of the image to generate
            model: Model ID (default: stability.stable-image-ultra-v1:0)
            size: Output size (ImageSize enum or string like "1024x1024")
            quality: Quality level
            n: Number of images to generate
            reference_images: Not supported for Bedrock Stability models

        Returns:
            ImageGenerationResponse with generated images

        Models:
            - stability.stable-image-ultra-v1:0: Stability AI Ultra
            - stability.stable-image-core-v1:0: Stability AI Core
            - amazon.titan-image-generator-v2:0: AWS Titan Image
        """

        model = model or self.DEFAULT_IMAGE_MODEL
        start_time = time.perf_counter()

        try:
            # Resolve any size format (preset, aspect ratio, pixels) to
            # pixel dimensions via ImageSizeResolver, matching other adapters.
            pixels = ImageSizeResolver.to_pixels_openai(size, model=model or "gpt-image-1")
            try:
                width, height = map(int, pixels.split("x"))
            except ValueError:
                width, height = 1024, 1024

            images = []

            async with self._session.client("bedrock-runtime") as client:
                if model.startswith("stability"):
                    # Stability AI models
                    body = {
                        "text_prompts": [{"text": prompt, "weight": 1.0}],
                        "cfg_scale": 7,
                        "steps": 50 if quality == ImageQuality.HD else 30,
                        "width": width,
                        "height": height,
                        "samples": n,
                    }

                    response = await client.invoke_model(
                        modelId=model,
                        body=json.dumps(body),
                    )

                    response_body = await response["body"].read()
                    result = json.loads(response_body)

                    for artifact in result.get("artifacts", []):
                        images.append(GeneratedImage(
                            base64=artifact["base64"],
                            format="png",
                        ))

                elif model.startswith("amazon.titan-image"):
                    # AWS Titan Image Generator
                    body = {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text": prompt,
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": n,
                            "height": height,
                            "width": width,
                            "cfgScale": 8.0,
                        },
                    }

                    response = await client.invoke_model(
                        modelId=model,
                        body=json.dumps(body),
                    )

                    response_body = await response["body"].read()
                    result = json.loads(response_body)

                    for img_base64 in result.get("images", []):
                        images.append(GeneratedImage(
                            base64=img_base64,
                            format="png",
                        ))

                else:
                    raise ProviderError(
                        message=f"Unsupported image generation model: {model}",
                        provider=self.provider_name,
                        original_error=f"Model {model} not recognized as an image generation model",
                    )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return ImageGenerationResponse(
                images=images,
                model=model,
                provider=self.provider_name,
                latency_ms=latency_ms,
                prompt=prompt,
            )

        except ProviderError:
            raise
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Bedrock image generation error after {latency_ms}ms: {e}")
            raise ProviderError(
                message=f"Bedrock image generation error: {e}",
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
        NOT SUPPORTED: Bedrock does not have native transcription.

        For transcription on AWS, use:
        - Amazon Transcribe (dedicated AWS service)
        - OpenAI whisper-1 or gpt-4o-transcribe

        To use Amazon Transcribe:
        ```python
        import boto3
        transcribe = boto3.client('transcribe')
        # Use start_transcription_job for async or
        # start_stream_transcription for real-time
        ```
        """
        raise NotImplementedError(
            "Bedrock does not have native audio transcription. "
            "Recommended alternatives:\n"
            "  - AWS: Amazon Transcribe (dedicated service, boto3.client('transcribe'))\n"
            "  - OpenAI: gpt-4o-transcribe (highest accuracy)\n"
            "  - OpenAI: whisper-1 (multilingual, stable)\n"
            "  - Google: gemini-3-flash-preview (native audio understanding)"
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
        NOT SUPPORTED: Bedrock does not have native text-to-speech.

        For TTS on AWS, use:
        - Amazon Polly (dedicated AWS service)
        - OpenAI tts-1 or tts-1-hd

        To use Amazon Polly:
        ```python
        import boto3
        polly = boto3.client('polly')
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'  # or other voice IDs
        )
        audio_stream = response['AudioStream'].read()
        ```
        """
        raise NotImplementedError(
            "Bedrock does not have native text-to-speech. "
            "Recommended alternatives:\n"
            "  - AWS: Amazon Polly (dedicated service, boto3.client('polly'))\n"
            "  - OpenAI: tts-1-hd (high quality)\n"
            "  - OpenAI: tts-1 (faster, standard quality)\n"
            "  - Google: Cloud Text-to-Speech API"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if Bedrock API is accessible."""
        try:
            async with self._session.client("bedrock") as client:
                # List foundation models to verify access
                await client.list_foundation_models(byOutputModality="TEXT")
                return True
        except Exception as e:
            logger.warning(f"Bedrock health check failed: {e}")
            return False

    @property
    def supports_vision(self) -> bool:
        # Only some models support vision (e.g., Mistral Large 3)
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
