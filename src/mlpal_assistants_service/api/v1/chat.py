"""Chat completion endpoint.

Thin API layer that delegates to ChatService for orchestration.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from mlpal_assistants_service.adapters.base import UnsupportedModalityError
from mlpal_assistants_service.api.deps import (
    ChatServiceDep,
    CurrentAPIKey,
)
from mlpal_assistants_service.core.exceptions import (
    ModelNotAvailableError,
    ModelNotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitExceededError,
    UnsupportedCapabilityError,
    UnsupportedEffortError,
    UnsupportedModelKwargsError,
    ValidationError,
    http_status_for_provider_error,
)
from mlpal_assistants_service.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter()

# Seconds of stream silence before we emit an SSE keepalive. Must be well under
# the client read-timeout and the ALB idle_timeout (300s) so a long byte-silent
# phase — e.g. an OpenAI reasoning model "thinking" before any token — can't
# trip those timeouts and stall the stream.
_STREAM_HEARTBEAT_INTERVAL = 15.0

# Bound the producer->consumer buffer so a slow SSE reader can't make the
# producer (draining the provider stream ahead of the reader) pile chunks into
# gateway memory unboundedly. When full, `queue.put` blocks — backpressure that
# naturally throttles the provider read to the reader's pace.
_STREAM_QUEUE_MAXSIZE = 256


def _stream_chunk_to_data(chunk) -> dict:
    """Serialize a StreamChunk to the SSE `data:` payload dict."""
    data: dict = {"content": chunk.content, "done": chunk.done}
    if chunk.tool_calls:
        data["tool_calls"] = [
            tc.model_dump(by_alias=True, exclude_none=True) for tc in chunk.tool_calls
        ]
    if chunk.finish_reason:
        data["finish_reason"] = chunk.finish_reason
    if chunk.cost:
        data["cost"] = {
            "model_name": chunk.cost.model_name,
            "provider": chunk.cost.provider,
            "tokens": {
                "input_tokens": chunk.cost.tokens.input_tokens,
                "output_tokens": chunk.cost.tokens.output_tokens,
                "total_tokens": chunk.cost.tokens.total_tokens,
                "cached_tokens": chunk.cost.tokens.cached_tokens,
                "cache_write_tokens": chunk.cost.tokens.cache_write_tokens,
                "reasoning_tokens": chunk.cost.tokens.reasoning_tokens,
            },
            "latency_ms": chunk.cost.latency_ms,
            "compute_units": chunk.cost.compute_units,
        }
    # The final chunk carries routing (meta-model resolution); it was being
    # dropped at this seam while non-stream responses always had it.
    routing = getattr(chunk, "routing", None)
    if routing:
        data["routing"] = routing.model_dump(exclude_none=True)
    return data


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    description="Generate a chat completion using any supported model.",
)
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    api_key: CurrentAPIKey,
    chat_service: ChatServiceDep,
) -> ChatCompletionResponse:
    """
    Create a chat completion.

    Supports models from OpenAI, Anthropic, Google, and more.
    """
    # Check permission
    if not api_key.has_permission("chat"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have permission for chat completions",
        )

    try:
        return await chat_service.chat(
            user_id=api_key.user_id,
            api_key_id=api_key.id,
            request=body,
            tier=api_key.rate_limit_tier,
            model_policy=api_key.model_policy,
            budgets=api_key.budgets,
            capture_policy=api_key.capture_policy,
        )
    except UnsupportedModalityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except UnsupportedCapabilityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except (UnsupportedModelKwargsError, UnsupportedEffortError, ValidationError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not found: {e.model}",
        )
    except ModelNotAvailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except QuotaExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.message,
        )
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=e.message,
            headers={"Retry-After": str(e.retry_after or 60)},
        )
    except ProviderError as e:
        raise HTTPException(
            status_code=http_status_for_provider_error(e),
            detail=f"Provider error: {e.message}",
        )


@router.post(
    "/completions/stream",
    summary="Create streaming chat completion",
    description="Generate a streaming chat completion.",
)
async def create_chat_completion_stream(
    request: Request,
    body: ChatCompletionRequest,
    api_key: CurrentAPIKey,
    chat_service: ChatServiceDep,
) -> StreamingResponse:
    """Create a streaming chat completion."""
    # Check permission
    if not api_key.has_permission("chat"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key does not have permission for chat completions",
        )

    async def generate():
        """Generate SSE stream."""
        # A producer task drains the adapter stream into a queue; this loop
        # reads it with a timeout and emits an SSE heartbeat comment whenever no
        # chunk has arrived for _STREAM_HEARTBEAT_INTERVAL seconds. Decoupling
        # via a queue (rather than wrapping `__anext__` in wait_for) keeps the
        # underlying async generator from being cancelled mid-iteration. The
        # queue is bounded (backpressure): if a full queue blocks the producer
        # on `put` and the client then disconnects, the `finally` below still
        # cancels + awaits the producer, which unblocks it — so it can't hang.
        queue: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

        async def _produce() -> None:
            try:
                async for chunk in chat_service.chat_stream(
                    user_id=api_key.user_id,
                    api_key_id=api_key.id,
                    request=body,
                    tier=api_key.rate_limit_tier,
                    model_policy=api_key.model_policy,
                    budgets=api_key.budgets,
                    capture_policy=api_key.capture_policy,
                ):
                    await queue.put(("chunk", chunk))
            except QuotaExceededError as e:
                await queue.put(("error", {"error": e.message, "done": True}))
            except ProviderError as e:
                await queue.put(("error", {"error": f"Provider error: {e.message}", "done": True}))
            except Exception as e:  # noqa: BLE001
                await queue.put(("error", {"error": str(e), "done": True}))
            finally:
                await queue.put(("end", None))

        producer = asyncio.create_task(_produce())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        queue.get(), timeout=_STREAM_HEARTBEAT_INTERVAL
                    )
                except TimeoutError:
                    # SSE comment line: ignored by SSE/JSON clients, but it's
                    # bytes on the wire that reset client/ALB idle timers.
                    yield ": ping\n\n"
                    continue

                if kind == "chunk":
                    yield f"data: {json.dumps(_stream_chunk_to_data(payload))}\n\n"
                elif kind == "error":
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
                else:  # "end"
                    break

            yield "data: [DONE]\n\n"
        finally:
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
