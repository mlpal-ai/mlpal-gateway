"""Provider call retry policy — one place, every SDK.

Why this exists (incident 2026-09-29): a non-streaming Opus request whose
generation ran past the 120 s read timeout was retried by THREE stacked
layers — the SDK (2 retries), tenacity on the adapter (3 attempts) and the
gateway's backend hop (2 backends) — up to 18 full-price invocations for one
request that returned a 504 to the client. Bedrock's invocation log showed
9 and 18 attempt chains with identical input token counts; our usage log
had one 0-token error row per chain.

Policy:
  * A timeout is NOT retried. The provider has already accepted the prompt
    and is generating; a second attempt bills the whole prompt again.
  * Pre-response faults are retried ONCE: connection refused/reset, connect
    timeout, 408/409/429 and 5xx (incl. Anthropic 529 overloaded). None of
    these have billed anything yet.
  * Everything else (4xx, parsing, validation) is the caller's — never retried.

SDK-level retries must be OFF (``max_retries=0`` / boto ``max_attempts=1``)
so this is the only retry layer. Every retry is a structured log line and a
``ProviderRetry`` metric so the cost of transient faults stays visible.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_sleep = asyncio.sleep  # seam: tests replace the backoff without touching asyncio

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429})
_TRANSIENT_BOTO_CODES = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
})


def read_timeout_seconds() -> float:
    """Whole-response wait for a non-streaming provider call."""
    from mlpal_assistants_service.core.config import get_settings

    return float(get_settings().provider_read_timeout_seconds)


def is_timeout(exc: BaseException) -> bool:
    """A read/overall timeout: the request reached the provider and the
    response never completed. Connect timeouts are NOT timeouts in this
    sense (nothing was sent) — see ``is_pre_response_fault``."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    name = type(exc).__name__
    if "ConnectTimeout" in name:
        return False
    # httpx.ReadTimeout / httpx.TimeoutException, anthropic/openai
    # APITimeoutError, botocore ReadTimeoutError.
    return "Timeout" in name


def _http_status(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)  # anthropic / openai APIStatusError
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)  # google-genai APIError
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if isinstance(response, dict):  # botocore ClientError
        meta = response.get("ResponseMetadata") or {}
        status = meta.get("HTTPStatusCode")
        if isinstance(status, int):
            return status
    status = getattr(response, "status_code", None)  # httpx.HTTPStatusError
    return status if isinstance(status, int) else None


def is_pre_response_fault(exc: BaseException) -> bool:
    """True when the provider has not started serving the request: a safe,
    unbilled retry. False for timeouts, client errors and anything unknown."""
    if is_timeout(exc):
        return False
    name = type(exc).__name__
    if "ConnectTimeout" in name or "Connection" in name or name == "ConnectError":
        return True
    if name == "RemoteProtocolError":  # httpx: peer closed before a response
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = ((response.get("Error") or {}).get("Code")) or ""
        if code in _TRANSIENT_BOTO_CODES:
            return True
    status = _http_status(exc)
    if status is None:
        return False
    return status in _TRANSIENT_HTTP_STATUSES or status >= 500


async def call_provider(
    fn: Callable[[], Awaitable[T]],
    *,
    provider: str,
    operation: str = "chat",
    max_attempts: int = 2,
    backoff_seconds: float = 1.0,
) -> T:
    """Run one provider call under the policy above. ``fn`` must build a
    fresh request each time it is called (no shared iterators)."""
    attempt = 1
    while True:
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — classified, then re-raised
            if attempt >= max_attempts or not is_pre_response_fault(e):
                raise
            _note_retry(provider, operation, attempt, e)
            await _sleep(backoff_seconds * attempt)
            attempt += 1


def _note_retry(provider: str, operation: str, attempt: int, exc: BaseException) -> None:
    logger.warning(
        "provider_retry provider=%s operation=%s attempt=%d error=%s status=%s",
        provider, operation, attempt, type(exc).__name__, _http_status(exc),
    )
    try:
        from mlpal_assistants_service.core.metrics import get_metrics

        get_metrics().put_metric_sync(
            "ProviderRetry", 1, dimensions={"provider": provider, "operation": operation}
        )
    except Exception:  # noqa: BLE001 — metrics never break a request
        logger.debug("ProviderRetry metric emit failed", exc_info=True)
