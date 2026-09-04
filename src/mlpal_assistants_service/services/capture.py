"""Opt-in request/response payload capture for the Traces debugger.

Privacy by default: nothing is captured unless the operator opts in. Effective
enablement resolves as (most-recent human intent wins):

    Redis runtime override (console toggle)  >  MLPAL_CAPTURE_PAYLOADS env
        >  config/gateway.yaml capture.enabled  >  False

Everything here stays OFF the request hot path: callers fire
``asyncio.create_task(capture_payload(...))`` (or capture inside an existing
background task), storage opens its own DB session, and any failure is logged
and dropped — capture is debug data, never billing data.

Storage: ``request_payloads`` — zlib-compressed body bytes (LLM payloads are
text; ~5–10× smaller), per-body size caps with a truncated flag, and a daily
retention purge. Auth headers are never stored (bodies only).
"""

from __future__ import annotations

import json
import logging
import os
import time
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select

from mlpal_assistants_service.core.file_config import file_config_section
from mlpal_assistants_service.db.models.request_payload import RequestPayload

logger = logging.getLogger(__name__)

REDIS_OVERRIDE_KEY = "capture:enabled"  # "true" | "false" (absent = no override)
_CACHE_TTL_SECONDS = 10.0  # worst-case toggle propagation delay


@dataclass(frozen=True)
class CaptureConfig:
    enabled: bool
    source: str  # "runtime" | "env" | "file" | "default"
    max_body_kb: int
    retention_days: int
    # Deployment default for keys with NO capture_policy ("inherit"). "on"
    # preserves historical behavior (subsystem enabled = capture everything);
    # "off" turns the subsystem into an allow-list where only keys with an
    # explicit {"mode": "on"} are captured.
    key_default: str = "on"


def _file_settings() -> tuple[bool | None, int, int, str]:
    section = file_config_section("capture")
    enabled = section.get("enabled")
    key_default = section.get("key_default")
    return (
        enabled if isinstance(enabled, bool) else None,
        int(section.get("max_body_kb", 256)),
        int(section.get("retention_days", 7)),
        key_default if key_default in ("on", "off") else "on",
    )


def resolve_config(runtime_override: bool | None) -> CaptureConfig:
    """Pure precedence resolution — unit-testable without I/O."""
    file_enabled, max_body_kb, retention_days, key_default = _file_settings()
    env_raw = os.environ.get("MLPAL_CAPTURE_PAYLOADS")

    if runtime_override is not None:
        enabled, source = runtime_override, "runtime"
    elif env_raw is not None:
        enabled, source = env_raw.strip().lower() in ("1", "true", "yes"), "env"
    elif file_enabled is not None:
        enabled, source = file_enabled, "file"
    else:
        enabled, source = False, "default"
    return CaptureConfig(enabled, source, max_body_kb, retention_days, key_default)


def key_allows_capture(
    capture_policy: dict | None,
    key_default: str,
    *models: str | None,
) -> bool:
    """Per-key capture decision — pure, zero I/O, called on the hot path
    BEFORE a capture task is spawned (so an opted-out key doesn't even pay
    the task spawn).

    Semantics (the operator subsystem toggle is checked separately and ANDed
    by callers; this function answers only "does the KEY allow it"):
      * policy None      -> the deployment key_default decides ("on"/"off")
      * mode "off"       -> False, always — the hard promise
      * mode "on"        -> True, subject to the models filter
      * "models" present -> capture only when ANY given tag (requested or
                            resolved) is in the list; no globbing — capture
                            is a debug tool, exactness beats convenience
    Malformed policies fail CLOSED (no capture): a privacy control must
    degrade toward not storing data.
    """
    if capture_policy is None:
        return key_default == "on"
    if not isinstance(capture_policy, dict):
        return False
    mode = capture_policy.get("mode")
    if mode == "off":
        return False
    if mode != "on":
        return False
    allowed = capture_policy.get("models")
    if not allowed:
        return True
    return any(m is not None and m in allowed for m in models)


class CaptureState:
    """Process-local cached view of the effective config. The per-request check
    must cost nothing — Redis is consulted at most once per TTL window."""

    def __init__(self) -> None:
        self._cached: CaptureConfig | None = None
        self._cached_at = 0.0

    async def config(self, redis: Any | None) -> CaptureConfig:
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < _CACHE_TTL_SECONDS:
            return self._cached
        override: bool | None = None
        if redis is not None:
            try:
                raw = await redis.get(REDIS_OVERRIDE_KEY)
                if raw is not None:
                    text = raw.decode() if isinstance(raw, bytes) else str(raw)
                    override = text.strip().lower() == "true"
            except Exception:  # noqa: BLE001 — Redis trouble must not affect requests
                override = None
        self._cached = resolve_config(override)
        self._cached_at = now
        return self._cached

    async def set_runtime_override(self, redis: Any, enabled: bool) -> CaptureConfig:
        await redis.set(REDIS_OVERRIDE_KEY, "true" if enabled else "false")
        self._cached = None  # take effect immediately on this instance
        return await self.config(redis)


capture_state = CaptureState()


# ── compression helpers (pure) ───────────────────────────────────────────────


def compress_body(body: Any, max_body_kb: int) -> tuple[bytes, int, bool]:
    """JSON-encode + zlib a body under the size cap.

    Returns (compressed, original_size_bytes, truncated). Truncation happens on
    the ENCODED text before compression so the stored payload stays valid UTF-8
    (decompressed output is marked, not silently cut mid-glyph)."""
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)
    raw = text.encode("utf-8")
    original = len(raw)
    cap = max_body_kb * 1024
    truncated = original > cap
    if truncated:
        raw = raw[:cap].decode("utf-8", errors="ignore").encode("utf-8")
        raw += b"\n... [truncated]"
    return zlib.compress(raw, 6), original, truncated


def decompress_body(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8")


# ── async storage (own session; log-and-drop on failure) ─────────────────────


async def capture_payload(
    trace_id: str,
    request_body: Any,
    response_body: Any,
    max_body_kb: int,
) -> None:
    """Persist one request/response pair. Runs only in background tasks."""
    try:
        import asyncio

        from mlpal_assistants_service.db.session import async_session_factory

        # JSON-encode + zlib are CPU work — run off the event loop so a large
        # body can never stall concurrent responses mid-write.
        req_gz, req_bytes, req_trunc = await asyncio.to_thread(
            compress_body, request_body, max_body_kb
        )
        resp_gz, resp_bytes, resp_trunc = await asyncio.to_thread(
            compress_body, response_body, max_body_kb
        )
        async with async_session_factory() as session:
            session.add(
                RequestPayload(
                    trace_id=trace_id,
                    request_gz=req_gz,
                    response_gz=resp_gz,
                    request_bytes=req_bytes,
                    response_bytes=resp_bytes,
                    truncated=req_trunc or resp_trunc,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 — debug data: never propagate, never retry
        logger.exception(f"payload capture failed trace={trace_id}")


async def fetch_payload(session: Any, trace_id: str) -> dict | None:
    row = (
        await session.execute(select(RequestPayload).where(RequestPayload.trace_id == trace_id))
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "trace_id": row.trace_id,
        "request": decompress_body(row.request_gz),
        "response": decompress_body(row.response_gz),
        "request_bytes": row.request_bytes,
        "response_bytes": row.response_bytes,
        "truncated": row.truncated,
        "captured_at": row.created_at.isoformat() if row.created_at else None,
    }


async def purge_expired(session: Any, retention_days: int) -> int:
    """Delete payloads past retention. Idempotent; safe on every instance."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    result = await session.execute(
        delete(RequestPayload).where(RequestPayload.created_at < cutoff)
    )
    await session.commit()
    return result.rowcount or 0
