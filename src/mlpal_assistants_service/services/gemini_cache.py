"""Explicit Gemini context caching for the stable /v2/messages prefix.

Gemini has no effective automatic caching for our request shape (measured: it
grabs only a ~2k-token block and is flaky), so we create an explicit
`cachedContent` for the part of the prompt that is stable and reused every turn:
the system instruction + tool declarations. That cache is created once and
reused across every turn of a conversation AND across every conversation that
shares the same agent config (e.g. all coding sessions share one system+tools
cache), so the create cost amortizes heavily.

Deliberately NOT cached here: the growing conversation tail. Caching a prefix
that changes every turn would mint a single-use cachedContent per turn — which
costs input-price to create and is never reused, i.e. strictly worse than not
caching. Advancing/chunked tail caching is a separate, stateful follow-up.

Everything is best-effort: any failure (Redis down, create rejected, provider
error) returns None and the caller falls back to the normal, uncached path.
Never raises, never changes model behavior — a cache reference is
behaviourally identical to sending system+tools inline.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis
from google.genai import types

from mlpal_assistants_service.core.config import get_settings

logger = logging.getLogger(__name__)

_KEY_PREFIX = "gcache:"
_CHARS_PER_TOKEN = 3  # conservative lower bound, for the pre-create size gate

_redis_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis | None:
    """Shared Redis so a cachedContent created by one pod is reused by all.
    Returns None if Redis isn't configured — caching is then skipped (we never
    create a cache we can't record, or we'd leak orphans)."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = aioredis.from_url(get_settings().redis_url)
        except Exception:  # noqa: BLE001
            logger.warning("gemini_cache: redis init failed; explicit caching disabled")
            return None
    return _redis_client


def _cache_key(model: str, system_instruction: str | None, gtools: Any) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update((system_instruction or "").encode())
    h.update(b"\x00")
    h.update(json.dumps(gtools or [], sort_keys=True, default=str).encode())
    return _KEY_PREFIX + h.hexdigest()


# Models whose API rejects explicit cachedContent ("Explicit caching is not
# supported for models/gemini-4-argon", verified 2026-10-14). They cache
# implicitly and report hits in usage_metadata, so there is nothing to create.
_NO_EXPLICIT_CACHE_PREFIXES = ("gemini-4",)


def supports_explicit_cache(model: str) -> bool:
    return not model.startswith(_NO_EXPLICIT_CACHE_PREFIXES)


def _too_small(system_instruction: str | None, gtools: Any) -> bool:
    approx_chars = len(system_instruction or "") + len(json.dumps(gtools or [], default=str))
    return approx_chars < get_settings().messages_v2_cache_min_tokens * _CHARS_PER_TOKEN


async def get_or_create_cache(
    client: Any,
    model: str,
    system_instruction: str | None,
    gtools: Any,
    ttl_seconds: int,
) -> str | None:
    """Return a live cachedContent name for (model, system, tools), creating it
    once and recording it in Redis for reuse. None → caller uses the normal path."""
    if not supports_explicit_cache(model):
        logger.debug(f"gemini_cache: {model} has no explicit cache; implicit caching applies")
        return None
    if _too_small(system_instruction, gtools):
        return None  # cost guard: not worth a create + storage, and Gemini would reject it
    redis = _get_redis()
    if redis is None:
        return None
    key = _cache_key(model, system_instruction, gtools)
    try:
        existing = await redis.get(key)
        if existing:
            return existing.decode() if isinstance(existing, (bytes, bytearray)) else existing
    except Exception:  # noqa: BLE001
        logger.warning("gemini_cache: redis get failed; skipping cache", exc_info=True)
        return None

    cfg = types.CreateCachedContentConfig(ttl=f"{ttl_seconds}s")
    if system_instruction:
        cfg.system_instruction = system_instruction
    if gtools:
        cfg.tools = gtools
    try:
        cache = await client.aio.caches.create(model=model, config=cfg)
    except Exception:  # noqa: BLE001 — too-small / unsupported / provider error
        logger.warning(f"gemini_cache: create failed for {model}; falling back uncached", exc_info=True)
        return None
    name = cache.name

    # SET NX so concurrent creators don't orphan caches: first writer wins;
    # losers delete the cache they just made and reuse the stored one.
    try:
        won = await redis.set(key, name, nx=True, ex=ttl_seconds)
        if won:
            logger.info(f"gemini_cache: created {name} model={model} ttl={ttl_seconds}s")
            return name
        stored = await redis.get(key)
        with contextlib.suppress(Exception):
            await client.aio.caches.delete(name=name)
        return stored.decode() if isinstance(stored, (bytes, bytearray)) else stored
    except Exception:  # noqa: BLE001
        logger.warning("gemini_cache: redis set failed; using created cache anyway", exc_info=True)
        return name  # created it; TTL will reap it even if unrecorded


async def invalidate(model: str, system_instruction: str | None, gtools: Any) -> None:
    """Drop the Redis mapping — e.g. after a 'cached content not found' (Gemini
    expired it before our TTL). The next request recreates. Never raises."""
    redis = _get_redis()
    if redis is None:
        return
    with contextlib.suppress(Exception):
        await redis.delete(_cache_key(model, system_instruction, gtools))
