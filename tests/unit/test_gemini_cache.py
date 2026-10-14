"""Unit tests for explicit Gemini prefix caching.

Covers the cache service (create-once / reuse / cost-guard / no-orphan race /
failure-safety), the cache_control TTL parser, and the adapter's decision to
swap the stable prefix for a cachedContent reference (including the AUTO-only
tool-mode guard). No network — Redis and the genai client are faked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.services import gemini_cache
from mlpal_assistants_service.services.messages_v2.translate_in import cache_control_ttl

BIG_SYSTEM = "x" * 20000  # comfortably over min_tokens*3 chars → passes the size gate


class FakeRedis:
    """Minimal async Redis stand-in supporting get / set(nx,ex) / delete."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


def _fake_genai(name: str = "cachedContents/abc") -> Any:
    client = MagicMock()
    created = MagicMock()
    created.name = name
    client.aio.caches.create = AsyncMock(return_value=created)
    client.aio.caches.delete = AsyncMock()
    return client


@pytest.fixture
def redis(monkeypatch) -> FakeRedis:
    fake = FakeRedis()
    monkeypatch.setattr(gemini_cache, "_get_redis", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# cache service
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_create_on_miss_then_reuse_without_second_create(redis):
    client = _fake_genai()
    name1 = await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", BIG_SYSTEM, None, 3600)
    assert name1 == "cachedContents/abc"
    assert client.aio.caches.create.await_count == 1

    # Second call for the same prefix must reuse the Redis entry, not re-create.
    name2 = await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", BIG_SYSTEM, None, 3600)
    assert name2 == name1
    assert client.aio.caches.create.await_count == 1


@pytest.mark.asyncio
async def test_small_prefix_is_not_cached(redis):
    client = _fake_genai()
    assert await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", "tiny", None, 3600) is None
    client.aio.caches.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_without_explicit_cache_is_skipped_before_any_create(redis):
    # Gemini 4 rejects cachedContent outright; the gate must short-circuit so a
    # client opting in gets the implicit cache, not a failing create per request.
    client = _fake_genai()
    assert gemini_cache.supports_explicit_cache("gemini-4-argon") is False
    assert gemini_cache.supports_explicit_cache("gemini-3.5-flash") is True
    assert await gemini_cache.get_or_create_cache(client, "gemini-4-argon", BIG_SYSTEM, None, 3600) is None
    client.aio.caches.create.assert_not_awaited()
    assert redis.store == {}


@pytest.mark.asyncio
async def test_no_redis_disables_caching(monkeypatch):
    monkeypatch.setattr(gemini_cache, "_get_redis", lambda: None)
    client = _fake_genai()
    assert await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", BIG_SYSTEM, None, 3600) is None
    client.aio.caches.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_failure_falls_back_to_none(redis):
    client = _fake_genai()
    client.aio.caches.create = AsyncMock(side_effect=RuntimeError("too small"))
    assert await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", BIG_SYSTEM, None, 3600) is None


@pytest.mark.asyncio
async def test_concurrent_create_race_deletes_own_orphan(monkeypatch):
    # Another pod stores the winning name AFTER our get (miss) but BEFORE our
    # SET NX — so NX fails and we must delete the cache we just created.
    class RacingRedis(FakeRedis):
        async def set(self, key, value, nx=False, ex=None):
            self.store[key] = "cachedContents/theirs"
            return None  # NX loses

    fake = RacingRedis()
    monkeypatch.setattr(gemini_cache, "_get_redis", lambda: fake)
    client = _fake_genai(name="cachedContents/mine")

    name = await gemini_cache.get_or_create_cache(client, "gemini-3.5-flash", BIG_SYSTEM, None, 3600)
    assert name == "cachedContents/theirs"          # we reuse the winner's cache
    client.aio.caches.delete.assert_awaited_once()  # and delete our own orphan


@pytest.mark.asyncio
async def test_invalidate_drops_mapping(redis):
    key = gemini_cache._cache_key("gemini-3.5-flash", BIG_SYSTEM, None)
    redis.store[key] = "cachedContents/x"
    await gemini_cache.invalidate("gemini-3.5-flash", BIG_SYSTEM, None)
    assert key not in redis.store


# --------------------------------------------------------------------------- #
# cache_control TTL parser
# --------------------------------------------------------------------------- #
def test_ttl_none_when_no_cache_control():
    assert cache_control_ttl({"messages": [{"role": "user", "content": "hi"}]}, 3600) is None


def test_ttl_default_when_marker_has_no_ttl():
    body = {"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]}
    assert cache_control_ttl(body, 3600) == 3600


def test_ttl_explicit_and_max_wins():
    body = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "u", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
        ],
    }
    assert cache_control_ttl(body, 999) == 3600  # 1h beats 5m


def test_ttl_detected_on_tool_definition():
    body = {"tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}]}
    assert cache_control_ttl(body, 300) == 300


# --------------------------------------------------------------------------- #
# adapter: prefix-cache application
# --------------------------------------------------------------------------- #
def _adapter():
    from mlpal_assistants_service.adapters.google import GoogleAdapter

    a = GoogleAdapter.__new__(GoogleAdapter)  # skip __init__ (needs API key)
    a._client = _fake_genai()
    return a


@pytest.mark.asyncio
async def test_apply_cache_noop_without_ttl():
    a = _adapter()
    config = {"system_instruction": "s", "tools": ["t"]}
    assert await a._apply_prefix_cache(config, "gemini-3.5-flash", None, None, None) is None
    assert "cached_content" not in config  # untouched


@pytest.mark.asyncio
async def test_apply_cache_skips_forced_tool_choice(monkeypatch):
    a = _adapter()
    monkeypatch.setattr(gemini_cache, "get_or_create_cache", AsyncMock(return_value="cachedContents/x"))
    config = {"system_instruction": "s", "tools": ["t"], "tool_config": {"mode": "any"}}
    # forced tool_choice can't ride a cache (AUTO-only) → normal path, cache untouched
    assert await a._apply_prefix_cache(config, "gemini-3.5-flash", ["tool"], "required", 3600) is None
    assert "cached_content" not in config
    gemini_cache.get_or_create_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_cache_strips_prefix_and_references_cache(monkeypatch):
    a = _adapter()
    monkeypatch.setattr(gemini_cache, "get_or_create_cache", AsyncMock(return_value="cachedContents/x"))
    config = {"system_instruction": "s", "tools": ["t"], "tool_config": {"mode": "auto"}, "temperature": 0.7}
    stripped = await a._apply_prefix_cache(config, "gemini-3.5-flash", ["tool"], "auto", 3600)

    assert stripped == {"system_instruction": "s", "tools": ["t"], "tool_config": {"mode": "auto"}}
    assert config["cached_content"] == "cachedContents/x"
    assert "system_instruction" not in config and "tools" not in config and "tool_config" not in config
    assert config["temperature"] == 0.7  # unrelated config preserved


# --------------------------------------------------------------------------- #
# edge gating: the flag/provider/cache_control safety gate
# --------------------------------------------------------------------------- #
def _edge_ctx(provider: str):
    from mlpal_assistants_service.services.messages_v2.edges import RequestContext

    return RequestContext(
        model_tag="m", provider=provider, provider_model_id="gemini-3.5-flash",
        backend="first_party", trace_id="t", api_key=None, headers={},
    )


def _req(cache_control: bool):
    from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest

    content = [{"type": "text", "text": "hi"}]
    if cache_control:
        content[0]["cache_control"] = {"type": "ephemeral"}
    body = {"model": "gemini-3.5-flash", "max_tokens": 64,
            "messages": [{"role": "user", "content": content}]}
    return ValidatedRequest(raw_body=b"{}", body=body, model="gemini-3.5-flash", stream=False)


def _edge():
    from mlpal_assistants_service.services.messages_v2.translating_edge import TranslatingEdge

    return TranslatingEdge(adapter=MagicMock())


@pytest.mark.parametrize("flag,provider,cc,expected", [
    (False, "google", True, False),   # flag off → never hint, even with cache_control
    (True, "openai", True, False),    # non-google never gets the google-only hint
    (True, "google", False, False),   # no client cache_control → no hint
    (True, "google", True, True),     # flag on + google + cache_control → hint
])
def test_edge_prefix_cache_gating(monkeypatch, flag, provider, cc, expected):
    from mlpal_assistants_service.core import config as cfg

    monkeypatch.setattr(cfg.get_settings(), "messages_v2_cache_google", flag)
    _, kwargs = _edge()._prepare(_req(cache_control=cc), _edge_ctx(provider))
    assert ("prefix_cache_ttl" in kwargs) is expected
