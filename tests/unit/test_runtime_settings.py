"""Runtime-overridable settings: whitelist, validation, precedence.

Contract (worklog 2026-08-17-ux-program): runtime override > env > default;
only HOT_SETTINGS names are overridable; backend CSVs validate against the
registered backends for the family; changes invalidate the factory's cached
resolution so they apply without restart.
"""

from __future__ import annotations

import pytest

from mlpal_assistants_service.adapters.factory import AdapterFactory
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.services import runtime_settings


class _FakeSession:
    """set_meta/get_meta take (session, ...) — capture writes in a dict."""

    def __init__(self, store):
        self.store = store


@pytest.fixture(autouse=True)
def _clean_store():
    runtime_settings._store.clear()
    yield
    runtime_settings._store.clear()


@pytest.mark.asyncio
async def test_set_value_rejects_unknown_setting_and_bad_backend(monkeypatch):
    async def fake_set_meta(session, key, value):
        pass

    monkeypatch.setattr(
        "mlpal_assistants_service.services.catalog_feed.set_meta", fake_set_meta
    )
    with pytest.raises(KeyError):
        await runtime_settings.set_value(None, "db_password", "nope")
    with pytest.raises(ValueError, match="unknown backend"):
        await runtime_settings.set_value(None, "openai_backends", "wat,first_party")
    with pytest.raises(ValueError, match="empty"):
        await runtime_settings.set_value(None, "openai_backends", " , ")


@pytest.mark.asyncio
async def test_set_and_clear_override_precedence(monkeypatch):
    writes = {}

    async def fake_set_meta(session, key, value):
        writes[key] = value

    monkeypatch.setattr(
        "mlpal_assistants_service.services.catalog_feed.set_meta", fake_set_meta
    )

    assert runtime_settings.get("openai_backends") is None
    await runtime_settings.set_value(None, "openai_backends", "azure, first_party")
    assert runtime_settings.get("openai_backends") == "azure,first_party"  # normalized
    assert runtime_settings.source_of("openai_backends") == "runtime"
    assert writes["setting:openai_backends"] == "azure,first_party"

    await runtime_settings.set_value(None, "openai_backends", None)
    assert runtime_settings.get("openai_backends") is None
    assert writes["setting:openai_backends"] is None


def test_factory_priority_honors_runtime_override(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "first_party")
    monkeypatch.setattr(s, "azure_openai_endpoint", None)
    monkeypatch.setattr(s, "azure_openai_api_key", None)
    f = AdapterFactory()
    assert f._priority("openai") == ["first_party"]

    monkeypatch.setitem(runtime_settings._store, "openai_backends", "azure,first_party")
    assert f._priority("openai") == ["azure", "first_party"]


@pytest.mark.asyncio
async def test_set_value_clears_factory_resolution(monkeypatch):
    async def fake_set_meta(session, key, value):
        pass

    monkeypatch.setattr(
        "mlpal_assistants_service.services.catalog_feed.set_meta", fake_set_meta
    )
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "sk-test")
    monkeypatch.setattr(s, "openai_backends", "first_party")
    f = AdapterFactory()
    f.clear_instances()
    f.resolve("openai", "gpt-5.2")
    assert ("openai", "gpt-5.2", frozenset()) in f._resolution

    await runtime_settings.set_value(None, "openai_backends", "first_party")
    assert ("openai", "gpt-5.2", frozenset()) not in f._resolution
    f.clear_instances()


@pytest.mark.asyncio
async def test_bundled_catalog_boot_apply_is_version_gated(monkeypatch):
    """apply_bundled_if_new reconciles once per bundled version, then no-ops —
    the merge-PR → deploy → DB path with no manual step (worklog 2026-08-17)."""
    from types import SimpleNamespace

    from mlpal_assistants_service.services import catalog_feed

    meta: dict[str, str] = {}
    calls = {"reconcile": 0}

    async def fake_get_meta(session, key):
        return meta.get(key)

    async def fake_set_meta(session, key, value):
        meta[key] = value

    async def fake_reconcile(session, registry, pricing, routing, retire_message=None):
        calls["reconcile"] += 1
        assert isinstance(routing, list)  # wrapper doc unwrapped
        return SimpleNamespace(inserted=1, updated=2, retired=0)

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def commit(self):
            pass

    monkeypatch.setattr(catalog_feed, "get_meta", fake_get_meta)
    monkeypatch.setattr(catalog_feed, "set_meta", fake_set_meta)
    monkeypatch.setattr(
        "mlpal_assistants_service.services.catalog_sync.reconcile", fake_reconcile
    )

    out = await catalog_feed.apply_bundled_if_new(FakeSession)
    assert out is not None and out["inserted"] == 1
    assert calls["reconcile"] == 1

    out2 = await catalog_feed.apply_bundled_if_new(FakeSession)
    assert out2 is None  # same bundled version — gated
    assert calls["reconcile"] == 1
