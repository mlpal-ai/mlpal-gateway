"""Multi-cloud serving backends: priority resolution, model maps, plugins.

Contract (worklog 2026-08-14-multicloud-backends):
- MLPAL_<FAMILY>_BACKENDS is priority-ordered; first backend that is
  configured AND serves the model wins; unconfigured/unknown entries are
  skipped, never fatal.
- Wire model IDs are mapped per backend (Azure deployment names, Bedrock/
  Vertex Claude maps); first-party is identity.
- Resolution is cached in-process — the hot path is a dict lookup.
- Plugins register via the `mlpal.adapters` entry-point group; `family:backend`
  names add serving backends, plain names add providers.
"""

from __future__ import annotations

import pytest

from mlpal_assistants_service.adapters.base import BaseAdapter
from mlpal_assistants_service.adapters.factory import AdapterFactory
from mlpal_assistants_service.adapters.openai import OpenAIAdapter
from mlpal_assistants_service.adapters.serving import (
    AzureOpenAIAdapter,
    BedrockAnthropicAdapter,
)
from mlpal_assistants_service.core.config import get_settings


@pytest.fixture()
def factory(monkeypatch):
    """Fresh factory with neutral backend config and no cached instances."""
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "first_party")
    monkeypatch.setattr(s, "google_backends", "first_party")
    monkeypatch.setattr(s, "anthropic_backends", "first_party")
    monkeypatch.setattr(s, "azure_openai_endpoint", None)
    monkeypatch.setattr(s, "azure_openai_api_key", None)
    monkeypatch.setattr(s, "azure_openai_deployments", None)
    monkeypatch.setattr(s, "vertex_project", None)
    monkeypatch.setattr(s, "bedrock_anthropic_models", None)
    monkeypatch.setattr(s, "vertex_anthropic_models", None)
    monkeypatch.setattr(s, "openai_api_key", "sk-test")
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    f = AdapterFactory()
    f.clear_instances()
    yield f
    f.clear_instances()


def test_first_party_default(factory):
    adapter, wire_id = factory.resolve("openai", "gpt-5.2")
    assert type(adapter) is OpenAIAdapter
    assert adapter.backend_name == "first_party"
    assert wire_id == "gpt-5.2"


def test_azure_wins_priority_with_deployment_map(factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "azure,first_party")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://res.services.ai.azure.com")
    monkeypatch.setattr(s, "azure_openai_api_key", "az-key")
    monkeypatch.setattr(s, "azure_openai_deployments", '{"gpt-5.2": "my-gpt52"}')

    adapter, wire_id = factory.resolve("openai", "gpt-5.2")
    assert isinstance(adapter, AzureOpenAIAdapter)
    assert wire_id == "my-gpt52"
    assert str(adapter._client.base_url).startswith(
        "https://res.services.ai.azure.com/openai/v1"
    )

    # Model not in the map → Azure doesn't serve it → falls to first_party.
    adapter2, wire2 = factory.resolve("openai", "gpt-4o")
    assert type(adapter2) is OpenAIAdapter
    assert wire2 == "gpt-4o"


def test_azure_identity_convention_without_map(factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "azure")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://res.openai.azure.com/")
    monkeypatch.setattr(s, "azure_openai_api_key", "az-key")

    adapter, wire_id = factory.resolve("openai", "gpt-5.2")
    assert isinstance(adapter, AzureOpenAIAdapter)
    assert wire_id == "gpt-5.2"  # deployment named after the model ID


def test_bedrock_claude_requires_explicit_map(factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "anthropic_backends", "bedrock,first_party")
    # No map → bedrock unconfigured → first_party serves.
    adapter, wire_id = factory.resolve("anthropic", "claude-haiku-4-5-20251001")
    assert adapter.backend_name == "first_party"

    factory.clear_instances()
    monkeypatch.setattr(
        s,
        "bedrock_anthropic_models",
        '{"claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5-20251001-v1:0"}',
    )
    adapter, wire_id = factory.resolve("anthropic", "claude-haiku-4-5-20251001")
    assert isinstance(adapter, BedrockAnthropicAdapter)
    assert wire_id == "anthropic.claude-haiku-4-5-20251001-v1:0"
    # Unmapped Claude model falls through to first_party.
    adapter2, _ = factory.resolve("anthropic", "claude-opus-5")
    assert adapter2.backend_name == "first_party"


def test_unconfigured_and_unknown_backends_skipped(factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "wat,azure,first_party")
    # azure has no endpoint/key → skipped; "wat" unknown → skipped.
    adapter, _ = factory.resolve("openai", "gpt-5.2")
    assert adapter.backend_name == "first_party"


def test_no_backend_serves_raises(factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", None)
    with pytest.raises(ValueError, match="No configured backend"):
        factory.resolve("openai", "gpt-5.2")
    assert factory.serving_backend_for("openai", "gpt-5.2") is None


def test_resolution_is_cached(factory):
    first = factory.resolve("openai", "gpt-5.2")
    second = factory.resolve("openai", "gpt-5.2")
    assert first is second  # same tuple object — pure dict hit


def test_family_enabled_via_cloud_backend_only(factory, monkeypatch):
    """Azure-only box: no OPENAI_API_KEY, but the family is enabled and its
    primary adapter is the Azure backend (founder bug: unserved vs served)."""
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", None)
    monkeypatch.setattr(s, "openai_backends", "azure")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://res.openai.azure.com")
    monkeypatch.setattr(s, "azure_openai_api_key", "az-key")

    assert factory.is_enabled("openai") is True
    primary = factory.primary("openai")
    assert isinstance(primary, AzureOpenAIAdapter)
    assert "openai" in factory.get_enabled()


def test_plugin_backend_and_provider_registration(factory, monkeypatch):
    class DummyBackend(BaseAdapter):
        provider_name = "openai"
        backend_name = "mycloud"

        def __init__(self):
            pass

        def serves(self, provider_model_id):
            return provider_model_id == "gpt-5.2"

        # abstract surface — not exercised here
        async def chat(self, *a, **k): ...
        async def chat_stream(self, *a, **k): ...
        async def embed(self, *a, **k): ...
        async def generate_image(self, *a, **k): ...
        async def transcribe(self, *a, **k): ...
        async def text_to_speech(self, *a, **k): ...
        async def health_check(self, *a, **k): ...

    class FakeEP:
        def __init__(self, name, cls):
            self.name = name
            self._cls = cls

        def load(self):
            return self._cls

    monkeypatch.setattr(
        "mlpal_assistants_service.adapters.factory.entry_points",
        lambda group: [FakeEP("openai:mycloud", DummyBackend)],
    )
    f = AdapterFactory()
    f.clear_instances()
    assert ("openai", "mycloud") in f._backend_classes

    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "mycloud,first_party")
    adapter, wire_id = f.resolve("openai", "gpt-5.2")
    assert adapter.backend_name == "mycloud"
    adapter2, _ = f.resolve("openai", "gpt-4o")  # plugin doesn't serve → fall through
    assert adapter2.backend_name == "first_party"
    f.clear_instances()


def test_broken_plugin_is_isolated(factory, monkeypatch):
    class FakeEP:
        name = "openai:broken"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(
        "mlpal_assistants_service.adapters.factory.entry_points",
        lambda group: [FakeEP()],
    )
    f = AdapterFactory()  # must not raise
    assert ("openai", "broken") not in f._backend_classes


def test_v2_native_backend_selection(monkeypatch):
    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        AnthropicBedrockBackend,
        AnthropicFirstPartyBackend,
        get_anthropic_backend,
    )

    s = get_settings()
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(s, "anthropic_backends", "first_party,bedrock")
    assert isinstance(get_anthropic_backend(s), AnthropicFirstPartyBackend)

    monkeypatch.setattr(s, "anthropic_api_key", None)
    backend = get_anthropic_backend(s)
    assert isinstance(backend, AnthropicBedrockBackend)
    assert "bedrock-runtime" in backend.url   # native Anthropic wire on bedrock-runtime (default)

    monkeypatch.setattr(s, "anthropic_backends", "first_party")
    with pytest.raises(ValueError, match="No usable native Anthropic backend"):
        get_anthropic_backend(s)


def test_v2_first_party_prepare_passthrough(monkeypatch):
    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        AnthropicFirstPartyBackend,
    )

    s = get_settings()
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    b = AnthropicFirstPartyBackend(s)
    body = b'{"model":"claude-opus-5","max_tokens":1}'
    content, headers = b.prepare(body, {"anthropic-beta": "context-1m-2025-08-07"})
    assert content == body  # byte-faithful
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-beta"] == "context-1m-2025-08-07"


def test_v2_bedrock_serves_only_mantle_allowlist(monkeypatch):
    """Mantle's population is a subset of bedrock-runtime's — the native
    backend must only claim allowlisted models so core falls back to the
    adapter path for the rest."""
    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        AnthropicBedrockBackend,
        AnthropicFirstPartyBackend,
    )

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    s = get_settings()
    monkeypatch.setattr(s, "bedrock_mantle_models", '["claude-opus-5"]')
    b = AnthropicBedrockBackend(s)
    assert b.serves("claude-opus-5") is True
    assert b.serves("claude-haiku-4-5-20251001") is False

    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    assert AnthropicFirstPartyBackend(s).serves("anything") is True


def test_translating_edge_wire_model_override():
    from mlpal_assistants_service.services.messages_v2.translating_edge import (
        TranslatingEdge,
    )

    class FakeAdapter:
        pass

    edge = TranslatingEdge(FakeAdapter(), wire_model_id="global.anthropic.claude-opus-5")
    assert edge._wire_model_id == "global.anthropic.claude-opus-5"
    assert TranslatingEdge(FakeAdapter())._wire_model_id is None


def test_v2_bedrock_prepare_adapts_and_signs(monkeypatch):
    import json

    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        AnthropicBedrockBackend,
    )

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    s = get_settings()
    monkeypatch.setattr(s, "bedrock_anthropic_models", '{"claude-opus-5": "global.anthropic.claude-opus-5"}')
    monkeypatch.setattr(s, "bedrock_native_endpoint", "runtime")
    b = AnthropicBedrockBackend(s)
    body = json.dumps({"model": "claude-opus-5", "max_tokens": 1, "messages": []}).encode()
    content, headers = b.prepare(body, {})
    adapted = json.loads(content)
    assert adapted["model"] == "global.anthropic.claude-opus-5"   # profile id from the verified map
    assert "anthropic_version" in adapted
    assert any(h.lower() == "authorization" for h in headers)  # SigV4 applied


def test_azure_claude_adapter_resolution(factory, monkeypatch):
    """Claude via Microsoft Foundry: same AIServices creds as Azure OpenAI,
    deployment map (or identity) for model IDs."""
    from mlpal_assistants_service.adapters.serving import AzureAnthropicAdapter

    s = get_settings()
    monkeypatch.setattr(s, "anthropic_backends", "azure,first_party")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://res.services.ai.azure.com")
    monkeypatch.setattr(s, "azure_openai_api_key", "az-key")
    monkeypatch.setattr(s, "azure_anthropic_deployments", '{"claude-haiku-4-5-20251001": "my-haiku"}')

    adapter, wire_id = factory.resolve("anthropic", "claude-haiku-4-5-20251001")
    assert isinstance(adapter, AzureAnthropicAdapter)
    assert wire_id == "my-haiku"
    assert str(adapter._client.base_url).startswith("https://res.services.ai.azure.com/anthropic")
    # Unmapped model falls through to first_party.
    adapter2, _ = factory.resolve("anthropic", "claude-opus-5")
    assert adapter2.backend_name == "first_party"


def test_v2_azure_native_backend(monkeypatch):
    import json

    from mlpal_assistants_service.services.messages_v2.anthropic_backend import (
        AnthropicAzureBackend,
        get_anthropic_backend,
    )

    s = get_settings()
    monkeypatch.setattr(s, "anthropic_api_key", None)
    monkeypatch.setattr(s, "anthropic_backends", "azure")
    monkeypatch.setattr(s, "azure_openai_endpoint", "https://res.services.ai.azure.com")
    monkeypatch.setattr(s, "azure_openai_api_key", "az-key")
    monkeypatch.setattr(s, "azure_anthropic_deployments", '{"claude-opus-5": "opus5-dep"}')

    b = get_anthropic_backend(s)
    assert isinstance(b, AnthropicAzureBackend)
    assert b.url == "https://res.services.ai.azure.com/anthropic/v1/messages"
    assert b.serves("claude-opus-5") and not b.serves("claude-sonnet-5")

    body = json.dumps({"model": "claude-opus-5", "max_tokens": 1}).encode()
    content, headers = b.prepare(body, {"anthropic-beta": "context-1m-2025-08-07"})
    assert json.loads(content)["model"] == "opus5-dep"
    assert headers["x-api-key"] == "az-key"
    assert headers["anthropic-beta"] == "context-1m-2025-08-07"

    # Identity convention: no map -> serves everything, body untouched.
    monkeypatch.setattr(s, "azure_anthropic_deployments", None)
    b2 = AnthropicAzureBackend(s)
    assert b2.serves("anything")
    content2, _ = b2.prepare(body, {})
    assert content2 == body
