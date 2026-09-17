"""`MLPAL_ANTHROPIC_BACKENDS=bedrock,first_party` must split cleanly:
the OpenAI wire (adapter path) takes Bedrock for mapped models, while the
Anthropic wire keeps first-party's byte-faithful passthrough for every model
the mantle endpoint does not serve, and count_tokens stays first-party."""

import json

import pytest

from mlpal_assistants_service.adapters.factory import AdapterFactory
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.services.messages_v2 import anthropic_backend as ab

MAP = {"claude-opus-5": "global.anthropic.claude-opus-5", "claude-sonnet-5": "global.anthropic.claude-sonnet-5"}


@pytest.fixture()
def bedrock_first(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "anthropic_backends", "bedrock,first_party")
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(s, "bedrock_anthropic_models", json.dumps(MAP))
    monkeypatch.setattr(s, "bedrock_mantle_models", "[]")
    monkeypatch.setattr(s, "bedrock_mantle_region", "us-east-2")
    monkeypatch.setattr(s, "azure_openai_endpoint", None)
    monkeypatch.setattr(s, "azure_openai_api_key", None)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    ab._backends.clear()
    ab._backend_lists.clear()
    f = AdapterFactory()
    f.clear_instances()
    yield s, f
    f.clear_instances()
    ab._backends.clear()
    ab._backend_lists.clear()


def test_native_backends_follow_priority_order(bedrock_first):
    s, _ = bedrock_first
    assert [b.name for b in ab.native_backends(s)] == ["bedrock", "first_party"]


def test_anthropic_wire_keeps_first_party_when_mantle_serves_nothing(bedrock_first):
    s, _ = bedrock_first
    for model in ("claude-opus-5", "claude-fable-5-1"):
        backend = ab.native_backend_for(s, model)
        assert backend is not None and backend.name == "first_party", model


def test_mantle_allowlist_wins_when_it_serves_the_model(bedrock_first, monkeypatch):
    s, _ = bedrock_first
    monkeypatch.setattr(s, "bedrock_mantle_models", json.dumps(["claude-opus-5"]))
    ab._backends.clear()
    ab._backend_lists.clear()
    assert ab.native_backend_for(s, "claude-opus-5").name == "bedrock"
    assert ab.native_backend_for(s, "claude-fable-5-1").name == "first_party"


def test_count_tokens_backend_skips_bedrock(bedrock_first):
    s, _ = bedrock_first
    assert ab.count_tokens_backend(s).name == "first_party"


def test_openai_wire_adapter_path_takes_bedrock_for_mapped_models_only(bedrock_first):
    _, f = bedrock_first
    assert f.serving_backend_for("anthropic", "claude-opus-5") == "bedrock"
    assert f.serving_backend_for("anthropic", "claude-fable-5-1") == "first_party"


def test_per_model_rules_match_cloud_prefixed_claude_ids():
    from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter

    a = AnthropicAdapter.__new__(AnthropicAdapter)
    assert AnthropicAdapter.canonical_model_id("global.anthropic.claude-opus-5") == "claude-opus-5"
    assert AnthropicAdapter.canonical_model_id("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "claude-haiku-4-5-20251001-v1:0"
    assert AnthropicAdapter.canonical_model_id("claude-opus-5@20260801") == "claude-opus-5"
    # Bedrock 400s "`temperature` is deprecated" when the skip rule misses the decorated id
    assert a._skips_temperature("global.anthropic.claude-opus-5")
    assert a._skips_temperature("global.anthropic.claude-sonnet-5")
    assert not a._skips_temperature("global.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert a.get_model_capabilities("global.anthropic.claude-opus-4-5-20251101-v1:0") is a.get_model_capabilities("claude-opus-4-5-20251101")
