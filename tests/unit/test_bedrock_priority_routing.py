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


def test_native_bedrock_backend_targets_runtime_by_profile_id():
    import json

    from mlpal_assistants_service.services.bedrock_mantle import BedrockMantleClient

    c = BedrockMantleClient(region="us-east-2", endpoint="runtime", model_map={"claude-opus-5": "global.anthropic.claude-opus-5"})
    assert c.url == "https://bedrock-runtime.us-east-2.amazonaws.com/anthropic/v1/messages"
    body, obj, removed = c.adapt_body(json.dumps({"model": "claude-opus-5", "max_tokens": 5, "messages": [],
                                                  "output_config": {"format": {"type": "json_schema"}}}).encode())
    assert obj["model"] == "global.anthropic.claude-opus-5" and obj["anthropic_version"] == "bedrock-2023-05-31"
    assert "output_config" in obj and removed == []          # runtime keeps output_config byte-faithful
    # unmapped model on runtime is passed through untouched (Bedrock answers with a clear 4xx)
    _, obj2, _ = c.adapt_body(json.dumps({"model": "claude-fable-5-1", "messages": []}).encode())
    assert obj2["model"] == "claude-fable-5-1"


def test_legacy_mantle_endpoint_keeps_prefix_and_output_config_strip():
    import json

    from mlpal_assistants_service.services.bedrock_mantle import BedrockMantleClient

    c = BedrockMantleClient(region="us-east-1", endpoint="mantle")
    assert c.url == "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
    _, obj, removed = c.adapt_body(json.dumps({"model": "claude-haiku-4-5", "messages": [],
                                               "output_config": {"format": {"type": "json_schema"}}}).encode())
    assert obj["model"] == "anthropic.claude-haiku-4-5" and "output_config" not in obj and removed


def test_runtime_override_moves_the_native_wire_too(bedrock_first, monkeypatch):
    """PUT /admin/v1/settings/anthropic_backends must flip BOTH wires: the
    adapter factory already honoured the override; the native selection
    read only the env value (found 2026-09-17)."""
    from mlpal_assistants_service.services import runtime_settings as rs

    s, f = bedrock_first
    monkeypatch.setattr(s, "bedrock_mantle_models", json.dumps(["claude-opus-5"]))
    ab._backends.clear()
    ab._backend_lists.clear()
    assert ab.native_backend_for(s, "claude-opus-5").name == "bedrock"
    assert f.serving_backend_for("anthropic", "claude-opus-5") == "bedrock"
    # runtime flip back to first-party (env untouched)
    monkeypatch.setitem(rs._store, "anthropic_backends", "first_party,bedrock")
    rs._invalidate_dependents()
    assert ab.effective_anthropic_backends(s) == "first_party,bedrock"
    assert ab.native_backend_for(s, "claude-opus-5").name == "first_party"
    assert f.serving_backend_for("anthropic", "claude-opus-5") == "first_party"
    # clearing the override returns to the env order
    monkeypatch.delitem(rs._store, "anthropic_backends")
    rs._invalidate_dependents()
    assert ab.native_backend_for(s, "claude-opus-5").name == "bedrock"
    assert f.serving_backend_for("anthropic", "claude-opus-5") == "bedrock"
