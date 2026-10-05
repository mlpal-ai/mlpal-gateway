"""Universal reasoning effort: one ordinal ladder, per-model rungs from the
catalog, deterministic clamp toward intent, never silent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from mlpal_assistants_service.adapters.anthropic import _apply_effort
from mlpal_assistants_service.core.exceptions import UnsupportedEffortError, ValidationError
from mlpal_assistants_service.schemas.chat import ChatCompletionRequest
from mlpal_assistants_service.services.messages_v2.translate_in import to_common
from mlpal_assistants_service.services.reasoning_effort import (
    LADDER,
    check_no_native_conflict,
    default_effort,
    resolve_effort,
    supported_levels,
)

ASTRA = {"effort_levels": ["low", "medium", "high", "xhigh", "max"], "default_effort": "medium"}
OPUS46 = {"effort_levels": ["none", "low", "medium", "high", "max"], "default_effort": "high"}
SONNET55 = {"effort_levels": ["none", "low", "medium", "high", "xhigh", "max"], "default_effort": "high"}
GEMINI38 = {"effort_levels": ["low", "medium", "high"], "default_effort": "high"}
PRO = {"effort_levels": ["medium", "high", "xhigh"]}
NO_LEVER = {"tools": True}


def test_ladder_is_ordered_and_supported_levels_follow_it():
    assert LADDER == ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert supported_levels({"effort_levels": ["max", "low", "bogus"]}) == ("low", "max")
    assert supported_levels(None) == ()
    assert default_effort(ASTRA) == "medium" and default_effort({"default_effort": "ultra"}) is None


@pytest.mark.parametrize("requested,caps,applied,clamped", [
    ("high", ASTRA, "high", False),        # supported → as is
    ("none", ASTRA, "low", True),          # below the floor → floor
    ("max", GEMINI38, "high", True),       # above the ceiling → ceiling
    ("xhigh", OPUS46, "high", True),       # gap in the middle → nearest lower (cost-conservative)
    ("low", PRO, "medium", True),          # below a raised floor → floor
    ("minimal", OPUS46, "none", True),     # gap → nearest lower
    ("high", NO_LEVER, None, True),        # no lever → nothing sent, reported as clamp
])
def test_resolution_matrix(requested, caps, applied, clamped):
    res = resolve_effort(requested, caps, model="m")
    assert (res.requested, res.applied, res.clamped) == (requested, applied, clamped)
    assert res.as_metadata() == {"requested": requested, "applied": applied, "clamped": clamped}


def test_omitted_effort_sends_nothing():
    res = resolve_effort(None, ASTRA)
    assert res.applied is None and not res.clamped and res.requested is None


def test_unknown_rung_is_a_validation_error():
    with pytest.raises(ValidationError):
        resolve_effort("ultra", ASTRA)


def test_strict_turns_a_clamp_into_400_naming_supported_set():
    with pytest.raises(UnsupportedEffortError) as e:
        resolve_effort("max", GEMINI38, strict=True, model="gemini-3.8-flash")
    assert e.value.supported == ["low", "medium", "high"]
    assert "gemini-3.8-flash" in e.value.message
    assert resolve_effort("high", GEMINI38, strict=True).applied == "high"


def test_native_kwargs_conflict_is_rejected_only_when_both_present():
    check_no_native_conflict(None, {"reasoning": {"effort": "low"}})
    check_no_native_conflict("high", {"service_tier": "flex"})
    with pytest.raises(ValidationError, match="reasoning"):
        check_no_native_conflict("high", {"reasoning": {"effort": "low"}})
    with pytest.raises(ValidationError, match="thinking_config"):
        check_no_native_conflict("low", {"thinking_config": {}})


def test_chat_schema_accepts_ladder_and_rejects_provider_spellings():
    ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], reasoning_effort="xhigh")
    with pytest.raises(PydanticValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], reasoning_effort="ultra")


# --- provider mappings -------------------------------------------------------

def test_anthropic_none_disables_thinking_and_rungs_ride_output_config():
    p: dict = {"model": "claude-opus-4-6"}
    _apply_effort(p, "none")
    assert p == {"model": "claude-opus-4-6", "thinking": {"type": "disabled"}}
    p = {"model": "claude-opus-4-6", "output_config": {"format": "json"}}
    _apply_effort(p, "xhigh")
    assert p["output_config"] == {"format": "json", "effort": "xhigh"}


def test_sonnet_5_5_none_is_between_tools_on_every_backend_spelling():
    """Probe-verified 2026-10-05: sonnet-5-5 returns 400 for `disabled` and
    wants `between_tools`; opus-5-5 is the reverse (no off switch at all)."""
    from mlpal_assistants_service.adapters.anthropic import thinking_off

    for spelling in ("claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5",
                     "anthropic.claude-sonnet-5-5", "claude-sonnet-5-5@20260928"):
        assert thinking_off(spelling) == {"type": "between_tools"}, spelling
        p = {"model": spelling}
        _apply_effort(p, "none")
        assert p["thinking"] == {"type": "between_tools"}
    for other in ("claude-sonnet-5", "claude-opus-5-5", "claude-opus-4-6", "claude-haiku-4-5-20251001"):
        assert thinking_off(other) == {"type": "disabled"}, other


def test_5_5_generation_structured_output_uses_native_format_not_forced_tool():
    """The 5.5 generation 400s on tool_choice type tool/any, so json_schema
    rides output_config.format; older Claude keeps the forced-tool trick."""
    from mlpal_assistants_service.adapters.anthropic import _apply_json_schema

    spec = {"name": "person", "schema": {"type": "object", "properties": {"n": {"type": "string"}}}}
    for m in ("claude-sonnet-5-5", "global.anthropic.claude-opus-5-5"):
        p = {"model": m, "output_config": {"effort": "low"}}
        assert _apply_json_schema(p, spec) is None
        assert p["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": spec["schema"]}}
        assert "tool_choice" not in p and "tools" not in p
    p = {"model": "claude-sonnet-5", "tools": [{"name": "t"}]}
    assert _apply_json_schema(p, spec) == "person"
    assert p["tool_choice"] == {"type": "tool", "name": "person"}
    assert [t["name"] for t in p["tools"]] == ["t", "person"]
    p = {}
    _apply_effort(p, None)
    assert p == {}


def test_anthropic_wire_explicit_effort_beats_thinking_budget():
    body = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 20000}, "output_config": {"effort": "low"}}
    assert to_common(body).reasoning_effort == "low"
    del body["output_config"]
    assert to_common(body).reasoning_effort == "high"   # budget band fallback
    del body["thinking"]
    assert to_common(body).reasoning_effort is None


@pytest.mark.asyncio
async def test_openai_adapter_sends_reasoning_effort_on_responses_wire():
    from mlpal_assistants_service.adapters.openai import OpenAIAdapter

    a = OpenAIAdapter.__new__(OpenAIAdapter)
    a.wire = "responses"
    a.validate_file_attachments = MagicMock()
    a.extract_files_from_messages = MagicMock(return_value=[])
    a._normalize_messages_for_responses = MagicMock(return_value=(None, [{"role": "user", "content": "hi"}]))
    a._model_skips_sampling_params = MagicMock(return_value=True)
    resp = SimpleNamespace(output=[], output_text="ok", usage=SimpleNamespace(
        input_tokens=1, output_tokens=1, input_tokens_details=None,
        output_tokens_details=SimpleNamespace(reasoning_tokens=7)), model="gpt-6-astra", id="r")
    create = AsyncMock(return_value=resp)
    a._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    try:
        await a.chat(model="gpt-6-astra", messages=[{"role": "user", "content": "hi"}], reasoning_effort="max")
    except Exception:
        pass  # response parsing of the stub is not under test
    assert create.call_args.kwargs["reasoning"] == {"effort": "max"}


def test_google_thought_tokens_are_output_tokens():
    from mlpal_assistants_service.adapters.google import _gemini_token_usage

    um = SimpleNamespace(prompt_token_count=17, candidates_token_count=213, thoughts_token_count=359,
                         cached_content_token_count=None, cache_tokens_details=None)
    u = _gemini_token_usage(um)
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (17, 572, 359)


def test_gemini_output_reserve_does_not_override_explicit_level(monkeypatch):
    from google.genai import types

    from mlpal_assistants_service.adapters import google as g

    monkeypatch.setattr(g, "get_settings", lambda: SimpleNamespace(google_thinking_output_reserve=1024))
    cfg = {"thinking_config": types.ThinkingConfig(thinking_level="high")}
    g._apply_thinking_output_reserve(cfg, "gemini-3.8-flash", max_tokens=1500, has_tools=False)
    assert str(cfg["thinking_config"].thinking_level).upper().endswith("HIGH")
    assert cfg["thinking_config"].thinking_budget is None


def test_anthropic_usage_surfaces_thinking_tokens_when_reported():
    from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter

    with_details = SimpleNamespace(input_tokens=1, output_tokens=50, cache_read_input_tokens=0, cache_creation=None,
                                   cache_creation_input_tokens=0, output_tokens_details=SimpleNamespace(thinking_tokens=37))
    assert AnthropicAdapter._token_usage(with_details).reasoning_tokens == 37
    without = SimpleNamespace(input_tokens=1, output_tokens=50, cache_read_input_tokens=0, cache_creation=None,
                              cache_creation_input_tokens=0)
    assert AnthropicAdapter._token_usage(without).reasoning_tokens is None


# --- Anthropic wire (v2 core + native edge) ----------------------------------

def _ctx(caps, **meta):
    from mlpal_assistants_service.services.messages_v2.edges import RequestContext

    model = "claude-sonnet-5-5" if caps is SONNET55 else "m"
    ctx = RequestContext(model_tag=model, provider="anthropic", provider_model_id=model, backend="first_party",
                         trace_id="t", api_key=None, headers={}, capabilities=caps)
    ctx.cc_metadata.update(meta)
    return ctx


def test_core_resolves_effort_before_response_and_records_source():
    from mlpal_assistants_service.services.messages_v2.core import (
        _effort_headers,
        _resolve_request_effort,
    )
    from mlpal_assistants_service.services.messages_v2.schemas import ValidatedRequest

    def req(body):
        return ValidatedRequest(raw_body=b"", body=body, model="m", stream=True)

    ctx = _ctx(GEMINI38)
    _resolve_request_effort(req({"output_config": {"effort": "max"}}), ctx)
    assert ctx.cc_metadata["reasoning_effort"] == {"requested": "max", "applied": "high", "clamped": True, "source": "explicit"}
    assert _effort_headers(ctx) == {"X-MLPal-Reasoning-Effort": "max->high"}
    ctx = _ctx(ASTRA)
    _resolve_request_effort(req({"thinking": {"type": "enabled", "budget_tokens": 20000}}), ctx)
    assert ctx.cc_metadata["reasoning_effort"]["source"] == "thinking"
    ctx = _ctx(ASTRA)
    _resolve_request_effort(req({}), ctx)
    assert "reasoning_effort" not in ctx.cc_metadata and _effort_headers(ctx) == {}
    with pytest.raises(ValidationError):
        _resolve_request_effort(ValidatedRequest(raw_body=b"", body={"output_config": {"effort": "high"}}, model="m",
                                                 stream=False, model_kwargs={"output_config": {"effort": "low"}}), _ctx(ASTRA))


def test_native_edge_rewrites_only_explicit_out_of_vocabulary_effort():
    from mlpal_assistants_service.services.messages_v2.anthropic_edge import _apply_resolved_effort

    # none on a model that accepts it → thinking disabled, effort dropped
    body = {"output_config": {"effort": "none"}}
    _apply_resolved_effort(body, _ctx(OPUS46, reasoning_effort={"requested": "none", "applied": "none", "clamped": False, "source": "explicit"}))
    assert body == {"thinking": {"type": "disabled"}}
    # same lever on sonnet-5-5 → its own off switch
    body = {"output_config": {"effort": "none"}}
    _apply_resolved_effort(body, _ctx(SONNET55, reasoning_effort={"requested": "none", "applied": "none", "clamped": False, "source": "explicit"}))
    assert body == {"thinking": {"type": "between_tools"}}
    # clamped rung → applied rung, other output_config keys kept
    body = {"output_config": {"effort": "xhigh", "format": "json"}}
    _apply_resolved_effort(body, _ctx(OPUS46, reasoning_effort={"requested": "xhigh", "applied": "high", "clamped": True, "source": "explicit"}))
    assert body["output_config"] == {"effort": "high", "format": "json"}
    # no lever → field dropped instead of a provider 400
    body = {"output_config": {"effort": "high"}}
    _apply_resolved_effort(body, _ctx(NO_LEVER, reasoning_effort={"requested": "high", "applied": None, "clamped": True, "source": "explicit"}))
    assert "output_config" not in body
    # a pure thinking budget is left to Anthropic
    body = {"thinking": {"type": "enabled", "budget_tokens": 20000}}
    _apply_resolved_effort(body, _ctx(OPUS46, reasoning_effort={"requested": "high", "applied": "high", "clamped": False, "source": "thinking"}))
    assert body == {"thinking": {"type": "enabled", "budget_tokens": 20000}}


def test_native_edge_stream_error_payload_keeps_anthropic_shape():
    import json

    from mlpal_assistants_service.services.messages_v2.anthropic_edge import _as_stream_error

    anthropic = b'{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}'
    assert _as_stream_error(anthropic, 400) == anthropic
    wrapped = json.loads(_as_stream_error(b"<html>502</html>", 502))
    assert wrapped["type"] == "error" and "502" in json.dumps(wrapped)



def test_gemini_tool_schema_sanitizer_folds_exclusive_bounds_and_drops_unknown_keywords():
    from google.genai import types

    from mlpal_assistants_service.adapters.google import sanitize_tool_schema

    gmail_like = {
        "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "title": "Args",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer", "exclusiveMinimum": 0, "examples": [1]},
            "limit": {"type": "integer", "exclusiveMaximum": 100, "maximum": 50},
            "ref": {"$ref": "#/$defs/Q"},
        },
        "required": ["id"],
        "$defs": {"Q": {"type": "string", "pattern": "^q"}},
    }
    out = sanitize_tool_schema(gmail_like)
    assert out["properties"]["id"] == {"type": "integer", "minimum": 0}
    assert out["properties"]["limit"] == {"type": "integer", "maximum": 50}   # explicit inclusive bound wins
    assert out["properties"]["ref"] == {"type": "string", "pattern": "^q"}
    assert "additionalProperties" not in out and "$schema" not in out and "title" not in out
    types.FunctionDeclaration(name="f", description="d", parameters=out)   # the SDK accepts it
