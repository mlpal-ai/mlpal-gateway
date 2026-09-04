"""cache_control on the OpenAI wire → Anthropic prompt-cache breakpoints."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter, _append_system_text
from mlpal_assistants_service.schemas.chat import ChatCompletionRequest, ChatMessage
from mlpal_assistants_service.services.chat import ChatService

CC = {"type": "ephemeral"}
CC_1H = {"type": "ephemeral", "ttl": "1h"}


@pytest.fixture
def adapter() -> AnthropicAdapter:
    a = AnthropicAdapter.__new__(AnthropicAdapter)
    a._process_file_attachment = MagicMock(return_value={"type": "image", "source": {}})
    return a


def test_system_without_breakpoint_stays_plain_string(adapter):
    system, msgs = adapter._normalize_messages(
        [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
         {"role": "user", "content": "hi"}]
    )
    assert system == "a\n\nb"
    assert msgs == [{"role": "user", "content": "hi"}]


def test_system_breakpoint_becomes_block_list(adapter):
    system, _ = adapter._normalize_messages(
        [{"role": "system", "content": "a"},
         {"role": "system", "content": "b", "cache_control": CC_1H},
         {"role": "user", "content": "hi"}]
    )
    assert system == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": CC_1H},
    ]


def test_plain_text_turn_breakpoint_wraps_in_text_block(adapter):
    _, msgs = adapter._normalize_messages(
        [{"role": "user", "content": "big context", "cache_control": CC},
         {"role": "user", "content": "q"}]
    )
    assert msgs[0] == {"role": "user", "content": [{"type": "text", "text": "big context", "cache_control": CC}]}
    assert msgs[1] == {"role": "user", "content": "q"}


def test_breakpoint_lands_on_last_block_of_multimodal_turn(adapter):
    _, msgs = adapter._normalize_messages(
        [{"role": "user", "content": "look", "files": [object()], "cache_control": CC}]
    )
    blocks = msgs[0]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[-1] == {"type": "image", "source": {}, "cache_control": CC}


def test_breakpoint_on_tool_call_and_tool_result_turns(adapter):
    _, msgs = adapter._normalize_messages([
        {"role": "assistant", "content": None, "cache_control": CC,
         "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "42", "cache_control": CC},
    ])
    assert msgs[0]["content"][-1]["type"] == "tool_use"
    assert msgs[0]["content"][-1]["cache_control"] == CC
    assert msgs[1]["content"][-1]["type"] == "tool_result"
    assert msgs[1]["content"][-1]["cache_control"] == CC


def test_append_system_text_handles_both_shapes():
    assert _append_system_text(None, "\nJSON") == "\nJSON"
    assert _append_system_text("sys", "\nJSON") == "sys\nJSON"
    assert _append_system_text([{"type": "text", "text": "sys", "cache_control": CC}], "\nJSON") == [
        {"type": "text", "text": "sys", "cache_control": CC},
        {"type": "text", "text": "\nJSON"},
    ]


def test_schema_rejects_more_than_four_breakpoints():
    msgs = [{"role": "user", "content": str(i), "cache_control": CC} for i in range(5)]
    with pytest.raises(ValidationError, match="at most 4"):
        ChatCompletionRequest(model="claude-fable-5-1", messages=msgs)
    ChatCompletionRequest(model="claude-fable-5-1", messages=msgs[:4])


def test_schema_rejects_unknown_ttl():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="x", cache_control={"type": "ephemeral", "ttl": "2h"})


def test_convert_messages_passes_cache_control_without_nulls():
    svc = ChatService.__new__(ChatService)
    out = svc._convert_messages([
        ChatMessage(role="system", content="s", cache_control={"type": "ephemeral"}),
        ChatMessage(role="user", content="u"),
    ])
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[1]


def test_openai_compatible_wire_strips_cache_control():
    from mlpal_assistants_service.adapters.openai import OpenAIAdapter

    a = OpenAIAdapter.__new__(OpenAIAdapter)
    params = a._completions_params(
        "m", [{"role": "system", "content": "s", "cache_control": CC}], 0.5, None, None, None,
        None, None, None, None,
    )
    assert params["messages"] == [{"role": "system", "content": "s"}]
