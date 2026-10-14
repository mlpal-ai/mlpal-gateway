"""Google adapter: every tool_choice shape the chat wire accepts becomes a valid
Gemini tool_config. The OpenAI-style forced dict used to pass through verbatim and
fail GenerateContentConfig validation (surfacing as a 503 for every Gemini model)."""

from __future__ import annotations

import pytest

from mlpal_assistants_service.adapters.google import GoogleAdapter


@pytest.fixture
def adapter():
    return GoogleAdapter(api_key="test")


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"function_calling_config": {"mode": "AUTO"}}),
        ("none", {"function_calling_config": {"mode": "NONE"}}),
        ("required", {"function_calling_config": {"mode": "ANY"}}),
        ("get_weather", {"function_calling_config": {"mode": "ANY", "allowed_function_names": ["get_weather"]}}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"function_calling_config": {"mode": "ANY", "allowed_function_names": ["get_weather"]}},
        ),
    ],
)
def test_tool_choice_shapes_convert_to_tool_config(adapter, choice, expected):
    assert adapter._convert_tool_choice(choice) == expected


def test_native_tool_config_passes_through(adapter):
    native = {"function_calling_config": {"mode": "ANY", "allowed_function_names": ["a", "b"]}}
    assert adapter._convert_tool_choice(native) == native
