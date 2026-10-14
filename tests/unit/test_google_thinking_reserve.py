"""Gemini output-token reserve: cap thinking so a tight max_tokens can't be
fully consumed by reasoning (which returned empty text on flash models)."""

from __future__ import annotations

from types import SimpleNamespace

from google.genai import types

import mlpal_assistants_service.adapters.google as g


def _reserve(monkeypatch, reserve):
    monkeypatch.setattr(g, "get_settings", lambda: SimpleNamespace(google_thinking_output_reserve=reserve))


def test_flash_floors_thinking_at_one_at_tiny_budget(monkeypatch):
    # Default (can_disable=False, min_active=1): never emit budget=0, which the
    # newest flash tiers reject. 1 token of thinking is immaterial vs the reserve.
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", 64, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 1


def test_mandatory_thinking_models_never_emit_zero(monkeypatch):
    # gemini-3.6-flash / gemini-3.5-flash-lite hard-reject thinking_budget=0.
    _reserve(monkeypatch, 256)
    for model in ("gemini-3.6-flash", "gemini-3.5-flash-lite"):
        cfg: dict = {}
        g._apply_thinking_output_reserve(cfg, model, 64, has_tools=False)
        assert cfg["thinking_config"].thinking_budget == 1, model


def test_discontinuous_lite_disables_in_forbidden_gap(monkeypatch):
    # gemini-2.5-flash-lite accepts {0} ∪ {>=512} but rejects 1..511. Both a tiny
    # budget and a mid-range one that lands in the gap must clamp to 0 (disable),
    # never to an API-rejected value.
    _reserve(monkeypatch, 256)
    for max_tokens in (64, 500):  # desired = -192 and 244, both in the gap
        cfg: dict = {}
        g._apply_thinking_output_reserve(cfg, "gemini-2.5-flash-lite", max_tokens, has_tools=False)
        assert cfg["thinking_config"].thinking_budget == 0, max_tokens


def test_discontinuous_lite_passes_through_above_min(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-2.5-flash-lite", 1024, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 768  # 1024-256, >= 512


def test_pro_floors_at_128(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.1-pro-preview", 64, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 128  # pro can't disable thinking


def test_gemini_4_disables_thinking_at_tiny_budget(monkeypatch):
    # Argon accepts budget=0 (verified live 2026-10-14), so a tight max_tokens
    # turns thinking off rather than returning empty text at finish=max_tokens.
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-4-argon", 256, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 0
    cfg = {}
    g._apply_thinking_output_reserve(cfg, "gemini-4-argon", 2048, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 1792


def test_large_budget_reserves_output(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", 2048, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 1792  # 2048-256


def test_skipped_with_tools(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", 64, has_tools=True)
    assert "thinking_config" not in cfg  # tool path owns thinking_config


def test_skipped_for_non_thinking_model(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-1.5-flash", 64, has_tools=False)
    assert "thinking_config" not in cfg


def test_disabled_when_reserve_zero(monkeypatch):
    _reserve(monkeypatch, 0)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", 64, has_tools=False)
    assert "thinking_config" not in cfg


def test_skipped_when_max_tokens_unset(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", None, has_tools=False)
    assert "thinking_config" not in cfg


def test_preserves_include_thoughts(monkeypatch):
    _reserve(monkeypatch, 256)
    cfg: dict = {"thinking_config": types.ThinkingConfig(include_thoughts=True)}
    g._apply_thinking_output_reserve(cfg, "gemini-3.5-flash", 2048, has_tools=False)
    assert cfg["thinking_config"].thinking_budget == 1792
    assert cfg["thinking_config"].include_thoughts is True
