"""Per-key payload-capture policy: resolution matrix, enforcement at both
capture seams (v1 chat + v2 messages), and the hot-path cost guarantee.

The privacy contract under test: an explicit {"mode": "off"} key NEVER
captures — no deployment default, operator toggle, or malformed input may
override it — and the per-request decision is a pure in-memory predicate
(zero I/O, zero task spawn for hard-off keys).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.services.capture import key_allows_capture


# ── resolution matrix ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "policy,key_default,models,expected",
    [
        # inherit: the deployment default decides
        (None, "on", ("gpt-5.2", "gpt-5.2"), True),
        (None, "off", ("gpt-5.2", "gpt-5.2"), False),
        # explicit off: hard, regardless of default
        ({"mode": "off"}, "on", ("gpt-5.2", "gpt-5.2"), False),
        ({"mode": "off", "models": ["gpt-5.2"]}, "on", ("gpt-5.2", "gpt-5.2"), False),
        # explicit on: captures even when the deployment default is off
        ({"mode": "on"}, "off", ("gpt-5.2", "gpt-5.2"), True),
        ({"mode": "on"}, "on", ("gpt-5.2", "gpt-5.2"), True),
        # models filter: exact tags, requested OR resolved may match
        ({"mode": "on", "models": ["gpt-5.2"]}, "on", ("mlpal", "gpt-5.2"), True),
        ({"mode": "on", "models": ["mlpal"]}, "on", ("mlpal", "gpt-5.2"), True),
        ({"mode": "on", "models": ["claude-opus-5"]}, "on", ("mlpal", "gpt-5.2"), False),
        ({"mode": "on", "models": []}, "on", ("gpt-5.2", "gpt-5.2"), True),  # empty = all
        # malformed policies fail CLOSED
        ({"mode": "banana"}, "on", ("gpt-5.2", "gpt-5.2"), False),
        ({}, "on", ("gpt-5.2", "gpt-5.2"), False),
        ("on", "on", ("gpt-5.2", "gpt-5.2"), False),
        (42, "on", ("gpt-5.2", "gpt-5.2"), False),
    ],
)
def test_key_allows_capture_matrix(policy, key_default, models, expected):
    assert key_allows_capture(policy, key_default, *models) is expected


def test_none_model_tags_never_match_filter():
    assert key_allows_capture({"mode": "on", "models": ["x"]}, "on", None, None) is False


# ── enforcement: v1 chat seam ────────────────────────────────────────────────
def _chat_service():
    from mlpal_assistants_service.services.chat import ChatService

    svc = ChatService.__new__(ChatService)
    svc.redis = None
    svc._background_tasks = set()
    return svc


@pytest.mark.asyncio
async def test_chat_hard_off_spawns_nothing(monkeypatch):
    """{"mode": "off"} short-circuits BEFORE the task spawn."""
    svc = _chat_service()
    spawned = []
    monkeypatch.setattr(svc, "_fire_and_forget", lambda coro: spawned.append(coro))

    svc._maybe_capture(
        "t-1", {"model": "gpt-5.2"}, {"content": "hi"},
        capture_policy={"mode": "off"},
        requested_model="gpt-5.2", resolved_model="gpt-5.2",
    )
    assert spawned == []


@pytest.mark.asyncio
async def test_chat_key_filter_blocks_inside_task(monkeypatch):
    """A key scoped to other models spawns the task but never stores."""
    import mlpal_assistants_service.services.chat as chat_mod

    stored = []
    monkeypatch.setattr(chat_mod, "capture_payload", AsyncMock(side_effect=lambda *a, **k: stored.append(a)))
    monkeypatch.setattr(
        chat_mod.capture_state, "config",
        AsyncMock(return_value=SimpleNamespace(enabled=True, max_body_kb=256, key_default="on")),
    )
    svc = _chat_service()
    svc._maybe_capture(
        "t-2", {"model": "gpt-5.2"}, {"content": "hi"},
        capture_policy={"mode": "on", "models": ["claude-opus-5"]},
        requested_model="gpt-5.2", resolved_model="gpt-5.2",
    )
    await asyncio.gather(*svc._background_tasks)
    assert stored == []


@pytest.mark.asyncio
async def test_chat_opt_in_captures_when_default_off(monkeypatch):
    import mlpal_assistants_service.services.chat as chat_mod

    stored = []
    monkeypatch.setattr(chat_mod, "capture_payload", AsyncMock(side_effect=lambda *a, **k: stored.append(a)))
    monkeypatch.setattr(
        chat_mod.capture_state, "config",
        AsyncMock(return_value=SimpleNamespace(enabled=True, max_body_kb=256, key_default="off")),
    )
    svc = _chat_service()
    svc._maybe_capture(
        "t-3", {"model": "gpt-5.2"}, {"content": "hi"},
        capture_policy={"mode": "on"},
        requested_model="gpt-5.2", resolved_model="gpt-5.2",
    )
    await asyncio.gather(*svc._background_tasks)
    assert len(stored) == 1


# ── enforcement: v2 messages seam ────────────────────────────────────────────
def test_v2_pre_spawn_short_circuit():
    from mlpal_assistants_service.services.messages_v2.core import _key_capture_possible

    ctx_off = SimpleNamespace(api_key=SimpleNamespace(capture_policy={"mode": "off"}))
    ctx_on = SimpleNamespace(api_key=SimpleNamespace(capture_policy={"mode": "on"}))
    ctx_inherit = SimpleNamespace(api_key=SimpleNamespace(capture_policy=None))
    ctx_no_attr = SimpleNamespace(api_key=SimpleNamespace())
    assert _key_capture_possible(ctx_off) is False
    assert _key_capture_possible(ctx_on) is True
    assert _key_capture_possible(ctx_inherit) is True
    assert _key_capture_possible(ctx_no_attr) is True  # spawn; task decides


@pytest.mark.asyncio
async def test_v2_capture_respects_key_filter(monkeypatch):
    import mlpal_assistants_service.services.capture as cap_mod
    from mlpal_assistants_service.services.messages_v2.core import _capture_v2

    stored = []
    monkeypatch.setattr(cap_mod, "capture_payload", AsyncMock(side_effect=lambda *a, **k: stored.append(a)))
    monkeypatch.setattr(
        cap_mod.capture_state, "config",
        AsyncMock(return_value=SimpleNamespace(enabled=True, max_body_kb=256, key_default="on")),
    )
    await _capture_v2(
        "t-4", b"{}", b"resp", None,
        capture_policy={"mode": "on", "models": ["other-model"]},
        requested_model="claude-opus-5", resolved_model="claude-opus-5",
    )
    assert stored == []
    await _capture_v2(
        "t-5", b"{}", b"resp", None,
        capture_policy={"mode": "on", "models": ["claude-opus-5"]},
        requested_model="claude-opus-5", resolved_model="claude-opus-5",
    )
    assert len(stored) == 1


# ── hot-path cost ────────────────────────────────────────────────────────────
def test_decision_is_pure_and_fast():
    """The per-request predicate must stay sub-microsecond in-memory work.
    Bound is deliberately loose (CI machines vary); the worklog records the
    measured figure. 200k decisions under 2s = <10µs each, ~100x headroom
    over the real cost measured at ~0.2µs."""
    policy = {"mode": "on", "models": ["gpt-5.2", "claude-opus-5", "mlpal"]}
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        key_allows_capture(policy, "on", "mlpal", "gpt-5.2")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"{n} decisions took {elapsed:.2f}s"
