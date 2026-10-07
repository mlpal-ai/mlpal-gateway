"""Harness attribution on W3C traceparent/baggage (yodex telemetry-v1):
allowlisted, bounded, percent-decoded; never fails a request; persisted on
every usage row of the request on either wire; queryable per session/run
scoped to the caller's user."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlpal_assistants_service.services import attribution as attr
from mlpal_assistants_service.services.attribution import (
    attribution_fields,
    bind_harness_attribution,
    current_attribution,
    harness_attribution,
    parse_baggage,
    parse_traceparent,
)

TP = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
BAG = ("mlpal.session=sess-1,mlpal.run=run-1,mlpal.prompt=p-1,mlpal.parent_run=run-0,"
       "mlpal.hop=coder%40v3,mlpal.origin=subagent,mlpal.workspace=mlpal%2Fgateway")


def test_traceparent_shape_is_enforced():
    assert parse_traceparent(TP) == {"harness_trace_id": "0af7651916cd43dd8448eb211c80319c", "harness_span_id": "b7ad6b7169203331"}
    assert parse_traceparent(TP.upper()) == parse_traceparent(TP)  # case-insensitive
    for bad in (None, "", "garbage", "01-" + TP[3:], TP[:-3], "00-" + "0" * 32 + "-b7ad6b7169203331-01",
                "00-0af7651916cd43dd8448eb211c80319c-" + "0" * 16 + "-01"):
        assert parse_traceparent(bad) == {}, bad


def test_baggage_allowlist_decoding_and_bounds():
    got = parse_baggage(BAG)
    assert got == {"session_id": "sess-1", "run_id": "run-1", "prompt_id": "p-1", "parent_run_id": "run-0",
                   "hop": "coder@v3", "origin": "subagent", "workspace": "mlpal/gateway"}
    # unknown keys, properties, oversize ids, non-printable, bad origin, duplicates
    got = parse_baggage("foo=bar,mlpal.run=run-2;prop=1,mlpal.session=" + "x" * 65 + ",mlpal.prompt=a%20b,"
                        "mlpal.origin=robot,mlpal.run=run-3,mlpal.hop=" + "h" * 129)
    assert got == {"run_id": "run-2"}
    assert parse_baggage(None) == {} and parse_baggage("=,,;") == {}


def test_bind_then_current_reads_the_request_context_and_is_empty_without_headers():
    bind_harness_attribution([(b"traceparent", TP.encode()), (b"Baggage", BAG.encode()), (b"x-other", b"1")])
    got = current_attribution()
    assert got["harness_span_id"] == "b7ad6b7169203331" and got["run_id"] == "run-1"
    assert attribution_fields({**got, "cc_session_id": "abc", "stream": True}) == got
    bind_harness_attribution([])
    assert current_attribution() == {} and harness_attribution({}) == {}


def test_span_attributes_are_set_when_a_span_is_recording(monkeypatch):
    span = MagicMock()
    span.is_recording.return_value = True
    monkeypatch.setattr(attr.trace, "get_current_span", lambda: span)
    bind_harness_attribution([(b"traceparent", TP.encode()), (b"baggage", b"mlpal.run=run-9")])
    current_attribution()
    assert {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list} == {
        "mlpal.harness_trace_id": "0af7651916cd43dd8448eb211c80319c", "mlpal.harness_span_id": "b7ad6b7169203331", "mlpal.run_id": "run-9"}


@pytest.mark.asyncio
async def test_usage_row_carries_attribution_on_any_wire_and_explicit_keys_win():
    from mlpal_assistants_service.services.usage import UsageService

    bind_harness_attribution([(b"baggage", b"mlpal.session=sess-7,mlpal.run=run-7")])
    svc = UsageService(session=MagicMock(), redis_client=None)
    svc._write_usage_to_db = AsyncMock()
    await svc.record_usage(user_id=1, api_key_id=2, trace_id="tr_x", model_tag="m", provider="anthropic",
                           operation="chat", input_tokens=1, output_tokens=1, compute_units=Decimal("0.1"),
                           status="error", cc_metadata={"run_id": "explicit", "stream": True})
    record = svc._write_usage_to_db.call_args.args[0]
    assert record["cc_metadata"] == {"session_id": "sess-7", "run_id": "explicit", "stream": True}
    bind_harness_attribution([])


@pytest.mark.asyncio
async def test_usage_query_requires_exactly_one_filter_and_sums_rows():
    from datetime import UTC, datetime

    from fastapi import HTTPException

    from mlpal_assistants_service.api.v1 import usage as usage_api

    row = MagicMock(trace_id="tr_1", model_tag="claude-sonnet-5-5", provider="anthropic", operation="chat",
                    input_tokens=100, output_tokens=10, compute_units=Decimal("0.5"), latency_ms=5, status="success",
                    created_at=datetime.now(UTC), cc_metadata={"run_id": "run-1", "session_id": "s", "cache_read_input_tokens": 60, "stream": True})
    rows = [row, row, row]
    repo = MagicMock()
    repo.get_user_usage_by_attribution = AsyncMock(return_value=rows)
    usage_api.UsageRepository = lambda session: repo  # type: ignore[assignment]
    svc = MagicMock(session=None)
    with pytest.raises(HTTPException):
        await usage_api.get_usage_by_attribution(user_id=1, usage_service=svc, session_id=None, run_id=None, limit=200)
    with pytest.raises(HTTPException):
        await usage_api.get_usage_by_attribution(user_id=1, usage_service=svc, session_id="s", run_id="r", limit=200)
    out = await usage_api.get_usage_by_attribution(user_id=1, usage_service=svc, session_id=None, run_id="run-1", limit=2)
    repo.get_user_usage_by_attribution.assert_awaited_once_with(1, "run_id", "run-1", limit=3)
    assert out.filter == {"run_id": "run-1"} and out.truncated is True and out.total_requests == 2
    assert out.total_compute_units == 1.0 and out.total_input_tokens == 200
    assert out.items[0].attribution == {"run_id": "run-1", "session_id": "s"} and out.items[0].cache_read_input_tokens == 60
