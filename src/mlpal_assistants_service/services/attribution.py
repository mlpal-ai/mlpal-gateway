"""Harness attribution carried on W3C `traceparent` / `baggage` headers
(yodex telemetry-v1, 2026-10-06).

Attribution only: tenancy always comes from the authenticated key. Values are
persisted on the usage row's `cc_metadata` JSONB (GIN-indexed, so a session or
run lookup is a containment query) and mirrored as span attributes. A
malformed header never fails a request — bad fields are dropped.

Baggage keys (W3C baggage, values percent-encoded):
  mlpal.session, mlpal.run, mlpal.prompt, mlpal.parent_run  → ids, ≤ 64 chars
  mlpal.hop (name@version), mlpal.workspace                 → ≤ 128 chars
  mlpal.origin                                              → fixed vocabulary
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from typing import Any
from urllib.parse import unquote

from opentelemetry import trace

_ID_MAX = 64
_LABEL_MAX = 128
ORIGINS = frozenset({"interactive", "one-shot", "subagent", "peer", "routine", "managed"})

# baggage key → (stored key, max length)
_BAGGAGE_KEYS: dict[str, tuple[str, int]] = {
    "mlpal.session": ("session_id", _ID_MAX),
    "mlpal.run": ("run_id", _ID_MAX),
    "mlpal.prompt": ("prompt_id", _ID_MAX),
    "mlpal.parent_run": ("parent_run_id", _ID_MAX),
    "mlpal.hop": ("hop", _LABEL_MAX),
    "mlpal.workspace": ("workspace", _LABEL_MAX),
    "mlpal.origin": ("origin", _LABEL_MAX),
}
ATTRIBUTION_KEYS = frozenset(k for k, _ in _BAGGAGE_KEYS.values()) | {"harness_trace_id", "harness_span_id"}
QUERYABLE_KEYS = frozenset({"session_id", "run_id"})

_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
_PRINTABLE = re.compile(r"^[\x21-\x7e]+$")  # printable ASCII, no spaces


def _clean(raw: str, max_len: int) -> str | None:
    value = unquote(raw.strip())
    if not value or len(value) > max_len or not _PRINTABLE.match(value):
        return None
    return value


def parse_baggage(header: str | None) -> dict[str, str]:
    """`baggage: k=v,k2=v2;prop` → allowlisted, decoded, bounded fields.
    Unknown keys, oversize or non-printable values and unknown origins are
    dropped silently; a duplicate key keeps its first occurrence."""
    out: dict[str, str] = {}
    if not header:
        return out
    for member in header.split(","):
        entry = member.split(";", 1)[0]  # strip baggage properties
        if "=" not in entry:
            continue
        key, raw = entry.split("=", 1)
        spec = _BAGGAGE_KEYS.get(key.strip())
        if spec is None or spec[0] in out:
            continue
        stored, max_len = spec
        value = _clean(raw, max_len)
        if value is None or (stored == "origin" and value not in ORIGINS):
            continue
        out[stored] = value
    return out


def parse_traceparent(header: str | None) -> dict[str, str]:
    m = _TRACEPARENT.match((header or "").strip().lower())
    if not m or m.group(1) == "0" * 32 or m.group(2) == "0" * 16:
        return {}
    return {"harness_trace_id": m.group(1), "harness_span_id": m.group(2)}


def harness_attribution(headers: Mapping[str, str]) -> dict[str, str]:
    """Everything the harness told us about this call, validated."""
    return {**parse_traceparent(headers.get("traceparent")), **parse_baggage(headers.get("baggage"))}


_attribution_var: ContextVar[dict[str, str] | None] = ContextVar("harness_attribution", default=None)


def bind_harness_attribution(raw_headers: Iterable[tuple[bytes, bytes]]) -> None:
    """Called by the observability middleware with the ASGI header list, so
    every usage row this request produces (either wire, including detached
    metering tasks, which inherit the context) carries the same fields."""
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in raw_headers}
    _attribution_var.set(harness_attribution(headers))


def current_attribution() -> dict[str, str]:
    """This request's validated attribution; also mirrors it onto the current
    span (idempotent, no-op without tracing)."""
    fields = _attribution_var.get() or {}
    if fields:
        span = trace.get_current_span()
        if span.is_recording():
            for key, value in fields.items():
                span.set_attribute(f"mlpal.{key}", value)
    return fields


def attribution_fields(cc_metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """The persisted attribution slice of a usage row's cc_metadata."""
    if not cc_metadata:
        return {}
    return {k: v for k, v in cc_metadata.items() if k in ATTRIBUTION_KEYS and isinstance(v, str)}
