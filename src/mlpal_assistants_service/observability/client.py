"""Per-request client fingerprint (viewer IP + user-agent hash).

Set once per request by ObservabilityMiddleware from the ASGI scope and read
by whatever records durable rows during that request (usage logs, key
creation). Context variables, not request objects, so the service layer's
signatures stay untouched and the value follows the request task through
streaming generators.

Two facts these fields are NOT: a security boundary (anyone hitting the ALB
directly can forge X-Forwarded-For) and content (never the body). They are
signals for the abuse detector — the joins that were done by hand against
CloudFront logs during the 2026-09-28 signup-farm incident.
"""

from __future__ import annotations

import hashlib
import ipaddress
from contextvars import ContextVar

client_ip_var: ContextVar[str | None] = ContextVar("client_ip", default=None)
client_ua_hash_var: ContextVar[str | None] = ContextVar("client_ua_hash", default=None)

UA_HASH_LEN = 16


def client_ip_from_headers(headers: list[tuple[bytes, bytes]], peer: str | None) -> str | None:
    """Viewer IP: first X-Forwarded-For hop (CloudFront prepends the viewer,
    the ALB appends the edge), else the transport peer."""
    for name, value in headers:
        if name == b"x-forwarded-for":
            first = value.decode("latin-1").split(",", 1)[0].strip()
            try:
                return str(ipaddress.ip_address(first))
            except ValueError:
                break
    return peer


def ua_hash_from_headers(headers: list[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        if name == b"user-agent":
            return hashlib.sha256(value).hexdigest()[:UA_HASH_LEN] if value else None
    return None


def bind_client_context(scope: dict) -> None:
    peer = scope.get("client")
    headers = scope.get("headers") or []
    client_ip_var.set(client_ip_from_headers(headers, peer[0] if peer else None))
    client_ua_hash_var.set(ua_hash_from_headers(headers))


def get_client_ip() -> str | None:
    return client_ip_var.get()


def get_client_ua_hash() -> str | None:
    return client_ua_hash_var.get()
