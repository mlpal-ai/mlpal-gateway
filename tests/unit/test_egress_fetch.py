"""Guarded download of caller-supplied URLs (CWE-918, reported 2026-10-04 by
Sithum Shihara / @zaara2004): every hop validated, redirects never followed
blindly, body capped, policy refusals surface as a 400 and are never swallowed
as a dropped attachment."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import httpx
import pytest

from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.seams import egress_guard as eg
from mlpal_assistants_service.seams.egress_guard import (
    EndpointRejected,
    fetch_user_url,
    fetch_user_url_sync,
    validate_endpoint,
    validate_endpoint_sync,
)

# host → addresses the fake resolver returns
HOSTS = {
    "public.example": ["93.184.216.34"],
    "internal.example": ["10.0.1.59"],
    "meta.example": ["169.254.169.254"],
    "mixed.example": ["93.184.216.34", "10.0.0.5"],  # one private address poisons the host
    "mapped.example": ["::ffff:127.0.0.1"],
    "v6public.example": ["2606:4700::1111"],
    "redirector.example": ["93.184.216.34"],
}


@pytest.fixture(autouse=True)
def _production_policy(monkeypatch):
    monkeypatch.setattr(get_settings(), "environment", "production")

    def fake_getaddrinfo(host, port, *a, **k):
        import ipaddress

        try:  # IP literals resolve to themselves, as the real resolver does
            return [(None, None, None, None, (str(ipaddress.ip_address(host)), port))]
        except ValueError:
            pass
        if host not in HOSTS:
            raise socket.gaierror(host)
        return [(None, None, None, None, (ip, port)) for ip in HOSTS[host]]

    monkeypatch.setattr(eg.socket, "getaddrinfo", fake_getaddrinfo)


# -- policy ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://public.example/x",  # not https
        "https://user:pw@public.example/x",  # embedded credentials
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://10.1.2.3/x",
        "https://192.168.1.1/x",
        "https://172.16.0.1/x",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::ffff:127.0.0.1]/x",
        "https://[::ffff:10.0.0.1]/x",
        "https://0.0.0.0/x",
        "https://224.0.0.1/x",
        "https://internal.example/x",  # name → private
        "https://meta.example/x",  # name → link-local
        "https://mixed.example/x",  # one of several addresses is private
        "https://mapped.example/x",
        "https://nxdomain.example/x",
    ],
)
@pytest.mark.asyncio
async def test_rejected_destinations(url):
    with pytest.raises(EndpointRejected):
        await validate_endpoint(url)
    with pytest.raises(EndpointRejected):
        validate_endpoint_sync(url)


@pytest.mark.asyncio
async def test_public_destinations_pass():
    for url in ("https://public.example/img.png", "https://v6public.example/a", "https://93.184.216.34/x"):
        await validate_endpoint(url)
        validate_endpoint_sync(url)


@pytest.mark.asyncio
async def test_development_allows_http_and_loopback_only(monkeypatch):
    monkeypatch.setattr(get_settings(), "environment", "development")
    await validate_endpoint("http://127.0.0.1:8099/local.png")
    with pytest.raises(EndpointRejected):
        await validate_endpoint("http://10.0.0.5/x")  # private stays blocked even in dev


# -- fetch: redirects, body cap, result shape ----------------------------------------
def _transport(routes: dict[str, tuple], hits: list[str]) -> httpx.MockTransport:
    """routes: url → (status, content, headers). A fresh Response per request:
    one httpx.Response cannot serve both a sync and an async client."""

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        status, content, headers = routes[str(request.url)]
        return httpx.Response(status, content=content, headers=headers)

    return httpx.MockTransport(handler)


PNG = b"\x89PNG" + b"\x00" * 40


@pytest.mark.asyncio
async def test_fetch_returns_content_and_lowercase_headers():
    hits: list[str] = []
    t = _transport({"https://public.example/img.png": (200, PNG, {"Content-Type": "image/png"})}, hits)
    got = await fetch_user_url("https://public.example/img.png", transport=t)
    assert got.content == PNG and got.headers["content-type"] == "image/png" and got.url.endswith("/img.png")
    got.raise_for_status()  # no-op: the fetch already checked the status
    got2 = fetch_user_url_sync("https://public.example/img.png", transport=t)
    assert got2.content == PNG and hits == ["https://public.example/img.png"] * 2


@pytest.mark.asyncio
async def test_redirect_to_private_is_refused_before_it_is_followed():
    hits: list[str] = []
    t = _transport({"https://redirector.example/go": (302, b"", {"Location": "https://internal.example/secret"})}, hits)
    with pytest.raises(EndpointRejected):
        await fetch_user_url("https://redirector.example/go", transport=t)
    with pytest.raises(EndpointRejected):
        fetch_user_url_sync("https://redirector.example/go", transport=t)
    assert hits == ["https://redirector.example/go"] * 2  # the private hop was never requested


@pytest.mark.asyncio
async def test_redirect_to_metadata_endpoint_is_refused():
    hits: list[str] = []
    t = _transport({"https://redirector.example/go": (301, b"", {"Location": "https://169.254.169.254/latest/"})}, hits)
    with pytest.raises(EndpointRejected):
        await fetch_user_url("https://redirector.example/go", transport=t)
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_public_redirect_is_followed_and_relative_locations_resolve():
    hits: list[str] = []
    t = _transport({
        "https://redirector.example/go": (302, b"", {"Location": "/final.png"}),
        "https://redirector.example/final.png": (200, PNG, {"content-type": "image/png"}),
    }, hits)
    got = await fetch_user_url("https://redirector.example/go", transport=t)
    assert got.content == PNG and got.url == "https://redirector.example/final.png"


@pytest.mark.asyncio
async def test_redirect_loop_stops_after_max_hops():
    hits: list[str] = []
    t = _transport({"https://redirector.example/loop": (302, b"", {"Location": "https://redirector.example/loop"})}, hits)
    with pytest.raises(EndpointRejected, match="too many redirects"):
        await fetch_user_url("https://redirector.example/loop", transport=t)
    assert len(hits) == eg.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_body_cap_and_upstream_errors():
    big = b"x" * (eg.MAX_FETCH_BYTES + 1)
    t = _transport({"https://public.example/big": (200, big, {}),
                    "https://public.example/missing": (404, b"", {})}, [])
    with pytest.raises(EndpointRejected, match="exceeds"):
        await fetch_user_url("https://public.example/big", transport=t)
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_user_url("https://public.example/missing", transport=t)


# -- surfacing -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rejection_is_a_400_not_a_dropped_attachment():
    from mlpal_assistants_service.main import egress_rejected_handler

    resp = await egress_rejected_handler(MagicMock(), EndpointRejected("endpoint must be an https:// URL"))
    assert resp.status_code == 400
    assert b'"url_rejected"' in resp.body and b"https://" in resp.body


def test_adapters_have_no_unguarded_user_url_fetch_left():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src/mlpal_assistants_service/adapters"
    offenders = []
    for f in root.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"client\.get\(|httpx\.Client\(", line):
                offenders.append(f"{f.name}:{i}")
    assert offenders == [], offenders  # every download goes through the egress guard
