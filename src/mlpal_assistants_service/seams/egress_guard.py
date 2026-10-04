"""Egress guard: the one place a user-supplied URL may be turned into a
request that leaves this process.

Two kinds of attacker-controlled URLs reach the gateway:

* BYOM endpoints (connections): validated at registration and at adapter
  construction via ``validate_endpoint``.
* URLs the gateway itself downloads on the caller's behalf — chat attachments
  (images, documents, audio, video), image-generation reference images,
  transcription audio. Until 2026-10-04 these were plain ``httpx.get`` calls
  in every adapter with redirects followed blindly: an authenticated caller
  could make the pod fetch loopback / RFC1918 / link-local destinations
  (CWE-918; reported privately by Sithum Shihara, @zaara2004). They now go
  through ``fetch_user_url`` / ``fetch_user_url_sync`` (or the context-manager
  forms ``guarded_fetch`` / ``guarded_fetch_sync``), which validate EVERY hop
  (initial URL and each redirect), cap the body size, and never follow a
  redirect without re-checking it.

Policy (production): https only, no embedded credentials, every address the
host resolves to must be publicly routable (no private, loopback, link-local
— includes 169.254.169.254 —, multicast, reserved or unspecified), at most
``MAX_REDIRECTS`` redirects, at most ``MAX_FETCH_BYTES`` bytes.

Development deployments (MLPAL_ENVIRONMENT=development) may use http:// and
loopback addresses so local rigs can test against a local vLLM/Ollama or a
local file server.

Residual risk: a DNS rebind between the hop's validation and its connection is
not caught (would need a pinning transport); tracked in
planning/designs/connections-byom.md as phase-2 hardening.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from mlpal_assistants_service.core.config import get_settings

MAX_REDIRECTS = 3
MAX_FETCH_BYTES = 25 * 1024 * 1024


class EndpointRejected(ValueError):
    """The URL failed the egress policy. Message is user-safe."""


def _is_blocked(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped  # ::ffff:127.0.0.1 is still loopback
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local  # includes 169.254.169.254 (cloud metadata)
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _dev() -> bool:
    return get_settings().environment == "development"


def _parse(url: str) -> tuple[str, int]:
    """Scheme / credential checks; returns (host, port)."""
    dev = _dev()
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (dev and parsed.scheme == "http"):
        raise EndpointRejected("endpoint must be an https:// URL")
    host = parsed.hostname
    if not host:
        raise EndpointRejected("endpoint URL has no host")
    if parsed.username or parsed.password:
        raise EndpointRejected("endpoint URL must not embed credentials")
    return host, parsed.port or (443 if parsed.scheme == "https" else 80)


def _check_resolved_sync(host: str, port: int) -> None:
    dev = _dev()
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise EndpointRejected(f"endpoint host does not resolve: {host}") from None
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if _is_blocked(addr) and not (dev and addr.is_loopback):
            raise EndpointRejected(
                "endpoint resolves to a private or reserved address — only "
                "publicly routable endpoints are allowed"
            )


def validate_endpoint_sync(url: str) -> None:
    """Synchronous twin of validate_endpoint (for the sync adapter paths)."""
    host, port = _parse(url)
    _check_resolved_sync(host, port)


async def validate_endpoint(url: str) -> None:
    """Raise EndpointRejected unless the URL is safe to connect to. DNS runs in
    the default executor."""
    host, port = _parse(url)
    await asyncio.get_running_loop().run_in_executor(None, _check_resolved_sync, host, port)


@dataclass
class FetchedResource:
    """What a guarded download yields. Shaped like the slice of httpx.Response
    the adapters used (``content``, ``headers``, ``raise_for_status``) so the
    call sites read the same as before."""

    content: bytes
    headers: dict[str, str] = field(default_factory=dict)  # lower-case keys
    url: str = ""  # final URL after redirects

    def raise_for_status(self) -> None:  # already raised inside the fetch
        return None


def _next_hop(current: str, response: httpx.Response) -> str:
    location = response.headers.get("location")
    if not location:
        raise EndpointRejected("redirect without a Location header")
    return urljoin(current, location)


def _too_large() -> EndpointRejected:
    return EndpointRejected(f"remote resource exceeds {MAX_FETCH_BYTES // (1024 * 1024)} MiB")


async def fetch_user_url(
    url: str, *, timeout: float = 30.0, transport: Any | None = None
) -> FetchedResource:
    """Download a caller-supplied URL with every hop validated and the body
    capped. Raises EndpointRejected on policy, httpx.HTTPStatusError on 4xx/5xx."""
    current = url
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            await validate_endpoint(current)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    current = _next_hop(current, response)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_FETCH_BYTES:
                        raise _too_large()
                    chunks.append(chunk)
                return FetchedResource(
                    b"".join(chunks), {k.lower(): v for k, v in response.headers.items()}, current
                )
    raise EndpointRejected("too many redirects")


def fetch_user_url_sync(
    url: str, *, timeout: float = 30.0, transport: Any | None = None
) -> FetchedResource:
    """Synchronous twin of fetch_user_url for the adapters' sync builders."""
    current = url
    with httpx.Client(timeout=timeout, follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            validate_endpoint_sync(current)
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    current = _next_hop(current, response)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_FETCH_BYTES:
                        raise _too_large()
                    chunks.append(chunk)
                return FetchedResource(
                    b"".join(chunks), {k.lower(): v for k, v in response.headers.items()}, current
                )
    raise EndpointRejected("too many redirects")


@asynccontextmanager
async def guarded_fetch(url: str, *, timeout: float = 30.0) -> AsyncIterator[FetchedResource]:
    """``async with guarded_fetch(url) as response:`` — drop-in for the
    ``async with httpx.AsyncClient() as client: response = await client.get(url)``
    shape the adapters used."""
    yield await fetch_user_url(url, timeout=timeout)


@contextmanager
def guarded_fetch_sync(url: str, *, timeout: float = 30.0) -> Iterator[FetchedResource]:
    """``with guarded_fetch_sync(url) as response:`` — sync drop-in."""
    yield fetch_user_url_sync(url, timeout=timeout)
