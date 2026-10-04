"""Client fingerprint: header parsing, middleware binding, and the values
landing on usage records and minted keys."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlpal_assistants_service.observability.client import (
    bind_client_context,
    client_ip_from_headers,
    client_ip_var,
    client_ua_hash_var,
    get_client_ip,
    get_client_ua_hash,
    ua_hash_from_headers,
)
from mlpal_assistants_service.observability.middleware import ObservabilityMiddleware


def test_client_ip_prefers_first_forwarded_hop() -> None:
    headers = [(b"x-forwarded-for", b"47.242.138.158, 130.176.1.1")]
    assert client_ip_from_headers(headers, "10.0.0.5") == "47.242.138.158"


def test_client_ip_falls_back_to_peer_on_garbage_header() -> None:
    assert client_ip_from_headers([(b"x-forwarded-for", b"not-an-ip")], "10.0.0.5") == "10.0.0.5"
    assert client_ip_from_headers([], "10.0.0.5") == "10.0.0.5"
    assert client_ip_from_headers([], None) is None


def test_client_ip_accepts_ipv6() -> None:
    assert client_ip_from_headers([(b"x-forwarded-for", b"2407:d840:50:d::9125")], None) == "2407:d840:50:d::9125"


def test_ua_hash_is_stable_short_and_absent_without_header() -> None:
    h = ua_hash_from_headers([(b"user-agent", b"mlpal-sillytavern-bridge/2.2.0")])
    assert h == ua_hash_from_headers([(b"user-agent", b"mlpal-sillytavern-bridge/2.2.0")])
    assert len(h) == 16
    assert ua_hash_from_headers([]) is None
    assert ua_hash_from_headers([(b"user-agent", b"")]) is None


@pytest.mark.asyncio
async def test_middleware_binds_context_for_downstream_app() -> None:
    seen: dict = {}

    async def app(scope, receive, send):
        seen["ip"] = get_client_ip()
        seen["ua"] = get_client_ua_hash()

    scope = {
        "type": "http", "method": "GET", "path": "/v1/models", "client": ("10.0.0.5", 1234),
        "headers": [(b"x-forwarded-for", b"38.83.41.46, 130.1.1.1"), (b"user-agent", b"Mozilla/5.0")],
    }
    with patch("mlpal_assistants_service.observability.middleware.get_metrics"):
        await ObservabilityMiddleware(app)(scope, AsyncMock(), AsyncMock())
    assert seen == {"ip": "38.83.41.46", "ua": ua_hash_from_headers(scope["headers"])}


@pytest.mark.asyncio
async def test_usage_record_carries_client_signals() -> None:
    from mlpal_assistants_service.services.usage import UsageService

    bind_client_context({"client": None, "headers": [(b"x-forwarded-for", b"1.2.3.4"), (b"user-agent", b"ua")]})
    svc = UsageService(MagicMock(), None)
    svc._sqs_client = None
    svc._write_usage_to_db = AsyncMock()
    svc._accrue_platform_fee = AsyncMock()
    await svc.record_usage(
        user_id="7", api_key_id="1", trace_id="t", model_tag="m", provider="p", operation="chat",
        input_tokens=1, output_tokens=1, compute_units=0,
    )
    record = svc._write_usage_to_db.await_args.args[0]
    assert record["client_ip"] == "1.2.3.4"
    assert record["client_ua_hash"] == ua_hash_from_headers([(b"user-agent", b"ua")])
    client_ip_var.set(None)
    client_ua_hash_var.set(None)


@pytest.mark.asyncio
async def test_minted_key_records_origin() -> None:
    from mlpal_assistants_service.schemas.api_key import APIKeyCreate
    from mlpal_assistants_service.services.api_key import APIKeyService

    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    bind_client_context({"client": None, "headers": [(b"x-forwarded-for", b"38.83.41.46"), (b"user-agent", b"Mozilla")]})
    key, _secret = await APIKeyService(session, None).create_key("300", APIKeyCreate(name="1"))
    assert key.created_ip == "38.83.41.46"
    assert key.created_ua_hash == ua_hash_from_headers([(b"user-agent", b"Mozilla")])
    client_ip_var.set(None)
    client_ua_hash_var.set(None)
