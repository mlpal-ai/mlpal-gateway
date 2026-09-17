"""OpenAI proprietary models served through Bedrock's OpenAI Responses wire.

Contract: a `bedrock` backend for the openai family — SigV4 instead of a
bearer key, model = inference-profile id from the explicit
MLPAL_BEDROCK_OPENAI_MODELS map, only mapped models served (everything else
falls through to first_party). Per-model rules see the bare OpenAI id behind
the profile id. URL-addressed MCP servers cannot ride this wire, so the
adapter declares it and the chat service serves such requests elsewhere.
"""

from __future__ import annotations

import json

import httpx
import pytest

from mlpal_assistants_service.adapters.aws_sigv4 import SigV4HttpxAuth
from mlpal_assistants_service.adapters.factory import AdapterFactory
from mlpal_assistants_service.adapters.openai import OpenAIAdapter
from mlpal_assistants_service.adapters.serving import BedrockOpenAIAdapter
from mlpal_assistants_service.core.config import get_settings
from mlpal_assistants_service.services.runtime_settings import _validate_backends_csv

MAP = {"gpt-6-astra": "global.openai.gpt-6-astra", "gpt-5.6-luna": "global.openai.gpt-5.6-luna"}


@pytest.fixture()
def bedrock_openai(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_backends", "bedrock,first_party")
    monkeypatch.setattr(s, "openai_api_key", "sk-test")
    monkeypatch.setattr(s, "bedrock_openai_models", json.dumps(MAP))
    monkeypatch.setattr(s, "bedrock_mantle_region", "us-east-2")
    monkeypatch.setattr(s, "azure_openai_endpoint", None)
    monkeypatch.setattr(s, "azure_openai_api_key", None)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    f = AdapterFactory()
    f.clear_instances()
    yield s, f
    f.clear_instances()


def test_mapped_models_go_to_bedrock_unmapped_stay_first_party(bedrock_openai):
    _, f = bedrock_openai
    adapter, wire = f.resolve("openai", "gpt-6-astra")
    assert isinstance(adapter, BedrockOpenAIAdapter) and wire == "global.openai.gpt-6-astra"
    assert str(adapter._client.base_url) == "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/"
    other, wire2 = f.resolve("openai", "gpt-4.1")
    assert type(other) is OpenAIAdapter and wire2 == "gpt-4.1"
    image, _ = f.resolve("openai", "gpt-image-2.5-flare")
    assert type(image) is OpenAIAdapter
    assert f.serving_backend_for("openai", "gpt-6-astra") == "bedrock"
    # backend failover has somewhere to go
    alt, _ = f.resolve("openai", "gpt-6-astra", frozenset({"bedrock"}))
    assert type(alt) is OpenAIAdapter


def test_backend_requires_map(bedrock_openai, monkeypatch):
    s, f = bedrock_openai
    monkeypatch.setattr(s, "bedrock_openai_models", None)
    f.clear_instances()
    with pytest.raises(RuntimeError, match="MLPAL_BEDROCK_OPENAI_MODELS"):
        BedrockOpenAIAdapter()
    # unconfigured backend is skipped, never fatal
    assert type(f.resolve("openai", "gpt-6-astra")[0]) is OpenAIAdapter


def test_per_model_rules_see_the_bare_id():
    assert OpenAIAdapter.canonical_model_id("global.openai.gpt-5.6-luna") == "gpt-5.6-luna"
    assert OpenAIAdapter.canonical_model_id("us.openai.gpt-6-astra") == "gpt-6-astra"
    assert OpenAIAdapter.canonical_model_id("gpt-4.1") == "gpt-4.1"
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    # gpt-5.6 family rejects temperature on both first-party and Bedrock
    assert adapter._model_skips_sampling_params("global.openai.gpt-5.6-luna")
    assert not adapter._model_skips_sampling_params("global.openai.gpt-4.1")


def test_mcp_passthrough_declared_unsupported():
    assert BedrockOpenAIAdapter.supports_mcp_passthrough is False
    assert OpenAIAdapter.supports_mcp_passthrough is True


def test_runtime_override_accepts_bedrock_for_openai(bedrock_openai):
    assert _validate_backends_csv("openai")("bedrock, first_party") == "bedrock,first_party"
    with pytest.raises(ValueError, match="unknown backend"):
        _validate_backends_csv("openai")("vertex")


def test_sigv4_auth_signs_the_exact_body(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    auth = SigV4HttpxAuth("us-east-2")
    req = httpx.Request(
        "POST", "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/responses",
        content=b'{"model":"global.openai.gpt-5.6-luna","input":"hi"}',
        headers={"Authorization": "Bearer unused-sigv4", "content-type": "application/json"},
    )
    signed = next(auth.auth_flow(req))
    assert signed.headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIATEST/")
    assert "bedrock/aws4_request" in signed.headers["Authorization"]
    assert "content-type" in signed.headers["Authorization"]  # signed header set
    assert "X-Amz-Date" in signed.headers
    assert signed.content == req.content


@pytest.mark.asyncio
async def test_remote_images_are_inlined_for_bedrock(bedrock_openai, monkeypatch):
    from mlpal_assistants_service.adapters.base import FileAttachment, FileSource, FileType

    adapter = BedrockOpenAIAdapter()
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 16

    def handler(request):
        assert request.url.host == "img.example"
        return httpx.Response(200, content=png, headers={"content-type": "image/png"})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    messages = [
        {"role": "user", "content": "what is this", "files": [
            FileAttachment(type=FileType.IMAGE, source=FileSource.URL, data="https://img.example/a.png")
        ]},
        {"role": "user", "images": [{"url": "https://img.example/b.png"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "and"},
                                     {"type": "input_image", "image_url": "https://img.example/c.png"}]},
    ]
    out = await adapter._inline_remote_files(messages)
    import base64
    b64 = base64.b64encode(png).decode()
    assert out[0]["files"][0].source == FileSource.BASE64 and out[0]["files"][0].data == b64
    assert out[0]["files"][0].mime_type == "image/png"
    assert out[1]["images"][0] == {"base64": b64, "mime_type": "image/png"}
    assert out[2]["content"][1]["image_url"] == f"data:image/png;base64,{b64}"
    assert messages[1]["images"][0] == {"url": "https://img.example/b.png"}  # input untouched
    # nothing remote → same object back, no client built
    plain = [{"role": "user", "content": "hi"}]
    assert await adapter._inline_remote_files(plain) is plain


@pytest.mark.asyncio
async def test_remote_image_fetch_failure_is_a_client_error(bedrock_openai, monkeypatch):
    from mlpal_assistants_service.core.exceptions import ProviderError

    adapter = BedrockOpenAIAdapter()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    with pytest.raises(ProviderError) as ei:
        await adapter._inline_remote_files([{"role": "user", "images": [{"url": "https://img.example/x"}]}])
    assert ei.value.status_code == 400
