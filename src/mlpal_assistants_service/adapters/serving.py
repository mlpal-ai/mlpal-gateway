"""Cloud serving backends: the same model families served through Azure,
Vertex AI, and Bedrock instead of (or in priority order with) the provider's
first-party API.

Design (worklog 2026-08-14-multicloud-backends):
- These are BACKENDS for existing families, not new catalog providers.
  Catalog rows are unchanged; `MLPAL_<FAMILY>_BACKENDS` picks who serves them.
- Each backend is a thin subclass of its family adapter: constructor swap
  (host/auth) + `serves()` / `backend_model_id()` data. All request/response
  logic is inherited — zero duplication, so family fixes apply everywhere.
- Model maps for Claude-on-cloud are EXPLICIT config (both clouds gate models
  behind per-account enablement; guessing would turn "not enabled" into
  confusing provider errors). `scripts/probe_backends.py` generates the map.
"""

from __future__ import annotations

import json
import logging

import httpx

from mlpal_assistants_service.adapters.anthropic import AnthropicAdapter
from mlpal_assistants_service.adapters.google import GoogleAdapter
from mlpal_assistants_service.adapters.openai import OpenAIAdapter
from mlpal_assistants_service.core.config import get_settings

logger = logging.getLogger(__name__)


def _parse_map(raw: str | None, setting_name: str) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{setting_name} is not valid JSON: {e}") from e
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(f"{setting_name} must be a JSON object of string→string")
    return parsed


class AzureOpenAIAdapter(OpenAIAdapter):
    """OpenAI family via Azure's `/openai/v1/` surface.

    The v1 surface is OpenAI-wire-compatible with the standard SDK; the only
    deltas are the base URL, the key, and that `model` means DEPLOYMENT name.
    Convention: name deployments after model IDs and no map is needed;
    MLPAL_AZURE_DEPLOYMENTS overrides per-model and makes `serves()` exact.
    """

    backend_name = "azure"

    def __init__(self) -> None:
        settings = get_settings()
        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise RuntimeError(
                "Azure backend requires MLPAL_AZURE_OPENAI_ENDPOINT and "
                "MLPAL_AZURE_OPENAI_API_KEY"
            )
        self._deployments = _parse_map(
            settings.azure_openai_deployments, "MLPAL_AZURE_DEPLOYMENTS"
        )
        super().__init__(
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_endpoint.rstrip("/") + "/openai/v1/",
        )

    def serves(self, provider_model_id: str) -> bool:
        # No map → identity convention: claim the family and let Azure's
        # DeploymentNotFound surface for undeployed models. With a map,
        # serves() is exact and the console can display truth.
        return provider_model_id in self._deployments if self._deployments else True

    def backend_model_id(self, provider_model_id: str) -> str:
        return self._deployments.get(provider_model_id, provider_model_id)


class BedrockOpenAIAdapter(OpenAIAdapter):
    """OpenAI proprietary models (gpt-6-astra, gpt-5.6 sol/terra/luna) via
    Bedrock's OpenAI Responses wire: `bedrock-runtime.<region>/openai/v1`,
    SigV4 instead of a bearer key, model = inference-profile id from the
    explicit MLPAL_BEDROCK_OPENAI_MODELS map. Verified 2026-09-17 against
    the adapter's parameter shapes: reasoning effort, tools, structured
    output, streaming, automatic prompt caching with OpenAI-identical
    accounting, store=false. Not on this wire: URL-addressed MCP servers
    (connector ARNs only) and `https://` image URLs (data:/s3:// only)."""

    backend_name = "bedrock"
    supports_mcp_passthrough = False

    def __init__(self) -> None:
        from mlpal_assistants_service.adapters.aws_sigv4 import SigV4HttpxAuth

        settings = get_settings()
        self._model_map = _parse_map(
            settings.bedrock_openai_models, "MLPAL_BEDROCK_OPENAI_MODELS"
        )
        if not self._model_map:
            raise RuntimeError(
                "Bedrock OpenAI backend requires MLPAL_BEDROCK_OPENAI_MODELS "
                "(run scripts/probe_backends.py openai to generate it)"
            )
        region = settings.bedrock_mantle_region
        super().__init__(
            api_key="unused-sigv4",
            base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/",
            http_client=httpx.AsyncClient(
                auth=SigV4HttpxAuth(region),
                limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
                timeout=httpx.Timeout(120.0, connect=10.0),
            ),
        )

    def serves(self, provider_model_id: str) -> bool:
        return provider_model_id in self._model_map

    def backend_model_id(self, provider_model_id: str) -> str:
        return self._model_map[provider_model_id]

    # Bedrock's OpenAI wire rejects `https://` image/file URLs (data: or
    # s3:// only). First-party OpenAI fetches them itself; here we do, once,
    # bounded, and hand the model the same bytes inline. Same request shape
    # as the parent — only the attachment source changes.
    _REMOTE_FETCH_TIMEOUT = 15.0
    _REMOTE_FETCH_MAX_BYTES = 20 * 1024 * 1024

    async def chat(self, model, messages, *args, **kwargs):
        messages = await self._inline_remote_files(messages)
        return await super().chat(model, messages, *args, **kwargs)

    async def chat_stream(self, model, messages, *args, **kwargs):
        messages = await self._inline_remote_files(messages)
        async for chunk in super().chat_stream(model, messages, *args, **kwargs):
            yield chunk

    async def _inline_remote_files(self, messages: list[dict]) -> list[dict]:
        import base64
        import copy

        from mlpal_assistants_service.adapters.base import FileAttachment, FileSource
        from mlpal_assistants_service.core.exceptions import ProviderError

        def _is_http(v: object) -> bool:
            return isinstance(v, str) and v.startswith(("http://", "https://"))

        async def _fetch(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
            try:
                async with client.stream("GET", url) as r:
                    r.raise_for_status()
                    buf = bytearray()
                    async for part in r.aiter_bytes():
                        buf += part
                        if len(buf) > self._REMOTE_FETCH_MAX_BYTES:
                            raise ProviderError(
                                f"remote file too large for inline delivery (> "
                                f"{self._REMOTE_FETCH_MAX_BYTES} bytes): {url}",
                                provider="openai", status_code=400,
                            )
                    mime = (r.headers.get("content-type") or "application/octet-stream").split(";")[0]
            except httpx.HTTPError as e:
                raise ProviderError(
                    f"could not fetch remote file {url}: {e}", provider="openai", status_code=400
                ) from e
            return base64.b64encode(bytes(buf)).decode(), mime

        needs = any(
            (isinstance(f, FileAttachment) and f.source == FileSource.URL)
            or (isinstance(f, dict) and _is_http(f.get("url")))
            for m in messages
            for key in ("files", "images", "documents")
            for f in (m.get(key) or [])
        ) or any(
            isinstance(part, dict) and (
                _is_http(part.get("image_url"))
                or _is_http((part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None)
            )
            for m in messages
            if isinstance(m.get("content"), list)
            for part in m["content"]
        )
        if not needs:
            return messages
        out = copy.deepcopy(messages)
        async with httpx.AsyncClient(
            timeout=self._REMOTE_FETCH_TIMEOUT, follow_redirects=True
        ) as client:
            for m in out:
                for key in ("files", "images", "documents"):
                    items = m.get(key) or []
                    for i, f in enumerate(items):
                        if isinstance(f, FileAttachment) and f.source == FileSource.URL:
                            data, mime = await _fetch(client, f.data)
                            items[i] = FileAttachment(
                                type=f.type, source=FileSource.BASE64, data=data,
                                mime_type=f.mime_type or mime, filename=f.filename,
                            )
                        elif isinstance(f, dict) and _is_http(f.get("url")):
                            data, mime = await _fetch(client, f["url"])
                            f.pop("url")
                            f["base64"] = data
                            f.setdefault("mime_type", mime)
                if isinstance(m.get("content"), list):
                    for part in m["content"]:
                        if not isinstance(part, dict):
                            continue
                        ref = part.get("image_url")
                        url = ref if _is_http(ref) else (ref or {}).get("url") if isinstance(ref, dict) else None
                        if _is_http(url):
                            data, mime = await _fetch(client, url)
                            inline = f"data:{mime};base64,{data}"
                            if isinstance(ref, dict):
                                ref["url"] = inline
                            else:
                                part["image_url"] = inline
        return out


class VertexGoogleAdapter(GoogleAdapter):
    """Gemini via Vertex AI. Same google-genai SDK, same model IDs — only the
    client constructor differs (ADC auth via GOOGLE_APPLICATION_CREDENTIALS)."""

    backend_name = "vertex"

    def __init__(self) -> None:
        from google import genai

        settings = get_settings()
        if not settings.vertex_project:
            raise RuntimeError("Vertex backend requires MLPAL_VERTEX_PROJECT")
        self._api_key = None
        self._client = genai.Client(
            vertexai=True,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )


class BedrockAnthropicAdapter(AnthropicAdapter):
    """Claude via AWS Bedrock. AsyncAnthropicBedrock is a drop-in for
    AsyncAnthropic (same `.messages` surface, SigV4 auth); model IDs come
    from the explicit MLPAL_BEDROCK_ANTHROPIC_MODELS map."""

    backend_name = "bedrock"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropicBedrock

        settings = get_settings()
        self._model_map = _parse_map(
            settings.bedrock_anthropic_models, "MLPAL_BEDROCK_ANTHROPIC_MODELS"
        )
        if not self._model_map:
            raise RuntimeError(
                "Bedrock Claude backend requires MLPAL_BEDROCK_ANTHROPIC_MODELS "
                "(run scripts/probe_backends.py to generate it)"
            )
        super().__init__(
            api_key="unused-sigv4",
            client=AsyncAnthropicBedrock(
                aws_region=settings.bedrock_mantle_region,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ),
            ),
        )

    def serves(self, provider_model_id: str) -> bool:
        return provider_model_id in self._model_map

    def backend_model_id(self, provider_model_id: str) -> str:
        return self._model_map[provider_model_id]


class VertexAnthropicAdapter(AnthropicAdapter):
    """Claude via Vertex AI (AnthropicVertex, ADC auth). Vertex requires
    per-model Model Garden enablement, so the map is explicit config."""

    backend_name = "vertex"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropicVertex

        settings = get_settings()
        if not settings.vertex_project:
            raise RuntimeError("Vertex backend requires MLPAL_VERTEX_PROJECT")
        self._model_map = _parse_map(
            settings.vertex_anthropic_models, "MLPAL_VERTEX_ANTHROPIC_MODELS"
        )
        if not self._model_map:
            raise RuntimeError(
                "Vertex Claude backend requires MLPAL_VERTEX_ANTHROPIC_MODELS "
                "(models need Model Garden enablement; run scripts/probe_backends.py)"
            )
        super().__init__(
            api_key="unused-adc",
            client=AsyncAnthropicVertex(
                project_id=settings.vertex_project,
                region=settings.vertex_location,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ),
            ),
        )

    def serves(self, provider_model_id: str) -> bool:
        return provider_model_id in self._model_map

    def backend_model_id(self, provider_model_id: str) -> str:
        return self._model_map[provider_model_id]


class AzureAnthropicAdapter(AnthropicAdapter):
    """Claude via Microsoft Foundry (GA). The same AIServices resource that
    serves /openai/v1 exposes the NATIVE Anthropic wire at /anthropic, so the
    standard Anthropic SDK works with a base_url + key swap. `model` means
    DEPLOYMENT name — identity convention (deployment named after the model
    ID) or MLPAL_AZURE_ANTHROPIC_DEPLOYMENTS for exact/non-identity maps."""

    backend_name = "azure"

    def __init__(self) -> None:
        from anthropic import AsyncAnthropic

        settings = get_settings()
        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise RuntimeError(
                "Azure backend requires MLPAL_AZURE_OPENAI_ENDPOINT and "
                "MLPAL_AZURE_OPENAI_API_KEY (the AIServices resource serves "
                "both OpenAI and Claude)"
            )
        self._deployments = _parse_map(
            settings.azure_anthropic_deployments, "MLPAL_AZURE_ANTHROPIC_DEPLOYMENTS"
        )
        super().__init__(
            api_key=settings.azure_openai_api_key,
            client=AsyncAnthropic(
                api_key=settings.azure_openai_api_key,
                base_url=settings.azure_openai_endpoint.rstrip("/") + "/anthropic",
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ),
            ),
        )

    def serves(self, provider_model_id: str) -> bool:
        return provider_model_id in self._deployments if self._deployments else True

    def backend_model_id(self, provider_model_id: str) -> str:
        return self._deployments.get(provider_model_id, provider_model_id)
