"""Anthropic backend selection + request building for the v2 native path.

Backend follows `MLPAL_ANTHROPIC_BACKENDS` (the same priority list the
adapter layer uses) — the first CONFIGURED entry serves the native
/v1/messages wire. `first_party` is api.anthropic.com with x-api-key auth;
`bedrock` is the bedrock-mantle Anthropic-format endpoint with SigV4 auth
and the mantle body/beta quirk-filters (reused from services/bedrock_mantle,
which has served the managed CDE passthrough since 2026-05).

Interface: `url` + `prepare(body_bytes, client_headers) -> (content, headers)`.
prepare owns any body adaptation because SigV4 signs the exact bytes.
"""

from __future__ import annotations

from collections.abc import Mapping

from mlpal_assistants_service.core.config import Settings


class AnthropicFirstPartyBackend:
    """Builds the URL + headers for a first-party Anthropic Messages call."""

    name = "first_party"

    def __init__(self, settings: Settings) -> None:
        self.url = settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        self._api_key = settings.anthropic_api_key
        self._default_version = settings.anthropic_api_version

    def serves(self, provider_model_id: str) -> bool:
        return True

    def prepare(
        self, body: bytes, client_headers: Mapping[str, str]
    ) -> tuple[bytes, dict[str, str]]:
        h = {
            "x-api-key": self._api_key,
            "anthropic-version": client_headers.get("anthropic-version")
            or self._default_version,
            "content-type": "application/json",
        }
        # Forward anthropic-beta unfiltered (locked decision for v2-A).
        beta = client_headers.get("anthropic-beta")
        if beta:
            h["anthropic-beta"] = beta
        return body, h


class AnthropicBedrockBackend:
    """Claude native wire via bedrock-mantle: adapt body (anthropic_version,
    model prefix, quirk-filters), filter betas mantle rejects, SigV4-sign."""

    name = "bedrock"

    def __init__(self, settings: Settings) -> None:
        import json as _json

        from mlpal_assistants_service.adapters.serving import _parse_map
        from mlpal_assistants_service.services.bedrock_mantle import BedrockMantleClient

        self._signer = BedrockMantleClient(
            region=settings.bedrock_mantle_region,
            endpoint=settings.bedrock_native_endpoint,
            model_map=_parse_map(
                settings.bedrock_anthropic_models, "MLPAL_BEDROCK_ANTHROPIC_MODELS"
            ),
        )
        self.url = self._signer.url
        # Mantle serves a SUBSET of bedrock-runtime (see config). Empty/unset
        # list → serve nothing natively; core falls back to the adapter path,
        # which is gated on the live-verified bedrock_anthropic_models map.
        self._models: frozenset[str] = frozenset(
            _json.loads(settings.bedrock_mantle_models or "[]")
        )

    def serves(self, provider_model_id: str) -> bool:
        return provider_model_id in self._models

    def prepare(
        self, body: bytes, client_headers: Mapping[str, str]
    ) -> tuple[bytes, dict[str, str]]:
        from mlpal_assistants_service.services.bedrock_mantle import (
            filter_anthropic_beta_header,
        )

        adapted, _, _removed = self._signer.adapt_body(body)
        beta, _dropped = filter_anthropic_beta_header(client_headers.get("anthropic-beta"))
        extra = {"anthropic-beta": beta} if beta else None
        return adapted, self._signer.sign(adapted, extra)


# Backend cache keyed by the config that shapes it — construction is per
# process, not per request (the bedrock backend builds a boto3 Session).
_backends: dict[tuple, object] = {}


class AnthropicAzureBackend:
    """Claude native wire via Microsoft Foundry: the AIServices resource
    exposes /anthropic/v1/messages byte-compatibly. `model` = DEPLOYMENT
    name — identity convention, or the MLPAL_AZURE_ANTHROPIC_DEPLOYMENTS map."""

    name = "azure"

    def __init__(self, settings: Settings) -> None:
        import json as _json

        self.url = (
            settings.azure_openai_endpoint.rstrip("/") + "/anthropic/v1/messages"
        )
        self._api_key = settings.azure_openai_api_key
        self._default_version = settings.anthropic_api_version
        self._deployments: dict[str, str] = _json.loads(
            settings.azure_anthropic_deployments or "{}"
        )

    def serves(self, provider_model_id: str) -> bool:
        # No map → identity convention (deployment named after the model ID);
        # Foundry 404s undeployed models, same contract as the OpenAI side.
        return provider_model_id in self._deployments if self._deployments else True

    def prepare(
        self, body: bytes, client_headers: Mapping[str, str]
    ) -> tuple[bytes, dict[str, str]]:
        import json as _json

        if self._deployments:
            obj = _json.loads(body)
            mapped = self._deployments.get(obj.get("model"))
            if mapped:
                obj["model"] = mapped
                body = _json.dumps(obj).encode()
        h = {
            "x-api-key": self._api_key,
            "anthropic-version": client_headers.get("anthropic-version")
            or self._default_version,
            "content-type": "application/json",
        }
        beta = client_headers.get("anthropic-beta")
        if beta:
            h["anthropic-beta"] = beta
        return body, h


NativeBackend = AnthropicFirstPartyBackend | AnthropicBedrockBackend | AnthropicAzureBackend
_backend_lists: dict[tuple, list] = {}


def effective_anthropic_backends(settings: Settings) -> str:
    """The priority list in force: the console/DB runtime override when set
    (PUT /admin/v1/settings/anthropic_backends), else the env value — the
    same precedence the adapter factory applies, so one flip moves BOTH
    wires (native passthrough here, adapter path there)."""
    from mlpal_assistants_service.services import runtime_settings

    return runtime_settings.get("anthropic_backends") or settings.anthropic_backends


def native_backends(settings: Settings) -> list[NativeBackend]:
    """Every configured native-path backend in MLPAL_ANTHROPIC_BACKENDS
    priority order. A native backend only takes a model it SERVES (bedrock:
    its mantle allowlist; first_party: everything), so callers walk this list
    and fall through — e.g. `bedrock,first_party` keeps first-party's
    byte-faithful passthrough for models the mantle endpoint lacks instead of
    dropping them onto the lossy translating edge."""
    priority = effective_anthropic_backends(settings)
    key = (
        priority,
        settings.anthropic_api_key,
        settings.anthropic_base_url,
        settings.bedrock_mantle_region,
        settings.bedrock_mantle_models,
        settings.bedrock_anthropic_models,
        settings.bedrock_native_endpoint,
        settings.azure_openai_endpoint,
        settings.azure_anthropic_deployments,
    )
    hit = _backend_lists.get(key)
    if hit is not None:
        return hit
    out: list[NativeBackend] = []
    for name in (n.strip() for n in priority.split(",")):
        if name == "first_party" and settings.anthropic_api_key:
            out.append(AnthropicFirstPartyBackend(settings))
        elif name == "bedrock":
            out.append(AnthropicBedrockBackend(settings))
        elif name == "azure" and settings.azure_openai_endpoint and settings.azure_openai_api_key:
            out.append(AnthropicAzureBackend(settings))
    _backend_lists[key] = out
    return out


def native_backend_for(settings: Settings, provider_model_id: str) -> NativeBackend | None:
    """First configured native backend that serves this model, else None
    (→ adapter path)."""
    for backend in native_backends(settings):
        if backend.serves(provider_model_id):
            return backend
    return None


def count_tokens_backend(settings: Settings) -> NativeBackend | None:
    """First native backend with a count_tokens surface. Bedrock (mantle) has
    none, so with `bedrock,first_party` counting still goes first-party."""
    for backend in native_backends(settings):
        if backend.name != "bedrock" and backend.url.endswith("/v1/messages"):
            return backend
    return None


def get_anthropic_backend(
    settings: Settings,
) -> AnthropicFirstPartyBackend | AnthropicBedrockBackend | AnthropicAzureBackend:
    """Resolve the native-path backend: first configured entry of
    MLPAL_ANTHROPIC_BACKENDS. `vertex` is adapter-path only for now (its
    native wire needs OAuth token refresh — tracked in the worklog)."""
    priority = effective_anthropic_backends(settings)
    key = (
        priority,
        settings.anthropic_api_key,
        settings.anthropic_base_url,
        settings.bedrock_mantle_region,
        settings.bedrock_mantle_models,
        settings.bedrock_anthropic_models,
        settings.bedrock_native_endpoint,
        settings.azure_openai_endpoint,
        settings.azure_anthropic_deployments,
    )
    hit = _backends.get(key)
    if hit is not None:
        return hit  # type: ignore[return-value]
    for name in (n.strip() for n in priority.split(",")):
        backend: object | None = None
        if name == "first_party" and settings.anthropic_api_key:
            backend = AnthropicFirstPartyBackend(settings)
        elif name == "bedrock":
            backend = AnthropicBedrockBackend(settings)
        elif name == "azure" and settings.azure_openai_endpoint and settings.azure_openai_api_key:
            backend = AnthropicAzureBackend(settings)
        if backend is not None:
            _backends[key] = backend
            return backend
    raise ValueError(
        f"No usable native Anthropic backend in "
        f"MLPAL_ANTHROPIC_BACKENDS={priority!r} "
        "(first_party needs ANTHROPIC_API_KEY; bedrock needs AWS creds; "
        "azure needs MLPAL_AZURE_OPENAI_{ENDPOINT,API_KEY})"
    )
