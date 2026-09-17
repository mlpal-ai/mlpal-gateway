"""Discover and LIVE-VERIFY what your cloud credentials can serve, and emit
the exact env values the serving backends need.

    uv run python scripts/probe_backends.py bedrock   # Claude on Bedrock; needs AWS creds
    uv run python scripts/probe_backends.py openai    # OpenAI models on Bedrock (Responses wire); needs AWS creds
    uv run python scripts/probe_backends.py vertex    # needs GOOGLE_APPLICATION_CREDENTIALS + MLPAL_VERTEX_PROJECT
    uv run python scripts/probe_backends.py azure     # needs MLPAL_AZURE_OPENAI_{ENDPOINT,API_KEY}

Each leg makes a 1-token call per candidate model — an entry only enters the
map if the call actually succeeds, so the map is truth, not hope. Output is
the JSON for MLPAL_BEDROCK_ANTHROPIC_MODELS / MLPAL_VERTEX_ANTHROPIC_MODELS,
or the deployment list for Azure.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from mlpal_assistants_service.core.config import get_settings


def _anthropic_catalog_ids() -> list[str]:
    from importlib.resources import files

    reg = json.loads(
        files("mlpal_assistants_service.catalog").joinpath("registry.json").read_text()
    )
    rows = reg if isinstance(reg, list) else reg.get("models", [])
    return [
        r["provider_model_id"]
        for r in rows
        if r.get("provider") == "anthropic"
        and r.get("is_active", True)
        and not r.get("is_deprecated")
    ]


def _openai_catalog_chat_ids() -> list[str]:
    from importlib.resources import files
    reg = json.loads(
        files("mlpal_assistants_service.catalog").joinpath("registry.json").read_text()
    )
    rows = reg if isinstance(reg, list) else reg.get("models", [])
    return [
        r["provider_model_id"]
        for r in rows
        if r.get("provider") == "openai"
        and r.get("is_active", True)
        and not r.get("is_deprecated")
        and (r.get("capabilities") or {}).get("operation", "chat") == "chat"
    ]


async def probe_bedrock_openai() -> dict[str, str]:
    """OpenAI proprietary models on Bedrock's Responses wire
    (bedrock-runtime/openai/v1, SigV4): {provider_model_id: profile id}."""
    import boto3
    import httpx

    from mlpal_assistants_service.adapters.aws_sigv4 import SigV4HttpxAuth

    settings = get_settings()
    region = settings.bedrock_mantle_region
    bedrock = boto3.client("bedrock", region_name=region)
    available = {
        p["inferenceProfileId"]
        for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]
        if ".openai." in p["inferenceProfileId"]
    } | {
        m["modelId"]
        for m in bedrock.list_foundation_models(byProvider="OpenAI")["modelSummaries"]
    }
    url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
    mapping: dict[str, str] = {}
    async with httpx.AsyncClient(auth=SigV4HttpxAuth(region), timeout=60) as c:
        for fp_id in _openai_catalog_chat_ids():
            for cand in (f"global.openai.{fp_id}", f"us.openai.{fp_id}", f"openai.{fp_id}"):
                if cand not in available:
                    continue
                print(f"  probing {fp_id} -> {cand}", file=sys.stderr)
                r = await c.post(
                    url, json={"model": cand, "input": "hi", "max_output_tokens": 16}
                )
                if r.status_code == 200:
                    mapping[fp_id] = cand
                    break
                print(f"    {cand}: HTTP {r.status_code}: {r.text[:110]}", file=sys.stderr)
    return mapping


async def _try_messages(client, model_id: str) -> bool:
    try:
        await client.messages.create(
            model=model_id, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
        )
        return True
    except Exception as e:  # noqa: BLE001 — probe: any failure means "not served"
        print(f"    {model_id}: {type(e).__name__}: {str(e)[:110]}", file=sys.stderr)
        return False


async def probe_bedrock() -> dict[str, str]:
    import boto3
    from anthropic import AsyncAnthropicBedrock

    settings = get_settings()
    region = settings.bedrock_mantle_region
    bedrock = boto3.client("bedrock", region_name=region)
    available: set[str] = {
        m["modelId"]
        for m in bedrock.list_foundation_models(byProvider="anthropic")["modelSummaries"]
    }
    available |= {
        p["inferenceProfileId"]
        for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]
        if "anthropic" in p["inferenceProfileId"]
    }

    client = AsyncAnthropicBedrock(aws_region=region)
    mapping: dict[str, str] = {}
    mantle: list[str] = []
    for fp_id in _anthropic_catalog_ids():
        # Candidates in preference order: global profile (routes to capacity
        # anywhere) > us profile > bare model ID; exact ID, then -v1:0 / -v1
        # suffixed variants (Bedrock suffixing is inconsistent across models).
        stems = [fp_id, f"{fp_id}-v1:0", f"{fp_id}-v1"]
        candidates = [
            f"{prefix}anthropic.{stem}"
            for prefix in ("global.", "us.", "")
            for stem in stems
        ]
        for cand in candidates:
            if cand not in available:
                continue
            print(f"  probing {fp_id} -> {cand}", file=sys.stderr)
            if await _try_messages(client, cand):
                mapping[fp_id] = cand
                break
        if await _probe_mantle(region, fp_id, mapping.get(fp_id)):
            mantle.append(fp_id)
    return mapping, mantle


async def _probe_mantle(region: str, fp_id: str, profile_id: str | None = None) -> bool:
    """Whether the NATIVE Anthropic wire on Bedrock serves this model —
    bedrock-runtime by profile id (default) or the legacy mantle host."""
    import httpx

    from mlpal_assistants_service.core.config import get_settings
    from mlpal_assistants_service.services.bedrock_mantle import BedrockMantleClient

    endpoint = get_settings().bedrock_native_endpoint
    signer = BedrockMantleClient(
        region=region, endpoint=endpoint, model_map={fp_id: profile_id} if profile_id else None
    )
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "model": fp_id if profile_id else f"anthropic.{fp_id}",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    body, _, _ = signer.adapt_body(body)
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(signer.url, content=body, headers=signer.sign(body))
    ok = r.status_code == 200
    print(f"    mantle {fp_id}: {'OK' if ok else r.status_code}", file=sys.stderr)
    return ok


async def probe_vertex() -> dict[str, str]:
    from anthropic import AsyncAnthropicVertex

    settings = get_settings()
    if not settings.vertex_project:
        sys.exit("Set MLPAL_VERTEX_PROJECT first")
    client = AsyncAnthropicVertex(
        project_id=settings.vertex_project, region=settings.vertex_location
    )
    mapping: dict[str, str] = {}
    for fp_id in _anthropic_catalog_ids():
        # Vertex IDs: >=4.6 generation drops the @date; older keep it.
        m = re.match(r"^(.*?)-(\d{8})$", fp_id)
        candidates = [fp_id] if not m else [f"{m.group(1)}@{m.group(2)}", fp_id]
        for cand in candidates:
            print(f"  probing {fp_id} -> {cand}", file=sys.stderr)
            if await _try_messages(client, cand):
                mapping[fp_id] = cand
                break
    return mapping


async def probe_azure() -> list[dict[str, str]]:
    import httpx

    settings = get_settings()
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        sys.exit("Set MLPAL_AZURE_OPENAI_ENDPOINT and MLPAL_AZURE_OPENAI_API_KEY first")
    # Data-plane deployments listing (the v1 /models route returns the whole
    # Azure model CATALOG, not what this resource actually deploys).
    url = settings.azure_openai_endpoint.rstrip("/") + "/openai/deployments"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(
            url,
            params={"api-version": "2023-03-15-preview"},
            headers={"api-key": settings.azure_openai_api_key},
        )
        r.raise_for_status()
        return sorted(
            ({"deployment": d["id"], "model": d["model"]} for d in r.json()["data"]),
            key=lambda d: d["deployment"],
        )


async def main() -> None:
    leg = sys.argv[1] if len(sys.argv) > 1 else ""
    if leg == "bedrock":
        mapping, mantle = await probe_bedrock()
        print(f"\nMLPAL_BEDROCK_ANTHROPIC_MODELS='{json.dumps(mapping)}'")
        print(f"MLPAL_BEDROCK_MANTLE_MODELS='{json.dumps(mantle)}'")
        print("MLPAL_ANTHROPIC_BACKENDS=first_party,bedrock  # or bedrock-first")
    elif leg == "openai":
        mapping = await probe_bedrock_openai()
        print(f"\nMLPAL_BEDROCK_OPENAI_MODELS='{json.dumps(mapping)}'")
        print("MLPAL_OPENAI_BACKENDS=bedrock,first_party  # or first_party,bedrock")
    elif leg == "vertex":
        mapping = await probe_vertex()
        print(f"\nMLPAL_VERTEX_ANTHROPIC_MODELS='{json.dumps(mapping)}'")
    elif leg == "azure":
        deployments = await probe_azure()
        print("\nDeployments on this resource:")
        for d in deployments:
            print(f"  {d['deployment']}  (model {d['model']})")
        mapping = {d["model"]: d["deployment"] for d in deployments}
        print(f"\nMLPAL_AZURE_DEPLOYMENTS='{json.dumps(mapping)}'")
        print("MLPAL_OPENAI_BACKENDS=azure,first_party  # or azure-first")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
