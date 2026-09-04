"""Gemini native image generation: the resolved aspect ratio and resolution
tier must reach the SDK's ImageConfig, and Gemini's inline_data parts must
come back as GeneratedImage rows."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

import mlpal_assistants_service.adapters.google as g
from mlpal_assistants_service.adapters.base import ImageQuality

PNG = b"\x89PNG-fake"


def _adapter(captured: dict) -> g.GoogleAdapter:
    async def generate_content(*, model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        part = SimpleNamespace(inline_data=SimpleNamespace(data=PNG, mime_type="image/png"))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    adapter = g.GoogleAdapter(api_key="test-key")
    adapter._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    return adapter


@pytest.mark.asyncio
async def test_pixels_select_aspect_and_tier():
    captured: dict = {}
    resp = await _adapter(captured).generate_image("a cat", model="gemini-3-pro-image", size="3840x2160")
    cfg = captured["config"].image_config
    assert (cfg.aspect_ratio, cfg.image_size) == ("16:9", "4K")
    assert captured["model"] == "gemini-3-pro-image"
    assert len(resp.images) == 1
    assert resp.images[0].base64 == base64.b64encode(PNG).decode()
    assert resp.images[0].format == "png"


@pytest.mark.asyncio
async def test_hd_quality_renders_2k():
    captured: dict = {}
    await _adapter(captured).generate_image("a cat", model="gemini-3.1-flash-image", size="1:1", quality=ImageQuality.HD)
    assert captured["config"].image_config.image_size == "2K"


@pytest.mark.asyncio
async def test_lite_clamped_to_1k_and_default_model_is_stable_tag():
    captured: dict = {}
    await _adapter(captured).generate_image("a cat", model="gemini-3.1-flash-lite-image", size="4K")
    assert captured["config"].image_config.image_size == "1K"

    captured.clear()
    await _adapter(captured).generate_image("a cat")
    assert captured["model"] == "gemini-3-pro-image"
