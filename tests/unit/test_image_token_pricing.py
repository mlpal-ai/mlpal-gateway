"""Token-priced image models (gpt-image-2.5): three list rates (text in,
image in, image out) reproduced exactly; per-image rows keep billing per image."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mlpal_assistants_service.adapters.base import ImageQuality, ImageSizeResolver
from mlpal_assistants_service.adapters.openai import _image_usage, _openai_image_quality
from mlpal_assistants_service.db.models.model_pricing import ModelPricing
from mlpal_assistants_service.schemas.images import ImageGenerationRequest
from mlpal_assistants_service.services.pricing import PricingService


def _row(**over) -> ModelPricing:
    base = {
        "model_tag": "gpt-image-2.5-flare", "operation": "image_generation", "tier": "premium",
        "input_rate": Decimal("5"), "output_rate": Decimal("30"), "image_input_rate": Decimal("8"),
        "rate_unit": "per_1m_tokens", "markup_multiplier": Decimal("3"), "cu_to_dollar": Decimal("10"),
        "input_cu_rate": Decimal("1.5"), "output_cu_rate": Decimal("9"),
    }
    base.update(over)
    return ModelPricing(**base)


@pytest.fixture
def pricing():
    svc = PricingService.__new__(PricingService)
    svc.get_pricing = AsyncMock(return_value=_row())
    return svc


@pytest.mark.asyncio
async def test_image_tokens_bill_three_rates_exactly(pricing):
    # flare high, 1024x1024 probe 2026-09-11: 17 text in, 1756 image out; plus a 500-token reference image
    cu = await pricing.calculate_compute_units(
        "gpt-image-2.5-flare", 17, 1756, operation="image_generation", image_input_units=500,
    )
    dollars = (Decimal(17) * 5 + Decimal(500) * 8 + Decimal(1756) * 30) / Decimal(1_000_000)
    assert cu * 10 == dollars


@pytest.mark.asyncio
async def test_null_image_input_rate_falls_back_to_input_rate(pricing):
    pricing.get_pricing = AsyncMock(return_value=_row(image_input_rate=None))
    cu = await pricing.calculate_compute_units("m", 0, 0, operation="image_generation", image_input_units=1000)
    assert cu * 10 == Decimal(1000) * 5 / Decimal(1_000_000)


@pytest.mark.asyncio
async def test_no_image_tokens_is_unchanged(pricing):
    assert await pricing.calculate_compute_units("m", 100, 10, operation="image_generation") == \
        await pricing.calculate_compute_units("m", 100, 10, operation="image_generation", image_input_units=0)


def test_redis_serializer_round_trips_image_input_rate_and_prefix_bumped():
    svc = PricingService.__new__(PricingService)
    from datetime import date

    row = _row()
    row.id = 1
    row.is_active = True
    row.effective_date = date(2026, 9, 11)
    row.cache_read_rate = None
    back = svc._dict_to_pricing(svc._pricing_to_dict(row))
    assert back.image_input_rate == Decimal("8") and back.cache_read_rate is None
    assert PricingService.CACHE_PREFIX == "pricing:v3:"   # new column → new prefix (serializer rule)


def test_openai_quality_ladder_maps_per_model():
    assert _openai_image_quality("gpt-image-2.5-flare", ImageQuality.MAX) == "max"
    assert _openai_image_quality("gpt-image-2", ImageQuality.STANDARD) == "medium"
    assert _openai_image_quality("gpt-image-2", ImageQuality.HD) == "high"
    assert _openai_image_quality("dall-e-3", ImageQuality.XHIGH) == "hd"
    assert _openai_image_quality("dall-e-3", ImageQuality.LOW) == "standard"


def test_image_usage_splits_text_and_image_input():
    u = _image_usage(SimpleNamespace(input_tokens=517, output_tokens=1756,
                                     input_tokens_details=SimpleNamespace(text_tokens=17, image_tokens=500)))
    assert (u.input_tokens, u.image_input_tokens, u.output_tokens) == (517, 500, 1756)
    assert _image_usage(None) is None


def test_request_schema_accepts_native_ladder():
    for q in ("standard", "hd", "low", "medium", "high", "xhigh", "max", "auto"):
        ImageGenerationRequest(model="gpt-image-2.5-flare", prompt="x", quality=q)


def test_gemini_treats_xhigh_and_max_as_hd():
    assert ImageSizeResolver.to_image_size_google("1024x1024", ImageQuality.MAX, "gemini-3-pro-image") == \
        ImageSizeResolver.to_image_size_google("1024x1024", ImageQuality.HD, "gemini-3-pro-image")
