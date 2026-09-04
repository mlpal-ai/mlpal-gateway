"""Tests for ImageSizeResolver, especially gpt-image-2 flexible resolutions.

gpt-image-2 accepts arbitrary sizes (each edge divisible by 16, longest edge
<= 3840, within OpenAI's pixel budget). The resolver must honor explicit pixel
sizes and true aspect ratios for it, while keeping the fixed-3-size behavior for
gpt-image-1/1.5 and DALL-E.
"""

import pytest

from mlpal_assistants_service.adapters.base import ImageQuality
from mlpal_assistants_service.adapters.base import ImageSizeResolver as R


class TestFlexibleGptImage2:
    def test_explicit_pixels_pass_through(self):
        assert R.to_pixels_openai("2048x2048", model="gpt-image-2") == "2048x2048"
        assert R.to_pixels_openai("1792x1024", model="gpt-image-2") == "1792x1024"
        assert R.to_pixels_openai("1280x720", model="gpt-image-2") == "1280x720"

    def test_aligns_to_multiple_of_16(self):
        # 1080 is not divisible by 16; OpenAI requires it -> snap to 1088.
        assert R.to_pixels_openai("1920x1080", model="gpt-image-2") == "1920x1088"

    def test_caps_longest_edge_at_3840(self):
        # Over-large request scales down proportionally to the 3840 max edge
        # (OpenAI then enforces the pixel budget and returns a clean 400 if over).
        assert R.to_pixels_openai("4096x4096", model="gpt-image-2") == "3840x3840"

    def test_auto_passes_through(self):
        assert R.to_pixels_openai("auto", model="gpt-image-2") == "auto"

    def test_aspect_maps_to_true_ratio_not_clamped(self):
        # Real 16:9, not the legacy 3:2 clamp (1536x1024).
        assert R.to_pixels_openai("16:9", model="gpt-image-2") == "1536x864"
        assert R.to_pixels_openai("9:16", model="gpt-image-2") == "864x1536"
        assert R.to_pixels_openai("1:1", model="gpt-image-2") == "1024x1024"


class TestFixedModelsUnchanged:
    def test_gpt_image_1_still_clamps_to_three_sizes(self):
        # Flexibility must NOT leak to gpt-image-1 (only supports the fixed 3).
        assert R.to_pixels_openai("2048x2048", model="gpt-image-1") == "1024x1024"
        assert R.to_pixels_openai("1792x1024", model="gpt-image-1") == "1536x1024"
        assert R.to_pixels_openai("16:9", model="gpt-image-1") == "1536x1024"

    def test_gpt_image_1_5_still_clamps(self):
        assert R.to_pixels_openai("2048x2048", model="gpt-image-1.5") == "1024x1024"

    def test_dalle3_unchanged(self):
        assert R.to_pixels_openai("16:9", model="dall-e-3") == "1792x1024"
        assert R.to_pixels_openai("2048x2048", model="dall-e-3") == "1024x1024"


class TestHelpers:
    @pytest.mark.parametrize(
        "n,expected",
        [(1080, 1088), (1024, 1024), (720, 720), (1000, 992), (8, 16)],
    )
    def test_align_edge(self, n, expected):
        assert R._align_edge(n) == expected

    def test_parse_explicit_pixels(self):
        assert R._parse_explicit_pixels("2048x2048") == (2048, 2048)
        assert R._parse_explicit_pixels("16:9") is None
        assert R._parse_explicit_pixels("square") is None


class TestGoogleImageSize:
    """Gemini 3.x takes resolution as `image_size` (512px/1K/2K/4K), separate
    from aspect ratio. Pixels imply a tier, `hd` lifts to 2K, models clamp."""

    PRO = "gemini-3-pro-image"
    FLASH = "gemini-3.1-flash-image"
    LITE = "gemini-3.1-flash-lite-image"

    def test_bare_tier_token_taken_literally(self):
        assert R.to_image_size_google("4K", model=self.PRO) == "4K"
        assert R.to_image_size_google("2k", model=self.PRO) == "2K"
        assert R.to_image_size_google("1K", model=self.PRO) == "1K"

    def test_bare_tier_token_has_square_aspect(self):
        assert R.to_aspect_ratio_google("4K") == "1:1"
        assert R.to_aspect_ratio_google("512px") == "1:1"

    @pytest.mark.parametrize(
        "pixels,tier",
        [("1024x1024", "1K"), ("1536x864", "1K"), ("1792x768", "1K"),
         ("2048x1152", "2K"), ("2048x2048", "2K"), ("2560x1440", "2K"),
         ("3840x2160", "4K"), ("4096x1792", "4K"), ("4096x4096", "4K")],
    )
    def test_explicit_pixels_infer_tier(self, pixels, tier):
        assert R.to_image_size_google(pixels, model=self.PRO) == tier

    def test_aspect_ratio_defaults_to_1k(self):
        assert R.to_image_size_google("16:9", model=self.PRO) == "1K"
        assert R.to_image_size_google("landscape", model=self.PRO) == "1K"

    def test_hd_lifts_to_2k(self):
        assert R.to_image_size_google("16:9", ImageQuality.HD, self.PRO) == "2K"
        assert R.to_image_size_google("1024x1024", ImageQuality.HD, self.PRO) == "2K"
        # ...but never lowers an explicit 4K.
        assert R.to_image_size_google("3840x2160", ImageQuality.HD, self.PRO) == "4K"

    def test_lite_is_1k_only(self):
        assert R.to_image_size_google("4K", model=self.LITE) == "1K"
        assert R.to_image_size_google("2048x2048", ImageQuality.HD, self.LITE) == "1K"

    def test_512px_only_for_flash(self):
        assert R.to_image_size_google("512px", model=self.FLASH) == "512px"
        assert R.to_image_size_google("512px", model=self.PRO) == "1K"
        assert R.to_image_size_google("512x512", model=self.FLASH) == "1K"  # pixels never go sub-1K
