"""Base adapter interface for AI providers.

Provides a unified abstraction for:
- Chat completion (text)
- Structured output (JSON schema)
- Vision (images)
- Documents (PDFs, text files)
- Audio (where supported)
- Tool/function calling
- Streaming
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def provider_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of the upstream HTTP status from a provider SDK error.

    Lets the gateway distinguish a client-caused bad request (e.g. OpenAI 400 for
    an invalid ``max_output_tokens``) from a genuine upstream/connection failure,
    so the API can return 400 instead of a misleading 502. Returns None when no
    status is present (connection errors, unknown SDK shapes).

    Covers the OpenAI/Anthropic SDKs (``status_code``), httpx-based errors
    (``response.status_code``) and google-genai (``code``).
    """
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


# =============================================================================
# File Types and Attachments
# =============================================================================

class FileType(str, Enum):
    """
    Supported file types for multimodal processing.

    Used throughout the adapters for:
    - Routing files to appropriate content block types
    - Validating model capabilities
    - Auto-detecting MIME types
    """

    IMAGE = "image"      # jpg, png, gif, webp, etc.
    PDF = "pdf"          # PDF documents
    AUDIO = "audio"      # mp3, wav, m4a, etc.
    VIDEO = "video"      # mp4, mov, etc. (limited support)
    CODE = "code"        # py, js, ts, etc. (sent as text)
    TEXT = "text"        # txt, md, json, etc. (sent as text)
    DOCUMENT = "document"  # docx, xlsx, etc. (provider-specific handling)
    OTHER = "other"      # Fallback for unrecognized types


class FileSource(str, Enum):
    """How the file data is provided."""

    BASE64 = "base64"    # Inline base64 encoded data
    URL = "url"          # Public URL (model fetches it)
    FILE_ID = "file_id"  # Pre-uploaded file reference (provider-specific)
    PATH = "path"        # Local file path (adapter handles encoding/upload)


@dataclass
class FileAttachment:
    """
    Unified file attachment for all providers.

    Each adapter converts this to provider-specific format.

    Examples:
        # Image from URL
        FileAttachment(
            type=FileType.IMAGE,
            source=FileSource.URL,
            data="https://example.com/image.png"
        )

        # PDF from base64
        FileAttachment(
            type=FileType.PDF,
            source=FileSource.BASE64,
            data="JVBERi0xLjQK...",
            mime_type="application/pdf"
        )

        # Audio from local path
        FileAttachment(
            type=FileType.AUDIO,
            source=FileSource.PATH,
            data="/path/to/audio.mp3"
        )

        # Pre-uploaded file
        FileAttachment(
            type=FileType.PDF,
            source=FileSource.FILE_ID,
            data="file_abc123",
            filename="report.pdf"
        )
    """

    type: FileType
    source: FileSource
    data: str  # base64 string, URL, file_id, or local path
    mime_type: str | None = None  # Auto-detected if not provided
    filename: str | None = None   # Optional, for display/logging

    def __post_init__(self) -> None:
        """Auto-detect mime_type if not provided."""
        if self.mime_type is None:
            self.mime_type = self._detect_mime_type()

    def _detect_mime_type(self) -> str:
        """Detect MIME type from file type or path extension."""
        # If we have a path or URL, try to detect from extension
        if self.source in (FileSource.PATH, FileSource.URL):
            ext = Path(self.data).suffix.lower()
            mime = EXTENSION_TO_MIME.get(ext)
            if mime:
                return mime

        # Fall back to type-based defaults
        return FILE_TYPE_DEFAULT_MIME.get(self.type, "application/octet-stream")


# Extension to MIME type mapping
EXTENSION_TO_MIME: dict[str, str] = {
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    # Video
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    # Documents
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".csv": "text/csv",
    # Text/Code
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".xml": "application/xml",
    ".html": "text/html",
    ".css": "text/css",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".java": "text/x-java",
    ".cpp": "text/x-c++src",
    ".c": "text/x-csrc",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
}

# Default MIME types per file type
FILE_TYPE_DEFAULT_MIME: dict[FileType, str] = {
    FileType.IMAGE: "image/png",
    FileType.PDF: "application/pdf",
    FileType.AUDIO: "audio/mpeg",
    FileType.VIDEO: "video/mp4",
    FileType.CODE: "text/plain",
    FileType.TEXT: "text/plain",
    FileType.DOCUMENT: "application/octet-stream",
    FileType.OTHER: "application/octet-stream",
}

# Extension to FileType mapping (for auto-detection)
EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # Images
    ".jpg": FileType.IMAGE, ".jpeg": FileType.IMAGE, ".png": FileType.IMAGE,
    ".gif": FileType.IMAGE, ".webp": FileType.IMAGE, ".bmp": FileType.IMAGE,
    ".svg": FileType.IMAGE,
    # Audio
    ".mp3": FileType.AUDIO, ".wav": FileType.AUDIO, ".m4a": FileType.AUDIO,
    ".webm": FileType.AUDIO, ".ogg": FileType.AUDIO, ".flac": FileType.AUDIO,
    ".aac": FileType.AUDIO,
    # Video
    ".mp4": FileType.VIDEO, ".mov": FileType.VIDEO, ".avi": FileType.VIDEO,
    ".mkv": FileType.VIDEO,
    # PDF
    ".pdf": FileType.PDF,
    # Documents
    ".docx": FileType.DOCUMENT, ".doc": FileType.DOCUMENT,
    ".xlsx": FileType.DOCUMENT, ".xls": FileType.DOCUMENT,
    ".pptx": FileType.DOCUMENT, ".ppt": FileType.DOCUMENT,
    ".csv": FileType.TEXT,
    # Text
    ".txt": FileType.TEXT, ".md": FileType.TEXT, ".json": FileType.TEXT,
    ".yaml": FileType.TEXT, ".yml": FileType.TEXT, ".xml": FileType.TEXT,
    ".html": FileType.TEXT, ".css": FileType.TEXT,
    # Code
    ".py": FileType.CODE, ".js": FileType.CODE, ".ts": FileType.CODE,
    ".java": FileType.CODE, ".cpp": FileType.CODE, ".c": FileType.CODE,
    ".go": FileType.CODE, ".rs": FileType.CODE, ".rb": FileType.CODE,
    ".php": FileType.CODE, ".swift": FileType.CODE, ".kt": FileType.CODE,
}


def detect_file_type(path_or_url: str) -> FileType:
    """Detect FileType from a file path or URL."""
    ext = Path(path_or_url).suffix.lower()
    return EXTENSION_TO_FILE_TYPE.get(ext, FileType.OTHER)


# =============================================================================
# Model Capabilities
# =============================================================================

@dataclass
class ModelCapabilities:
    """
    Describes what a specific model can handle.

    Used for:
    - Validating file attachments before sending
    - Providing helpful error messages
    - Routing to appropriate models
    """

    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_pdf: bool = False
    supports_documents: bool = False  # docx, xlsx, etc.
    supports_tools: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = True
    supports_mcp: bool = False
    max_file_size_mb: int = 20  # Default 20MB limit
    max_context_tokens: int = 128000
    max_output_tokens: int = 16384

    def supports_file_type(self, file_type: FileType) -> bool:
        """Check if this model supports a given file type."""
        mapping = {
            FileType.IMAGE: self.supports_images,
            FileType.PDF: self.supports_pdf,
            FileType.AUDIO: self.supports_audio,
            FileType.VIDEO: self.supports_video,
            FileType.DOCUMENT: self.supports_documents or self.supports_pdf,
            FileType.CODE: True,  # Code is sent as text
            FileType.TEXT: True,  # Text is always supported
            FileType.OTHER: False,
        }
        return mapping.get(file_type, False)

    def get_unsupported_types(self, file_types: list[FileType]) -> list[FileType]:
        """Return list of file types that are not supported."""
        return [ft for ft in file_types if not self.supports_file_type(ft)]


# =============================================================================
# Token Usage and Responses
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    # Prompt-cache WRITES by price tier: `5m` = the standard 1.25x write
    # (Anthropic 5-minute TTL, and OpenAI's untiered cache write); `1h` = the
    # 2x Anthropic 1-hour TTL. Like cached_tokens, whether input_tokens already
    # contains them follows the adapter's cached_tokens_included_in_input.
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    # Hidden reasoning/thinking tokens (subset of output_tokens). None = the
    # provider does not report them separately (Anthropic).
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class AdapterResponse:
    """Standardized response from any provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage
    finish_reason: str = "stop"
    latency_ms: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens


# =============================================================================
# Generation Response Types
# =============================================================================

class ImageSize(str, Enum):
    """Standard image sizes for generation."""

    SQUARE = "1024x1024"
    LANDSCAPE = "1792x1024"     # 16:9 (DALL-E 3 supported)
    PORTRAIT = "1024x1792"      # 9:16 (DALL-E 3 supported)
    WIDE = "1792x1024"          # 16:9 (alias for LANDSCAPE)
    TALL = "1024x1792"          # 9:16 (alias for PORTRAIT)
    AUTO = "auto"               # Let provider decide


class ImageQuality(str, Enum):
    """Image quality levels."""

    STANDARD = "standard"
    HD = "hd"
    HIGH = "high"
    AUTO = "auto"


class ImageSizeResolver:
    """
    Production-grade image size resolver for unified cross-provider experience.

    Accepts multiple input formats and resolves to provider-specific outputs:
    - Semantic presets: "square", "landscape", "portrait", "wide", "tall"
    - Aspect ratios: "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"
    - Pixel dimensions: "1024x1024", "1792x1024", "1536x1024", etc.

    Usage:
        resolver = ImageSizeResolver()
        # For OpenAI (needs pixel dimensions)
        pixels = resolver.to_pixels(user_input, provider="openai")
        # For Google (needs aspect ratio)
        aspect = resolver.to_aspect_ratio(user_input)
    """

    # Semantic preset to aspect ratio mapping
    PRESETS_TO_ASPECT: dict[str, str] = {
        "square": "1:1",
        "landscape": "16:9",
        "portrait": "9:16",
        "wide": "16:9",
        "tall": "9:16",
        "auto": "1:1",
        # Additional semantic names
        "widescreen": "16:9",
        "vertical": "9:16",
        "horizontal": "16:9",
        "standard": "4:3",
        "classic": "3:2",
    }

    # Pixel dimensions to aspect ratio mapping
    PIXELS_TO_ASPECT: dict[str, str] = {
        # Square
        "1024x1024": "1:1",
        "512x512": "1:1",
        "256x256": "1:1",
        # 16:9 Landscape
        "1792x1024": "16:9",
        "1920x1080": "16:9",
        # 9:16 Portrait
        "1024x1792": "9:16",
        "1080x1920": "9:16",
        # 3:2 (OpenAI gpt-image)
        "1536x1024": "3:2",
        # 2:3 (OpenAI gpt-image)
        "1024x1536": "2:3",
        # 4:3
        "1024x768": "4:3",
        "1536x1152": "4:3",
        # 3:4
        "768x1024": "3:4",
        "1152x1536": "3:4",
        # 21:9 Ultrawide
        "2560x1080": "21:9",
    }

    # Provider-specific pixel dimension mappings
    # Maps aspect ratio to best available pixel dimensions per provider
    OPENAI_ASPECT_TO_PIXELS: dict[str, str] = {
        # gpt-image-1/1.5 supported sizes
        "1:1": "1024x1024",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
        # Fallback mappings for unsupported ratios (map to closest)
        "16:9": "1536x1024",  # Closest to 16:9
        "9:16": "1024x1536",  # Closest to 9:16
        "4:3": "1536x1024",   # Closest to 4:3
        "3:4": "1024x1536",   # Closest to 3:4
        "4:5": "1024x1536",   # Closest to 4:5
        "5:4": "1536x1024",   # Closest to 5:4
        "21:9": "1536x1024",  # Map ultrawide to landscape
    }

    # OpenAI image models that accept arbitrary pixel sizes (not just the fixed
    # three). gpt-image-2's rules: each edge a multiple of 16, longest edge
    # <= 3840, and total pixels within OpenAI's (dynamic) budget. We pre-apply
    # the cheap, deterministic rules (align-to-16, cap longest edge) and let
    # OpenAI enforce the budget — it returns a precise 400 that the gateway now
    # surfaces as a 400 (see http_status_for_provider_error).
    OPENAI_FLEXIBLE_IMAGE_MODELS: set[str] = {"gpt-image-2"}
    _OPENAI_FLEX_MAX_EDGE = 3840
    _PIXEL_ALIGN = 16

    # True-aspect default pixel sizes for flexible models when the caller gives
    # an aspect ratio / preset instead of explicit pixels (all edges divisible
    # by 16, ~1-1.6MP). Unlike the legacy map below, these honor the *actual*
    # ratio (e.g. real 16:9) instead of clamping to 3:2 / 2:3.
    OPENAI_FLEXIBLE_ASPECT_TO_PIXELS: dict[str, str] = {
        "1:1": "1024x1024",
        "3:2": "1536x1024",
        "2:3": "1024x1536",
        "16:9": "1536x864",
        "9:16": "864x1536",
        "4:3": "1280x960",
        "3:4": "960x1280",
        "5:4": "1280x1024",
        "4:5": "1024x1280",
        "21:9": "1792x768",
    }

    DALLE_ASPECT_TO_PIXELS: dict[str, str] = {
        # DALL-E 3 supported sizes
        "1:1": "1024x1024",
        "16:9": "1792x1024",
        "9:16": "1024x1792",
        # Fallback mappings
        "3:2": "1792x1024",
        "2:3": "1024x1792",
        "4:3": "1792x1024",
        "3:4": "1024x1792",
        "4:5": "1024x1792",
        "5:4": "1792x1024",
        "21:9": "1792x1024",
    }

    # Google supported aspect ratios (Gemini 3 / Imagen)
    GOOGLE_SUPPORTED_ASPECTS: set[str] = {
        "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9"
    }

    # Gemini 3.x native image models take output resolution as a knob
    # (`image_config.image_size`) separate from aspect ratio. Values are
    # case-sensitive on the wire ("1K", not "1k"). Lite is 1K-only; only
    # 3.1 Flash can render below 1K.
    GOOGLE_IMAGE_SIZE_ALIASES: dict[str, str] = {
        "512px": "512px",
        "0.5k": "512px",
        "1k": "1K",
        "2k": "2K",
        "4k": "4K",
    }
    GOOGLE_1K_ONLY_IMAGE_MODELS: set[str] = {"gemini-3.1-flash-lite-image"}
    GOOGLE_512PX_IMAGE_MODELS: set[str] = {"gemini-3.1-flash-image"}
    # Thresholds for inferring a Gemini tier from explicit pixels. A tier is
    # reached by either its edge length or its pixel count, so both a square
    # 2048x2048 (4MP) and an ultra-wide 4096x1792 (7MP) land where a caller
    # expects. Megapixel cutoffs sit at the midpoints of ~1/~4/~16MP.
    _GOOGLE_2K_MIN_EDGE = 2048
    _GOOGLE_4K_MIN_EDGE = 3072
    _GOOGLE_2K_MIN_PIXELS = 2_000_000
    _GOOGLE_4K_MIN_PIXELS = 8_000_000

    @classmethod
    def normalize_to_aspect_ratio(cls, size_input: str | ImageSize) -> str:
        """
        Normalize any size input to an aspect ratio string.

        Args:
            size_input: Can be:
                - ImageSize enum
                - Semantic preset: "square", "landscape", "portrait"
                - Aspect ratio: "16:9", "1:1", "9:16"
                - Pixel dimensions: "1024x1024", "1792x1024"

        Returns:
            Aspect ratio string like "16:9", "1:1", etc.
        """
        # Handle enum
        if isinstance(size_input, ImageSize):
            size_input = size_input.value

        size_str = size_input.lower().strip()

        # Check semantic presets
        if size_str in cls.PRESETS_TO_ASPECT:
            return cls.PRESETS_TO_ASPECT[size_str]

        # A bare resolution tier ("2K", "512px") carries no aspect information.
        if size_str in cls.GOOGLE_IMAGE_SIZE_ALIASES:
            return "1:1"

        # Check if it's already an aspect ratio (contains ':')
        if ":" in size_str:
            # Validate it's a known ratio or parse it
            parts = size_str.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return size_str

        # Check pixel dimensions
        if "x" in size_str:
            if size_str in cls.PIXELS_TO_ASPECT:
                return cls.PIXELS_TO_ASPECT[size_str]
            # Try to calculate aspect ratio from dimensions
            parts = size_str.split("x")
            if len(parts) == 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                    return cls._calculate_aspect_ratio(w, h)
                except ValueError:
                    pass

        # Default fallback
        return "1:1"

    @staticmethod
    def _parse_explicit_pixels(size_input: str | ImageSize) -> tuple[int, int] | None:
        """Return (w, h) if the input is explicit pixel dimensions, else None."""
        if isinstance(size_input, ImageSize):
            size_input = size_input.value
        s = str(size_input).lower().strip()
        if "x" in s:
            parts = s.split("x")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                return int(parts[0]), int(parts[1])
        return None

    @classmethod
    def _align_edge(cls, n: int) -> int:
        """Round a pixel dimension to the nearest multiple of 16 (min 16)."""
        return max(cls._PIXEL_ALIGN, round(n / cls._PIXEL_ALIGN) * cls._PIXEL_ALIGN)

    @classmethod
    def _to_flexible_openai_pixels(cls, w: int, h: int) -> str:
        """Fit explicit dims to a flexible OpenAI model's deterministic rules:
        scale down proportionally so the longest edge <= 3840, then snap both
        edges to a multiple of 16. (Total-pixel budget is left to OpenAI.)"""
        longest = max(w, h)
        if longest > cls._OPENAI_FLEX_MAX_EDGE:
            scale = cls._OPENAI_FLEX_MAX_EDGE / longest
            w = round(w * scale)
            h = round(h * scale)
        return f"{cls._align_edge(w)}x{cls._align_edge(h)}"

    @classmethod
    def to_pixels_openai(cls, size_input: str | ImageSize, model: str = "gpt-image-1") -> str:
        """
        Convert any size input to OpenAI pixel dimensions.

        Args:
            size_input: Any supported size format
            model: OpenAI model (gpt-image-2, gpt-image-1, dall-e-3, etc.)

        Returns:
            Pixel dimension string like "1024x1024", or "auto" for flexible
            models when the caller asked the model to choose.
        """
        # Flexible models (gpt-image-2) honor explicit pixel sizes and "auto",
        # and map aspect/preset inputs to the *true* aspect ratio.
        if model in cls.OPENAI_FLEXIBLE_IMAGE_MODELS:
            normalized = (
                size_input.value if isinstance(size_input, ImageSize) else str(size_input)
            ).strip().lower()
            if normalized == "auto":
                return "auto"
            explicit = cls._parse_explicit_pixels(size_input)
            if explicit:
                return cls._to_flexible_openai_pixels(*explicit)
            aspect = cls.normalize_to_aspect_ratio(size_input)
            return cls.OPENAI_FLEXIBLE_ASPECT_TO_PIXELS.get(aspect, "1024x1024")

        aspect = cls.normalize_to_aspect_ratio(size_input)

        # Fixed-size models: clamp to the nearest supported size.
        if model.startswith("dall-e") or model == "dall-e-3":
            return cls.DALLE_ASPECT_TO_PIXELS.get(aspect, "1024x1024")
        else:
            # gpt-image-1 / gpt-image-1.5 (three fixed sizes)
            return cls.OPENAI_ASPECT_TO_PIXELS.get(aspect, "1024x1024")

    @classmethod
    def to_aspect_ratio_google(cls, size_input: str | ImageSize) -> str:
        """
        Convert any size input to Google-compatible aspect ratio.

        Args:
            size_input: Any supported size format

        Returns:
            Aspect ratio string supported by Google
        """
        aspect = cls.normalize_to_aspect_ratio(size_input)

        # If aspect ratio is supported by Google, use it directly
        if aspect in cls.GOOGLE_SUPPORTED_ASPECTS:
            return aspect

        # Map to closest supported ratio
        aspect_mapping = {
            "21:9": "16:9",  # Map ultrawide to widescreen
        }
        return aspect_mapping.get(aspect, "1:1")

    @classmethod
    def to_image_size_google(
        cls,
        size_input: str | ImageSize,
        quality: ImageQuality = ImageQuality.STANDARD,
        model: str = "",
    ) -> str:
        """
        Resolve the Gemini ``image_size`` tier ("512px" | "1K" | "2K" | "4K").

        Resolution order:
        1. A bare tier token ("2K", "4k", "512px") selects that tier.
        2. Explicit pixels infer the tier from edge length / megapixels
           (e.g. "3840x2160" -> "4K", "2048x2048" -> "2K", "1024x1024" -> "1K").
        3. Otherwise (aspect ratio / preset) the tier is 1K.

        Then ``quality`` hd/high raises anything below 2K to 2K — including an
        explicit "1K" or sub-2K pixels — because Gemini has no separate quality
        knob, so resolution is how "hd" is honored. Finally the result is
        clamped to what ``model`` can actually render.
        """
        if isinstance(size_input, ImageSize):
            size_input = size_input.value
        size_str = size_input.lower().strip()
        wants_hd = quality in (ImageQuality.HD, ImageQuality.HIGH)

        if size_str in cls.GOOGLE_IMAGE_SIZE_ALIASES:
            tier = cls.GOOGLE_IMAGE_SIZE_ALIASES[size_str]
        elif (dims := cls._parse_explicit_pixels(size_str)) is not None:
            longest, pixels = max(dims), dims[0] * dims[1]
            if longest >= cls._GOOGLE_4K_MIN_EDGE or pixels >= cls._GOOGLE_4K_MIN_PIXELS:
                tier = "4K"
            elif longest >= cls._GOOGLE_2K_MIN_EDGE or pixels >= cls._GOOGLE_2K_MIN_PIXELS:
                tier = "2K"
            else:
                tier = "1K"
        else:
            tier = "2K" if wants_hd else "1K"

        if wants_hd and tier in ("512px", "1K"):
            tier = "2K"

        if model in cls.GOOGLE_1K_ONLY_IMAGE_MODELS:
            return "1K"
        if tier == "512px" and model not in cls.GOOGLE_512PX_IMAGE_MODELS:
            return "1K"
        return tier

    @classmethod
    def _calculate_aspect_ratio(cls, width: int, height: int) -> str:
        """Calculate simplified aspect ratio from dimensions."""
        # Map to common aspect ratios
        ratio = width / height
        common_ratios = [
            (1.0, "1:1"),
            (16/9, "16:9"),
            (9/16, "9:16"),
            (4/3, "4:3"),
            (3/4, "3:4"),
            (3/2, "3:2"),
            (2/3, "2:3"),
            (5/4, "5:4"),
            (4/5, "4:5"),
            (21/9, "21:9"),
        ]

        # Find closest match
        closest = min(common_ratios, key=lambda x: abs(x[0] - ratio))
        return closest[1]

    @classmethod
    def get_supported_formats(cls) -> dict[str, list[str]]:
        """Return documentation of all supported input formats."""
        return {
            "semantic_presets": list(cls.PRESETS_TO_ASPECT.keys()),
            "aspect_ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9"],
            "pixel_dimensions": list(cls.PIXELS_TO_ASPECT.keys()),
            "providers": {
                "openai_gpt_image": ["1024x1024", "1536x1024", "1024x1536"],
                "openai_dalle3": ["1024x1024", "1792x1024", "1024x1792"],
                "google_gemini": sorted(cls.GOOGLE_SUPPORTED_ASPECTS),
                "google_gemini_image_size": ["512px", "1K", "2K", "4K"],
            }
        }


@dataclass
class GeneratedImage:
    """A single generated image."""

    data: bytes | None = None       # Raw bytes (if available)
    base64: str | None = None       # Base64 encoded
    url: str | None = None          # URL (for hosted images)
    format: str = "png"
    revised_prompt: str | None = None


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embeddings: list[list[float]]   # List of vectors
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: int = 0
    dimensions: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ImageGenerationResponse:
    """Response from image generation."""

    images: list[GeneratedImage]    # List of generated images
    model: str
    provider: str
    latency_ms: int = 0
    prompt: str = ""
    revised_prompt: str | None = None  # Some models revise prompts
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AudioResponse:
    """Response from TTS or transcription."""

    content: str | bytes            # Text for STT, bytes for TTS
    model: str
    provider: str
    latency_ms: int = 0
    format: str | None = None       # Audio format for TTS
    language: str | None = None     # Detected language for STT
    timestamp: datetime = field(default_factory=datetime.utcnow)



@dataclass
class StreamChunk:
    """A chunk from a streaming response."""

    content: str
    done: bool = False
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Reasoning/"thinking" text, kept distinct from `content` so callers can
    # render it as a separate channel (e.g. an Anthropic `thinking` block) and
    # keep it out of the final answer. Populated only when the caller opts in
    # via chat_stream(stream_thinking=True); None otherwise. v1 ignores it.
    thinking: str | None = None


# =============================================================================
# Tool Definitions
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of a tool/function for the model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    strict: bool = False  # For strict schema validation


@dataclass
class ToolCall:
    """A tool call made by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str
    content: str
    is_error: bool = False


# =============================================================================
# Exceptions
# =============================================================================

class UnsupportedModalityError(Exception):
    """
    Raised when a file type is not supported by the model.

    Provides helpful information about:
    - Which file types are not supported
    - What the model does support
    - Suggestions for alternatives
    """

    def __init__(
        self,
        unsupported_types: list[FileType],
        model: str,
        provider: str,
        capabilities: ModelCapabilities,
    ):
        self.unsupported_types = unsupported_types
        self.model = model
        self.provider = provider
        self.capabilities = capabilities

        # Build helpful error message
        types_str = ", ".join(t.value for t in unsupported_types)
        supported = []
        if capabilities.supports_images:
            supported.append("images")
        if capabilities.supports_pdf:
            supported.append("PDFs")
        if capabilities.supports_audio:
            supported.append("audio")
        if capabilities.supports_video:
            supported.append("video")
        if capabilities.supports_documents:
            supported.append("documents")

        supported_str = ", ".join(supported) if supported else "text only"

        message = (
            f"Model '{model}' ({provider}) does not support: {types_str}. "
            f"This model supports: {supported_str}."
        )

        # Add suggestions
        suggestions = self._get_suggestions()
        if suggestions:
            message += f" Suggestions: {suggestions}"

        super().__init__(message)

    def _get_suggestions(self) -> str:
        """Generate helpful suggestions based on unsupported types."""
        suggestions = []

        for file_type in self.unsupported_types:
            if file_type == FileType.AUDIO:
                suggestions.append(
                    "For audio, consider using OpenAI's gpt-4o-audio-preview "
                    "or Google's Gemini models."
                )
            elif file_type == FileType.VIDEO:
                suggestions.append(
                    "For video, Google's Gemini models support video input."
                )
            elif file_type == FileType.PDF:
                suggestions.append(
                    "For PDFs, use OpenAI gpt-4o, Anthropic Claude, or Google Gemini."
                )

        return " ".join(suggestions)


# =============================================================================
# Base Adapter
# =============================================================================

class BaseAdapter(ABC):
    """
    Abstract base class for provider adapters.

    All provider adapters must implement this interface to ensure
    consistent behavior across different AI providers.

    Capabilities:
    - Chat completion (text)
    - Structured output (JSON schema)
    - Vision (image input)
    - Documents (PDFs, text files)
    - Audio (where supported)
    - Tool/function calling
    - Streaming

    File Handling:
    - Files are passed via the `files` key in messages
    - Each adapter converts FileAttachment to provider-specific format
    - Unsupported file types raise UnsupportedModalityError
    """

    # Whether this adapter's TokenUsage.input_tokens already CONTAINS the
    # cached_tokens subset. OpenAI (prompt_tokens) and Google
    # (promptTokenCount) include it; Anthropic's usage.input_tokens and
    # Bedrock Converse's inputTokens exclude cache reads. Billing subtracts
    # the cached portion only when it is included. (Verified against provider
    # usage docs 2026-09-01.)
    cached_tokens_included_in_input: bool = True

    provider_name: str

    # Serving-backend identity. "first_party" = the provider's own API;
    # cloud backends (azure, vertex, bedrock) override. Used by the factory's
    # priority resolution and surfaced in /v1/models.
    backend_name: str = "first_party"

    def serves(self, provider_model_id: str) -> bool:
        """Whether this backend can serve the given provider model ID.

        First-party adapters serve their whole family. Cloud backends
        override with their deployment/enablement map so priority
        resolution can fall through to the next backend.
        """
        return True

    def backend_model_id(self, provider_model_id: str) -> str:
        """Map a provider model ID to this backend's wire ID (identity for
        first-party; deployment name on Azure, dashed dateless IDs on
        Vertex Claude, prefixed IDs on Bedrock)."""
        return provider_model_id

    # =========================================================================
    # Model Capabilities
    # =========================================================================

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a specific model.

        Override in subclasses to provide model-specific capabilities.
        Default returns basic text-only capabilities.
        """
        return ModelCapabilities()

    def validate_file_attachments(
        self,
        model: str,
        files: list[FileAttachment],
    ) -> None:
        """
        Validate that the model supports all file types in the request.

        Raises:
            UnsupportedModalityError: If any file type is not supported
        """
        if not files:
            return

        capabilities = self.get_model_capabilities(model)
        file_types = [f.type for f in files]
        unsupported = capabilities.get_unsupported_types(file_types)

        if unsupported:
            raise UnsupportedModalityError(
                unsupported_types=unsupported,
                model=model,
                provider=self.provider_name,
                capabilities=capabilities,
            )

    # =========================================================================
    # Model-specific kwargs (pass-through with loud rejection)
    # =========================================================================

    # Fields the gateway owns — never overridable via model_kwargs, on any
    # adapter, even accept_all ones (byom/bedrock).
    RESERVED_KWARGS: frozenset = frozenset(
        {
            "model", "messages", "input", "system", "stream", "tools",
            "tool_choice", "max_tokens", "max_output_tokens", "temperature",
            "top_p", "stop", "response_format", "mcp_servers", "api_key",
            "extra_headers", "extra_body", "base_url", "n_retries", "timeout",
        }
    )
    # Curated provider-native keys this adapter forwards. Empty = none.
    SUPPORTED_KWARGS: frozenset = frozenset()
    # byom custom endpoints / bedrock additionalModelRequestFields: the
    # downstream surface is designed for arbitrary keys — accept everything
    # non-reserved and let the endpoint validate (still loud, just theirs).
    accept_all_kwargs: bool = False

    def validate_model_kwargs(self, model_kwargs: dict | None) -> None:
        """Reject (never drop) kwargs this adapter can't forward.

        Raises UnsupportedModelKwargsError listing the offending keys and the
        full supported set, before any provider call is made.
        """
        if not model_kwargs:
            return
        from mlpal_assistants_service.core.exceptions import (
            UnsupportedModelKwargsError,
        )

        keys = set(model_kwargs)
        reserved = keys & self.RESERVED_KWARGS
        if reserved:
            raise UnsupportedModelKwargsError(
                offending=sorted(reserved),
                allowed=sorted(self.SUPPORTED_KWARGS),
                provider=self.provider_name,
            )
        if self.accept_all_kwargs:
            return
        offending = keys - self.SUPPORTED_KWARGS
        if offending:
            raise UnsupportedModelKwargsError(
                offending=sorted(offending),
                allowed=sorted(self.SUPPORTED_KWARGS),
                provider=self.provider_name,
            )

    # =========================================================================
    # Core Chat Methods
    # =========================================================================

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict[str, Any] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> AdapterResponse:
        """
        Execute a chat completion.

        Args:
            model: Provider-specific model identifier
            messages: List of messages in standard format:
                [
                    {
                        "role": "user/assistant/system",
                        "content": "...",
                        "files": [FileAttachment, ...]  # Optional
                    }
                ]
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop: Stop sequences
            tools: List of tool definitions
            tool_choice: Tool choice strategy
            response_format: Response format spec (pass-through to provider)
            mcp_servers: MCP server configs (pass-through to provider)

        Returns:
            AdapterResponse with standardized fields

        Raises:
            ProviderError: On provider API errors
            UnsupportedModalityError: If files are not supported
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict[str, Any] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        stream_thinking: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Execute a streaming chat completion.

        Args:
            Same as chat(), plus:
            stream_thinking: When True and the model reasons, surface the
                reasoning/summary text as StreamChunk.thinking chunks (kept out
                of `content`). Opt-in so existing (v1) callers are unaffected.

        Yields:
            StreamChunk objects as content is generated
        """
        pass

    # =========================================================================
    # Generation Methods
    # =========================================================================

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for text inputs.

        Args:
            texts: List of text strings to embed
            model: Provider-specific embedding model ID (uses default if None)
            dimensions: Output dimensions (for models that support it)

        Returns:
            EmbeddingResponse with list of embedding vectors

        Raises:
            ProviderError: On provider API errors
            NotImplementedError: If provider doesn't support embeddings

        Example:
            >>> response = await adapter.embed(["Hello world", "How are you?"])
            >>> print(len(response.embeddings))  # 2
            >>> print(len(response.embeddings[0]))  # e.g., 3072 for text-embedding-3-large
        """
        pass

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: ImageSize | str = ImageSize.SQUARE,
        quality: ImageQuality = ImageQuality.STANDARD,
        n: int = 1,
        reference_images: list[FileAttachment] | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate images from text prompt, optionally with reference images.

        Supports:
        - Text-to-image: Just provide a prompt
        - Image-to-image: Provide prompt + reference_images

        Args:
            prompt: Text description of the image to generate
            model: Provider-specific image model ID (uses default if None)
            size: Output image size (ImageSize enum or string like "1024x1024")
            quality: Image quality level
            n: Number of images to generate (1-4, provider-dependent)
            reference_images: Optional list of reference images for image-to-image

        Returns:
            ImageGenerationResponse with generated images

        Raises:
            ProviderError: On provider API errors
            NotImplementedError: If provider doesn't support image generation

        Example:
            >>> # Text-to-image
            >>> response = await adapter.generate_image(
            ...     prompt="A futuristic city at sunset",
            ...     size=ImageSize.LANDSCAPE
            ... )
            >>> image = response.images[0]
            >>>
            >>> # Image-to-image with reference
            >>> refs = [FileAttachment(type=FileType.IMAGE, source=FileSource.PATH, data="sketch.png")]
            >>> response = await adapter.generate_image(
            ...     prompt="Transform this sketch into a realistic image",
            ...     reference_images=refs
            ... )
        """
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio: FileAttachment,
        model: str | None = None,
        language: str | None = None,
    ) -> AudioResponse:
        """
        Transcribe audio to text (Speech-to-Text).

        Args:
            audio: Audio file attachment (supports mp3, wav, m4a, etc.)
            model: Provider-specific transcription model ID (uses default if None)
            language: Optional language code (e.g., "en", "es") for better accuracy

        Returns:
            AudioResponse with transcribed text as content

        Raises:
            ProviderError: On provider API errors
            NotImplementedError: If provider doesn't support transcription

        Example:
            >>> audio = FileAttachment(
            ...     type=FileType.AUDIO,
            ...     source=FileSource.PATH,
            ...     data="/path/to/recording.mp3"
            ... )
            >>> response = await adapter.transcribe(audio)
            >>> print(response.content)  # Transcribed text
        """
        pass

    @abstractmethod
    async def text_to_speech(
        self,
        text: str,
        model: str | None = None,
        voice: str = "alloy",
        format: str = "mp3",
        speed: float = 1.0,
    ) -> AudioResponse:
        """
        Convert text to speech audio (Text-to-Speech).

        Args:
            text: Text to convert to speech
            model: Provider-specific TTS model ID (uses default if None)
            voice: Voice ID/name to use (provider-specific)
            format: Audio format (mp3, wav, opus, etc.)
            speed: Playback speed (0.25 to 4.0, default 1.0)

        Returns:
            AudioResponse with audio bytes as content

        Raises:
            ProviderError: On provider API errors
            NotImplementedError: If provider doesn't support TTS

        Example:
            >>> response = await adapter.text_to_speech(
            ...     text="Hello, this is a test.",
            ...     voice="nova",
            ...     format="mp3"
            ... )
            >>> with open("speech.mp3", "wb") as f:
            ...     f.write(response.content)
        """
        pass

    # =========================================================================
    # Tool Execution Support
    # =========================================================================

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_executor: Any,  # Callable to execute tools
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AdapterResponse:
        """
        Execute chat with automatic tool execution loop.

        This is a helper method that handles the tool calling loop:
        1. Call model with tools
        2. If model returns tool calls, execute them
        3. Send tool results back to model
        4. Repeat until model returns final response or max iterations

        Args:
            model: Model identifier
            messages: Initial messages
            tools: Available tools
            tool_executor: Async callable that takes ToolCall and returns ToolResult
            max_iterations: Maximum tool calling iterations
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Final AdapterResponse after tool execution completes
        """
        current_messages = list(messages)
        total_usage = TokenUsage()
        total_latency = 0

        for _ in range(max_iterations):
            response = await self.chat(
                model=model,
                messages=current_messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Accumulate usage
            total_usage.input_tokens += response.usage.input_tokens
            total_usage.output_tokens += response.usage.output_tokens
            total_usage.total_tokens += response.usage.total_tokens
            total_latency += response.latency_ms

            # Check if there are tool calls
            if not response.tool_calls:
                # No more tool calls, return final response
                response.usage = total_usage
                response.latency_ms = total_latency
                return response

            # Add assistant message with tool calls
            current_messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            })

            # Execute tools and add results
            for tool_call in response.tool_calls:
                tc = ToolCall(
                    id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["arguments"],
                )
                result = await tool_executor(tc)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })

        # Max iterations reached
        response.usage = total_usage
        response.latency_ms = total_latency
        return response

    # =========================================================================
    # File Handling Utilities
    # =========================================================================

    def extract_files_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[FileAttachment]:
        """
        Extract all file attachments from messages.

        Used for validation before making API calls.
        """
        files = []
        for msg in messages:
            msg_files = msg.get("files", [])
            for f in msg_files:
                if isinstance(f, FileAttachment):
                    files.append(f)
                elif isinstance(f, dict):
                    # Convert dict to FileAttachment
                    files.append(FileAttachment(
                        type=FileType(f.get("type", "other")),
                        source=FileSource(f.get("source", "base64")),
                        data=f.get("data", ""),
                        mime_type=f.get("mime_type"),
                        filename=f.get("filename"),
                    ))
        return files

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def convert_tools(
        self,
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """
        Convert tool definitions to provider-specific format.

        Override in subclasses if provider needs different format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                },
            }
            for tool in tools
        ]

    async def health_check(self) -> bool:
        """Check if provider API is accessible."""
        return True

    @property
    def supports_vision(self) -> bool:
        """Whether this adapter supports vision (image input)."""
        return False

    @property
    def supports_audio(self) -> bool:
        """Whether this adapter supports audio input."""
        return False

    @property
    def supports_video(self) -> bool:
        """Whether this adapter supports video input."""
        return False

    @property
    def supports_pdf(self) -> bool:
        """Whether this adapter supports PDF input."""
        return False

    @property
    def supports_tools(self) -> bool:
        """Whether this adapter supports tool/function calling."""
        return False

    @property
    def supports_structured_output(self) -> bool:
        """Whether this adapter supports structured output."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this adapter supports streaming."""
        return True
