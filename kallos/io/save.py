"""Save a linear-RGB float32 tensor back to disk or memory.

Handles sRGB gamma encoding, 8-bit/16-bit dithering, ICC profile tagging,
and EXIF preservation. The format choice drives the bit depth:

- JPEG → 8-bit, quality 95, 4:4:4 subsampling, ICC + EXIF
- PNG  → 8-bit, ICC + EXIF
- TIFF → 16-bit, LZW-compressed, ICC + EXIF
- HEIC → 8-bit, ICC + EXIF (requires pillow-heif)

Two entry points: `save_image(...)` writes to a file path; `serialize_image(...)`
returns bytes (used by the FastAPI /save endpoint to stream a real browser
download instead of writing to a hidden folder on disk).

We synthesize a clean sRGB v4 ICC profile via PIL.ImageCms at import time so
wide-gamut displays (macOS, modern Windows, most phones) render colors
correctly. Without the tag, viewers guess — usually right, sometimes not.
"""

from __future__ import annotations

import io
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageCms

# Register HEIF/HEIC support with Pillow at import time.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_AVAILABLE = True
except Exception:  # noqa: BLE001 — falls back gracefully if the wheel is missing
    _HEIC_AVAILABLE = False

Image = NDArray[np.float32]


class SaveFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    HEIC = "heic"


# Filename extensions and MIME types for each format.
EXTENSION = {
    SaveFormat.JPEG: ".jpg",
    SaveFormat.PNG: ".png",
    SaveFormat.TIFF: ".tif",
    SaveFormat.HEIC: ".heic",
}
MIME_TYPE = {
    SaveFormat.JPEG: "image/jpeg",
    SaveFormat.PNG: "image/png",
    SaveFormat.TIFF: "image/tiff",
    SaveFormat.HEIC: "image/heic",
}


# Build an sRGB profile once at import; it's ~3KB and shared across all saves.
_SRGB_ICC: bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def save_image(
    img: Image,
    path: str | Path,
    fmt: SaveFormat,
    *,
    exif_bytes: bytes | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Write a linear-RGB tensor to disk in the chosen format. Returns the path."""
    out = Path(path)
    with out.open("wb") as f:
        _write_to(img, f, fmt, exif_bytes=exif_bytes, jpeg_quality=jpeg_quality)
    return out


def serialize_image(
    img: Image,
    fmt: SaveFormat,
    *,
    exif_bytes: bytes | None = None,
    jpeg_quality: int = 95,
) -> bytes:
    """Encode a linear-RGB tensor to a byte string in the chosen format.

    Used by the HTTP layer to stream a download response without touching disk.
    """
    buf = io.BytesIO()
    _write_to(img, buf, fmt, exif_bytes=exif_bytes, jpeg_quality=jpeg_quality)
    return buf.getvalue()


def _write_to(
    img: Image,
    sink: BinaryIO,
    fmt: SaveFormat,
    *,
    exif_bytes: bytes | None,
    jpeg_quality: int,
) -> None:
    """Encode `img` into the given binary sink. Sink can be a file or BytesIO."""
    encoded = _linear_to_srgb(img)

    if fmt == SaveFormat.TIFF:
        u16 = _to_uint16_dithered(encoded)
        # ICC travels as TIFF tag 34675 (InterColorProfile).
        extratags: list[tuple[int, int, int, bytes, bool]] = [
            (34675, 7, len(_SRGB_ICC), _SRGB_ICC, False),
        ]
        tifffile.imwrite(
            sink,
            u16,
            photometric="rgb",
            compression="zlib",
            extratags=extratags,
            metadata=None,
        )
        return

    u8 = _to_uint8_dithered(encoded)
    pil = PILImage.fromarray(u8, mode="RGB")
    kwargs: dict[str, object] = {"icc_profile": _SRGB_ICC}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    if fmt == SaveFormat.JPEG:
        kwargs["quality"] = jpeg_quality
        kwargs["subsampling"] = 0  # 4:4:4 — keep sharp edges crisp
        pil.save(sink, format="JPEG", **kwargs)
    elif fmt == SaveFormat.PNG:
        pil.save(sink, format="PNG", **kwargs)
    elif fmt == SaveFormat.HEIC:
        if not _HEIC_AVAILABLE:
            raise RuntimeError(
                "HEIC support not installed. Run `pip install pillow-heif`."
            )
        kwargs["quality"] = jpeg_quality
        pil.save(sink, format="HEIF", **kwargs)


def _linear_to_srgb(x: Image) -> Image:
    """Convert linear values to sRGB gamma-encoded values. IEC 61966-2-1."""
    a = 0.055
    threshold = 0.0031308
    x = np.clip(x, 0.0, 1.0)
    lo = x * 12.92
    hi = (1 + a) * np.power(x, 1.0 / 2.4) - a
    return np.where(x <= threshold, lo, hi).astype(np.float32)


def _to_uint8_dithered(x: Image) -> NDArray[np.uint8]:
    """Round to 8-bit with triangular dither to avoid banding in smooth gradients."""
    rng = np.random.default_rng(0)
    n = rng.uniform(-0.5, 0.5, size=x.shape) + rng.uniform(-0.5, 0.5, size=x.shape)
    scaled = x * 255.0 + n.astype(np.float32)
    return np.clip(scaled, 0, 255).round().astype(np.uint8)


def _to_uint16_dithered(x: Image) -> NDArray[np.uint16]:
    """16-bit dither — the noise floor is so far below visible that this is
    barely necessary, but a single LSB of triangular noise still helps any
    downstream tools that quantize further."""
    rng = np.random.default_rng(0)
    n = rng.uniform(-0.5, 0.5, size=x.shape) + rng.uniform(-0.5, 0.5, size=x.shape)
    scaled = x * 65535.0 + n.astype(np.float32)
    return np.clip(scaled, 0, 65535).round().astype(np.uint16)

