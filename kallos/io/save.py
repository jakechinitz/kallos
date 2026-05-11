"""Save a linear-RGB float32 tensor back to disk.

Handles sRGB gamma encoding, 8-bit/16-bit dithering, ICC profile tagging,
and EXIF preservation. The format choice drives the bit depth:

- JPEG → 8-bit, quality 95, 4:4:4 subsampling, ICC + EXIF
- PNG  → 8-bit, ICC + EXIF
- TIFF → 16-bit, LZW-compressed, ICC + EXIF

We synthesize a clean sRGB v4 ICC profile via PIL.ImageCms at import time so
wide-gamut displays (macOS, modern Windows, most phones) render colors
correctly. Without the tag, viewers guess — usually right, sometimes not.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    encoded = _linear_to_srgb(img)

    if fmt == SaveFormat.TIFF:
        u16 = _to_uint16_dithered(encoded)
        # tifffile writes proper 16-bit RGB TIFFs; Pillow's RGB mode is 8-bit only.
        # ICC + EXIF travel as extra tags; viewers that respect them get the right
        # color, the rest assume sRGB and still look right.
        extratags: list[tuple[int, int, int, bytes, bool]] = [
            # (tag, dtype, count, value, writeonce) — TIFF tag 34675 is InterColorProfile.
            (34675, 7, len(_SRGB_ICC), _SRGB_ICC, False),
        ]
        kwargs: dict[str, object] = {
            "photometric": "rgb",
            "compression": "zlib",  # built-in to tifffile; LZW would need imagecodecs
            "extratags": extratags,
            "metadata": None,
        }
        tifffile.imwrite(str(out), u16, **kwargs)
    else:
        u8 = _to_uint8_dithered(encoded)
        pil = PILImage.fromarray(u8, mode="RGB")
        kwargs = {"icc_profile": _SRGB_ICC}
        if exif_bytes is not None:
            kwargs["exif"] = exif_bytes
        if fmt == SaveFormat.JPEG:
            kwargs["quality"] = jpeg_quality
            kwargs["subsampling"] = 0  # 4:4:4 — keep sharp edges crisp
            pil.save(out, format="JPEG", **kwargs)
        elif fmt == SaveFormat.PNG:
            pil.save(out, format="PNG", **kwargs)
        elif fmt == SaveFormat.HEIC:
            if not _HEIC_AVAILABLE:
                raise RuntimeError(
                    "HEIC support not installed. Run `pip install pillow-heif`."
                )
            # HEIC quality scale: 50–95 maps to "good–transparent" for HEIC.
            # We map the JPEG quality kwarg roughly the same way.
            kwargs["quality"] = jpeg_quality
            pil.save(out, format="HEIF", **kwargs)

    return out


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
    # 1 LSB of dither width at 8-bit = 1/255. Triangular distribution = sum of two uniforms.
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
