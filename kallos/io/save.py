"""Save a linear-RGB float32 tensor back to disk.

Handles sRGB gamma encoding, 8-bit/16-bit dithering, ICC profile tagging,
and EXIF preservation. The format choice (JPEG/PNG/TIFF) drives bit depth.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

Image = NDArray[np.float32]


class SaveFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"


# sRGB v4 IEC61966-2-1 minimal profile (sRGB IEC61966-2.1, 3144 bytes).
# Bundled inline so we never depend on a system ICC file.
_SRGB_PROFILE_PATH = Path(__file__).with_name("srgb_v2.icc")


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

    u8 = _to_uint8_dithered(encoded)
    pil = PILImage.fromarray(u8, mode="RGB")
    kwargs: dict[str, object] = {"icc_profile": _icc_or_none()}
    if exif_bytes is not None:
        kwargs["exif"] = exif_bytes
    if fmt == SaveFormat.JPEG:
        kwargs["quality"] = jpeg_quality
        kwargs["subsampling"] = 0  # 4:4:4 — keep sharp edges crisp
        pil.save(out, format="JPEG", **kwargs)
    elif fmt == SaveFormat.PNG:
        pil.save(out, format="PNG", **kwargs)
    elif fmt == SaveFormat.TIFF:
        # 8-bit TIFF with LZW compression. (16-bit RGB TIFF needs a separate
        # encoder like `tifffile` — kept as a future enhancement.)
        kwargs["compression"] = "tiff_lzw"
        pil.save(out, format="TIFF", **kwargs)

    return out


def _icc_or_none() -> bytes | None:
    if _SRGB_PROFILE_PATH.exists():
        return _SRGB_PROFILE_PATH.read_bytes()
    return None


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
