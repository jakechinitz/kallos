"""Load an image file into kallos' canonical format.

Canonical format: numpy.float32, shape (H, W, 3), values in [0, 1],
**linear sRGB primaries** (not gamma encoded). EXIF orientation is applied
to the pixels here so downstream code never has to think about it.

RAF (Fuji RAW) is decoded via libraw with gamma=(1,1) so we get true linear
data; WB is left at unity and applied later in the pipeline.

JPEG/PNG are decoded via Pillow, gamma-decoded to linear, EXIF orientation
baked into the pixels.

TIFF is decoded via tifffile so we handle 16-bit RGB correctly (Pillow's
RGB mode is 8-bit only).

We also extract a tiny `ExifSummary` (ISO, shutter speed, focal length) so
Auto Enhance can be camera-aware — noisy at high ISO, suggest deblur when the
shutter speed violates the 1/focal-length reciprocal rule, and so on.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import piexif
import tifffile
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

# Register HEIF/HEIC support with Pillow at import time. Safe to skip if the
# wheel is unavailable — HEIC simply won't be in SUPPORTED_EXTENSIONS.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_EXTENSIONS = {".heic", ".heif"}
except Exception:  # noqa: BLE001
    _HEIC_EXTENSIONS = set()

Image = NDArray[np.float32]

RAW_EXTENSIONS = {".raf", ".dng", ".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
SUPPORTED_EXTENSIONS = (
    RAW_EXTENSIONS | TIFF_EXTENSIONS | _HEIC_EXTENSIONS | {".jpg", ".jpeg", ".png"}
)

MAX_MEGAPIXELS = 100  # reject inputs above this; some pixel-shift RAFs are huge


@dataclass
class ExifSummary:
    """The handful of EXIF fields Auto Enhance actually uses.

    All fields are optional — many JPEGs lack EXIF, and RAFs we read minimally.
    Values are pre-parsed into plain numbers so callers don't need to know
    EXIF's rational-number quirks.
    """

    iso: int | None = None                    # e.g. 200, 1600, 6400
    shutter_seconds: float | None = None      # e.g. 1/250 -> 0.004
    focal_length_mm: float | None = None      # native focal length
    focal_length_35mm: float | None = None    # 35mm-equivalent (when present)
    camera_make: str | None = None
    camera_model: str | None = None


@dataclass
class LoadResult:
    """What a successful load gives back."""

    image: Image
    is_raw: bool
    camera_wb: tuple[float, float, float, float] | None
    daylight_wb: tuple[float, float, float, float] | None
    exif_bytes: bytes | None
    exif: ExifSummary
    source_name: str


def load_image(path: str | Path, *, max_long_edge: int | None = None) -> LoadResult:
    """Load an image from disk. Optionally downsample to `max_long_edge` pixels."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    if ext in RAW_EXTENSIONS:
        return _load_raw(p, max_long_edge)
    if ext in TIFF_EXTENSIONS:
        return _load_tiff(p, max_long_edge)
    return _load_pillow(p, max_long_edge)


def load_image_from_bytes(
    data: bytes, filename: str, *, max_long_edge: int | None = None
) -> LoadResult:
    """Same as `load_image` but from an in-memory byte stream (e.g. an upload)."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")
    if ext in RAW_EXTENSIONS:
        raise ValueError("RAW files must be loaded from a path, not bytes")
    if ext in TIFF_EXTENSIONS:
        # tifffile accepts a BytesIO directly.
        arr = tifffile.imread(io.BytesIO(data))
        return _finish_tiff(arr, filename, max_long_edge, raw_exif=None)

    pil = PILImage.open(io.BytesIO(data))
    return _finish_pillow(pil, filename, max_long_edge)


def _load_raw(path: Path, max_long_edge: int | None) -> LoadResult:
    import rawpy  # type: ignore[import-untyped]

    exif = _exif_summary_from_path(path)

    with rawpy.imread(str(path)) as raw:
        camera_wb = tuple(float(x) for x in raw.camera_whitebalance[:4])
        daylight_wb = tuple(float(x) for x in raw.daylight_whitebalance[:4])
        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.sRGB,
        )

    _check_megapixels(rgb.shape)
    img = rgb.astype(np.float32) / 65535.0
    if max_long_edge is not None:
        img = _downsample(img, max_long_edge)

    return LoadResult(
        image=img,
        is_raw=True,
        camera_wb=(camera_wb[0], camera_wb[1], camera_wb[2], camera_wb[3]),
        daylight_wb=(daylight_wb[0], daylight_wb[1], daylight_wb[2], daylight_wb[3]),
        exif_bytes=None,
        exif=exif,
        source_name=path.stem,
    )


def _load_pillow(path: Path, max_long_edge: int | None) -> LoadResult:
    pil = PILImage.open(path)
    return _finish_pillow(pil, str(path), max_long_edge)


def _load_tiff(path: Path, max_long_edge: int | None) -> LoadResult:
    arr = tifffile.imread(str(path))
    raw_exif = _try_extract_tiff_exif(path)
    return _finish_tiff(arr, str(path), max_long_edge, raw_exif=raw_exif)


def _try_extract_tiff_exif(path: Path) -> bytes | None:
    try:
        with PILImage.open(path) as pil:
            return pil.info.get("exif")
    except Exception:  # noqa: BLE001
        return None


def _finish_tiff(
    arr: NDArray[np.generic],
    source: str,
    max_long_edge: int | None,
    *,
    raw_exif: bytes | None,
) -> LoadResult:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Unsupported TIFF channel count: {arr.shape[2]}")

    _check_megapixels(arr.shape)

    # Normalize to float32 [0,1] based on the source bit depth.
    if arr.dtype == np.uint16:
        f = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        f = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        f = arr.astype(np.float32)
    else:
        f = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)

    if max_long_edge is not None:
        f = _downsample(f, max_long_edge)

    f = _srgb_to_linear(f)

    exif_summary, exif_bytes = _parse_exif(raw_exif)

    return LoadResult(
        image=f,
        is_raw=False,
        camera_wb=None,
        daylight_wb=None,
        exif_bytes=exif_bytes,
        exif=exif_summary,
        source_name=Path(source).stem,
    )


def _finish_pillow(pil: PILImage.Image, source: str, max_long_edge: int | None) -> LoadResult:
    # Bake EXIF orientation into the pixels so we deal in upright tensors only.
    pil = ImageOps.exif_transpose(pil)
    raw_exif = pil.info.get("exif")

    if pil.mode not in ("RGB", "RGBA", "L"):
        pil = pil.convert("RGB")
    if pil.mode in ("RGBA", "L"):
        pil = pil.convert("RGB")

    _check_megapixels((pil.height, pil.width))

    if max_long_edge is not None:
        pil.thumbnail((max_long_edge, max_long_edge), PILImage.LANCZOS)

    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = _srgb_to_linear(arr)

    exif_summary, exif_bytes = _parse_exif(raw_exif)

    return LoadResult(
        image=arr,
        is_raw=False,
        camera_wb=None,
        daylight_wb=None,
        exif_bytes=exif_bytes,
        exif=exif_summary,
        source_name=Path(source).stem,
    )


def _parse_exif(raw_exif: bytes | None) -> tuple[ExifSummary, bytes | None]:
    """Pull ISO/shutter/focal/camera fields out of EXIF, return summary + bytes
    (with the orientation tag stripped since we've baked it into pixels)."""
    if not raw_exif:
        return ExifSummary(), None
    try:
        exif_dict = piexif.load(raw_exif)
    except Exception:  # noqa: BLE001 — bad EXIF shouldn't block opening
        return ExifSummary(), None

    zeroth = exif_dict.get("0th", {})
    exif = exif_dict.get("Exif", {})

    summary = ExifSummary(
        iso=_int(exif.get(piexif.ExifIFD.ISOSpeedRatings)),
        shutter_seconds=_rational(exif.get(piexif.ExifIFD.ExposureTime)),
        focal_length_mm=_rational(exif.get(piexif.ExifIFD.FocalLength)),
        focal_length_35mm=_int(exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)),
        camera_make=_str(zeroth.get(piexif.ImageIFD.Make)),
        camera_model=_str(zeroth.get(piexif.ImageIFD.Model)),
    )

    # Strip orientation; we've baked it into the pixels already.
    zeroth.pop(piexif.ImageIFD.Orientation, None)
    try:
        out_bytes = piexif.dump(exif_dict)
    except Exception:  # noqa: BLE001
        out_bytes = None
    return summary, out_bytes


def _exif_summary_from_path(path: Path) -> ExifSummary:
    """Best-effort EXIF extraction for RAW files. Many RAFs carry standard EXIF
    in the JPEG preview / metadata block that piexif can read."""
    try:
        data = path.read_bytes()
        summary, _ = _parse_exif(_extract_exif_bytes(data))
        return summary
    except Exception:  # noqa: BLE001
        return ExifSummary()


def _extract_exif_bytes(data: bytes) -> bytes | None:
    """Find an EXIF APP1 segment in arbitrary bytes (works for JPEG previews
    embedded in RAFs). Returns the payload starting with the Exif\\0\\0 header
    that piexif expects, or None if not found."""
    marker = b"\xff\xe1"
    idx = data.find(marker)
    while idx != -1:
        size = int.from_bytes(data[idx + 2 : idx + 4], "big")
        segment = data[idx + 4 : idx + 2 + size]
        if segment.startswith(b"Exif\x00\x00"):
            return segment
        idx = data.find(marker, idx + 2)
    return None


def _int(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _rational(v: object) -> float | None:
    if v is None:
        return None
    if isinstance(v, tuple) and len(v) == 2 and v[1]:
        return float(v[0]) / float(v[1])
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _str(v: object) -> str | None:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore").strip("\x00 ")
    if isinstance(v, str):
        return v.strip()
    return None


def _check_megapixels(shape: tuple[int, ...]) -> None:
    mp = (shape[0] * shape[1]) / 1_000_000
    if mp > MAX_MEGAPIXELS:
        raise ValueError(
            f"Image too large: {mp:.0f}MP exceeds the {MAX_MEGAPIXELS}MP limit."
        )


def _downsample(img: Image, max_long_edge: int) -> Image:
    h, w = img.shape[:2]
    if max(h, w) <= max_long_edge:
        return img
    scale = max_long_edge / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if img.ndim == 3:
        channels = []
        for c in range(img.shape[2]):
            cpil = PILImage.fromarray(
                np.clip(img[:, :, c] * 65535.0, 0, 65535).astype(np.uint16), mode="I;16"
            )
            cpil = cpil.resize((new_w, new_h), PILImage.LANCZOS)
            channels.append(np.asarray(cpil, dtype=np.float32) / 65535.0)
        return np.stack(channels, axis=-1)
    cpil = PILImage.fromarray(np.clip(img * 65535.0, 0, 65535).astype(np.uint16), mode="I;16")
    cpil = cpil.resize((new_w, new_h), PILImage.LANCZOS)
    return np.asarray(cpil, dtype=np.float32) / 65535.0


def _srgb_to_linear(x: Image) -> Image:
    """Convert sRGB gamma-encoded values to linear light. IEC 61966-2-1."""
    a = 0.055
    threshold = 0.04045
    lo = x / 12.92
    hi = ((x + a) / (1 + a)) ** 2.4
    return np.where(x <= threshold, lo, hi).astype(np.float32)
