"""Load an image file into kallos' canonical format.

Canonical format: numpy.float32, shape (H, W, 3), values in [0, 1],
**linear sRGB primaries** (not gamma encoded). EXIF orientation is applied
to the pixels here so downstream code never has to think about it.

RAF (Fuji RAW) is decoded via libraw with gamma=(1,1) so we get true linear
data; WB is left at unity and applied later in the pipeline.

JPEG/PNG/TIFF are decoded via Pillow, gamma-decoded to linear, and have their
EXIF orientation baked into the pixels.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import piexif
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

Image = NDArray[np.float32]

RAW_EXTENSIONS = {".raf", ".dng", ".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MAX_MEGAPIXELS = 100  # reject inputs above this; some pixel-shift RAFs are huge


@dataclass
class LoadResult:
    """What a successful load gives back."""

    image: Image
    is_raw: bool
    camera_wb: tuple[float, float, float, float] | None
    daylight_wb: tuple[float, float, float, float] | None
    exif_bytes: bytes | None
    source_name: str


def load_image(path: str | Path, *, max_long_edge: int | None = None) -> LoadResult:
    """Load an image from disk. Optionally downsample to `max_long_edge` pixels.

    `max_long_edge` lets the caller request a preview-sized tensor cheaply
    (Pillow / rawpy do the resize in their decoders when possible).
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    if ext in RAW_EXTENSIONS:
        return _load_raw(p, max_long_edge)
    return _load_pillow(p, max_long_edge)


def load_image_from_bytes(
    data: bytes, filename: str, *, max_long_edge: int | None = None
) -> LoadResult:
    """Same as `load_image` but from an in-memory byte stream (e.g. an upload).

    For RAW we still need a file on disk for rawpy; the caller should write the
    bytes to a temp file first. For Pillow-supported formats this works inline.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")
    if ext in RAW_EXTENSIONS:
        raise ValueError("RAW files must be loaded from a path, not bytes")

    pil = PILImage.open(io.BytesIO(data))
    return _finish_pillow(pil, filename, max_long_edge)


def _load_raw(path: Path, max_long_edge: int | None) -> LoadResult:
    # Imported lazily so non-RAW users don't pay the libraw startup cost.
    import rawpy  # type: ignore[import-untyped]

    with rawpy.imread(str(path)) as raw:
        camera_wb = tuple(float(x) for x in raw.camera_whitebalance[:4])
        daylight_wb = tuple(float(x) for x in raw.daylight_whitebalance[:4])
        # 16-bit, linear (gamma=(1,1)), no auto-bright, unity WB, sRGB primaries.
        # This gives us a clean linear tensor with no surprises baked in.
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
        exif_bytes=None,  # piexif doesn't handle RAF; we write minimal EXIF on save
        source_name=path.stem,
    )


def _load_pillow(path: Path, max_long_edge: int | None) -> LoadResult:
    pil = PILImage.open(path)
    return _finish_pillow(pil, str(path), max_long_edge)


def _finish_pillow(pil: PILImage.Image, source: str, max_long_edge: int | None) -> LoadResult:
    # Bake EXIF orientation into the pixels so we deal in upright tensors only.
    pil = ImageOps.exif_transpose(pil)

    exif_bytes: bytes | None = None
    try:
        raw_exif = pil.info.get("exif")
        if raw_exif:
            # Round-trip through piexif to strip the orientation tag (we baked it in).
            exif_dict = piexif.load(raw_exif)
            exif_dict.get("0th", {}).pop(piexif.ImageIFD.Orientation, None)
            exif_bytes = piexif.dump(exif_dict)
    except Exception:  # noqa: BLE001 — bad EXIF shouldn't block opening the image
        exif_bytes = None

    if pil.mode not in ("RGB", "RGBA", "L"):
        pil = pil.convert("RGB")
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    if pil.mode == "L":
        pil = pil.convert("RGB")

    _check_megapixels((pil.height, pil.width))

    if max_long_edge is not None:
        pil.thumbnail((max_long_edge, max_long_edge), PILImage.LANCZOS)

    arr = np.asarray(pil, dtype=np.float32) / 255.0
    # PNG/JPEG ship gamma-encoded sRGB; linearize for the pipeline.
    arr = _srgb_to_linear(arr)

    return LoadResult(
        image=arr,
        is_raw=False,
        camera_wb=None,
        daylight_wb=None,
        exif_bytes=exif_bytes,
        source_name=Path(source).stem,
    )


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
    # Resample via Pillow (LANCZOS) — better than naive numpy striding.
    scale = max_long_edge / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Pillow expects uint8/uint16; round-trip via float32 sRGB-linear is fine because
    # both source and target are linear floats.
    pil = PILImage.fromarray(np.clip(img * 65535.0, 0, 65535).astype(np.uint16), mode="I;16")
    pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
    out = np.asarray(pil, dtype=np.float32) / 65535.0
    # PIL's I;16 is single-channel; reattach RGB by resizing each channel.
    # Simpler: do it per-channel.
    if img.shape[2] == 3:
        channels = []
        for c in range(3):
            cpil = PILImage.fromarray(
                np.clip(img[:, :, c] * 65535.0, 0, 65535).astype(np.uint16), mode="I;16"
            )
            cpil = cpil.resize((new_w, new_h), PILImage.LANCZOS)
            channels.append(np.asarray(cpil, dtype=np.float32) / 65535.0)
        out = np.stack(channels, axis=-1)
    return out


def _srgb_to_linear(x: Image) -> Image:
    """Convert sRGB gamma-encoded values to linear light. IEC 61966-2-1."""
    a = 0.055
    threshold = 0.04045
    lo = x / 12.92
    hi = ((x + a) / (1 + a)) ** 2.4
    return np.where(x <= threshold, lo, hi).astype(np.float32)
