"""Auto Enhance — detail only.

Philosophy: trust the camera. Fuji's JPEG engine already nailed exposure,
white balance, color rendering, and tone curve. Don't fight any of that.
Auto's only job is to add the one thing the camera *didn't* do: extra
detail that survives pinch-zoom.

What Auto touches:
- Clarity: large-radius midtone microcontrast. The iPhone-look ingredient.
- Sharpen: small edge-aware top-up.
- AI Deblur: auto-on when EXIF says the shutter was too slow for the
  focal length (handheld shake recovery, the one thing the camera can't fix).

What Auto leaves alone:
- Brightness, Contrast, Warmth, Vibrance, Denoise, WB preset — all zero.
  These are scene-of-capture decisions that the camera made on purpose;
  re-doing them risks ruining the photo to fix something that isn't broken.

If a particular shot needs exposure or color help, that's what the
Advanced sliders are for — but Auto won't surprise you with them.
"""

from __future__ import annotations

from kallos.io.load import ExifSummary
from kallos.state import Settings


def auto_settings(
    img,  # noqa: ARG001 — kept in the signature so callers can pass it; not used today
    *,
    exif: ExifSummary | None = None,
    is_raw: bool = False,
) -> Settings:
    """Compute detail-only Auto Enhance values.

    `is_raw` toggles between two intensity profiles:
    - RAW (camera did nothing): both Clarity and Sharpen lean in.
    - JPEG (camera already sharpened): Clarity does the heavy lifting,
      Sharpen stays light to avoid double-sharpening edges.
    """
    exif = exif or ExifSummary()

    if is_raw:
        clarity = 12.0
        sharpen = 12.0
    else:
        clarity = 25.0
        sharpen = 5.0

    return Settings(
        # Detail-only: everything else stays at the camera's choice.
        brightness=0.0,
        contrast=0.0,
        warmth=0.0,
        vibrance=0.0,
        clarity=clarity,
        sharpen=sharpen,
        denoise=0.0,
        ai_deblur=_looks_shaky(exif),
        wb_preset="as_shot",
    )


def _looks_shaky(exif: ExifSummary) -> bool:
    """1 / (focal_length × 1.5) reciprocal rule — when violated, the shot is
    a strong handheld-shake candidate and AI Deblur should auto-enable."""
    f = exif.focal_length_35mm or (
        exif.focal_length_mm * 1.5 if exif.focal_length_mm else None
    )
    if f is None or exif.shutter_seconds is None:
        return False
    safe_shutter = 1.0 / (f * 1.5)
    return exif.shutter_seconds > safe_shutter
