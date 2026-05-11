"""The image processing pipeline. Reads top to bottom like a recipe.

Two entry points:
- `render_preview` for live slider updates (no AI deblur, uses cache).
- `render_final` for Save (full quality, applies AI deblur if toggled).

The order matters. The reasoning, briefly:
- WB before exposure: WB is per-channel gain; tonal ops assume neutral gray.
- Denoise before deblur: deblur sharpens noise into something worse.
- Sharpen after contrast: contrast reshapes edge gradients.
- Vibrance last: keeps it from amplifying noise introduced earlier.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kallos.ops import color, denoise, sharpen, tone, wb
from kallos.ops.deblur import apply_deblur
from kallos.state import Session, Settings

Image = NDArray[np.float32]


def render_preview(session: Session) -> Image:
    """Run the slider pipeline. Uses the cached intermediate when possible."""
    s = session.settings

    # The cached intermediate is the tensor after WB + brightness + denoise —
    # the expensive ops. Tonal/color/sharpen tweaks recompute cheaply from it.
    if session.intermediate is None or _cache_dirty(session, s):
        x = wb.apply_preset(session.original, s.wb_preset, camera_wb=session.camera_wb)
        x = wb.apply_warmth(x, s.warmth)
        x = tone.apply_brightness(x, s.brightness)
        x = denoise.apply_denoise(x, s.denoise)
        session.intermediate = x
        session.intermediate_for = _intermediate_key(s)
    else:
        x = session.intermediate

    x = tone.apply_contrast(x, s.contrast)
    x = sharpen.apply_sharpen(x, s.sharpen)
    x = color.apply_vibrance(x, s.vibrance)
    return x


def render_final(session: Session, *, source: Image | None = None) -> Image:
    """Full-quality render for Save. Runs AI deblur if the toggle is on.

    Pass `source` to render at a different resolution than the cached preview
    (e.g. the full-res tensor).
    """
    s = session.settings
    base = source if source is not None else session.original

    x = wb.apply_preset(base, s.wb_preset, camera_wb=session.camera_wb)
    x = wb.apply_warmth(x, s.warmth)
    x = tone.apply_brightness(x, s.brightness)
    x = denoise.apply_denoise(x, s.denoise)
    if s.ai_deblur:
        x = apply_deblur(x, use_ai=True)
    x = tone.apply_contrast(x, s.contrast)
    x = sharpen.apply_sharpen(x, s.sharpen)
    x = color.apply_vibrance(x, s.vibrance)
    return x


def _intermediate_key(s: Settings) -> Settings:
    """The slice of settings the cached intermediate depends on."""
    return Settings(
        wb_preset=s.wb_preset,
        warmth=s.warmth,
        brightness=s.brightness,
        denoise=s.denoise,
    )


def _cache_dirty(session: Session, current: Settings) -> bool:
    if session.intermediate_for is None:
        return True
    key = _intermediate_key(current)
    return (
        key.wb_preset != session.intermediate_for.wb_preset
        or key.warmth != session.intermediate_for.warmth
        or key.brightness != session.intermediate_for.brightness
        or key.denoise != session.intermediate_for.denoise
    )


def encode_for_display(img: Image) -> NDArray[np.uint8]:
    """Gamma-encode a linear tensor to 8-bit sRGB for the browser preview.

    No dithering here — preview JPEG already adds enough noise that banding
    isn't visible. Dithering is reserved for the final Save path.
    """
    a = 0.055
    threshold = 0.0031308
    x = np.clip(img, 0.0, 1.0)
    lo = x * 12.92
    hi = (1 + a) * np.power(x, 1.0 / 2.4) - a
    encoded = np.where(x <= threshold, lo, hi)
    return np.clip(encoded * 255.0, 0, 255).round().astype(np.uint8)
