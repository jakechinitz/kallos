"""White balance. Applies a preset (camera multipliers or fixed daylight),
then layers a continuous Warmth offset on top.

WB is a per-channel gain in linear light. It must run before tonal ops because
tonal ops assume the gray balance is correct.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]

# Approximate channel multipliers for common color temperatures (relative to D65 = 1,1,1).
# Derived from blackbody spectra under sRGB primaries; close enough for a "warmth" slider.
_PRESETS: dict[str, tuple[float, float, float]] = {
    "as_shot": (1.0, 1.0, 1.0),       # caller substitutes camera_wb if available
    "auto": (1.0, 1.0, 1.0),          # gray-world handled separately
    "daylight": (1.0, 1.0, 1.0),      # ~5500K, the reference
    "tungsten": (1.5, 1.0, 0.55),     # ~3200K — pull blue down, push red up
    "shade": (0.85, 1.0, 1.15),       # ~7500K — opposite of tungsten
}


def apply_preset(
    img: Image,
    preset: str,
    *,
    camera_wb: tuple[float, float, float, float] | None = None,
) -> Image:
    """Apply a WB preset. For 'as_shot' on a RAW, uses the camera multipliers."""
    if preset == "as_shot" and camera_wb is not None:
        # rawpy WB is RGBG; collapse to RGB by averaging the two greens.
        r, g1, b, g2 = camera_wb
        green = (g1 + g2) / 2.0
        gains = _normalize_to_green((r, green, b))
    elif preset == "auto":
        gains = _gray_world_gains(img)
    else:
        gains = _PRESETS.get(preset, (1.0, 1.0, 1.0))

    return _apply_gains(img, gains)


def apply_warmth(img: Image, amount: float) -> Image:
    """Continuous warmth shift in [-100, +100]. Positive = warmer, negative = cooler."""
    if amount == 0:
        return img
    # ±100 corresponds to ~±1500K. Implement as a smooth blend between two endpoints.
    t = amount / 100.0  # -1..+1
    # Warm endpoint (red up, blue down); cool endpoint (red down, blue up).
    warm = (1.15, 1.0, 0.85)
    cool = (0.85, 1.0, 1.15)
    target = warm if t > 0 else cool
    strength = abs(t)
    gains = tuple(1.0 + (target[i] - 1.0) * strength for i in range(3))
    return _apply_gains(img, gains)


def _apply_gains(img: Image, gains: tuple[float, float, float]) -> Image:
    out = img.copy()
    for c in range(3):
        out[:, :, c] *= gains[c]
    return np.clip(out, 0.0, None).astype(np.float32)


def _normalize_to_green(gains: tuple[float, float, float]) -> tuple[float, float, float]:
    g = gains[1] or 1.0
    return (gains[0] / g, 1.0, gains[2] / g)


def _gray_world_gains(img: Image) -> tuple[float, float, float]:
    """Simple gray-world: scale each channel so its mean equals the overall mean."""
    means = img.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    if target <= 0:
        return (1.0, 1.0, 1.0)
    return (target / max(means[0], 1e-6), target / max(means[1], 1e-6), target / max(means[2], 1e-6))
