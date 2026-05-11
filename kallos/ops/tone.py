"""Tonal operations: brightness (exposure) and contrast (S-curve).

Both run in linear light. Brightness is a true exposure adjustment in stops;
contrast is a smooth S-curve around mid-gray that preserves shadow and
highlight detail better than a linear contrast stretch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]

# Mid-gray pivot for the contrast S-curve, in linear light. 0.18 is the photo
# industry's "18% gray" reference — visually a comfortable mid-tone.
MID_GRAY = 0.18


def apply_brightness(img: Image, amount: float) -> Image:
    """Exposure adjustment. amount in [-100, +100] maps to ±2 EV."""
    if amount == 0:
        return img
    stops = (amount / 100.0) * 2.0
    gain = float(2.0**stops)
    return np.clip(img * gain, 0.0, None).astype(np.float32)


def apply_contrast(img: Image, amount: float) -> Image:
    """Soft S-curve in linear light. amount in [-100, +100].

    Implemented as a smoothstep-like sigmoid around mid-gray. Negative amount
    flattens the curve (less contrast); positive amount steepens it.
    """
    if amount == 0:
        return img
    strength = amount / 100.0  # -1..+1
    # Map strength to a "slope" multiplier at the pivot. >1 steepens, <1 flattens.
    slope = float(2.0**strength)  # ~0.5x at -100, 2x at +100

    # Convert to log-luminance space around the pivot so the curve is symmetric.
    eps = 1e-6
    pivot = MID_GRAY
    ratio = (img + eps) / (pivot + eps)
    out = pivot * np.power(ratio, slope)
    return np.clip(out, 0.0, None).astype(np.float32)
