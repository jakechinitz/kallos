"""Vibrance: saturation that protects skin tones and already-saturated colors.

Works in HSV. Boosts low-saturation pixels more than high-saturation ones, and
attenuates the boost in the skin-tone hue range (~15°–45° in HSV degrees).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]


def apply_vibrance(img: Image, amount: float) -> Image:
    """Vibrance in [-100, +100]. Positive boosts dull colors, negative desaturates."""
    if amount == 0:
        return img

    # cv2.cvtColor expects float32 in [0,1] for sRGB-linear-ish data.
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]  # 0..360
    s = hsv[:, :, 1]  # 0..1

    strength = amount / 100.0  # -1..+1

    # Protect already-saturated pixels: weight by (1 - s) so dull colors get most of the lift.
    weight = 1.0 - s
    # Protect skin tones: hues near 15..45 degrees get a softer touch.
    skin_mask = np.exp(-(((h - 30.0) / 25.0) ** 2))  # gaussian centered at 30°
    weight = weight * (1.0 - 0.6 * skin_mask)

    delta = strength * weight
    new_s = np.clip(s + delta, 0.0, 1.0)
    hsv[:, :, 1] = new_s

    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0.0, None).astype(np.float32)
