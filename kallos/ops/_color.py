"""sRGB <-> linear-light conversions (IEC 61966-2-1).

Lives in `kallos.ops` because Clarity, Sharpen, Vibrance, and Denoise all
need to round-trip through OpenCV color spaces — and OpenCV's RGB<->Lab
and RGB<->HSV conversions assume sRGB-encoded input. Our pipeline tensors
are in *linear* RGB, so any op that uses cv2.cvtColor must encode to
sRGB first and decode back to linear after.

Underscore prefix marks this as ops-internal — it isn't part of the
public op contract.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]


def linear_to_srgb(x: Image) -> Image:
    """Linear-light [0,1] -> sRGB-encoded [0,1] via the IEC piecewise curve."""
    a = 0.055
    threshold = 0.0031308
    x = np.clip(x, 0.0, 1.0)
    lo = x * 12.92
    hi = (1 + a) * np.power(x, 1.0 / 2.4) - a
    return np.where(x <= threshold, lo, hi).astype(np.float32)


def srgb_to_linear(x: Image) -> Image:
    """sRGB-encoded [0,1] -> linear-light [0,1] via the IEC piecewise curve."""
    a = 0.055
    threshold = 0.04045
    lo = x / 12.92
    hi = ((x + a) / (1 + a)) ** 2.4
    return np.where(x <= threshold, lo, hi).astype(np.float32)
