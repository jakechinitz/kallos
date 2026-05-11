"""Non-local means denoising via OpenCV.

Operates on 8-bit data internally (OpenCV's NL-means is 8-bit only), with a
float round-trip. Default strength is intentionally low because aggressive
denoising destroys text and fine detail — the exact thing this app exists
to preserve.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]


def apply_denoise(img: Image, amount: float) -> Image:
    """NL-means denoise. amount in [0, 100]."""
    if amount <= 0:
        return img

    # cv2 NL-means is 8-bit; convert linear-RGB-float to gamma-encoded uint8 first.
    # We keep this conversion local to denoise — outside this op everything stays linear.
    enc = _linear_to_srgb_u8(img)

    # Map amount 0..100 to NL-means h parameter 0..15. h=10 is a common default.
    h_lum = float(amount * 0.15)
    h_col = float(amount * 0.15)

    denoised = cv2.fastNlMeansDenoisingColored(
        enc,
        None,
        h=h_lum,
        hColor=h_col,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return _srgb_u8_to_linear(denoised)


def _linear_to_srgb_u8(x: Image) -> NDArray[np.uint8]:
    a = 0.055
    threshold = 0.0031308
    x = np.clip(x, 0.0, 1.0)
    lo = x * 12.92
    hi = (1 + a) * np.power(x, 1.0 / 2.4) - a
    encoded = np.where(x <= threshold, lo, hi)
    return np.clip(encoded * 255.0, 0, 255).round().astype(np.uint8)


def _srgb_u8_to_linear(x: NDArray[np.uint8]) -> Image:
    f = x.astype(np.float32) / 255.0
    a = 0.055
    threshold = 0.04045
    lo = f / 12.92
    hi = ((f + a) / (1 + a)) ** 2.4
    return np.where(f <= threshold, lo, hi).astype(np.float32)
