"""Auto Enhance: pick sane slider values from image statistics.

The heuristics are deliberately conservative — Auto should produce a
flattering baseline you'd want to tweak from, not a maxed-out look.

What we measure:
- Median luminance: tells us if the image is under/over exposed.
- Histogram percentiles (1st, 99th): tells us how much dynamic range is used.
- Color cast: tells us how much warmth correction to apply (negative on a
  warm image, positive on a cool one — we *correct* casts, not amplify them).
- Estimated noise (local std in flat areas): tells us how much denoise.

What we set:
- Brightness: push median toward 0.18 (mid-gray).
- Contrast: small boost if the histogram doesn't reach the endpoints.
- Warmth: counter-balance the color cast.
- Vibrance: small fixed bump (+15) — almost every photo benefits.
- Sharpen: fixed +20 — slightly above default. Most images are slightly soft.
- Denoise: scaled by estimated noise, but capped low so we don't kill text.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from kallos.state import Settings

Image = NDArray[np.float32]

# Photo industry mid-gray reference in linear light.
MID_GRAY = 0.18


def auto_settings(img: Image) -> Settings:
    """Compute Auto Enhance slider values for a linear-RGB tensor."""
    lum = _luminance(img)

    # --- brightness: aim median luminance at MID_GRAY ---
    med = float(np.median(lum)) + 1e-6
    stops_needed = math.log2(MID_GRAY / med)
    # Clamp to a tasteful range; auto shouldn't massively re-expose.
    stops_needed = max(-1.5, min(1.5, stops_needed))
    brightness = (stops_needed / 2.0) * 100.0

    # --- contrast: how compressed is the histogram? ---
    p1, p99 = np.percentile(lum, [1, 99])
    span = float(p99 - p1)
    # A full-range image spans ~1.0. Bump contrast when span is short.
    contrast = max(0.0, min(40.0, (1.0 - span) * 80.0))

    # --- warmth: counter the dominant color cast ---
    means = img.reshape(-1, 3).mean(axis=0)
    rb_ratio = float(means[0] - means[2])  # >0 means image is warm
    # Counter the cast: pull warmth toward zero of the cast.
    warmth = max(-25.0, min(25.0, -rb_ratio * 200.0))

    # --- denoise: estimate noise level in flat areas ---
    noise = _estimate_noise(lum)
    # Empirically: noise ~0.005 is clean, ~0.02 is grainy ISO 3200ish.
    denoise = max(0.0, min(30.0, (noise - 0.005) * 1500.0))

    return Settings(
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        warmth=round(warmth, 1),
        vibrance=15.0,
        sharpen=20.0,
        denoise=round(denoise, 1),
        ai_deblur=False,
    )


def _luminance(img: Image) -> NDArray[np.float32]:
    # Rec. 709 coefficients on linear RGB.
    return (
        0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    ).astype(np.float32)


def _estimate_noise(lum: NDArray[np.float32]) -> float:
    """Estimate noise as the local std in the flattest areas of the image.

    Sample 32x32 patches, take the std of each, return the median of the
    bottom 10% (the flattest patches — areas where any variance is noise).
    """
    h, w = lum.shape
    if h < 64 or w < 64:
        return float(lum.std())
    patch = 32
    ys = np.arange(0, h - patch, patch)
    xs = np.arange(0, w - patch, patch)
    stds = []
    for y in ys:
        for x in xs:
            stds.append(float(lum[y:y + patch, x:x + patch].std()))
    stds.sort()
    flat = stds[: max(1, len(stds) // 10)]
    return float(np.median(flat))
