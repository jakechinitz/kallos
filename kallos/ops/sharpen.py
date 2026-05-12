"""Edge-aware unsharp mask in the L channel of Lab.

Working on L (luminance) only avoids color fringing that you get from a naive
3-channel unsharp mask. The mask is gated by a soft edge map so flat areas
(sky, walls) don't get noisy.

cv2's RGB<->Lab assumes sRGB-encoded input, so we encode-then-decode around
the conversion to avoid the tonal shift you'd get from passing linear values.

The radius (gaussian sigma) auto-adapts to the amount: low amounts use a
slightly larger radius for "presence" sharpening (landscapes, portraits);
high amounts use a small radius for crisp letterform sharpening (text, fine
detail). One slider, two regimes.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from kallos.ops._color import linear_to_srgb, srgb_to_linear

Image = NDArray[np.float32]


def apply_sharpen(img: Image, amount: float, *, radius: float | None = None) -> Image:
    """Unsharp mask. amount in [0, 100].

    radius (gaussian sigma in pixels) defaults to an amount-adaptive value:
    radius=1.2 at amount=0 fading via a sqrt curve to radius=0.7 at amount=100,
    so the tight (text-crisp) regime kicks in around the middle of the slider.
    Pass an explicit radius to override.
    """
    if amount <= 0:
        return img

    if radius is None:
        # Sqrt curve so the tight (text-crisp) radius regime kicks in around the
        # middle of the slider rather than only at the top.
        t = max(0.0, min(1.0, amount / 100.0))
        radius = 1.2 - 0.5 * (t**0.5)

    # Encode to sRGB so cv2's Lab conversion is correct, then decode back at the end.
    srgb = linear_to_srgb(img)
    lab = cv2.cvtColor(srgb, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]  # 0..100

    blurred = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    detail = L - blurred

    # Soft edge gate: amplify edges, attenuate flat areas. Uses the magnitude
    # of the local gradient as a confidence map.
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    cap = float(np.median(grad)) + 3.0 * float(np.median(np.abs(grad - np.median(grad))) + 1e-6)
    edge_gate = np.clip(grad / max(cap, 1e-6), 0.0, 1.0)

    # amount=100 -> strength 1.5 added detail on edges. Tuned empirically.
    strength = (amount / 100.0) * 1.5

    l_sharp = L + strength * detail * edge_gate
    lab[:, :, 0] = np.clip(l_sharp, 0.0, 100.0)

    out_srgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return srgb_to_linear(np.clip(out_srgb, 0.0, 1.0))
