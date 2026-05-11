"""Edge-aware unsharp mask in the L channel of Lab.

Working on L (luminance) only avoids color fringing that you get from a naive
3-channel unsharp mask. The mask is gated by a soft edge map so flat areas
(sky, walls) don't get noisy.

The radius (gaussian sigma) auto-adapts to the amount: low amounts use a
slightly larger radius for "presence" sharpening (landscapes, portraits);
high amounts use a small radius for crisp letterform sharpening (text, fine
detail). One slider, two regimes.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]


def apply_sharpen(img: Image, amount: float, *, radius: float | None = None) -> Image:
    """Unsharp mask. amount in [0, 100].

    radius (gaussian sigma in pixels) defaults to an amount-adaptive value:
    radius=1.2 at amount=0 fading smoothly to radius=0.7 at amount=100.
    Pass an explicit radius to override.
    """
    if amount <= 0:
        return img

    if radius is None:
        # Smooth linear interpolation: 1.2 px (presence) → 0.7 px (text-crisp).
        t = max(0.0, min(1.0, amount / 100.0))
        radius = 1.2 - 0.5 * t

    # OpenCV's RGB<->Lab assumes sRGB-encoded RGB input. We're in linear sRGB,
    # but for sharpening purposes the L channel is still a useful proxy for
    # perceived luminance and the round-trip preserves the original pixels.
    # We clip to [0,1] since cv2 Lab conversion expects bounded values.
    bounded = np.clip(img, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(bounded, cv2.COLOR_RGB2Lab)
    l = lab[:, :, 0]  # 0..100

    # Blurred version for the unsharp mask.
    blurred = cv2.GaussianBlur(l, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    detail = l - blurred

    # Soft edge gate: amplify edges, attenuate flat areas. Uses the magnitude
    # of the local gradient as a confidence map.
    gx = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # Normalize to roughly 0..1 using a soft cap (median + a few MADs).
    cap = float(np.median(grad)) + 3.0 * float(np.median(np.abs(grad - np.median(grad))) + 1e-6)
    edge_gate = np.clip(grad / max(cap, 1e-6), 0.0, 1.0)

    # amount=100 -> strength 1.5 added detail on edges. Tuned empirically.
    strength = (amount / 100.0) * 1.5

    l_sharp = l + strength * detail * edge_gate
    lab[:, :, 0] = np.clip(l_sharp, 0.0, 100.0)

    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(out, 0.0, None).astype(np.float32)
