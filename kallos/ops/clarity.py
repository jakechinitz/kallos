"""Clarity — large-radius midtone local-contrast enhancement.

This is the algorithm that makes iPhone shots feel "crisp when you zoom in".
It is *not* sharpening. Sharpen uses a tiny radius (~1px) and crispens
existing edges; Clarity uses a large radius (~20px) and adds "presence" to
textures (wood grain, fabric, book spines, brick) without producing the
visible halos that aggressive sharpening creates.

Algorithm:
1. Convert to Lab, work on the L (luminance) channel only — keeps colors stable.
2. Blur L with a large gaussian (sigma proportional to image size so the
   effect looks the same on the 1280px preview and the 24MP save).
3. detail = L - blurred  (the local-contrast residual)
4. Restrict the effect to midtones with a soft gaussian mask centered at
   L=50 — keeps deep shadows readable and bright highlights from blooming.
5. L' = L + strength * detail * midtone_mask

Negative amounts soften (less local contrast — useful for portraits).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]

# Reference sigma at the preview resolution (1280px long edge). The actual
# sigma scales linearly with image dimensions so the perceptual effect is
# consistent between preview and full-res save.
_REFERENCE_SIGMA = 20.0
_REFERENCE_LONG_EDGE = 1280.0


def apply_clarity(img: Image, amount: float) -> Image:
    """Local-contrast clarity. amount in [-100, +100]. 0 is identity."""
    if amount == 0:
        return img

    bounded = np.clip(img, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(bounded, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]  # 0..100

    h, w = L.shape
    sigma = _REFERENCE_SIGMA * max(h, w) / _REFERENCE_LONG_EDGE
    base = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = L - base

    # Midtone mask: 1 at L=50, fades smoothly to ~0 at L=0 and L=100.
    midtone = np.exp(-((L - 50.0) / 35.0) ** 2).astype(np.float32)

    # Amount -> strength. Cap at 0.6 so even +100 doesn't go cartoony.
    strength = (amount / 100.0) * 0.6
    L_new = L + strength * detail * midtone

    lab[:, :, 0] = np.clip(L_new, 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return np.clip(out, 0.0, None).astype(np.float32)
