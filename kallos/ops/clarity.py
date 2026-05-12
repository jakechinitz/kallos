"""Clarity — large-radius midtone local-contrast enhancement.

This is the algorithm that makes iPhone shots feel "crisp when you zoom in".
It is *not* sharpening. Sharpen uses a tiny radius (~1px) and crispens
existing edges; Clarity uses a large radius (~20px) and adds "presence" to
textures (wood grain, fabric, book spines, brick) without producing the
visible halos that aggressive sharpening creates.

Algorithm:
1. Encode linear RGB → sRGB (cv2's Lab conversion assumes sRGB-encoded input;
   passing linear values yields wrong Lab coordinates and a tonal shift).
2. Convert to Lab, work on the L channel only — keeps colors stable.
3. Blur L with a large gaussian (sigma proportional to image size so the
   effect looks the same on the 1280px preview and the 24MP save).
4. detail = L - blurred  (the local-contrast residual)
5. Restrict the effect to midtones with a soft gaussian mask centered at
   L=50 — keeps deep shadows readable and bright highlights from blooming.
6. L' = L + strength * detail * midtone_mask
7. Lab → sRGB → linear back to the pipeline.

Negative amounts soften (less local contrast — useful for portraits).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from kallos.ops._color import linear_to_srgb, srgb_to_linear

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

    # cv2's RGB<->Lab assumes sRGB-encoded input. Encode now, decode at the end.
    srgb = linear_to_srgb(img)
    lab = cv2.cvtColor(srgb, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]  # 0..100

    h, w = L.shape
    sigma = _REFERENCE_SIGMA * max(h, w) / _REFERENCE_LONG_EDGE
    base = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = L - base

    # Midtone mask: 1 at L=50, fades smoothly to ~0 at L=0 and L=100.
    midtone = np.exp(-((L - 50.0) / 35.0) ** 2).astype(np.float32)

    # Cap effect at 0.6 so even +100 doesn't go cartoony.
    strength = (amount / 100.0) * 0.6
    L_new = L + strength * detail * midtone

    lab[:, :, 0] = np.clip(L_new, 0.0, 100.0)
    out_srgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return srgb_to_linear(np.clip(out_srgb, 0.0, 1.0))
