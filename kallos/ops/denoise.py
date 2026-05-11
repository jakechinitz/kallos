"""Non-local means denoise with an edge-protective blend.

NL-means already does a decent job preserving edges, but it still softens
text and fine textures slightly. We push it further: compute an edge mask
from the luminance gradient and blend the denoised output back toward the
original wherever a real edge lives. Flat regions get the full denoise;
edges (text, hair, leaves) stay exactly as captured.

This is the closest classical analogue to "this part of the image needs
enhancement, this part doesn't" — local, automatic, no segmentation model.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]


def apply_denoise(img: Image, amount: float) -> Image:
    """NL-means denoise + edge-protective blend. amount in [0, 100]."""
    if amount <= 0:
        return img

    # cv2 NL-means is 8-bit; convert linear-RGB-float to gamma-encoded uint8 first.
    enc = _linear_to_srgb_u8(img)

    # Map amount 0..100 to h parameter 0..15.
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
    denoised_lin = _srgb_u8_to_linear(denoised)

    # Edge-protect: where there's a real edge, keep more of the original.
    mask = _edge_mask(img)[..., None]  # (H, W, 1)
    return (denoised_lin * (1.0 - mask) + img * mask).astype(np.float32)


def _edge_mask(img: Image) -> NDArray[np.float32]:
    """Soft luminance-edge mask in [0, 1]. 0 = flat (denoise fully), 1 = edge
    (leave original). Normalized against the image's own edge distribution so
    it works for both low-contrast and high-contrast scenes."""
    lum = (
        0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    ).astype(np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # Scale to roughly 0..1 using a robust upper bound (median + 3 MADs).
    med = float(np.median(grad))
    mad = float(np.median(np.abs(grad - med)))
    cap = med + 3.0 * mad + 1e-6
    mask = np.clip(grad / cap, 0.0, 1.0)
    # Smooth so the boundary between "denoise" and "preserve" isn't jagged.
    return cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=1.5)


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
