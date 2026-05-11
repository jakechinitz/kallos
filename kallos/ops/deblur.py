"""AI deblur via ONNX runtime, with a classical Wiener-deconvolution fallback.

The primary path is a pretrained motion-deblur model (NAFNet trained on the
REDS dataset). The model weights are downloaded on first use to
~/.cache/kallos/models/ and run through onnxruntime CPU.

The fallback is a Wiener deconvolution with an estimated linear-motion PSF.
It's deterministic, fast, and surprisingly OK for short handheld shake; it
runs automatically when the ONNX model can't be loaded (offline, download
failed, or the user prefers no-ML).

This op is heavier than the others, so it's invoked only on the Save path
(or via a small "preview deblur" button on a center crop).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from kallos.models.registry import OnnxSession, try_load_deblur_model

Image = NDArray[np.float32]


def apply_deblur(img: Image, *, use_ai: bool = True) -> Image:
    """Run deblur on the full image. If the AI model is unavailable, falls
    back to Wiener deconvolution with an estimated PSF."""
    if use_ai:
        session = try_load_deblur_model()
        if session is not None:
            return _run_onnx_tiled(img, session)
    return _wiener_fallback(img)


def _run_onnx_tiled(
    img: Image,
    session: OnnxSession,
    *,
    tile: int = 512,
    overlap: int = 32,
) -> Image:
    """Tiled inference with feathered seams. Handles arbitrary input sizes."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    step = tile - overlap
    feather = _feather_mask(tile, overlap)

    for y in range(0, max(1, h), step):
        for x in range(0, max(1, w), step):
            y0, x0 = y, x
            y1, x1 = min(y + tile, h), min(x + tile, w)
            patch = img[y0:y1, x0:x1, :]
            ph, pw = patch.shape[:2]
            if ph < tile or pw < tile:
                padded = np.zeros((tile, tile, 3), dtype=np.float32)
                padded[:ph, :pw, :] = patch
                patch = padded
            result = session.run(patch)
            f = feather[:ph, :pw, None] if (ph < tile or pw < tile) else feather[..., None]
            out[y0:y1, x0:x1, :] += result[:ph, :pw, :] * f
            weight[y0:y1, x0:x1, :] += f
            if x1 >= w:
                break
        if y1 >= h:
            break

    return (out / np.maximum(weight, 1e-6)).astype(np.float32)


def _feather_mask(tile: int, overlap: int) -> NDArray[np.float32]:
    """1D ramp on each side, multiplied to a 2D mask, for smooth tile blending."""
    ramp = np.ones(tile, dtype=np.float32)
    if overlap > 0:
        edge = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return ramp[:, None] * ramp[None, :]


def _wiener_fallback(img: Image, *, angle_deg: float = 0.0, length: int = 11) -> Image:
    """Wiener deconvolution with an estimated linear-motion PSF.

    angle/length default to a small horizontal blur — a reasonable guess for
    handheld shots without knowing the actual motion. In practice this gives
    a mild deblur that still helps text legibility.
    """
    psf = _motion_psf(length=length, angle_deg=angle_deg)
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = _wiener_channel(img[:, :, c], psf, k=0.01)
    return np.clip(out, 0.0, None).astype(np.float32)


def _motion_psf(length: int, angle_deg: float) -> NDArray[np.float32]:
    psf = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    angle_rad = math.radians(angle_deg)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * math.cos(angle_rad)))
        y = int(round(center + offset * math.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1.0
    s = psf.sum()
    return psf / s if s > 0 else psf


def _wiener_channel(channel: NDArray[np.float32], psf: NDArray[np.float32], k: float) -> NDArray[np.float32]:
    h, w = channel.shape
    # Pad PSF to image size and center via fftshift.
    psf_padded = np.zeros_like(channel)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf
    psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

    H = np.fft.fft2(psf_padded)
    G = np.fft.fft2(channel)
    H_conj = np.conj(H)
    denom = H * H_conj + k
    F = (H_conj / denom) * G
    return np.real(np.fft.ifft2(F)).astype(np.float32)


