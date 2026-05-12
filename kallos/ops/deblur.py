"""AI deblur via ONNX runtime.

The primary (and currently only) path is a pretrained motion-deblur model
(NAFNet trained on the REDS dataset). Drop the ONNX file into
~/.cache/kallos/models/nafnet_motion_deblur.onnx and the toggle lights up.

Without a model present, AI Deblur is a no-op. We used to fall back to
Wiener deconvolution with a fixed-PSF guess (11-px horizontal motion) but
that assumption is wrong for almost all real handheld shake, and the
resulting ringing + amplified noise made images visibly worse than the
input. Better to leave the photo alone than degrade it under a 'deblur'
label. The Wiener routine is still in this file as `_wiener_fallback` for
anyone who wants to opt in via code, but nothing calls it by default.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from kallos.models.registry import OnnxSession, try_load_deblur_model

Image = NDArray[np.float32]


def has_deblur_model() -> bool:
    """Cheap check: is a real ONNX model available? Used by Auto Enhance to
    decide whether to enable the AI Deblur toggle."""
    return try_load_deblur_model() is not None


def apply_deblur(img: Image, *, use_ai: bool = True) -> Image:
    """Run deblur. If no AI model is present, returns input unchanged
    (Wiener fallback was removed because wrong-PSF deconvolution hurts more
    than it helps)."""
    if not use_ai:
        return img
    session = try_load_deblur_model()
    if session is None:
        return img
    return _run_onnx_tiled(img, session)


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
    """Wiener deconvolution with a fixed linear-motion PSF guess. NOT USED by
    default — see module docstring. Left here as a reference implementation
    for anyone experimenting with classical deblur.
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


def _wiener_channel(
    channel: NDArray[np.float32], psf: NDArray[np.float32], k: float
) -> NDArray[np.float32]:
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
