"""Auto Enhance — opinionated, Leica-flavored.

Leica's processing ethos (insofar as a list can capture it):
- Restraint over pump. Colors pure, not boosted. Saturation conservative.
- Highlights are sacred. Sky and skin should never plasticize.
- Microcontrast for presence, not global crunch.
- Preserve the light's character. Golden-hour stays golden; tungsten just
  loses its edge.
- The camera knows things. ISO, shutter speed, focal length — read them.
- Noise that lives in the texture is information; smooth flats, never edges.

We translate that into the six sliders kallos exposes:

- Brightness: push median luminance toward mid-gray, but back off when
  highlights would clip. Better an under-exposed-feeling result than a
  blown sky.
- Contrast: small bump, scaled by how compressed the histogram is.
- Warmth: detect color cast, but only partially correct it. A warm scene
  should still feel warm — we just take the curse off.
- Vibrance: small fixed lift (+8). Restrained on purpose. Skin tones are
  protected by the vibrance op itself.
- Sharpen: modest (+12) by default, less if the shot looks shake-blurred,
  more if it's loaded with fine edges (text, foliage).
- Denoise: driven by ISO when EXIF is present, otherwise by measured noise
  in flat patches. Capped low so text doesn't melt.
- AI Deblur: auto-enabled when the shutter speed violates the
  1/(focal_length × 1.5) reciprocal-rule guideline — a strong hint of
  handheld shake.
- WB preset: "auto" only when the image lacks camera_wb (i.e. not a RAW
  As-Shot) AND we have no EXIF context.

Each decision is small and explainable. None of this is a model; all of it
is photographer-common-sense in code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from kallos.io.load import ExifSummary
from kallos.state import Settings

Image = NDArray[np.float32]

# Photo industry mid-gray reference in linear light.
MID_GRAY = 0.18

# How close to clipping (in linear light) before we treat the highlights as
# "at risk" and back off on brightness boosts.
HIGHLIGHT_DANGER = 0.85


def auto_settings(img: Image, *, exif: ExifSummary | None = None) -> Settings:
    """Compute Leica-flavored Auto Enhance slider values."""
    exif = exif or ExifSummary()
    stats = _measure(img)

    brightness = _choose_brightness(stats)
    contrast = _choose_contrast(stats)
    warmth = _choose_warmth(stats)
    sharpen = _choose_sharpen(stats, exif)
    denoise = _choose_denoise(stats, exif)
    deblur = _choose_deblur(exif)

    return Settings(
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        warmth=round(warmth, 1),
        vibrance=8.0,       # restrained — Leica colors are pure, not pumped
        sharpen=round(sharpen, 1),
        denoise=round(denoise, 1),
        ai_deblur=deblur,
        wb_preset="as_shot",
    )


# ---------------------------------------------------------------------------
# Image stats — one pass, used by every decision below.
# ---------------------------------------------------------------------------

@dataclass
class _Stats:
    lum_median: float
    lum_p1: float
    lum_p99: float
    lum_span: float
    flat_noise: float
    edge_density: float
    illuminant: tuple[float, float, float]  # normalized R, G, B of the implicit light


def _measure(img: Image) -> _Stats:
    lum = _luminance(img)
    p1, p50, p99 = np.percentile(lum, [1, 50, 99])
    return _Stats(
        lum_median=float(p50),
        lum_p1=float(p1),
        lum_p99=float(p99),
        lum_span=float(p99 - p1),
        flat_noise=_flat_patch_noise(lum),
        edge_density=_edge_density(lum),
        illuminant=_shades_of_gray(img, p=6.0),
    )


# ---------------------------------------------------------------------------
# Per-slider decisions.
# ---------------------------------------------------------------------------

def _choose_brightness(s: _Stats) -> float:
    """Aim median at mid-gray, but never at the cost of clipping highlights.

    Leica's rule of thumb: protect the highlights, you can always rescue the
    shadows. So we compute the exposure needed to hit mid-gray, then clamp
    based on how close the 99th percentile is to clipping.
    """
    want = math.log2(MID_GRAY / max(s.lum_median, 1e-4))  # stops
    # Highlight-aware ceiling: don't push p99 past 0.95.
    if s.lum_p99 > 0.01:
        ceiling = math.log2(0.95 / s.lum_p99)
    else:
        ceiling = 3.0
    stops = min(want, ceiling)
    # Be conservative on negative exposure too — most users prefer slightly
    # bright over slightly dark.
    stops = max(-1.0, min(1.5, stops))
    return (stops / 2.0) * 100.0


def _choose_contrast(s: _Stats) -> float:
    """Small bump, more on flat images. Capped at +25 — anything bigger is
    HDR-look territory and not Leica."""
    flatness = max(0.0, 1.0 - s.lum_span)  # 0 = full range, 1 = totally flat
    return min(25.0, flatness * 50.0)


def _choose_warmth(s: _Stats) -> float:
    """Detect the cast in the implicit illuminant and only partially correct.

    Mistake to avoid: neutralizing a warm scene. Golden-hour light is *the
    look*. We pull it toward neutral by ~40% so the photo loses the harshest
    edge of the cast without losing its character.
    """
    r, _g, b = s.illuminant
    # Negative if image is warm-cast (more red), positive if cool-cast.
    cast = float(r - b)
    # Map a cast of ±0.15 (a meaningfully strong tint) to ±25 on the slider.
    # Multiply by -0.40 (partial correction, opposite sign).
    return max(-25.0, min(25.0, -cast * 0.40 * 167.0))


def _choose_sharpen(s: _Stats, exif: ExifSummary) -> float:
    """Modest baseline; adjust for shake risk and edge density.

    - Shake-blurred shots get *less* sharpen — sharpening blur is amplifying
      the wrong thing; deblur handles that case.
    - Edge-dense images (text, foliage, architecture) get a small extra bump.
    """
    base = 12.0
    if _looks_shaky(exif):
        base = 6.0
    # edge_density ranges ~0..1; scale a small bump.
    base += min(8.0, s.edge_density * 16.0)
    return min(30.0, max(0.0, base))


def _choose_denoise(s: _Stats, exif: ExifSummary) -> float:
    """Prefer ISO when we know it (more reliable than measuring noise that
    might be confused with texture). Fall back to flat-patch noise."""
    if exif.iso is not None:
        # Below 400: clean. 400-1600: gentle. 1600+: scale up. Cap at 30 — beyond
        # that, you're erasing texture, not fixing noise.
        iso = exif.iso
        if iso <= 400:
            return 0.0
        if iso <= 1600:
            return float((iso - 400) / 1200.0) * 8.0     # 0 → 8
        if iso <= 6400:
            return 8.0 + float((iso - 1600) / 4800.0) * 14.0  # 8 → 22
        return 22.0 + min(8.0, float((iso - 6400) / 6400.0) * 8.0)
    # No EXIF: measure.
    noise = s.flat_noise
    return max(0.0, min(20.0, (noise - 0.005) * 1500.0))


def _choose_deblur(exif: ExifSummary) -> bool:
    """Auto-enable AI Deblur when the shutter speed strongly suggests shake.

    The classic rule says shake risk starts when shutter < 1/(focal_length_35mm).
    Modern stabilization extends that; we use 1/(focal × 1.5) as the trigger
    so the toggle fires only on clearly-risky shots.
    """
    return _looks_shaky(exif)


def _looks_shaky(exif: ExifSummary) -> bool:
    f = exif.focal_length_35mm or (
        # If we only have native focal length, assume APS-C (1.5x) which covers
        # Fuji X-series perfectly. Worst case the slider is a small false positive.
        exif.focal_length_mm * 1.5 if exif.focal_length_mm else None
    )
    if f is None or exif.shutter_seconds is None:
        return False
    safe_shutter = 1.0 / (f * 1.5)
    return exif.shutter_seconds > safe_shutter


# ---------------------------------------------------------------------------
# Stats primitives.
# ---------------------------------------------------------------------------

def _luminance(img: Image) -> NDArray[np.float32]:
    return (
        0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    ).astype(np.float32)


def _flat_patch_noise(lum: NDArray[np.float32]) -> float:
    """Estimate noise as the median std in the flattest 10% of 32x32 patches."""
    h, w = lum.shape
    if h < 64 or w < 64:
        return float(lum.std())
    patch = 32
    ys = np.arange(0, h - patch, patch)
    xs = np.arange(0, w - patch, patch)
    stds: list[float] = []
    for y in ys:
        for x in xs:
            stds.append(float(lum[y : y + patch, x : x + patch].std()))
    stds.sort()
    flat = stds[: max(1, len(stds) // 10)]
    return float(np.median(flat))


def _edge_density(lum: NDArray[np.float32]) -> float:
    """How much of the image is occupied by significant edges, 0..1.

    Used to nudge sharpen up on text- or foliage-heavy frames.
    """
    g = cv2.Sobel(lum, cv2.CV_32F, 1, 1, ksize=3)
    mag = np.abs(g)
    threshold = float(np.median(mag) + 2 * np.std(mag))
    return float((mag > threshold).mean())


def _shades_of_gray(img: Image, *, p: float = 6.0) -> tuple[float, float, float]:
    """Shades-of-Gray illuminant estimation (Finlayson & Trezzi 2004).

    Better than gray-world: averages each channel after raising pixels to the
    p-th power, which weights brighter patches more — closer to how human
    color constancy actually works.
    """
    eps = 1e-8
    means = []
    for c in range(3):
        ch = np.clip(img[:, :, c], 0, None)
        means.append(float(np.mean(ch**p) ** (1.0 / p)) + eps)
    # Normalize so the result is roughly [0..1] regardless of scene brightness.
    s = sum(means) / 3.0
    return (means[0] / max(s, eps), means[1] / max(s, eps), means[2] / max(s, eps))
