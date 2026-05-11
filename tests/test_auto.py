"""Tests for Auto Enhance — verifying the EXIF-driven smarts actually trigger."""

from __future__ import annotations

import numpy as np
import pytest

from kallos.auto import auto_settings
from kallos.io.load import ExifSummary


def test_auto_settings_in_valid_range(small_image):
    s = auto_settings(small_image)
    assert -100 <= s.brightness <= 100
    assert -100 <= s.contrast <= 100
    assert -100 <= s.warmth <= 100
    assert 0 <= s.sharpen <= 100
    assert 0 <= s.denoise <= 100
    assert s.wb_preset in {"as_shot", "auto", "daylight", "tungsten", "shade"}


def test_auto_is_restrained_vibrance(small_image):
    # Leica-style: vibrance is deliberately low.
    s = auto_settings(small_image)
    assert s.vibrance <= 10.0


def test_high_iso_lifts_denoise(small_image):
    clean = auto_settings(small_image, exif=ExifSummary(iso=100))
    grainy = auto_settings(small_image, exif=ExifSummary(iso=3200))
    assert grainy.denoise > clean.denoise


def test_iso_400_keeps_denoise_low(small_image):
    s = auto_settings(small_image, exif=ExifSummary(iso=400))
    assert s.denoise <= 1.0  # below 400 ISO returns 0.0; allow a touch of float slack


def test_slow_shutter_with_long_lens_enables_deblur(small_image):
    # 200mm at 1/60s — well below the 1/(200*1.5) safe shutter.
    shaky = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=200, shutter_seconds=1 / 60,
    ))
    assert shaky.ai_deblur is True


def test_fast_shutter_keeps_deblur_off(small_image):
    crisp = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=50, shutter_seconds=1 / 500,
    ))
    assert crisp.ai_deblur is False


def test_shaky_shot_gets_less_sharpen(small_image):
    crisp = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=50, shutter_seconds=1 / 500,
    ))
    shaky = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=200, shutter_seconds=1 / 60,
    ))
    # Sharpen blur is the wrong move; deblur handles it.
    assert shaky.sharpen <= crisp.sharpen


def test_no_exif_falls_back_to_measurement(small_image):
    # Without EXIF, the function must still return *something* sensible
    # (it falls back to measuring noise in flat patches).
    s = auto_settings(small_image, exif=None)
    assert 0 <= s.denoise <= 30
    assert s.ai_deblur is False  # no EXIF -> can't tell, so don't enable


def test_warmth_only_partially_corrects_a_cast():
    # Moderate warm cast — auto should nudge cooler but not fully neutralize.
    # (The neutralization gain that would zero the cast is much larger than
    # what we apply; we use a 40% correction factor on purpose.)
    rng = np.random.default_rng(0)
    img = rng.uniform(0.3, 0.6, (128, 128, 3)).astype(np.float32)
    img[..., 0] *= 1.08
    img[..., 2] *= 0.92
    img = np.clip(img, 0, 1).astype(np.float32)
    s = auto_settings(img)
    assert s.warmth < 0          # correcting toward cool
    assert s.warmth > -20.0      # but partial, not maxed


def test_jpeg_path_is_lighter_than_raw_path(small_image):
    raw = auto_settings(small_image, is_raw=True)
    jpg = auto_settings(small_image, is_raw=False)

    # Camera already sharpened/denoised JPEGs; Auto should hold back.
    assert jpg.denoise == 0.0
    assert jpg.sharpen <= raw.sharpen
    assert jpg.vibrance <= raw.vibrance
    # Brightness/contrast are scene-dependent, not camera-dependent, so they
    # should be the same regardless of source.
    assert jpg.brightness == raw.brightness
    assert jpg.contrast == raw.contrast


def test_jpeg_gets_more_clarity_than_raw(small_image):
    # Clarity is the iPhone-look ingredient the camera doesn't apply,
    # so the JPEG path leans into it harder than the RAW path.
    raw = auto_settings(small_image, is_raw=True)
    jpg = auto_settings(small_image, is_raw=False)
    assert jpg.clarity > raw.clarity


def test_jpeg_ignores_iso_for_denoise(small_image):
    # On a JPEG, the camera already denoised — high ISO shouldn't crank
    # kallos's denoise on top.
    s = auto_settings(small_image, exif=ExifSummary(iso=6400), is_raw=False)
    assert s.denoise == 0.0


def test_jpeg_still_enables_deblur_on_shake(small_image):
    # Shake recovery is the one thing the camera can't do, so JPEG path
    # must still auto-enable it.
    s = auto_settings(
        small_image,
        exif=ExifSummary(focal_length_35mm=200, shutter_seconds=1 / 60),
        is_raw=False,
    )
    assert s.ai_deblur is True


@pytest.mark.parametrize("iso,expected_min,expected_max", [
    (100, 0.0, 0.5),
    (800, 1.5, 5.0),
    (3200, 12.0, 20.0),
    (12800, 22.0, 30.0),
])
def test_denoise_scales_smoothly_with_iso(small_image, iso, expected_min, expected_max):
    s = auto_settings(small_image, exif=ExifSummary(iso=iso))
    assert expected_min <= s.denoise <= expected_max, f"iso={iso} got denoise={s.denoise}"
