"""Tests for Auto Enhance — detail-only philosophy.

Auto is intentionally narrow: it touches Clarity, Sharpen, and AI Deblur.
Everything else (brightness, contrast, warmth, vibrance, denoise, WB) stays
at the camera's choice. These tests pin that contract.
"""

from __future__ import annotations

from kallos.auto import auto_settings
from kallos.io.load import ExifSummary


def test_auto_leaves_tone_and_color_alone(small_image):
    s = auto_settings(small_image)
    assert s.brightness == 0
    assert s.contrast == 0
    assert s.warmth == 0
    assert s.vibrance == 0
    assert s.denoise == 0
    assert s.wb_preset == "as_shot"


def test_auto_touches_only_detail_controls(small_image):
    s = auto_settings(small_image)
    assert s.clarity > 0
    assert s.sharpen > 0


def test_auto_settings_in_valid_range(small_image):
    s = auto_settings(small_image)
    assert 0 <= s.clarity <= 100
    assert 0 <= s.sharpen <= 100


def test_jpeg_path_leans_on_clarity(small_image):
    # JPEG: camera already sharpened, so Clarity does the heavy lifting.
    jpg = auto_settings(small_image, is_raw=False)
    assert jpg.clarity > jpg.sharpen


def test_raw_path_balances_clarity_and_sharpen(small_image):
    # RAW: nothing was sharpened in-camera, so Sharpen comes up to match Clarity.
    raw = auto_settings(small_image, is_raw=True)
    jpg = auto_settings(small_image, is_raw=False)
    assert raw.sharpen > 0
    assert raw.clarity > 0
    # JPEG leans on Clarity; RAW evens it out.
    assert raw.sharpen > jpg.sharpen


def test_slow_shutter_with_long_lens_enables_deblur(small_image, monkeypatch):
    # Auto only enables AI Deblur when (a) the shot looks shaky AND (b) a
    # model is available. Mock the latter so we're testing the shake logic.
    monkeypatch.setattr("kallos.auto.has_deblur_model", lambda: True)
    shaky = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=200, shutter_seconds=1 / 60,
    ))
    assert shaky.ai_deblur is True


def test_fast_shutter_keeps_deblur_off(small_image, monkeypatch):
    monkeypatch.setattr("kallos.auto.has_deblur_model", lambda: True)
    crisp = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=50, shutter_seconds=1 / 500,
    ))
    assert crisp.ai_deblur is False


def test_deblur_works_for_jpeg_too(small_image, monkeypatch):
    monkeypatch.setattr("kallos.auto.has_deblur_model", lambda: True)
    s = auto_settings(
        small_image,
        exif=ExifSummary(focal_length_35mm=200, shutter_seconds=1 / 60),
        is_raw=False,
    )
    assert s.ai_deblur is True


def test_deblur_stays_off_when_no_model_even_if_shaky(small_image, monkeypatch):
    """Without a real model, a 'deblur on' toggle would do nothing — Auto
    shouldn't lie about it."""
    monkeypatch.setattr("kallos.auto.has_deblur_model", lambda: False)
    s = auto_settings(small_image, exif=ExifSummary(
        focal_length_35mm=200, shutter_seconds=1 / 60,
    ))
    assert s.ai_deblur is False


def test_no_exif_keeps_deblur_off(small_image):
    # Without EXIF we can't tell, so don't auto-enable.
    s = auto_settings(small_image, exif=None)
    assert s.ai_deblur is False
