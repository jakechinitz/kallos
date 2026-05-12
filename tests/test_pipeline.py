"""End-to-end pipeline tests."""

from __future__ import annotations

import numpy as np

from kallos.auto import auto_settings
from kallos.pipeline import encode_for_display, render_final, render_preview
from kallos.state import Session, Settings


def test_default_settings_are_near_identity(small_image):
    session = Session(original=small_image)
    out = render_preview(session)
    # All defaults are no-ops; output should match input.
    assert np.allclose(out, small_image)


def test_render_preview_with_all_sliders(small_image):
    session = Session(original=small_image)
    session.settings = Settings(
        brightness=20, contrast=30, warmth=15,
        vibrance=20, sharpen=25, denoise=10,
    )
    out = render_preview(session)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32
    # The pipeline should produce something visibly different.
    assert not np.allclose(out, small_image)


def test_intermediate_cache_reused_for_tonal_changes(small_image):
    session = Session(original=small_image)
    session.settings = Settings(brightness=10, denoise=5, contrast=20)
    render_preview(session)
    cached_id = id(session.intermediate)
    # Change only a downstream slider (contrast). Cache must be reused.
    session.settings = Settings(brightness=10, denoise=5, contrast=40)
    render_preview(session)
    assert id(session.intermediate) == cached_id


def test_intermediate_cache_invalidated_for_upstream_changes(small_image):
    session = Session(original=small_image)
    session.settings = Settings(brightness=10, denoise=5)
    render_preview(session)
    cached_id = id(session.intermediate)
    # Brightness is upstream of the cache point; must invalidate.
    session.settings = Settings(brightness=30, denoise=5)
    render_preview(session)
    assert id(session.intermediate) != cached_id


def test_auto_enhance_produces_plausible_settings(small_image):
    s = auto_settings(small_image)
    # Sliders must be within the declared ranges.
    assert -100 <= s.brightness <= 100
    assert -100 <= s.contrast <= 100
    assert -100 <= s.warmth <= 100
    assert 0 <= s.sharpen <= 100
    assert 0 <= s.denoise <= 100


def test_render_final_runs_without_ai(small_image):
    session = Session(original=small_image)
    session.settings = Settings(brightness=10, sharpen=20, ai_deblur=False)
    out = render_final(session)
    assert out.shape == small_image.shape


def test_encode_for_display_returns_uint8(small_image):
    enc = encode_for_display(small_image)
    assert enc.dtype == np.uint8
    assert enc.shape == small_image.shape


def test_clarity_does_not_shift_neutral_gray():
    """Regression test for the color-space bug where Clarity ran cv2 Lab
    conversion on linear data instead of sRGB-encoded data, causing a
    visible tonal/color shift even on flat fields. A flat gray image must
    stay flat gray under any clarity amount."""
    from kallos.ops.clarity import apply_clarity

    flat = np.full((64, 64, 3), 0.18, dtype=np.float32)  # mid-gray, linear
    out = apply_clarity(flat, 50)
    # No spatial structure -> detail = 0 -> output should be very close to input.
    assert np.abs(out - flat).max() < 0.01


def test_sharpen_does_not_shift_flat_field():
    """Same regression for Sharpen — a flat field has no edges, so Sharpen
    must produce essentially the input back regardless of amount."""
    from kallos.ops.sharpen import apply_sharpen

    flat = np.full((64, 64, 3), 0.4, dtype=np.float32)
    out = apply_sharpen(flat, 80)
    assert np.abs(out - flat).max() < 0.01


def test_vibrance_does_not_shift_neutral_gray():
    """Vibrance on pure gray has zero saturation to boost; output must match input."""
    from kallos.ops.color import apply_vibrance

    gray = np.full((64, 64, 3), 0.4, dtype=np.float32)
    out = apply_vibrance(gray, 80)
    assert np.abs(out - gray).max() < 0.01
