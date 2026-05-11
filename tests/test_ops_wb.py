"""Tests for kallos.ops.wb (white balance)."""

from __future__ import annotations

import numpy as np

from kallos.ops.wb import apply_preset, apply_warmth


def test_warmth_identity(small_image):
    out = apply_warmth(small_image, 0)
    assert np.array_equal(out, small_image)


def test_warmth_positive_warms(small_image):
    out = apply_warmth(small_image, 60)
    # Warming raises red relative to blue.
    rb_before = float(small_image[..., 0].mean() - small_image[..., 2].mean())
    rb_after  = float(out[..., 0].mean() - out[..., 2].mean())
    assert rb_after > rb_before


def test_warmth_negative_cools(small_image):
    out = apply_warmth(small_image, -60)
    rb_before = float(small_image[..., 0].mean() - small_image[..., 2].mean())
    rb_after  = float(out[..., 0].mean() - out[..., 2].mean())
    assert rb_after < rb_before


def test_preset_daylight_is_identity_without_wb(small_image):
    # Daylight preset is the reference; without camera_wb it should be a no-op gain.
    out = apply_preset(small_image, "daylight")
    assert np.allclose(out, small_image)


def test_preset_tungsten_shifts_warm(small_image):
    out = apply_preset(small_image, "tungsten")
    assert out[..., 0].mean() > small_image[..., 0].mean()
    assert out[..., 2].mean() < small_image[..., 2].mean()
