"""Tests for kallos.ops.tone (brightness, contrast)."""

from __future__ import annotations

import numpy as np

from kallos.ops.tone import apply_brightness, apply_contrast


def test_brightness_identity(small_image):
    out = apply_brightness(small_image, 0)
    assert np.array_equal(out, small_image)


def test_brightness_positive_brightens(small_image):
    out = apply_brightness(small_image, 50)
    assert out.mean() > small_image.mean()


def test_brightness_negative_darkens(small_image):
    out = apply_brightness(small_image, -50)
    assert out.mean() < small_image.mean()


def test_brightness_preserves_shape_dtype(small_image):
    out = apply_brightness(small_image, 25)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32


def test_contrast_identity(small_image):
    out = apply_contrast(small_image, 0)
    assert np.array_equal(out, small_image)


def test_contrast_positive_increases_std(small_image):
    # Positive contrast should expand the histogram (higher std).
    out = apply_contrast(small_image, 60)
    assert out.std() > small_image.std()


def test_contrast_negative_decreases_std(small_image):
    out = apply_contrast(small_image, -60)
    assert out.std() < small_image.std()
