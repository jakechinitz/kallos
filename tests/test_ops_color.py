"""Tests for kallos.ops.color (vibrance)."""

from __future__ import annotations

import numpy as np

from kallos.ops.color import apply_vibrance


def test_vibrance_identity(small_image):
    out = apply_vibrance(small_image, 0)
    assert np.array_equal(out, small_image)


def test_vibrance_positive_increases_saturation(small_image):
    import cv2

    out = apply_vibrance(small_image, 60)
    hsv_before = cv2.cvtColor(small_image, cv2.COLOR_RGB2HSV)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    assert hsv_after[:, :, 1].mean() > hsv_before[:, :, 1].mean()


def test_vibrance_shape_preserved(small_image):
    out = apply_vibrance(small_image, 30)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32
