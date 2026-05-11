"""Tests for kallos.ops.sharpen."""

from __future__ import annotations

import numpy as np

from kallos.ops.sharpen import apply_sharpen


def test_sharpen_identity(small_image):
    out = apply_sharpen(small_image, 0)
    assert np.array_equal(out, small_image)


def test_sharpen_changes_edges(small_image):
    out = apply_sharpen(small_image, 50)
    # Some pixels must change at non-zero amount.
    assert not np.allclose(out, small_image)


def test_sharpen_shape_preserved(small_image):
    out = apply_sharpen(small_image, 50)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32
