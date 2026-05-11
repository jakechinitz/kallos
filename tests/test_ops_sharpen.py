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


def test_sharpen_adaptive_radius_vs_fixed_radius(small_image):
    # The adaptive-radius default should differ from a fixed-radius override
    # at the same amount — proving the auto path is doing something.
    auto = apply_sharpen(small_image, 80)
    fixed = apply_sharpen(small_image, 80, radius=2.0)
    assert not np.allclose(auto, fixed)


def test_sharpen_amount_monotonic(small_image):
    # An edge step at a scale the radius can actually blur. Higher amount must
    # produce a larger total change from the input.
    base = np.full((64, 64, 3), 0.4, dtype=np.float32)
    base[:, 32:, :] = 0.6  # vertical edge in the middle

    low = apply_sharpen(base, 20)
    high = apply_sharpen(base, 80)
    assert np.abs(high - base).sum() > np.abs(low - base).sum()
