"""Tests for kallos.ops.denoise."""

from __future__ import annotations

import numpy as np

from kallos.ops.denoise import apply_denoise


def test_denoise_identity(small_image):
    out = apply_denoise(small_image, 0)
    assert np.array_equal(out, small_image)


def test_denoise_reduces_noise():
    rng = np.random.default_rng(0)
    base = np.full((128, 128, 3), 0.5, dtype=np.float32)
    noisy = np.clip(base + rng.normal(0, 0.05, base.shape).astype(np.float32), 0, 1)
    out = apply_denoise(noisy, 40)
    assert out.std() < noisy.std()


def test_denoise_shape_preserved(small_image):
    out = apply_denoise(small_image, 30)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32
