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


def test_denoise_preserves_edges_more_than_flats():
    """Edge-protected denoise: a strong edge should change less than a flat
    noisy region under the same denoise strength."""
    rng = np.random.default_rng(0)
    base = np.zeros((128, 128, 3), dtype=np.float32)
    base[:, :64, :] = 0.3
    base[:, 64:, :] = 0.7
    noisy = np.clip(base + rng.normal(0, 0.04, base.shape).astype(np.float32), 0, 1)

    out = apply_denoise(noisy, 50)

    # The edge column (x=64) should change less than a flat column far from it.
    edge_col = np.abs(out[:, 63:65, :] - noisy[:, 63:65, :]).mean()
    flat_col = np.abs(out[:, 10:20, :] - noisy[:, 10:20, :]).mean()
    assert edge_col < flat_col, (
        f"edge should be preserved more than flats: edge_change={edge_col:.4f}, "
        f"flat_change={flat_col:.4f}"
    )
