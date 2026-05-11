"""Tests for kallos.ops.clarity."""

from __future__ import annotations

import numpy as np

from kallos.ops.clarity import apply_clarity


def test_clarity_identity(small_image):
    out = apply_clarity(small_image, 0)
    assert np.array_equal(out, small_image)


def test_clarity_changes_local_contrast(small_image):
    out = apply_clarity(small_image, 50)
    assert not np.allclose(out, small_image)


def test_clarity_shape_preserved(small_image):
    out = apply_clarity(small_image, 40)
    assert out.shape == small_image.shape
    assert out.dtype == np.float32


def test_clarity_increases_midtone_contrast():
    """A midtone-heavy image should gain local contrast (higher std in
    midtones) with positive clarity."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.35, 0.65, (256, 256, 3)).astype(np.float32)
    out = apply_clarity(base, 80)
    # Std of midtones increases (microcontrast lifts the texture).
    assert out.std() > base.std()


def test_clarity_negative_softens():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.35, 0.65, (256, 256, 3)).astype(np.float32)
    out = apply_clarity(base, -60)
    assert out.std() < base.std()


def test_clarity_protects_shadows_and_highlights():
    """Pixels near pure black or pure white should be barely touched."""
    img = np.zeros((128, 128, 3), dtype=np.float32)
    img[:, :64, :] = 0.02   # near black
    img[:, 64:, :] = 0.98   # near white
    out = apply_clarity(img, 80)
    # The midtone mask should suppress the effect at both extremes.
    extreme_delta = np.abs(out - img).max()
    assert extreme_delta < 0.05
