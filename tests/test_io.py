"""Tests for kallos.io (load/save round-trip)."""

from __future__ import annotations

import numpy as np

from kallos.io import SaveFormat, save_image
from kallos.io.load import load_image


def test_jpeg_roundtrip(small_image, tmp_path):
    out_path = tmp_path / "out.jpg"
    save_image(small_image, out_path, SaveFormat.JPEG)
    assert out_path.exists()

    loaded = load_image(out_path)
    # JPEG is lossy + we gamma-encoded then re-linearized; expect approximate match.
    assert loaded.image.shape == small_image.shape
    assert loaded.image.dtype == np.float32
    assert np.mean(np.abs(loaded.image - small_image)) < 0.05


def test_png_roundtrip(small_image, tmp_path):
    out_path = tmp_path / "out.png"
    save_image(small_image, out_path, SaveFormat.PNG)
    assert out_path.exists()
    loaded = load_image(out_path)
    # PNG is lossless at 8-bit but we lose a bit through the gamma round-trip.
    assert np.mean(np.abs(loaded.image - small_image)) < 0.01


def test_tiff_roundtrip(small_image, tmp_path):
    out_path = tmp_path / "out.tif"
    save_image(small_image, out_path, SaveFormat.TIFF)
    assert out_path.exists()
    loaded = load_image(out_path)
    # 8-bit TIFF round-trip should be lossless aside from gamma quantization.
    assert np.mean(np.abs(loaded.image - small_image)) < 0.01
