"""Shared test fixtures: a small synthetic image with known properties.

Using a synthetic fixture (instead of a real photo) makes tests deterministic
and keeps the repo small. The image has gradient + colored regions so the
ops have something nontrivial to act on.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def small_image() -> NDArray[np.float32]:
    """64x96 linear-RGB float32 image with gradients and color blocks."""
    h, w = 64, 96
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Horizontal luminance gradient.
    img[:, :, :] = np.linspace(0.05, 0.95, w, dtype=np.float32)[None, :, None]
    # Add a warm patch and a cool patch.
    img[16:48, 16:32, 0] *= 1.4  # red boost
    img[16:48, 16:32, 2] *= 0.6  # blue cut -> warm
    img[16:48, 64:80, 0] *= 0.7
    img[16:48, 64:80, 2] *= 1.3
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def jpeg_bytes(small_image, tmp_path):
    """Same image, encoded as a JPEG byte string."""
    from PIL import Image as PILImage
    u8 = (np.clip(small_image, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)
    pil = PILImage.fromarray(u8, mode="RGB")
    path = tmp_path / "fixture.jpg"
    pil.save(path, format="JPEG", quality=95)
    return path.read_bytes(), str(path)
