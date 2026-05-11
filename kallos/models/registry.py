"""Optional ONNX model discovery and lazy loading.

By design, kallos works fully without any ML weights — the deblur op falls
back to a classical Wiener deconvolution if no model is available. If you
drop a compatible NAFNet motion-deblur ONNX file into ~/.cache/kallos/models/
named `nafnet_motion_deblur.onnx`, the op will use it automatically.

We intentionally don't hardcode a download URL here. The ONNX export of
NAFNet-REDS is community-distributed and locations change; the user is one
release-notes update away from being told where to fetch it. Keeping this
local-only avoids shipping a broken download UX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def cache_dir() -> Path:
    p = Path.home() / ".cache" / "kallos" / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


DEBLUR_MODEL_PATH = cache_dir() / "nafnet_motion_deblur.onnx"


class OnnxSession:
    """Thin wrapper around an onnxruntime InferenceSession.

    Expects an NCHW float32 input in [0, 1] and produces the same shape.
    """

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def run(self, patch: NDArray[np.float32]) -> NDArray[np.float32]:
        # patch: (H, W, 3) float32 [0,1]
        x = patch.transpose(2, 0, 1)[None, ...]  # 1xCxHxW
        outputs: list[Any] = self._sess.run(None, {self._input_name: x})
        y = outputs[0][0].transpose(1, 2, 0)
        return np.clip(y, 0.0, None).astype(np.float32)


def try_load_deblur_model() -> OnnxSession | None:
    """Return a session if the model is present and loadable, else None."""
    if not DEBLUR_MODEL_PATH.exists():
        return None
    try:
        return OnnxSession(DEBLUR_MODEL_PATH)
    except Exception:  # noqa: BLE001 — any failure means fall back to Wiener
        return None
