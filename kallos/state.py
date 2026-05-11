"""In-memory session state and the slider settings schema.

A `Session` holds the original linear-RGB tensor, an optional cached
intermediate (post WB + denoise), and the current `Settings`. The settings
dataclass is the single source of truth for which sliders exist; the FastAPI
layer and the frontend both speak this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kallos.io.load import ExifSummary

Image = NDArray[np.float32]


@dataclass
class Settings:
    """Slider values. All defaults must produce an identity pipeline."""

    brightness: float = 0.0      # -100..+100  -> ±2 EV
    contrast: float = 0.0        # -100..+100  -> S-curve strength
    warmth: float = 0.0          # -100..+100  -> ±1500K offset on top of wb_preset
    vibrance: float = 0.0        # -100..+100  -> skin-protected saturation
    sharpen: float = 0.0         # 0..100      -> unsharp mask amount
    denoise: float = 0.0         # 0..100      -> NL-means strength
    ai_deblur: bool = False      # toggle; applied on Save only
    wb_preset: str = "as_shot"   # as_shot | auto | daylight | tungsten | shade

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def affects_intermediate(self, other: Settings) -> bool:
        """Return True if changing self -> other invalidates the cached intermediate.

        The pipeline caches the post-WB, post-denoise tensor. Anything that runs
        at or before that point invalidates the cache; everything after is cheap
        to recompute.
        """
        return (
            self.wb_preset != other.wb_preset
            or self.warmth != other.warmth
            or self.brightness != other.brightness
            or self.denoise != other.denoise
        )


@dataclass
class Session:
    """All the state for one open image.

    `original` is the untouched linear-RGB tensor at preview resolution.
    `full_res` is the same image at full resolution (lazily kept for Save).
    `intermediate` is a cache of the pipeline output after the expensive ops.
    """

    original: Image
    full_res: Image | None = None
    settings: Settings = field(default_factory=Settings)
    intermediate: Image | None = None
    intermediate_for: Settings | None = None
    # Camera-provided WB multipliers (RGBG, scaled); None for non-RAW.
    camera_wb: tuple[float, float, float, float] | None = None
    daylight_wb: tuple[float, float, float, float] | None = None
    # Source filename, used to suggest a save path.
    source_name: str = "image"
    # True if the original came from a RAW file (rawpy decode), False otherwise.
    # Auto Enhance uses this to choose between "do the whole job" and "top up
    # what the camera already did".
    is_raw: bool = True
    # EXIF bytes captured at load; passed through on save when possible.
    exif_bytes: bytes | None = None
    # Parsed EXIF (ISO/shutter/focal length) used by Auto Enhance.
    exif: ExifSummary = field(default_factory=ExifSummary)

    def invalidate_cache(self) -> None:
        self.intermediate = None
        self.intermediate_for = None
