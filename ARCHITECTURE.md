# kallos — code map

Read this first. It tells you where to look for any given change.

## Big picture

A FastAPI server serves a single HTML page. The page sends slider settings to
`POST /render`; the server runs the pipeline and returns a JPEG preview. Save
goes through the same pipeline at full resolution.

```
browser  <--HTTP-->  FastAPI app  <-->  pipeline  <-->  ops (pure functions)
                          |
                          +-- state (in-memory: original tensor, cache, settings)
```

## What's where

| If you want to change... | Look at |
| --- | --- |
| The look of the app (HTML/CSS/JS) | `kallos/static/` |
| Which sliders exist and their ranges | `kallos/state.py` (`Settings` dataclass) |
| The order operations run in | `kallos/pipeline.py` |
| How "Auto Enhance" picks its values | `kallos/auto.py` (reads `ExifSummary` from `kallos/io/load.py`) |
| One specific operation (e.g. sharpen) | `kallos/ops/<name>.py` |
| RAW decoding / image loading | `kallos/io/load.py` |
| Saving JPEG/PNG/TIFF | `kallos/io/save.py` |
| FastAPI routes / API contract | `kallos/app.py` |
| Server entry point + browser-open | `kallos/__main__.py` |
| AI deblur model handling | `kallos/ops/deblur.py` + `kallos/models/registry.py` |

## Image data conventions

A "tensor" in kallos is always `numpy.ndarray`, shape `(H, W, 3)`, dtype
`float32`, value range `[0.0, 1.0]`, in **linear sRGB primaries** (not gamma
encoded). Every op takes one in and returns one out.

The only conversion to/from gamma-encoded display sRGB happens at the edges:
loaders linearize on the way in, savers gamma-encode and dither on the way out.

## Op contract

Every op in `kallos/ops/` exposes:

```python
def apply(img: NDArray[np.float32], amount: float, **opts) -> NDArray[np.float32]: ...
```

- `amount = 0` (or default) must be a no-op or return `img` unchanged.
- Output must have the same shape and dtype as input.
- No side effects, no hidden state.

This contract is what makes the codebase easy to reason about and easy for
an LLM (or you) to edit one op without breaking the rest.

## Pipeline order (in `pipeline.py`)

```
load → wb (preset + warmth) → brightness → denoise →
[deblur on save only] → contrast → sharpen → vibrance → save
```

The reasoning behind this order is documented in `pipeline.py` itself.

## Adding a new slider

1. Add the field to `Settings` in `kallos/state.py`.
2. Add a new file in `kallos/ops/your_op.py` exposing `apply(img, amount)`.
3. Insert one call in `kallos/pipeline.py` at the right point.
4. Add the slider to `kallos/static/index.html` + wire it in `app.js`.
5. Add `tests/test_ops_your_op.py` with the standard smoke + identity tests.

That's it. No registry, no factory, no plugin system.
