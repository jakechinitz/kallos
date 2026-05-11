# kallos

A simple beautifier for photos. Sliders, not menus.

Made for shooters (especially Fuji owners) who want photos that pop without
fighting Lightroom. One click of **Auto Enhance** gets you a polished baseline.
Six sliders fine-tune from there. An optional AI deblur mode recovers detail
from camera shake.

## What it can do

- Open **Fuji RAF**, JPEG, PNG, TIFF
- One-click **Auto Enhance** with smart defaults per image
- Sliders: Brightness, Contrast, Warmth, Vibrance, Sharpen, Denoise
- **AI Deblur** toggle for shake recovery (NAFNet motion model, with classical Wiener fallback)
- **Compare** view modes: edited / side-by-side / wipe-slider
- Save as JPEG (quality 95), PNG, or 8-bit TIFF with sRGB ICC + EXIF preserved

## Run it

### macOS / Linux

```bash
./run.sh
```

### Windows

Double-click `run.bat` (or run it from a terminal).

The first launch creates a Python virtual environment and installs dependencies
(~30 seconds). After that the app opens in your default browser at
<http://127.0.0.1:8765>.

Python 3.10 or newer is required. If you don't have it, run.sh / run.bat will
tell you and point you to the download.

## Tour

1. Drag an image onto the window (or click to pick one).
2. Click the big **Auto Enhance** button.
3. Nudge any slider to fine-tune.
4. Click **Compare** to A/B against the original.
5. Click **Save** to write the result alongside the source file.

## For the curious

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) for a one-page map of the code.
Every image operation is a pure function in `kallos/ops/`. The pipeline is a
flat recipe in `kallos/pipeline.py`.
