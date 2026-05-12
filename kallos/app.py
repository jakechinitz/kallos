"""FastAPI app: serves the UI and the image-processing endpoints.

Endpoints:
- GET  /                       — the single HTML page
- POST /upload                 — multipart upload; returns dimensions + WB info
- POST /render                 — body: settings JSON; returns a JPEG preview
- POST /auto                   — returns auto-suggested Settings for the session
- POST /save                   — body: settings + format; streams the saved file
                                 back as a download (Content-Disposition)
- POST /reset                  — clears the session
- GET  /original.jpg           — preview JPEG of the untouched original

The state is a single in-process Session — this is a local app for one user,
not a multi-tenant service. Keeping it singleton-y means no auth, no DB, no
fuss.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from kallos.auto import auto_for_text, auto_settings
from kallos.io import EXTENSION, MIME_TYPE, SaveFormat, serialize_image
from kallos.io.load import RAW_EXTENSIONS, load_image, load_image_from_bytes
from kallos.pipeline import encode_for_display, render_final, render_preview
from kallos.state import Session, Settings

PREVIEW_LONG_EDGE = 1280

app = FastAPI(title="kallos")

# In-memory state for the single open image.
_session: Session | None = None


# ---------------------------------------------------------------------------
# No-cache for everything (this is a local single-user dev tool; the cost of
# a fresh fetch is zero, the cost of a stale UI is real — see commit history
# for the time the bug fixes "didn't land" because of browser caching).
# ---------------------------------------------------------------------------

@app.middleware("http")
async def no_cache(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["cache-control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["pragma"] = "no-cache"
    response.headers["expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Upload / reset
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    global _session
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    data = await file.read()

    if suffix in RAW_EXTENSIONS:
        # rawpy needs a path, so spill to a temp file.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            preview = load_image(tmp_path, max_long_edge=PREVIEW_LONG_EDGE)
            full = load_image(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        preview = load_image_from_bytes(data, file.filename, max_long_edge=PREVIEW_LONG_EDGE)
        full = load_image_from_bytes(data, file.filename)

    _session = Session(
        original=preview.image,
        full_res=full.image,
        camera_wb=preview.camera_wb,
        daylight_wb=preview.daylight_wb,
        source_name=preview.source_name,
        is_raw=preview.is_raw,
        exif_bytes=preview.exif_bytes,
        exif=preview.exif,
    )

    h, w = preview.image.shape[:2]
    return JSONResponse(
        {
            "width": int(w),
            "height": int(h),
            "is_raw": preview.is_raw,
            "source_name": preview.source_name,
            "settings": asdict(_session.settings),
        }
    )


@app.post("/reset")
def reset() -> JSONResponse:
    if _session is None:
        return JSONResponse({"ok": True})
    _session.settings = Settings()
    _session.invalidate_cache()
    return JSONResponse({"settings": asdict(_session.settings)})


# ---------------------------------------------------------------------------
# Auto Enhance
# ---------------------------------------------------------------------------

@app.post("/auto")
def auto() -> JSONResponse:
    if _session is None:
        raise HTTPException(400, "No image loaded")
    s = auto_settings(_session.original, exif=_session.exif, is_raw=_session.is_raw)
    _session.settings = s
    _session.invalidate_cache()
    return JSONResponse(asdict(s))


@app.post("/auto-text")
def auto_text() -> JSONResponse:
    """Text-optimized Auto: pushes Sharpen + Clarity for book spines / signs /
    documents, and enables AI Deblur (text suffers most from shake)."""
    if _session is None:
        raise HTTPException(400, "No image loaded")
    s = auto_for_text(exif=_session.exif)
    _session.settings = s
    _session.invalidate_cache()
    return JSONResponse(asdict(s))


# ---------------------------------------------------------------------------
# Render preview
# ---------------------------------------------------------------------------

@app.post("/render")
async def render(settings: dict[str, Any]) -> Response:
    if _session is None:
        raise HTTPException(400, "No image loaded")
    _session.settings = Settings.from_dict(settings)
    img = await asyncio.to_thread(render_preview, _session)
    jpeg = _encode_jpeg(img, quality=85)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/original.jpg")
def original() -> Response:
    if _session is None:
        raise HTTPException(400, "No image loaded")
    jpeg = _encode_jpeg(_session.original, quality=85)
    return Response(content=jpeg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Save — streams the rendered file as a browser download
# ---------------------------------------------------------------------------

@app.post("/save")
async def save(payload: dict[str, Any]) -> Response:
    """Render the full-resolution image and stream it as a download.

    No disk write — the user's browser handles where to put the file
    (their default Downloads folder, or wherever they pick).
    """
    if _session is None:
        raise HTTPException(400, "No image loaded")

    settings = Settings.from_dict(payload.get("settings", {}))
    fmt_name = str(payload.get("format", "jpeg")).lower()
    try:
        fmt = SaveFormat(fmt_name)
    except ValueError as exc:
        raise HTTPException(400, f"Unknown format: {fmt_name}") from exc

    _session.settings = settings
    full = _session.full_res if _session.full_res is not None else _session.original
    rendered = await asyncio.to_thread(lambda: render_final(_session, source=full))

    body = await asyncio.to_thread(
        serialize_image, rendered, fmt, exif_bytes=_session.exif_bytes
    )

    filename = f"{_session.source_name}_kallos{EXTENSION[fmt]}"
    return Response(
        content=body,
        media_type=MIME_TYPE[fmt],
        headers={"content-disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _encode_jpeg(img, *, quality: int) -> bytes:  # type: ignore[no-untyped-def]
    """Encode a linear-RGB tensor as a JPEG byte string for the browser."""
    u8 = encode_for_display(img)
    pil = PILImage.fromarray(u8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()
