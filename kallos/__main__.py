"""Entry point: `python -m kallos`.

Starts uvicorn bound to localhost only, then opens the browser to the UI.
Localhost-only binding keeps Windows Defender's firewall prompt away.
"""

from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8765


def _open_browser_when_ready() -> None:
    # Tiny delay so uvicorn has time to start listening.
    time.sleep(0.8)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main() -> None:
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    uvicorn.run("kallos.app:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
