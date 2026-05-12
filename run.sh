#!/usr/bin/env bash
# kallos launcher for macOS / Linux.
# Creates a venv on first run, installs deps, then opens the app in your browser.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "kallos needs Python 3.10 or newer."
    echo "Install it from https://www.python.org/downloads/ then run this again."
    exit 1
fi

# Accept any Python 3.10+ — whatever the user has installed (3.10 through
# 3.13 and beyond). The venv we create will use this interpreter.
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1; then
    echo "Your Python is too old. kallos needs 3.10 or newer."
    echo "You have: $(python3 --version 2>&1)"
    echo "Install a newer Python from https://www.python.org/downloads/ then run this again."
    exit 1
fi

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "Setting up kallos for the first time. This takes 1-2 minutes..."
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip
    if ! python -m pip install -r requirements.txt; then
        echo
        echo "Dependency install failed. Read the messages above to see what went wrong."
        echo "You can retry by deleting the .venv folder and running this script again."
        exit 1
    fi
    # Best-effort: try to add HEIC support. Failure is fine — JPEG/PNG/TIFF/RAF
    # all still work without it.
    if ! python -m pip install "pillow-heif>=0.18,<2" --only-binary=:all: >/dev/null 2>&1; then
        echo
        echo "Note: HEIC reading is unavailable for this Python version."
        echo "This is fine — JPEG, PNG, TIFF, and Fuji RAF all still work."
        echo
    fi
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# Self-heal: if the venv exists but is missing a key package, reinstall.
# Catches the case where an earlier launch's install half-finished.
if ! python -c "import uvicorn, kallos" >/dev/null 2>&1; then
    echo "Dependencies look incomplete - reinstalling..."
    if ! python -m pip install -r requirements.txt; then
        echo
        echo "Dependency install failed. Try deleting the .venv folder and running fresh."
        exit 1
    fi
fi

exec python -m kallos
