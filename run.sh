#!/usr/bin/env bash
# kallos launcher for macOS / Linux.
# Creates a venv on first run, installs deps, then opens the app in your browser.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "kallos needs Python 3.10+."
    echo "Install it from https://www.python.org/downloads/ then run this again."
    exit 1
fi

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "Setting up kallos for the first time. This takes about 30 seconds..."
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

exec python -m kallos
