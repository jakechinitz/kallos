@echo off
REM kallos launcher for Windows.
REM Creates a venv on first run, installs deps, then opens the app in your browser.

cd /d "%~dp0"

REM Use the official `py` launcher with -3, which picks the highest installed
REM Python 3.x. Falls back to `python` if the py launcher isn't available.
REM This dodges the Windows Store Python alias trap and works on whatever
REM Python 3.10+ you have installed (3.10, 3.11, 3.12, 3.13, anything newer).
set "PYCMD="
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3"
) else (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PYCMD=python"
    )
)

if "%PYCMD%"=="" (
    echo kallos needs Python 3.10 or newer.
    echo Opening the download page...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verify the version is at least 3.10.
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Your Python is too old. kallos needs 3.10 or newer.
    echo Opening the download page...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Setting up kallos for the first time. This takes about 30 seconds.
    echo Windows Defender may scan the new files - that is normal.
    %PYCMD% -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
) else (
    call .venv\Scripts\activate.bat
)

python -m kallos
pause
