@echo off
REM kallos launcher for Windows.
REM Creates a venv on first run, installs deps, then opens the app in your browser.

cd /d "%~dp0"

REM Use the official py launcher to dodge the Windows Store Python alias trap.
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    py -3.10 --version >nul 2>&1
    if errorlevel 1 (
        echo kallos needs Python 3.10 or newer.
        echo Opening the download page...
        start https://www.python.org/downloads/
        pause
        exit /b 1
    )
    set "PYCMD=py -3.10"
) else (
    set "PYCMD=py -3.11"
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
