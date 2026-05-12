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
    echo Setting up kallos for the first time. This takes 1-2 minutes.
    echo Windows Defender may scan the new files - that is normal.
    echo.
    %PYCMD% -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the virtual environment.
        echo If the folder is on OneDrive, move it somewhere local first.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency install failed. Read the messages above to see what went wrong.
        echo Common causes:
        echo   - Folder is on OneDrive   ^(move it somewhere local^)
        echo   - No internet connection
        echo   - A package wheel isn't available for your Python version
        echo.
        echo You can retry by deleting the .venv folder and running this script again.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

REM Self-heal: if the venv exists but is missing a key package, reinstall.
REM This catches the case where an earlier launch's install half-finished.
python -c "import uvicorn, kallos" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Dependencies look incomplete - reinstalling...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency install failed again. Try deleting the .venv folder
        echo and running this script fresh.
        pause
        exit /b 1
    )
)

python -m kallos
pause
