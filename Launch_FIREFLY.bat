@echo off
:: FIREFLY - Windows launcher
::
:: On first run (or after upgrading from a v1.x install whose venv
:: predates PySide6 / napari / torch) this script:
::   1. detects the missing venv (or stale one without PySide6),
::   2. installs dependencies with pip output visible so the user can
::      see progress -- installs take 3-8 minutes on first run because
::      napari + PyTorch are large wheels,
::   3. launches FIREFLY when the install finishes.
::
:: On subsequent runs (healthy venv present) it launches FIREFLY
:: directly with no install step.

setlocal

set "FOLDER=%~dp0"
set "APP=%FOLDER%app_qt.py"
set "VENV=%FOLDER%sptpalm-env"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "REQ=%FOLDER%requirements.txt"

:: -- Sanity check: app file present ------------------------------------
if not exist "%APP%" (
    echo.
    echo  FIREFLY files not found at %FOLDER%.
    echo  Re-extract the FIREFLY folder and try again.
    echo.
    pause
    exit /b 1
)

:: -- Locate a system Python (used only to create the venv) -------------
set "PYTHON="
for /f "delims=" %%i in ('where python3 2^>nul') do (
    set "PYTHON=%%i"
    goto :found_python
)
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYTHON=%%i"
    goto :found_python
)
echo.
echo  Python 3 is not installed.
echo  Install it from https://www.python.org/downloads/
echo  (tick "Add Python to PATH" during installation)
echo.
pause
exit /b 1
:found_python

:: -- Check whether the venv exists AND has PySide6 ---------------------
:: A bare `exist` check on python.exe isn't enough -- a v1.x venv would
:: pass that but lack PySide6 / napari.  Probe with `python -c "import
:: PySide6"` and reinstall if it fails.
set "NEEDS_SETUP=0"
if not exist "%VENV_PY%" (
    set "NEEDS_SETUP=1"
) else (
    "%VENV_PY%" -c "import PySide6" >nul 2>&1
    if errorlevel 1 set "NEEDS_SETUP=1"
)

if "%NEEDS_SETUP%"=="1" goto :setup
goto :launch

:setup
echo.
echo  ============================================================
echo    FIREFLY first-time setup
echo    Installs PySide6, napari, PyTorch, scipy, etc.
echo    Expect 3-8 minutes depending on network speed.
echo  ============================================================
echo.

cd /d "%FOLDER%"

if not exist "%VENV_PY%" (
    echo ^> python -m venv sptpalm-env
    "%PYTHON%" -m venv sptpalm-env
    if errorlevel 1 (
        echo.
        echo  Failed to create the virtual environment.
        echo  Check that Python 3 has the venv module (it does by default).
        pause
        exit /b 1
    )
)

echo.
echo ^> upgrading pip
"%VENV_PY%" -m pip install --upgrade pip

echo.
echo ^> installing requirements (this is the slow step)
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  Dependency installation failed.  Re-run this script to retry,
    echo  or run manually:
    echo      %VENV_PY% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo    Setup complete - launching FIREFLY...
echo  ============================================================
echo.

:launch
cd /d "%FOLDER%"
"%VENV_PY%" "%APP%"
