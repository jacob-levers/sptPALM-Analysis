@echo off
:: sptPALM Launch Script for Windows
:: On first run the app installs its own dependencies automatically.
:: Python 3 must already be installed (python.org).

setlocal

set "FOLDER=%~dp0"
set "VENV_PY=%FOLDER%sptpalm-env\Scripts\python.exe"
set "APP=%FOLDER%app_tk.py"

:: Prefer venv Python (present after first-time setup), fall back to system Python
if exist "%VENV_PY%" (
    set "PYTHON=%VENV_PY%"
    goto run
)

for /f "delims=" %%i in ('where python3 2^>nul') do (
    set "PYTHON=%%i"
    goto run
)
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYTHON=%%i"
    goto run
)

echo.
echo  Python 3 not found.
echo  Please install Python 3 from https://www.python.org/downloads/
echo  Make sure to tick "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:run
cd /d "%FOLDER%"
"%PYTHON%" "%APP%"
