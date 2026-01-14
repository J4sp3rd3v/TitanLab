@echo off
title TITAN STUDIO V8
color 0b
cls

echo [TITAN] Checking Python...
python --version >nul 2>&1
if errorlevel 1 goto NoPython

if not exist "venv" (
    echo [TITAN] Creating venv...
    python -m venv venv
)

call venv\Scripts\activate

echo [TITAN] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto InstallError

cls
echo.
echo    [ SYSTEM ONLINE ]
echo    [ LAUNCHING STUDIO INTERFACE... ]
echo.
echo    Go to: http://127.0.0.1:7860
echo.
python titan_ui.py
pause
exit

:NoPython
echo [!] Python not found!
pause
exit

:InstallError
echo [!] Install failed!
pause
exit
