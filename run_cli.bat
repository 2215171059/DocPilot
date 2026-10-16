@echo off
chcp 65001 >nul
cd /d "%~dp0"
title DocPilot - CLI

REM ============================================================
REM  DocPilot CLI launcher (Windows)
REM  Double-click to chat with the graph in the terminal.
REM
REM  Debug commands available inside the session:
REM     :history   :state   :quit
REM
REM  First time here? Create the environment once:
REM     python -m venv .venv
REM     .venv\Scripts\python.exe -m pip install -r requirements.txt
REM ============================================================

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "PYTHONPATH=%~dp0src"

"%PYTHON%" scripts\run_cli.py

echo.
echo [DocPilot] session ended.
pause
