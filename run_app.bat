@echo off
chcp 65001 >nul
cd /d "%~dp0"
title DocPilot - Web UI

REM ============================================================
REM  DocPilot Web UI launcher (Windows)
REM  Double-click this file to start the Streamlit interface.
REM
REM  What this script handles for you:
REM   1. It prefers the project's own .venv, instead of whatever
REM      "python" happens to be first on PATH.
REM   2. Streamlit apps must be started via "streamlit run app.py".
REM      Running "python app.py" will NOT start the UI.
REM   3. --server.headless skips Streamlit's first-run email prompt.
REM
REM  First time here? Create the environment once:
REM     python -m venv .venv
REM     .venv\Scripts\python.exe -m pip install -r requirements.txt
REM ============================================================

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "PYTHONPATH=%~dp0src"

echo ============================================================
echo   DocPilot  -  LangGraph knowledge-base assistant
echo ------------------------------------------------------------
echo   URL : http://127.0.0.1:8501
echo   Stop: press Ctrl+C in this window
echo ============================================================
echo.

"%PYTHON%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

echo.
echo [DocPilot] server stopped.
pause
