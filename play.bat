@echo off
setlocal

set VENV="%~dp0venv"
set PYTHON=%VENV%\Scripts\python.exe
set PIP=%VENV%\Scripts\pip.exe

REM -- Check Python is available ------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM -- Create venv on first run -------------------------------------------------
if not exist %VENV% (
    echo First run: creating virtual environment...
    python -m venv %VENV%
    if errorlevel 1 ( echo Failed to create venv. & pause & exit /b 1 )
)

REM -- Install / upgrade dependencies -------------------------------------------
%PIP% show pygame >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies -- this takes a few minutes on first run...
    %PIP% install -r "%~dp0requirements.txt" -q
    if errorlevel 1 ( echo Dependency install failed. & pause & exit /b 1 )
)

REM -- Launch game --------------------------------------------------------------
echo Launching Ultimate Tic-Tac-Toe...
%PYTHON% "%~dp0mlp\play.py"
endlocal
