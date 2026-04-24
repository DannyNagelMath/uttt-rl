@echo off
REM Build the Windows release executable using PyInstaller.
REM Run this from the project root with the venv activated:
REM   venv\Scripts\activate
REM   build_release.bat

REM Requires pyinstaller: pip install pyinstaller
REM Output: dist\uttt-agent\uttt-agent.exe  (folder mode for fast startup)

set MODEL=best_models\run2_selfplay_500000.zip
set ENTRY=mlp\play.py

echo Building release with model: %MODEL%
echo.

pyinstaller ^
  --name uttt-agent ^
  --onedir ^
  --windowed ^
  --add-data "%MODEL%;." ^
  --add-data "uttt_game.py;." ^
  --add-data "mlp\uttt_env.py;." ^
  --add-data "mlp\utils.py;." ^
  --paths mlp ^
  --paths . ^
  %ENTRY%

echo.
if exist dist\uttt-agent\uttt-agent.exe (
    echo Build succeeded: dist\uttt-agent\uttt-agent.exe
) else (
    echo Build FAILED - check output above
)
