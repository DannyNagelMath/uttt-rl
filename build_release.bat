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

REM Copy model to a fixed name so play.py can find it inside the bundle
copy /Y "%MODEL%" best_model.zip

pyinstaller ^
  --name uttt-agent ^
  --onedir ^
  --windowed ^
  --runtime-hook hook_torch_dll.py ^
  --add-data "best_model.zip;." ^
  --paths mlp ^
  --paths . ^
  %ENTRY%

del best_model.zip

echo.
if exist dist\uttt-agent\uttt-agent.exe (
    echo Build succeeded: dist\uttt-agent\uttt-agent.exe
) else (
    echo Build FAILED - check output above
)
