@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Training.cmd first.
  pause
  exit /b 1
)
set "STARFALL_RECORDING=%~1"
if not defined STARFALL_RECORDING set "STARFALL_RECORDING=videoscreenstafall.mp4"
if not exist "%STARFALL_RECORDING%" (
  echo Drag a recorded gameplay video onto this launcher.
  pause
  exit /b 1
)
if not exist "weights\starfall-detector-1988\best_openvino_model\best.xml" (
  echo Preparing the trained detector for CPU preview...
  ".venv\Scripts\python.exe" -c "import torch; from ultralytics import YOLO; torch.set_num_threads(2); YOLO('weights/starfall-detector-1988/best.pt').export(format='openvino', imgsz=416, batch=1, dynamic=False)"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
echo Showing the recording with large detection boxes. Press q to finish early.
echo The annotated MP4 is saved in reports. The original recording is preserved.
".venv\Scripts\python.exe" -m starfall.cli preview-video "%STARFALL_RECORDING%"
pause
