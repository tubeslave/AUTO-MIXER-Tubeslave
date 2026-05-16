@echo off
echo === Auto Mixer Tubeslave - Starting Backend ===
echo.

cd /d %~dp0

if not exist backend\venv (
    echo Creating virtual environment...
    python -m venv backend\venv
)

echo Activating virtual environment...
call backend\venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r backend\requirements.txt

echo.
echo Starting WebSocket server...
python -m backend.server

pause
