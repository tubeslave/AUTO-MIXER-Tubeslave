@echo off
echo === Auto Mixer Tubeslave - Starting Backend ===
echo.

cd backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting WebSocket server...
python server.py

pause
