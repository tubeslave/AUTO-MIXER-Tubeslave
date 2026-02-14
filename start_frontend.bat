@echo off
echo === Auto Mixer Tubeslave - Starting Frontend ===
echo.

cd frontend

if not exist node_modules (
    echo Installing npm dependencies...
    call npm install
)

echo.
echo Starting React development server...
call npm start

pause
