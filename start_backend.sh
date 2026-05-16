#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Auto Mixer Tubeslave - Starting Backend ==="
echo ""

if [ ! -d "backend/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv backend/venv
fi

echo "Activating virtual environment..."
source backend/venv/bin/activate

echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "Starting WebSocket server..."
python -m backend.server
