#!/bin/bash

echo "Building Python backend with PyInstaller..."

cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Building executable with PyInstaller..."
venv/bin/pyinstaller --clean server.spec

if [ -f "dist/server" ]; then
    echo "✓ Backend built successfully: backend/dist/server"
else
    echo "✗ Build failed!"
    deactivate
    exit 1
fi

deactivate
cd ..

echo "Backend build complete!"