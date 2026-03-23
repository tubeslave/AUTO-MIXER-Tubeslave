#!/bin/bash

set -e

echo "======================================"
echo "  AUTO MIXER Tubeslave - Build .app  "
echo "======================================"
echo ""

echo "Step 1/3: Building Python backend..."
./build_backend.sh

echo ""
echo "Step 2/3: Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi

echo ""
echo "Step 3/3: Building Electron app..."
npm run dist

echo ""
echo "======================================"
echo "  ✓ Build Complete!"
echo "======================================"
echo ""
echo "Output: frontend/dist/AUTO MIXER Tubeslave.app"
echo ""
echo "To install:"
echo "  cp -r \"frontend/dist/AUTO MIXER Tubeslave.app\" /Applications/"
echo ""
