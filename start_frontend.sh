#!/bin/bash

echo "=== Auto Mixer Tubeslave - Starting Frontend ==="
echo ""

cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

echo ""
echo "Starting React development server..."
npm start
