#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-external/diffvox}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -e "$TARGET_DIR" ]; then
  echo "Target already exists: $TARGET_DIR"
  exit 0
fi

git clone --depth 1 https://github.com/SonyResearch/diffvox.git "$TARGET_DIR"
"$PYTHON_BIN" -m venv "$TARGET_DIR/.venv"
"$TARGET_DIR/.venv/bin/python" -m pip install --upgrade pip
"$TARGET_DIR/.venv/bin/python" -m pip install -r "$TARGET_DIR/requirements.txt"

echo "DiffVox installed at $TARGET_DIR"
echo "Use it only in offline proposal mode until A/B validation passes."
