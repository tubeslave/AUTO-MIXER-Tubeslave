#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="8787"
HOST="${HOST:-127.0.0.1}"

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. I will not stop any process."
  echo "Inspect it with: lsof -nP -iTCP:${PORT} -sTCP:LISTEN"
  echo "If you decide it is safe, stop the owner yourself, for example: kill <PID>"
  exit 1
fi

exec env HOST="${HOST}" PORT="${PORT}" node server.js
