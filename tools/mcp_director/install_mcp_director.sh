#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="$SCRIPT_DIR/server.js"

if [[ ! -x /usr/bin/python3 ]]; then
  echo "automixer_mcp_install_failed: /usr/bin/python3 is required" >&2
  exit 1
fi

node --check "$SERVER"

cat <<JSON
{
  "mcpServers": {
    "automixer-wing-lab": {
      "command": "node",
      "args": ["$SERVER"],
      "env": {
        "AUTOMIXER_WING_LAB_AUTOWRITE": "1"
      }
    }
  }
}
JSON
