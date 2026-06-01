#!/usr/bin/env bash
# Capture a short UDP/OSC packet trace for Automixer latency smoke tests.

set -u

OUTPUT="reports/latency/bench_smoke.pcap"
STATUS_PATH=""
PORT="30223"
DURATION="5"
IFACE=""

usage() {
  cat <<'EOF'
Usage: tools/latency/capture_network.sh [options]

Options:
  --output PATH      pcap output path (default: reports/latency/bench_smoke.pcap)
  --status PATH      JSON status path (default: OUTPUT.status.json)
  --port PORT        benchmark OSC UDP port to include (default: 30223)
  --duration SEC     capture duration in seconds (default: 5)
  --iface IFACE      tcpdump interface (default: lo0 on macOS, lo elsewhere)
  -h, --help         show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --status)
      STATUS_PATH="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --iface)
      IFACE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$STATUS_PATH" ]; then
  STATUS_PATH="${OUTPUT}.status.json"
fi

if [ -z "$IFACE" ]; then
  case "$(uname -s)" in
    Darwin) IFACE="lo0" ;;
    *) IFACE="lo" ;;
  esac
fi

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$STATUS_PATH")"

write_empty_pcap() {
  python3 - "$OUTPUT" <<'PY'
from pathlib import Path
import struct
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("wb") as handle:
    handle.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
PY
}

write_status() {
  local status="$1"
  local message="$2"
  local exit_code="$3"
  local log_path="${4:-}"
  python3 - "$STATUS_PATH" "$status" "$message" "$exit_code" "$OUTPUT" "$IFACE" "$PORT" "$DURATION" "$log_path" <<'PY'
from pathlib import Path
import json
import os
import sys

status_path = Path(sys.argv[1])
status, message, exit_code = sys.argv[2], sys.argv[3], int(sys.argv[4])
output, iface, port, duration, log_path = sys.argv[5:10]
pcap = Path(output)
log_text = ""
if log_path:
    try:
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        log_text = ""

payload = {
    "status": status,
    "message": message,
    "exit_code": exit_code,
    "output": output,
    "output_exists": pcap.exists(),
    "output_size_bytes": pcap.stat().st_size if pcap.exists() else 0,
    "interface": iface,
    "port": int(port),
    "duration_s": float(duration),
    "filter": f"udp and (port {port} or port 2222 or port 2223)",
    "tcpdump_log_tail": log_text,
}
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if ! command -v tcpdump >/dev/null 2>&1; then
  write_empty_pcap
  write_status "tcpdump_missing" "tcpdump is not installed; wrote an empty pcap placeholder." 0 ""
  echo "tcpdump missing; wrote empty pcap placeholder at $OUTPUT"
  exit 0
fi

rm -f "$OUTPUT"
LOG_PATH="$(mktemp "${TMPDIR:-/tmp}/automixer_tcpdump.XXXXXX.log")"
FILTER="udp and (port ${PORT} or port 2222 or port 2223)"

tcpdump -i "$IFACE" -nn -s 0 -U -w "$OUTPUT" "$FILTER" >"$LOG_PATH" 2>&1 &
TCPDUMP_PID=$!

sleep "$DURATION"

if kill -0 "$TCPDUMP_PID" >/dev/null 2>&1; then
  kill -INT "$TCPDUMP_PID" >/dev/null 2>&1 || true
fi

wait "$TCPDUMP_PID"
TCPDUMP_STATUS=$?

if [ ! -s "$OUTPUT" ]; then
  write_empty_pcap
  write_status "tcpdump_failed" "tcpdump did not produce a pcap; wrote an empty placeholder." "$TCPDUMP_STATUS" "$LOG_PATH"
  echo "tcpdump failed; wrote empty pcap placeholder at $OUTPUT"
  exit 0
fi

write_status "captured" "tcpdump completed and wrote pcap." "$TCPDUMP_STATUS" "$LOG_PATH"
echo "pcap written to $OUTPUT"
