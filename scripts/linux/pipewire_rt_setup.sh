#!/usr/bin/env bash
#
# Linux audio: PipeWire / real-time hints for low-latency capture.
# Run review steps manually; this script does not modify system files.
#
# References: pipewire wiki (rt kit), limits.conf for @audio rtprio/memlock.

set -euo pipefail

echo "=== PipeWire / RT quick check (read-only) ==="
echo

if command -v pw-cli >/dev/null 2>&1; then
  echo "--- pw-cli info (if available) ---"
  pw-cli info 2>/dev/null | head -40 || true
else
  echo "pw-cli not found (install pipewire or use distro packages)."
fi

echo
echo "--- Current user limits (RLIMIT) ---"
ulimit -a 2>/dev/null || true

echo
echo "Suggested manual steps (not executed here):"
echo "  1. Add user to group 'audio' (and 'pipewire' if your distro uses it)."
echo "  2. In /etc/security/limits.d/99-audio.conf (example):"
echo "       @audio   -  rtprio     95"
echo "       @audio   -  memlock    unlimited"
echo "  3. Ensure WirePlumber / pipewire-pulse is running; prefer pipewire-jack for JACK apps."
echo "  4. For this project, capture uses sounddevice/PortAudio — use PW's ALSA or"
echo "     JACK device that maps to your Dante/virtual card."
echo "  5. Set AUTOMIXER audio.source to 'pipewire' in config (same code path as sounddevice)."
echo
