#!/usr/bin/env python3
"""Validate dLive connection inputs without DNS, sockets, or MIDI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_client import validate_dlive_connection_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a dLive MIDI/TCP endpoint without contacting it."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--endpoint", choices=("surface", "mixrack"), required=True)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--midi-base-channel", type=int, default=0)
    parser.add_argument("--user-profile", type=int, default=0)
    args = parser.parse_args()

    try:
        report = validate_dlive_connection_config(
            host=args.host,
            endpoint=args.endpoint,
            tls=args.tls,
            port=args.port,
            midi_base_channel=args.midi_base_channel,
            user_profile=args.user_profile,
        )
    except ValueError as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
