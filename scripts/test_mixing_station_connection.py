#!/usr/bin/env python3
"""Check Mixing Station Desktop REST API availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from integrations.mixing_station.rest_client import MixingStationRestClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    client = MixingStationRestClient(host=args.host, port=args.port)
    health = client.health_check()
    app_state = client.get_app_state() if health.success else health
    mixer_state = client.get_mixer_state() if health.success else None
    print(json.dumps({
        "health": health.__dict__,
        "app_state": app_state.__dict__,
        "mixer_state": mixer_state.__dict__ if mixer_state else None,
    }, indent=2, ensure_ascii=True))
    return 0 if health.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
