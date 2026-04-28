#!/usr/bin/env python3
"""Replay Automixer-to-Mixing-Station JSONL logs for visual analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from integrations.mixing_station.adapter import MixingStationAdapter
from integrations.mixing_station.config import MixingStationConfig
from integrations.mixing_station.logger import correction_from_log_row
from integrations.mixing_station.models import AutomixCorrection


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_replay_corrections(path: str | Path, *, dry_run: bool) -> List[AutomixCorrection]:
    corrections: List[AutomixCorrection] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corrections.append(correction_from_log_row(row, dry_run=dry_run))
    return corrections


def replay(
    corrections: Iterable[AutomixCorrection],
    adapter: MixingStationAdapter,
    *,
    speed: float = 1.0,
) -> int:
    sent = 0
    previous_ts = None
    speed = max(0.01, float(speed))
    for correction in corrections:
        if previous_ts is not None:
            delay = (correction.timestamp - previous_ts).total_seconds() / speed
            if 0.0 < delay < 10.0:
                time.sleep(delay)
        previous_ts = correction.timestamp
        result = adapter.send_correction(correction)
        print(json.dumps(result.to_dict(), ensure_ascii=True))
        sent += int(result.success)
    return sent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/automix_to_mixing_station.jsonl")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--dry-run", default="true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--transport", default="websocket")
    args = parser.parse_args()

    dry_run = parse_bool(args.dry_run)
    config = MixingStationConfig.from_file()
    config.enabled = True
    config.dry_run = dry_run
    config.host = args.host
    config.rest_port = args.port
    config.transport = args.transport
    adapter = MixingStationAdapter(config)
    corrections = load_replay_corrections(args.log, dry_run=dry_run)
    sent = replay(corrections, adapter, speed=args.speed)
    return 0 if sent == len(corrections) else 1


if __name__ == "__main__":
    raise SystemExit(main())
