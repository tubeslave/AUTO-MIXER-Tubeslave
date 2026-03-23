#!/usr/bin/env python3
"""
Calibrate Ableton Track Delay UI coordinates from current mouse position.

DEPRECATED: UI automation отключён для Ableton (не работает должным образом).
Track Delay настраивайте вручную. Скрипт оставлен для справки.

Usage (если UI automation будет восстановлен):
  1. Open Ableton Live and make sure the target window is visible.
  2. Click the Track Delay field for channel 1 (or another channel).
  3. Run:
       cd backend && PYTHONPATH=. python ../scripts/calibrate_ableton_track_delay.py --channel 1
  4. Copy the printed JSON block into `config/default_config.json` or user config.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ableton_client import AbletonClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=1, help="Channel whose Track Delay field is focused")
    args = parser.parse_args()

    client = AbletonClient()
    if not client.calibrate_track_delay_ui_from_mouse(channel=args.channel):
        print("Calibration failed.")
        return 1

    payload = {
        "ableton_ui": {
            "window_title": client.live_window_title,
            "track_delay_x_ratio": round(client.track_delay_x_ratio, 6),
            "track_delay_first_row_center_y_ratio": round(
                client.track_delay_first_row_center_y_ratio, 6
            ),
            "track_delay_row_pitch_y_ratio": round(client.track_delay_row_pitch_y_ratio, 6),
            "track_delay_base_channel": client.track_delay_base_channel,
        }
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
