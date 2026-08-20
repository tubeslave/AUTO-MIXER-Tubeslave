#!/usr/bin/env python3
"""Review an operator-supplied dLive MIDI capture without opening transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from dlive_midi_capture_evidence import review_dlive_fader_midi_capture  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--endpoint", choices=("surface", "mixrack"), required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--midi-base-channel", type=int, default=11)
    parser.add_argument("--user-profile", type=int, default=0)
    parser.add_argument("--input-channel", type=int, required=True)
    parser.add_argument("--target-db", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _write_output(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        report = review_dlive_fader_midi_capture(
            capture,
            host=args.host,
            endpoint=args.endpoint,
            port=args.port,
            tls=args.tls,
            midi_base_channel=args.midi_base_channel,
            user_profile=args.user_profile,
            input_channel=args.input_channel,
            target_db=args.target_db,
        )
        if args.output is not None:
            _write_output(args.output, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema": "automixer.dlive_midi_capture_review.v1",
            "status": "blocked",
            "reason": str(exc) or type(exc).__name__,
            "console_contacted_by_automixer": False,
            "midi_sent_by_automixer": False,
            "mutation_write_attempt_count": 0,
            "live_write": False,
            "execution_allowed": False,
            "live_authority_granted": False,
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
