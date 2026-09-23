#!/usr/bin/env python3
"""Controller-backed STUDIO file mastering; no console or online-model access."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_workbench.mastering.offline import deliver_master


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--name', default='Master')
    parser.add_argument('--target-lufs', type=float, default=-14.5)
    parser.add_argument('--ceiling-dbtp', type=float, default=-1.2)
    args = parser.parse_args()
    result = deliver_master(args.source, args.output_dir, name=args.name,
                            target_lufs=args.target_lufs, ceiling_dbtp=args.ceiling_dbtp)
    print(json.dumps({'status': result['status'], 'artifacts': result['artifacts'],
                      'baseline_eligible': False}, indent=2))
    return 0 if result['status'] == 'pending_human_review' else 2


if __name__ == '__main__':
    raise SystemExit(main())
