"""CLI entrypoint for the offline AI mixing test chain."""

from __future__ import annotations

import argparse
import json

from .runner import OfflineTestRunner


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m ai_mixing_pipeline.offline_test_runner")
    parser.add_argument("--input", default="offline_test_input", help="Input offline test directory")
    parser.add_argument("--output", default="offline_test_output", help="Output offline test directory")
    parser.add_argument("--config", default="configs/ai_mixing_roles.yaml", help="Roles config YAML")
    parser.add_argument(
        "--mode",
        default="offline_test",
        choices=["observe", "suggest", "offline_test", "shadow_mix", "assisted_offline"],
        help="Pipeline mode",
    )
    args = parser.parse_args()
    runner = OfflineTestRunner(
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
        mode=args.mode,
    )
    result = runner.run()
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
