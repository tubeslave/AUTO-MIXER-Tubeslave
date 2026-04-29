"""CLI for `python -m automixer.production_mix_v1`."""

from __future__ import annotations

import argparse
import json

from .config import load_production_mix_config
from .models import MODE_LIVE, MODE_OFFLINE
from .pipeline import ProductionMixPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production_mix_v1 safely")
    parser.add_argument("--config", default="", help="Pipeline config YAML path")
    parser.add_argument("--input-dir", default="", help="Directory of audio stems")
    parser.add_argument("--output-dir", default="production_mix_v1_out")
    parser.add_argument("--mode", choices=[MODE_OFFLINE, MODE_LIVE], default=MODE_OFFLINE)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live-apply", action="store_true", help="Disable dry-run flag only")
    parser.add_argument("--smoke", action="store_true", help="Use generated smoke stems")
    parser.add_argument(
        "--enable-optional-critics",
        action="store_true",
        help="Try MuQ/MERT/Essentia adapters with local/fallback behavior",
    )
    args = parser.parse_args(argv)

    dry_run = True
    if args.live_apply:
        dry_run = False
    config = load_production_mix_config(args.config or None)
    pipeline = ProductionMixPipeline(
        config,
        enable_optional_critics=bool(args.enable_optional_critics),
    )
    result = pipeline.run(
        input_dir=args.input_dir or None,
        output_dir=args.output_dir,
        mode=args.mode,
        dry_run=dry_run,
        smoke=bool(args.smoke),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "selected": result.decision.selected.candidate.name,
                "accepted": result.decision.accepted,
                "dry_run": result.dry_run,
                "artifacts": result.artifacts,
                "warnings": result.warnings,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
