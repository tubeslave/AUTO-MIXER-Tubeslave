#!/usr/bin/env python3
"""Write offline smart DSP replay/evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from smart_dsp_replay_evaluation import (  # noqa: E402
    evaluate_smart_dsp_replay_scenarios,
    utc_now,
    write_smart_dsp_replay_evaluation_artifacts,
)


DEFAULT_ARTIFACT_ROOT = Path("artifacts/smart_dsp_replay_evaluation")


def _default_run_id() -> str:
    return utc_now().replace("-", "").replace(":", "").replace("Z", "Z_smart_dsp_replay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root directory for generated offline artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id directory name. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="Required pass rate for the default scenario pack.",
    )
    parser.add_argument(
        "--generated-at-utc",
        default=None,
        help="Optional deterministic timestamp for tests/reports.",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or _default_run_id()
    output_dir = args.artifact_root / run_id
    report = evaluate_smart_dsp_replay_scenarios(
        generated_at_utc=args.generated_at_utc,
        min_pass_rate=args.min_pass_rate,
    )
    paths = write_smart_dsp_replay_evaluation_artifacts(
        report=report,
        output_dir=output_dir,
    )
    print(paths["summary_json"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
