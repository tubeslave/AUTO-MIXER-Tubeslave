from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_ops.architect_agent import ArchitectAgent
from src.agent_ops.coordinator import Coordinator
from src.agent_ops.evaluator_agent import EvaluatorAgent
from src.agent_ops.trainer_agent import TrainerAgent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone agent operations.")
    parser.add_argument(
        "--task",
        choices=("train_pipeline", "discover_datasets"),
        default="train_pipeline",
    )
    parser.add_argument(
        "--report",
        default="/Users/dmitrijvolkov/Downloads/deep-research-report.md",
        help="Research report to attach to dataset discovery.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/training_datasets/audio_mixing",
        help="Dataset-discovery output directory.",
    )
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument("--search-query", action="append", default=[])
    parser.add_argument("--max-dataset-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--search-limit", type=int, default=8)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-convert-jsonl", action="store_true")
    return parser


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    if args.task == "discover_datasets":
        return {
            "task": "discover_datasets",
            "dataset_discovery": {
                "report_path": args.report,
                "output_dir": args.output_dir,
                "dataset_ids": args.dataset_id,
                "search_queries": args.search_query,
                "max_dataset_bytes": args.max_dataset_bytes,
                "timeout_sec": args.timeout_sec,
                "search_limit": args.search_limit,
                "download": not args.no_download,
                "convert_jsonl": not args.no_convert_jsonl,
            },
            "model_dir": "artifacts",
        }

    return {
        "train_config": {
            "mode": "script",
            "script_path": "scripts/train.py",
            "args": [],
        },
        "model_dir": "artifacts",
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    agents = {
        "architect": ArchitectAgent(),
        "trainer": TrainerAgent(project_root=PROJECT_ROOT),
        "evaluator": EvaluatorAgent(),
    }
    coordinator = Coordinator(agents)

    payload = build_payload(args)
    result = coordinator.run(args.task, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
