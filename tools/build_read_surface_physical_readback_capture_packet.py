#!/usr/bin/env python3
"""Build an offline operator packet for read-surface physical readback capture."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.read_surface_physical_readback_capture_plan import (  # noqa: E402
    build_read_surface_physical_readback_capture_plan,
)


DEFAULT_OUTPUT = "Docs/reports/READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PACKET_LATEST.md"
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_read_surface_physical_readback_capture_plan.py",
    "tests/test_no_wing_release_check.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def build_capture_packet(
    *,
    capture_plan: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    plan = (
        capture_plan
        if capture_plan is not None
        else build_read_surface_physical_readback_capture_plan()
    )
    summary = _mapping(plan.get("summary"))
    safety = _mapping(plan.get("safety"))
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    step = steps[0] if steps and isinstance(steps[0], Mapping) else {}
    physical_ids = _string_list(summary.get("physical_fixture_missing_surface_ids"))
    next_surface_id = summary.get("next_physical_readback_surface_id")
    next_action_id = summary.get("next_physical_readback_action_id")
    packet_status = (
        "operator_capture_required"
        if next_surface_id
        else "read_surface_physical_readback_complete"
    )
    return {
        "type": "read_surface_physical_readback_capture_packet",
        "generated_at": generated_at or utc_now(),
        "status": packet_status,
        "source_plan_status": plan.get("status"),
        "source_method_id": plan.get("method_id"),
        "read_only": safety.get("read_only") is True,
        "live_write": safety.get("live_write") is True,
        "mutates_console_state": safety.get("mutates_console_state") is True,
        "queries_console": safety.get("queries_console") is True,
        "executes_tests": safety.get("executes_tests") is True,
        "operator_required": safety.get("operator_required") is True,
        "execution_allowed_by_packet": False,
        "live_widening_allowed": safety.get("live_widening_allowed") is True,
        "live_authority_granted": safety.get("live_authority_granted") is True,
        "full_autonomous_ready": safety.get("full_autonomous_ready") is True,
        "physical_fixture_missing_surface_ids": physical_ids,
        "physical_gap_surface_count": len(physical_ids),
        "next_physical_readback_surface_id": next_surface_id,
        "next_physical_readback_action_id": next_action_id,
        "target_action_consistency": summary.get("target_action_consistency") is True,
        "capture_step_count": int(summary.get("capture_step_count") or len(steps)),
        "required_artifact_contracts": _string_list(
            summary.get("required_artifact_contracts")
        ),
        "blocked_by": _string_list(summary.get("blocked_by")),
        "step": dict(step) if step else {},
        "operator_sequence": [
            "Confirm the intended console, show state, and read-only capture scope.",
            "Use operator-approved read-only tooling outside this packet to capture the physical readback artifact.",
            "Reject the artifact if the capture path would transmit a mixer-control command.",
            "Validate the artifact against the required physical readback contract.",
            "Review the artifact manually before changing fixture baselines or autonomy readiness.",
        ],
        "commands": {
            "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
        },
    }


def _bool(value: Any) -> str:
    return str(bool(value)).lower()


def render_markdown(packet: Mapping[str, Any]) -> str:
    step = _mapping(packet.get("step"))
    commands = _mapping(packet.get("commands"))
    lines = [
        "# Read Surface Physical Readback Capture Packet",
        "",
        f"Generated: `{packet['generated_at']}`",
        f"Status: `{packet['status']}`",
        "",
        "## Safety Boundary",
        "",
        f"- read_only: `{_bool(packet['read_only'])}`",
        f"- live_write: `{_bool(packet['live_write'])}`",
        f"- mutates_console_state: `{_bool(packet['mutates_console_state'])}`",
        f"- queries_console: `{_bool(packet['queries_console'])}`",
        f"- executes_tests: `{_bool(packet['executes_tests'])}`",
        f"- operator_required: `{_bool(packet['operator_required'])}`",
        "- execution_allowed_by_packet: "
        f"`{_bool(packet['execution_allowed_by_packet'])}`",
        f"- live_widening_allowed: `{_bool(packet['live_widening_allowed'])}`",
        f"- live_authority_granted: `{_bool(packet['live_authority_granted'])}`",
        f"- full_autonomous_ready: `{_bool(packet['full_autonomous_ready'])}`",
        "",
        "## Next Capture Target",
        "",
        f"- surface_id: `{packet['next_physical_readback_surface_id']}`",
        f"- remediation_action_id: `{packet['next_physical_readback_action_id']}`",
        f"- capture_step_count: `{packet['capture_step_count']}`",
        f"- physical_gap_surface_count: `{packet['physical_gap_surface_count']}`",
        f"- target_action_consistency: `{_bool(packet['target_action_consistency'])}`",
        "- required_artifact_contracts: "
        f"`{json.dumps(packet['required_artifact_contracts'])}`",
        f"- blocked_by: `{json.dumps(packet['blocked_by'])}`",
        "",
        "## Operator Sequence",
        "",
    ]
    for index, item in enumerate(packet.get("operator_sequence") or [], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(["", "## Step Details", ""])
    if step:
        lines.extend(
            [
                f"- target_console_surface: `{step.get('target_console_surface')}`",
                f"- read_mode: `{step.get('read_mode')}`",
                f"- contract_method_id: `{step.get('contract_method_id')}`",
                "- expected_artifacts: "
                f"`{json.dumps(step.get('expected_artifacts') or [])}`",
                "- blockers_to_clear: "
                f"`{json.dumps(step.get('blockers_to_clear') or [])}`",
                "",
                "## Forbidden Actions",
                "",
            ]
        )
        for item in step.get("forbidden_actions") or []:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    focused_validation = commands.get("focused_validation")
    if isinstance(focused_validation, list):
        lines.extend(
            [
                "",
                "## Focused Validation",
                "",
                "```bash",
                _command_text([str(item) for item in focused_validation]),
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline read-surface physical readback capture packet."
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_capture_packet()
    text = (
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(packet)
    )
    write_output(Path(args.output), text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
