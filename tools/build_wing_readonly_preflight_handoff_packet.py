#!/usr/bin/env python3
"""Build an offline operator packet for the WING read-only preflight step."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import no_wing_release_check

DEFAULT_URL = "ws://localhost:8765"
DEFAULT_CHANNELS = "3"
DEFAULT_VALUE_TYPES = "fader,gain,pan"
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
DEFAULT_REVIEW_OUTPUT = "Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md"
DEFAULT_DECISION_TEMPLATE_OUTPUT = (
    "artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json"
)
DEFAULT_DECISION_ROOT = "artifacts/wing_readonly_preflight_review_decisions"
DEFAULT_DECISION_REVIEW_OUTPUT = "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
DEFAULT_PACKET_OUTPUT = "Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md"
DEFAULT_PLAN_OUTPUT = "artifacts/wing_readonly_preflight_plan/latest_plan.json"
MAX_WING_INPUT_CHANNEL = 48
FOCUSED_VALIDATION_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "-m",
    "pytest",
    "-q",
    "tests/test_wing_readonly_preflight_evidence.py",
    "tests/test_wing_readonly_preflight_review_decision.py",
    "tests/test_wing_readonly_preflight_review_tool.py",
    "tests/test_wing_readonly_preflight_review_decision_tool.py",
    "tests/test_no_wing_release_check.py",
]
BACKEND_PORT_PRESTART_CHECK_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
]
FORBIDDEN_LIVE_REQUEST_TYPES = list(
    no_wing_release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
)
REQUIRED_REVIEW_CHECKLIST_IDS = list(
    no_wing_release_check.WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
)
NOT_CLOSED_BLOCKERS = list(
    no_wing_release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_channels(channels: str) -> list[int]:
    parsed: list[int] = []
    for item in _split_csv(channels):
        try:
            channel = int(item)
        except ValueError as exc:
            raise ValueError("channels_must_be_comma_separated_integers") from exc
        if channel <= 0:
            raise ValueError("channels_must_be_positive")
        if channel > MAX_WING_INPUT_CHANNEL:
            raise ValueError("channels_must_be_in_1_48")
        parsed.append(channel)
    if not parsed:
        raise ValueError("at_least_one_channel_required")
    return parsed


def capture_command(
    *,
    url: str,
    channels: str,
    value_types: str,
    artifact_root: str,
    expected_plan: str,
) -> list[str]:
    return [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        url,
        "--channels",
        channels,
        "--operator-confirm-channels",
        channels,
        "--operator-confirm-session-ready",
        "--value-types",
        value_types,
        "--require-physical-readback",
        "--artifact-root",
        artifact_root,
        "--expected-plan",
        expected_plan,
    ]


def plan_command(
    *,
    url: str,
    channels: str,
    value_types: str,
    artifact_root: str,
    plan_output: str,
) -> list[str]:
    return [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        url,
        "--channels",
        channels,
        "--operator-confirm-channels",
        channels,
        "--operator-confirm-session-ready",
        "--value-types",
        value_types,
        "--require-physical-readback",
        "--artifact-root",
        artifact_root,
        "--plan-only",
        "--plan-output",
        plan_output,
    ]


def artifact_review_command(
    *,
    artifact_root: str,
    review_output: str,
    decision_template_output: str,
) -> list[str]:
    return [
        "python3",
        "tools/review_wing_readonly_preflight_artifacts.py",
        "--artifact-root",
        artifact_root,
        "--output",
        review_output,
        "--decision-template-output",
        decision_template_output,
    ]


def decision_review_command(
    *,
    artifact_root: str,
    decision_root: str,
    decision_review_output: str,
) -> list[str]:
    return [
        "python3",
        "tools/review_wing_readonly_preflight_decisions.py",
        "--artifact-root",
        artifact_root,
        "--decision-root",
        decision_root,
        "--output",
        decision_review_output,
    ]


def build_handoff_packet(
    *,
    url: str = DEFAULT_URL,
    channels: str = DEFAULT_CHANNELS,
    value_types: str = DEFAULT_VALUE_TYPES,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    review_output: str = DEFAULT_REVIEW_OUTPUT,
    decision_template_output: str = DEFAULT_DECISION_TEMPLATE_OUTPUT,
    decision_root: str = DEFAULT_DECISION_ROOT,
    decision_review_output: str = DEFAULT_DECISION_REVIEW_OUTPUT,
    plan_output: str = DEFAULT_PLAN_OUTPUT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    channel_numbers = _validate_channels(channels)
    preview = plan_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        plan_output=plan_output,
    )
    capture = capture_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        expected_plan=plan_output,
    )
    review = artifact_review_command(
        artifact_root=artifact_root,
        review_output=review_output,
        decision_template_output=decision_template_output,
    )
    decision = decision_review_command(
        artifact_root=artifact_root,
        decision_root=decision_root,
        decision_review_output=decision_review_output,
    )
    return {
        "type": "wing_readonly_preflight_handoff_packet",
        "generated_at": generated_at or utc_now(),
        "status": "operator_hardware_step_required",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_failure_detail_fields": [
            "listener_processes",
            "operator_next_step",
        ],
        "manual_review_required_before_readiness_update": True,
        "execution_allowed_by_packet": False,
        "readiness_matrix_update_allowed_by_packet": False,
        "readiness_closure_allowed_by_packet": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "url": url,
        "channels": channel_numbers,
        "value_types": _split_csv(value_types),
        "artifact_root": artifact_root,
        "decision_root": decision_root,
        "expected_run_artifact_glob": f"{artifact_root}/<run_id>/summary.json",
        "expected_event_log_glob": f"{artifact_root}/<run_id>/events.jsonl",
        "plan_output": plan_output,
        "decision_template_output": decision_template_output,
        "review_output": review_output,
        "decision_review_output": decision_review_output,
        "commands": {
            "preview_readonly_request_plan": preview,
            "capture_readonly_preflight": capture,
            "review_artifact_and_write_pending_decision_template": review,
            "review_edited_decision_json": decision,
            "prestart_backend_port_check": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
            "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
        },
        "operator_sequence": [
            (
                "Before starting the intended backend/WING session, run "
                "prestart_backend_port_check and resolve only confirmed "
                "unexpected listeners."
            ),
            "Run preview_readonly_request_plan and confirm the request sequence is read-only.",
            "Confirm backend is connected to the intended WING Rack and ready for read-only inspection.",
            "Confirm the selected channel numbers are safe to inspect.",
            "Run capture_readonly_preflight.",
            "Run review_artifact_and_write_pending_decision_template.",
            "Edit the pending decision template into an accepted or rejected JSON record.",
            "Run review_edited_decision_json.",
            "Only after accepted_decision_ready, move to WING_SUPERVISED_FADER_GAIN_PROOF.",
        ],
        "required_review_checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
        "forbidden_live_request_types": list(FORBIDDEN_LIVE_REQUEST_TYPES),
        "blockers_not_closed_by_packet": list(NOT_CLOSED_BLOCKERS),
        "next_manual_artifact_required": "real wing_readonly_preflight.v1 summary.json",
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    commands = packet.get("commands") if isinstance(packet.get("commands"), Mapping) else {}
    lines = [
        "# WING Read-Only Preflight Handoff Packet",
        "",
        f"Generated: `{packet['generated_at']}`",
        f"Status: `{packet['status']}`",
        "",
        "## Safety Boundary",
        "",
        f"- read_only: `{str(packet['read_only']).lower()}`",
        f"- live_write: `{str(packet['live_write']).lower()}`",
        f"- mutates_console_state: `{str(packet['mutates_console_state']).lower()}`",
        f"- queries_console: `{str(packet['queries_console']).lower()}`",
        f"- executes_tests: `{str(packet['executes_tests']).lower()}`",
        f"- operator_required: `{str(packet['operator_required']).lower()}`",
        "- expected_plan_required_for_real_run: "
        f"`{str(packet['expected_plan_required_for_real_run']).lower()}`",
        "- physical_preflight_requires_existing_backend_connection: "
        f"`{str(packet['physical_preflight_requires_existing_backend_connection']).lower()}`",
        "- backend_port_prestart_check_required: "
        f"`{str(packet['backend_port_prestart_check_required']).lower()}`",
        "- backend_port_prestart_check_must_pass_before_starting_backend: "
        f"`{str(packet['backend_port_prestart_check_must_pass_before_starting_backend']).lower()}`",
        "- manual_review_required_before_readiness_update: "
        f"`{str(packet['manual_review_required_before_readiness_update']).lower()}`",
        "- execution_allowed_by_packet: "
        f"`{str(packet['execution_allowed_by_packet']).lower()}`",
        "- readiness_matrix_update_allowed_by_packet: "
        f"`{str(packet['readiness_matrix_update_allowed_by_packet']).lower()}`",
        "- readiness_closure_allowed_by_packet: "
        f"`{str(packet['readiness_closure_allowed_by_packet']).lower()}`",
        f"- live_authority_granted: `{str(packet['live_authority_granted']).lower()}`",
        f"- full_autonomous_ready: `{str(packet['full_autonomous_ready']).lower()}`",
        "",
        "## Operator Sequence",
        "",
    ]
    for index, item in enumerate(packet.get("operator_sequence") or [], start=1):
        lines.append(f"{index}. {item}")
    lines.extend(["", "## Commands", ""])
    for name in (
        "prestart_backend_port_check",
        "preview_readonly_request_plan",
        "capture_readonly_preflight",
        "review_artifact_and_write_pending_decision_template",
        "review_edited_decision_json",
        "focused_validation",
    ):
        command = commands.get(name)
        if isinstance(command, list):
            lines.extend([f"### {name}", "", "```bash", _command_text(command), "```", ""])
    lines.extend(
        [
            "## Expected Artifacts",
            "",
            f"- summary: `{packet['expected_run_artifact_glob']}`",
            f"- events: `{packet['expected_event_log_glob']}`",
            f"- pending decision template: `{packet['decision_template_output']}`",
            f"- artifact review report: `{packet['review_output']}`",
            f"- decision review report: `{packet['decision_review_output']}`",
            "",
            "## Required Checklist",
            "",
        ]
    )
    for item in packet.get("required_review_checklist_ids") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Forbidden Request Types", ""])
    for item in packet.get("forbidden_live_request_types") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Blockers Not Closed By This Packet", ""])
    for item in packet.get("blockers_not_closed_by_packet") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next Required Artifact", "", str(packet["next_manual_artifact_required"]), ""])
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline operator handoff packet for WING read-only preflight."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--channels", default=DEFAULT_CHANNELS)
    parser.add_argument("--value-types", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--review-output", default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--decision-template-output", default=DEFAULT_DECISION_TEMPLATE_OUTPUT)
    parser.add_argument("--decision-root", default=DEFAULT_DECISION_ROOT)
    parser.add_argument("--decision-review-output", default=DEFAULT_DECISION_REVIEW_OUTPUT)
    parser.add_argument("--plan-output", default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=DEFAULT_PACKET_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        packet = build_handoff_packet(
            url=args.url,
            channels=args.channels,
            value_types=args.value_types,
            artifact_root=args.artifact_root,
            review_output=args.review_output,
            decision_template_output=args.decision_template_output,
            decision_root=args.decision_root,
            decision_review_output=args.decision_review_output,
            plan_output=args.plan_output,
        )
    except ValueError as exc:
        if not args.quiet:
            print(str(exc))
        return 2
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
