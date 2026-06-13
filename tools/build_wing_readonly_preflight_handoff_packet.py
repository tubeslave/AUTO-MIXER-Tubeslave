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

from backend.wing_readonly_preflight_contract import (  # noqa: E402
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS,
    WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED,
    WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_RUNNER_URL,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX,
    expected_backend_port_prestart_gate,
    expected_supervised_prestart_gate,
)
from tools import no_wing_release_check

DEFAULT_URL = WING_READONLY_PREFLIGHT_RUNNER_URL
DEFAULT_CHANNELS = WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
DEFAULT_VALUE_TYPES = WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
DEFAULT_ARTIFACT_ROOT = WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
DEFAULT_REVIEW_OUTPUT = WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
DEFAULT_DECISION_TEMPLATE_OUTPUT = (
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
)
DEFAULT_DECISION_ROOT = WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
DEFAULT_DECISION_REVIEW_OUTPUT = WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
DEFAULT_PACKET_OUTPUT = WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
DEFAULT_PLAN_OUTPUT = WING_READONLY_PREFLIGHT_PLAN_OUTPUT
DEFAULT_PRESTART_SUMMARY = WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
MAX_WING_INPUT_CHANNEL = 48
SUPPORTED_VALUE_TYPES = set(WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES)
EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE = {
    request_type: list(response_types)
    for request_type, response_types in (
        WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
    )
}
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
BACKEND_PORT_PRESTART_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
FORBIDDEN_LIVE_REQUEST_TYPES = list(
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
)
REQUIRED_REVIEW_CHECKLIST_IDS = list(
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
)
NOT_CLOSED_BLOCKERS = list(
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_value_types(value_types: str) -> list[str]:
    parsed = _split_csv(value_types)
    if not parsed:
        raise ValueError("value_types_must_include_at_least_one_value_type")
    if len(set(parsed)) != len(parsed):
        raise ValueError("value_types_must_not_contain_duplicates")
    unsupported = sorted(set(parsed) - SUPPORTED_VALUE_TYPES)
    if unsupported:
        raise ValueError(f"value_types_unsupported:{','.join(unsupported)}")
    return parsed


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
    prestart_summary: str,
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
        "--prestart-summary",
        prestart_summary,
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
    prestart_summary: str,
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
        "--prestart-summary",
        prestart_summary,
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
    prestart_summary: str = DEFAULT_PRESTART_SUMMARY,
    generated_at: str | None = None,
) -> dict[str, Any]:
    channel_numbers = _validate_channels(channels)
    value_type_names = _validate_value_types(value_types)
    supervised_gate_step_index = (
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX
    )
    supervised_gate_step = WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP
    prestart_step_index = WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX
    prestart_step = WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP
    mvp_blocker_anchor_digest_review_step = (
        "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
        "current first open blocker before supervised proof planning."
    )
    expected_prestart_gate = expected_backend_port_prestart_gate(
        operator_sequence_step=prestart_step,
        operator_sequence_step_index=prestart_step_index,
    )
    expected_supervised_gate = expected_supervised_prestart_gate(
        operator_sequence_step=supervised_gate_step,
        operator_sequence_step_index=supervised_gate_step_index,
    )
    preview = plan_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        plan_output=plan_output,
        prestart_summary=prestart_summary,
    )
    capture = capture_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        expected_plan=plan_output,
        prestart_summary=prestart_summary,
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
    mvp_blocker_anchor = no_wing_release_check.wing_mvp_first_open_blocker_anchor()
    mvp_blocker_anchor_sha256 = no_wing_release_check.wing_mvp_blocker_anchor_sha256(
        mvp_blocker_anchor
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
        "backend_port_prestart_summary_required": True,
        "backend_port_prestart_summary_path_placeholder": prestart_summary,
        "backend_port_prestart_check_operator_sequence_step_index": prestart_step_index,
        "backend_port_prestart_check_must_be_first_operator_step": True,
        "backend_port_prestart_check_command": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
        "backend_port_prestart_check_failure_detail_fields": list(
            BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "manual_review_required_before_readiness_update": True,
        "execution_allowed_by_packet": False,
        "readiness_matrix_update_allowed_by_packet": False,
        "readiness_closure_allowed_by_packet": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "url": url,
        "channels": channel_numbers,
        "value_types": value_type_names,
        "expected_response_types_by_request_type": dict(
            EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE
        ),
        "artifact_root": artifact_root,
        "decision_root": decision_root,
        "expected_run_artifact_glob": (
            f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
        ),
        "expected_event_log_glob": (
            f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
        ),
        "plan_output": plan_output,
        "decision_template_output": decision_template_output,
        "decision_template_output_matches_contract": (
            decision_template_output
            == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
        ),
        "review_output": review_output,
        "review_output_matches_contract": (
            review_output == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
        ),
        "decision_review_output": decision_review_output,
        "decision_review_output_matches_contract": (
            decision_review_output
            == WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
        ),
        "commands": {
            "preview_readonly_request_plan": preview,
            "capture_readonly_preflight": capture,
            "review_artifact_and_write_pending_decision_template": review,
            "review_edited_decision_json": decision,
            "prestart_backend_port_check": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
            "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
        },
        "mvp_blocker_anchor": mvp_blocker_anchor,
        "mvp_blocker_anchor_sha256": mvp_blocker_anchor_sha256,
        "operator_sequence": [
            prestart_step,
            "Run preview_readonly_request_plan and confirm the request sequence is read-only.",
            "Confirm backend is connected to the intended WING Rack and ready for read-only inspection.",
            "Confirm the selected channel numbers are safe to inspect.",
            "Run capture_readonly_preflight.",
            "Run review_artifact_and_write_pending_decision_template.",
            "Edit the pending decision template into an accepted or rejected JSON record.",
            "Run review_edited_decision_json.",
            mvp_blocker_anchor_digest_review_step,
            supervised_gate_step,
        ],
        "backend_port_prestart_gate": dict(expected_prestart_gate),
        "expected_backend_port_prestart_gate": dict(expected_prestart_gate),
        "supervised_prestart_gate": dict(expected_supervised_gate),
        "expected_supervised_prestart_gate": dict(expected_supervised_gate),
        "required_review_checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
        "forbidden_live_request_types": list(FORBIDDEN_LIVE_REQUEST_TYPES),
        "blockers_not_closed_by_packet": list(NOT_CLOSED_BLOCKERS),
        "next_manual_artifact_required": (
            WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
        ),
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
        "- backend_port_prestart_summary_required: "
        f"`{str(packet['backend_port_prestart_summary_required']).lower()}`",
        "- backend_port_prestart_summary_path_placeholder: "
        f"`{packet['backend_port_prestart_summary_path_placeholder']}`",
        "- backend_port_prestart_check_operator_sequence_step_index: "
        f"`{packet['backend_port_prestart_check_operator_sequence_step_index']}`",
        "- backend_port_prestart_check_must_be_first_operator_step: "
        f"`{str(packet['backend_port_prestart_check_must_be_first_operator_step']).lower()}`",
        "- backend_port_prestart_check_command: "
        f"`{_command_text(packet['backend_port_prestart_check_command'])}`",
        "- backend_port_prestart_check_failure_detail_fields: "
        f"`{json.dumps(packet['backend_port_prestart_check_failure_detail_fields'])}`",
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
    supervised_gate = (
        packet.get("supervised_prestart_gate")
        if isinstance(packet.get("supervised_prestart_gate"), Mapping)
        else {}
    )
    backend_port_prestart_gate = (
        packet.get("backend_port_prestart_gate")
        if isinstance(packet.get("backend_port_prestart_gate"), Mapping)
        else {}
    )
    expected_backend_port_prestart_gate = (
        packet.get("expected_backend_port_prestart_gate")
        if isinstance(packet.get("expected_backend_port_prestart_gate"), Mapping)
        else {}
    )
    expected_supervised_gate = (
        packet.get("expected_supervised_prestart_gate")
        if isinstance(packet.get("expected_supervised_prestart_gate"), Mapping)
        else {}
    )
    if backend_port_prestart_gate:
        lines.extend(
            [
                "",
                "## Backend Port Prestart Gate",
                "",
                f"- required: `{str(backend_port_prestart_gate.get('required')).lower()}`",
                "- operator_sequence_step_index: "
                f"`{backend_port_prestart_gate.get('operator_sequence_step_index')}`",
                "- operator_sequence_step_must_be_first: "
                f"`{str(backend_port_prestart_gate.get('operator_sequence_step_must_be_first')).lower()}`",
                "- runs_command: "
                f"`{backend_port_prestart_gate.get('runs_command')}`",
                "- must_pass_before_backend_wing_session: "
                f"`{str(backend_port_prestart_gate.get('must_pass_before_backend_wing_session')).lower()}`",
                f"- grants_live_authority: `{str(backend_port_prestart_gate.get('grants_live_authority')).lower()}`",
                "- required_markers: "
                f"`{json.dumps(backend_port_prestart_gate.get('required_markers') or [])}`",
            ]
        )
    if expected_backend_port_prestart_gate:
        lines.extend(
            [
                "",
                "## Expected Backend Port Prestart Gate",
                "",
                "- expected_backend_port_prestart_gate: "
                f"`{json.dumps(expected_backend_port_prestart_gate, sort_keys=True)}`",
                "- actual_matches_expected: "
                f"`{str(backend_port_prestart_gate == expected_backend_port_prestart_gate).lower()}`",
            ]
        )
    if supervised_gate:
        lines.extend(
            [
                "",
                "## Supervised Prestart Gate",
                "",
                f"- required: `{str(supervised_gate.get('required')).lower()}`",
                "- operator_sequence_step_index: "
                f"`{supervised_gate.get('operator_sequence_step_index')}`",
                "- operator_sequence_step_must_be_final: "
                f"`{str(supervised_gate.get('operator_sequence_step_must_be_final')).lower()}`",
                "- requires_operator_decision_status: "
                f"`{supervised_gate.get('requires_operator_decision_status')}`",
                "- requires_accepted_candidate_prestart_trace: "
                f"`{str(supervised_gate.get('requires_accepted_candidate_prestart_trace')).lower()}`",
                f"- unlocks_runbook: `{supervised_gate.get('unlocks_runbook')}`",
                f"- grants_live_authority: `{str(supervised_gate.get('grants_live_authority')).lower()}`",
                "- required_markers: "
                f"`{json.dumps(supervised_gate.get('required_markers') or [])}`",
            ]
        )
    if expected_supervised_gate:
        lines.extend(
            [
                "",
                "## Expected Supervised Prestart Gate",
                "",
                "- expected_supervised_prestart_gate: "
                f"`{json.dumps(expected_supervised_gate, sort_keys=True)}`",
                "- actual_matches_expected: "
                f"`{str(supervised_gate == expected_supervised_gate).lower()}`",
            ]
        )
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
            "## Expected Response Types",
            "",
            "- expected_response_types_by_request_type: "
            f"`{json.dumps(packet['expected_response_types_by_request_type'], sort_keys=True)}`",
            "",
            "## Expected Artifacts",
            "",
            f"- summary: `{packet['expected_run_artifact_glob']}`",
            f"- events: `{packet['expected_event_log_glob']}`",
            f"- pending decision template: `{packet['decision_template_output']}`",
            "- pending decision template matches contract: "
            f"`{str(packet['decision_template_output_matches_contract']).lower()}`",
            f"- artifact review report: `{packet['review_output']}`",
            "- artifact review report matches contract: "
            f"`{str(packet['review_output_matches_contract']).lower()}`",
            f"- decision review report: `{packet['decision_review_output']}`",
            "- decision review report matches contract: "
            f"`{str(packet['decision_review_output_matches_contract']).lower()}`",
        ]
    )
    blocker_anchor = (
        packet.get("mvp_blocker_anchor")
        if isinstance(packet.get("mvp_blocker_anchor"), Mapping)
        else {}
    )
    if blocker_anchor:
        lines.extend(
            [
                "## MVP Blocker Anchor",
                "",
                f"- blocker_table: `{blocker_anchor.get('blocker_table')}`",
                f"- selected_blocker: `{blocker_anchor.get('selected_blocker')}`",
                "- selected_blocker_status: "
                f"`{blocker_anchor.get('selected_blocker_status')}`",
                f"- selected_action: {blocker_anchor.get('selected_action')}",
                "- selected_evidence_file_count: "
                f"`{blocker_anchor.get('selected_evidence_file_count')}`",
                "- packet_does_not_close_blocker: "
                f"`{str(blocker_anchor.get('packet_does_not_close_blocker')).lower()}`",
                "- mvp_blocker_anchor_sha256: "
                f"`{packet.get('mvp_blocker_anchor_sha256')}`",
                "",
                "### Selected Evidence Files",
                "",
            ]
        )
        for item in blocker_anchor.get("selected_evidence_files") or []:
            lines.append(f"- `{item}`")
        lines.append("")
    lines.extend(["## Required Checklist", ""])
    for item in packet.get("required_review_checklist_ids") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Forbidden Request Types", ""])
    for item in packet.get("forbidden_live_request_types") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Blockers Not Closed By This Packet", ""])
    for item in packet.get("blockers_not_closed_by_packet") or []:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Next Required Artifact",
            "",
            "- next_manual_artifact_required: "
            f"`{packet['next_manual_artifact_required']}`",
            "",
        ]
    )
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
    parser.add_argument("--prestart-summary", default=DEFAULT_PRESTART_SUMMARY)
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
            prestart_summary=args.prestart_summary,
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
