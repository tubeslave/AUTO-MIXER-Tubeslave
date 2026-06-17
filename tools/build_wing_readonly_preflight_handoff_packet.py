#!/usr/bin/env python3
"""Build an offline operator packet for the WING read-only preflight step."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.wing_readonly_preflight_contract import (  # noqa: E402
    WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP,
    WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS,
    WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT,
    WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP,
    WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED,
    WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
    WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
    WING_READONLY_PREFLIGHT_PROOF_RUNNER,
    WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_RUNNER_URL,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS,
    WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP,
    WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX,
    expected_backend_port_prestart_gate,
    expected_handoff_operator_sequence,
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
ALLOWED_REQUEST_TYPES = list(WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES)
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
ARTIFACT_MANIFEST_VERIFIER_COMMAND = [
    "PYTHONPATH=backend",
    "python3",
    "tools/verify_no_wing_artifact_manifest.py",
    "<no_wing_artifact_dir>",
]
REQUEST_ID_POLICY = WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
PROOF_ARTIFACT_SCHEMA = WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
PROOF_RUNNER = WING_READONLY_PREFLIGHT_PROOF_RUNNER
SELECTED_MVP_BLOCKER = WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
SELECTED_MVP_BLOCKER_STATUS = WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
FAIL_CLOSED_CONDITIONS = list(WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS)
BACKEND_PORT_PRESTART_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT = (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
)
BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
)
PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
)
PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
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


def _sha256_existing_file(path_value: str) -> str:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            return ""
    return ""


def _load_existing_json_object(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [path, REPO_ROOT / path]
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _prestart_backend_check_detail(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    for item in results:
        if (
            isinstance(item, Mapping)
            and item.get("name")
            == no_wing_release_check.WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
        ):
            detail = item.get("detail")
            return detail if isinstance(detail, Mapping) else {}
    return {}


def _prestart_backend_check_status(payload: Mapping[str, Any]) -> Any:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for item in results:
        if (
            isinstance(item, Mapping)
            and item.get("name")
            == no_wing_release_check.WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
        ):
            return item.get("status")
    return None


def _prestart_summary_value(
    payload: Mapping[str, Any],
    field: str,
    expected_value: Any,
) -> Any:
    return payload.get(field) if payload else expected_value


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


def request_address(channel: int, value_type: str) -> str:
    if value_type == "fader":
        return f"/ch/{channel}/fdr"
    if value_type == "gain":
        return f"/ch/{channel}/in/set/trim"
    if value_type == "pan":
        return f"/ch/{channel}/pan"
    raise ValueError(f"value_types_unsupported:{value_type}")


def request_id_for(channel: int, value_type: str) -> str:
    return f"wing-readonly-preflight:ch{channel}:{value_type}"


def build_request_sequence(
    channels: Sequence[int],
    value_types: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        *[
            {
                "type": "query_mixer_parameter",
                "request_id": request_id_for(channel, value_type),
                "channel": channel,
                "value_type": value_type,
                "address": request_address(channel, value_type),
            }
            for channel in channels
            for value_type in value_types
        ],
    ]


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


def validate_inputs_command(
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
        "--validate-run-inputs-only",
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
    backend_connection_step_index = (
        WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX
    )
    backend_connection_step = WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP
    mvp_blocker_anchor_digest_review_step = (
        WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP
    )
    mvp_blocker_anchor_digest_review_step_index = (
        WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX
    )
    expected_prestart_gate = expected_backend_port_prestart_gate(
        operator_sequence_step=prestart_step,
        operator_sequence_step_index=prestart_step_index,
    )
    expected_supervised_gate = expected_supervised_prestart_gate(
        operator_sequence_step=supervised_gate_step,
        operator_sequence_step_index=supervised_gate_step_index,
    )
    request_sequence = build_request_sequence(channel_numbers, value_type_names)
    preview = plan_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        plan_output=plan_output,
        prestart_summary=prestart_summary,
    )
    validate_inputs = validate_inputs_command(
        url=url,
        channels=channels,
        value_types=value_types,
        artifact_root=artifact_root,
        expected_plan=plan_output,
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
    expected_run_artifact_glob = (
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
    )
    expected_event_log_glob = (
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
    )
    mvp_blocker_anchor = no_wing_release_check.wing_mvp_first_open_blocker_anchor()
    mvp_blocker_anchor_sha256 = no_wing_release_check.wing_mvp_blocker_anchor_sha256(
        mvp_blocker_anchor
    )
    project_director_action_anchor = (
        no_wing_release_check.wing_mvp_project_director_selected_action_anchor()
    )
    project_director_selected_action_sha256 = str(
        project_director_action_anchor["selected_action_sha256"]
    )
    actual_project_director_selected_action_sha256 = hashlib.sha256(
        str(project_director_action_anchor["selected_action"]).encode("utf-8")
    ).hexdigest()
    project_director_selected_action_sha256_available = bool(
        project_director_selected_action_sha256
    )
    project_director_selected_action_sha256_matches_text = (
        project_director_selected_action_sha256
        == actual_project_director_selected_action_sha256
    )
    prestart_summary_sha256 = _sha256_existing_file(prestart_summary)
    prestart_summary_payload = _load_existing_json_object(prestart_summary)
    prestart_backend_check_detail = _prestart_backend_check_detail(
        prestart_summary_payload
    )
    prestart_summary_type = _prestart_summary_value(
        prestart_summary_payload,
        "type",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
    )
    prestart_summary_status = _prestart_summary_value(
        prestart_summary_payload,
        "status",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
    )
    prestart_summary_connection_policy = _prestart_summary_value(
        prestart_summary_payload,
        "connection_policy",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    )
    prestart_summary_read_only = _prestart_summary_value(
        prestart_summary_payload,
        "read_only",
        True,
    )
    prestart_summary_live_write = _prestart_summary_value(
        prestart_summary_payload,
        "live_write",
        False,
    )
    prestart_summary_contacts_backend = _prestart_summary_value(
        prestart_summary_payload,
        "contacts_backend",
        False,
    )
    prestart_summary_contacts_wing = _prestart_summary_value(
        prestart_summary_payload,
        "contacts_wing",
        False,
    )
    prestart_summary_mutates_console_state = _prestart_summary_value(
        prestart_summary_payload,
        "mutates_console_state",
        False,
    )
    prestart_summary_queries_console = _prestart_summary_value(
        prestart_summary_payload,
        "queries_console",
        False,
    )
    prestart_summary_executes_tests = _prestart_summary_value(
        prestart_summary_payload,
        "executes_tests",
        False,
    )
    prestart_backend_check_status = (
        _prestart_backend_check_status(prestart_summary_payload)
        if prestart_summary_payload
        else WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
    )
    environment_blocked_checks = prestart_summary_payload.get(
        "environment_blocked_checks"
    )
    environment_blocked_check_details = prestart_summary_payload.get(
        "environment_blocked_check_details"
    )
    prestart_required_false_all_values = {
        field: (
            prestart_backend_check_detail.get(field)
            if prestart_summary_payload
            else False
        )
        for field in PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
    }
    prestart_required_false_values = {
        field: value
        for field, value in prestart_required_false_all_values.items()
        if field != "live_authority_granted"
    }
    prestart_required_false_plan_values = {
        field: (
            prestart_backend_check_detail.get(field)
            if prestart_summary_payload
            else False
        )
        for field in PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
    }
    prestart_denial_match_flags = {
        match_field: (
            prestart_required_false_all_values.get(detail_field)
            == prestart_required_false_plan_values.get(plan_field)
        )
        for detail_field, plan_field, match_field in zip(
            PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
            PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
            PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
        )
    }
    expected_operator_sequence = expected_handoff_operator_sequence(
        backend_port_prestart_step=prestart_step,
        backend_connection_step=backend_connection_step,
        mvp_blocker_anchor_digest_review_step=mvp_blocker_anchor_digest_review_step,
        supervised_prestart_step=supervised_gate_step,
    )
    operator_sequence = list(expected_operator_sequence)
    operator_sequence_count = len(operator_sequence)
    operator_sequence_sha256 = (
        no_wing_release_check.wing_readonly_operator_sequence_sha256(
            operator_sequence
        )
    )
    expected_operator_sequence_sha256 = (
        no_wing_release_check.wing_readonly_operator_sequence_sha256(
            expected_operator_sequence
        )
    )
    operator_sequence_matches_contract = operator_sequence == expected_operator_sequence
    operator_sequence_sha256_matches_contract = (
        operator_sequence_sha256 == expected_operator_sequence_sha256
    )
    required_supervised_proof_plan_summary_fields = list(
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS
    )
    expected_required_supervised_proof_plan_summary_fields = list(
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS
    )
    required_supervised_proof_plan_summary_fields_include_invalid_records = (
        "readonly_preflight_review_decision_invalid_records"
        in required_supervised_proof_plan_summary_fields
    )
    backend_port_prestart_gate = dict(expected_prestart_gate)
    expected_backend_port_prestart_gate_dict = dict(expected_prestart_gate)
    supervised_prestart_gate = dict(expected_supervised_gate)
    expected_supervised_prestart_gate_dict = dict(expected_supervised_gate)
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
        "backend_port_prestart_summary_sha256": prestart_summary_sha256,
        "backend_port_prestart_summary_sha256_available": bool(
            prestart_summary_sha256
        ),
        "backend_port_prestart_summary_type": prestart_summary_type,
        "backend_port_prestart_summary_status": prestart_summary_status,
        "backend_port_prestart_summary_connection_policy": (
            prestart_summary_connection_policy
        ),
        "backend_port_prestart_summary_read_only": prestart_summary_read_only,
        "backend_port_prestart_summary_live_write": prestart_summary_live_write,
        "backend_port_prestart_summary_contacts_backend": (
            prestart_summary_contacts_backend
        ),
        "backend_port_prestart_summary_contacts_wing": (
            prestart_summary_contacts_wing
        ),
        "backend_port_prestart_summary_mutates_console_state": (
            prestart_summary_mutates_console_state
        ),
        "backend_port_prestart_summary_queries_console": (
            prestart_summary_queries_console
        ),
        "backend_port_prestart_summary_executes_tests": (
            prestart_summary_executes_tests
        ),
        "backend_port_prestart_summary_environment_blocked_checks": (
            environment_blocked_checks
            if isinstance(environment_blocked_checks, list)
            else []
        ),
        "backend_port_prestart_summary_environment_blocked_check_details": (
            environment_blocked_check_details
            if isinstance(environment_blocked_check_details, list)
            else []
        ),
        "backend_port_prestart_check_status": prestart_backend_check_status,
        **prestart_required_false_values,
        **prestart_required_false_plan_values,
        **prestart_denial_match_flags,
        "backend_port_prestart_check_operator_sequence_step_index": prestart_step_index,
        "backend_port_prestart_check_must_be_first_operator_step": True,
        "backend_port_prestart_check_command": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
        "backend_port_prestart_check_command_text": (
            BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
        ),
        "artifact_manifest_verifier_required": True,
        "artifact_manifest_verifier_command": _command_text(
            ARTIFACT_MANIFEST_VERIFIER_COMMAND
        ),
        "artifact_manifest_verifier_must_pass_before_backend_wing_session": True,
        "artifact_manifest_verifier_read_only": True,
        "artifact_manifest_verifier_live_write": False,
        "artifact_manifest_verifier_contacts_backend": False,
        "artifact_manifest_verifier_contacts_wing": False,
        "artifact_manifest_verifier_mutates_console_state": False,
        "artifact_manifest_verifier_queries_console": False,
        "artifact_manifest_verifier_executes_tests": False,
        "artifact_manifest_verifier_readiness_closure_allowed": False,
        "artifact_manifest_verifier_live_authority_granted": False,
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
        "proof_artifact_schema": PROOF_ARTIFACT_SCHEMA,
        "proof_runner": PROOF_RUNNER,
        "selected_mvp_blocker": SELECTED_MVP_BLOCKER,
        "selected_mvp_blocker_status": SELECTED_MVP_BLOCKER_STATUS,
        "allowed_request_types": list(ALLOWED_REQUEST_TYPES),
        "request_id_policy": REQUEST_ID_POLICY,
        "expected_response_types_by_request_type": dict(
            EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE
        ),
        "request_sequence": request_sequence,
        "request_count": len(request_sequence),
        "fail_closed_conditions": list(FAIL_CLOSED_CONDITIONS),
        "artifact_root": artifact_root,
        "decision_root": decision_root,
        "expected_artifacts": [expected_run_artifact_glob, expected_event_log_glob],
        "expected_run_artifact_glob": expected_run_artifact_glob,
        "expected_event_log_glob": expected_event_log_glob,
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
            "validate_readonly_preflight_inputs": validate_inputs,
            "capture_readonly_preflight": capture,
            "review_artifact_and_write_pending_decision_template": review,
            "review_edited_decision_json": decision,
            "prestart_backend_port_check": list(BACKEND_PORT_PRESTART_CHECK_COMMAND),
            "verify_no_wing_artifact_manifest": list(
                ARTIFACT_MANIFEST_VERIFIER_COMMAND
            ),
            "focused_validation": list(FOCUSED_VALIDATION_COMMAND),
        },
        "mvp_blocker_anchor": mvp_blocker_anchor,
        "mvp_blocker_anchor_sha256": mvp_blocker_anchor_sha256,
        "project_director_selected_action_anchor": project_director_action_anchor,
        "expected_project_director_selected_action_anchor": (
            project_director_action_anchor
        ),
        "project_director_selected_action_anchor_matches_current_directive": True,
        "project_director_selected_action_source": (
            project_director_action_anchor["selected_action_source"]
        ),
        "expected_project_director_selected_action_source": (
            project_director_action_anchor["selected_action_source"]
        ),
        "project_director_selected_action_source_matches_current_directive": True,
        "project_director_selected_action_sha256": (
            project_director_selected_action_sha256
        ),
        "actual_project_director_selected_action_sha256": (
            actual_project_director_selected_action_sha256
        ),
        "expected_project_director_selected_action_sha256": (
            project_director_selected_action_sha256
        ),
        "project_director_selected_action_sha256_available": (
            project_director_selected_action_sha256_available
        ),
        "project_director_selected_action_sha256_available_matches_digest": (
            project_director_selected_action_sha256_available
            == bool(project_director_selected_action_sha256)
        ),
        "project_director_selected_action_sha256_matches_anchor": True,
        "project_director_selected_action_sha256_matches_text": (
            project_director_selected_action_sha256_matches_text
        ),
        "project_director_selected_action_sha256_matches_current_directive": True,
        "project_director_current_first_command_baseline_coverage_required": (
            project_director_action_anchor[
                "current_first_command_baseline_coverage_required"
            ]
        ),
        "project_director_current_first_command_baseline_coverage_passed": (
            project_director_action_anchor[
                "current_first_command_baseline_coverage_passed"
            ]
        ),
        "project_director_current_first_command_baseline_coverage_marker_count": (
            project_director_action_anchor[
                "current_first_command_baseline_coverage_marker_count"
            ]
        ),
        "project_director_current_first_command_baseline_missing_markers": (
            project_director_action_anchor[
                "current_first_command_baseline_missing_markers"
            ]
        ),
        "operator_sequence": operator_sequence,
        "operator_sequence_sha256": operator_sequence_sha256,
        "expected_operator_sequence_sha256": expected_operator_sequence_sha256,
        "operator_sequence_matches_contract": operator_sequence_matches_contract,
        "operator_sequence_sha256_matches_contract": (
            operator_sequence_sha256_matches_contract
        ),
        "operator_sequence_count": operator_sequence_count,
        "expected_operator_sequence_count": (
            WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT
        ),
        "operator_sequence_count_matches_contract": (
            operator_sequence_count
            == WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT
        ),
        "backend_connection_confirmation_step_index": backend_connection_step_index,
        "validate_run_inputs_only_step_index": (
            WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX
        ),
        "validate_run_inputs_only_step": (
            WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP
        ),
        "validate_run_inputs_only_before_capture": (
            operator_sequence.index(WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP)
            < operator_sequence.index(WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP)
        ),
        "validate_run_inputs_only_transport_opened": False,
        "validate_run_inputs_only_contacts_backend": False,
        "validate_run_inputs_only_queries_console": False,
        "validate_run_inputs_only_grants_live_authority": False,
        "backend_port_prestart_gate": backend_port_prestart_gate,
        "expected_backend_port_prestart_gate": expected_backend_port_prestart_gate_dict,
        "backend_port_prestart_gate_matches_expected": (
            backend_port_prestart_gate == expected_backend_port_prestart_gate_dict
        ),
        "supervised_prestart_gate": supervised_prestart_gate,
        "expected_supervised_prestart_gate": expected_supervised_prestart_gate_dict,
        "supervised_prestart_gate_matches_expected": (
            supervised_prestart_gate == expected_supervised_prestart_gate_dict
        ),
        "required_supervised_proof_plan_summary_fields": (
            required_supervised_proof_plan_summary_fields
        ),
        "expected_required_supervised_proof_plan_summary_fields": list(
            expected_required_supervised_proof_plan_summary_fields
        ),
        "required_supervised_proof_plan_summary_fields_match_contract": (
            required_supervised_proof_plan_summary_fields
            == expected_required_supervised_proof_plan_summary_fields
        ),
        "required_supervised_proof_plan_summary_fields_include_invalid_records": (
            required_supervised_proof_plan_summary_fields_include_invalid_records
        ),
        "mvp_blocker_anchor_digest_review_step_index": (
            mvp_blocker_anchor_digest_review_step_index
        ),
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
        "- backend_port_prestart_summary_sha256: "
        f"`{packet['backend_port_prestart_summary_sha256']}`",
        "- backend_port_prestart_summary_sha256_available: "
        f"`{str(packet['backend_port_prestart_summary_sha256_available']).lower()}`",
        "- backend_port_prestart_summary_type: "
        f"`{packet['backend_port_prestart_summary_type']}`",
        "- backend_port_prestart_summary_status: "
        f"`{packet['backend_port_prestart_summary_status']}`",
        "- backend_port_prestart_summary_connection_policy: "
        f"`{packet['backend_port_prestart_summary_connection_policy']}`",
        "- backend_port_prestart_summary_read_only: "
        f"`{str(packet['backend_port_prestart_summary_read_only']).lower()}`",
        "- backend_port_prestart_summary_live_write: "
        f"`{str(packet['backend_port_prestart_summary_live_write']).lower()}`",
        "- backend_port_prestart_summary_contacts_backend: "
        f"`{str(packet['backend_port_prestart_summary_contacts_backend']).lower()}`",
        "- backend_port_prestart_summary_contacts_wing: "
        f"`{str(packet['backend_port_prestart_summary_contacts_wing']).lower()}`",
        "- backend_port_prestart_summary_mutates_console_state: "
        f"`{str(packet['backend_port_prestart_summary_mutates_console_state']).lower()}`",
        "- backend_port_prestart_summary_queries_console: "
        f"`{str(packet['backend_port_prestart_summary_queries_console']).lower()}`",
        "- backend_port_prestart_summary_executes_tests: "
        f"`{str(packet['backend_port_prestart_summary_executes_tests']).lower()}`",
        "- backend_port_prestart_summary_environment_blocked_checks: "
        f"`{json.dumps(packet['backend_port_prestart_summary_environment_blocked_checks'])}`",
        "- backend_port_prestart_summary_environment_blocked_check_details: "
        f"`{json.dumps(packet['backend_port_prestart_summary_environment_blocked_check_details'])}`",
        "- backend_port_prestart_check_status: "
        f"`{packet['backend_port_prestart_check_status']}`",
        *[
            f"- {field}: `{str(packet[field]).lower()}`"
            for field in PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        ],
        *[
            f"- {field}: `{str(packet[field]).lower()}`"
            for field in PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        ],
        *[
            f"- {field}: `{str(packet[field]).lower()}`"
            for field in PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
        ],
        "- backend_port_prestart_check_operator_sequence_step_index: "
        f"`{packet['backend_port_prestart_check_operator_sequence_step_index']}`",
        "- backend_port_prestart_check_must_be_first_operator_step: "
        f"`{str(packet['backend_port_prestart_check_must_be_first_operator_step']).lower()}`",
        "- backend_port_prestart_check_command: "
        f"`{_command_text(packet['backend_port_prestart_check_command'])}`",
        "- backend_port_prestart_check_command_text: "
        f"`{packet['backend_port_prestart_check_command_text']}`",
        "- artifact_manifest_verifier_required: "
        f"`{str(packet['artifact_manifest_verifier_required']).lower()}`",
        "- artifact_manifest_verifier_command: "
        f"`{packet['artifact_manifest_verifier_command']}`",
        "- artifact_manifest_verifier_must_pass_before_backend_wing_session: "
        f"`{str(packet['artifact_manifest_verifier_must_pass_before_backend_wing_session']).lower()}`",
        "- artifact_manifest_verifier_read_only: "
        f"`{str(packet['artifact_manifest_verifier_read_only']).lower()}`",
        "- artifact_manifest_verifier_live_write: "
        f"`{str(packet['artifact_manifest_verifier_live_write']).lower()}`",
        "- artifact_manifest_verifier_contacts_backend: "
        f"`{str(packet['artifact_manifest_verifier_contacts_backend']).lower()}`",
        "- artifact_manifest_verifier_contacts_wing: "
        f"`{str(packet['artifact_manifest_verifier_contacts_wing']).lower()}`",
        "- artifact_manifest_verifier_mutates_console_state: "
        f"`{str(packet['artifact_manifest_verifier_mutates_console_state']).lower()}`",
        "- artifact_manifest_verifier_queries_console: "
        f"`{str(packet['artifact_manifest_verifier_queries_console']).lower()}`",
        "- artifact_manifest_verifier_executes_tests: "
        f"`{str(packet['artifact_manifest_verifier_executes_tests']).lower()}`",
        "- artifact_manifest_verifier_readiness_closure_allowed: "
        f"`{str(packet['artifact_manifest_verifier_readiness_closure_allowed']).lower()}`",
        "- artifact_manifest_verifier_live_authority_granted: "
        f"`{str(packet['artifact_manifest_verifier_live_authority_granted']).lower()}`",
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
        "## Proof Identity",
        "",
        f"- proof_artifact_schema: `{packet['proof_artifact_schema']}`",
        f"- proof_runner: `{packet['proof_runner']}`",
        f"- selected_mvp_blocker: `{packet['selected_mvp_blocker']}`",
        "- selected_mvp_blocker_status: "
        f"`{packet['selected_mvp_blocker_status']}`",
        "- project_director_selected_action_source: "
        f"`{packet['project_director_selected_action_source']}`",
        "- expected_project_director_selected_action_source: "
        f"`{packet['expected_project_director_selected_action_source']}`",
        "- project_director_selected_action_source_matches_current_directive: "
        f"`{str(packet['project_director_selected_action_source_matches_current_directive']).lower()}`",
        "- project_director_selected_action_sha256: "
        f"`{packet['project_director_selected_action_sha256']}`",
        "- actual_project_director_selected_action_sha256: "
        f"`{packet['actual_project_director_selected_action_sha256']}`",
        "- expected_project_director_selected_action_sha256: "
        f"`{packet['expected_project_director_selected_action_sha256']}`",
        "- project_director_selected_action_sha256_available: "
        f"`{str(packet['project_director_selected_action_sha256_available']).lower()}`",
        "- project_director_selected_action_sha256_available_matches_digest: "
        f"`{str(packet['project_director_selected_action_sha256_available_matches_digest']).lower()}`",
        "- project_director_selected_action_sha256_matches_anchor: "
        f"`{str(packet['project_director_selected_action_sha256_matches_anchor']).lower()}`",
        "- project_director_selected_action_sha256_matches_text: "
        f"`{str(packet['project_director_selected_action_sha256_matches_text']).lower()}`",
        "- project_director_selected_action_sha256_matches_current_directive: "
        f"`{str(packet['project_director_selected_action_sha256_matches_current_directive']).lower()}`",
        "- project_director_selected_action_anchor_matches_current_directive: "
        f"`{str(packet['project_director_selected_action_anchor_matches_current_directive']).lower()}`",
        "- project_director_current_first_command_baseline_coverage_required: "
        f"`{str(packet['project_director_current_first_command_baseline_coverage_required']).lower()}`",
        "- project_director_current_first_command_baseline_coverage_passed: "
        f"`{str(packet['project_director_current_first_command_baseline_coverage_passed']).lower()}`",
        "- project_director_current_first_command_baseline_coverage_marker_count: "
        f"`{packet['project_director_current_first_command_baseline_coverage_marker_count']}`",
        "- project_director_current_first_command_baseline_missing_markers: "
        f"`{json.dumps(packet['project_director_current_first_command_baseline_missing_markers'])}`",
        "",
        "## Operator Sequence",
        "",
        "- operator_sequence_count: "
        f"`{packet['operator_sequence_count']}`",
        "- expected_operator_sequence_count: "
        f"`{packet['expected_operator_sequence_count']}`",
        "- operator_sequence_count_matches_contract: "
        f"`{str(packet['operator_sequence_count_matches_contract']).lower()}`",
        "- operator_sequence_sha256: "
        f"`{packet['operator_sequence_sha256']}`",
        "- expected_operator_sequence_sha256: "
        f"`{packet['expected_operator_sequence_sha256']}`",
        "- operator_sequence_matches_contract: "
        f"`{str(packet['operator_sequence_matches_contract']).lower()}`",
        "- operator_sequence_sha256_matches_contract: "
        f"`{str(packet['operator_sequence_sha256_matches_contract']).lower()}`",
        "- mvp_blocker_anchor_digest_review_step_index: "
        f"`{packet['mvp_blocker_anchor_digest_review_step_index']}`",
        "- validate_run_inputs_only_step_index: "
        f"`{packet['validate_run_inputs_only_step_index']}`",
        "- validate_run_inputs_only_before_capture: "
        f"`{str(packet['validate_run_inputs_only_before_capture']).lower()}`",
        "- validate_run_inputs_only_transport_opened: "
        f"`{str(packet['validate_run_inputs_only_transport_opened']).lower()}`",
        "- validate_run_inputs_only_contacts_backend: "
        f"`{str(packet['validate_run_inputs_only_contacts_backend']).lower()}`",
        "- validate_run_inputs_only_queries_console: "
        f"`{str(packet['validate_run_inputs_only_queries_console']).lower()}`",
        "- validate_run_inputs_only_grants_live_authority: "
        f"`{str(packet['validate_run_inputs_only_grants_live_authority']).lower()}`",
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
                "- backend_port_prestart_gate_matches_expected: "
                f"`{str(packet.get('backend_port_prestart_gate_matches_expected')).lower()}`",
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
                "- required_supervised_proof_plan_summary_fields: "
                f"`{json.dumps(packet.get('required_supervised_proof_plan_summary_fields') or [])}`",
                "- expected_required_supervised_proof_plan_summary_fields: "
                f"`{json.dumps(packet.get('expected_required_supervised_proof_plan_summary_fields') or [])}`",
                "- required_supervised_proof_plan_summary_fields_match_contract: "
                f"`{str(packet.get('required_supervised_proof_plan_summary_fields_match_contract')).lower()}`",
                "- required_supervised_proof_plan_summary_fields_include_invalid_records: "
                f"`{str(packet.get('required_supervised_proof_plan_summary_fields_include_invalid_records')).lower()}`",
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
                "- supervised_prestart_gate_matches_expected: "
                f"`{str(packet.get('supervised_prestart_gate_matches_expected')).lower()}`",
                "- actual_matches_expected: "
                f"`{str(supervised_gate == expected_supervised_gate).lower()}`",
            ]
        )
    lines.extend(["", "## Commands", ""])
    for name in (
        "prestart_backend_port_check",
        "verify_no_wing_artifact_manifest",
        "preview_readonly_request_plan",
        "validate_readonly_preflight_inputs",
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
            "- allowed_request_types: "
            f"`{json.dumps(packet['allowed_request_types'])}`",
            f"- request_id_policy: `{packet['request_id_policy']}`",
            "- expected_response_types_by_request_type: "
            f"`{json.dumps(packet['expected_response_types_by_request_type'], sort_keys=True)}`",
            "",
            "## Expected Request Sequence",
            "",
            f"- request_count: `{packet['request_count']}`",
            "- request_sequence: "
            f"`{json.dumps(packet['request_sequence'])}`",
            "",
            "## Fail-Closed Conditions",
            "",
            "- fail_closed_conditions: "
            f"`{json.dumps(packet['fail_closed_conditions'])}`",
            "",
            "## Expected Artifacts",
            "",
            "- expected_artifacts: "
            f"`{json.dumps(packet['expected_artifacts'])}`",
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
    project_director_anchor = (
        packet.get("project_director_selected_action_anchor")
        if isinstance(packet.get("project_director_selected_action_anchor"), Mapping)
        else {}
    )
    if project_director_anchor:
        lines.extend(
            [
                "## Project Director Selected Action Anchor",
                "",
                f"- anchor_schema: `{project_director_anchor.get('anchor_schema')}`",
                f"- blocker_table: `{project_director_anchor.get('blocker_table')}`",
                f"- selected_blocker: `{project_director_anchor.get('selected_blocker')}`",
                "- selected_blocker_status: "
                f"`{project_director_anchor.get('selected_blocker_status')}`",
                "- selected_action_source: "
                f"`{project_director_anchor.get('selected_action_source')}`",
                "- selected_action_sha256: "
                f"`{project_director_anchor.get('selected_action_sha256')}`",
                "- current_first_command_baseline_coverage_required: "
                f"`{str(project_director_anchor.get('current_first_command_baseline_coverage_required')).lower()}`",
                "- current_first_command_baseline_coverage_passed: "
                f"`{str(project_director_anchor.get('current_first_command_baseline_coverage_passed')).lower()}`",
                "- current_first_command_baseline_coverage_marker_count: "
                f"`{project_director_anchor.get('current_first_command_baseline_coverage_marker_count')}`",
                "- current_first_command_baseline_missing_markers: "
                f"`{json.dumps(project_director_anchor.get('current_first_command_baseline_missing_markers') or [])}`",
                "- packet_does_not_close_blocker: "
                f"`{str(project_director_anchor.get('packet_does_not_close_blocker')).lower()}`",
                "- expected_project_director_selected_action_anchor: "
                f"`{json.dumps(packet.get('expected_project_director_selected_action_anchor'))}`",
                f"- selected_action: {project_director_anchor.get('selected_action')}",
                "",
            ]
        )
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
