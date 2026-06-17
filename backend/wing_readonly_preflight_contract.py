"""Shared read-only WING preflight contract constants.

The values here are report-only safety boundaries. They do not open a mixer
transport or grant live-write authority.
"""

from __future__ import annotations

import posixpath
from typing import Any, Iterable, Mapping


WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA = "wing_readonly_preflight.v1"
WING_READONLY_PREFLIGHT_PROOF_RUNNER = "wing_readonly_preflight_runner"
WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER = "WING read-only connection proof"
WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS = (
    "tooling_ready_waits_real_wing"
)
WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME = "summary.json"
WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME = "events.jsonl"
WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT = (
    f"{WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA} "
    f"{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
)
WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED = (
    f"real {WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT}"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER = (
    "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER = (
    "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER = (
    "accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER = (
    "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
)
WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS = (
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_operator_session_ready",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_event_log_run_id_binding",
    "confirm_no_live_authority",
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS = (
    "review_id",
    "artifact_source_id",
    "run_id",
    "artifact_captured_at",
    "artifact_root",
    "summary_path",
    "summary_sha256",
    "events_path",
    "events_sha256",
    "channels",
    "value_types",
    "operator_confirmed_channels",
    "operator_confirmed_session_ready",
    "readback_count",
    "physical_readback_count",
    "request_id_policy",
    "request_id_correlation_required",
    "event_log_run_id_matches_summary",
    "decision",
    "reviewed_by",
    "reviewed_at",
    "checklist_ids",
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS = (
    "operator_must_replace_placeholders_before_ledger_acceptance"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT = (
    "artifacts/wing_readonly_preflight_review_decisions"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT = (
    f"{WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT}/pending_review_decision_template.json"
)
WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT = (
    "Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT = (
    "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
)
WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT = (
    "Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md"
)
WING_READONLY_PREFLIGHT_PLAN_OUTPUT = (
    "artifacts/wing_readonly_preflight_plan/latest_plan.json"
)
WING_READONLY_PREFLIGHT_RUNNER_URL = "ws://localhost:8765"
WING_READONLY_PREFLIGHT_RUNNER_CHANNELS = "3"
WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES = ("fader", "gain", "pan")
WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES = ",".join(
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
)
WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES = (
    "verified_from_physical_readback",
)
WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "physical_wing",
    "wing",
)
WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER = (
    "artifacts/no_wing_release_check/<run_id>/summary.json"
)
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE = "no_wing_release_check"
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS = "passed"
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY = (
    "no_hardware_no_network_no_live_writes"
)
WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME = "backend_port_free"
WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS = (
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
)
WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS = (
    "safe_to_stop_listener_by_gate",
    "live_authority_granted",
    "backend_port_prestart_gate_execution_allowed",
    "backend_port_prestart_gate_readiness_closure_allowed",
    "backend_port_prestart_gate_live_authority_granted",
)
WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS = tuple(
    f"plan_{field}"
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
)
WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS = tuple(
    f"{field}_matches_plan"
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
)
WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_WS_PROOF_FIELDS = (
    "artifact_manifest_verifier_must_pass_before_websocket",
    "artifact_manifest_verifier_passed_before_websocket",
    "artifact_manifest_verifier_websocket_opened_after_pass",
)
WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS = "passed"
WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON = "wing_readonly_preflight_passed"
WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY = (
    "required_for_every_query_mixer_parameter_response"
)
WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES = (
    "get_operator_mode",
    "get_state",
    "query_mixer_parameter",
)
WING_READONLY_PREFLIGHT_OPERATOR_MODE_RESPONSE_TYPES = ("operator_mode_status",)
WING_READONLY_PREFLIGHT_STATE_RESPONSE_TYPES = ("state_update",)
WING_READONLY_PREFLIGHT_READBACK_RESPONSE_TYPES = (
    "mixer_parameter_value",
    "mixer_parameter_read_blocked",
)
WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE = {
    "get_operator_mode": WING_READONLY_PREFLIGHT_OPERATOR_MODE_RESPONSE_TYPES,
    "get_state": WING_READONLY_PREFLIGHT_STATE_RESPONSE_TYPES,
    "query_mixer_parameter": WING_READONLY_PREFLIGHT_READBACK_RESPONSE_TYPES,
}

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = (
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT = " ".join(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = (
    "listener_processes",
    "listener_process_error",
    "listener_confirmation_checklist",
    "preflight_blocked_until_listener_confirmed",
    "intended_backend_confirmation_required",
    "safe_to_stop_listener_by_gate",
    "listener_confirmation_template_status",
    "listener_confirmation_template_path",
    "listener_confirmation_template_acceptance_status",
    "listener_confirmation_template_safe_to_stop_listener_by_gate",
    "listener_confirmation_template_live_authority_granted",
    "listener_confirmation_template_accepted_by_template",
    "listener_confirmation_template_operator_placeholder_fields",
    "listener_confirmation_template_execution_allowed_by_template",
    "listener_confirmation_template_readiness_closure_allowed_by_template",
    "listener_confirmation_template_required_edits",
    "backend_port_prestart_gate_execution_allowed",
    "backend_port_prestart_gate_readiness_closure_allowed",
    "backend_port_prestart_gate_live_authority_granted",
    "fail_closed_reason",
    "operator_next_step",
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME = (
    "prestart_backend_port_check"
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX = 0
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP = (
    "Before starting the intended backend/WING session, run "
    "prestart_backend_port_check and resolve only confirmed "
    "unexpected listeners."
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS = (
    "Before starting the intended backend/WING session",
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME,
    "unexpected listeners",
)

WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_OPERATOR_STEP_INDEX = 1
WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_OPERATOR_STEP = (
    "Run verify_no_wing_artifact_manifest against the same passed no-WING "
    "artifact directory before any backend/WING WebSocket."
)
WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_REQUIRED_MARKERS = (
    "verify_no_wing_artifact_manifest",
    "same passed no-WING artifact directory",
    "before any backend/WING WebSocket",
)

WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX = 3
WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP = (
    "Confirm existing backend connection is the intended WING Rack and ready "
    "for read-only inspection."
)
WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_REQUIRED_MARKERS = (
    "existing backend connection",
    "intended WING Rack",
    "read-only inspection",
)

WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX = 5
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX = 11
WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX = (
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX - 1
)
WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT = (
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX + 1
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP = (
    "Only after accepted_decision_ready, a valid accepted-candidate "
    "prestart trace, a valid accepted-candidate verifier WebSocket proof, "
    "a valid accepted-candidate artifact-root trace, and "
    "a valid accepted-candidate request-sequence trace, a valid "
    "accepted-candidate contact trace, plus "
    "expected-artifact-bound accepted source id, move to "
    "WING_SUPERVISED_FADER_GAIN_PROOF."
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS = (
    "accepted_decision_ready",
    "valid accepted-candidate prestart trace",
    "valid accepted-candidate verifier WebSocket proof",
    "valid accepted-candidate artifact-root trace",
    "valid accepted-candidate request-sequence trace",
    "valid accepted-candidate contact trace",
    "expected-artifact-bound accepted source id",
    "WING_SUPERVISED_FADER_GAIN_PROOF",
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK = (
    "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS = (
    "accepted_decision_ready"
)
WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES = (
    "fader",
    "gain",
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS = (
    "allowed_supervised_live_value_types",
    "readonly_preflight_review_decision_candidate_prestart_contract_valid",
    "readonly_preflight_review_decision_candidate_prestart_source_ids",
    "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source",
    "readonly_preflight_review_decision_expected_candidate_prestart_source_ids",
    "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
    "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    "readonly_preflight_review_decision_candidate_request_sequence_contract_valid",
    "readonly_preflight_review_decision_candidate_request_sequence_source_ids",
    "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source",
    "readonly_preflight_review_decision_expected_candidate_request_sequence_source_ids",
    "readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source",
    "readonly_preflight_review_decision_candidate_artifact_root_contract_valid",
    "readonly_preflight_review_decision_candidate_artifact_roots",
    "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source",
    "readonly_preflight_review_decision_expected_candidate_artifact_roots",
    "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source",
    "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source",
    "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids",
    "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete",
    "readonly_preflight_review_decision_invalid_records",
)
WING_READONLY_PREFLIGHT_PREVIEW_PLAN_OPERATOR_STEP = (
    "Run preview_readonly_request_plan and confirm the request sequence is read-only."
)
WING_READONLY_PREFLIGHT_CHANNEL_SAFETY_CONFIRMATION_STEP = (
    "Confirm the selected channel numbers are safe to inspect."
)
WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP = "Run capture_readonly_preflight."
WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP = (
    "Run validate_readonly_preflight_inputs before capture; it must validate "
    "the prestart summary and expected plan without opening a websocket."
)
WING_READONLY_PREFLIGHT_REVIEW_OPERATOR_STEP = (
    "Run review_artifact_and_write_pending_decision_template."
)
WING_READONLY_PREFLIGHT_DECISION_TEMPLATE_EDIT_STEP = (
    "Edit the pending decision template into an accepted or rejected JSON record."
)
WING_READONLY_PREFLIGHT_DECISION_REVIEW_OPERATOR_STEP = (
    "Run review_edited_decision_json."
)
WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP = (
    "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
    "current first open blocker before supervised proof planning."
)

WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES = (
    "create_operator_proposal",
    "apply_operator_proposal",
    "rollback_last_manual_write",
    "emergency_stop",
    "set_operator_mode",
    "set_fader",
    "set_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_compressor",
    "route_output",
    "load_snap",
    "save_snap",
    "reset",
    "mute",
    "unmute",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
    "connect_wing",
    "disconnect_wing",
    "connect_mixer",
    "disconnect_mixer",
)
WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS = (
    "wing_readonly_connection_proof",
    "physical_console_readback_proof",
    "supervised_fader_write_proof",
    "supervised_gain_down_trim_proof",
    "readiness_matrix_update_based_on_real_proof",
)

WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS = (
    "backend_not_in_wing_mode",
    "event_log_wing_mode_not_proven",
    "transport_timeout_or_connection_loss",
    "ws_handshake_rejected",
    "operator_channel_confirmation_mismatch",
    "operator_session_confirmation_missing",
    "physical_readback_missing",
    "readback_identity_mismatch",
    "readback_request_id_mismatch",
    "expected_plan_missing_or_mismatch",
    "prestart_summary_missing_or_invalid",
    "prestart_summary_sha256_mismatch",
    "stale_backend_port_prestart_detail_contract",
    "existing_backend_connection_boundary_missing",
    "forbidden_request_type_sent",
    "unexpected_response_type",
    "event_log_run_id_mismatch",
    "event_log_missing_or_invalid",
    "manual_review_decision_missing_or_invalid",
)


def expected_backend_port_prestart_gate(
    *,
    operator_sequence_step: str = WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP,
    operator_sequence_step_index: int = (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX
    ),
) -> dict[str, object]:
    return {
        "required": True,
        "operator_sequence_step_index": operator_sequence_step_index,
        "operator_sequence_step_must_be_first": True,
        "operator_sequence_step": operator_sequence_step,
        "required_markers": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS
        ),
        "runs_command": WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME,
        "must_pass_before_backend_wing_session": True,
        "grants_live_authority": False,
    }


def expected_supervised_prestart_gate(
    *,
    operator_sequence_step: str = WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP,
    operator_sequence_step_index: int | None = (
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX
    ),
) -> dict[str, object]:
    return {
        "required": True,
        "operator_sequence_step_index": operator_sequence_step_index,
        "operator_sequence_step_must_be_final": True,
        "operator_sequence_step": operator_sequence_step,
        "required_markers": list(
            WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS
        ),
        "unlocks_runbook": WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK,
        "grants_live_authority": False,
        "requires_operator_decision_status": (
            WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS
        ),
        "requires_accepted_candidate_prestart_trace": True,
        "requires_accepted_candidate_verifier_ws_proof": True,
        "requires_accepted_candidate_artifact_root_trace": True,
        "requires_accepted_candidate_request_sequence_trace": True,
        "requires_accepted_candidate_expected_artifact_trace": True,
        "requires_accepted_candidate_contact_trace": True,
    }


def expected_handoff_operator_sequence(
    *,
    backend_port_prestart_step: str = (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP
    ),
    artifact_manifest_verifier_step: str = (
        WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_OPERATOR_STEP
    ),
    backend_connection_step: str = (
        WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP
    ),
    mvp_blocker_anchor_digest_review_step: str = (
        WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP
    ),
    supervised_prestart_step: str = WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP,
) -> list[str]:
    return [
        backend_port_prestart_step,
        artifact_manifest_verifier_step,
        WING_READONLY_PREFLIGHT_PREVIEW_PLAN_OPERATOR_STEP,
        backend_connection_step,
        WING_READONLY_PREFLIGHT_CHANNEL_SAFETY_CONFIRMATION_STEP,
        WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP,
        WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP,
        WING_READONLY_PREFLIGHT_REVIEW_OPERATOR_STEP,
        WING_READONLY_PREFLIGHT_DECISION_TEMPLATE_EDIT_STEP,
        WING_READONLY_PREFLIGHT_DECISION_REVIEW_OPERATOR_STEP,
        mvp_blocker_anchor_digest_review_step,
        supervised_prestart_step,
    ]


def _contract_string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _contract_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def is_sha256_hexdigest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _candidate_expected_artifact_binding_valid(row: Mapping[str, Any]) -> bool:
    return not _candidate_expected_artifact_binding_violations(row)


def _candidate_expected_artifact_binding_violations(
    row: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    expected_artifacts = _contract_string_list(row.get("expected_artifacts"))
    summary_path = str(row.get("summary_path") or row.get("source_id") or "")
    events_path = str(row.get("events_path") or "")
    expected_summary_artifact = str(row.get("expected_summary_artifact") or "")
    expected_events_artifact = str(row.get("expected_events_artifact") or "")
    if len(expected_artifacts) != 2:
        violations.append("expected_artifacts_count_mismatch")
    if not expected_summary_artifact:
        violations.append("expected_summary_artifact_missing")
    elif summary_path and expected_summary_artifact != summary_path:
        violations.append("expected_summary_artifact_mismatch")
    if not expected_events_artifact:
        violations.append("expected_events_artifact_missing")
    elif events_path and expected_events_artifact != events_path:
        violations.append("expected_events_artifact_mismatch")
    if row.get("summary_path_matches_expected_artifact") is not True:
        violations.append("summary_path_not_bound_to_expected_artifact")
    if row.get("events_path_matches_expected_artifact") is not True:
        violations.append("events_path_not_bound_to_expected_artifact")
    return violations


def wing_readonly_preflight_candidate_prestart_contract_violations(
    row: Mapping[str, Any],
) -> list[str]:
    """Return report-only prestart contract violations for one candidate row."""

    violations = _candidate_expected_artifact_binding_violations(row)
    if row.get("backend_port_prestart_check_required") is not True:
        violations.append("backend_port_prestart_check_required_not_true")
    if row.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
        violations.append(
            "backend_port_prestart_check_must_pass_before_starting_backend_not_true"
        )
    if _contract_string_list(row.get("backend_port_prestart_check_command")) != (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append("backend_port_prestart_check_command_mismatch")
    command_text = str(row.get("backend_port_prestart_check_command_text") or "")
    if command_text != WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT:
        violations.append("backend_port_prestart_check_command_text_mismatch")
    if _contract_string_list(
        row.get("backend_port_prestart_check_failure_detail_fields")
    ) != (WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS):
        violations.append("backend_port_prestart_check_failure_detail_fields_mismatch")
    if row.get("backend_port_prestart_summary_required") is not True:
        violations.append("backend_port_prestart_summary_required_not_true")
    if not str(row.get("backend_port_prestart_summary_path") or ""):
        violations.append("backend_port_prestart_summary_path_missing")
    prestart_summary_sha256 = str(
        row.get("backend_port_prestart_summary_sha256") or ""
    )
    if not prestart_summary_sha256:
        violations.append("backend_port_prestart_summary_sha256_missing")
    elif not is_sha256_hexdigest(prestart_summary_sha256):
        violations.append("backend_port_prestart_summary_sha256_invalid")
    if (
        row.get("backend_port_prestart_summary_type")
        != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
    ):
        violations.append("backend_port_prestart_summary_type_mismatch")
    if (
        row.get("backend_port_prestart_summary_status")
        != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
    ):
        violations.append("backend_port_prestart_summary_status_not_passed")
    if (
        row.get("backend_port_prestart_summary_connection_policy")
        != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
    ):
        violations.append("backend_port_prestart_summary_connection_policy_mismatch")
    if row.get("backend_port_prestart_summary_read_only") is not True:
        violations.append("backend_port_prestart_summary_read_only_not_true")
    if row.get("backend_port_prestart_summary_live_write") is not False:
        violations.append("backend_port_prestart_summary_live_write_not_false")
    if row.get("backend_port_prestart_summary_contacts_backend") is not False:
        violations.append("backend_port_prestart_summary_contacts_backend_not_false")
    if row.get("backend_port_prestart_summary_contacts_wing") is not False:
        violations.append("backend_port_prestart_summary_contacts_wing_not_false")
    if row.get("backend_port_prestart_summary_mutates_console_state") is not False:
        violations.append(
            "backend_port_prestart_summary_mutates_console_state_not_false"
        )
    if row.get("backend_port_prestart_summary_queries_console") is not False:
        violations.append("backend_port_prestart_summary_queries_console_not_false")
    if row.get("backend_port_prestart_summary_executes_tests") is not False:
        violations.append("backend_port_prestart_summary_executes_tests_not_false")
    if row.get("artifact_manifest_verifier_must_pass_before_websocket") is not True:
        violations.append(
            "artifact_manifest_verifier_must_pass_before_websocket_not_true"
        )
    if row.get("artifact_manifest_verifier_passed_before_websocket") is not True:
        violations.append(
            "artifact_manifest_verifier_passed_before_websocket_not_true"
        )
    if row.get("artifact_manifest_verifier_websocket_opened_after_pass") is not True:
        violations.append(
            "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
        )
    if (
        row.get("backend_port_prestart_check_status")
        != WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
    ):
        violations.append("backend_port_prestart_check_status_not_passed")
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        if row.get(field) is not False:
            violations.append(f"{field}_not_false")
        if row.get(f"plan_{field}") is not False:
            violations.append(f"plan_{field}_not_false")
        if row.get(f"{field}_matches_plan") is not True:
            violations.append(f"{field}_matches_plan_not_true")
    return violations


def wing_readonly_preflight_candidate_verifier_websocket_proof_violations(
    row: Mapping[str, Any],
) -> list[str]:
    """Return verifier-before-WebSocket proof violations for one candidate row."""

    violations: list[str] = []
    for field in WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_WS_PROOF_FIELDS:
        if row.get(field) is not True:
            violations.append(f"{field}_not_true")
    return violations


def wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Return accepted source ids whose recorded prestart trace is incomplete."""

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    violations_by_source: dict[str, list[str]] = {}
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        normalized_source_id = str(source_id)
        row = rows_by_source.get(normalized_source_id)
        if not isinstance(row, Mapping):
            violations_by_source[normalized_source_id] = ["candidate_review_missing"]
            continue
        violations = wing_readonly_preflight_candidate_prestart_contract_violations(row)
        if violations:
            violations_by_source[normalized_source_id] = violations
    return violations_by_source


def wing_readonly_preflight_accepted_candidate_prestart_source_ids(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return accepted source ids whose candidate rows bind a valid prestart trace.

    This is a report-only gate. It does not inspect files, contact the backend,
    or grant authority; it only checks that review/plan summaries agree on the
    no-WING prestart contract already recorded in candidate review rows.
    """

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    valid_source_ids: list[str] = []
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        row = rows_by_source.get(str(source_id))
        if not isinstance(row, Mapping):
            continue
        if wing_readonly_preflight_candidate_prestart_contract_violations(row):
            continue
        valid_source_ids.append(str(source_id))
    return valid_source_ids


def wing_readonly_preflight_accepted_candidate_verifier_websocket_proof_violations_by_source(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Return accepted source ids lacking verifier-before-WebSocket proof."""

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    violations_by_source: dict[str, list[str]] = {}
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        normalized_source_id = str(source_id)
        row = rows_by_source.get(normalized_source_id)
        if not isinstance(row, Mapping):
            violations_by_source[normalized_source_id] = ["candidate_review_missing"]
            continue
        violations = (
            wing_readonly_preflight_candidate_verifier_websocket_proof_violations(row)
        )
        if violations:
            violations_by_source[normalized_source_id] = violations
    return violations_by_source


def wing_readonly_preflight_accepted_candidate_verifier_websocket_proof_source_ids(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return accepted source ids with complete verifier-before-WebSocket proof."""

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    valid_source_ids: list[str] = []
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        row = rows_by_source.get(str(source_id))
        if not isinstance(row, Mapping):
            continue
        if wing_readonly_preflight_candidate_verifier_websocket_proof_violations(row):
            continue
        valid_source_ids.append(str(source_id))
    return valid_source_ids


def wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    return wing_readonly_preflight_accepted_candidate_verifier_websocket_proof_violations_by_source(
        accepted_source_ids,
        candidate_reviews,
    )


def wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> list[str]:
    return wing_readonly_preflight_accepted_candidate_verifier_websocket_proof_source_ids(
        accepted_source_ids,
        candidate_reviews,
    )


def wing_readonly_preflight_candidate_request_sequence_plan_violations(
    row: Mapping[str, Any],
) -> list[str]:
    """Return report-only request-sequence/expected-plan trace violations."""

    violations: list[str] = []
    request_sequence = _contract_mapping_list(row.get("request_sequence"))
    plan_request_sequence = _contract_mapping_list(row.get("plan_request_sequence"))
    if not request_sequence:
        violations.append("request_sequence_missing")
    if not plan_request_sequence:
        violations.append("plan_request_sequence_missing")
    if row.get("request_sequence_matches_plan") is not True:
        violations.append("request_sequence_matches_plan_not_true")
    elif request_sequence and plan_request_sequence and request_sequence != plan_request_sequence:
        violations.append("request_sequence_plan_mismatch")
    return violations


def wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Return accepted source ids whose request sequence is not bound to plan."""

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    violations_by_source: dict[str, list[str]] = {}
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        normalized_source_id = str(source_id)
        row = rows_by_source.get(normalized_source_id)
        if not isinstance(row, Mapping):
            violations_by_source[normalized_source_id] = ["candidate_review_missing"]
            continue
        violations = wing_readonly_preflight_candidate_request_sequence_plan_violations(
            row
        )
        if violations:
            violations_by_source[normalized_source_id] = violations
    return violations_by_source


def wing_readonly_preflight_accepted_candidate_request_sequence_source_ids(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return accepted source ids whose request sequence matches expected plan."""

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    valid_source_ids: list[str] = []
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        row = rows_by_source.get(str(source_id))
        if not isinstance(row, Mapping):
            continue
        if wing_readonly_preflight_candidate_request_sequence_plan_violations(row):
            continue
        valid_source_ids.append(str(source_id))
    return valid_source_ids


def _candidate_artifact_root_path_violations(
    row: Mapping[str, Any],
    *,
    source_id: str,
    artifact_root: str,
) -> list[str]:
    violations: list[str] = []
    normalized_root = posixpath.normpath(artifact_root.rstrip("/"))
    if not normalized_root:
        return ["artifact_root_missing"]
    if ".." in artifact_root.split("/"):
        violations.append("artifact_root_contains_parent_traversal")
    path_fields = (
        ("source_id", source_id),
        ("summary_path", str(row.get("summary_path") or "")),
        ("events_path", str(row.get("events_path") or "")),
        ("expected_summary_artifact", str(row.get("expected_summary_artifact") or "")),
        ("expected_events_artifact", str(row.get("expected_events_artifact") or "")),
    )
    for field, path in path_fields:
        if not path:
            violations.append(f"{field}_missing")
            continue
        if ".." in path.split("/"):
            violations.append(f"{field}_contains_parent_traversal")
        normalized_path = posixpath.normpath(path)
        if not normalized_path.startswith(f"{normalized_root}/"):
            violations.append(f"{field}_not_under_artifact_root")
    return violations


def wing_readonly_preflight_accepted_candidate_artifact_root_contract(
    accepted_source_ids: Iterable[str],
    candidate_reviews: Iterable[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return accepted source ids whose candidate rows bind one artifact root.

    This report-only gate checks that the accepted source id and the associated
    summary/event artifact paths all live under the declared artifact root. It
    does not read files, contact WING, mutate state, or grant authority.
    """

    rows_by_source = {
        str(row.get("source_id")): row
        for row in candidate_reviews
        if isinstance(row, Mapping) and row.get("source_id")
    }
    artifact_roots: list[str] = []
    violations_by_source: dict[str, list[str]] = {}
    for source_id in accepted_source_ids:
        if not source_id:
            continue
        normalized_source_id = str(source_id)
        row = rows_by_source.get(normalized_source_id)
        if not isinstance(row, Mapping):
            violations_by_source[normalized_source_id] = ["candidate_review_missing"]
            continue
        artifact_root = str(row.get("artifact_root") or "")
        if not artifact_root:
            violations_by_source[normalized_source_id] = ["artifact_root_missing"]
            continue
        artifact_roots.append(artifact_root)
        violations = _candidate_artifact_root_path_violations(
            row,
            source_id=normalized_source_id,
            artifact_root=artifact_root,
        )
        if violations:
            violations_by_source[normalized_source_id] = violations
    return artifact_roots, violations_by_source
