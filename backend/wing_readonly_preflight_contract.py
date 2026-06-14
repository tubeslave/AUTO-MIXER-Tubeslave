"""Shared read-only WING preflight contract constants.

The values here are report-only safety boundaries. They do not open a mixer
transport or grant live-write authority.
"""

from __future__ import annotations

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
WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS = (
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_operator_session_ready",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_no_live_authority",
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS = (
    "review_id",
    "artifact_source_id",
    "run_id",
    "artifact_captured_at",
    "summary_path",
    "summary_sha256",
    "events_path",
    "events_sha256",
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

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = (
    "listener_processes",
    "listener_process_error",
    "listener_confirmation_checklist",
    "preflight_blocked_until_listener_confirmed",
    "intended_backend_confirmation_required",
    "safe_to_stop_listener_by_gate",
    "listener_confirmation_template_status",
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

WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX = 2
WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP = (
    "Confirm existing backend connection is the intended WING Rack and ready "
    "for read-only inspection."
)
WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_REQUIRED_MARKERS = (
    "existing backend connection",
    "intended WING Rack",
    "read-only inspection",
)

WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX = 9
WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX = (
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX - 1
)
WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT = (
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX + 1
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP = (
    "Only after accepted_decision_ready, a valid accepted-candidate "
    "prestart trace, and expected-artifact-bound accepted source id, move "
    "to WING_SUPERVISED_FADER_GAIN_PROOF."
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS = (
    "accepted_decision_ready",
    "valid accepted-candidate prestart trace",
    "expected-artifact-bound accepted source id",
    "WING_SUPERVISED_FADER_GAIN_PROOF",
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK = (
    "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS = (
    "accepted_decision_ready"
)
WING_READONLY_PREFLIGHT_PREVIEW_PLAN_OPERATOR_STEP = (
    "Run preview_readonly_request_plan and confirm the request sequence is read-only."
)
WING_READONLY_PREFLIGHT_CHANNEL_SAFETY_CONFIRMATION_STEP = (
    "Confirm the selected channel numbers are safe to inspect."
)
WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP = "Run capture_readonly_preflight."
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
    }


def expected_handoff_operator_sequence(
    *,
    backend_port_prestart_step: str = (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP
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
        WING_READONLY_PREFLIGHT_PREVIEW_PLAN_OPERATOR_STEP,
        backend_connection_step,
        WING_READONLY_PREFLIGHT_CHANNEL_SAFETY_CONFIRMATION_STEP,
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


def _candidate_expected_artifact_binding_valid(row: Mapping[str, Any]) -> bool:
    expected_artifacts = _contract_string_list(row.get("expected_artifacts"))
    if len(expected_artifacts) != 2:
        return False
    if not str(row.get("expected_summary_artifact") or ""):
        return False
    if not str(row.get("expected_events_artifact") or ""):
        return False
    if row.get("summary_path_matches_expected_artifact") is not True:
        return False
    if row.get("events_path_matches_expected_artifact") is not True:
        return False
    return True


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
        if not _candidate_expected_artifact_binding_valid(row):
            continue
        if row.get("backend_port_prestart_check_required") is not True:
            continue
        if row.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
            continue
        if _contract_string_list(row.get("backend_port_prestart_check_command")) != (
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ):
            continue
        if _contract_string_list(
            row.get("backend_port_prestart_check_failure_detail_fields")
        ) != (WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS):
            continue
        if row.get("backend_port_prestart_summary_required") is not True:
            continue
        if not str(row.get("backend_port_prestart_summary_path") or ""):
            continue
        if not str(row.get("backend_port_prestart_summary_sha256") or ""):
            continue
        if (
            row.get("backend_port_prestart_summary_type")
            != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
        ):
            continue
        if (
            row.get("backend_port_prestart_summary_status")
            != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
        ):
            continue
        if (
            row.get("backend_port_prestart_summary_connection_policy")
            != WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
        ):
            continue
        if row.get("backend_port_prestart_summary_live_write") is not False:
            continue
        if row.get("backend_port_prestart_summary_contacts_backend") is not False:
            continue
        if row.get("backend_port_prestart_summary_queries_console") is not False:
            continue
        if row.get("backend_port_prestart_summary_executes_tests") is not False:
            continue
        if (
            row.get("backend_port_prestart_check_status")
            != WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
        ):
            continue
        valid_source_ids.append(str(source_id))
    return valid_source_ids
