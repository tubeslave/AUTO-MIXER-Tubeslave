"""Shared read-only WING preflight contract constants.

The values here are report-only safety boundaries. They do not open a mixer
transport or grant live-write authority.
"""

from __future__ import annotations


WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA = "wing_readonly_preflight.v1"
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
WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER = (
    "artifacts/no_wing_release_check/<run_id>/summary.json"
)

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

WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX = 9
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP = (
    "Only after accepted_decision_ready and a valid accepted-candidate "
    "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS = (
    "accepted_decision_ready",
    "valid accepted-candidate prestart trace",
    "WING_SUPERVISED_FADER_GAIN_PROOF",
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK = (
    "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
)
WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS = (
    "accepted_decision_ready"
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
