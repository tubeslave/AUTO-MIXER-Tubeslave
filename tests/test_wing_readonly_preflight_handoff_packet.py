import hashlib
import json
import subprocess
import sys
from pathlib import Path

from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_REQUIRED_MARKERS,
    WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP,
    WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS,
    WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT,
    WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED,
    WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
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
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS,
    WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP,
    WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX,
    expected_backend_port_prestart_gate,
    expected_handoff_operator_sequence,
    expected_supervised_prestart_gate,
)
from tools import build_wing_readonly_preflight_handoff_packet as handoff
from tools import no_wing_release_check

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_no_wing_prestart_summary(
    path: Path,
    *,
    status: str = WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
    environment_blocked: bool = False,
) -> None:
    detail = {
        "backend_port_prestart_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "safe_to_stop_listener_by_gate": False,
        "live_authority_granted": False,
        "backend_port_prestart_gate_execution_allowed": False,
        "backend_port_prestart_gate_readiness_closure_allowed": False,
        "backend_port_prestart_gate_live_authority_granted": False,
        **{
            field: False
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        },
        **{
            field: True
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
        },
    }
    payload = {
        "type": WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
        "status": status,
        "connection_policy": WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
        "read_only": True,
        "live_write": False,
        "contacts_backend": False,
        "contacts_wing": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "results": [
            {
                "name": no_wing_release_check.WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME,
                "status": WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
                "detail": detail,
            }
        ],
    }
    if environment_blocked:
        payload["failed_checks"] = ["frontend_e2e"]
        payload["environment_blocked_checks"] = ["frontend_e2e"]
        payload["environment_blocked_check_details"] = [
            {
                "name": "frontend_e2e",
                "reason": "local_dev_server_bind_denied",
                "address": "127.0.0.1:3000",
                "operator_next_step": (
                    "Rerun the no-WING quick check in an environment that can "
                    "bind the frontend dev server."
                ),
                "stdout_log": "frontend-stdout.log",
                "stderr_log": "frontend-stderr.log",
            }
        ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_handoff_packet_builds_readonly_operator_sequence():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert handoff.DEFAULT_PACKET_OUTPUT == WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
    assert handoff.DEFAULT_PLAN_OUTPUT == WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    assert handoff.DEFAULT_URL == WING_READONLY_PREFLIGHT_RUNNER_URL
    assert handoff.DEFAULT_CHANNELS == WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
    assert handoff.DEFAULT_VALUE_TYPES == WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
    assert handoff.DEFAULT_ARTIFACT_ROOT == WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    assert handoff.SUPPORTED_VALUE_TYPES == set(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert packet["type"] == "wing_readonly_preflight_handoff_packet"
    assert packet["status"] == "operator_hardware_step_required"
    assert packet["read_only"] is True
    assert packet["live_write"] is False
    assert packet["mutates_console_state"] is False
    assert packet["queries_console"] is False
    assert packet["executes_tests"] is False
    assert packet["operator_required"] is True
    assert packet["expected_plan_required_for_real_run"] is True
    assert packet["physical_preflight_requires_existing_backend_connection"] is True
    assert packet["backend_port_prestart_check_required"] is True
    assert packet["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert packet["backend_port_prestart_summary_required"] is True
    assert packet["backend_port_prestart_summary_path_placeholder"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert packet["backend_port_prestart_summary_sha256"] == ""
    assert packet["backend_port_prestart_summary_sha256_available"] is False
    assert packet["backend_port_prestart_summary_type"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
    )
    assert packet["backend_port_prestart_summary_status"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
    )
    assert packet["backend_port_prestart_summary_connection_policy"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
    )
    assert packet["backend_port_prestart_summary_read_only"] is True
    assert packet["backend_port_prestart_summary_live_write"] is False
    assert packet["backend_port_prestart_summary_contacts_backend"] is False
    assert packet["backend_port_prestart_summary_contacts_wing"] is False
    assert packet["backend_port_prestart_summary_mutates_console_state"] is False
    assert packet["backend_port_prestart_summary_queries_console"] is False
    assert packet["backend_port_prestart_summary_executes_tests"] is False
    assert packet["backend_port_prestart_check_status"] == (
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
    )
    assert packet["backend_port_prestart_check_operator_sequence_step_index"] == 0
    assert packet["backend_port_prestart_check_must_be_first_operator_step"] is True
    assert packet["backend_connection_confirmation_step_index"] == (
        WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX
    )
    assert packet["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert (
        packet["backend_port_prestart_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    )
    assert packet["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert packet["manual_review_required_before_readiness_update"] is True
    assert packet["execution_allowed_by_packet"] is False
    assert packet["readiness_matrix_update_allowed_by_packet"] is False
    assert packet["live_authority_granted"] is False
    assert packet["full_autonomous_ready"] is False
    assert packet["next_manual_artifact_required"] == (
        WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
    )
    assert packet["expected_run_artifact_glob"].endswith(
        f"/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
    )
    assert packet["expected_event_log_glob"].endswith(
        f"/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
    )
    assert packet["expected_artifacts"] == [
        packet["expected_run_artifact_glob"],
        packet["expected_event_log_glob"],
    ]
    assert packet["channels"] == [3]
    assert packet["value_types"] == ["fader", "gain", "pan"]
    assert packet["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert packet["proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert packet["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert packet["selected_mvp_blocker_status"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert packet["allowed_request_types"] == list(
        WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
    )
    assert packet["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    assert packet["expected_response_types_by_request_type"] == {
        request_type: list(response_types)
        for request_type, response_types in (
            WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
        )
    }
    assert packet["fail_closed_conditions"] == list(
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
    )
    assert packet["request_count"] == 5
    assert packet["request_sequence"] == [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        {
            "type": "query_mixer_parameter",
            "request_id": "wing-readonly-preflight:ch3:fader",
            "channel": 3,
            "value_type": "fader",
            "address": "/ch/3/fdr",
        },
        {
            "type": "query_mixer_parameter",
            "request_id": "wing-readonly-preflight:ch3:gain",
            "channel": 3,
            "value_type": "gain",
            "address": "/ch/3/in/set/trim",
        },
        {
            "type": "query_mixer_parameter",
            "request_id": "wing-readonly-preflight:ch3:pan",
            "channel": 3,
            "value_type": "pan",
            "address": "/ch/3/pan",
        },
    ]
    assert packet["commands"]["preview_readonly_request_plan"] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--prestart-summary",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        "--plan-only",
        "--plan-output",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    ]
    assert packet["commands"]["validate_readonly_preflight_inputs"] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--prestart-summary",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        "--expected-plan",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
        "--validate-run-inputs-only",
    ]
    assert packet["commands"]["capture_readonly_preflight"] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--operator-confirm-channels",
        "3",
        "--operator-confirm-session-ready",
        "--value-types",
        "fader,gain,pan",
        "--require-physical-readback",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--prestart-summary",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        "--expected-plan",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    ]
    assert packet["commands"]["prestart_backend_port_check"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert packet["commands"]["verify_no_wing_artifact_manifest"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/verify_no_wing_artifact_manifest.py",
        "<no_wing_artifact_dir>",
    ]
    assert packet["artifact_manifest_verifier_required"] is True
    assert packet["artifact_manifest_verifier_command"] == (
        "PYTHONPATH=backend python3 "
        "tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>"
    )
    assert (
        packet["artifact_manifest_verifier_must_pass_before_backend_wing_session"]
        is True
    )
    assert packet["artifact_manifest_verifier_read_only"] is True
    assert packet["artifact_manifest_verifier_live_write"] is False
    assert packet["artifact_manifest_verifier_contacts_backend"] is False
    assert packet["artifact_manifest_verifier_contacts_wing"] is False
    assert packet["artifact_manifest_verifier_mutates_console_state"] is False
    assert packet["artifact_manifest_verifier_queries_console"] is False
    assert packet["artifact_manifest_verifier_executes_tests"] is False
    assert (
        packet["artifact_manifest_verifier_readiness_matrix_update_allowed"]
        is False
    )
    assert packet["artifact_manifest_verifier_readiness_closure_allowed"] is False
    assert packet["artifact_manifest_verifier_live_authority_granted"] is False
    assert packet["mvp_blocker_anchor"] == (
        no_wing_release_check.wing_mvp_first_open_blocker_anchor()
    )
    assert packet["mvp_blocker_anchor_sha256"] == (
        no_wing_release_check.wing_mvp_blocker_anchor_sha256(
            packet["mvp_blocker_anchor"]
        )
    )
    assert packet["project_director_selected_action_anchor"] == (
        no_wing_release_check.wing_mvp_project_director_selected_action_anchor()
    )
    assert packet["expected_project_director_selected_action_anchor"] == (
        packet["project_director_selected_action_anchor"]
    )
    assert packet["project_director_selected_action_anchor_matches_current_directive"] is True
    assert packet["project_director_selected_action_source"] == (
        packet["project_director_selected_action_anchor"]["selected_action_source"]
    )
    assert packet["project_director_selected_action_source"] == "Current First Command"
    assert packet["expected_project_director_selected_action_source"] == (
        "Current First Command"
    )
    assert packet["project_director_selected_action_source_matches_current_directive"] is True
    assert packet["project_director_selected_action_sha256"] == (
        packet["project_director_selected_action_anchor"]["selected_action_sha256"]
    )
    assert packet["project_director_selected_action_sha256"] == hashlib.sha256(
        packet["project_director_selected_action_anchor"]["selected_action"].encode(
            "utf-8"
        )
    ).hexdigest()
    assert packet["actual_project_director_selected_action_sha256"] == (
        packet["project_director_selected_action_sha256"]
    )
    assert packet["expected_project_director_selected_action_sha256"] == (
        packet["project_director_selected_action_sha256"]
    )
    assert packet["project_director_selected_action_sha256_available"] is True
    assert (
        packet["project_director_selected_action_sha256_available_matches_digest"]
        is True
    )
    assert packet["project_director_selected_action_sha256_matches_anchor"] is True
    assert packet["project_director_selected_action_sha256_matches_text"] is True
    assert (
        packet["project_director_selected_action_sha256_matches_current_directive"]
        is True
    )
    assert (
        packet["project_director_current_first_command_baseline_coverage_required"]
        is True
    )
    assert (
        packet["project_director_current_first_command_baseline_coverage_passed"]
        is True
    )
    assert (
        packet[
            "project_director_current_first_command_baseline_coverage_marker_count"
        ]
        == packet["project_director_selected_action_anchor"][
            "current_first_command_baseline_coverage_marker_count"
        ]
    )
    assert (
        packet["project_director_current_first_command_baseline_missing_markers"]
        == []
    )
    assert (
        packet["mvp_blocker_anchor"]["selected_blocker"]
        == "WING read-only connection proof"
    )
    assert packet["mvp_blocker_anchor"]["selected_evidence_file_count"] == len(
        packet["mvp_blocker_anchor"]["selected_evidence_files"]
    )
    assert "tools/wing_readonly_preflight_runner.py" in packet[
        "mvp_blocker_anchor"
    ]["selected_evidence_files"]
    assert (
        packet["operator_sequence"][0]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP
    )
    assert len(packet["operator_sequence"]) == (
        WING_READONLY_PREFLIGHT_HANDOFF_OPERATOR_SEQUENCE_COUNT
    )
    assert packet["operator_sequence_sha256"] == (
        no_wing_release_check.wing_readonly_operator_sequence_sha256(
            packet["operator_sequence"]
        )
    )
    assert packet["operator_sequence"] == expected_handoff_operator_sequence()
    assert packet["expected_operator_sequence_sha256"] == (
        no_wing_release_check.wing_readonly_operator_sequence_sha256(
            expected_handoff_operator_sequence()
        )
    )
    assert packet["operator_sequence_matches_contract"] is True
    assert packet["operator_sequence_sha256_matches_contract"] is True
    first_gate_step = packet["operator_sequence"][0]
    assert packet["backend_port_prestart_gate"] == expected_backend_port_prestart_gate(
        operator_sequence_step=first_gate_step,
        operator_sequence_step_index=(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_STEP_INDEX
        ),
    )
    assert packet["backend_port_prestart_gate"]["required_markers"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS
    )
    assert packet["expected_backend_port_prestart_gate"] == packet[
        "backend_port_prestart_gate"
    ]
    assert packet["backend_port_prestart_gate_matches_expected"] is True
    anchor_digest_review_step = packet["operator_sequence"][-2]
    assert "mvp_blocker_anchor" in anchor_digest_review_step
    assert "mvp_blocker_anchor_sha256" in anchor_digest_review_step
    assert "current first open blocker" in anchor_digest_review_step
    assert "supervised proof planning" in anchor_digest_review_step
    backend_connection_step = packet["operator_sequence"][
        WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP_INDEX
    ]
    assert backend_connection_step == (
        WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_STEP
    )
    for marker in WING_READONLY_PREFLIGHT_BACKEND_CONNECTION_CONFIRMATION_REQUIRED_MARKERS:
        assert marker in backend_connection_step
    assert packet["operator_sequence"].index(backend_connection_step) < packet[
        "operator_sequence"
    ].index("Run capture_readonly_preflight.")
    validate_step = packet["operator_sequence"][
        WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX
    ]
    assert validate_step == WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP
    assert packet["validate_run_inputs_only_step_index"] == (
        WING_READONLY_PREFLIGHT_VALIDATE_INPUTS_OPERATOR_STEP_INDEX
    )
    assert packet["validate_run_inputs_only_step"] == validate_step
    assert packet["validate_run_inputs_only_before_capture"] is True
    assert packet["validate_run_inputs_only_transport_opened"] is False
    assert packet["validate_run_inputs_only_read_only"] is True
    assert packet["validate_run_inputs_only_live_write"] is False
    assert packet["validate_run_inputs_only_contacts_backend"] is False
    assert packet["validate_run_inputs_only_contacts_wing"] is False
    assert packet["validate_run_inputs_only_queries_console"] is False
    assert packet["validate_run_inputs_only_write_request_sent"] is False
    assert packet["validate_run_inputs_only_proposal_request_sent"] is False
    assert packet["validate_run_inputs_only_readiness_closure_allowed"] is False
    assert packet["validate_run_inputs_only_grants_live_authority"] is False
    assert packet["operator_sequence"].index(validate_step) < packet[
        "operator_sequence"
    ].index(WING_READONLY_PREFLIGHT_CAPTURE_OPERATOR_STEP)
    supervised_gate_step = packet["operator_sequence"][-1]
    assert "accepted_decision_ready" in supervised_gate_step
    assert "valid accepted-candidate prestart trace" in supervised_gate_step
    assert "valid accepted-candidate artifact-root trace" in supervised_gate_step
    assert "expected-artifact-bound accepted source id" in supervised_gate_step
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in supervised_gate_step
    assert (
        supervised_gate_step == WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP
    )
    assert packet["supervised_prestart_gate"] == expected_supervised_prestart_gate(
        operator_sequence_step=supervised_gate_step,
        operator_sequence_step_index=(
            WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_STEP_INDEX
        ),
    )
    assert packet["supervised_prestart_gate"]["required_markers"] == list(
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS
    )
    assert packet["supervised_prestart_gate"]["unlocks_runbook"] == (
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK
    )
    assert (
        packet["supervised_prestart_gate"][
            "requires_accepted_candidate_expected_artifact_trace"
        ]
        is True
    )
    assert packet["expected_supervised_prestart_gate"] == packet[
        "supervised_prestart_gate"
    ]
    assert packet["supervised_prestart_gate_matches_expected"] is True
    assert packet["required_supervised_proof_plan_summary_fields"] == list(
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_roots"
        in packet["required_supervised_proof_plan_summary_fields"]
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source"
        in packet["required_supervised_proof_plan_summary_fields"]
    )
    for field_name in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert field_name in packet["required_supervised_proof_plan_summary_fields"]
    assert packet["expected_required_supervised_proof_plan_summary_fields"] == list(
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS
    )
    assert (
        packet["required_supervised_proof_plan_summary_fields_match_contract"]
        is True
    )
    assert (
        packet[
            "required_supervised_proof_plan_summary_fields_include_invalid_records"
        ]
        is True
    )
    rendered = handoff.render_markdown(packet)
    assert "## Backend Port Prestart Gate" in rendered
    assert (
        "backend_port_prestart_summary_type: "
        f"`{WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE}`"
    ) in rendered
    assert (
        "backend_port_prestart_summary_status: "
        f"`{WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS}`"
    ) in rendered
    assert (
        "backend_port_prestart_summary_connection_policy: "
        f"`{WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY}`"
    ) in rendered
    assert "backend_port_prestart_summary_read_only: `true`" in rendered
    assert "backend_port_prestart_summary_live_write: `false`" in rendered
    assert "backend_port_prestart_summary_contacts_backend: `false`" in rendered
    assert "backend_port_prestart_summary_contacts_wing: `false`" in rendered
    assert (
        "backend_port_prestart_summary_mutates_console_state: `false`"
        in rendered
    )
    assert "backend_port_prestart_summary_queries_console: `false`" in rendered
    assert "backend_port_prestart_summary_executes_tests: `false`" in rendered
    assert (
        "backend_port_prestart_check_status: "
        f"`{WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS}`"
    ) in rendered
    assert "## Expected Backend Port Prestart Gate" in rendered
    assert "expected_backend_port_prestart_gate" in rendered
    assert "backend_port_prestart_gate_matches_expected: `true`" in rendered
    assert "## Supervised Prestart Gate" in rendered
    assert "## Expected Supervised Prestart Gate" in rendered
    assert "expected_supervised_prestart_gate" in rendered
    assert "supervised_prestart_gate_matches_expected: `true`" in rendered
    assert "## Proof Identity" in rendered
    assert WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA in rendered
    assert WING_READONLY_PREFLIGHT_PROOF_RUNNER in rendered
    assert WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER in rendered
    assert WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS in rendered
    assert "## MVP Blocker Anchor" in rendered
    assert "WING read-only connection proof" in rendered
    assert "selected_evidence_file_count" in rendered
    assert "mvp_blocker_anchor_sha256" in rendered
    assert packet["mvp_blocker_anchor_sha256"] in rendered
    assert "## Project Director Selected Action Anchor" in rendered
    assert "project_director_selected_action_source: `Current First Command`" in rendered
    assert (
        "expected_project_director_selected_action_source: `Current First Command`"
        in rendered
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: `true`"
        in rendered
    )
    assert "project_director_selected_action_sha256" in rendered
    assert packet["project_director_selected_action_sha256"] in rendered
    assert "actual_project_director_selected_action_sha256" in rendered
    assert "expected_project_director_selected_action_sha256" in rendered
    assert (
        "project_director_selected_action_sha256_available_matches_digest: `true`"
        in rendered
    )
    assert "project_director_selected_action_sha256_matches_anchor: `true`" in rendered
    assert "project_director_selected_action_sha256_matches_text: `true`" in rendered
    assert (
        "project_director_selected_action_sha256_matches_current_directive: `true`"
        in rendered
    )
    assert (
        "project_director_selected_action_anchor_matches_current_directive: `true`"
        in rendered
    )
    assert "expected_project_director_selected_action_anchor" in rendered
    assert (
        "project_director_current_first_command_baseline_coverage_required: `true`"
        in rendered
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: `true`"
        in rendered
    )
    assert (
        "current_first_command_baseline_missing_markers: `[]`"
        in rendered
    )
    assert "operator_sequence_sha256" in rendered
    assert packet["operator_sequence_sha256"] in rendered
    assert "expected_operator_sequence_sha256" in rendered
    assert packet["expected_operator_sequence_sha256"] in rendered
    assert "operator_sequence_matches_contract: `true`" in rendered
    assert "operator_sequence_sha256_matches_contract: `true`" in rendered
    assert "validate_readonly_preflight_inputs" in rendered
    assert "validate_run_inputs_only_before_capture: `true`" in rendered
    assert "validate_run_inputs_only_transport_opened: `false`" in rendered
    assert "validate_run_inputs_only_read_only: `true`" in rendered
    assert "validate_run_inputs_only_live_write: `false`" in rendered
    assert "validate_run_inputs_only_contacts_backend: `false`" in rendered
    assert "validate_run_inputs_only_contacts_wing: `false`" in rendered
    assert "validate_run_inputs_only_queries_console: `false`" in rendered
    assert "validate_run_inputs_only_write_request_sent: `false`" in rendered
    assert "validate_run_inputs_only_proposal_request_sent: `false`" in rendered
    assert (
        "validate_run_inputs_only_readiness_closure_allowed: `false`" in rendered
    )
    assert "validate_run_inputs_only_grants_live_authority: `false`" in rendered
    assert "tools/wing_readonly_preflight_runner.py" in rendered
    assert "actual_matches_expected: `true`" in rendered
    assert "operator_sequence_step_must_be_final: `true`" in rendered
    assert "grants_live_authority: `false`" in rendered
    assert "valid accepted-candidate prestart trace" in rendered
    assert "valid accepted-candidate artifact-root trace" in rendered
    assert "expected-artifact-bound accepted source id" in rendered
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in rendered
    assert "required_supervised_proof_plan_summary_fields" in rendered
    assert "expected_required_supervised_proof_plan_summary_fields" in rendered
    assert (
        "required_supervised_proof_plan_summary_fields_match_contract: `true`"
        in rendered
    )
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: `true`"
        in rendered
    )
    for field_name in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert field_name in rendered
    assert "decision_template_required_fields" in rendered
    assert "decision_template_prestart_proof_fields" in rendered
    assert "decision_template_required_fields_match_contract: `true`" in rendered
    assert (
        "decision_template_prestart_proof_fields_match_contract: `true`" in rendered
    )
    assert "decision_template_prestart_proof_fields_required: `true`" in rendered
    assert (
        "decision_template_required_fields_include_prestart_proof_fields: `true`"
        in rendered
    )
    assert (
        "decision_template_preserve_prestart_proof_fields_before_operator_edit: `true`"
        in rendered
    )
    assert "backend_port_prestart_summary_path" in rendered
    assert "backend_port_prestart_summary_sha256" in rendered
    assert "artifact_manifest_verifier_websocket_opened_after_pass" in rendered
    assert "readonly_preflight_review_decision_expected_candidate_prestart_source_ids" in rendered
    assert "readonly_preflight_review_decision_invalid_records" in rendered
    assert (
        "next_manual_artifact_required: "
        f"`{WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED}`"
    ) in rendered
    assert "--decision-template-output" in packet["commands"][
        "review_artifact_and_write_pending_decision_template"
    ]
    assert packet["decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert packet["decision_template_output_matches_contract"] is True
    assert packet["decision_template_required_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert packet["expected_decision_template_required_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert packet["decision_template_required_fields_count"] == len(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert packet["expected_decision_template_required_fields_count"] == len(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert packet["decision_template_required_fields_match_contract"] is True
    assert packet["decision_template_prestart_proof_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
    )
    assert packet["expected_decision_template_prestart_proof_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
    )
    assert packet["decision_template_prestart_proof_fields_count"] == len(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
    )
    assert packet["expected_decision_template_prestart_proof_fields_count"] == len(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
    )
    assert packet["decision_template_prestart_proof_fields_match_contract"] is True
    assert packet["decision_template_prestart_proof_fields_required"] is True
    assert (
        packet["decision_template_required_fields_include_prestart_proof_fields"]
        is True
    )
    assert (
        packet["decision_template_preserve_prestart_proof_fields_before_operator_edit"]
        is True
    )
    assert WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT in packet[
        "commands"
    ]["review_artifact_and_write_pending_decision_template"]
    assert packet["commands"]["review_edited_decision_json"] == [
        "python3",
        "tools/review_wing_readonly_preflight_decisions.py",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--decision-root",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
        "--output",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    ]
    assert packet["review_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert packet["review_output_matches_contract"] is True
    assert packet["decision_review_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert packet["decision_review_output_matches_contract"] is True
    assert "confirm_no_live_authority" in packet["required_review_checklist_ids"]
    assert "confirm_operator_session_ready" in packet["required_review_checklist_ids"]
    assert "confirm_event_log_run_id_binding" in packet["required_review_checklist_ids"]
    assert packet["required_review_checklist_ids"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )
    forbidden_request_types = set(packet["forbidden_live_request_types"])
    assert "apply_operator_proposal" in forbidden_request_types
    assert "create_operator_proposal" in forbidden_request_types
    assert "rollback_last_manual_write" in forbidden_request_types
    assert "emergency_stop" in forbidden_request_types
    assert "set_fader" in forbidden_request_types
    assert "set_gain" in forbidden_request_types
    assert "set_trim" in forbidden_request_types
    assert "set_input_trim" in forbidden_request_types
    assert "set_channel_trim" in forbidden_request_types
    assert "set_preamp_gain" in forbidden_request_types
    assert "set_eq" in forbidden_request_types
    assert "set_compressor" in forbidden_request_types
    assert "route_output" in forbidden_request_types
    assert "load_snap" in forbidden_request_types
    assert "save_snap" in forbidden_request_types
    assert "reset" in forbidden_request_types
    assert "mute" in forbidden_request_types
    assert "unmute" in forbidden_request_types
    assert "set_fx_parameter" in forbidden_request_types
    assert len(packet["forbidden_live_request_types"]) == len(forbidden_request_types)
    assert forbidden_request_types == set(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert "wing_readonly_connection_proof" in packet["blockers_not_closed_by_packet"]
    assert packet["blockers_not_closed_by_packet"] == list(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )


def test_handoff_packet_forbidden_live_request_types_matches_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")
    expected_forbidden = set(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    packet_forbidden = set(packet["forbidden_live_request_types"])

    assert packet_forbidden == expected_forbidden
    assert len(packet["forbidden_live_request_types"]) == len(packet_forbidden)


def test_handoff_packet_response_type_map_matches_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["expected_response_types_by_request_type"] == {
        request_type: list(response_types)
        for request_type, response_types in (
            WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
        )
    }
    assert handoff.EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE == packet[
        "expected_response_types_by_request_type"
    ]


def test_handoff_packet_allowed_requests_match_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["allowed_request_types"] == list(
        WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
    )
    assert handoff.ALLOWED_REQUEST_TYPES == packet["allowed_request_types"]
    assert packet["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    assert handoff.REQUEST_ID_POLICY == packet["request_id_policy"]


def test_handoff_packet_proof_identity_matches_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert packet["proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert packet["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert packet["selected_mvp_blocker_status"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert handoff.PROOF_ARTIFACT_SCHEMA == packet["proof_artifact_schema"]
    assert handoff.PROOF_RUNNER == packet["proof_runner"]
    assert handoff.SELECTED_MVP_BLOCKER == packet["selected_mvp_blocker"]
    assert handoff.SELECTED_MVP_BLOCKER_STATUS == packet["selected_mvp_blocker_status"]


def test_handoff_packet_fail_closed_conditions_match_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["fail_closed_conditions"] == list(
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
    )
    assert handoff.FAIL_CLOSED_CONDITIONS == packet["fail_closed_conditions"]


def test_handoff_packet_request_sequence_uses_runner_shape():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["request_sequence"] == handoff.build_request_sequence(
        [3],
        ["fader", "gain", "pan"],
    )
    assert packet["request_count"] == len(packet["request_sequence"])
    assert handoff.request_id_for(3, "gain") == "wing-readonly-preflight:ch3:gain"
    assert handoff.request_address(3, "gain") == "/ch/3/in/set/trim"


def test_handoff_packet_review_checklist_ids_match_shared_contract():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")

    assert packet["required_review_checklist_ids"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )


def test_handoff_packet_supports_custom_channel_confirmation():
    packet = handoff.build_handoff_packet(channels="2,4", generated_at="2026-06-07T00:00:00Z")

    assert packet["channels"] == [2, 4]
    preview = packet["commands"]["preview_readonly_request_plan"]
    assert preview[preview.index("--channels") + 1] == "2,4"
    assert preview[preview.index("--operator-confirm-channels") + 1] == "2,4"
    assert "--operator-confirm-session-ready" in preview
    command = packet["commands"]["capture_readonly_preflight"]
    assert command[command.index("--channels") + 1] == "2,4"
    assert command[command.index("--operator-confirm-channels") + 1] == "2,4"
    assert "--operator-confirm-session-ready" in command


def test_handoff_packet_rejects_invalid_channels():
    try:
        handoff.build_handoff_packet(channels="0")
    except ValueError as exc:
        assert str(exc) == "channels_must_be_positive"
    else:
        raise AssertionError("expected invalid channel to fail")


def test_handoff_packet_rejects_invalid_value_types(tmp_path: Path):
    invalid_cases = [
        ("", "value_types_must_include_at_least_one_value_type"),
        ("fader,fader", "value_types_must_not_contain_duplicates"),
        ("fader,eq", "value_types_unsupported:eq"),
    ]

    for value_types, expected_error in invalid_cases:
        try:
            handoff.build_handoff_packet(value_types=value_types)
        except ValueError as exc:
            assert str(exc) == expected_error
        else:
            raise AssertionError("expected invalid value type to fail")

    output = tmp_path / "handoff.md"
    assert (
        handoff.main(
            [
                "--value-types",
                "fader,eq",
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 2
    )
    assert not output.exists()


def test_handoff_packet_rejects_channels_outside_wing_input_range(tmp_path: Path):
    try:
        handoff.build_handoff_packet(channels="49")
    except ValueError as exc:
        assert str(exc) == "channels_must_be_in_1_48"
    else:
        raise AssertionError("expected out-of-range channel to fail")

    output = tmp_path / "handoff.md"
    assert handoff.main(["--channels", "49", "--output", str(output), "--quiet"]) == 2
    assert not output.exists()


def test_handoff_markdown_renders_commands_and_boundaries():
    packet = handoff.build_handoff_packet(generated_at="2026-06-07T00:00:00Z")
    markdown = handoff.render_markdown(packet)

    assert "WING Read-Only Preflight Handoff Packet" in markdown
    assert "live_write: `false`" in markdown
    assert "expected_plan_required_for_real_run: `true`" in markdown
    assert "physical_preflight_requires_existing_backend_connection: `true`" in markdown
    assert "backend_port_prestart_check_required: `true`" in markdown
    assert "backend_port_prestart_check_must_pass_before_starting_backend: `true`" in markdown
    assert "backend_port_prestart_summary_required: `true`" in markdown
    assert "backend_port_prestart_summary_path_placeholder" in markdown
    assert WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER in markdown
    assert "backend_port_prestart_summary_sha256: ``" in markdown
    assert "backend_port_prestart_summary_sha256_available: `false`" in markdown
    assert "backend_port_prestart_check_operator_sequence_step_index: `0`" in markdown
    assert "backend_port_prestart_check_must_be_first_operator_step: `true`" in markdown
    assert "backend_port_prestart_check_command" in markdown
    assert "backend_port_prestart_check_command_text" in markdown
    assert WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT in markdown
    assert "backend_port_prestart_check_failure_detail_fields" in markdown
    for field_name in WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS:
        assert field_name in markdown
    assert "prestart_backend_port_check" in markdown
    assert "tools/no_wing_release_check.py" in markdown
    assert "verify_no_wing_artifact_manifest" in markdown
    assert "tools/verify_no_wing_artifact_manifest.py" in markdown
    assert "artifact_manifest_verifier_required: `true`" in markdown
    assert "artifact_manifest_verifier_command" in markdown
    assert (
        "artifact_manifest_verifier_must_pass_before_backend_wing_session: `true`"
        in markdown
    )
    assert "artifact_manifest_verifier_read_only: `true`" in markdown
    assert "artifact_manifest_verifier_live_write: `false`" in markdown
    assert "artifact_manifest_verifier_contacts_backend: `false`" in markdown
    assert "artifact_manifest_verifier_contacts_wing: `false`" in markdown
    assert "artifact_manifest_verifier_mutates_console_state: `false`" in markdown
    assert "artifact_manifest_verifier_queries_console: `false`" in markdown
    assert "artifact_manifest_verifier_executes_tests: `false`" in markdown
    assert (
        "artifact_manifest_verifier_readiness_matrix_update_allowed: `false`"
        in markdown
    )
    assert (
        "artifact_manifest_verifier_readiness_closure_allowed: `false`"
        in markdown
    )
    assert "artifact_manifest_verifier_live_authority_granted: `false`" in markdown
    assert (
        markdown.index("### prestart_backend_port_check")
        < markdown.index("### verify_no_wing_artifact_manifest")
        < markdown.index("### preview_readonly_request_plan")
    )
    assert "manual_review_required_before_readiness_update: `true`" in markdown
    assert "preview_readonly_request_plan" in markdown
    assert "--plan-only" in markdown
    assert "--operator-confirm-session-ready" in markdown
    assert "--prestart-summary" in markdown
    assert "capture_readonly_preflight" in markdown
    assert "review_edited_decision_json" in markdown
    assert "Expected Response Types" in markdown
    assert "allowed_request_types" in markdown
    assert "query_mixer_parameter" in markdown
    assert "request_id_policy" in markdown
    assert WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY in markdown
    assert "expected_response_types_by_request_type" in markdown
    assert "mixer_parameter_read_blocked" in markdown
    assert "Expected Request Sequence" in markdown
    assert "request_sequence" in markdown
    assert "wing-readonly-preflight:ch3:fader" in markdown
    assert "/ch/3/in/set/trim" in markdown
    assert "Fail-Closed Conditions" in markdown
    assert "fail_closed_conditions" in markdown
    assert "unexpected_response_type" in markdown
    assert "event_log_run_id_mismatch" in markdown
    assert "manual_review_decision_missing_or_invalid" in markdown
    assert "expected_artifacts" in markdown
    assert WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME in markdown
    assert WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME in markdown
    assert "Expected Supervised Prestart Gate" in markdown
    assert "actual_matches_expected: `true`" in markdown
    assert "Proof Identity" in markdown
    assert "proof_artifact_schema" in markdown
    assert WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA in markdown
    assert WING_READONLY_PREFLIGHT_PROOF_RUNNER in markdown
    assert WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER in markdown
    assert WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS in markdown
    assert "pending_review_decision_template.json" in markdown
    assert "pending decision template matches contract: `true`" in markdown
    assert "artifact review report matches contract: `true`" in markdown
    assert "decision review report matches contract: `true`" in markdown
    assert "confirm_operator_session_ready" in markdown
    assert "full_autonomous_ready: `false`" in markdown
    assert (
        "next_manual_artifact_required: "
        f"`{WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED}`"
    ) in markdown


def test_handoff_tool_main_writes_markdown_and_json(tmp_path: Path):
    markdown_output = tmp_path / "handoff.md"
    json_output = tmp_path / "handoff.json"

    assert handoff.main(["--output", str(markdown_output), "--quiet"]) == 0
    assert "WING Read-Only Preflight Handoff Packet" in markdown_output.read_text(
        encoding="utf-8"
    )

    assert handoff.main(["--format", "json", "--output", str(json_output), "--quiet"]) == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["type"] == "wing_readonly_preflight_handoff_packet"
    assert payload["expected_plan_required_for_real_run"] is True
    assert payload["physical_preflight_requires_existing_backend_connection"] is True
    assert payload["backend_port_prestart_check_required"] is True
    assert payload["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert payload["backend_port_prestart_summary_required"] is True
    assert payload["backend_port_prestart_summary_path_placeholder"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert payload["backend_port_prestart_summary_sha256"] == ""
    assert payload["backend_port_prestart_summary_sha256_available"] is False
    assert payload["backend_port_prestart_check_operator_sequence_step_index"] == 0
    assert payload["backend_port_prestart_check_must_be_first_operator_step"] is True
    assert payload["allowed_request_types"] == list(
        WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
    )
    assert payload["request_id_policy"] == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    assert payload["proof_artifact_schema"] == WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    assert payload["proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert payload["selected_mvp_blocker"] == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    assert payload["selected_mvp_blocker_status"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert payload["expected_response_types_by_request_type"] == {
        request_type: list(response_types)
        for request_type, response_types in (
            WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
        )
    }
    assert payload["fail_closed_conditions"] == list(
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
    )
    assert payload["request_count"] == 5
    assert payload["request_sequence"] == handoff.build_request_sequence(
        [3],
        ["fader", "gain", "pan"],
    )
    assert payload["expected_artifacts"] == [
        payload["expected_run_artifact_glob"],
        payload["expected_event_log_glob"],
    ]
    assert payload["next_manual_artifact_required"] == (
        WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
    )
    assert payload["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert (
        payload["backend_port_prestart_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    )
    assert payload["manual_review_required_before_readiness_update"] is True
    assert payload["live_authority_granted"] is False


def test_handoff_packet_hashes_existing_prestart_summary(tmp_path: Path):
    prestart_summary = tmp_path / "summary.json"
    write_no_wing_prestart_summary(prestart_summary)
    expected_sha256 = hashlib.sha256(prestart_summary.read_bytes()).hexdigest()

    packet = handoff.build_handoff_packet(
        prestart_summary=str(prestart_summary),
        generated_at="2026-06-07T00:00:00Z",
    )

    assert packet["backend_port_prestart_summary_path_placeholder"] == str(
        prestart_summary
    )
    assert packet["backend_port_prestart_summary_sha256"] == expected_sha256
    assert packet["backend_port_prestart_summary_sha256_available"] is True
    assert packet["backend_port_prestart_summary_type"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
    )
    assert packet["backend_port_prestart_summary_status"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
    )
    assert packet["backend_port_prestart_summary_connection_policy"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
    )
    assert packet["backend_port_prestart_summary_read_only"] is True
    assert packet["backend_port_prestart_summary_live_write"] is False
    assert packet["backend_port_prestart_summary_contacts_backend"] is False
    assert packet["backend_port_prestart_summary_contacts_wing"] is False
    assert packet["backend_port_prestart_summary_mutates_console_state"] is False
    assert packet["backend_port_prestart_summary_queries_console"] is False
    assert packet["backend_port_prestart_summary_executes_tests"] is False
    assert packet["backend_port_prestart_check_status"] == (
        WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
    )
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS:
        assert packet[field] is False
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS:
        assert packet[field] is True
    assert packet["safe_to_stop_listener_by_gate_matches_plan"] is True
    assert packet["live_authority_granted_matches_plan"] is True
    assert (
        packet["backend_port_prestart_gate_execution_allowed_matches_plan"]
        is True
    )
    assert (
        packet["backend_port_prestart_gate_readiness_closure_allowed_matches_plan"]
        is True
    )
    assert (
        packet["backend_port_prestart_gate_live_authority_granted_matches_plan"]
        is True
    )
    assert packet["commands"]["preview_readonly_request_plan"][
        packet["commands"]["preview_readonly_request_plan"].index("--prestart-summary")
        + 1
    ] == str(prestart_summary)
    assert packet["commands"]["capture_readonly_preflight"][
        packet["commands"]["capture_readonly_preflight"].index("--prestart-summary")
        + 1
    ] == str(prestart_summary)


def test_handoff_packet_preserves_environment_blocked_prestart_summary_metadata(
    tmp_path: Path,
):
    prestart_summary = tmp_path / "environment-blocked-summary.json"
    write_no_wing_prestart_summary(
        prestart_summary,
        status="failed",
        environment_blocked=True,
    )
    expected_sha256 = hashlib.sha256(prestart_summary.read_bytes()).hexdigest()

    packet = handoff.build_handoff_packet(
        prestart_summary=str(prestart_summary),
        generated_at="2026-06-07T00:00:00Z",
    )
    rendered = handoff.render_markdown(packet)

    assert packet["backend_port_prestart_summary_sha256"] == expected_sha256
    assert packet["backend_port_prestart_summary_sha256_available"] is True
    assert packet["backend_port_prestart_summary_type"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
    )
    assert packet["backend_port_prestart_summary_status"] == "failed"
    assert packet["backend_port_prestart_summary_status"] != (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
    )
    assert packet["backend_port_prestart_summary_environment_blocked_checks"] == [
        "frontend_e2e"
    ]
    assert packet[
        "backend_port_prestart_summary_environment_blocked_check_details"
    ][0]["reason"] == "local_dev_server_bind_denied"
    assert packet[
        "backend_port_prestart_summary_environment_blocked_check_details"
    ][0]["address"] == "127.0.0.1:3000"
    assert packet["execution_allowed_by_packet"] is False
    assert packet["live_authority_granted"] is False
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS:
        assert packet[field] is False
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS:
        assert packet[field] is True
    assert packet["safe_to_stop_listener_by_gate_matches_plan"] is True
    assert packet["live_authority_granted_matches_plan"] is True
    assert (
        packet["backend_port_prestart_gate_execution_allowed_matches_plan"]
        is True
    )
    assert (
        packet["backend_port_prestart_gate_readiness_closure_allowed_matches_plan"]
        is True
    )
    assert (
        packet["backend_port_prestart_gate_live_authority_granted_matches_plan"]
        is True
    )
    assert "backend_port_prestart_summary_status: `failed`" in rendered
    assert "local_dev_server_bind_denied" in rendered
    assert "127.0.0.1:3000" in rendered
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS:
        assert f"{field}: `false`" in rendered
    for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS:
        assert f"{field}: `true`" in rendered
    assert "safe_to_stop_listener_by_gate_matches_plan: `true`" in rendered
    assert "live_authority_granted_matches_plan: `true`" in rendered


def test_handoff_packet_denial_match_flags_detect_prestart_summary_drift(
    tmp_path: Path,
):
    prestart_summary = tmp_path / "drifted-summary.json"
    write_no_wing_prestart_summary(prestart_summary)
    payload = json.loads(prestart_summary.read_text(encoding="utf-8"))
    detail = payload["results"][0]["detail"]
    detail["safe_to_stop_listener_by_gate"] = True
    detail["live_authority_granted"] = True
    detail["backend_port_prestart_gate_execution_allowed"] = True
    detail["backend_port_prestart_gate_readiness_closure_allowed"] = True
    detail["backend_port_prestart_gate_live_authority_granted"] = True
    prestart_summary.write_text(json.dumps(payload), encoding="utf-8")

    packet = handoff.build_handoff_packet(
        prestart_summary=str(prestart_summary),
        generated_at="2026-06-07T00:00:00Z",
    )
    rendered = handoff.render_markdown(packet)

    assert packet["safe_to_stop_listener_by_gate"] is True
    assert packet["backend_port_prestart_gate_execution_allowed"] is True
    assert packet["backend_port_prestart_gate_readiness_closure_allowed"] is True
    assert packet["backend_port_prestart_gate_live_authority_granted"] is True
    assert packet["safe_to_stop_listener_by_gate_matches_plan"] is False
    assert packet["live_authority_granted_matches_plan"] is False
    assert (
        packet["backend_port_prestart_gate_execution_allowed_matches_plan"]
        is False
    )
    assert (
        packet["backend_port_prestart_gate_readiness_closure_allowed_matches_plan"]
        is False
    )
    assert (
        packet["backend_port_prestart_gate_live_authority_granted_matches_plan"]
        is False
    )
    assert "safe_to_stop_listener_by_gate_matches_plan: `false`" in rendered
    assert "live_authority_granted_matches_plan: `false`" in rendered


def test_handoff_packet_denial_plan_fields_detect_prestart_summary_drift(
    tmp_path: Path,
):
    prestart_summary = tmp_path / "plan-drifted-summary.json"
    write_no_wing_prestart_summary(prestart_summary)
    payload = json.loads(prestart_summary.read_text(encoding="utf-8"))
    detail = payload["results"][0]["detail"]
    detail["plan_safe_to_stop_listener_by_gate"] = True
    prestart_summary.write_text(json.dumps(payload), encoding="utf-8")

    packet = handoff.build_handoff_packet(
        prestart_summary=str(prestart_summary),
        generated_at="2026-06-07T00:00:00Z",
    )
    rendered = handoff.render_markdown(packet)

    assert packet["safe_to_stop_listener_by_gate"] is False
    assert packet["plan_safe_to_stop_listener_by_gate"] is True
    assert packet["safe_to_stop_listener_by_gate_matches_plan"] is False
    assert "plan_safe_to_stop_listener_by_gate: `true`" in rendered
    assert "safe_to_stop_listener_by_gate_matches_plan: `false`" in rendered


def test_handoff_tool_runs_as_repo_local_script_path(tmp_path: Path):
    output = tmp_path / "handoff.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_wing_readonly_preflight_handoff_packet.py",
            "--channels",
            "3",
            "--format",
            "json",
            "--output",
            str(output),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["type"] == "wing_readonly_preflight_handoff_packet"
    assert payload["commands"]["preview_readonly_request_plan"][1] == (
        "tools/wing_readonly_preflight_runner.py"
    )
    assert payload["live_authority_granted"] is False


def test_handoff_tool_main_fails_closed_for_invalid_channel(tmp_path: Path):
    output = tmp_path / "handoff.md"

    assert handoff.main(["--channels", "bad", "--output", str(output), "--quiet"]) == 2
    assert not output.exists()


def test_handoff_tool_is_transport_free():
    source = Path("tools/build_wing_readonly_preflight_handoff_packet.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader(",
        "set_gain(",
        "apply_operator_proposal(",
    ]
    for token in forbidden:
        assert token not in source
