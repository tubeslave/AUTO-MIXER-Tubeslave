import json
import socket
import sys
from pathlib import Path

import pytest

from backend.operator_mode_policy import MODE_SUPERVISED, build_operator_mode_status
from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS,
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES,
    WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS,
    WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES,
    WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES,
    WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_RUNNER_URL,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
    WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
    WING_READONLY_PREFLIGHT_PROOF_RUNNER,
    WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
    WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
    WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
    WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK,
)
from tools import no_wing_release_check as release_check
from tools.no_wing_release_check import (
    CheckResult,
    CommandCheck,
    autonomy_evidence_overlay_summary,
    autonomy_gap_rollup_summary,
    bus_group_main_dca_readback_artifact_intake_summary,
    bus_group_main_dca_readback_contract_summary,
    build_checks,
    channel_basic_controls_readback_artifact_intake_summary,
    channel_basic_controls_readback_contract_summary,
    channel_processing_readback_artifact_intake_summary,
    channel_processing_readback_contract_summary,
    current_supervised_live_write_kinds,
    current_supervised_live_write_surfaces,
    dlive_state_readback_artifact_intake_summary,
    dlive_state_readback_contract_summary,
    fx_slots_inserts_readback_artifact_intake_summary,
    fx_slots_inserts_readback_contract_summary,
    gain_proof_artifact_root_for_run,
    matrix_state_readback_artifact_intake_summary,
    matrix_state_readback_contract_summary,
    next_wing_proof_plan_summary,
    proof_report_path_for_run,
    raw_mixer_readback_artifact_intake_summary,
    raw_mixer_readback_contract_summary,
    read_surface_physical_readback_acceptance_template_path_for_run,
    read_surface_physical_readback_acceptance_template_artifact_summary,
    read_surface_physical_readback_artifact_review_queue_artifact_summary,
    read_surface_physical_readback_artifact_review_queue_path_for_run,
    read_surface_physical_readback_capture_packet_artifact_summary,
    read_surface_physical_readback_capture_packet_path_for_run,
    read_surface_physical_readback_capture_template_artifact_summary,
    read_surface_physical_readback_capture_template_path_for_run,
    read_surface_physical_readback_artifact_review_queue_summary,
    read_surface_physical_readback_fixture_acceptance_ledger_summary,
    read_surface_physical_readback_fixture_closure_plan_artifact_summary,
    read_surface_physical_readback_fixture_closure_plan_summary,
    read_surface_physical_readback_fixture_closure_plan_path_for_run,
    read_surface_physical_readback_capture_plan_summary,
    wing_readonly_preflight_evidence_summary,
    wing_readonly_preflight_handoff_packet_artifact_summary,
    wing_readonly_preflight_plan_artifact_summary,
    wing_readonly_preflight_review_decision_summary,
    read_surface_fixture_evidence_summary,
    read_surface_remediation_plan_summary,
    routing_sends_readback_contract_summary,
    routing_sends_readback_artifact_intake_summary,
    run_command_check,
    run_release_checks,
    snapshots_scenes_readback_artifact_intake_summary,
    supervised_gain_dry_run_summary,
    user_presets_readback_contract_summary,
    user_presets_readback_artifact_intake_summary,
    validate_repo_local_commands,
    validate_no_live_commands,
    validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot,
    validate_bus_group_main_dca_physical_readback_contract_snapshot,
    validate_channel_basic_controls_physical_readback_artifact_intake_snapshot,
    validate_channel_basic_controls_physical_readback_contract_snapshot,
    validate_channel_processing_physical_readback_artifact_intake_snapshot,
    validate_channel_processing_physical_readback_contract_snapshot,
    validate_dlive_state_physical_readback_artifact_intake_snapshot,
    validate_dlive_state_physical_readback_contract_snapshot,
    validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot,
    validate_fx_slots_inserts_physical_readback_contract_snapshot,
    validate_matrix_state_physical_readback_artifact_intake_snapshot,
    validate_matrix_state_physical_readback_contract_snapshot,
    validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot,
    validate_read_surface_physical_readback_artifact_review_queue_snapshot,
    validate_read_surface_physical_readback_acceptance_template_artifact_snapshot,
    validate_read_surface_physical_readback_capture_packet_artifact_snapshot,
    validate_read_surface_physical_readback_capture_template_artifact_snapshot,
    validate_read_surface_physical_readback_fixture_acceptance_ledger_snapshot,
    validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot,
    validate_read_surface_physical_readback_fixture_closure_plan_snapshot,
    validate_wing_readonly_preflight_evidence_snapshot,
    validate_wing_readonly_preflight_handoff_packet_artifact_snapshot,
    validate_wing_readonly_preflight_plan_artifact_snapshot,
    validate_wing_readonly_preflight_review_decision_ledger_snapshot,
    validate_read_surface_fixture_evidence_snapshot,
    validate_read_surface_physical_readback_capture_plan_snapshot,
    validate_raw_mixer_state_physical_readback_artifact_intake_snapshot,
    validate_raw_mixer_state_physical_readback_contract_snapshot,
    validate_read_surface_remediation_plan_snapshot,
    validate_routing_sends_physical_readback_contract_snapshot,
    validate_routing_sends_physical_readback_artifact_intake_snapshot,
    validate_snapshots_scenes_physical_readback_artifact_intake_snapshot,
    validate_snapshots_scenes_physical_readback_contract_snapshot,
    validate_user_presets_physical_readback_artifact_intake_snapshot,
    validate_user_presets_physical_readback_contract_snapshot,
    validate_supervised_proof_report_autonomy_overlay,
    validate_supervised_proof_evaluator_physical_readback_policy,
    validate_supervised_proof_report_next_wing_plan,
    validate_wing_operator_proposal_runner_physical_readback_policy,
    validate_supervised_proof_report_gain_dry_run,
    validate_supervised_proof_report_gap_rollup,
    wing_readonly_preflight_handoff_packet_path_for_run,
    wing_readonly_preflight_plan_path_for_run,
    wing_readonly_preflight_prestart_summary_path_for_run,
    write_summary,
)

GAIN_REQUIRED_EVIDENCE_COMPONENTS = [
    "physical_pre_read",
    "bounded_gain_down_apply",
    "apply_readback",
    "rollback_readback",
    "emergency_stop_block",
    "unsafe_gain_up_followup_blocked",
    "gain_up_operator_authored_opt_in",
]


def test_wing_readonly_preflight_prestart_contract_uses_shared_backend_constants():
    assert release_check.WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert list(
        release_check.WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    ) == list(WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS)
    assert "listener_process_error" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_status" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_live_authority_granted" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_accepted_by_template" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_execution_allowed_by_template" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_readiness_closure_allowed_by_template" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert "listener_confirmation_template_required_edits" in (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )


def test_wing_readonly_handoff_gate_contract_uses_shared_constants():
    assert (
        release_check.WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_COMMAND_NAME
    )
    assert list(
        release_check.WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS
    ) == list(WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_GATE_REQUIRED_MARKERS)
    assert list(
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_SUPERVISED_PROOF_GATE_MARKERS
    ) == list(WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS)
    assert (
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_RUNBOOK
        == "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
    )


def test_wing_readonly_handoff_safety_contract_uses_shared_constants():
    assert (
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
        == WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS == (
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
        == WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_PLAN_OUTPUT
        == WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_RUNNER_URL
        == WING_READONLY_PREFLIGHT_RUNNER_URL
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
        == WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
        == WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
        == WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
        == WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
        == WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE
        == {
            request_type: list(response_types)
            for request_type, response_types in (
                WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
            )
        }
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
        == WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    )
    assert "apply_operator_proposal" in (
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert "set_gain" in WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    assert "connect_wing" in (
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert "wing_readonly_connection_proof" in (
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    assert "readiness_matrix_update_based_on_real_proof" in (
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )


def test_wing_readonly_review_checklist_contract_uses_shared_constants():
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
        == WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )
    assert "confirm_artifact_paths" in WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    assert "confirm_no_live_authority" in (
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )


def test_wing_readonly_review_decision_template_contract_uses_shared_constants():
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_OUTPUT
        == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_REVIEW_DECISION_OUTPUT
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert "review_id" in (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert "checklist_ids" in (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )


def test_wing_readonly_prestart_summary_contract_uses_shared_constants():
    assert (
        release_check.WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
        == WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
        == WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
        == WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
        == WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
    )
    assert (
        release_check.WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
        == WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
    )


def test_wing_readonly_prestart_summary_writer_uses_shared_contract_constants(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE",
        "contract_test_summary_type",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS",
        "contract_test_summary_status",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY",
        "contract_test_connection_policy",
    )
    backend_result = CheckResult(
        name=WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME,
        status=WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS,
        started_at="2026-06-13T00:00:00Z",
        finished_at="2026-06-13T00:00:00Z",
        duration_sec=0.0,
        detail={"listener_detected": False},
    )

    output = tmp_path / "summary.json"
    release_check.write_wing_readonly_prestart_summary(
        output,
        generated_at="2026-06-13T00:00:00Z",
        backend_port_result=backend_result,
    )

    payload = json.loads(output.read_text())
    assert payload["type"] == "contract_test_summary_type"
    assert payload["status"] == "contract_test_summary_status"
    assert payload["connection_policy"] == "contract_test_connection_policy"
    assert (
        payload["results"][0]["name"]
        == WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_NAME
    )


def test_wing_readonly_preflight_runbook_lists_prestart_failure_detail_fields():
    runbook_text = Path("Docs/runbooks/WING_READONLY_PREFLIGHT.md").read_text()

    assert "backend_port_free" in runbook_text
    assert "command_line" in runbook_text
    for field_name in WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS:
        assert field_name in runbook_text


def test_offline_no_wing_runbook_lists_wing_proof_candidate_prestart_gate():
    runbook_text = Path("Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md").read_text()

    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        in runbook_text
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
        in runbook_text
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in runbook_text
    )
    assert "backend-port prestart contract" in runbook_text


def test_no_wing_release_check_commands_have_no_live_write_tokens(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_package=True)

    validate_no_live_commands(checks)

    command_text = "\n".join(check.shell_display() for check in checks)
    assert "wing_operator_proposal_validation_runner.py" not in command_text
    assert "wing_supervised_smoke_runner.py" not in command_text
    assert "apply_operator_proposal" not in command_text
    assert "set_fader" not in command_text
    assert "set_gain" not in command_text
    assert "set_eq" not in command_text
    assert "set_compressor" not in command_text
    assert "route_output" not in command_text
    command_tokens = [token for check in checks for token in check.command]
    assert "reset" not in command_tokens
    assert "mute" not in command_text
    assert any(check.name == "write_surface_static_scan" for check in checks)
    assert any(check.name == "memory_lint" for check in checks)
    assert any(check.name == "frontend_dist_package" for check in checks)
    assert all(check.timeout_sec == release_check.DEFAULT_COMMAND_TIMEOUT_SEC for check in checks)


def test_no_wing_release_check_allows_benign_words_containing_forbidden_tokens():
    checks = [
        CommandCheck(
            name="resettable_fixture",
            command=("python3", "tools/offline_resettable_fixture.py"),
            cwd=".",
        ),
        CommandCheck(
            name="commute_report_check",
            command=("python3", "tools/commute_report.py"),
            cwd=".",
        ),
        CommandCheck(
            name="set_gainstaging_fixture",
            command=("python3", "tools/offline_set_gainstaging.py"),
            cwd=".",
        ),
        CommandCheck(
            name="wingclientless_parser",
            command=("python3", "tools/wingclientless_parser.py"),
            cwd=".",
        ),
    ]

    validate_no_live_commands(checks)


def test_no_wing_release_check_can_disable_command_timeouts(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, command_timeout_sec=None)

    assert checks
    assert all(check.timeout_sec is None for check in checks)


def test_no_wing_release_check_allows_configured_git_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIXER_GIT_BIN", "/custom/git")

    checks = build_checks(run_dir=tmp_path, include_frontend=False)
    diff_check = next(check for check in checks if check.name == "diff_check")

    assert diff_check.command == ("/custom/git", "diff", "--check")


def test_no_wing_release_check_repo_local_command_plan_passes(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=True, include_package=True)

    validate_repo_local_commands(checks)


def test_no_wing_release_check_rejects_external_project_command_paths(tmp_path):
    repo_root = tmp_path / "AUTO-MIXER-Tubeslave-main"
    external_project = tmp_path / "paperclip"
    repo_root.mkdir()
    external_project.mkdir()
    checks = [
        CommandCheck(
            name="bad_cwd",
            command=("python3", "tools/offline_check.py"),
            cwd=str(external_project),
        ),
        CommandCheck(
            name="bad_absolute_script",
            command=("python3", str(external_project / "runner.py")),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="bad_relative_script",
            command=("python3", "../paperclip/runner.py"),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="bad_env_path",
            command=("python3", "tools/offline_check.py"),
            cwd=str(repo_root),
            env={"PYTHONPATH": "../paperclip"},
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_repo_local_commands(checks, repo_root=repo_root)

    message = str(excinfo.value)
    assert "repo boundary" in message
    assert "bad_cwd:cwd:" in message
    assert "bad_absolute_script:command_path:" in message
    assert "bad_relative_script:command_path:../paperclip/runner.py" in message
    assert "bad_env_path:env_path:PYTHONPATH" in message


def test_no_wing_release_check_derives_supervised_boundary_from_operator_policy():
    policy_status = build_operator_mode_status(
        mode=MODE_SUPERVISED,
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=False,
    )

    assert current_supervised_live_write_kinds() == tuple(
        policy_status["capabilities"]["allowed_live_write_kinds"]
    )
    assert current_supervised_live_write_kinds() == ("fader", "gain", "pan")
    assert current_supervised_live_write_surfaces() == tuple(
        policy_status["capabilities"]["allowed_live_write_surfaces"]
    )
    assert current_supervised_live_write_surfaces() == ("preamp", "trim", "fader", "pan")


def test_no_wing_release_check_validates_read_surface_fixture_evidence_snapshot():
    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["snapshot_status"] == "fixture_evidence_incomplete"
    assert result.detail["surface_count"] >= 1
    assert result.detail["surface_with_fixture_count"] >= 1
    assert result.detail["status_counts"]["partial_fixture_evidence"] >= 1
    assert result.detail["physical_fixture_missing_surface_ids"]
    assert (
        result.detail["next_physical_readback_surface_id"]
        == result.detail["physical_fixture_missing_surface_ids"][0]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_read_surface_fixture_summary_row_drift(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))
    corrupted["summary"]["surface_count"] += 1
    corrupted["summary"]["surface_with_fixture_count"] = 0
    corrupted["summary"]["status_counts"]["partial_fixture_evidence"] = 0
    corrupted["summary"]["status_counts"]["fixture_evidence_present"] = "not-an-int"
    corrupted["summary"]["physical_fixture_missing_surface_ids"] = []
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["surfaces"][0]["status"] = "unsupported_fixture_status"

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_raw_mixer_state_cache_status_unknown" in result.detail["violations"]
    assert "read_surface_fixture_status_count_fixture_evidence_present_not_integer" in result.detail["violations"]
    assert "read_surface_fixture_surface_count_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_surface_with_fixture_count_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_status_counts_mismatch" in result.detail["violations"]
    assert "read_surface_fixture_physical_missing_surface_ids_mismatch" in result.detail["violations"]
    assert (
        "read_surface_fixture_next_physical_readback_surface_id_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_read_surface_fixture_stale_declarations(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))

    channel_row = next(
        row
        for row in corrupted["surfaces"]
        if row["surface_id"] == "channel_identity_and_basic_controls"
    )
    channel_row["fixture_paths"] = ["tests/missing_read_surface_fixture.py"]
    channel_row["test_names"] = ["test_missing_read_surface_fixture"]

    raw_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "raw_mixer_state_cache")
    raw_row["fixture_paths"] = ["backend/server.py"]
    raw_row["test_names"] = ["test_read_surface_inventory_is_read_only_and_unique"]

    dlive_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "dlive_state")
    dlive_row["status"] = "missing_fixture_evidence"

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_fixture_path_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_test_name_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_raw_mixer_state_cache_fixture_path_not_tests" in result.detail["violations"]
    assert "read_surface_fixture_row_raw_mixer_state_cache_test_name_missing" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_fixture_paths" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_test_names" in result.detail["violations"]


def test_no_wing_release_check_rejects_read_surface_contract_gap_inconsistency(monkeypatch):
    snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    corrupted = json.loads(json.dumps(snapshot))

    channel_row = next(
        row
        for row in corrupted["surfaces"]
        if row["surface_id"] == "channel_identity_and_basic_controls"
    )
    channel_row["status"] = "partial_fixture_evidence"
    channel_row["covered_contracts"].append("physical_readback_fixture")
    channel_row["remaining_gaps"] = [
        gap
        for gap in channel_row["remaining_gaps"]
        if gap not in {"physical_readback_fixture_missing", "hardware_parity_unproven"}
    ]

    matrix_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "matrix_state")
    matrix_row["status"] = "fixture_evidence_present"
    matrix_row["covered_contracts"] = [
        contract
        for contract in matrix_row["covered_contracts"]
        if contract != "physical_readback_fixture"
    ]
    matrix_row["remaining_gaps"].append("physical_readback_fixture_missing")
    matrix_row["remaining_gaps"].append("hardware_parity_unproven")

    dlive_row = next(row for row in corrupted["surfaces"] if row["surface_id"] == "dlive_state")
    dlive_row["status"] = "missing_fixture_evidence"
    dlive_row["covered_contracts"] = ["cached_dlive_midi_state"]
    dlive_row["fixture_paths"] = []
    dlive_row["test_names"] = []

    monkeypatch.setattr(
        release_check,
        "build_read_surface_fixture_evidence_snapshot",
        lambda: corrupted,
    )

    result = validate_read_surface_fixture_evidence_snapshot()

    assert result.status == "failed"
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_has_physical_contract" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_missing_physical_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_channel_identity_and_basic_controls_partial_status_missing_hardware_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_missing_physical_contract" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_has_physical_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_matrix_state_present_status_has_hardware_gap" in result.detail["violations"]
    assert "read_surface_fixture_row_dlive_state_missing_status_has_covered_contracts" in result.detail["violations"]


def test_no_wing_release_check_validates_read_surface_remediation_plan_alignment():
    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "passed"
    assert result.detail["plan_status"] == "read_remediation_required"
    assert result.detail["physical_fixture_missing_surface_ids"]
    assert (
        result.detail["next_physical_readback_surface_id"]
        == result.detail["expected_next_physical_readback_surface_id"]
    )
    assert (
        result.detail["next_physical_readback_action_id"]
        == result.detail["expected_next_physical_readback_action_id"]
    )
    assert result.detail["next_physical_readback_target_action_ids"] == [
        result.detail["expected_next_physical_readback_action_id"]
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_read_surface_remediation_target_drift(monkeypatch):
    fixture_snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    plan = release_check.build_read_surface_remediation_plan(
        fixture_evidence_snapshot=fixture_snapshot
    )
    corrupted = json.loads(json.dumps(plan))
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["next_physical_readback_action_id"] = "remediate_dlive_state"
    next_action = next(
        action
        for action in corrupted["actions"]
        if action["id"] == "remediate_raw_mixer_state_cache"
    )
    next_action["next_physical_readback_target"] = False
    next_action["missing_contracts"] = []

    monkeypatch.setattr(
        release_check,
        "build_read_surface_remediation_plan",
        lambda **_: corrupted,
    )

    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "failed"
    assert "read_surface_remediation_next_target_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_id_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_surface_mismatch" in result.detail["violations"]
    assert "read_surface_remediation_next_action_target_flag_missing" in result.detail["violations"]


def test_no_wing_release_check_rejects_multiple_remediation_next_target_flags(monkeypatch):
    fixture_snapshot = release_check.build_read_surface_fixture_evidence_snapshot()
    plan = release_check.build_read_surface_remediation_plan(
        fixture_evidence_snapshot=fixture_snapshot
    )
    corrupted = json.loads(json.dumps(plan))
    expected_next_action_id = corrupted["summary"]["next_physical_readback_action_id"]
    extra_action = next(
        action
        for action in corrupted["actions"]
        if action["id"] != expected_next_action_id
    )
    extra_action["next_physical_readback_target"] = True

    monkeypatch.setattr(
        release_check,
        "build_read_surface_remediation_plan",
        lambda **_: corrupted,
    )

    result = validate_read_surface_remediation_plan_snapshot()

    assert result.status == "failed"
    assert (
        "read_surface_remediation_next_target_action_flags_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_read_surface_capture_plan_alignment():
    result = validate_read_surface_physical_readback_capture_plan_snapshot()

    assert result.status == "passed"
    assert result.detail["plan_status"] == "operator_capture_required"
    assert result.detail["capture_step_count"] == 1
    assert result.detail["physical_fixture_missing_surface_ids"]
    assert (
        result.detail["next_physical_readback_surface_id"]
        == result.detail["expected_next_physical_readback_surface_id"]
    )
    assert (
        result.detail["next_physical_readback_action_id"]
        == result.detail["expected_next_physical_readback_action_id"]
    )
    assert result.detail["target_action_consistency"] is True
    assert result.detail["step_surface_id"] == result.detail["next_physical_readback_surface_id"]
    assert result.detail["step_remediation_action_id"] == result.detail["next_physical_readback_action_id"]
    assert "physical_readback_fixture" in result.detail["required_artifact_contracts"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_capture_plan_execution_authority(monkeypatch):
    capture_plan = release_check.build_read_surface_physical_readback_capture_plan()
    corrupted = json.loads(json.dumps(capture_plan))
    corrupted["safety"]["execution_allowed_by_plan"] = True
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["live_authority_granted"] = True
    corrupted["safety"]["operator_required"] = False
    corrupted["summary"]["target_action_consistency"] = False
    corrupted["steps"][0]["execution_allowed_by_plan"] = True
    corrupted["steps"][0]["live_write"] = True
    corrupted["steps"][0]["operator_required"] = False

    monkeypatch.setattr(
        release_check,
        "build_read_surface_physical_readback_capture_plan",
        lambda **_: corrupted,
    )

    result = validate_read_surface_physical_readback_capture_plan_snapshot()

    assert result.status == "failed"
    assert "read_surface_capture_plan_execution_allowed_by_plan_not_false" in result.detail["violations"]
    assert "read_surface_capture_plan_live_write_not_false" in result.detail["violations"]
    assert "read_surface_capture_plan_live_authority_granted_not_false" in result.detail["violations"]
    assert "read_surface_capture_plan_operator_required_not_true" in result.detail["violations"]
    assert "read_surface_capture_plan_target_action_consistency_mismatch" in result.detail["violations"]
    assert "read_surface_capture_plan_step_operator_required_not_true" in result.detail["violations"]
    assert "read_surface_capture_plan_step_execution_allowed_by_plan_not_false" in result.detail["violations"]
    assert "read_surface_capture_plan_step_live_write_not_false" in result.detail["violations"]


def test_no_wing_release_check_validates_wing_readonly_preflight_evidence_snapshot():
    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["evidence_status"] == "operator_preflight_required"
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["accepted_as_readonly_preflight_candidate"] is False
    assert result.detail["readiness_matrix_update_allowed"] is False
    assert result.detail["readiness_closure_allowed_by_evidence"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["mvp_blocker_candidate_count"] == 0
    assert result.detail["mvp_blockers_closed_by_evidence"] == []
    assert result.detail["artifact_operator_next_steps"] == []
    assert result.detail["mvp_blocker_matrix_ids"] == [
        "wing_readonly_connection_proof",
        "physical_console_readback_proof",
        "operator_visible_evidence_report",
        "readiness_matrix_update_based_on_real_proof",
    ]
    assert result.detail["capture_tool_path"] == "tools/wing_readonly_preflight_runner.py"
    assert result.detail["capture_tool_available"] is True
    assert result.detail["plan_command"] == [
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
    assert result.detail["plan_command_uses_plan_only"] is True
    expected_target_defaults = {
        "url": WING_READONLY_PREFLIGHT_RUNNER_URL,
        "channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "value_types": WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "artifact_root": WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    }
    assert result.detail["expected_target_defaults"] == expected_target_defaults
    assert result.detail["plan_command_target_defaults"] == expected_target_defaults
    assert result.detail["plan_command_target_defaults_match_contract"] is True
    assert result.detail["plan_command_output"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert result.detail["plan_command_channels"] == "3"
    assert result.detail["plan_command_operator_confirmed_channels"] == "3"
    assert result.detail["plan_command_confirms_session_ready"] is True
    assert result.detail["plan_command_value_types"] == "fader,gain,pan"
    assert result.detail["plan_command_artifact_root"] == (
        "artifacts/wing_readonly_preflight"
    )
    assert result.detail["plan_command_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert result.detail["plan_command_matches_capture"] is True
    assert result.detail["capture_command_template"] == [
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
    assert result.detail["plan_command_output_matches_contract"] is True
    assert result.detail["capture_command_channels"] == "3"
    assert result.detail["capture_command_operator_confirmed_channels"] == "3"
    assert result.detail["capture_command_requires_operator_channel_confirmation"] is True
    assert result.detail["capture_command_operator_confirmation_matches_channels"] is True
    assert result.detail["capture_command_confirms_session_ready"] is True
    assert result.detail["capture_command_requires_physical_readback"] is True
    assert result.detail["capture_command_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert result.detail["capture_command_url"] == WING_READONLY_PREFLIGHT_RUNNER_URL
    assert result.detail["capture_command_target_defaults"] == expected_target_defaults
    assert result.detail["capture_command_target_defaults_match_contract"] is True
    assert result.detail["capture_command_prestart_summary_matches_plan"] is True
    assert result.detail["capture_command_expected_plan"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert result.detail["capture_command_expected_plan_matches_contract"] is True
    assert result.detail["capture_command_expected_plan_matches_plan_output"] is True
    assert result.detail["review_tool_path"] == (
        "tools/review_wing_readonly_preflight_artifacts.py"
    )
    assert result.detail["review_tool_available"] is True
    assert result.detail["review_command"] == [
        "python3",
        "tools/review_wing_readonly_preflight_artifacts.py",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--output",
        WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
        "--decision-template-output",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    ]
    assert result.detail["review_command_writes_decision_template"] is True
    assert result.detail["review_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert result.detail["review_decision_template_output_matches_contract"] is True
    assert result.detail["review_exit_codes"] == {
        "candidate_ready": 0,
        "artifact_missing": 2,
        "invalid_artifact": 3,
    }
    assert result.detail["review_manual_checklist_ids"] == [
        "confirm_artifact_paths",
        "confirm_operator_channel_scope",
        "confirm_operator_session_ready",
        "confirm_physical_readbacks",
        "confirm_event_log_readonly_sequence",
        "confirm_no_live_authority",
    ]
    assert result.detail["review_manual_checklist_complete"] is True
    assert result.detail["review_decision_template_required_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert result.detail["review_decision_template_contract_complete"] is True
    assert result.detail["review_decision_template_status"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert result.detail["review_decision_template_status_matches_contract"] is True
    assert result.detail["handoff_packet_tool_path"] == (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
    )
    assert result.detail["handoff_packet_tool_available"] is True
    assert result.detail["handoff_packet_command"] == [
        "python3",
        "tools/build_wing_readonly_preflight_handoff_packet.py",
        "--url",
        "ws://localhost:8765",
        "--channels",
        "3",
        "--value-types",
        "fader,gain,pan",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--plan-output",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
        "--prestart-summary",
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        "--output",
        WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    ]
    assert result.detail["handoff_packet_output"] == (
        WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
    )
    assert result.detail["handoff_packet_output_matches_contract"] is True
    assert result.detail["handoff_packet_plan_output"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert result.detail["handoff_packet_plan_output_matches_contract"] is True
    assert result.detail["handoff_packet_plan_output_matches_plan_command"] is True
    assert result.detail["handoff_packet_url"] == "ws://localhost:8765"
    assert result.detail["handoff_packet_target_defaults"] == expected_target_defaults
    assert result.detail["handoff_packet_target_defaults_match_contract"] is True
    assert result.detail["handoff_packet_artifact_root"] == (
        "artifacts/wing_readonly_preflight"
    )
    assert result.detail["handoff_packet_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert result.detail["handoff_packet_channels"] == "3"
    assert result.detail["handoff_packet_value_types"] == "fader,gain,pan"
    assert result.detail["handoff_packet_url_matches_capture"] is True
    assert result.detail["handoff_packet_artifact_root_matches_capture"] is True
    assert result.detail["handoff_packet_prestart_summary_matches_capture"] is True
    assert result.detail["handoff_packet_channel_matches_capture"] is True
    assert result.detail["handoff_packet_value_types_match_capture"] is True
    assert result.detail["blocked_by"] == [
        "operator_must_run_wing_readonly_preflight_runner"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_surfaces_wing_preflight_artifact_operator_next_steps(
    monkeypatch,
):
    evidence = release_check.build_wing_readonly_preflight_evidence()
    evidence["artifacts"] = [
        {
            "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
            "status": "invalid_readonly_preflight_artifact",
            "accepted_as_readonly_preflight_candidate": False,
            "operator_next_step": (
                "Regenerate the offline expected plan before opening the "
                "backend/WING WebSocket."
            ),
            "violations": ["expected_plan_mismatch"],
        },
        {
            "source_id": "artifacts/wing_readonly_preflight/run/retry-summary.json",
            "status": "invalid_readonly_preflight_artifact",
            "accepted_as_readonly_preflight_candidate": False,
            "operator_next_step": (
                "Regenerate the offline expected plan before opening the "
                "backend/WING WebSocket."
            ),
            "violations": ["expected_plan_mismatch"],
        },
    ]
    evidence["summary"]["artifact_count"] = 2
    evidence["summary"]["invalid_candidate_count"] = 2
    evidence["summary"]["blocked_by"] = [
        "operator_must_replace_invalid_readonly_preflight_artifact"
    ]
    monkeypatch.setattr(
        release_check,
        "build_wing_readonly_preflight_evidence",
        lambda: evidence,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["artifact_operator_next_steps"] == [
        "Regenerate the offline expected plan before opening the "
        "backend/WING WebSocket."
    ]
    assert "wing_readonly_preflight_invalid_artifacts_present" in result.detail[
        "violations"
    ]


def valid_wing_readonly_preflight_plan_payload(plan_path: Path) -> dict:
    return {
        "type": "wing_readonly_preflight_offline_plan",
        "status": "ready_for_operator_review",
        "proof_artifact_schema": WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA,
        "proof_runner": WING_READONLY_PREFLIGHT_PROOF_RUNNER,
        "selected_mvp_blocker": WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
        "selected_mvp_blocker_status": (
            WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
        ),
        "url": "ws://localhost:8765",
        "artifact_root": "artifacts/wing_readonly_preflight",
        "channels": [3],
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
        "operator_session_confirmation_required": True,
        "operator_confirmed_session_ready": True,
        "value_types": ["fader", "gain", "pan"],
        "require_physical_readback": True,
        "read_only": True,
        "live_write": False,
        "operator_required": True,
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": [
            "PYTHONPATH=backend",
            "python3",
            "tools/no_wing_release_check.py",
            "--quick",
        ],
        "backend_port_prestart_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "backend_port_prestart_summary_required": True,
        "backend_port_prestart_summary_path": (
            WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
        ),
        "backend_port_prestart_summary_sha256": "a" * 64,
        "backend_port_prestart_summary_type": (
            WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
        ),
        "backend_port_prestart_summary_status": (
            WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS
        ),
        "backend_port_prestart_summary_connection_policy": (
            WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
        ),
        "backend_port_prestart_summary_live_write": False,
        "backend_port_prestart_summary_contacts_backend": False,
        "backend_port_prestart_summary_queries_console": False,
        "backend_port_prestart_summary_executes_tests": False,
        "backend_port_prestart_check_status": (
            WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS
        ),
        "manual_review_required_before_readiness_update": True,
        "request_id_policy": WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY,
        "contacts_backend": False,
        "queries_console": False,
        "execution_allowed_by_plan": False,
        "live_authority_granted": False,
        "allowed_request_types": list(WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES),
        "expected_response_types_by_request_type": {
            request_type: list(response_types)
            for request_type, response_types in (
                WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
            )
        },
        "forbidden_request_types": [
            "apply_operator_proposal",
            "connect_mixer",
            "disconnect_mixer",
            "emergency_stop",
            "rollback_last_manual_write",
            "set_compressor",
            "set_eq",
            "set_fader",
            "set_gain",
            "set_gate",
            "set_operator_mode",
        ],
        "request_sequence": [
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
        ],
        "expected_artifacts": [
            (
                "artifacts/wing_readonly_preflight/<run_id>/"
                f"{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
            ),
            (
                "artifacts/wing_readonly_preflight/<run_id>/"
                f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
            ),
        ],
        "post_run_review_commands": [
            " ".join(release_check.wing_readonly_preflight_review_command()),
            " ".join(release_check.wing_readonly_preflight_review_decision_command()),
        ],
        "fail_closed_conditions": list(
            release_check.WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS
        ),
        "next_step": (
            "Run without --plan-only only when the intended backend/WING "
            "session is ready."
        ),
        "plan_output": str(plan_path),
    }


def valid_wing_readonly_preflight_handoff_packet_payload(plan_path: Path) -> dict:
    return {
        "type": "wing_readonly_preflight_handoff_packet",
        "generated_at": "2026-06-07T00:00:00Z",
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
        "backend_port_prestart_summary_path_placeholder": (
            WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
        ),
        "backend_port_prestart_check_operator_sequence_step_index": 0,
        "backend_port_prestart_check_must_be_first_operator_step": True,
        "backend_port_prestart_check_command": [
            "PYTHONPATH=backend",
            "python3",
            "tools/no_wing_release_check.py",
            "--quick",
        ],
        "backend_port_prestart_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "manual_review_required_before_readiness_update": True,
        "execution_allowed_by_packet": False,
        "readiness_matrix_update_allowed_by_packet": False,
        "readiness_closure_allowed_by_packet": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "url": "ws://localhost:8765",
        "channels": [3],
        "value_types": ["fader", "gain", "pan"],
        "artifact_root": "artifacts/wing_readonly_preflight",
        "decision_root": WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
        "expected_run_artifact_glob": (
            "artifacts/wing_readonly_preflight/<run_id>/"
            f"{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
        ),
        "expected_event_log_glob": (
            "artifacts/wing_readonly_preflight/<run_id>/"
            f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
        ),
        "plan_output": str(plan_path),
        "decision_template_output": (
            WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
        ),
        "decision_template_output_matches_contract": True,
        "review_output": WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
        "review_output_matches_contract": True,
        "decision_review_output": WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
        "decision_review_output_matches_contract": True,
        "commands": {
            "prestart_backend_port_check": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "preview_readonly_request_plan": [
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
                str(plan_path),
            ],
            "capture_readonly_preflight": [
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
                str(plan_path),
            ],
            "review_artifact_and_write_pending_decision_template": [
                "python3",
                "tools/review_wing_readonly_preflight_artifacts.py",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--output",
                WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
                "--decision-template-output",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
            ],
            "review_edited_decision_json": [
                "python3",
                "tools/review_wing_readonly_preflight_decisions.py",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--decision-root",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
                "--output",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
            ],
            "focused_validation": [
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
            ],
        },
        "mvp_blocker_anchor": release_check.wing_mvp_first_open_blocker_anchor(),
        "mvp_blocker_anchor_sha256": release_check.wing_mvp_blocker_anchor_sha256(
            release_check.wing_mvp_first_open_blocker_anchor()
        ),
        "operator_sequence": [
            (
                "Before starting the intended backend/WING session, run "
                "prestart_backend_port_check and resolve only confirmed "
                "unexpected listeners."
            ),
            "Run preview_readonly_request_plan and confirm the request sequence is read-only.",
            "Confirm backend is connected to the intended WING Rack.",
            "Confirm the selected channel numbers are safe to inspect.",
            "Run capture_readonly_preflight.",
            "Run review_artifact_and_write_pending_decision_template.",
            "Edit the pending decision template into an accepted or rejected JSON record.",
            "Run review_edited_decision_json.",
            (
                "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
                "current first open blocker before supervised proof planning."
            ),
            (
                "Only after accepted_decision_ready and a valid accepted-candidate "
                "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
            ),
        ],
        "backend_port_prestart_gate": {
            "required": True,
            "operator_sequence_step_index": 0,
            "operator_sequence_step_must_be_first": True,
            "operator_sequence_step": (
                "Before starting the intended backend/WING session, run "
                "prestart_backend_port_check and resolve only confirmed "
                "unexpected listeners."
            ),
            "required_markers": [
                "Before starting the intended backend/WING session",
                "prestart_backend_port_check",
                "unexpected listeners",
            ],
            "runs_command": "prestart_backend_port_check",
            "must_pass_before_backend_wing_session": True,
            "grants_live_authority": False,
        },
        "expected_backend_port_prestart_gate": {
            "required": True,
            "operator_sequence_step_index": 0,
            "operator_sequence_step_must_be_first": True,
            "operator_sequence_step": (
                "Before starting the intended backend/WING session, run "
                "prestart_backend_port_check and resolve only confirmed "
                "unexpected listeners."
            ),
            "required_markers": [
                "Before starting the intended backend/WING session",
                "prestart_backend_port_check",
                "unexpected listeners",
            ],
            "runs_command": "prestart_backend_port_check",
            "must_pass_before_backend_wing_session": True,
            "grants_live_authority": False,
        },
        "supervised_prestart_gate": {
            "required": True,
            "operator_sequence_step_index": 9,
            "operator_sequence_step_must_be_final": True,
            "operator_sequence_step": (
                "Only after accepted_decision_ready and a valid accepted-candidate "
                "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
            ),
            "required_markers": [
                "accepted_decision_ready",
                "valid accepted-candidate prestart trace",
                "WING_SUPERVISED_FADER_GAIN_PROOF",
            ],
            "unlocks_runbook": "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
            "grants_live_authority": False,
            "requires_operator_decision_status": "accepted_decision_ready",
            "requires_accepted_candidate_prestart_trace": True,
        },
        "expected_supervised_prestart_gate": {
            "required": True,
            "operator_sequence_step_index": 9,
            "operator_sequence_step_must_be_final": True,
            "operator_sequence_step": (
                "Only after accepted_decision_ready and a valid accepted-candidate "
                "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
            ),
            "required_markers": [
                "accepted_decision_ready",
                "valid accepted-candidate prestart trace",
                "WING_SUPERVISED_FADER_GAIN_PROOF",
            ],
            "unlocks_runbook": "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
            "grants_live_authority": False,
            "requires_operator_decision_status": "accepted_decision_ready",
            "requires_accepted_candidate_prestart_trace": True,
        },
        "required_review_checklist_ids": [
            "confirm_artifact_paths",
            "confirm_operator_channel_scope",
            "confirm_operator_session_ready",
            "confirm_physical_readbacks",
            "confirm_event_log_readonly_sequence",
            "confirm_no_live_authority",
        ],
        "forbidden_live_request_types": list(
            release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
        ),
        "blockers_not_closed_by_packet": [
            "wing_readonly_connection_proof",
            "physical_console_readback_proof",
            "supervised_fader_write_proof",
            "supervised_gain_down_trim_proof",
            "readiness_matrix_update_based_on_real_proof",
        ],
        "next_manual_artifact_required": (
            WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
        ),
    }


def test_no_wing_release_check_fail_closed_conditions_use_shared_contract():
    assert release_check.WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS == (
        WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS
    )


def valid_read_surface_capture_packet_payload() -> dict:
    return {
        "type": "read_surface_physical_readback_capture_packet",
        "generated_at": "2026-06-07T00:00:00Z",
        "status": "operator_capture_required",
        "source_plan_status": "operator_capture_required",
        "source_method_id": "read_surface_physical_readback_capture_plan",
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_packet": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "physical_fixture_missing_surface_ids": [
            "raw_mixer_state_cache",
            "channel_identity_and_basic_controls",
        ],
        "physical_gap_surface_count": 2,
        "remaining_capture_target_count": 2,
        "remaining_capture_targets": [
            {
                "surface_id": "raw_mixer_state_cache",
                "remediation_action_id": "remediate_raw_mixer_state_cache",
                "target_console_surface": "all_cached_addresses",
                "read_mode": "cached_state",
                "requires_connection": True,
                "contract_method_id": "raw_mixer_state_physical_readback_contract",
                "required_artifact_contracts": [
                    "physical_readback_fixture",
                    "raw_mixer_state_physical_readback_contract",
                ],
                "expected_artifacts": [
                    "Read DTO for all_cached_addresses with explicit coverage flags.",
                    "Raw mixer-state physical readback artifact validated by raw_mixer_state_physical_readback_contract.",
                ],
                "blockers_to_clear": [
                    "physical_readback_fixture_missing",
                    "hardware_parity_unproven",
                ],
                "next_physical_readback_target": True,
                "execution_allowed_by_target": False,
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
            },
            {
                "surface_id": "channel_identity_and_basic_controls",
                "remediation_action_id": "remediate_channel_identity_and_basic_controls",
                "target_console_surface": "channels",
                "read_mode": "product_snapshot",
                "requires_connection": True,
                "contract_method_id": "channel_basic_controls_physical_readback_contract",
                "required_artifact_contracts": [
                    "physical_readback_fixture",
                    "channel_basic_controls_physical_readback_contract",
                ],
                "expected_artifacts": [
                    "Read DTO for channels with explicit coverage flags.",
                    "Channel identity/basic-control physical readback artifact validated by channel_basic_controls_physical_readback_contract.",
                ],
                "blockers_to_clear": [
                    "physical_readback_fixture_missing",
                    "hardware_parity_unproven",
                ],
                "next_physical_readback_target": False,
                "execution_allowed_by_target": False,
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
            },
        ],
        "next_physical_readback_surface_id": "raw_mixer_state_cache",
        "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
        "target_action_consistency": True,
        "capture_step_count": 1,
        "required_artifact_contracts": [
            "physical_readback_fixture",
            "raw_mixer_state_physical_readback_contract",
        ],
        "blocked_by": [
            "physical_readback_fixtures_incomplete",
            "operator_must_capture_and_review_physical_readback_artifact",
        ],
        "step": {
            "surface_id": "raw_mixer_state_cache",
            "remediation_action_id": "remediate_raw_mixer_state_cache",
            "target_console_surface": "all_cached_addresses",
            "read_mode": "cached_state",
            "contract_method_id": "raw_mixer_state_physical_readback_contract",
            "expected_artifacts": [
                "Read DTO for all_cached_addresses with explicit coverage flags.",
                "Raw mixer-state physical readback artifact validated by raw_mixer_state_physical_readback_contract.",
            ],
            "blockers_to_clear": [
                "physical_readback_fixture_missing",
                "hardware_parity_unproven",
            ],
            "forbidden_actions": [
                "do not apply fader, gain, trim, pan, mute, EQ, dynamics, gate, routing, DCA, matrix, FX, scene, or preset changes",
                "do not transmit OSC, MIDI, TCP, or websocket mixer-control messages",
                "do not change safety rows, operator mode, promotion readiness, or live-write policy from this plan",
            ],
        },
        "operator_sequence": [
            "Confirm the intended console, show state, and read-only capture scope.",
            "Use operator-approved read-only tooling outside this packet to capture the physical readback artifact.",
            "Reject the artifact if the capture path would transmit a mixer-control command.",
            "Validate the artifact against the required physical readback contract.",
            "Review the artifact manually before changing fixture baselines or autonomy readiness.",
        ],
        "commands": {
            "focused_validation": [
                "PYTHONPATH=backend",
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_read_surface_physical_readback_capture_plan.py",
                "tests/test_no_wing_release_check.py",
            ],
        },
    }


def valid_read_surface_capture_template_payload() -> dict:
    return {
        "type": "read_surface_physical_readback_artifact_capture_template",
        "generated_at": "2026-06-07T00:00:00Z",
        "status": "operator_capture_template_ready",
        "source_packet_type": "read_surface_physical_readback_capture_packet",
        "source_packet_status": "operator_capture_required",
        "source_method_id": "read_surface_physical_readback_capture_plan",
        "selected_surface_id": "raw_mixer_state_cache",
        "include_all_targets": False,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_template": False,
        "accepted_as_physical_fixture_candidate": False,
        "fixture_closure_allowed_by_template": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "all_remaining_target_count": 2,
        "capture_template_count": 1,
        "capture_templates": [
            {
                "template_id": "pending-raw-mixer-state-cache-physical-readback",
                "surface_id": "raw_mixer_state_cache",
                "remediation_action_id": "remediate_raw_mixer_state_cache",
                "target_console_surface": "all_cached_addresses",
                "read_mode": "cached_state",
                "requires_connection": True,
                "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
                "required_artifact_contracts": [
                    "physical_readback_fixture",
                    "raw_mixer_state_physical_readback_contract",
                ],
                "expected_artifacts": [
                    "Raw mixer-state physical readback artifact validated by raw_mixer_state_physical_readback_contract.",
                ],
                "blockers_to_clear": [
                    "physical_readback_fixture_missing",
                    "hardware_parity_unproven",
                ],
                "artifact_source_id": "<path-to-operator-captured-json>",
                "captured_at": "<UTC ISO timestamp, e.g. 2026-06-07T12:00:00Z>",
                "captured_by": "<operator-name>",
                "source": "<physical_readback|wing_physical_readback|dlive_physical_readback>",
                "console_model": "<console model and firmware>",
                "capture_scope": "<manual capture scope and show-state notes>",
                "physical_readback_payload": (
                    "<replace with JSON payload that matches required_contract_method_id>"
                ),
                "physical_readback_payload_template": {
                    "template_status": "operator_must_fill_from_physical_readback",
                    "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
                    "artifact_type": "raw_mixer_state_physical_readback",
                    "schema_id": "raw_mixer_state_physical_readback.v1",
                    "required_top_level_fields": [
                        "type",
                        "schema_id",
                        "surface_id",
                        "remediation_action_id",
                        "captured_at",
                        "source",
                        "console_model",
                        "address_count",
                        "readback_rows",
                        "safety",
                    ],
                    "required_field_sets": {
                        "required_row_fields": [
                            "address",
                            "value",
                            "status",
                            "source",
                            "captured_at",
                        ],
                        "required_false_safety_flags": [
                            "live_write",
                            "mutates_console_state",
                            "live_widening_allowed",
                            "full_autonomous_ready",
                        ],
                    },
                    "payload": {
                        "type": "raw_mixer_state_physical_readback",
                        "schema_id": "raw_mixer_state_physical_readback.v1",
                        "surface_id": "raw_mixer_state_cache",
                        "remediation_action_id": "remediate_raw_mixer_state_cache",
                        "captured_at": (
                            "<UTC ISO timestamp, e.g. 2026-06-07T12:00:00Z>"
                        ),
                        "source": (
                            "<physical_readback|wing_physical_readback|dlive_physical_readback>"
                        ),
                        "console_model": "<console model and firmware>",
                        "address_count": "<integer address_count>",
                        "readback_rows": [
                            {
                                "address": "<operator-captured address>",
                                "value": "<operator-captured value>",
                                "status": "<verified_from_physical_readback>",
                                "source": (
                                    "<physical_readback|wing_physical_readback|dlive_physical_readback>"
                                ),
                                "captured_at": (
                                    "<UTC ISO timestamp, e.g. 2026-06-07T12:00:00Z>"
                                ),
                            }
                        ],
                        "safety": {
                            "read_only": True,
                            "live_write": False,
                            "mutates_console_state": False,
                            "live_widening_allowed": False,
                            "full_autonomous_ready": False,
                            "live_authority_granted": False,
                            "accepted_as_physical_fixture_candidate": False,
                        },
                    },
                },
                "safety": {
                    "read_only": True,
                    "live_write": False,
                    "mutates_console_state": False,
                    "queries_console": False,
                    "executes_tests": False,
                    "operator_required": True,
                    "execution_allowed_by_template": False,
                    "accepted_as_physical_fixture_candidate": False,
                    "fixture_closure_allowed_by_template": False,
                    "live_widening_allowed": False,
                    "live_authority_granted": False,
                    "full_autonomous_ready": False,
                },
            }
        ],
        "required_operator_edits": [
            "artifact_source_id",
            "captured_at",
            "captured_by",
            "source",
            "console_model",
            "capture_scope",
            "physical_readback_payload",
            "physical_readback_payload_template.payload",
        ],
        "blocked_by": [
            "operator_must_capture_and_review_physical_readback_artifact",
            "operator_must_replace_capture_template_placeholders",
            "operator_must_validate_filled_artifact_with_required_contract",
            "operator_must_review_artifact_before_fixture_acceptance",
        ],
        "operator_sequence": [
            "Confirm the selected surface and console/show state before capture.",
            "Replace every placeholder with real operator-captured physical readback data.",
            "Reject the capture path if it would send a mixer-control message or mutate console state.",
            "Keep authority, execution, fixture-closure, and autonomous-readiness flags false.",
            "Validate the filled artifact with the listed required contract before review queue intake.",
            "Use the review queue and acceptance ledger before any separate fixture-closure decision.",
        ],
        "commands": {
            "focused_validation": [
                "PYTHONPATH=backend",
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_read_surface_physical_readback_capture_template.py",
                "tests/test_no_wing_release_check.py",
            ],
        },
    }


def valid_read_surface_artifact_review_queue_payload() -> dict:
    queue_items = []
    for spec in release_check.INTAKE_SPECS:
        queue_items.append(
            {
                "method_id": spec["method_id"],
                "available": True,
                "schema_id": "physical_readback.v1",
                "intake_status": "operator_artifact_missing",
                "surface_id": spec["surface_id"],
                "remediation_action_id": spec["remediation_action_id"],
                "required_contract_method_id": spec["required_contract_method_id"],
                "status": "operator_artifact_missing",
                "artifact_count": 0,
                "valid_candidate_count": 0,
                "invalid_candidate_count": 0,
                "accepted_as_physical_fixture_candidate": False,
                "source_accepted_as_physical_fixture_candidate_claimed": False,
                "operator_review_required": True,
                "source_operator_review_required_claimed": True,
                "physical_fixture_gap_remains": True,
                "source_physical_fixture_gap_remains_claimed": True,
                "fixture_closure_allowed_by_queue": False,
                "fixture_closure_allowed_by_intake": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "target_consistency": True,
                "candidate_source_id_missing": False,
                "unsafe_authority_flags": [],
                "artifact_source_ids": [],
                "blocked_by": [
                    "operator_must_capture_physical_readback_artifact",
                ],
                "violations": [],
            }
        )
    return {
        "type": "read_surface_physical_readback_artifact_review_queue",
        "generated_at": "2026-06-07T00:00:00Z",
        "method_id": "read_surface_physical_readback_artifact_review_queue",
        "status": "operator_artifact_capture_required",
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "intake_surface_count": len(queue_items),
            "available_intake_count": len(queue_items),
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "review_candidate_count": 0,
            "missing_artifact_surface_count": len(queue_items),
            "unsafe_authority_claim_count": 0,
            "candidate_source_id_missing_count": 0,
            "target_inconsistent_surface_count": 0,
            "review_candidate_surface_ids": [],
            "invalid_surface_ids": [],
            "missing_artifact_surface_ids": [
                spec["surface_id"] for spec in release_check.INTAKE_SPECS
            ],
            "unsafe_authority_surface_ids": [],
            "candidate_source_id_missing_surface_ids": [],
            "target_inconsistent_surface_ids": [],
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [],
            "blocked_by": ["operator_must_capture_missing_physical_readback_artifacts"],
            "next_global_step": "Operator captures missing physical readback artifacts.",
            "queue_item_method_ids": [
                spec["method_id"] for spec in release_check.INTAKE_SPECS
            ],
        },
        "queue_items": queue_items,
        "operator_sequence": [
            "Confirm every listed candidate came from a physical readback artifact intake.",
            "Reject the queue if any candidate reports live, mutation, query, or closure authority.",
            "Review valid candidates manually before preparing acceptance records.",
            "Capture or replace any missing or invalid physical readback artifacts.",
            "Run the focused validation command before using the acceptance template.",
        ],
        "commands": {
            "focused_validation": [
                "PYTHONPATH=backend",
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_read_surface_physical_readback_artifact_review_queue.py",
                "tests/test_no_wing_release_check.py",
            ],
        },
    }


def valid_read_surface_acceptance_template_payload() -> dict:
    return {
        "type": "read_surface_physical_readback_fixture_acceptance_template",
        "generated_at": "2026-06-07T00:00:00Z",
        "status": "operator_candidate_missing",
        "source_queue_status": "operator_artifact_capture_required",
        "source_queue_method_id": "read_surface_physical_readback_artifact_review_queue",
        "selected_surface_id": None,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_template": False,
        "fixture_closure_allowed_by_template": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "review_candidate_count": 0,
        "template_record_count": 0,
        "template_records": [],
        "required_operator_edits": [
            "decision",
            "reviewed_by",
            "reviewed_at",
            "review_notes",
        ],
        "blocked_by": ["operator_must_capture_missing_physical_readback_artifacts"],
        "operator_sequence": [
            "Confirm the artifact is a valid review candidate from the review queue.",
            "Confirm candidate_trace still matches the review queue candidate.",
            "Set decision to accepted or rejected after manual review.",
            "Set reviewed_by and reviewed_at using a UTC ISO timestamp.",
            "Keep all authority and fixture-closure flags false.",
            "Run the fixture acceptance ledger before any separate closure decision.",
        ],
        "commands": {
            "focused_validation": [
                "PYTHONPATH=backend",
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_read_surface_physical_readback_fixture_acceptance_template.py",
                "tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py",
            ],
        },
    }


def valid_read_surface_fixture_closure_plan_artifact_payload() -> dict:
    return {
        "type": "read_surface_physical_readback_fixture_closure_plan",
        "generated_at": "2026-06-07T00:00:00Z",
        "method_id": "read_surface_physical_readback_fixture_closure_plan",
        "source_method_id": "read_surface_physical_readback_fixture_acceptance_ledger",
        "status": "fixture_closure_not_ready",
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "ledger_status": "operator_acceptance_required",
            "review_candidate_count": 0,
            "acceptance_record_count": 0,
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "closure_candidate_count": 0,
            "closure_candidate_surface_ids": [],
            "rejected_surface_ids": [],
            "invalid_record_ids": [],
            "unreviewed_candidate_surface_ids": [],
            "manual_fixture_closure_candidate_available": False,
            "manual_fixture_closure_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "accepted_physical_readback_candidate_required_before_fixture_closure"
            ],
            "next_global_step": "Operator needs accepted physical readback candidates first.",
        },
        "closure_candidates": [],
        "operator_sequence": [
            "Confirm every physical readback candidate was reviewed by an operator.",
            "Reject this plan if any acceptance record is invalid, ambiguous, or unreviewed.",
            "Keep fixture_update_allowed_by_plan and fixture_closure_allowed_by_plan false.",
            "Perform any fixture baseline update as a separate manual repository change.",
            "Run the focused validation command before considering a separate closure decision.",
        ],
        "commands": {
            "focused_validation": [
                "PYTHONPATH=backend",
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_read_surface_physical_readback_fixture_closure_plan.py",
                "tests/test_no_wing_release_check.py",
            ],
        },
    }


def test_no_wing_release_check_validates_read_surface_capture_packet_artifact(
    tmp_path,
):
    packet_path = tmp_path / "read_surface_physical_readback_capture_packet.json"
    packet_path.write_text(
        json.dumps(valid_read_surface_capture_packet_payload()),
        encoding="utf-8",
    )

    result = validate_read_surface_physical_readback_capture_packet_artifact_snapshot(
        packet_path
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "read_surface_physical_readback_capture_packet_artifact_is_readonly_operator_gate"
    )
    assert result.detail["packet_path"] == str(packet_path)
    assert result.detail["packet_status"] == "operator_capture_required"
    assert result.detail["source_plan_status"] == "operator_capture_required"
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_packet"] is False
    assert result.detail["live_widening_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["physical_gap_surface_count"] == 2
    assert result.detail["remaining_capture_target_count"] == 2
    assert result.detail["remaining_capture_target_surface_ids"] == [
        "raw_mixer_state_cache",
        "channel_identity_and_basic_controls",
    ]
    assert result.detail["remaining_capture_target_action_ids"] == [
        "remediate_raw_mixer_state_cache",
        "remediate_channel_identity_and_basic_controls",
    ]
    assert result.detail["remaining_capture_target_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract",
        "channel_basic_controls_physical_readback_contract",
    ]
    assert result.detail["remaining_capture_target_forbidden_flags"] == []
    assert result.detail["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert (
        result.detail["next_physical_readback_action_id"]
        == "remediate_raw_mixer_state_cache"
    )
    assert result.detail["target_action_consistency"] is True
    assert result.detail["capture_step_count"] == 1
    assert "physical_readback_fixture" in result.detail["required_artifact_contracts"]
    assert result.detail["step_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["step_target_console_surface"] == "all_cached_addresses"
    assert result.detail["step_read_mode"] == "cached_state"
    assert (
        result.detail["step_contract_method_id"]
        == "raw_mixer_state_physical_readback_contract"
    )
    assert result.detail["step_expected_artifact_count"] == 2
    assert result.detail["step_blocker_count"] == 2
    assert result.detail["operator_sequence_count"] == 5
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_read_surface_capture_packet_artifact(
    tmp_path,
):
    result = validate_read_surface_physical_readback_capture_packet_artifact_snapshot(
        tmp_path / "missing-capture-packet.json"
    )

    assert result.status == "failed"
    assert (
        "read_surface_physical_readback_capture_packet_artifact_missing_report"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_read_surface_capture_packet_artifact(
    tmp_path,
):
    packet_path = tmp_path / "unsafe-capture-packet.json"
    payload = valid_read_surface_capture_packet_payload()
    payload["queries_console"] = True
    payload["execution_allowed_by_packet"] = True
    payload["live_authority_granted"] = True
    payload["physical_gap_surface_count"] = 3
    payload["remaining_capture_targets"][0]["queries_console"] = True
    payload["step"]["forbidden_actions"] = [
        "do not apply fader, gain, trim, pan, mute, EQ, dynamics, gate, routing, DCA, matrix, FX, scene, or preset changes",
    ]
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_read_surface_physical_readback_capture_packet_artifact_snapshot(
        packet_path
    )

    assert result.status == "failed"
    assert "read_surface_capture_packet_artifact_queries_console_not_false" in result.detail[
        "violations"
    ]
    assert (
        "read_surface_capture_packet_artifact_execution_allowed_by_packet_not_false"
        in result.detail["violations"]
    )
    assert (
        "read_surface_capture_packet_artifact_live_authority_granted_not_false"
        in result.detail["violations"]
    )
    assert "read_surface_capture_packet_artifact_gap_count_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "read_surface_capture_packet_artifact_target_0_queries_console_not_false"
        in result.detail["violations"]
    )
    assert (
        "read_surface_capture_packet_artifact_forbidden_action_missing:do not transmit osc"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_read_surface_capture_template_artifact(
    tmp_path,
):
    template_path = tmp_path / "read_surface_physical_readback_capture_template.json"
    template_path.write_text(
        json.dumps(valid_read_surface_capture_template_payload()),
        encoding="utf-8",
    )

    result = validate_read_surface_physical_readback_capture_template_artifact_snapshot(
        template_path
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "read_surface_physical_readback_capture_template_artifact_is_pending_readonly_operator_gate"
    )
    assert result.detail["template_path"] == str(template_path)
    assert result.detail["template_status"] == "operator_capture_template_ready"
    assert result.detail["source_packet_status"] == "operator_capture_required"
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_template"] is False
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["fixture_closure_allowed_by_template"] is False
    assert result.detail["live_widening_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["all_remaining_target_count"] == 2
    assert result.detail["capture_template_count"] == 1
    assert result.detail["capture_template_surface_ids"] == ["raw_mixer_state_cache"]
    assert result.detail["capture_template_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract"
    ]
    assert result.detail["payload_template_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract"
    ]
    assert result.detail["payload_template_artifact_types"] == [
        "raw_mixer_state_physical_readback"
    ]
    assert result.detail["payload_template_schema_ids"] == [
        "raw_mixer_state_physical_readback.v1"
    ]
    assert result.detail["payload_template_field_counts"] == [10]
    assert result.detail["required_operator_edits"] == [
        "artifact_source_id",
        "captured_at",
        "captured_by",
        "source",
        "console_model",
        "capture_scope",
        "physical_readback_payload",
        "physical_readback_payload_template.payload",
    ]
    assert result.detail["operator_sequence_count"] == 6
    assert result.detail["record_unsafe_flags"] == []
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_read_surface_capture_template_artifact(
    tmp_path,
):
    result = validate_read_surface_physical_readback_capture_template_artifact_snapshot(
        tmp_path / "missing-capture-template.json"
    )

    assert result.status == "failed"
    assert (
        "read_surface_physical_readback_capture_template_artifact_missing_report"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_read_surface_capture_template_artifact(
    tmp_path,
):
    template_path = tmp_path / "unsafe-capture-template.json"
    payload = valid_read_surface_capture_template_payload()
    payload["live_authority_granted"] = True
    payload["accepted_as_physical_fixture_candidate"] = True
    payload["capture_template_count"] = 2
    payload["capture_templates"][0]["artifact_source_id"] = "raw-mixer-physical.json"
    payload["capture_templates"][0]["captured_at"] = "2026-06-07T12:00:00Z"
    payload["capture_templates"][0]["physical_readback_payload"] = {
        "type": "raw_mixer_state_physical_readback",
    }
    payload["capture_templates"][0]["physical_readback_payload_template"]["payload"][
        "safety"
    ]["live_authority_granted"] = True
    payload["capture_templates"][0]["physical_readback_payload_template"]["payload"][
        "safety"
    ]["accepted_as_physical_fixture_candidate"] = True
    payload["capture_templates"][0]["safety"]["live_authority_granted"] = True
    payload["capture_templates"][0]["safety"][
        "accepted_as_physical_fixture_candidate"
    ] = True
    template_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_read_surface_physical_readback_capture_template_artifact_snapshot(
        template_path
    )

    assert result.status == "failed"
    assert "capture_template_artifact_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert (
        "capture_template_artifact_accepted_as_physical_fixture_candidate_not_false"
        in result.detail["violations"]
    )
    assert "capture_template_artifact_record_count_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "capture_template_artifact_record_0_artifact_source_id_not_pending"
        in result.detail["violations"]
    )
    assert (
        "capture_template_artifact_record_0_captured_at_not_pending"
        in result.detail["violations"]
    )
    assert (
        "capture_template_artifact_record_0_payload_not_pending"
        in result.detail["violations"]
    )
    assert "capture_template_artifact_record_0_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert (
        "capture_template_artifact_record_0_payload_template_live_authority_granted_not_false"
        in result.detail["violations"]
    )
    assert (
        "capture_template_artifact_record_0_payload_template_accepted_as_physical_fixture_candidate_not_false"
        in result.detail["violations"]
    )
    assert (
        "capture_template_artifact_record_0_accepted_as_physical_fixture_candidate_not_false"
        in result.detail["violations"]
    )
    assert set(result.detail["record_unsafe_flags"]) == {
        "record_0_payload_template_live_authority_granted",
        "record_0_payload_template_accepted_as_physical_fixture_candidate",
        "record_0_accepted_as_physical_fixture_candidate",
        "record_0_live_authority_granted",
    }


def test_no_wing_release_check_validates_artifact_review_queue_artifact(tmp_path):
    queue_path = tmp_path / "read_surface_physical_readback_artifact_review_queue.json"
    queue_path.write_text(
        json.dumps(valid_read_surface_artifact_review_queue_payload()),
        encoding="utf-8",
    )

    result = validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot(
        queue_path
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "physical_readback_artifact_review_queue_artifact_is_read_only_operator_gate"
    )
    assert result.detail["queue_path"] == str(queue_path)
    assert result.detail["queue_status"] == "operator_artifact_capture_required"
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_queue"] is False
    assert result.detail["fixture_closure_allowed_by_queue"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["intake_surface_count"] == len(release_check.INTAKE_SPECS)
    assert result.detail["available_intake_count"] == len(release_check.INTAKE_SPECS)
    assert result.detail["artifact_count"] == 0
    assert result.detail["review_candidate_count"] == 0
    assert result.detail["missing_artifact_surface_count"] == len(release_check.INTAKE_SPECS)
    assert result.detail["unsafe_authority_claim_count"] == 0
    assert result.detail["candidate_source_id_missing_count"] == 0
    assert result.detail["target_inconsistent_surface_count"] == 0
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["operator_sequence_count"] == 5
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_artifact_review_queue_artifact(
    tmp_path,
):
    result = validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot(
        tmp_path / "missing-artifact-review-queue.json"
    )

    assert result.status == "failed"
    assert (
        "read_surface_physical_readback_artifact_review_queue_artifact_missing_report"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_artifact_review_queue_artifact(
    tmp_path,
):
    queue_path = tmp_path / "unsafe-artifact-review-queue.json"
    payload = valid_read_surface_artifact_review_queue_payload()
    payload["safety"]["fixture_closure_allowed_by_queue"] = True
    payload["safety"]["live_authority_granted"] = True
    payload["summary"]["fixture_closure_allowed_by_queue"] = True
    payload["summary"]["live_authority_granted"] = True
    payload["summary"]["unsafe_authority_claim_count"] = 1
    payload["summary"]["candidate_source_id_missing_count"] = 1
    payload["summary"]["candidate_source_id_missing_surface_ids"] = [
        payload["queue_items"][0]["surface_id"]
    ]
    payload["queue_items"][0]["unsafe_authority_flags"] = ["live_write"]
    payload["queue_items"][0]["candidate_source_id_missing"] = True
    payload["queue_items"][0]["fixture_closure_allowed_by_queue"] = True
    payload["queue_items"][0]["live_authority_granted"] = True
    queue_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot(
        queue_path
    )

    assert result.status == "failed"
    assert "artifact_review_queue_artifact_fixture_closure_allowed_by_queue_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_summary_fixture_closure_allowed_by_queue_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_summary_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_item_0_fixture_closure_allowed_by_queue_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_item_0_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_unsafe_authority_claim_present" in result.detail[
        "violations"
    ]
    assert "artifact_review_queue_artifact_candidate_source_id_missing_present" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_read_surface_acceptance_template_artifact(
    tmp_path,
):
    template_path = tmp_path / "read_surface_physical_readback_acceptance_template.json"
    template_path.write_text(
        json.dumps(valid_read_surface_acceptance_template_payload()),
        encoding="utf-8",
    )

    result = validate_read_surface_physical_readback_acceptance_template_artifact_snapshot(
        template_path
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "read_surface_physical_readback_acceptance_template_artifact_is_readonly_operator_gate"
    )
    assert result.detail["template_path"] == str(template_path)
    assert result.detail["template_status"] == "operator_candidate_missing"
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_template"] is False
    assert result.detail["fixture_closure_allowed_by_template"] is False
    assert result.detail["live_widening_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["review_candidate_count"] == 0
    assert result.detail["template_record_count"] == 0
    assert result.detail["candidate_trace_count"] == 0
    assert result.detail["required_operator_edits"] == [
        "decision",
        "reviewed_by",
        "reviewed_at",
        "review_notes",
    ]
    assert result.detail["operator_sequence_count"] == 6
    assert result.detail["record_unsafe_flags"] == []
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_read_surface_acceptance_template_artifact(
    tmp_path,
):
    result = validate_read_surface_physical_readback_acceptance_template_artifact_snapshot(
        tmp_path / "missing-acceptance-template.json"
    )

    assert result.status == "failed"
    assert (
        "read_surface_physical_readback_acceptance_template_artifact_missing_report"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_read_surface_acceptance_template_artifact(
    tmp_path,
):
    template_path = tmp_path / "unsafe-acceptance-template.json"
    payload = valid_read_surface_acceptance_template_payload()
    payload["status"] = "operator_acceptance_template_ready"
    payload["live_authority_granted"] = True
    payload["template_record_count"] = 1
    payload["template_records"] = [
        {
            "review_id": "pending-raw-mixer",
            "surface_id": "raw_mixer_state",
            "artifact_source_id": "raw-mixer.json",
            "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
            "candidate_trace": {
                "source_queue_method_id": "read_surface_physical_readback_artifact_review_queue",
                "source_queue_status": "operator_review_candidates_available",
                "candidate_status": "operator_review_candidate_available",
                "intake_status": "artifact_source_ready",
                "artifact_source_ids": ["raw-mixer.json"],
                "valid_candidate_count": 1,
                "invalid_candidate_count": 0,
                "target_consistency": True,
                "candidate_source_id_missing": False,
                "accepted_as_physical_fixture_candidate": True,
                "fixture_closure_allowed_by_queue": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "decision": "accepted",
            "reviewed_by": "operator",
            "reviewed_at": "2026-06-07T12:00:00Z",
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "execution_allowed_by_acceptance": False,
            "fixture_closure_allowed_by_acceptance": True,
            "fixture_closure_allowed": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        }
    ]
    template_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_read_surface_physical_readback_acceptance_template_artifact_snapshot(
        template_path
    )

    assert result.status == "failed"
    assert "acceptance_template_artifact_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "acceptance_template_artifact_record_0_decision_not_pending" in result.detail[
        "violations"
    ]
    assert "acceptance_template_artifact_record_0_reviewed_by_not_pending" in result.detail[
        "violations"
    ]
    assert "acceptance_template_artifact_record_0_reviewed_at_not_pending" in result.detail[
        "violations"
    ]
    assert (
        "acceptance_template_artifact_record_0_fixture_closure_allowed_by_acceptance_not_false"
        in result.detail["violations"]
    )
    assert result.detail["record_unsafe_flags"] == [
        "record_0_fixture_closure_allowed_by_acceptance"
    ]


def test_no_wing_release_check_validates_fixture_closure_plan_artifact(tmp_path):
    plan_path = tmp_path / "read_surface_physical_readback_fixture_closure_plan.json"
    plan_path.write_text(
        json.dumps(valid_read_surface_fixture_closure_plan_artifact_payload()),
        encoding="utf-8",
    )

    result = validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot(
        plan_path
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "physical_readback_fixture_closure_plan_artifact_is_manual_read_only_gate"
    )
    assert result.detail["plan_path"] == str(plan_path)
    assert result.detail["plan_status"] == "fixture_closure_not_ready"
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_plan"] is False
    assert result.detail["fixture_update_allowed_by_plan"] is False
    assert result.detail["fixture_closure_allowed_by_plan"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["closure_candidate_count"] == 0
    assert result.detail["manual_fixture_closure_required"] is True
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["operator_sequence_count"] == 5
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_fixture_closure_plan_artifact(
    tmp_path,
):
    result = validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot(
        tmp_path / "missing-fixture-closure-plan.json"
    )

    assert result.status == "failed"
    assert (
        "read_surface_physical_readback_fixture_closure_plan_artifact_missing_report"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_fixture_closure_plan_artifact(
    tmp_path,
):
    plan_path = tmp_path / "unsafe-fixture-closure-plan.json"
    payload = valid_read_surface_fixture_closure_plan_artifact_payload()
    payload["safety"]["fixture_closure_allowed_by_plan"] = True
    payload["safety"]["live_authority_granted"] = True
    payload["summary"]["fixture_closure_allowed_by_plan"] = True
    payload["summary"]["live_authority_granted"] = True
    payload["summary"]["closure_candidate_count"] = 1
    payload["closure_candidates"] = []
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot(
        plan_path
    )

    assert result.status == "failed"
    assert "fixture_closure_plan_artifact_fixture_closure_allowed_by_plan_not_false" in result.detail[
        "violations"
    ]
    assert "fixture_closure_plan_artifact_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert (
        "fixture_closure_plan_artifact_summary_fixture_closure_allowed_by_plan_not_false"
        in result.detail["violations"]
    )
    assert (
        "fixture_closure_plan_artifact_summary_live_authority_granted_not_false"
        in result.detail["violations"]
    )
    assert "fixture_closure_plan_artifact_candidate_count_mismatch" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_wing_preflight_plan_artifact(tmp_path):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )

    result = validate_wing_readonly_preflight_plan_artifact_snapshot(plan_path)

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "wing_readonly_preflight_plan_artifact_is_offline_readonly_preview"
    )
    assert result.detail["plan_path"] == str(plan_path)
    assert result.detail["plan_status"] == "ready_for_operator_review"
    assert result.detail["plan_output"] == str(plan_path)
    assert result.detail["channels"] == [3]
    assert result.detail["operator_confirmed_channels"] == [3]
    assert result.detail["value_types"] == ["fader", "gain", "pan"]
    assert result.detail["request_count"] == 5
    assert result.detail["expected_request_count"] == 5
    assert result.detail["request_sequence_matches_expected"] is True
    assert result.detail["expected_response_types_by_request_type"] == {
        request_type: list(response_types)
        for request_type, response_types in (
            WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE.items()
        )
    }
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["contacts_backend"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["backend_port_prestart_summary_contacts_backend"] is False
    assert result.detail["backend_port_prestart_summary_queries_console"] is False
    assert result.detail["backend_port_prestart_summary_executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["expected_plan_required_for_real_run"] is True
    assert (
        result.detail["physical_preflight_requires_existing_backend_connection"]
        is True
    )
    assert result.detail["manual_review_required_before_readiness_update"] is True
    assert result.detail["execution_allowed_by_plan"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["selected_mvp_blocker"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    )
    assert result.detail["selected_mvp_blocker_status"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert "set_gain" in result.detail["forbidden_request_types"]
    assert result.detail["fail_closed_conditions"] == list(
        release_check.WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS
    )
    assert result.detail["required_fail_closed_conditions"] == list(
        release_check.WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS
    )
    assert (
        "existing_backend_connection_boundary_missing"
        in result.detail["fail_closed_conditions"]
    )
    assert (
        "stale_backend_port_prestart_detail_contract"
        in result.detail["fail_closed_conditions"]
    )
    assert result.detail["post_run_review_commands"] == [
        " ".join(release_check.wing_readonly_preflight_review_command()),
        " ".join(release_check.wing_readonly_preflight_review_decision_command()),
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_wing_preflight_plan_prestart_contract_drift(
    monkeypatch,
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE",
        "stale_summary_type",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS",
        "stale_summary_status",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY",
        "stale_connection_policy",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_STATUS",
        "stale_backend_check_status",
    )

    result = validate_wing_readonly_preflight_plan_artifact_snapshot(plan_path)

    assert result.status == "failed"
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_type_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_status_not_passed"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_policy_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_check_status_not_passed"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_missing_wing_preflight_plan_artifact(tmp_path):
    result = validate_wing_readonly_preflight_plan_artifact_snapshot(
        tmp_path / "missing-plan.json"
    )

    assert result.status == "failed"
    assert "wing_readonly_preflight_plan_artifact_missing_report" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_unsafe_wing_preflight_plan_artifact(tmp_path):
    plan_path = tmp_path / "unsafe-plan.json"
    payload = valid_wing_readonly_preflight_plan_payload(plan_path)
    payload["contacts_backend"] = True
    payload["live_authority_granted"] = True
    payload["backend_port_prestart_summary_contacts_backend"] = True
    payload["backend_port_prestart_summary_queries_console"] = True
    payload["backend_port_prestart_summary_executes_tests"] = True
    payload["expected_plan_required_for_real_run"] = False
    payload["physical_preflight_requires_existing_backend_connection"] = False
    payload["manual_review_required_before_readiness_update"] = False
    payload["fail_closed_conditions"] = ["backend_not_in_wing_mode"]
    payload["post_run_review_commands"] = []
    payload["request_sequence"].append({"type": "set_gain", "channel": 3})
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_plan_artifact_snapshot(plan_path)

    assert result.status == "failed"
    assert "wing_readonly_preflight_plan_contacts_backend_not_false" in result.detail[
        "violations"
    ]
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_contacts_backend_not_false"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_queries_console_not_false"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_backend_port_prestart_summary_executes_tests_not_false"
        in result.detail["violations"]
    )
    assert "wing_readonly_preflight_plan_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_plan_expected_plan_required_not_true" in result.detail[
        "violations"
    ]
    assert (
        "wing_readonly_preflight_plan_existing_backend_connection_boundary_not_true"
        in result.detail["violations"]
    )
    assert "wing_readonly_preflight_plan_manual_review_boundary_not_true" in result.detail[
        "violations"
    ]
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_physical_readback_missing_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_event_log_wing_mode_not_proven_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_transport_timeout_or_connection_loss_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_ws_handshake_rejected_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_existing_backend_connection_boundary_missing_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_prestart_summary_missing_or_invalid_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_condition_prestart_summary_sha256_mismatch_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_fail_closed_conditions_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_plan_post_run_review_commands_mismatch"
        in result.detail["violations"]
    )
    assert "wing_readonly_preflight_plan_request_sequence_mismatch" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_plan_unsafe_request_type_present" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_wing_preflight_plan_response_map_drift(
    tmp_path,
):
    plan_path = tmp_path / "response-map-drift.json"
    payload = valid_wing_readonly_preflight_plan_payload(plan_path)
    payload["expected_response_types_by_request_type"] = {
        "get_operator_mode": ["operator_mode_status"]
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_plan_artifact_snapshot(plan_path)

    assert result.status == "failed"
    assert (
        "wing_readonly_preflight_plan_response_type_mapping_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_plan_with_unexpected_fail_closed_condition(
    tmp_path,
):
    plan_path = tmp_path / "extra-fail-closed-plan.json"
    payload = valid_wing_readonly_preflight_plan_payload(plan_path)
    payload["fail_closed_conditions"] = list(
        release_check.WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS
    ) + ["unexpected_condition"]
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_plan_artifact_snapshot(plan_path)

    assert result.status == "failed"
    assert (
        "wing_readonly_preflight_plan_fail_closed_conditions_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "wing_readonly_preflight_handoff_packet.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    packet_path.write_text(
        json.dumps(valid_wing_readonly_preflight_handoff_packet_payload(plan_path)),
        encoding="utf-8",
    )

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "wing_readonly_preflight_handoff_packet_is_readonly_operator_gate"
    )
    assert result.detail["packet_path"] == str(packet_path)
    assert result.detail["expected_plan_output"] == str(plan_path)
    assert result.detail["packet_status"] == "operator_hardware_step_required"
    assert result.detail["plan_artifact_status"] == "passed"
    assert result.detail["plan_artifact_violations"] == []
    assert result.detail["read_only"] is True
    assert result.detail["live_write"] is False
    assert result.detail["mutates_console_state"] is False
    assert result.detail["queries_console"] is False
    assert result.detail["executes_tests"] is False
    assert result.detail["operator_required"] is True
    assert result.detail["execution_allowed_by_packet"] is False
    assert result.detail["readiness_matrix_update_allowed_by_packet"] is False
    assert result.detail["readiness_closure_allowed_by_packet"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["url"] == "ws://localhost:8765"
    assert result.detail["plan_output"] == str(plan_path)
    assert result.detail["artifact_root"] == "artifacts/wing_readonly_preflight"
    assert result.detail["channels"] == [3]
    assert result.detail["value_types"] == ["fader", "gain", "pan"]
    assert result.detail["preview_command_uses_plan_only"] is True
    assert result.detail["capture_command_requires_physical_readback"] is True
    assert result.detail["prestart_backend_port_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert result.detail["backend_port_prestart_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert result.detail["backend_port_prestart_check_operator_sequence_step_index"] == 0
    assert result.detail["backend_port_prestart_check_must_be_first_operator_step"] is True
    assert (
        "Before starting the intended backend/WING session"
        in result.detail["backend_port_prestart_check_operator_sequence_step"]
    )
    assert "prestart_backend_port_check" in result.detail[
        "backend_port_prestart_check_operator_sequence_step"
    ]
    assert result.detail["backend_port_prestart_check_operator_sequence_step_is_first"] is True
    assert result.detail["backend_port_prestart_gate"] == result.detail[
        "expected_backend_port_prestart_gate"
    ]
    assert result.detail["declared_expected_backend_port_prestart_gate"] == result.detail[
        "expected_backend_port_prestart_gate"
    ]
    assert result.detail["backend_port_prestart_gate"]["operator_sequence_step_index"] == 0
    assert result.detail["backend_port_prestart_gate"][
        "operator_sequence_step_must_be_first"
    ] is True
    assert result.detail["backend_port_prestart_gate"][
        "must_pass_before_backend_wing_session"
    ] is True
    assert result.detail["backend_port_prestart_gate"]["grants_live_authority"] is False
    assert result.detail["review_command_writes_decision_template"] is True
    assert result.detail["review_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert result.detail["review_decision_template_output_matches_contract"] is True
    assert result.detail["review_decision_template_output_contract_flag"] is True
    assert result.detail["review_command_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert result.detail["review_command_decision_template_output_matches_packet"] is True
    assert result.detail["review_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert result.detail["review_output_matches_contract"] is True
    assert result.detail["review_output_contract_flag"] is True
    assert result.detail["review_command_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert result.detail["review_command_output_matches_packet"] is True
    assert result.detail["decision_review_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert result.detail["decision_review_output_matches_contract"] is True
    assert result.detail["decision_review_output_contract_flag"] is True
    assert result.detail["decision_review_command_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert result.detail["decision_review_command_output_matches_packet"] is True
    assert "prestart_backend_port_check" in result.detail["command_names"]
    assert "preview_readonly_request_plan" in result.detail["command_names"]
    assert "confirm_no_live_authority" in result.detail[
        "required_review_checklist_ids"
    ]
    assert "apply_operator_proposal" in result.detail["forbidden_live_request_types"]
    assert "set_gain" in result.detail["forbidden_live_request_types"]
    assert "set_eq" in result.detail["forbidden_live_request_types"]
    assert "route_output" in result.detail["forbidden_live_request_types"]
    assert result.detail["required_forbidden_live_request_types"] == list(
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    assert result.detail["blockers_not_closed_by_packet"] == list(
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    assert result.detail["required_not_closed_blockers"] == list(
        release_check.WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    assert "physical_console_readback_proof" in result.detail[
        "blockers_not_closed_by_packet"
    ]
    assert result.detail["mvp_blocker_anchor"] == (
        release_check.wing_mvp_first_open_blocker_anchor()
    )
    assert result.detail["expected_mvp_blocker_anchor"] == (
        release_check.wing_mvp_first_open_blocker_anchor()
    )
    assert result.detail["mvp_blocker_anchor_matches_current_first_open"] is True
    assert result.detail["mvp_blocker_anchor_sha256"] == (
        release_check.wing_mvp_blocker_anchor_sha256(
            result.detail["mvp_blocker_anchor"]
        )
    )
    assert result.detail["actual_mvp_blocker_anchor_sha256"] == result.detail[
        "mvp_blocker_anchor_sha256"
    ]
    assert result.detail["expected_mvp_blocker_anchor_sha256"] == result.detail[
        "mvp_blocker_anchor_sha256"
    ]
    assert result.detail["mvp_blocker_anchor_sha256_matches_anchor"] is True
    assert result.detail["mvp_blocker_anchor_sha256_matches_current_first_open"] is True
    assert (
        result.detail["mvp_blocker_anchor"]["selected_blocker"]
        == WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    )
    assert result.detail["mvp_blocker_anchor"]["selected_evidence_file_count"] == len(
        result.detail["mvp_blocker_anchor"]["selected_evidence_files"]
    )
    assert result.detail["operator_sequence_count"] == 10
    assert (
        result.detail["operator_sequence_has_mvp_blocker_anchor_digest_review"]
        is True
    )
    assert result.detail[
        "operator_sequence_mvp_blocker_anchor_digest_review_step"
    ] == (
        "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
        "current first open blocker before supervised proof planning."
    )
    assert result.detail[
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index"
    ] == 8
    assert result.detail["operator_sequence_has_supervised_prestart_gate"] is True
    assert (
        result.detail["operator_sequence_supervised_prestart_gate_step"]
        == (
            "Only after accepted_decision_ready and a valid accepted-candidate "
            "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
        )
    )
    assert result.detail["operator_sequence_supervised_prestart_gate_step_index"] == 9
    assert result.detail["operator_sequence_supervised_prestart_gate_is_final"] is True
    assert result.detail["supervised_prestart_gate"] == result.detail[
        "expected_supervised_prestart_gate"
    ]
    assert result.detail["declared_expected_supervised_prestart_gate"] == result.detail[
        "expected_supervised_prestart_gate"
    ]
    assert result.detail["supervised_prestart_gate"][
        "operator_sequence_step_index"
    ] == 9
    assert result.detail["supervised_prestart_gate"][
        "operator_sequence_step_must_be_final"
    ] is True
    assert result.detail["supervised_prestart_gate"]["grants_live_authority"] is False
    assert result.detail["required_supervised_prestart_gate_markers"] == [
        "accepted_decision_ready",
        "valid accepted-candidate prestart trace",
        "WING_SUPERVISED_FADER_GAIN_PROOF",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_stale_handoff_packet_mvp_blocker_anchor(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "stale-handoff-anchor.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["mvp_blocker_anchor"]["selected_blocker_status"] = "closed"
    payload["mvp_blocker_anchor"]["packet_does_not_close_blocker"] = False
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["mvp_blocker_anchor_matches_current_first_open"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_closes_blocker"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_mvp_blocker_anchor_digest_mismatch(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-anchor-digest-mismatch.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["mvp_blocker_anchor_sha256"] = "0" * 64
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["mvp_blocker_anchor_matches_current_first_open"] is True
    assert result.detail["mvp_blocker_anchor_sha256_matches_anchor"] is False
    assert (
        result.detail["mvp_blocker_anchor_sha256_matches_current_first_open"]
        is False
    )
    assert (
        "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_sha256_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_expected_mvp_blocker_anchor_sha256_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_decision_template_output_mismatch(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-template-output-mismatch.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    stale_output = "artifacts/wing_readonly_preflight_review_decisions/stale_template.json"
    payload["decision_template_output"] = stale_output
    payload["decision_template_output_matches_contract"] = False
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["review_decision_template_output"] == stale_output
    assert result.detail["review_decision_template_output_matches_contract"] is False
    assert result.detail["review_decision_template_output_contract_flag"] is False
    assert result.detail["review_command_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert result.detail["review_command_decision_template_output_matches_packet"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_decision_template_output_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_decision_template_output_contract_flag_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_decision_template_output_command_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_review_output_mismatch(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-review-output-mismatch.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    stale_review_output = "Docs/reports/STALE_WING_READONLY_PREFLIGHT_REVIEW.md"
    stale_decision_output = "Docs/reports/STALE_WING_READONLY_DECISION_REVIEW.md"
    payload["review_output"] = stale_review_output
    payload["review_output_matches_contract"] = False
    payload["decision_review_output"] = stale_decision_output
    payload["decision_review_output_matches_contract"] = False
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["review_output"] == stale_review_output
    assert result.detail["review_output_matches_contract"] is False
    assert result.detail["review_output_contract_flag"] is False
    assert result.detail["review_command_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert result.detail["review_command_output_matches_packet"] is False
    assert result.detail["decision_review_output"] == stale_decision_output
    assert result.detail["decision_review_output_matches_contract"] is False
    assert result.detail["decision_review_output_contract_flag"] is False
    assert result.detail["decision_review_command_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert result.detail["decision_review_command_output_matches_packet"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_review_output_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_review_output_contract_flag_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_review_output_command_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_decision_review_output_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_decision_review_output_contract_flag_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_decision_review_output_command_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_without_anchor_digest_review_step(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-without-anchor-digest-review-step.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["operator_sequence"] = [
        item
        for item in payload["operator_sequence"]
        if "mvp_blocker_anchor_sha256" not in item
    ]
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["operator_sequence_count"] == 9
    assert (
        result.detail["operator_sequence_has_mvp_blocker_anchor_digest_review"]
        is False
    )
    assert result.detail[
        "operator_sequence_mvp_blocker_anchor_digest_review_step"
    ] == ""
    assert result.detail[
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index"
    ] is None
    assert (
        "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_digest_review_step_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_missing_wing_preflight_handoff_packet_artifact(
    tmp_path,
):
    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        tmp_path / "missing-handoff.json",
        plan_path=tmp_path / "wing_readonly_preflight_plan.json",
    )

    assert result.status == "failed"
    assert "wing_readonly_preflight_handoff_packet_missing_report" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_handoff_packet_without_first_prestart_gate(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-without-first-prestart-gate.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["backend_port_prestart_check_operator_sequence_step_index"] = 1
    payload["backend_port_prestart_check_must_be_first_operator_step"] = False
    payload["operator_sequence"].insert(0, "Start backend/WING session first.")
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["backend_port_prestart_check_operator_sequence_step_index"] == 1
    assert result.detail["backend_port_prestart_check_must_be_first_operator_step"] is False
    assert (
        result.detail["backend_port_prestart_check_operator_sequence_step_is_first"]
        is False
    )
    assert (
        "wing_readonly_preflight_handoff_packet_backend_port_prestart_step_index_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_backend_port_prestart_check_must_be_first_operator_step_not_true"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_backend_port_prestart_step_not_first"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_with_prestart_gate_mismatch(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-with-prestart-gate-mismatch.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["backend_port_prestart_gate"]["grants_live_authority"] = True
    payload["expected_backend_port_prestart_gate"][
        "operator_sequence_step_must_be_first"
    ] = False
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["backend_port_prestart_gate"] != result.detail[
        "expected_backend_port_prestart_gate"
    ]
    assert result.detail["declared_expected_backend_port_prestart_gate"] != result.detail[
        "expected_backend_port_prestart_gate"
    ]
    assert (
        "wing_readonly_preflight_handoff_packet_backend_port_prestart_gate_contract_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_expected_backend_port_prestart_gate_contract_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_without_supervised_prestart_gate(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-without-prestart-gate.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["operator_sequence"][-1] = (
        "Only after accepted_decision_ready, move to WING_SUPERVISED_FADER_GAIN_PROOF."
    )
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["operator_sequence_has_supervised_prestart_gate"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_with_split_supervised_prestart_gate(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-with-split-prestart-gate.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["operator_sequence"].insert(
        -1,
        "Verify valid accepted-candidate prestart trace before proof planning.",
    )
    payload["operator_sequence"][-1] = (
        "Only after accepted_decision_ready, move to WING_SUPERVISED_FADER_GAIN_PROOF."
    )
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["operator_sequence_has_supervised_prestart_gate"] is False
    assert result.detail["operator_sequence_supervised_prestart_gate_step"] == ""
    assert (
        "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_with_nonfinal_supervised_prestart_gate(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-with-nonfinal-prestart-gate.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["operator_sequence"].append(
        "Only after accepted_decision_ready, move to WING_SUPERVISED_FADER_GAIN_PROOF."
    )
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["operator_sequence_has_supervised_prestart_gate"] is True
    assert result.detail["operator_sequence_supervised_prestart_gate_step_index"] == 9
    assert result.detail["operator_sequence_supervised_prestart_gate_is_final"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_not_final"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_without_structured_supervised_prestart_gate(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-without-structured-prestart-gate.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload.pop("supervised_prestart_gate")
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["operator_sequence_has_supervised_prestart_gate"] is True
    assert result.detail["supervised_prestart_gate"] == {}
    assert (
        "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_contract_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_packet_with_declared_expected_gate_mismatch(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "handoff-with-expected-prestart-gate-mismatch.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["expected_supervised_prestart_gate"]["grants_live_authority"] = True
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["supervised_prestart_gate"] == result.detail[
        "expected_supervised_prestart_gate"
    ]
    assert result.detail["declared_expected_supervised_prestart_gate"] != result.detail[
        "expected_supervised_prestart_gate"
    ]
    assert (
        "wing_readonly_preflight_handoff_packet_expected_supervised_prestart_gate_contract_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_unsafe_wing_preflight_handoff_packet_artifact(
    tmp_path,
):
    plan_path = tmp_path / "wing_readonly_preflight_plan.json"
    packet_path = tmp_path / "unsafe-handoff.json"
    plan_path.write_text(
        json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
        encoding="utf-8",
    )
    payload = valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
    payload["live_authority_granted"] = True
    payload["queries_console"] = True
    payload["url"] = "ws://localhost:9999"
    payload["expected_plan_required_for_real_run"] = False
    payload["physical_preflight_requires_existing_backend_connection"] = False
    payload["backend_port_prestart_check_command"] = ["python3", "other_check.py"]
    payload["manual_review_required_before_readiness_update"] = False
    payload["commands"]["preview_readonly_request_plan"].remove("--plan-only")
    payload["blockers_not_closed_by_packet"].remove("physical_console_readback_proof")
    payload["forbidden_live_request_types"].remove("set_gain")
    packet_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert "wing_readonly_preflight_handoff_packet_queries_console_not_false" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_live_authority_granted_not_false" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_url_mismatch" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_expected_plan_required_for_real_run_not_true" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_physical_preflight_requires_existing_backend_connection_not_true" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_manual_review_required_before_readiness_update_not_true" in result.detail[
        "violations"
    ]
    assert (
        "wing_readonly_preflight_handoff_packet_backend_port_prestart_check_command_mismatch"
        in result.detail["violations"]
    )
    assert "wing_readonly_preflight_handoff_packet_preview_readonly_request_plan_mismatch" in result.detail[
        "violations"
    ]
    assert "wing_readonly_preflight_handoff_packet_preview_plan_only_missing" in result.detail[
        "violations"
    ]
    assert (
        "wing_readonly_preflight_handoff_packet_blocker_physical_console_readback_proof_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_forbidden_set_gain_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_handoff_with_invalid_wing_preflight_plan(
    tmp_path,
):
    plan_path = tmp_path / "invalid-wing-readonly-plan.json"
    packet_path = tmp_path / "handoff-with-invalid-plan.json"
    plan_payload = valid_wing_readonly_preflight_plan_payload(plan_path)
    plan_payload["live_authority_granted"] = True
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    packet_path.write_text(
        json.dumps(valid_wing_readonly_preflight_handoff_packet_payload(plan_path)),
        encoding="utf-8",
    )

    result = validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
        packet_path,
        plan_path=plan_path,
    )

    assert result.status == "failed"
    assert result.detail["plan_artifact_status"] == "failed"
    assert (
        "wing_readonly_preflight_plan_live_authority_granted_not_false"
        in result.detail["plan_artifact_violations"]
    )
    assert (
        "wing_readonly_preflight_handoff_packet_plan_artifact_invalid"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_missing_wing_preflight_runner(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_RUNNER_TOOL",
        "tools/missing_wing_readonly_preflight_runner.py",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["capture_tool_available"] is False
    assert result.detail["capture_tool_path"] == (
        "tools/missing_wing_readonly_preflight_runner.py"
    )
    assert result.detail["capture_command_template"][1] == (
        "tools/missing_wing_readonly_preflight_runner.py"
    )
    assert "wing_readonly_preflight_runner_tool_missing" in result.detail["violations"]


def test_no_wing_release_check_rejects_incomplete_wing_preflight_capture_template(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_capture_command_template",
        lambda: [
            "python3",
            "tools/wing_readonly_preflight_runner.py",
            "--channels",
            "3",
        ],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert (
        "wing_readonly_preflight_capture_operator_confirmation_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_capture_physical_readback_missing"
        in result.detail["violations"]
    )
    assert result.detail["capture_command_requires_operator_channel_confirmation"] is False
    assert result.detail["capture_command_operator_confirmation_matches_channels"] is False
    assert result.detail["capture_command_requires_physical_readback"] is False


def test_no_wing_release_check_rejects_mismatched_wing_preflight_confirmation(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_capture_command_template",
        lambda: [
            "python3",
            "tools/wing_readonly_preflight_runner.py",
            "--channels",
            "3",
            "--operator-confirm-channels",
            "4",
            "--require-physical-readback",
        ],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["capture_command_channels"] == "3"
    assert result.detail["capture_command_operator_confirmed_channels"] == "4"
    assert result.detail["capture_command_operator_confirmation_matches_channels"] is False
    assert (
        "wing_readonly_preflight_capture_operator_confirmation_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_missing_wing_preflight_review_tool(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_REVIEW_TOOL",
        "tools/missing_wing_readonly_preflight_review_tool.py",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_tool_available"] is False
    assert result.detail["review_tool_path"] == (
        "tools/missing_wing_readonly_preflight_review_tool.py"
    )
    assert result.detail["review_command"][1] == (
        "tools/missing_wing_readonly_preflight_review_tool.py"
    )
    assert "wing_readonly_preflight_review_tool_missing" in result.detail["violations"]


def test_no_wing_release_check_rejects_missing_wing_preflight_handoff_tool(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL",
        "tools/missing_wing_readonly_preflight_handoff_packet.py",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["handoff_packet_tool_available"] is False
    assert result.detail["handoff_packet_tool_path"] == (
        "tools/missing_wing_readonly_preflight_handoff_packet.py"
    )
    assert result.detail["handoff_packet_command"][1] == (
        "tools/missing_wing_readonly_preflight_handoff_packet.py"
    )
    assert (
        "wing_readonly_preflight_handoff_packet_tool_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_handoff_channel_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_handoff_packet_command",
        lambda: [
            "python3",
            "tools/build_wing_readonly_preflight_handoff_packet.py",
            "--url",
            "ws://localhost:8765",
            "--channels",
            "4",
            "--value-types",
            "fader,gain,pan",
            "--artifact-root",
            "artifacts/wing_readonly_preflight",
            "--plan-output",
            WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
            "--output",
            WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
        ],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["handoff_packet_channels"] == "4"
    assert result.detail["handoff_packet_channel_matches_capture"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_channel_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_handoff_output_mismatch(
    monkeypatch,
):
    handoff_command = release_check.wing_readonly_preflight_handoff_packet_command()
    output_index = handoff_command.index("--output") + 1
    handoff_command[output_index] = "Docs/reports/STALE_WING_HANDOFF_PACKET.md"
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_handoff_packet_command",
        lambda: handoff_command,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["handoff_packet_output"] == (
        "Docs/reports/STALE_WING_HANDOFF_PACKET.md"
    )
    assert result.detail["handoff_packet_output_matches_contract"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_output_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_handoff_plan_output_mismatch(
    monkeypatch,
):
    handoff_command = release_check.wing_readonly_preflight_handoff_packet_command()
    plan_output_index = handoff_command.index("--plan-output") + 1
    handoff_command[plan_output_index] = (
        "artifacts/wing_readonly_preflight_plan/stale_plan.json"
    )
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_handoff_packet_command",
        lambda: handoff_command,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["handoff_packet_plan_output"] == (
        "artifacts/wing_readonly_preflight_plan/stale_plan.json"
    )
    assert result.detail["handoff_packet_plan_output_matches_contract"] is False
    assert result.detail["handoff_packet_plan_output_matches_plan_command"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_plan_output_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_handoff_target_defaults_mismatch(
    monkeypatch,
):
    handoff_command = release_check.wing_readonly_preflight_handoff_packet_command()
    url_index = handoff_command.index("--url") + 1
    handoff_command[url_index] = "ws://localhost:9999"
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_handoff_packet_command",
        lambda: handoff_command,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["handoff_packet_target_defaults"] == {
        "url": "ws://localhost:9999",
        "channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "value_types": WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "artifact_root": WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    }
    assert result.detail["handoff_packet_target_defaults_match_contract"] is False
    assert (
        "wing_readonly_preflight_handoff_packet_target_defaults_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_runner_argparse_target_defaults_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_runner_argparse_target_defaults",
        lambda: {
            "url": "ws://localhost:9999",
            "channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
            "value_types": WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
            "artifact_root": WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        },
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["runner_argparse_target_defaults"] == {
        "url": "ws://localhost:9999",
        "channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "value_types": WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "artifact_root": WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    }
    assert result.detail["runner_argparse_target_defaults_match_contract"] is False
    assert (
        "wing_readonly_preflight_runner_argparse_target_defaults_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_runner_supported_value_types_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_runner_supported_value_types",
        lambda: ["fader", "gain"],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["expected_supported_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert result.detail["runner_supported_value_types"] == ["fader", "gain"]
    assert result.detail["runner_supported_value_types_match_contract"] is False
    assert (
        "wing_readonly_preflight_runner_supported_value_types_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_tracks_wing_preflight_proof_runner_contract():
    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["expected_proof_runner"] == (
        WING_READONLY_PREFLIGHT_PROOF_RUNNER
    )
    assert result.detail["runner_proof_runner"] == (
        WING_READONLY_PREFLIGHT_PROOF_RUNNER
    )
    assert result.detail["evidence_proof_runner"] == (
        WING_READONLY_PREFLIGHT_PROOF_RUNNER
    )
    assert result.detail["runner_proof_runner_match_contract"] is True
    assert result.detail["evidence_proof_runner_match_contract"] is True


def test_no_wing_release_check_rejects_wing_preflight_runner_proof_runner_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_runner_run_summary_contract",
        lambda: {
            "proof_runner": "unexpected_runner",
            "status": WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
            "reason": WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
        },
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["runner_proof_runner"] == "unexpected_runner"
    assert result.detail["runner_proof_runner_match_contract"] is False
    assert (
        "wing_readonly_preflight_runner_proof_runner_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_evidence_proof_runner_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_EVIDENCE_PROOF_RUNNER",
        "unexpected_runner",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["evidence_proof_runner"] == "unexpected_runner"
    assert result.detail["evidence_proof_runner_match_contract"] is False
    assert (
        "wing_readonly_preflight_evidence_proof_runner_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_tracks_wing_preflight_run_summary_contract():
    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["expected_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert result.detail["expected_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert result.detail["runner_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert result.detail["runner_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert result.detail["evidence_expected_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert result.detail["evidence_expected_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert result.detail["runner_run_summary_status_match_contract"] is True
    assert result.detail["runner_run_summary_reason_match_contract"] is True
    assert result.detail["evidence_run_summary_status_match_contract"] is True
    assert result.detail["evidence_run_summary_reason_match_contract"] is True


def test_no_wing_release_check_rejects_wing_preflight_runner_run_summary_contract_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_runner_run_summary_contract",
        lambda: {"status": "blocked", "reason": "wing_readonly_preflight_blocked"},
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["runner_run_summary_status"] == "blocked"
    assert result.detail["runner_run_summary_reason"] == (
        "wing_readonly_preflight_blocked"
    )
    assert result.detail["runner_run_summary_status_match_contract"] is False
    assert result.detail["runner_run_summary_reason_match_contract"] is False
    assert (
        "wing_readonly_preflight_runner_run_summary_status_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_runner_run_summary_reason_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_evidence_run_summary_contract_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_EVIDENCE_EXPECTED_RUN_SUMMARY_STATUS",
        "blocked",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_EVIDENCE_EXPECTED_RUN_SUMMARY_REASON",
        "wing_readonly_preflight_blocked",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["evidence_expected_run_summary_status"] == "blocked"
    assert result.detail["evidence_expected_run_summary_reason"] == (
        "wing_readonly_preflight_blocked"
    )
    assert result.detail["evidence_run_summary_status_match_contract"] is False
    assert result.detail["evidence_run_summary_reason_match_contract"] is False
    assert (
        "wing_readonly_preflight_evidence_run_summary_status_mismatch"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_evidence_run_summary_reason_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_tracks_wing_preflight_evidence_allowed_value_types():
    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "passed"
    assert result.detail["expected_supported_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert result.detail["evidence_allowed_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert result.detail["evidence_allowed_value_types_match_contract"] is True


def test_no_wing_release_check_rejects_wing_preflight_evidence_allowed_value_types_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_EVIDENCE_ALLOWED_VALUE_TYPES",
        {"fader", "gain"},
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["expected_supported_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert result.detail["evidence_allowed_value_types"] == ["fader", "gain"]
    assert result.detail["evidence_allowed_value_types_match_contract"] is False
    assert (
        "wing_readonly_preflight_evidence_allowed_value_types_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_runner_physical_readback_contract_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_runner_physical_readback_contract",
        lambda: {
            "statuses": [],
            "sources": sorted(WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES),
        },
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["expected_physical_readback_statuses"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert result.detail["runner_physical_readback_statuses"] == []
    assert result.detail["runner_physical_readback_statuses_match_contract"] is False
    assert (
        "wing_readonly_preflight_runner_physical_readback_statuses_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_evidence_physical_readback_contract_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_EVIDENCE_PHYSICAL_READBACK_SOURCES",
        {"physical_readback"},
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["expected_physical_readback_sources"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )
    assert result.detail["evidence_physical_readback_sources"] == ["physical_readback"]
    assert result.detail["evidence_physical_readback_sources_match_contract"] is False
    assert (
        "wing_readonly_preflight_evidence_physical_readback_sources_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_plan_command_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_plan_command",
        lambda: [
            "python3",
            "tools/wing_readonly_preflight_runner.py",
            "--url",
            "ws://localhost:8765",
            "--channels",
            "4",
            "--operator-confirm-channels",
            "4",
                "--operator-confirm-session-ready",
            "--value-types",
            "fader,gain,pan",
            "--require-physical-readback",
            "--artifact-root",
            "artifacts/wing_readonly_preflight",
        ],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["plan_command_uses_plan_only"] is False
    assert result.detail["plan_command_channels"] == "4"
    assert result.detail["plan_command_matches_capture"] is False
    assert "wing_readonly_preflight_plan_only_missing" in result.detail["violations"]
    assert "wing_readonly_preflight_plan_output_missing" in result.detail["violations"]
    assert "wing_readonly_preflight_plan_command_mismatch" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_review_command_without_decision_template_output(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_review_command",
        lambda: [
            "python3",
            "tools/review_wing_readonly_preflight_artifacts.py",
            "--artifact-root",
            "artifacts/wing_readonly_preflight",
            "--output",
            WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
        ],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_command_writes_decision_template"] is False
    assert result.detail["review_decision_template_output"] is None
    assert result.detail["review_decision_template_output_matches_contract"] is False
    assert (
        "wing_readonly_preflight_review_decision_template_output_missing"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_review_decision_template_output_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_review_decision_template_output_mismatch(
    monkeypatch,
):
    review_command = release_check.wing_readonly_preflight_review_command()
    output_index = review_command.index("--decision-template-output") + 1
    review_command[output_index] = (
        "artifacts/wing_readonly_preflight_review_decisions/stale_template.json"
    )
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_review_command",
        lambda: review_command,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_command_writes_decision_template"] is True
    assert result.detail["review_decision_template_output"] == review_command[output_index]
    assert result.detail["review_decision_template_output_matches_contract"] is False
    assert (
        "wing_readonly_preflight_review_decision_template_output_missing"
        not in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_review_decision_template_output_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_reads_review_checklist_from_exported_tool_behavior(
    monkeypatch,
    tmp_path: Path,
):
    fake_review_tool = tmp_path / "fake_wing_readonly_preflight_review_tool.py"
    fake_review_tool.write_text(
        "\n".join(
            [
                "# confirm_operator_channel_scope",
                "# confirm_physical_readbacks",
                "# confirm_event_log_readonly_sequence",
                "# confirm_no_live_authority",
                "def manual_review_checklist():",
                "    return [{'id': 'confirm_artifact_paths'}]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_REVIEW_TOOL",
        str(fake_review_tool),
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_tool_available"] is True
    assert result.detail["review_manual_checklist_ids"] == ["confirm_artifact_paths"]
    assert result.detail["review_manual_checklist_complete"] is False
    assert result.detail["review_decision_template_required_fields"] == []
    assert result.detail["review_decision_template_contract_complete"] is False
    assert (
        "wing_readonly_preflight_review_manual_checklist_incomplete"
        in result.detail["violations"]
    )
    assert (
        "wing_readonly_preflight_review_decision_template_contract_incomplete"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_incomplete_wing_preflight_review_checklist(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_review_manual_checklist_ids",
        lambda: ["confirm_artifact_paths"],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_manual_checklist_ids"] == ["confirm_artifact_paths"]
    assert result.detail["review_manual_checklist_complete"] is False
    assert (
        "wing_readonly_preflight_review_manual_checklist_incomplete"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_incomplete_wing_preflight_decision_template_contract(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_review_decision_template_required_fields",
        lambda: ["review_id", "artifact_source_id"],
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_decision_template_required_fields"] == [
        "review_id",
        "artifact_source_id",
    ]
    assert result.detail["review_decision_template_contract_complete"] is False
    assert (
        "wing_readonly_preflight_review_decision_template_contract_incomplete"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_stale_wing_preflight_decision_template_status(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "wing_readonly_preflight_review_decision_template_status",
        lambda: "accepted_decision_ready",
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    assert result.detail["review_decision_template_status"] == "accepted_decision_ready"
    assert result.detail["review_decision_template_status_matches_contract"] is False
    assert (
        "wing_readonly_preflight_review_decision_template_status_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_wing_preflight_evidence_authority_claim(
    monkeypatch,
):
    corrupted = {
        "type": "wing_readonly_preflight_evidence",
        "method_id": "wing_readonly_preflight_evidence",
        "status": "readonly_preflight_candidate_available",
        "safety": {
            "read_only": True,
            "live_write": True,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": False,
            "execution_allowed_by_evidence": True,
            "readiness_closure_allowed_by_evidence": True,
            "live_widening_allowed": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
        },
        "summary": {
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 1,
            "accepted_as_readonly_preflight_candidate": True,
            "readiness_matrix_update_allowed": True,
            "readiness_closure_allowed_by_evidence": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "mvp_blocker_candidate_count": 4,
            "mvp_blockers_closed_by_evidence": ["wing_readonly_connection_proof"],
            "blocked_by": [],
        },
        "mvp_blocker_matrix": [
            {
                "id": "wing_readonly_connection_proof",
                "status": "candidate_available",
                "closed": True,
                "manual_review_required": False,
                "automatic_live_authority_allowed": True,
            }
        ],
        "artifacts": [],
    }

    monkeypatch.setattr(
        release_check,
        "build_wing_readonly_preflight_evidence",
        lambda: corrupted,
    )

    result = validate_wing_readonly_preflight_evidence_snapshot()

    assert result.status == "failed"
    violations = set(result.detail["violations"])
    assert "wing_readonly_preflight_evidence_live_write_not_false" in violations
    assert "wing_readonly_preflight_evidence_operator_required_not_true" in violations
    assert (
        "wing_readonly_preflight_evidence_execution_allowed_by_evidence_not_false"
        in violations
    )
    assert (
        "wing_readonly_preflight_evidence_readiness_closure_allowed_by_evidence_not_false"
        in violations
    )
    assert "wing_readonly_preflight_evidence_live_widening_allowed_not_false" in violations
    assert "wing_readonly_preflight_evidence_live_authority_granted_not_false" in violations
    assert "wing_readonly_preflight_evidence_full_autonomous_ready_not_false" in violations
    assert "wing_readonly_preflight_invalid_artifacts_present" in violations
    assert "wing_readonly_preflight_candidate_summary_mismatch" in violations
    assert "wing_readonly_preflight_readiness_update_allowed_mismatch" in violations
    assert "wing_readonly_preflight_readiness_closure_allowed" in violations
    assert "wing_readonly_preflight_live_authority_granted" in violations
    assert "wing_readonly_preflight_full_autonomous_ready" in violations
    assert "wing_readonly_preflight_evidence_matrix_count_mismatch" in violations
    assert "wing_readonly_preflight_matrix_ids_mismatch" in violations
    assert "wing_readonly_preflight_matrix_row_closed_not_false" in violations
    assert "wing_readonly_preflight_matrix_manual_review_not_true" in violations
    assert "wing_readonly_preflight_matrix_live_authority_not_false" in violations


def test_no_wing_release_check_validates_wing_preflight_review_decision_ledger_snapshot():
    result = validate_wing_readonly_preflight_review_decision_ledger_snapshot()

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "wing_readonly_preflight_review_decision_is_readonly_operator_gate"
    )
    assert result.detail["ledger_status"] == "operator_review_required"
    assert result.detail["review_tool_path"] == (
        "tools/review_wing_readonly_preflight_decisions.py"
    )
    assert result.detail["review_tool_available"] is True
    assert result.detail["review_command"] == [
        "python3",
        "tools/review_wing_readonly_preflight_decisions.py",
        "--artifact-root",
        "artifacts/wing_readonly_preflight",
        "--decision-root",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
        "--output",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
    ]
    assert result.detail["review_exit_codes"] == {
        "accepted_decision_ready": 0,
        "preflight_candidate_missing": 2,
        "decision_missing_or_invalid": 3,
        "candidate_rejected": 4,
    }
    assert result.detail["preflight_evidence_status"] == "operator_preflight_required"
    assert result.detail["decision_search_roots"] == [
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
    ]
    assert result.detail["review_candidate_count"] == 0
    assert result.detail["decision_record_count"] == 0
    assert result.detail["accepted_record_count"] == 0
    assert result.detail["rejected_record_count"] == 0
    assert result.detail["invalid_record_count"] == 0
    assert result.detail["unreviewed_candidate_count"] == 0
    assert result.detail["manual_readiness_update_candidate_available"] is False
    assert result.detail["accepted_candidate_prestart_contract_valid"] is False
    assert result.detail["accepted_candidate_prestart_source_ids"] == []
    assert result.detail["operator_review_required"] is True
    assert result.detail["readiness_matrix_update_allowed_by_decision"] is False
    assert result.detail["readiness_closure_allowed_by_decision"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["candidate_reviews"] == []
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_missing_wing_preflight_decision_review_tool(
    monkeypatch,
):
    monkeypatch.setattr(
        release_check,
        "WING_READONLY_PREFLIGHT_REVIEW_DECISION_TOOL",
        "tools/missing_wing_readonly_preflight_decisions.py",
    )

    result = validate_wing_readonly_preflight_review_decision_ledger_snapshot()

    assert result.status == "failed"
    assert result.detail["review_tool_available"] is False
    assert result.detail["review_tool_path"] == (
        "tools/missing_wing_readonly_preflight_decisions.py"
    )
    assert result.detail["review_command"][1] == (
        "tools/missing_wing_readonly_preflight_decisions.py"
    )
    assert (
        "wing_readonly_preflight_review_decision_tool_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_raw_mixer_readback_contract_snapshot():
    result = validate_raw_mixer_state_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert result.detail["surface_id"] == "raw_mixer_state_cache"
    assert result.detail["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert result.detail["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["fixture_next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["remediation_next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "manual_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_raw_mixer_contract_drift(monkeypatch):
    contract = release_check.build_raw_mixer_state_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "raw_mixer_state_physical_readback.v0"
    corrupted["surface_id"] = "channel_identity_and_basic_controls"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["next_physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True

    monkeypatch.setattr(
        release_check,
        "build_raw_mixer_state_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_raw_mixer_state_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "raw_mixer_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_live_write_not_false" in result.detail["violations"]
    assert "raw_mixer_readback_required_top_level_fields_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_allows_cached_source" in result.detail["violations"]
    assert "raw_mixer_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_forbidden_mutation_fields_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_next_target_mismatch" in result.detail["violations"]
    assert "raw_mixer_readback_contract_live_authority_not_false" in result.detail["violations"]


def test_no_wing_release_check_validates_raw_mixer_artifact_intake_snapshot():
    result = validate_raw_mixer_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "raw_mixer_state_physical_readback.v1"
    assert result.detail["surface_id"] == "raw_mixer_state_cache"
    assert result.detail["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert result.detail["required_contract_method_id"] == "raw_mixer_state_physical_readback_contract"
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "operator_must_capture_raw_mixer_state_physical_readback_artifact" in result.detail["blocked_by"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_raw_mixer_artifact_intake_authority(monkeypatch):
    intake = release_check.build_raw_mixer_state_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_raw_mixer_state_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_raw_mixer_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "raw_mixer_artifact_intake_live_write_not_false" in result.detail["violations"]
    assert "raw_mixer_artifact_intake_fixture_closure_allowed_by_intake_not_false" in result.detail["violations"]
    assert "raw_mixer_artifact_intake_physical_gap_not_true" in result.detail["violations"]
    assert "raw_mixer_artifact_intake_fixture_closure_not_false" in result.detail["violations"]
    assert "raw_mixer_artifact_intake_live_authority_not_false" in result.detail["violations"]
    assert "raw_mixer_artifact_intake_full_autonomous_not_false" in result.detail["violations"]


def test_no_wing_release_check_validates_channel_basic_controls_readback_contract_snapshot():
    result = validate_channel_basic_controls_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "channel_basic_controls_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_identity_and_basic_controls"
    assert result.detail["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert result.detail["physical_readback_surface_id"] == "channel_identity_and_basic_controls"
    assert result.detail["physical_readback_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_channel_statuses"]
    assert "manual_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "cache_matches_physical_readback" in result.detail["required_coverage_flags"]
    assert result.detail["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_basic_controls_contract_drift(monkeypatch):
    contract = release_check.build_channel_basic_controls_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "channel_basic_controls_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_coverage_flags"] = []
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_channel_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["fader"]

    monkeypatch.setattr(
        release_check,
        "build_channel_basic_controls_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_channel_basic_controls_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "channel_basic_controls_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_basic_controls_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_controls_readback_required_control_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_controls_readback_required_coverage_flags_mismatch"
        in result.detail["violations"]
    )
    assert "channel_basic_controls_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_readback_allows_cached_source" in result.detail["violations"]
    assert "channel_basic_controls_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert (
        "channel_basic_controls_readback_forbidden_mutation_fields_mismatch"
        in result.detail["violations"]
    )
    assert "channel_basic_controls_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "channel_basic_controls_contract_live_authority_not_false" in result.detail["violations"]
    assert "channel_basic_controls_promotion_family_missing:gain" in result.detail["violations"]
    assert "channel_basic_controls_promotion_family_missing:mute" in result.detail["violations"]


def test_no_wing_release_check_validates_channel_basic_controls_artifact_intake_snapshot():
    result = validate_channel_basic_controls_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "channel_basic_controls_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_identity_and_basic_controls"
    assert result.detail["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert (
        result.detail["required_contract_method_id"]
        == "channel_basic_controls_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_channel_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert (
        "operator_must_capture_channel_basic_controls_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_basic_controls_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_channel_basic_controls_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_channel_basic_controls_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_channel_basic_controls_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "channel_basic_artifact_intake_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_basic_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "channel_basic_artifact_intake_physical_gap_not_true" in result.detail["violations"]
    assert (
        "channel_basic_artifact_intake_fixture_closure_not_false"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_artifact_intake_live_authority_not_false"
        in result.detail["violations"]
    )
    assert (
        "channel_basic_artifact_intake_full_autonomous_not_false"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_channel_processing_readback_contract_snapshot():
    result = validate_channel_processing_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "channel_processing_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_eq_dynamics_gate"
    assert result.detail["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert result.detail["physical_readback_surface_id"] == "channel_eq_dynamics_gate"
    assert result.detail["physical_readback_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_channel_statuses"]
    assert "eq_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_hpf_fields"] == ["enabled", "frequency_hz"]
    assert result.detail["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in result.detail["required_compressor_fields"]
    assert "range_db" in result.detail["required_gate_fields"]
    assert result.detail["minimum_eq_band_count"] == 4
    assert result.detail["promotion_families_requiring_surface"][:4] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_processing_contract_drift(monkeypatch):
    contract = release_check.build_channel_processing_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "channel_processing_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_eq_band_fields"] = ["band"]
    corrupted["artifact_contract"]["required_compressor_fields"] = ["enabled"]
    corrupted["artifact_contract"]["required_gate_fields"] = ["enabled"]
    corrupted["artifact_contract"]["minimum_eq_band_count"] = 2
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_channel_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["eq"]

    monkeypatch.setattr(
        release_check,
        "build_channel_processing_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_channel_processing_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "channel_processing_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_processing_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_readback_required_eq_band_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_readback_required_compressor_fields_mismatch"
        in result.detail["violations"]
    )
    assert "channel_processing_readback_required_gate_fields_mismatch" in result.detail["violations"]
    assert (
        "channel_processing_readback_artifact_minimum_eq_band_count_mismatch"
        in result.detail["violations"]
    )
    assert "channel_processing_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_allows_cached_source" in result.detail["violations"]
    assert "channel_processing_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "channel_processing_readback_forbidden_mutation_fields_mismatch" in result.detail["violations"]
    assert "channel_processing_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "channel_processing_contract_live_authority_not_false" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:compressor" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:gate" in result.detail["violations"]
    assert "channel_processing_promotion_family_missing:deessing" in result.detail["violations"]


def test_no_wing_release_check_validates_channel_processing_artifact_intake_snapshot():
    result = validate_channel_processing_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "channel_processing_physical_readback.v1"
    assert result.detail["surface_id"] == "channel_eq_dynamics_gate"
    assert result.detail["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert (
        result.detail["required_contract_method_id"]
        == "channel_processing_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_channel_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ]
    assert (
        "operator_must_capture_channel_processing_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_channel_processing_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_channel_processing_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_channel_processing_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_channel_processing_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "channel_processing_artifact_intake_live_write_not_false" in result.detail["violations"]
    assert (
        "channel_processing_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "channel_processing_artifact_intake_physical_gap_not_true" in result.detail["violations"]
    assert (
        "channel_processing_artifact_intake_fixture_closure_not_false"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_artifact_intake_live_authority_not_false"
        in result.detail["violations"]
    )
    assert (
        "channel_processing_artifact_intake_full_autonomous_not_false"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_bus_group_main_dca_readback_contract_snapshot():
    result = validate_bus_group_main_dca_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "bus_group_main_dca_physical_readback.v1"
    assert result.detail["surface_id"] == "bus_group_main_dca_state"
    assert result.detail["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert result.detail["physical_readback_surface_id"] == "bus_group_main_dca_state"
    assert result.detail["physical_readback_action_id"] == "remediate_bus_group_main_dca_state"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_surface_statuses"]
    assert "dca_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_bus_control_fields"] == [
        "fader_db",
        "mute",
        "pan",
        "width",
        "mono",
    ]
    assert "true_peak_dbtp" in result.detail["required_metering_fields"]
    assert "member_buses" in result.detail["required_dca_assignment_fields"]
    assert result.detail["minimum_bus_count"] == 1
    assert result.detail["minimum_main_count"] == 1
    assert result.detail["minimum_dca_count"] == 1
    assert result.detail["promotion_families_requiring_surface"][:3] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_bus_group_main_dca_contract_drift(monkeypatch):
    contract = release_check.build_bus_group_main_dca_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "bus_group_main_dca_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_bus_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_metering_fields"] = ["lufs"]
    corrupted["artifact_contract"]["minimum_bus_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_surface_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["dca"]

    monkeypatch.setattr(
        release_check,
        "build_bus_group_main_dca_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_bus_group_main_dca_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "bus_group_main_dca_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_contract_live_write_not_false" in result.detail["violations"]
    assert (
        "bus_group_main_dca_readback_required_top_level_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "bus_group_main_dca_readback_required_bus_control_fields_mismatch"
        in result.detail["violations"]
    )
    assert (
        "bus_group_main_dca_readback_required_metering_fields_mismatch"
        in result.detail["violations"]
    )
    assert "bus_group_main_dca_readback_artifact_minimum_bus_count_mismatch" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_allows_cached_source" in result.detail["violations"]
    assert "bus_group_main_dca_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "bus_group_main_dca_contract_live_authority_not_false" in result.detail["violations"]
    assert "bus_group_main_dca_promotion_family_missing:bus_group_matrix" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_promotion_family_missing:master_bus" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_bus_group_main_dca_artifact_intake_snapshot():
    result = validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "bus_group_main_dca_physical_readback.v1"
    assert result.detail["surface_id"] == "bus_group_main_dca_state"
    assert result.detail["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert (
        result.detail["required_contract_method_id"]
        == "bus_group_main_dca_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_bus_count"] == 0
    assert result.detail["valid_main_count"] == 0
    assert result.detail["valid_dca_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ]
    assert (
        "operator_must_capture_bus_group_main_dca_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_bus_group_main_dca_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_bus_group_main_dca_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_bus_group_main_dca_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "bus_group_main_dca_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "bus_group_main_dca_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "bus_group_main_dca_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert (
        "bus_group_main_dca_artifact_intake_fixture_closure_not_false"
        in result.detail["violations"]
    )
    assert "bus_group_main_dca_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "bus_group_main_dca_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_matrix_state_readback_contract_snapshot():
    result = validate_matrix_state_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "matrix_state_physical_readback.v1"
    assert result.detail["surface_id"] == "matrix_state"
    assert result.detail["remediation_action_id"] == "remediate_matrix_state"
    assert result.detail["physical_readback_surface_id"] == "matrix_state"
    assert result.detail["physical_readback_action_id"] == "remediate_matrix_state"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_matrix_statuses"]
    assert "matrix_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert result.detail["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert result.detail["required_eq_band_fields"] == [
        "band",
        "frequency_hz",
        "gain_db",
        "q",
    ]
    assert "gain_reduction_db" in result.detail["required_dynamics_fields"]
    assert "post_insert_weight" in result.detail["required_insert_fields"]
    assert "true_peak_dbtp" in result.detail["required_metering_fields"]
    assert result.detail["minimum_matrix_count"] == 1
    assert result.detail["minimum_eq_band_count"] == 8
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_matrix_state_contract_drift(monkeypatch):
    contract = release_check.build_matrix_state_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "matrix_state_physical_readback.v0"
    corrupted["surface_id"] = "raw_mixer_state_cache"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_control_fields"] = ["fader_db"]
    corrupted["artifact_contract"]["required_metering_fields"] = ["lufs"]
    corrupted["artifact_contract"]["minimum_matrix_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_matrix_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_matrix_state_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_matrix_state_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "matrix_state_readback_contract_schema_id_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_contract_surface_id_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_contract_live_write_not_false" in result.detail["violations"]
    assert "matrix_state_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_required_control_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_required_metering_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_artifact_minimum_matrix_count_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_allows_cached_source" in result.detail["violations"]
    assert "matrix_state_readback_accepted_statuses_mismatch" in result.detail["violations"]
    assert "matrix_state_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "matrix_state_contract_surface_summary_mismatch" in result.detail["violations"]
    assert "matrix_state_contract_live_authority_not_false" in result.detail["violations"]
    assert "matrix_state_promotion_family_missing:bus_group_matrix" in result.detail[
        "violations"
    ]
    assert "matrix_state_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_matrix_state_artifact_intake_snapshot():
    result = validate_matrix_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "matrix_state_physical_readback.v1"
    assert result.detail["surface_id"] == "matrix_state"
    assert result.detail["remediation_action_id"] == "remediate_matrix_state"
    assert (
        result.detail["required_contract_method_id"]
        == "matrix_state_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_matrix_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "bus_group_matrix" in result.detail["promotion_families_requiring_surface"]
    assert "fx_send_insert" in result.detail["promotion_families_requiring_surface"]
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["blocked_by"] == [
        "operator_must_capture_matrix_state_physical_readback_artifact"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_matrix_state_artifact_intake_authority(monkeypatch):
    intake = release_check.build_matrix_state_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_matrix_state_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_matrix_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "matrix_state_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "matrix_state_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "matrix_state_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert "matrix_state_artifact_intake_fixture_closure_not_false" in result.detail[
        "violations"
    ]
    assert "matrix_state_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "matrix_state_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_fx_slots_inserts_readback_contract_snapshot():
    result = validate_fx_slots_inserts_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "fx_slots_inserts_physical_readback.v1"
    assert result.detail["surface_id"] == "fx_slots_and_inserts"
    assert result.detail["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert result.detail["physical_readback_surface_id"] == "fx_slots_and_inserts"
    assert result.detail["physical_readback_action_id"] == "remediate_fx_slots_and_inserts"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "fx_parameter_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "parameter_schema" in result.detail["required_fx_slot_fields"]
    assert "safe_range" in result.detail["required_parameter_fields"]
    assert "physical_model_schema_verified" in result.detail[
        "required_parameter_schema_fields"
    ]
    assert "scoped_diff_marker" in result.detail["required_insert_coverage_flags"]
    assert result.detail["minimum_fx_slot_count"] == 1
    assert result.detail["minimum_fx_parameter_count"] == 4
    assert result.detail["minimum_insert_count"] == 1
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_fx_slots_inserts_contract_drift(monkeypatch):
    contract = release_check.build_fx_slots_inserts_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "fx_slots_inserts_physical_readback.v0"
    corrupted["surface_id"] = "matrix_state"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_fx_slot_fields"] = ["slot"]
    corrupted["artifact_contract"]["required_parameter_fields"] = ["value"]
    corrupted["artifact_contract"]["minimum_fx_parameter_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "routing_and_sends"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_fx_slots_inserts_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_fx_slots_inserts_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "fx_slots_inserts_readback_contract_schema_id_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_contract_surface_id_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_contract_live_write_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_fx_slot_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_required_parameter_fields_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "fx_slots_inserts_readback_artifact_minimum_fx_parameter_count_mismatch"
        in result.detail["violations"]
    )
    assert "fx_slots_inserts_readback_allowed_sources_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_allows_cached_source" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_accepted_statuses_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_contract_surface_summary_mismatch" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_contract_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_promotion_family_missing:mix_balance" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_fx_slots_inserts_artifact_intake_snapshot():
    result = validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "fx_slots_inserts_physical_readback.v1"
    assert result.detail["surface_id"] == "fx_slots_and_inserts"
    assert result.detail["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert (
        result.detail["required_contract_method_id"]
        == "fx_slots_inserts_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_fx_slot_count"] == 0
    assert result.detail["valid_insert_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "fx_send_insert" in result.detail["promotion_families_requiring_surface"]
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert (
        "operator_must_capture_fx_slots_inserts_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_fx_slots_inserts_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_fx_slots_inserts_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_fx_slots_inserts_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "fx_slots_inserts_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "fx_slots_inserts_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "fx_slots_inserts_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_artifact_intake_fixture_closure_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_fx_slots_inserts_artifact_intake_mixed_candidates(
    monkeypatch,
):
    intake = release_check.build_fx_slots_inserts_physical_readback_artifact_intake()
    mixed = json.loads(json.dumps(intake))
    mixed["status"] = "invalid_operator_artifact_present"
    mixed["summary"]["artifact_count"] = 2
    mixed["summary"]["valid_candidate_count"] = 1
    mixed["summary"]["invalid_candidate_count"] = 1
    mixed["summary"]["valid_fx_slot_count"] = 1
    mixed["summary"]["valid_insert_count"] = 1
    mixed["summary"]["accepted_as_physical_fixture_candidate"] = False
    mixed["summary"]["blocked_by"] = [
        "operator_must_replace_invalid_fx_slots_inserts_physical_readback_artifact"
    ]
    mixed["artifacts"] = [
        {"source_id": "valid_candidate.json"},
        {"source_id": "invalid_candidate.json"},
    ]

    monkeypatch.setattr(
        release_check,
        "build_fx_slots_inserts_physical_readback_artifact_intake",
        lambda: mixed,
    )

    result = validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "fx_slots_inserts_artifact_intake_invalid_candidate_present" in result.detail[
        "violations"
    ]
    assert "fx_slots_inserts_artifact_intake_candidate_flag_mismatch" not in result.detail[
        "violations"
    ]
    assert result.detail["accepted_as_physical_fixture_candidate"] is False


def test_no_wing_release_check_validates_routing_sends_readback_contract_snapshot():
    result = validate_routing_sends_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "routing_sends_physical_readback.v1"
    assert result.detail["surface_id"] == "routing_and_sends"
    assert result.detail["remediation_action_id"] == "remediate_routing_and_sends"
    assert result.detail["physical_readback_surface_id"] == "routing_and_sends"
    assert result.detail["physical_readback_action_id"] == "remediate_routing_and_sends"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "routing_write_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "wing_routing_hint" in result.detail["required_static_route_fields"]
    assert "pre_on" in result.detail["required_bus_send_fields"]
    assert "main" in result.detail["required_main_send_fields"]
    assert "target_matrix" in result.detail["required_matrix_send_fields"]
    assert "source_channel" in result.detail["required_output_route_fields"]
    assert "cache_matches_physical_readback" in result.detail["required_send_coverage_flags"]
    assert (
        "cache_matches_physical_readback"
        in result.detail["required_output_route_coverage_flags"]
    )
    assert result.detail["minimum_static_dante_route_count"] == 1
    assert result.detail["minimum_channel_bus_send_count"] == 1
    assert result.detail["minimum_channel_main_send_count"] == 1
    assert result.detail["minimum_channel_matrix_send_count"] == 1
    assert result.detail["minimum_output_route_count"] == 1
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_routing_sends_contract_drift(monkeypatch):
    contract = release_check.build_routing_sends_physical_readback_contract()
    corrupted = json.loads(json.dumps(contract))
    corrupted["schema_id"] = "routing_sends_physical_readback.v0"
    corrupted["surface_id"] = "matrix_state"
    corrupted["safety"]["live_write"] = True
    corrupted["artifact_contract"]["required_top_level_fields"] = ["type"]
    corrupted["artifact_contract"]["required_static_route_fields"] = ["start"]
    corrupted["artifact_contract"]["required_bus_send_fields"] = ["channel"]
    corrupted["artifact_contract"]["minimum_channel_bus_send_count"] = 0
    corrupted["artifact_contract"]["allowed_sources"].append("mixer_state_cache")
    corrupted["artifact_contract"]["accepted_row_statuses"] = ["verified_from_cache"]
    corrupted["artifact_contract"]["forbidden_mutation_evidence_fields"] = []
    corrupted["summary"]["physical_readback_surface_id"] = "dlive_state"
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["promotion_families_requiring_surface"] = ["routing"]

    monkeypatch.setattr(
        release_check,
        "build_routing_sends_physical_readback_contract",
        lambda: corrupted,
    )

    result = validate_routing_sends_physical_readback_contract_snapshot()

    assert result.status == "failed"
    assert "routing_sends_readback_contract_schema_id_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_contract_surface_id_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_contract_live_write_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_top_level_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_static_route_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_required_bus_send_fields_mismatch" in result.detail[
        "violations"
    ]
    assert (
        "routing_sends_readback_artifact_minimum_channel_bus_send_count_mismatch"
        in result.detail["violations"]
    )
    assert "routing_sends_readback_allowed_sources_mismatch" in result.detail["violations"]
    assert "routing_sends_readback_allows_cached_source" in result.detail["violations"]
    assert "routing_sends_readback_accepted_statuses_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_readback_forbidden_mutation_fields_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_contract_surface_summary_mismatch" in result.detail[
        "violations"
    ]
    assert "routing_sends_contract_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_promotion_family_missing:fx_send_insert" in result.detail[
        "violations"
    ]
    assert "routing_sends_promotion_family_missing:scene_preset" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_routing_sends_artifact_intake_snapshot():
    result = validate_routing_sends_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "routing_sends_physical_readback.v1"
    assert result.detail["surface_id"] == "routing_and_sends"
    assert result.detail["remediation_action_id"] == "remediate_routing_and_sends"
    assert (
        result.detail["required_contract_method_id"]
        == "routing_sends_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_static_dante_route_count"] == 0
    assert result.detail["valid_channel_bus_send_count"] == 0
    assert result.detail["valid_channel_main_send_count"] == 0
    assert result.detail["valid_channel_matrix_send_count"] == 0
    assert result.detail["valid_output_route_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert "fx_send_insert" in result.detail["promotion_families_requiring_surface"]
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert (
        "operator_must_capture_routing_sends_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_routing_sends_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_routing_sends_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_routing_sends_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_routing_sends_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "routing_sends_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "routing_sends_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "routing_sends_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert "routing_sends_artifact_intake_fixture_closure_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "routing_sends_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_validates_snapshots_scenes_readback_contract_snapshot():
    result = validate_snapshots_scenes_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "snapshots_scenes_physical_readback.v1"
    assert result.detail["surface_id"] == "snapshots_scenes"
    assert result.detail["remediation_action_id"] == "remediate_snapshots_scenes"
    assert result.detail["physical_readback_surface_id"] == "snapshots_scenes"
    assert result.detail["physical_readback_action_id"] == "remediate_snapshots_scenes"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "scene_recall_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "active_snapshot_index" in result.detail["required_active_scene_fields"]
    assert "scene_content_fingerprint" in result.detail["required_snapshot_fields"]
    assert "rollback_reference" in result.detail["required_coverage_flags"]
    assert "scene_load_allowed" in result.detail["required_false_safety_flags"]
    assert result.detail["minimum_snapshot_count"] == 1
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_snapshots_scenes_artifact_intake_snapshot():
    result = validate_snapshots_scenes_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "snapshots_scenes_physical_readback.v1"
    assert result.detail["surface_id"] == "snapshots_scenes"
    assert result.detail["remediation_action_id"] == "remediate_snapshots_scenes"
    assert (
        result.detail["required_contract_method_id"]
        == "snapshots_scenes_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_snapshot_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["scene_load_allowed"] is False
    assert result.detail["scene_save_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert (
        "operator_must_capture_snapshots_scenes_physical_readback_artifact"
        in result.detail["blocked_by"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_snapshots_scenes_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_snapshots_scenes_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["safety"]["scene_load_allowed"] = True
    corrupted["safety"]["scene_save_allowed"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["scene_load_allowed"] = True
    corrupted["summary"]["scene_save_allowed"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_snapshots_scenes_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_snapshots_scenes_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "snapshots_scenes_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "snapshots_scenes_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "snapshots_scenes_artifact_intake_scene_load_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_scene_save_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_fixture_closure_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_scene_load_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_scene_save_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_snapshots_scenes_artifact_intake_mixed_candidates(
    monkeypatch,
):
    intake = release_check.build_snapshots_scenes_physical_readback_artifact_intake()
    mixed = json.loads(json.dumps(intake))
    mixed["status"] = "invalid_operator_artifact_present"
    mixed["summary"]["artifact_count"] = 2
    mixed["summary"]["valid_candidate_count"] = 1
    mixed["summary"]["invalid_candidate_count"] = 1
    mixed["summary"]["valid_snapshot_count"] = 1
    mixed["summary"]["accepted_as_physical_fixture_candidate"] = False
    mixed["summary"]["blocked_by"] = [
        "operator_must_replace_invalid_snapshots_scenes_physical_readback_artifact"
    ]
    mixed["artifacts"] = [
        {"source_id": "valid_scene_candidate.json"},
        {"source_id": "invalid_scene_candidate.json"},
    ]

    monkeypatch.setattr(
        release_check,
        "build_snapshots_scenes_physical_readback_artifact_intake",
        lambda: mixed,
    )

    result = validate_snapshots_scenes_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "snapshots_scenes_artifact_intake_invalid_candidate_present" in result.detail[
        "violations"
    ]
    assert "snapshots_scenes_artifact_intake_candidate_flag_mismatch" not in result.detail[
        "violations"
    ]
    assert result.detail["accepted_as_physical_fixture_candidate"] is False


def test_no_wing_release_check_validates_user_presets_readback_contract_snapshot():
    result = validate_user_presets_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "user_presets_physical_readback.v1"
    assert result.detail["surface_id"] == "user_presets"
    assert result.detail["remediation_action_id"] == "remediate_user_presets"
    assert result.detail["physical_readback_surface_id"] == "user_presets"
    assert result.detail["physical_readback_action_id"] == "remediate_user_presets"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "preset_store_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "preset_content_fingerprint" in result.detail["required_preset_fields"]
    assert "rollback_reference" in result.detail["required_coverage_flags"]
    assert "preset_recall_allowed" in result.detail["required_false_safety_flags"]
    assert result.detail["minimum_preset_count"] == 1
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_user_presets_artifact_intake_snapshot():
    result = validate_user_presets_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["policy"] == "user_presets_artifact_intake_is_read_only_operator_review_gate"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "user_presets_physical_readback.v1"
    assert result.detail["surface_id"] == "user_presets"
    assert result.detail["remediation_action_id"] == "remediate_user_presets"
    assert result.detail["required_contract_method_id"] == "user_presets_physical_readback_contract"
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_preset_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["preset_recall_allowed"] is False
    assert result.detail["preset_store_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "scene_preset" in result.detail["promotion_families_requiring_surface"]
    assert "operator_must_capture_user_presets_physical_readback_artifact" in result.detail[
        "blocked_by"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_artifact_review_queue_snapshot():
    result = validate_read_surface_physical_readback_artifact_review_queue_snapshot()

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "physical_readback_artifact_review_queue_is_read_only_operator_gate"
    )
    assert result.detail["queue_status"] == "operator_artifact_capture_required"
    assert result.detail["intake_surface_count"] == 10
    assert result.detail["available_intake_count"] == result.detail["intake_surface_count"]
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["review_candidate_count"] == 0
    assert result.detail["missing_artifact_surface_count"] == result.detail[
        "intake_surface_count"
    ]
    assert result.detail["unsafe_authority_claim_count"] == 0
    assert result.detail["candidate_source_id_missing_count"] == 0
    assert result.detail["target_inconsistent_surface_count"] == 0
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_queue"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "raw_mixer_state_cache" in result.detail["missing_artifact_surface_ids"]
    assert "user_presets" in result.detail["missing_artifact_surface_ids"]
    assert "dlive_state" in result.detail["missing_artifact_surface_ids"]
    assert "raw_mixer_state_physical_readback_artifact_intake" in result.detail[
        "queue_item_method_ids"
    ]
    assert "dlive_state_physical_readback_artifact_intake" in result.detail[
        "queue_item_method_ids"
    ]
    assert "operator_must_capture_missing_physical_readback_artifacts" in result.detail[
        "blocked_by"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_fixture_acceptance_ledger_snapshot():
    result = validate_read_surface_physical_readback_fixture_acceptance_ledger_snapshot()

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "physical_readback_fixture_acceptance_ledger_is_read_only_operator_gate"
    )
    assert result.detail["ledger_status"] == "operator_acceptance_required"
    assert result.detail["queue_status"] == "operator_artifact_capture_required"
    assert result.detail["acceptance_record_count"] == 0
    assert result.detail["accepted_record_count"] == 0
    assert result.detail["rejected_record_count"] == 0
    assert result.detail["invalid_record_count"] == 0
    assert result.detail["candidate_trace_missing_count"] == 0
    assert result.detail["candidate_trace_mismatch_count"] == 0
    assert (
        result.detail["acceptance_template_tool_path"]
        == "tools/build_read_surface_physical_readback_acceptance_template.py"
    )
    assert result.detail["acceptance_template_tool_available"] is True
    assert result.detail["operator_acceptance_required"] is True
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_acceptance"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_fixture_closure_plan_snapshot():
    result = validate_read_surface_physical_readback_fixture_closure_plan_snapshot()

    assert result.status == "passed"
    assert (
        result.detail["policy"]
        == "physical_readback_fixture_closure_plan_is_manual_read_only_gate"
    )
    assert result.detail["plan_status"] == "fixture_closure_not_ready"
    assert result.detail["ledger_status"] == "operator_acceptance_required"
    assert result.detail["closure_candidate_count"] == 0
    assert result.detail["accepted_record_count"] == 0
    assert result.detail["invalid_record_count"] == 0
    assert result.detail["unreviewed_candidate_count"] == 0
    assert result.detail["manual_fixture_closure_candidate_available"] is False
    assert result.detail["manual_fixture_closure_required"] is True
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_update_allowed_by_plan"] is False
    assert result.detail["fixture_closure_allowed_by_plan"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_dlive_state_readback_contract_snapshot():
    result = validate_dlive_state_physical_readback_contract_snapshot()

    assert result.status == "passed"
    assert result.detail["contract_status"] == "contract_ready_physical_fixture_missing"
    assert result.detail["schema_id"] == "dlive_state_physical_readback.v1"
    assert result.detail["surface_id"] == "dlive_state"
    assert result.detail["remediation_action_id"] == "remediate_dlive_state"
    assert result.detail["physical_readback_surface_id"] == "dlive_state"
    assert result.detail["physical_readback_action_id"] == "remediate_dlive_state"
    assert result.detail["fixture_physical_gap_declared"] is True
    assert result.detail["remediation_physical_gap_declared"] is True
    assert result.detail["artifact_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["live_authority_granted"] is False
    assert "physical_readback" in result.detail["allowed_sources"]
    assert "mixer_state_cache" not in result.detail["allowed_sources"]
    assert "verified_from_physical_readback" in result.detail["accepted_row_statuses"]
    assert "midi_send_events" in result.detail["forbidden_mutation_evidence_fields"]
    assert "fader_raw_14bit" in result.detail["required_channel_fields"]
    assert "midi_base_channel" in result.detail["required_midi_state_fields"]
    assert "wing_surface_equivalent" in result.detail["required_parity_fields"]
    assert "wing_parity_mapping" in result.detail["required_coverage_flags"]
    assert "midi_send_allowed" in result.detail["required_false_safety_flags"]
    assert "tcp_send_allowed" in result.detail["required_false_safety_flags"]
    assert "dlive_write_allowed" in result.detail["required_false_safety_flags"]
    assert result.detail["minimum_channel_count"] == 1
    assert "performance_adaptation" in result.detail["promotion_families_requiring_surface"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_dlive_state_artifact_intake_snapshot():
    result = validate_dlive_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "passed"
    assert result.detail["intake_status"] == "operator_artifact_missing"
    assert result.detail["schema_id"] == "dlive_state_physical_readback.v1"
    assert result.detail["surface_id"] == "dlive_state"
    assert result.detail["remediation_action_id"] == "remediate_dlive_state"
    assert (
        result.detail["required_contract_method_id"]
        == "dlive_state_physical_readback_contract"
    )
    assert result.detail["artifact_count"] == 0
    assert result.detail["valid_candidate_count"] == 0
    assert result.detail["invalid_candidate_count"] == 0
    assert result.detail["valid_channel_count"] == 0
    assert result.detail["accepted_as_physical_fixture_candidate"] is False
    assert result.detail["operator_review_required"] is True
    assert result.detail["physical_fixture_gap_remains"] is True
    assert result.detail["fixture_closure_allowed_by_intake"] is False
    assert result.detail["midi_send_allowed"] is False
    assert result.detail["tcp_send_allowed"] is False
    assert result.detail["dlive_write_allowed"] is False
    assert result.detail["dlive_apply_allowed"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["full_autonomous_ready"] is False
    assert "fader" in result.detail["promotion_families_requiring_surface"]
    assert "gain" in result.detail["promotion_families_requiring_surface"]
    assert "mute" in result.detail["promotion_families_requiring_surface"]
    assert "routing" in result.detail["promotion_families_requiring_surface"]
    assert "performance_adaptation" in result.detail[
        "promotion_families_requiring_surface"
    ]
    assert result.detail["blocked_by"] == [
        "operator_must_capture_dlive_state_physical_readback_artifact"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_dlive_state_artifact_intake_authority(
    monkeypatch,
):
    intake = release_check.build_dlive_state_physical_readback_artifact_intake()
    corrupted = json.loads(json.dumps(intake))
    corrupted["safety"]["live_write"] = True
    corrupted["safety"]["fixture_closure_allowed_by_intake"] = True
    corrupted["safety"]["midi_send_allowed"] = True
    corrupted["safety"]["tcp_send_allowed"] = True
    corrupted["safety"]["dlive_write_allowed"] = True
    corrupted["safety"]["dlive_apply_allowed"] = True
    corrupted["summary"]["physical_fixture_gap_remains"] = False
    corrupted["summary"]["fixture_closure_allowed_by_intake"] = True
    corrupted["summary"]["midi_send_allowed"] = True
    corrupted["summary"]["tcp_send_allowed"] = True
    corrupted["summary"]["dlive_write_allowed"] = True
    corrupted["summary"]["dlive_apply_allowed"] = True
    corrupted["summary"]["live_authority_granted"] = True
    corrupted["summary"]["full_autonomous_ready"] = True

    monkeypatch.setattr(
        release_check,
        "build_dlive_state_physical_readback_artifact_intake",
        lambda: corrupted,
    )

    result = validate_dlive_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "dlive_state_artifact_intake_live_write_not_false" in result.detail[
        "violations"
    ]
    assert (
        "dlive_state_artifact_intake_fixture_closure_allowed_by_intake_not_false"
        in result.detail["violations"]
    )
    assert "dlive_state_artifact_intake_midi_send_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_tcp_send_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_dlive_write_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_dlive_apply_allowed_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_physical_gap_not_true" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_fixture_closure_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_midi_send_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_tcp_send_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_dlive_write_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_dlive_apply_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_live_authority_not_false" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_full_autonomous_not_false" in result.detail[
        "violations"
    ]


def test_no_wing_release_check_rejects_dlive_state_artifact_intake_mixed_candidates(
    monkeypatch,
):
    intake = release_check.build_dlive_state_physical_readback_artifact_intake()
    mixed = json.loads(json.dumps(intake))
    mixed["status"] = "invalid_operator_artifact_present"
    mixed["summary"]["artifact_count"] = 2
    mixed["summary"]["valid_candidate_count"] = 1
    mixed["summary"]["invalid_candidate_count"] = 1
    mixed["summary"]["valid_channel_count"] = 1
    mixed["summary"]["accepted_as_physical_fixture_candidate"] = False
    mixed["summary"]["blocked_by"] = [
        "operator_must_replace_invalid_dlive_state_physical_readback_artifact"
    ]
    mixed["artifacts"] = [
        {"source_id": "valid_dlive_state_candidate.json"},
        {"source_id": "invalid_dlive_state_candidate.json"},
    ]

    monkeypatch.setattr(
        release_check,
        "build_dlive_state_physical_readback_artifact_intake",
        lambda: mixed,
    )

    result = validate_dlive_state_physical_readback_artifact_intake_snapshot()

    assert result.status == "failed"
    assert "dlive_state_artifact_intake_invalid_candidate_present" in result.detail[
        "violations"
    ]
    assert "dlive_state_artifact_intake_candidate_flag_mismatch" not in result.detail[
        "violations"
    ]
    assert result.detail["accepted_as_physical_fixture_candidate"] is False


def test_no_wing_release_check_quick_mode_still_runs_write_surface_scan(tmp_path):
    checks = build_checks(run_dir=tmp_path, include_frontend=False, quick=True)

    check_names = [check.name for check in checks]
    assert "write_surface_static_scan" in check_names
    assert "read_surface_fixture_evidence" in check_names
    assert "memory_lint" in check_names
    assert "gain_dry_run_proof" in check_names
    assert "wing_readonly_preflight_plan" in check_names
    assert "read_surface_physical_readback_capture_packet" in check_names
    assert "read_surface_physical_readback_capture_template" in check_names
    assert "read_surface_physical_readback_artifact_review_queue" in check_names
    assert "read_surface_physical_readback_acceptance_template" in check_names
    assert "read_surface_physical_readback_fixture_closure_plan" in check_names
    assert "wing_readonly_preflight_handoff_packet" in check_names
    assert "supervised_proof_artifact_report" in check_names
    assert check_names.index("write_surface_static_scan") < check_names.index("backend_pytest")
    assert check_names.index("read_surface_fixture_evidence") < check_names.index("backend_pytest")
    assert check_names.index("write_surface_static_scan") < check_names.index(
        "read_surface_fixture_evidence"
    )
    assert check_names.index("offline_operator_validation_harness") < check_names.index(
        "wing_readonly_preflight_plan"
    )
    assert check_names.index("wing_readonly_preflight_plan") < check_names.index(
        "read_surface_physical_readback_capture_packet"
    )
    assert check_names.index("read_surface_physical_readback_capture_packet") < check_names.index(
        "read_surface_physical_readback_capture_template"
    )
    assert check_names.index("read_surface_physical_readback_capture_template") < check_names.index(
        "read_surface_physical_readback_artifact_review_queue"
    )
    assert check_names.index(
        "read_surface_physical_readback_artifact_review_queue"
    ) < check_names.index(
        "read_surface_physical_readback_acceptance_template"
    )
    assert check_names.index(
        "read_surface_physical_readback_acceptance_template"
    ) < check_names.index(
        "read_surface_physical_readback_fixture_closure_plan"
    )
    assert check_names.index(
        "read_surface_physical_readback_fixture_closure_plan"
    ) < check_names.index(
        "wing_readonly_preflight_handoff_packet"
    )
    assert check_names.index("wing_readonly_preflight_handoff_packet") < check_names.index(
        "gain_dry_run_proof"
    )
    assert check_names.index("gain_dry_run_proof") < check_names.index(
        "supervised_proof_artifact_report"
    )
    assert check_names.index("supervised_proof_artifact_report") < check_names.index(
        "memory_lint"
    )
    assert check_names.index("memory_lint") < check_names.index("diff_check")
    assert any(
        "tests/test_write_surface_inventory_static_scan.py" in check.command
        for check in checks
        if check.name == "write_surface_static_scan"
    )
    fixture_check = next(check for check in checks if check.name == "read_surface_fixture_evidence")
    assert "tests/test_read_surface_fixture_evidence.py" in fixture_check.command
    assert "tests/test_read_surface_physical_readback_capture_packet.py" in fixture_check.command
    assert "tests/test_read_surface_physical_readback_capture_template.py" in fixture_check.command
    assert "tests/test_read_surface_physical_readback_artifact_review_queue.py" in fixture_check.command
    assert (
        "tests/test_read_surface_physical_readback_fixture_acceptance_template.py"
        in fixture_check.command
    )
    assert (
        "tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py"
        in fixture_check.command
    )
    assert (
        "tests/test_read_surface_physical_readback_fixture_closure_plan.py"
        in fixture_check.command
    )
    assert "tests/test_raw_mixer_state_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_raw_mixer_state_physical_readback_artifact_intake.py" in fixture_check.command
    assert "tests/test_channel_basic_controls_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_channel_basic_controls_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_channel_processing_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_channel_processing_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_bus_group_main_dca_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_bus_group_main_dca_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_matrix_state_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_matrix_state_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_fx_slots_inserts_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_fx_slots_inserts_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_routing_sends_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_routing_sends_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_snapshots_scenes_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_snapshots_scenes_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_user_presets_physical_readback_contract.py" in fixture_check.command
    assert (
        "tests/test_dlive_state_physical_readback_artifact_intake.py"
        in fixture_check.command
    )
    assert "tests/test_dlive_state_physical_readback_contract.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_evidence.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_runner.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_review_tool.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_review_decision.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_review_decision_tool.py" in fixture_check.command
    assert "tests/test_wing_readonly_preflight_handoff_packet.py" in fixture_check.command
    report_check = next(check for check in checks if check.name == "supervised_proof_artifact_report")
    assert "tools/evaluate_supervised_proof_artifacts.py" in report_check.command
    assert "--quiet" in report_check.command
    assert "--require-physical" in report_check.command
    assert str(proof_report_path_for_run(tmp_path)) in report_check.command
    assert str(gain_proof_artifact_root_for_run(tmp_path)) in report_check.command
    assert report_check.command.count("--artifact-root") == 2
    handoff_check = next(
        check for check in checks if check.name == "wing_readonly_preflight_handoff_packet"
    )
    assert "tools/build_wing_readonly_preflight_handoff_packet.py" in handoff_check.command
    assert "--url" in handoff_check.command
    assert "ws://localhost:8765" in handoff_check.command
    assert "--channels" in handoff_check.command
    assert "3" in handoff_check.command
    assert "--value-types" in handoff_check.command
    assert "fader,gain,pan" in handoff_check.command
    assert "--artifact-root" in handoff_check.command
    assert "artifacts/wing_readonly_preflight" in handoff_check.command
    assert "--plan-output" in handoff_check.command
    assert str(wing_readonly_preflight_plan_path_for_run(tmp_path)) in handoff_check.command
    assert "--format" in handoff_check.command
    assert "json" in handoff_check.command
    assert "--quiet" in handoff_check.command
    assert str(wing_readonly_preflight_handoff_packet_path_for_run(tmp_path)) in (
        handoff_check.command
    )
    read_capture_packet_check = next(
        check for check in checks if check.name == "read_surface_physical_readback_capture_packet"
    )
    assert "tools/build_read_surface_physical_readback_capture_packet.py" in (
        read_capture_packet_check.command
    )
    assert "--format" in read_capture_packet_check.command
    assert "json" in read_capture_packet_check.command
    assert "--quiet" in read_capture_packet_check.command
    assert str(read_surface_physical_readback_capture_packet_path_for_run(tmp_path)) in (
        read_capture_packet_check.command
    )
    read_capture_template_check = next(
        check
        for check in checks
        if check.name == "read_surface_physical_readback_capture_template"
    )
    assert "tools/build_read_surface_physical_readback_capture_template.py" in (
        read_capture_template_check.command
    )
    assert "--format" in read_capture_template_check.command
    assert "json" in read_capture_template_check.command
    assert "--quiet" in read_capture_template_check.command
    assert str(read_surface_physical_readback_capture_template_path_for_run(tmp_path)) in (
        read_capture_template_check.command
    )
    read_artifact_review_queue_check = next(
        check
        for check in checks
        if check.name == "read_surface_physical_readback_artifact_review_queue"
    )
    assert "tools/build_read_surface_physical_readback_artifact_review_queue.py" in (
        read_artifact_review_queue_check.command
    )
    assert "--format" in read_artifact_review_queue_check.command
    assert "json" in read_artifact_review_queue_check.command
    assert "--quiet" in read_artifact_review_queue_check.command
    assert str(read_surface_physical_readback_artifact_review_queue_path_for_run(tmp_path)) in (
        read_artifact_review_queue_check.command
    )
    wing_plan_check = next(
        check for check in checks if check.name == "wing_readonly_preflight_plan"
    )
    assert "tools/wing_readonly_preflight_runner.py" in wing_plan_check.command
    assert "--plan-only" in wing_plan_check.command
    assert "--plan-output" in wing_plan_check.command
    assert str(wing_readonly_preflight_plan_path_for_run(tmp_path)) in (
        wing_plan_check.command
    )
    assert "--prestart-summary" in wing_plan_check.command
    assert str(wing_readonly_preflight_prestart_summary_path_for_run(tmp_path)) in (
        wing_plan_check.command
    )
    assert "--operator-confirm-channels" in wing_plan_check.command
    assert "--operator-confirm-session-ready" in wing_plan_check.command
    assert "--require-physical-readback" in wing_plan_check.command
    read_acceptance_template_check = next(
        check
        for check in checks
        if check.name == "read_surface_physical_readback_acceptance_template"
    )
    assert "tools/build_read_surface_physical_readback_acceptance_template.py" in (
        read_acceptance_template_check.command
    )
    assert "--quiet" in read_acceptance_template_check.command
    assert str(read_surface_physical_readback_acceptance_template_path_for_run(tmp_path)) in (
        read_acceptance_template_check.command
    )
    read_fixture_closure_plan_check = next(
        check
        for check in checks
        if check.name == "read_surface_physical_readback_fixture_closure_plan"
    )
    assert "tools/build_read_surface_physical_readback_fixture_closure_plan.py" in (
        read_fixture_closure_plan_check.command
    )
    assert "--format" in read_fixture_closure_plan_check.command
    assert "json" in read_fixture_closure_plan_check.command
    assert "--quiet" in read_fixture_closure_plan_check.command
    assert str(read_surface_physical_readback_fixture_closure_plan_path_for_run(tmp_path)) in (
        read_fixture_closure_plan_check.command
    )
    gain_check = next(check for check in checks if check.name == "gain_dry_run_proof")
    assert "tools/gain_proof_dry_run_runner.py" in gain_check.command
    assert str(gain_proof_artifact_root_for_run(tmp_path)) in gain_check.command


def valid_proof_report_with_gap_rollup() -> dict:
    return {
        "type": "supervised_proof_artifact_report",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "value_type_counts": {"gain": 1},
            "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
            "autonomy_gap_rollup": {
                "read_only": True,
                "live_write": False,
                "live_widening_allowed": False,
                "full_autonomous_ready": False,
                "source": "autonomy_mode_transition_gate.rows[mode=auto]",
                "full_autonomous_blocked": True,
                "full_auto_gap_bucket_counts": {"read_surface": 1, "evidence_requirement": 19},
                "evidence_requirement_ids_by_action_family": {
                    "gain": list(GAIN_REQUIRED_EVIDENCE_COMPONENTS),
                    "bus_group_matrix": [
                        "bus_group_assignment_cache_freshness",
                        "bus_group_fader_readback",
                        "matrix_send_readback",
                        "main_bus_true_peak_headroom",
                        "bus_group_physical_fixture",
                        "hardware_readback_and_rollback_proof",
                    ],
                    "mute": [
                        "current_mute_cache_freshness",
                        "mute_owner_intent_evidence",
                        "signal_return_or_idle_window_evidence",
                        "manual_mute_authority_review",
                        "mute_physical_fixture",
                        "hardware_readback_rollback_and_emergency_proof",
                    ],
                },
                "next_required_evidence_by_action_family": {
                    "bus_group_matrix": [
                        "Add bus/group/matrix safety review before promotion.",
                    ],
                    "mute": [
                        "Add fresh mute-state intent and physical proof before promotion.",
                    ],
                },
                "supervised_current_live_write_kinds": ["fader", "gain", "pan"],
                "supervised_current_live_write_surfaces": ["preamp", "trim", "fader", "pan"],
                "next_required_evidence": [
                    "Reduce every Full Autonomous readiness gap bucket to zero before reviewing auto mode.",
                    "Keep live apply limited to current supervised write kinds until physical readback and rollback evidence exists.",
                ],
                "claim": (
                    "Full Autonomous remains blocked; this rollup is report-only and does not widen "
                    "the current supervised fader/gain live-write boundary."
                ),
            }
        },
        "rows": [
            {
                "proof_id": "gain-proof-dry-run",
                "value_type": "gain",
                "channel": 1,
                "proof_mode": "dry_run",
                "status": "dry_run_ready",
                "promotion_allowed": False,
                "autonomous_ready": False,
                "blockers": ["physical_proof_required_for_promotion"],
                "evidence": {
                    "pre_read_status": "verified_from_physical_readback",
                    "pre_read_source": "fake_mixer",
                    "readback_status": "verified_from_physical_readback",
                    "readback_source": "fake_mixer",
                    "rollback_readback_status": "verified_from_physical_readback",
                    "rollback_readback_source": "fake_mixer",
                    "operator_approval_recorded": True,
                    "idempotency_key_present": True,
                },
            }
        ],
        "autonomy_mode_transition_gate": {
            "summary": {
                "autonomous_apply_allowed_modes": [],
                "blocked_modes": ["auto"],
            }
        },
        "autonomy_evidence_overlay": {
            "type": "autonomy_evidence_overlay",
            "status": "dry_run_only",
            "safety": {
                "read_only": True,
                "live_write": False,
                "live_widening_allowed": False,
                "automatic_safety_update_allowed": False,
                "full_autonomous_ready": False,
                "operator_required": True,
            },
            "summary": {
                "target_count": 3,
                "status_counts": {
                    "dry_run_only": 1,
                    "manual_safety_review_candidate": 0,
                    "no_current_evidence": 2,
                    "remediation_required": 0,
                },
                "manual_safety_review_candidate_target_ids": [],
                "remediation_required_target_ids": [],
                "dry_run_only_target_ids": ["gain_trim_supervised_promotion"],
                "no_current_evidence_target_ids": [
                    "fader_supervised_promotion",
                    "pan_phase_supervised_promotion",
                ],
                "autonomous_ready_target_ids": [],
                "global_blockers": [
                    "overlay_does_not_modify_safety_rows",
                    "manual_safety_review_required_before_promotion",
                    "full_autonomous_mode_remains_blocked",
                ],
            },
            "rows": [
                {
                    "id": "fader_autonomy_evidence_overlay",
                    "target_id": "fader_supervised_promotion",
                    "value_type": "fader",
                    "status": "no_current_evidence",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": True,
                    "autonomous_ready": False,
                    "proof_artifact_count": 0,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 0,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": [],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["proof_artifact_missing"],
                    "next_required_actions": ["Run one physical supervised fader proof."],
                },
                {
                    "id": "gain_autonomy_evidence_overlay",
                    "target_id": "gain_trim_supervised_promotion",
                    "value_type": "gain",
                    "status": "dry_run_only",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": False,
                    "autonomous_ready": False,
                    "proof_artifact_count": 1,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 1,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["physical_proof_required_for_promotion"],
                    "next_required_actions": ["Run one small supervised gain proof on WING."],
                },
                {
                    "id": "pan_autonomy_evidence_overlay",
                    "target_id": "pan_phase_supervised_promotion",
                    "value_type": "pan",
                    "status": "no_current_evidence",
                    "safety_review_candidate": False,
                    "supervised_live_allowed_static": False,
                    "autonomous_ready": False,
                    "proof_artifact_count": 0,
                    "physical_candidate_count": 0,
                    "dry_run_ready_count": 0,
                    "blocked_artifact_count": 0,
                    "remediation_action_count": 0,
                    "physical_candidate_proof_ids": [],
                    "dry_run_ready_proof_ids": [],
                    "blocked_proof_ids": [],
                    "remediation_action_ids": [],
                    "blockers": ["proof_artifact_missing"],
                    "next_required_actions": ["Run one physical supervised pan proof."],
                },
            ],
        },
        "autonomy_promotion_readiness": {
            "type": "autonomy_promotion_readiness",
            "summary": {
                "wing_readonly_preflight_evidence": {
                    "evidence_status": "readonly_preflight_candidate_available",
                    "readiness_update_candidate_available": True,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 0,
                    "prestart_backend_port_check_required": True,
                    "prestart_backend_port_check_must_pass_before_starting_backend": True,
                    "prestart_backend_port_check_command": [
                        "PYTHONPATH=backend",
                        "python3",
                        "tools/no_wing_release_check.py",
                        "--quick",
                    ],
                    "prestart_backend_port_check_failure_detail_fields": list(
                        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "blocked_by": [
                        "operator_must_review_readonly_preflight_candidate_before_readiness_update"
                    ],
                },
            },
        },
        "wing_readonly_preflight_review_decision_ledger": {
            "type": "wing_readonly_preflight_review_decision_ledger",
            "status": "operator_review_decision_records_available",
            "summary": {
                "ledger_status": "operator_review_decision_records_available",
                "accepted_record_count": 1,
                "rejected_record_count": 0,
                "accepted_source_ids": [
                    "artifacts/wing_readonly_preflight/run/summary.json"
                ],
                "accepted_candidate_prestart_contract_valid": True,
                "accepted_candidate_prestart_source_ids": [
                    "artifacts/wing_readonly_preflight/run/summary.json"
                ],
                "candidate_reviews": [
                    {
                        "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
                        "status": "valid_readonly_preflight_artifact",
                        "backend_port_prestart_check_required": True,
                        "backend_port_prestart_check_must_pass_before_starting_backend": True,
                        "backend_port_prestart_check_command": [
                            "PYTHONPATH=backend",
                            "python3",
                            "tools/no_wing_release_check.py",
                            "--quick",
                        ],
                        "backend_port_prestart_check_failure_detail_fields": list(
                            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                        ),
                    }
                ],
                "rejected_source_ids": [],
                "invalid_record_count": 0,
                "unreviewed_candidate_count": 0,
                "manual_readiness_update_candidate_available": True,
                "readiness_matrix_update_allowed_by_decision": False,
                "readiness_closure_allowed_by_decision": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "blocked_by": [
                    "readiness_matrix_update_still_requires_separate_manual_action"
                ],
            },
        },
        "next_wing_proof_plan": {
            "type": "supervised_wing_proof_plan",
            "status": "operator_proof_required",
            "safety": {
                "read_only": True,
                "live_write": False,
                "execution_allowed_by_plan": False,
                "live_widening_allowed": False,
                "automatic_safety_update_allowed": False,
                "full_autonomous_ready": False,
                "operator_required": True,
            },
            "summary": {
                "step_count": 1,
                "planned_value_types": ["gain"],
                "planned_channels": [3],
                "readonly_preflight_required": True,
                "readonly_preflight_candidate_available": True,
                "readonly_preflight_evidence_status": "readonly_preflight_candidate_available",
                "readonly_preflight_blocked_by": [
                    "operator_must_review_readonly_preflight_candidate_before_readiness_update"
                ],
                "readonly_preflight_prestart_check_required": True,
                "readonly_preflight_prestart_check_must_pass_before_starting_backend": True,
                "readonly_preflight_prestart_check_command": [
                    "PYTHONPATH=backend",
                    "python3",
                    "tools/no_wing_release_check.py",
                    "--quick",
                ],
                "readonly_preflight_prestart_check_failure_detail_fields": list(
                    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                ),
                "readonly_preflight_next_required_actions": [
                    "Manually review the read-only preflight candidate before updating readiness rows."
                ],
                "readonly_preflight_step_count": 0,
                "readonly_preflight_review_decision_required": True,
                "readonly_preflight_review_decision_accepted": True,
                "readonly_preflight_review_decision_status": "operator_review_decision_records_available",
                "readonly_preflight_review_decision_blocked_by": [
                    "readiness_matrix_update_still_requires_separate_manual_action"
                ],
                "readonly_preflight_review_decision_accepted_record_count": 1,
                "readonly_preflight_review_decision_accepted_source_ids": [
                    "artifacts/wing_readonly_preflight/run/summary.json"
                ],
                "readonly_preflight_review_decision_candidate_prestart_contract_valid": True,
                "readonly_preflight_review_decision_candidate_prestart_source_ids": [
                    "artifacts/wing_readonly_preflight/run/summary.json"
                ],
                "readonly_preflight_review_decision_rejected_source_ids": [],
                "readonly_preflight_review_decision_invalid_record_count": 0,
                "readonly_preflight_review_decision_unreviewed_candidate_count": 0,
                "global_blockers": [
                    "operator_must_approve_and_execute_manually",
                    "proof_plan_does_not_contact_mixer",
                    "safety_matrix_review_required_after_artifacts",
                ],
            },
            "readonly_preflight_steps": [],
            "steps": [
                {
                    "id": "gain_physical_proof_ch3",
                    "value_type": "gain",
                    "channel": 3,
                    "delta_db": -0.25,
                    "live_execution_required": True,
                    "execution_allowed_by_plan": False,
                    "operator_approval_required": True,
                    "command": [
                        "python3",
                        "tools/wing_operator_proposal_validation_runner.py",
                        "--url",
                        "ws://localhost:8765",
                        "--channel",
                        "3",
                        "--value-type",
                        "gain",
                        "--delta-db",
                        "-0.25",
                        "--operator-confirm-channel",
                        "3",
                        "--operator-confirm-value-type",
                        "gain",
                        "--operator-confirm-delta-db",
                        "-0.25",
                        "--execute",
                        "--prove-blocked-apply",
                        "--prove-emergency-stop",
                    ],
                    "rationale": "physical proof required",
                    "prerequisites": ["operator approval"],
                    "abort_conditions": ["operator concern"],
                    "expected_artifacts": ["summary.json"],
                    "evidence_blockers_to_review": ["physical_candidate_missing"],
                    "required_evidence_components": list(GAIN_REQUIRED_EVIDENCE_COMPONENTS),
                }
            ],
        },
    }


def test_no_wing_release_check_validates_proof_report_gap_rollup_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "passed"
    assert result.detail["autonomy_gap_rollup_present"] is True
    assert result.detail["full_auto_gap_bucket_counts"] == {
        "read_surface": 1,
        "evidence_requirement": 19,
    }
    assert result.detail["evidence_requirement_ids_by_action_family"]["gain"] == (
        GAIN_REQUIRED_EVIDENCE_COMPONENTS
    )
    assert result.detail["required_gain_evidence_components"] == (
        GAIN_REQUIRED_EVIDENCE_COMPONENTS
    )
    assert result.detail["evidence_requirement_ids_by_action_family"]["bus_group_matrix"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert result.detail["evidence_requirement_ids_by_action_family"]["mute"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert "bus_group_matrix" in result.detail["next_required_evidence_by_action_family"]
    assert result.detail["supervised_current_live_write_kinds"] == ["fader", "gain", "pan"]
    assert result.detail["supervised_current_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]
    assert result.detail["policy_supervised_live_write_kinds"] == ["fader", "gain", "pan"]
    assert result.detail["policy_supervised_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
        "pan",
    ]


def test_no_wing_release_check_validates_gain_dry_run_report_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_gain_dry_run(report_path)

    assert result.status == "passed"
    assert result.detail["gain_dry_run_row_count"] == 1
    assert result.detail["dry_run_ready_proof_ids"] == ["gain-proof-dry-run"]
    assert result.detail["value_type_counts"] == {"gain": 1}


def test_no_wing_release_check_validates_autonomy_evidence_overlay_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "passed"
    assert result.detail["overlay_status"] == "dry_run_only"
    assert result.detail["row_count"] == 3
    assert result.detail["dry_run_only_target_ids"] == ["gain_trim_supervised_promotion"]
    assert result.detail["no_current_evidence_target_ids"] == [
        "fader_supervised_promotion",
        "pan_phase_supervised_promotion",
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_next_wing_proof_plan_schema(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "passed"
    assert result.detail["plan_status"] == "operator_proof_required"
    assert result.detail["step_count"] == 1
    assert result.detail["planned_value_types"] == ["gain"]
    assert result.detail["planned_channels"] == [3]
    assert result.detail["readonly_preflight_required"] is True
    assert result.detail["readonly_preflight_candidate_available"] is True
    assert result.detail["readonly_preflight_evidence_status"] == "readonly_preflight_candidate_available"
    assert result.detail["readonly_preflight_review_decision_required"] is True
    assert result.detail["readonly_preflight_review_decision_accepted"] is True
    assert (
        result.detail["readonly_preflight_review_decision_status"]
        == "operator_review_decision_records_available"
    )
    assert result.detail["readonly_preflight_review_decision_accepted_record_count"] == 1
    assert (
        result.detail[
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert result.detail[
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == ["artifacts/wing_readonly_preflight/run/summary.json"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_next_wing_plan_with_stale_review_source_ids(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["next_wing_proof_plan"]["summary"][
        "readonly_preflight_review_decision_accepted_source_ids"
    ] = ["artifacts/wing_readonly_preflight/stale/summary.json"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert (
        "next_wing_proof_plan_review_decision_accepted_sources_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_next_wing_plan_without_candidate_prestart_trace(
    tmp_path,
):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["wing_readonly_preflight_review_decision_ledger"]["summary"][
        "candidate_reviews"
    ][0]["backend_port_prestart_check_required"] = False
    payload["next_wing_proof_plan"]["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
    ] = True
    payload["next_wing_proof_plan"]["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] = ["artifacts/wing_readonly_preflight/run/summary.json"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert (
        "next_wing_proof_plan_review_decision_candidate_prestart_contract_mismatch"
        in result.detail["violations"]
    )
    assert (
        "next_wing_proof_plan_review_decision_candidate_prestart_sources_mismatch"
        in result.detail["violations"]
    )
    assert (
        "next_wing_proof_plan_review_decision_accepted_mismatch"
        in result.detail["violations"]
    )
    assert (
        "next_wing_proof_plan_review_decision_candidate_prestart_blocker_missing"
        in result.detail["violations"]
    )


def test_no_wing_release_check_allows_next_wing_plan_blocked_by_candidate_prestart_trace(
    tmp_path,
):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["wing_readonly_preflight_review_decision_ledger"]["summary"][
        "candidate_reviews"
    ][0]["backend_port_prestart_check_required"] = False
    plan = payload["next_wing_proof_plan"]
    plan["status"] = "manual_review_ready"
    plan["summary"]["step_count"] = 0
    plan["summary"]["planned_value_types"] = []
    plan["summary"]["planned_channels"] = []
    plan["summary"]["readonly_preflight_review_decision_accepted"] = False
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
    ] = False
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] = []
    plan["summary"]["global_blockers"] = [
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof",
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof",
        "readiness_matrix_update_still_requires_separate_manual_action",
    ]
    plan["steps"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "passed"
    assert result.detail["readonly_preflight_review_decision_accepted"] is False
    assert (
        result.detail[
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert result.detail[
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_next_wing_plan_stale_operator_confirmation(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    command = payload["next_wing_proof_plan"]["steps"][0]["command"]
    command[command.index("--operator-confirm-channel") + 1] = "4"
    command[command.index("--operator-confirm-delta-db") + 1] = "-0.50"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert (
        "next_wing_proof_plan_step_0_operator_confirm_channel_mismatch"
        in result.detail["violations"]
    )
    assert (
        "next_wing_proof_plan_step_0_operator_confirm_delta_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_next_wing_gain_plan_without_evidence_components(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["next_wing_proof_plan"]["steps"][0]["required_evidence_components"] = [
        "physical_pre_read"
    ]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert (
        "next_wing_proof_plan_step_0_gain_evidence_components_mismatch"
        in result.detail["violations"]
    )


def test_no_wing_release_check_allows_next_wing_plan_blocked_by_preflight(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    preflight = payload["autonomy_promotion_readiness"]["summary"][
        "wing_readonly_preflight_evidence"
    ]
    preflight["evidence_status"] = "operator_preflight_required"
    preflight["readiness_update_candidate_available"] = False
    preflight["valid_candidate_count"] = 0
    preflight["invalid_candidate_count"] = 0
    preflight["blocked_by"] = ["operator_must_run_wing_readonly_preflight_runner"]
    decision = payload["wing_readonly_preflight_review_decision_ledger"]
    decision["status"] = "operator_review_required"
    decision["summary"]["accepted_record_count"] = 0
    decision["summary"]["rejected_record_count"] = 0
    decision["summary"]["accepted_source_ids"] = []
    decision["summary"]["rejected_source_ids"] = []
    decision["summary"]["invalid_record_count"] = 0
    decision["summary"]["unreviewed_candidate_count"] = 0
    decision["summary"]["manual_readiness_update_candidate_available"] = False
    decision["summary"]["blocked_by"] = [
        "operator_must_capture_valid_readonly_preflight_candidate_first"
    ]
    plan = payload["next_wing_proof_plan"]
    plan["status"] = "blocked_by_readonly_preflight"
    plan["summary"]["step_count"] = 0
    plan["summary"]["planned_value_types"] = []
    plan["summary"]["planned_channels"] = []
    plan["summary"]["readonly_preflight_candidate_available"] = False
    plan["summary"]["readonly_preflight_evidence_status"] = "operator_preflight_required"
    plan["summary"]["readonly_preflight_blocked_by"] = [
        "operator_must_run_wing_readonly_preflight_runner"
    ]
    plan["summary"]["readonly_preflight_step_count"] = 1
    plan["summary"]["readonly_preflight_review_decision_accepted"] = False
    plan["summary"]["readonly_preflight_review_decision_status"] = "operator_review_required"
    plan["summary"]["readonly_preflight_review_decision_blocked_by"] = [
        "operator_must_capture_valid_readonly_preflight_candidate_first"
    ]
    plan["summary"]["readonly_preflight_review_decision_accepted_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"] = []
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
    ] = False
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] = []
    plan["summary"]["readonly_preflight_review_decision_rejected_source_ids"] = []
    plan["summary"]["readonly_preflight_review_decision_invalid_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_unreviewed_candidate_count"] = 0
    plan["summary"]["global_blockers"] = [
        "readonly_preflight_candidate_required_before_supervised_proof",
        "operator_must_run_wing_readonly_preflight_runner",
    ]
    plan["readonly_preflight_steps"] = [
        {
            "id": "wing_readonly_preflight_required",
            "read_only": True,
            "live_execution_required": False,
            "execution_allowed_by_plan": False,
            "operator_approval_required": True,
            "command": [
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
            ],
            "rationale": "read-only preflight required",
            "prerequisites": ["operator present"],
            "abort_conditions": ["backend not wing"],
            "expected_artifacts": [WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT],
        }
    ]
    plan["steps"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "passed"
    assert result.detail["plan_status"] == "blocked_by_readonly_preflight"
    assert result.detail["step_count"] == 0
    assert result.detail["readonly_preflight_candidate_available"] is False
    assert result.detail["readonly_preflight_evidence_status"] == "operator_preflight_required"
    assert result.detail["readonly_preflight_step_count"] == 1
    assert "readonly_preflight_candidate_required_before_supervised_proof" in result.detail["global_blockers"]
    assert result.detail["violations"] == []


def test_no_wing_release_check_allows_next_wing_plan_waiting_on_review_decision(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    decision = payload["wing_readonly_preflight_review_decision_ledger"]
    decision["status"] = "operator_review_required"
    decision["summary"]["accepted_record_count"] = 0
    decision["summary"]["rejected_record_count"] = 0
    decision["summary"]["accepted_source_ids"] = []
    decision["summary"]["rejected_source_ids"] = []
    decision["summary"]["invalid_record_count"] = 0
    decision["summary"]["unreviewed_candidate_count"] = 1
    decision["summary"]["manual_readiness_update_candidate_available"] = False
    decision["summary"]["blocked_by"] = [
        "operator_must_record_readonly_preflight_review_decision"
    ]
    plan = payload["next_wing_proof_plan"]
    plan["status"] = "manual_review_ready"
    plan["summary"]["step_count"] = 0
    plan["summary"]["planned_value_types"] = []
    plan["summary"]["planned_channels"] = []
    plan["summary"]["readonly_preflight_review_decision_accepted"] = False
    plan["summary"]["readonly_preflight_review_decision_status"] = "operator_review_required"
    plan["summary"]["readonly_preflight_review_decision_blocked_by"] = [
        "operator_must_record_readonly_preflight_review_decision"
    ]
    plan["summary"]["readonly_preflight_review_decision_accepted_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"] = []
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
    ] = False
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] = []
    plan["summary"]["readonly_preflight_review_decision_rejected_source_ids"] = []
    plan["summary"]["readonly_preflight_review_decision_invalid_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_unreviewed_candidate_count"] = 1
    plan["summary"]["global_blockers"] = [
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof",
        "operator_must_record_readonly_preflight_review_decision",
    ]
    plan["steps"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "passed"
    assert result.detail["plan_status"] == "manual_review_ready"
    assert result.detail["step_count"] == 0
    assert result.detail["readonly_preflight_candidate_available"] is True
    assert result.detail["readonly_preflight_review_decision_accepted"] is False
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in result.detail["global_blockers"]
    )
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_next_wing_proof_plan_that_grants_execution(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    plan = payload["next_wing_proof_plan"]
    plan["safety"]["execution_allowed_by_plan"] = True
    plan["safety"]["live_write"] = True
    plan["safety"]["operator_required"] = False
    plan["summary"]["global_blockers"] = []
    plan["steps"][0]["execution_allowed_by_plan"] = True
    plan["steps"][0]["operator_approval_required"] = False
    plan["steps"][0]["delta_db"] = 0.25
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert "next_wing_proof_plan_execution_allowed_by_plan_not_false" in result.detail["violations"]
    assert "next_wing_proof_plan_live_write_not_false" in result.detail["violations"]
    assert "next_wing_proof_plan_operator_required_not_true" in result.detail["violations"]
    assert "next_wing_proof_plan_operator_blocker_missing" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_execution_allowed" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_operator_approval_not_required" in result.detail["violations"]
    assert "next_wing_proof_plan_step_0_delta_not_safe_negative_proof" in result.detail["violations"]


def test_no_wing_release_check_rejects_next_wing_steps_without_accepted_review_decision(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    decision = payload["wing_readonly_preflight_review_decision_ledger"]
    decision["status"] = "operator_review_required"
    decision["summary"]["accepted_record_count"] = 0
    decision["summary"]["accepted_source_ids"] = []
    decision["summary"]["invalid_record_count"] = 0
    decision["summary"]["unreviewed_candidate_count"] = 1
    decision["summary"]["manual_readiness_update_candidate_available"] = False
    decision["summary"]["blocked_by"] = [
        "operator_must_record_readonly_preflight_review_decision"
    ]
    plan = payload["next_wing_proof_plan"]
    plan["summary"]["readonly_preflight_review_decision_accepted"] = False
    plan["summary"]["readonly_preflight_review_decision_status"] = "operator_review_required"
    plan["summary"]["readonly_preflight_review_decision_blocked_by"] = [
        "operator_must_record_readonly_preflight_review_decision"
    ]
    plan["summary"]["readonly_preflight_review_decision_accepted_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"] = []
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_valid"
    ] = False
    plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] = []
    plan["summary"]["readonly_preflight_review_decision_invalid_record_count"] = 0
    plan["summary"]["readonly_preflight_review_decision_unreviewed_candidate_count"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_next_wing_plan(report_path)

    assert result.status == "failed"
    assert (
        "next_wing_proof_plan_operator_proof_without_review_decision"
        in result.detail["violations"]
    )
    assert (
        "next_wing_proof_plan_steps_without_accepted_review_decision"
        in result.detail["violations"]
    )


def test_no_wing_release_check_validates_wing_runner_physical_readback_policy():
    result = validate_wing_operator_proposal_runner_physical_readback_policy()

    assert result.status == "passed"
    assert result.detail["physical_readback_statuses"] == [
        "verified_from_physical_readback"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_validates_supervised_evaluator_physical_readback_policy():
    result = validate_supervised_proof_evaluator_physical_readback_policy()

    assert result.status == "passed"
    assert result.detail["physical_readback_statuses"] == [
        "verified_from_physical_readback"
    ]
    assert result.detail["violations"] == []


def test_no_wing_release_check_rejects_wing_runner_cache_as_physical(tmp_path):
    runner_path = tmp_path / "wing_operator_proposal_validation_runner.py"
    runner_path.write_text(
        """
PHYSICAL_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
def run(args):
    read_current_parameter(require_physical_readback=bool(args.execute))
    result.get("readback_status") in PHYSICAL_READBACK_STATUSES
    rollback_readback_status not in PHYSICAL_READBACK_STATUSES
    return {
        "physical_readback_required": bool(args.execute),
        "status": "verified_from_cache",
    }
""",
        encoding="utf-8",
    )

    result = validate_wing_operator_proposal_runner_physical_readback_policy(runner_path)

    assert result.status == "failed"
    assert "physical_readback_statuses_allows_cache" in result.detail["violations"]
    assert (
        "physical_readback_statuses_include_non_physical_status"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_evaluator_cache_as_physical(tmp_path):
    evaluator_path = tmp_path / "supervised_proof_artifact_evaluator.py"
    evaluator_path.write_text(
        """
PHYSICAL_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
def _append_physical_blockers(artifact, blockers):
    if artifact.pre_read_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_pre_read_status_required")
    if artifact.readback_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_apply_readback_status_required")
    if artifact.rollback_readback_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_rollback_readback_status_required")
""",
        encoding="utf-8",
    )

    result = validate_supervised_proof_evaluator_physical_readback_policy(evaluator_path)

    assert result.status == "failed"
    assert "physical_readback_statuses_allows_cache" in result.detail["violations"]
    assert (
        "physical_readback_statuses_include_non_physical_status"
        in result.detail["violations"]
    )


def test_no_wing_release_check_rejects_autonomy_overlay_that_claims_authority(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    overlay = payload["autonomy_evidence_overlay"]
    overlay["safety"]["live_write"] = True
    overlay["safety"]["live_widening_allowed"] = True
    overlay["safety"]["automatic_safety_update_allowed"] = True
    overlay["safety"]["full_autonomous_ready"] = True
    overlay["safety"]["operator_required"] = False
    overlay["summary"]["autonomous_ready_target_ids"] = ["gain_trim_supervised_promotion"]
    overlay["summary"]["global_blockers"] = []
    overlay["rows"][1]["autonomous_ready"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "failed"
    assert "autonomy_overlay_live_write_not_false" in result.detail["violations"]
    assert "autonomy_overlay_live_widening_allowed_not_false" in result.detail["violations"]
    assert "autonomy_overlay_automatic_safety_update_allowed_not_false" in result.detail["violations"]
    assert "autonomy_overlay_full_autonomous_ready_not_false" in result.detail["violations"]
    assert "autonomy_overlay_operator_required_not_true" in result.detail["violations"]
    assert "autonomy_overlay_autonomous_ready_targets_not_empty" in result.detail["violations"]
    assert "autonomy_overlay_row_1_autonomous_ready_not_false" in result.detail["violations"]


def test_no_wing_release_check_rejects_autonomy_overlay_summary_row_drift(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    overlay = payload["autonomy_evidence_overlay"]
    overlay["summary"]["target_count"] = 4
    overlay["summary"]["status_counts"]["dry_run_only"] = 2
    overlay["summary"]["dry_run_only_target_ids"] = [
        "gain_trim_supervised_promotion",
        "fader_supervised_promotion",
    ]
    overlay["summary"]["no_current_evidence_target_ids"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_autonomy_overlay(report_path)

    assert result.status == "failed"
    assert "autonomy_overlay_target_count_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_status_counts_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_dry_run_only_target_ids_mismatch" in result.detail["violations"]
    assert "autonomy_overlay_no_current_evidence_target_ids_mismatch" in result.detail["violations"]


def test_no_wing_release_check_rejects_gain_dry_run_report_that_claims_promotion(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["rows"][0]["promotion_allowed"] = True
    payload["rows"][0]["autonomous_ready"] = True
    payload["rows"][0]["blockers"] = []
    payload["rows"][0]["evidence"]["pre_read_source"] = "physical_readback"
    payload["summary"]["dry_run_ready_proof_ids"] = []
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_gain_dry_run(report_path)

    assert result.status == "failed"
    assert "summary_dry_run_ready_proof_ids_missing" in result.detail["violations"]
    assert "gain_dry_run_promotion_allowed_not_false" in result.detail["violations"]
    assert "gain_dry_run_autonomous_ready_not_false" in result.detail["violations"]
    assert "gain_dry_run_physical_proof_blocker_missing" in result.detail["violations"]
    assert "gain_dry_run_pre_read_source_not_fake_mixer" in result.detail["violations"]


def test_no_wing_release_check_rejects_missing_proof_report_gap_rollup(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    report_path.write_text(
        json.dumps(
            {
                "type": "supervised_proof_artifact_report",
                "summary": {},
                "autonomy_mode_transition_gate": {
                    "summary": {"autonomous_apply_allowed_modes": [], "blocked_modes": ["auto"]}
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "failed"
    assert "autonomy_gap_rollup_read_only_not_true" in result.detail["violations"]
    assert "autonomy_gap_rollup_bucket_counts_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_source_not_auto_mode_gate_row" in result.detail["violations"]
    assert "autonomy_gap_rollup_evidence_by_family_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_next_by_family_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_gain_evidence_components_mismatch" in (
        result.detail["violations"]
    )


def test_no_wing_release_check_rejects_gap_rollup_without_gain_evidence_components(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    payload["summary"]["autonomy_gap_rollup"][
        "evidence_requirement_ids_by_action_family"
    ]["gain"] = ["physical_pre_read"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "failed"
    assert result.detail["required_gain_evidence_components"] == (
        GAIN_REQUIRED_EVIDENCE_COMPONENTS
    )
    assert "autonomy_gap_rollup_gain_evidence_components_mismatch" in (
        result.detail["violations"]
    )


def test_no_wing_release_check_rejects_misleading_proof_report_gap_rollup_claims(tmp_path):
    report_path = tmp_path / "supervised_proof_artifact_report.json"
    payload = valid_proof_report_with_gap_rollup()
    rollup = payload["summary"]["autonomy_gap_rollup"]
    rollup["source"] = "manual_summary"
    rollup["claim"] = "Full Autonomous is ready for wider live apply."
    rollup["next_required_evidence"] = []
    rollup["supervised_current_live_write_kinds"] = ["fader", "gain", "eq"]
    rollup["supervised_current_live_write_surfaces"] = [
        "preamp",
        "trim",
        "fader",
        "send",
    ]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_supervised_proof_report_gap_rollup(report_path)

    assert result.status == "failed"
    assert "autonomy_gap_rollup_source_not_auto_mode_gate_row" in result.detail["violations"]
    assert "autonomy_gap_rollup_claim_does_not_preserve_auto_block" in result.detail["violations"]
    assert "autonomy_gap_rollup_claim_does_not_preserve_live_boundary" in result.detail["violations"]
    assert "autonomy_gap_rollup_next_required_evidence_missing" in result.detail["violations"]
    assert "autonomy_gap_rollup_unexpected_supervised_live_write_kinds" in result.detail["violations"]
    assert (
        "autonomy_gap_rollup_unexpected_supervised_live_write_surfaces"
        in result.detail["violations"]
    )
    assert result.detail["unexpected_supervised_live_write_kinds"] == ["eq"]
    assert result.detail["unexpected_supervised_live_write_surfaces"] == ["send"]


def test_no_wing_release_check_runs_gap_rollup_schema_after_report(tmp_path, monkeypatch):
    def fake_free_port(
        port: int,
        host: str = "127.0.0.1",
        listener_confirmation_path: Path | None = None,
    ) -> CheckResult:
        return CheckResult(
            name="backend_port_free",
            status="passed",
            started_at="2026-06-05T00:00:00Z",
            finished_at="2026-06-05T00:00:00Z",
            duration_sec=0.0,
            detail={"host": host, "port": port, "listener_detected": False},
        )

    def fake_command_check(check: CommandCheck, log_dir: Path) -> CheckResult:
        if check.name == "wing_readonly_preflight_plan":
            plan_path = wing_readonly_preflight_plan_path_for_run(log_dir.parent)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(valid_wing_readonly_preflight_plan_payload(plan_path)),
                encoding="utf-8",
            )
        if check.name == "wing_readonly_preflight_handoff_packet":
            plan_path = wing_readonly_preflight_plan_path_for_run(log_dir.parent)
            packet_path = wing_readonly_preflight_handoff_packet_path_for_run(
                log_dir.parent
            )
            packet_path.parent.mkdir(parents=True, exist_ok=True)
            packet_path.write_text(
                json.dumps(
                    valid_wing_readonly_preflight_handoff_packet_payload(plan_path)
                ),
                encoding="utf-8",
            )
        if check.name == "read_surface_physical_readback_capture_packet":
            packet_path = read_surface_physical_readback_capture_packet_path_for_run(
                log_dir.parent
            )
            packet_path.parent.mkdir(parents=True, exist_ok=True)
            packet_path.write_text(
                json.dumps(valid_read_surface_capture_packet_payload()),
                encoding="utf-8",
            )
        if check.name == "read_surface_physical_readback_capture_template":
            template_path = read_surface_physical_readback_capture_template_path_for_run(
                log_dir.parent
            )
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(
                json.dumps(valid_read_surface_capture_template_payload()),
                encoding="utf-8",
            )
        if check.name == "read_surface_physical_readback_artifact_review_queue":
            queue_path = read_surface_physical_readback_artifact_review_queue_path_for_run(
                log_dir.parent
            )
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(
                json.dumps(valid_read_surface_artifact_review_queue_payload()),
                encoding="utf-8",
            )
        if check.name == "read_surface_physical_readback_acceptance_template":
            template_path = read_surface_physical_readback_acceptance_template_path_for_run(
                log_dir.parent
            )
            template_path.parent.mkdir(parents=True, exist_ok=True)
            template_path.write_text(
                json.dumps(valid_read_surface_acceptance_template_payload()),
                encoding="utf-8",
            )
        if check.name == "read_surface_physical_readback_fixture_closure_plan":
            plan_path = read_surface_physical_readback_fixture_closure_plan_path_for_run(
                log_dir.parent
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps(valid_read_surface_fixture_closure_plan_artifact_payload()),
                encoding="utf-8",
            )
        if check.name == "supervised_proof_artifact_report":
            report_path = proof_report_path_for_run(log_dir.parent)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(valid_proof_report_with_gap_rollup()), encoding="utf-8")
        return CheckResult(
            name=check.name,
            status="passed",
            started_at="2026-06-05T00:00:00Z",
            finished_at="2026-06-05T00:00:00Z",
            duration_sec=0.0,
            command=check.shell_display(),
            cwd=check.cwd,
            returncode=0,
        )

    monkeypatch.setattr(release_check, "check_port_free", fake_free_port)
    monkeypatch.setattr(release_check, "run_command_check", fake_command_check)

    summary = run_release_checks(
        artifact_root=tmp_path,
        include_frontend=False,
        quick=True,
        command_timeout_sec=None,
    )

    result_names = [result["name"] for result in summary["results"]]
    assert summary["status"] == "passed"
    assert "read_surface_fixture_evidence_schema" in result_names
    assert "read_surface_remediation_plan_schema" in result_names
    assert "read_surface_physical_readback_capture_plan_schema" in result_names
    assert "wing_readonly_preflight_evidence_schema" in result_names
    assert "wing_readonly_preflight_plan_artifact_schema" in result_names
    assert "wing_readonly_preflight_handoff_packet_artifact_schema" in result_names
    assert "wing_readonly_preflight_review_decision_ledger_schema" in result_names
    assert "read_surface_physical_readback_capture_packet_artifact_schema" in result_names
    assert (
        "read_surface_physical_readback_capture_template_artifact_schema"
        in result_names
    )
    assert (
        "read_surface_physical_readback_artifact_review_queue_artifact_schema"
        in result_names
    )
    assert (
        "read_surface_physical_readback_acceptance_template_artifact_schema"
        in result_names
    )
    assert "read_surface_physical_readback_fixture_closure_plan_artifact_schema" in result_names
    assert "raw_mixer_state_physical_readback_contract_schema" in result_names
    assert "raw_mixer_state_physical_readback_artifact_intake_schema" in result_names
    assert "channel_basic_controls_physical_readback_contract_schema" in result_names
    assert "channel_basic_controls_physical_readback_artifact_intake_schema" in result_names
    assert "channel_processing_physical_readback_contract_schema" in result_names
    assert "channel_processing_physical_readback_artifact_intake_schema" in result_names
    assert "bus_group_main_dca_physical_readback_contract_schema" in result_names
    assert "bus_group_main_dca_physical_readback_artifact_intake_schema" in result_names
    assert "matrix_state_physical_readback_contract_schema" in result_names
    assert "matrix_state_physical_readback_artifact_intake_schema" in result_names
    assert "fx_slots_inserts_physical_readback_contract_schema" in result_names
    assert "fx_slots_inserts_physical_readback_artifact_intake_schema" in result_names
    assert "routing_sends_physical_readback_contract_schema" in result_names
    assert "routing_sends_physical_readback_artifact_intake_schema" in result_names
    assert "snapshots_scenes_physical_readback_contract_schema" in result_names
    assert "snapshots_scenes_physical_readback_artifact_intake_schema" in result_names
    assert "user_presets_physical_readback_contract_schema" in result_names
    assert "user_presets_physical_readback_artifact_intake_schema" in result_names
    assert "dlive_state_physical_readback_contract_schema" in result_names
    assert "dlive_state_physical_readback_artifact_intake_schema" in result_names
    assert "read_surface_physical_readback_artifact_review_queue_schema" in result_names
    assert "read_surface_physical_readback_fixture_acceptance_ledger_schema" in result_names
    assert "read_surface_physical_readback_fixture_closure_plan_schema" in result_names
    assert "supervised_proof_report_gain_dry_run_schema" in result_names
    assert "autonomy_evidence_overlay_schema" in result_names
    assert "next_wing_proof_plan_schema" in result_names
    assert "wing_operator_proposal_runner_physical_readback_policy" in result_names
    assert "supervised_proof_evaluator_physical_readback_policy" in result_names
    assert "supervised_proof_report_gap_rollup_schema" in result_names
    assert result_names.index("read_surface_fixture_evidence") < result_names.index(
        "read_surface_fixture_evidence_schema"
    )
    assert result_names.index("read_surface_fixture_evidence_schema") < result_names.index(
        "read_surface_remediation_plan_schema"
    )
    assert result_names.index("read_surface_remediation_plan_schema") < result_names.index(
        "read_surface_physical_readback_capture_plan_schema"
    )
    assert result_names.index("read_surface_physical_readback_capture_plan_schema") < result_names.index(
        "wing_readonly_preflight_evidence_schema"
    )
    assert result_names.index("wing_readonly_preflight_evidence_schema") < result_names.index(
        "wing_readonly_preflight_review_decision_ledger_schema"
    )
    assert result_names.index("wing_readonly_preflight_plan") < result_names.index(
        "wing_readonly_preflight_plan_artifact_schema"
    )
    assert result_names.index(
        "wing_readonly_preflight_plan_artifact_schema"
    ) < result_names.index("read_surface_physical_readback_capture_packet")
    assert result_names.index("read_surface_physical_readback_capture_packet") < result_names.index(
        "read_surface_physical_readback_capture_packet_artifact_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_capture_packet_artifact_schema"
    ) < result_names.index("read_surface_physical_readback_capture_template")
    assert result_names.index("read_surface_physical_readback_capture_template") < result_names.index(
        "read_surface_physical_readback_capture_template_artifact_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_capture_template_artifact_schema"
    ) < result_names.index("read_surface_physical_readback_artifact_review_queue")
    assert result_names.index(
        "read_surface_physical_readback_artifact_review_queue"
    ) < result_names.index(
        "read_surface_physical_readback_artifact_review_queue_artifact_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_artifact_review_queue_artifact_schema"
    ) < result_names.index(
        "read_surface_physical_readback_acceptance_template"
    )
    assert result_names.index(
        "read_surface_physical_readback_acceptance_template"
    ) < result_names.index(
        "read_surface_physical_readback_acceptance_template_artifact_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_acceptance_template_artifact_schema"
    ) < result_names.index("read_surface_physical_readback_fixture_closure_plan")
    assert result_names.index(
        "read_surface_physical_readback_fixture_closure_plan"
    ) < result_names.index(
        "read_surface_physical_readback_fixture_closure_plan_artifact_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_fixture_closure_plan_artifact_schema"
    ) < result_names.index("wing_readonly_preflight_handoff_packet")
    assert result_names.index("wing_readonly_preflight_handoff_packet") < result_names.index(
        "wing_readonly_preflight_handoff_packet_artifact_schema"
    )
    assert result_names.index(
        "wing_readonly_preflight_handoff_packet_artifact_schema"
    ) < result_names.index("gain_dry_run_proof")
    assert result_names.index("wing_readonly_preflight_review_decision_ledger_schema") < result_names.index(
        "raw_mixer_state_physical_readback_contract_schema"
    )
    assert result_names.index("raw_mixer_state_physical_readback_contract_schema") < result_names.index(
        "raw_mixer_state_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("raw_mixer_state_physical_readback_artifact_intake_schema") < result_names.index(
        "channel_basic_controls_physical_readback_contract_schema"
    )
    assert result_names.index("channel_basic_controls_physical_readback_contract_schema") < result_names.index(
        "channel_basic_controls_physical_readback_artifact_intake_schema"
    )
    assert result_names.index(
        "channel_basic_controls_physical_readback_artifact_intake_schema"
    ) < result_names.index(
        "channel_processing_physical_readback_contract_schema"
    )
    assert result_names.index("channel_processing_physical_readback_contract_schema") < result_names.index(
        "channel_processing_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("channel_processing_physical_readback_artifact_intake_schema") < result_names.index(
        "bus_group_main_dca_physical_readback_contract_schema"
    )
    assert result_names.index("bus_group_main_dca_physical_readback_contract_schema") < result_names.index(
        "bus_group_main_dca_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("bus_group_main_dca_physical_readback_artifact_intake_schema") < result_names.index(
        "matrix_state_physical_readback_contract_schema"
    )
    assert result_names.index("matrix_state_physical_readback_contract_schema") < result_names.index(
        "matrix_state_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("matrix_state_physical_readback_artifact_intake_schema") < result_names.index(
        "fx_slots_inserts_physical_readback_contract_schema"
    )
    assert result_names.index("fx_slots_inserts_physical_readback_contract_schema") < result_names.index(
        "fx_slots_inserts_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("fx_slots_inserts_physical_readback_artifact_intake_schema") < result_names.index(
        "routing_sends_physical_readback_contract_schema"
    )
    assert result_names.index("routing_sends_physical_readback_contract_schema") < result_names.index(
        "routing_sends_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("routing_sends_physical_readback_artifact_intake_schema") < result_names.index(
        "snapshots_scenes_physical_readback_contract_schema"
    )
    assert result_names.index("snapshots_scenes_physical_readback_contract_schema") < result_names.index(
        "snapshots_scenes_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("snapshots_scenes_physical_readback_artifact_intake_schema") < result_names.index(
        "user_presets_physical_readback_contract_schema"
    )
    assert result_names.index("user_presets_physical_readback_contract_schema") < result_names.index(
        "user_presets_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("user_presets_physical_readback_artifact_intake_schema") < result_names.index(
        "dlive_state_physical_readback_contract_schema"
    )
    assert result_names.index("dlive_state_physical_readback_contract_schema") < result_names.index(
        "dlive_state_physical_readback_artifact_intake_schema"
    )
    assert result_names.index("dlive_state_physical_readback_artifact_intake_schema") < result_names.index(
        "read_surface_physical_readback_artifact_review_queue_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_artifact_review_queue_schema"
    ) < result_names.index(
        "read_surface_physical_readback_fixture_acceptance_ledger_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_fixture_acceptance_ledger_schema"
    ) < result_names.index(
        "read_surface_physical_readback_fixture_closure_plan_schema"
    )
    assert result_names.index(
        "read_surface_physical_readback_fixture_closure_plan_schema"
    ) < result_names.index(
        "backend_pytest"
    )
    assert result_names.index("supervised_proof_artifact_report") < result_names.index(
        "supervised_proof_report_gain_dry_run_schema"
    )
    assert result_names.index("supervised_proof_report_gain_dry_run_schema") < result_names.index(
        "autonomy_evidence_overlay_schema"
    )
    assert result_names.index("autonomy_evidence_overlay_schema") < result_names.index(
        "next_wing_proof_plan_schema"
    )
    assert result_names.index("next_wing_proof_plan_schema") < result_names.index(
        "wing_operator_proposal_runner_physical_readback_policy"
    )
    assert result_names.index(
        "wing_operator_proposal_runner_physical_readback_policy"
    ) < result_names.index(
        "supervised_proof_evaluator_physical_readback_policy"
    )
    assert result_names.index(
        "supervised_proof_evaluator_physical_readback_policy"
    ) < result_names.index(
        "supervised_proof_report_gap_rollup_schema"
    )


def test_no_wing_release_check_rejects_forbidden_live_command():
    checks = [
        CommandCheck(
            name="bad_live_runner",
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_supervised_smoke_runner",
            command=("python3", "tools/wing_supervised_smoke_runner.py"),
            cwd=".",
        )
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "forbidden live command tokens" in message
    assert "bad_live_runner:wing_operator_proposal_validation_runner.py" in message
    assert "bad_supervised_smoke_runner:wing_supervised_smoke_runner.py" in message


def test_no_wing_release_check_rejects_current_cycle_forbidden_command_boundary():
    current_cycle_forbidden_tokens = (
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
    )
    missing = [
        token
        for token in current_cycle_forbidden_tokens
        if token not in release_check.FORBIDDEN_COMMAND_TOKENS
    ]
    assert missing == []

    checks = [
        CommandCheck(
            name=f"bad_{token}_runner",
            command=("python3", f"tools/{token}_audit.py"),
            cwd=".",
        )
        for token in current_cycle_forbidden_tokens
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    for token in current_cycle_forbidden_tokens:
        assert f"bad_{token}_runner:{token}" in message


def test_no_wing_release_check_rejects_forbidden_live_command_in_check_name():
    checks = [
        CommandCheck(
            name="planned_set_gain_check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
        CommandCheck(
            name="planned_load_scene_check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "planned_set_gain_check:set_gain" in message
    assert "planned_load_scene_check:load_scene" in message


def test_no_wing_release_check_rejects_forbidden_live_tokens_case_insensitively():
    checks = [
        CommandCheck(
            name="Planned_Set_Gain_Check",
            command=("python3", "tools/offline_safe_wrapper.py"),
            cwd=".",
        ),
        CommandCheck(
            name="mixed_case_wrapper",
            command=("python3", "tools/LOAD_SNAP_PLAN.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "Planned_Set_Gain_Check:set_gain" in message
    assert "mixed_case_wrapper:load_snap" in message


def test_no_wing_release_check_rejects_expanded_forbidden_live_commands():
    checks = [
        CommandCheck(
            name="bad_eq_runner",
            command=("python3", "tools/apply_set_eq_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_compressor_runner",
            command=("python3", "tools/apply_set_compressor_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_runner",
            command=("python3", "tools/reset_eq.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_mute_runner",
            command=("python3", "tools/mute_channels.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_eq_runner:set_eq" in message
    assert "bad_compressor_runner:set_compressor" in message
    assert "bad_reset_runner:reset" in message
    assert "bad_mute_runner:mute" in message


def test_no_wing_release_check_rejects_unmute_as_live_write():
    checks = [
        CommandCheck(
            name="bad_unmute_runner",
            command=("python3", "tools/unmute_channels.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    assert "bad_unmute_runner:unmute" in str(excinfo.value)


def test_no_wing_release_check_rejects_mute_aliases_as_live_writes():
    checks = [
        CommandCheck(
            name="bad_set_mute_runner",
            command=("python3", "tools/apply_set_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_channel_mute_runner",
            command=("python3", "tools/apply_set_channel_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_bus_mute_runner",
            command=("python3", "tools/apply_set_bus_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_set_main_mute_runner",
            command=("python3", "tools/apply_set_main_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_mute_channel_runner",
            command=("python3", "tools/mute_channel_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_unmute_channel_runner",
            command=("python3", "tools/unmute_channel_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_set_mute_runner:set_mute" in message
    assert "bad_set_channel_mute_runner:set_channel_mute" in message
    assert "bad_set_bus_mute_runner:set_bus_mute" in message
    assert "bad_set_main_mute_runner:set_main_mute" in message
    assert "bad_mute_channel_runner:mute_channel" in message
    assert "bad_unmute_channel_runner:unmute_channel" in message


def test_no_wing_release_check_rejects_live_write_method_variants():
    checks = [
        CommandCheck(
            name="bad_channel_gain_runner",
            command=("python3", "tools/proof_set_channel_gain.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_fx_runner",
            command=("python3", "tools/apply_set_fx_model_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_route_runner",
            command=("python3", "tools/route_multiple_outputs_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_input_patch_runner",
            command=("python3", "tools/set_channel_input_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_phase_runner",
            command=("python3", "tools/reset_phase_delay.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_channel_gain_runner:set_channel_gain" in message
    assert "bad_fx_runner:set_fx_model" in message
    assert "bad_route_runner:route_multiple_outputs" in message
    assert "bad_input_patch_runner:set_channel_input" in message
    assert "bad_phase_runner:reset_phase_delay" in message


def test_no_wing_release_check_rejects_group_dca_matrix_balance_aliases():
    checks = [
        CommandCheck(
            name="bad_group_balance_runner",
            command=("python3", "tools/apply_set_group_fader_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_balance_runner",
            command=("python3", "tools/apply_set_matrix_fader_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_dca_level_runner",
            command=("python3", "tools/apply_set_dca_level_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_dca_mute_runner",
            command=("python3", "tools/apply_set_dca_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_group_mute_runner",
            command=("python3", "tools/apply_set_group_mute_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_mute_runner",
            command=("python3", "tools/apply_set_matrix_mute_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_group_balance_runner:set_group_fader" in message
    assert "bad_matrix_balance_runner:set_matrix_fader" in message
    assert "bad_dca_level_runner:set_dca_level" in message
    assert "bad_dca_mute_runner:set_dca_mute" in message
    assert "bad_group_mute_runner:set_group_mute" in message
    assert "bad_matrix_mute_runner:set_matrix_mute" in message


def test_no_wing_release_check_rejects_bus_main_matrix_processing_aliases():
    checks = [
        CommandCheck(
            name="bad_bus_eq_runner",
            command=("python3", "tools/apply_set_bus_eq_on_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_main_eq_runner",
            command=("python3", "tools/apply_set_main_eq_band_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_matrix_eq_runner",
            command=("python3", "tools/apply_set_matrix_eq_band_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_bus_dynamics_runner",
            command=("python3", "tools/apply_set_bus_compressor_gain_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_bus_pan_runner",
            command=("python3", "tools/apply_set_bus_pan_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_main_pan_runner",
            command=("python3", "tools/apply_set_main_pan_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_bus_eq_runner:set_bus_eq_on" in message
    assert "bad_main_eq_runner:set_main_eq_band" in message
    assert "bad_matrix_eq_runner:set_matrix_eq_band" in message
    assert "bad_bus_dynamics_runner:set_bus_compressor_gain" in message
    assert "bad_bus_pan_runner:set_bus_pan" in message
    assert "bad_main_pan_runner:set_main_pan" in message


def test_no_wing_release_check_rejects_fx_return_and_send_aliases():
    checks = [
        CommandCheck(
            name="bad_fx_return_runner",
            command=("python3", "tools/apply_set_fx_return_level_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_channel_main_send_runner",
            command=("python3", "tools/apply_set_channel_main_send_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reverb_send_runner",
            command=("python3", "tools/apply_set_reverb_send_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_fx_return_runner:set_fx_return_level" in message
    assert "bad_channel_main_send_runner:set_channel_main_send" in message
    assert "bad_reverb_send_runner:set_reverb_send" in message


def test_no_wing_release_check_rejects_trim_and_preamp_aliases():
    checks = [
        CommandCheck(
            name="bad_trim_runner",
            command=("python3", "tools/apply_set_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_input_trim_runner",
            command=("python3", "tools/apply_set_input_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_channel_trim_runner",
            command=("python3", "tools/apply_set_channel_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_preamp_trim_runner",
            command=("python3", "tools/apply_set_preamp_trim_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_preamp_gain_runner",
            command=("python3", "tools/apply_set_preamp_gain_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_trim_runner:set_trim" in message
    assert "bad_input_trim_runner:set_input_trim" in message
    assert "bad_channel_trim_runner:set_channel_trim" in message
    assert "bad_preamp_trim_runner:set_preamp_trim" in message
    assert "bad_preamp_gain_runner:set_preamp_gain" in message


def test_no_wing_release_check_rejects_channel_reset_variants():
    checks = [
        CommandCheck(
            name="bad_reset_eq_runner",
            command=("python3", "tools/reset_channel_eq.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_hpf_runner",
            command=("python3", "tools/reset_channel_hpf.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_dynamics_runner",
            command=("python3", "tools/reset_channel_dynamics.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_reset_processing_runner",
            command=("python3", "tools/reset_channel_processing.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_reset_eq_runner:reset_channel_eq" in message
    assert "bad_reset_hpf_runner:reset_channel_hpf" in message
    assert "bad_reset_dynamics_runner:reset_channel_dynamics" in message
    assert "bad_reset_processing_runner:reset_channel_processing" in message


def test_no_wing_release_check_rejects_scene_and_snapshot_aliases():
    checks = [
        CommandCheck(
            name="bad_recall_snap_runner",
            command=("python3", "tools/recall_snap_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_store_snap_runner",
            command=("python3", "tools/store_snap_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_restore_snapshot_runner",
            command=("python3", "tools/restore_snapshot_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_load_scene_runner",
            command=("python3", "tools/load_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_save_scene_runner",
            command=("python3", "tools/save_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_recall_scene_runner",
            command=("python3", "tools/recall_scene_plan.py"),
            cwd=".",
        ),
        CommandCheck(
            name="bad_restore_scene_runner",
            command=("python3", "tools/restore_scene_plan.py"),
            cwd=".",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        validate_no_live_commands(checks)

    message = str(excinfo.value)
    assert "bad_recall_snap_runner:recall_snap" in message
    assert "bad_store_snap_runner:store_snap" in message
    assert "bad_restore_snapshot_runner:restore_snapshot" in message
    assert "bad_load_scene_runner:load_scene" in message
    assert "bad_save_scene_runner:save_scene" in message
    assert "bad_recall_scene_runner:recall_scene" in message
    assert "bad_restore_scene_runner:restore_scene" in message


def test_no_wing_release_check_summary_reports_failed_checks(tmp_path):
    result = CheckResult(
        name="example_check",
        status="failed",
        started_at="2026-05-26T00:00:00Z",
        finished_at="2026-05-26T00:00:01Z",
        duration_sec=1.0,
        returncode=1,
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-05-26T00:00:00Z")

    assert summary["status"] == "failed"
    assert summary["live_write"] is False
    assert summary["failed_checks"] == ["example_check"]
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"
    assert "`example_check`: `failed`" in (tmp_path / "summary.md").read_text()


def test_no_wing_release_check_summary_surfaces_backend_port_owner(tmp_path):
    result = CheckResult(
        name="backend_port_free",
        status="failed",
        started_at="2026-06-12T00:00:00Z",
        finished_at="2026-06-12T00:00:01Z",
        duration_sec=1.0,
        detail={
            "host": "127.0.0.1",
            "port": 8765,
            "policy": "port_must_not_have_unexpected_listener",
            "listener_detected": True,
            "listener_processes": [
                {
                    "pid": 87210,
                    "command": "Python",
                    "name": "TCP *:8765 (LISTEN)",
                    "command_line": "python3 backend/server.py --mode wing",
                }
            ],
            "listener_confirmation_checklist": list(
                release_check.BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST
            ),
            "backend_port_prestart_check_required": True,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
            ),
            "backend_port_prestart_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_gate_execution_allowed": False,
            "backend_port_prestart_gate_readiness_closure_allowed": False,
            "backend_port_prestart_gate_live_authority_granted": False,
            "preflight_blocked_until_listener_confirmed": True,
            "intended_backend_confirmation_required": True,
            "safe_to_stop_listener_by_gate": False,
            "listener_confirmation_template_path": (
                "artifacts/no_wing_release_check/run/backend-listener-confirmation-template.json"
            ),
            "listener_confirmation_template_status": "pending_operator_edit",
            "listener_confirmation_template_acceptance_status": (
                release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_STATUS
            ),
            "listener_confirmation_template_accepted_by_template": False,
            "listener_confirmation_template_execution_allowed_by_template": False,
            "listener_confirmation_template_readiness_closure_allowed_by_template": False,
            "listener_confirmation_template_operator_reviewed_at": (
                "REPLACE_WITH_OPERATOR_REVIEW_UTC"
            ),
            "listener_confirmation_template_operator_name": "REPLACE_WITH_OPERATOR_NAME",
            "listener_confirmation_template_operator_placeholder_fields": [
                "operator_reviewed_at",
                "operator_name",
            ],
            "listener_confirmation_template_candidate_count": 1,
            "listener_confirmation_template_required_edits": list(
                release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_REQUIRED_EDITS
            ),
            "listener_confirmation_template_safe_to_stop_listener_by_gate": False,
            "listener_confirmation_template_live_authority_granted": False,
            "fail_closed_reason": "backend_port_listener_requires_operator_confirmation",
            "operator_next_step": (
                "Stop or reconfigure the unexpected local listener only after "
                "confirming it is not the intended Automixer backend, then rerun "
                "the no-WING gate and WING read-only preflight."
            ),
        },
    )

    write_summary(tmp_path, [result], generated_at="2026-06-12T00:00:00Z")

    markdown_summary = (tmp_path / "summary.md").read_text()
    assert "`backend_port_free`: `failed`" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert '"command": "Python"' in markdown_summary
    assert '"command_line": "python3 backend/server.py --mode wing"' in markdown_summary
    assert '"pid": 87210' in markdown_summary
    assert "listener_confirmation_checklist" in markdown_summary
    assert "do not stop or reconfigure the listener from this no-WING gate" in (
        markdown_summary
    )
    assert "backend_port_prestart_check_required: `True`" in markdown_summary
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: `True`"
        in markdown_summary
    )
    assert "backend_port_prestart_check_command" in markdown_summary
    assert "tools/no_wing_release_check.py" in markdown_summary
    assert "backend_port_prestart_check_failure_detail_fields" in markdown_summary
    assert "backend_port_prestart_gate_execution_allowed: `False`" in markdown_summary
    assert (
        "backend_port_prestart_gate_readiness_closure_allowed: `False`"
        in markdown_summary
    )
    assert (
        "backend_port_prestart_gate_live_authority_granted: `False`"
        in markdown_summary
    )
    assert "listener_confirmation_template_required_edits" in markdown_summary
    assert "preflight_blocked_until_listener_confirmed: `True`" in markdown_summary
    assert "intended_backend_confirmation_required: `True`" in markdown_summary
    assert "safe_to_stop_listener_by_gate: `False`" in markdown_summary
    assert "listener_confirmation_template_path" in markdown_summary
    assert "backend-listener-confirmation-template.json" in markdown_summary
    assert "listener_confirmation_template_status: `pending_operator_edit`" in (
        markdown_summary
    )
    assert "listener_confirmation_template_acceptance_status" in markdown_summary
    assert "operator_review_required" in markdown_summary
    assert (
        "listener_confirmation_template_safe_to_stop_listener_by_gate: `False`"
        in markdown_summary
    )
    assert (
        "listener_confirmation_template_live_authority_granted: `False`"
        in markdown_summary
    )
    assert "listener_confirmation_template_accepted_by_template: `False`" in (
        markdown_summary
    )
    assert (
        "listener_confirmation_template_execution_allowed_by_template: `False`"
        in markdown_summary
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: `False`"
        in markdown_summary
    )
    assert "listener_confirmation_template_operator_reviewed_at" in markdown_summary
    assert "REPLACE_WITH_OPERATOR_REVIEW_UTC" in markdown_summary
    assert "listener_confirmation_template_operator_name" in markdown_summary
    assert "REPLACE_WITH_OPERATOR_NAME" in markdown_summary
    assert "listener_confirmation_template_operator_placeholder_fields" in (
        markdown_summary
    )
    assert "listener_confirmation_template_candidate_count: `1`" in markdown_summary
    assert "listener_confirmation_template_required_edits" in markdown_summary
    assert "rerun the no-WING gate with --backend-listener-confirmation" in (
        markdown_summary
    )
    assert (
        "fail_closed_reason: `backend_port_listener_requires_operator_confirmation`"
        in markdown_summary
    )
    assert "operator_next_step" in markdown_summary
    assert "WING read-only preflight" in markdown_summary


def test_no_wing_release_check_summary_surfaces_confirmed_backend_listener(tmp_path):
    result = CheckResult(
        name="backend_port_free",
        status="passed",
        started_at="2026-06-13T00:00:00Z",
        finished_at="2026-06-13T00:00:01Z",
        duration_sec=1.0,
        detail={
            "host": "127.0.0.1",
            "port": 8765,
            "policy": "port_must_be_free_or_operator_confirmed_intended_backend",
            "listener_detected": True,
            "listener_processes": [
                {
                    "pid": 87210,
                    "command": "Python",
                    "name": "TCP *:8765 (LISTEN)",
                    "command_line": "python3 backend/server.py --mode wing",
                }
            ],
            "listener_confirmation_path": "operator-confirmation.json",
            "listener_confirmation_status": "accepted",
            "intended_backend_confirmed": True,
            "matched_listener_pid": 87210,
            "preflight_blocked_until_listener_confirmed": False,
            "intended_backend_confirmation_required": False,
            "safe_to_stop_listener_by_gate": False,
            "live_authority_granted": False,
            "operator_next_step": (
                "Proceed only with the operator-confirmed intended backend "
                "session and WING read-only preflight; this gate still grants "
                "no live authority."
            ),
        },
    )

    write_summary(tmp_path, [result], generated_at="2026-06-13T00:00:00Z")

    markdown_summary = (tmp_path / "summary.md").read_text()
    assert "`backend_port_free`: `passed`" in markdown_summary
    assert "listener_confirmation_path: `operator-confirmation.json`" in markdown_summary
    assert "listener_confirmation_status: `accepted`" in markdown_summary
    assert "intended_backend_confirmed: `True`" in markdown_summary
    assert "matched_listener_pid: `87210`" in markdown_summary
    assert "preflight_blocked_until_listener_confirmed: `False`" in markdown_summary
    assert "live_authority_granted: `False`" in markdown_summary
    assert "fail_closed_reason" not in markdown_summary


def test_no_wing_release_check_summary_embeds_autonomy_gap_rollup(tmp_path):
    result = CheckResult(
        name="supervised_proof_report_gap_rollup_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "autonomy_gap_rollup_present": True,
            "full_auto_gap_bucket_counts": {
                "read_surface": 14,
                "write_surface": 19,
                "evidence_requirement": 19,
            },
            "evidence_requirement_ids_by_action_family": {
                "gain": list(GAIN_REQUIRED_EVIDENCE_COMPONENTS),
                "bus_group_matrix": [
                    "bus_group_assignment_cache_freshness",
                    "bus_group_fader_readback",
                ],
                "mute": [
                    "current_mute_cache_freshness",
                    "mute_owner_intent_evidence",
                ],
            },
            "next_required_evidence_by_action_family": {
                "bus_group_matrix": ["Add bus/group/matrix safety review before promotion."],
                "mute": ["Add fresh mute-state intent before promotion."],
            },
            "supervised_current_live_write_kinds": ["fader", "gain", "pan"],
            "policy_supervised_live_write_kinds": ["fader", "gain", "pan"],
            "supervised_current_live_write_surfaces": ["preamp", "trim", "fader", "pan"],
            "policy_supervised_live_write_surfaces": ["preamp", "trim", "fader", "pan"],
            "unexpected_supervised_live_write_surfaces": [],
            "unexpected_supervised_live_write_kinds": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert autonomy_gap_rollup_summary([result]) == summary["autonomy_gap_rollup"]
    assert json_summary["autonomy_gap_rollup"]["full_auto_gap_bucket_counts"] == {
        "read_surface": 14,
        "write_surface": 19,
        "evidence_requirement": 19,
    }
    assert json_summary["autonomy_gap_rollup"]["evidence_requirement_ids_by_action_family"][
        "gain"
    ] == GAIN_REQUIRED_EVIDENCE_COMPONENTS
    assert json_summary["autonomy_gap_rollup"]["evidence_requirement_ids_by_action_family"][
        "bus_group_matrix"
    ] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
    ]
    assert json_summary["autonomy_gap_rollup"]["next_required_evidence_by_action_family"][
        "mute"
    ] == ["Add fresh mute-state intent before promotion."]
    assert json_summary["autonomy_gap_rollup"]["live_write"] is False
    assert json_summary["autonomy_gap_rollup"]["full_autonomous_ready"] is False
    assert "## Autonomy Gap Rollup" in markdown_summary
    assert "full_auto_gap_bucket_counts" in markdown_summary
    assert '"read_surface": 14' in markdown_summary
    assert "evidence_requirement_ids_by_action_family" in markdown_summary
    assert "bounded_gain_down_apply" in markdown_summary
    assert "bus_group_assignment_cache_freshness" in markdown_summary
    assert "next_required_evidence_by_action_family" in markdown_summary
    assert (
        'supervised_current_live_write_kinds: `["fader", "gain", "pan"]`'
        in markdown_summary
    )
    assert (
        'supervised_current_live_write_surfaces: `["preamp", "trim", "fader", "pan"]`'
        in markdown_summary
    )
    assert (
        'policy_supervised_live_write_surfaces: `["preamp", "trim", "fader", "pan"]`'
        in markdown_summary
    )


def test_no_wing_release_check_summary_embeds_gain_dry_run_proof(tmp_path):
    result = CheckResult(
        name="supervised_proof_report_gain_dry_run_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "gain_dry_run_row_count": 1,
            "dry_run_ready_proof_ids": ["gain-proof-dry-run"],
            "value_type_counts": {"gain": 1},
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert supervised_gain_dry_run_summary([result]) == summary["supervised_gain_dry_run"]
    assert json_summary["supervised_gain_dry_run"]["live_write"] is False
    assert json_summary["supervised_gain_dry_run"]["promotion_allowed"] is False
    assert json_summary["supervised_gain_dry_run"]["autonomous_ready"] is False
    assert json_summary["supervised_gain_dry_run"]["gain_dry_run_row_count"] == 1
    assert "## Supervised Gain Dry-Run Proof" in markdown_summary
    assert 'dry_run_ready_proof_ids: `["gain-proof-dry-run"]`' in markdown_summary


def test_no_wing_release_check_summary_embeds_autonomy_evidence_overlay(tmp_path):
    result = CheckResult(
        name="autonomy_evidence_overlay_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "overlay_status": "dry_run_only",
            "row_count": 3,
            "status_counts": {
                "dry_run_only": 1,
                "manual_safety_review_candidate": 0,
                "no_current_evidence": 2,
                "remediation_required": 0,
            },
            "manual_safety_review_candidate_target_ids": [],
            "remediation_required_target_ids": [],
            "dry_run_only_target_ids": ["gain_trim_supervised_promotion"],
            "no_current_evidence_target_ids": [
                "fader_supervised_promotion",
                "pan_phase_supervised_promotion",
            ],
            "global_blockers": [
                "overlay_does_not_modify_safety_rows",
                "manual_safety_review_required_before_promotion",
                "full_autonomous_mode_remains_blocked",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert autonomy_evidence_overlay_summary([result]) == summary["autonomy_evidence_overlay"]
    assert json_summary["autonomy_evidence_overlay"]["live_write"] is False
    assert json_summary["autonomy_evidence_overlay"]["live_widening_allowed"] is False
    assert json_summary["autonomy_evidence_overlay"]["automatic_safety_update_allowed"] is False
    assert json_summary["autonomy_evidence_overlay"]["full_autonomous_ready"] is False
    assert json_summary["autonomy_evidence_overlay"]["operator_required"] is True
    assert json_summary["autonomy_evidence_overlay"]["dry_run_only_target_ids"] == [
        "gain_trim_supervised_promotion"
    ]
    assert "## Autonomy Evidence Overlay" in markdown_summary
    assert 'dry_run_only_target_ids: `["gain_trim_supervised_promotion"]`' in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_fixture_evidence(tmp_path):
    result = CheckResult(
        name="read_surface_fixture_evidence_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "snapshot_status": "fixture_evidence_incomplete",
            "surface_count": 10,
            "surface_with_fixture_count": 10,
            "declared_fixture_path_count": 10,
            "declared_test_name_count": 25,
            "status_counts": {"partial_fixture_evidence": 10},
            "physical_fixture_missing_surface_ids": ["channel_identity_and_basic_controls"],
            "next_physical_readback_surface_id": "channel_identity_and_basic_controls",
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_fixture_evidence_summary([result]) == summary["read_surface_fixture_evidence"]
    assert json_summary["read_surface_fixture_evidence"]["live_write"] is False
    assert json_summary["read_surface_fixture_evidence"]["full_autonomous_ready"] is False
    assert json_summary["read_surface_fixture_evidence"]["surface_count"] == 10
    assert json_summary["read_surface_fixture_evidence"]["declared_fixture_path_count"] == 10
    assert json_summary["read_surface_fixture_evidence"]["declared_test_name_count"] == 25
    assert json_summary["read_surface_fixture_evidence"]["physical_gap_surface_count"] == 1
    assert json_summary["read_surface_fixture_evidence"]["summary_recomputed_from_rows"] is True
    assert json_summary["read_surface_fixture_evidence"]["fixture_declaration_validation"] is True
    assert json_summary["read_surface_fixture_evidence"]["contract_gap_consistency_enforced"] is True
    assert json_summary["read_surface_fixture_evidence"]["physical_fixture_missing_surface_ids"] == [
        "channel_identity_and_basic_controls"
    ]
    assert (
        json_summary["read_surface_fixture_evidence"]["next_physical_readback_surface_id"]
        == "channel_identity_and_basic_controls"
    )
    assert "## Read Surface Fixture Evidence" in markdown_summary
    assert "declared_fixture_path_count" in markdown_summary
    assert "contract_gap_consistency_enforced: `true`" in markdown_summary
    assert "physical_fixture_missing_surface_ids" in markdown_summary
    assert "next_physical_readback_surface_id" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_remediation_plan(tmp_path):
    result = CheckResult(
        name="read_surface_remediation_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "read_remediation_required",
            "action_count": 10,
            "incomplete_surface_count": 10,
            "physical_fixture_missing_surface_ids": ["raw_mixer_state_cache"],
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "next_physical_readback_target_action_ids": [
                "remediate_raw_mixer_state_cache"
            ],
            "expected_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "expected_next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_remediation_plan_summary([result]) == summary["read_surface_remediation_plan"]
    assert json_summary["read_surface_remediation_plan"]["read_only"] is True
    assert json_summary["read_surface_remediation_plan"]["live_write"] is False
    assert json_summary["read_surface_remediation_plan"]["mutates_console_state"] is False
    assert json_summary["read_surface_remediation_plan"]["full_autonomous_ready"] is False
    assert json_summary["read_surface_remediation_plan"]["physical_gap_surface_count"] == 1
    assert (
        json_summary["read_surface_remediation_plan"]["next_physical_readback_action_id"]
        == "remediate_raw_mixer_state_cache"
    )
    assert json_summary["read_surface_remediation_plan"][
        "next_physical_readback_target_action_ids"
    ] == ["remediate_raw_mixer_state_cache"]
    assert json_summary["read_surface_remediation_plan"]["target_action_consistency"] is True
    assert "## Read Surface Remediation Plan" in markdown_summary
    assert "next_physical_readback_action_id" in markdown_summary
    assert "next_physical_readback_target_action_ids" in markdown_summary
    assert "target_action_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_capture_plan(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_capture_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "operator_capture_required",
            "capture_step_count": 1,
            "physical_fixture_missing_surface_ids": ["raw_mixer_state_cache"],
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "expected_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "expected_next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "target_action_consistency": True,
            "required_artifact_contracts": [
                "physical_readback_fixture",
                "raw_mixer_state_physical_readback_contract",
            ],
            "blocked_by": [
                "physical_readback_fixtures_incomplete",
                "operator_must_capture_and_review_physical_readback_artifact",
            ],
            "step_surface_id": "raw_mixer_state_cache",
            "step_remediation_action_id": "remediate_raw_mixer_state_cache",
            "step_forbidden_actions": [
                "do not apply fader, gain, trim, pan, mute, EQ, dynamics, gate, routing, DCA, matrix, FX, scene, or preset changes",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert (
        read_surface_physical_readback_capture_plan_summary([result])
        == summary["read_surface_physical_readback_capture_plan"]
    )
    capture_plan = json_summary["read_surface_physical_readback_capture_plan"]
    assert capture_plan["read_only"] is True
    assert capture_plan["live_write"] is False
    assert capture_plan["queries_console"] is False
    assert capture_plan["executes_tests"] is False
    assert capture_plan["operator_required"] is True
    assert capture_plan["execution_allowed_by_plan"] is False
    assert capture_plan["live_authority_granted"] is False
    assert capture_plan["full_autonomous_ready"] is False
    assert capture_plan["capture_step_count"] == 1
    assert capture_plan["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert capture_plan["target_action_consistency"] is True
    assert "physical_readback_fixture" in capture_plan["required_artifact_contracts"]
    assert "operator_must_capture_and_review_physical_readback_artifact" in capture_plan["blocked_by"]
    assert "## Read Surface Physical Readback Capture Plan" in markdown_summary
    assert "execution_allowed_by_plan: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "step_remediation_action_id" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_capture_packet_artifact(
    tmp_path,
):
    result = CheckResult(
        name="read_surface_physical_readback_capture_packet_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "packet_path": "/tmp/read_surface_physical_readback_capture_packet.json",
            "packet_type": "read_surface_physical_readback_capture_packet",
            "packet_status": "operator_capture_required",
            "source_plan_status": "operator_capture_required",
            "source_method_id": "read_surface_physical_readback_capture_plan",
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_packet": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "physical_fixture_missing_surface_ids": ["raw_mixer_state_cache"],
            "physical_gap_surface_count": 1,
            "remaining_capture_target_count": 1,
            "remaining_capture_target_surface_ids": ["raw_mixer_state_cache"],
            "remaining_capture_target_action_ids": ["remediate_raw_mixer_state_cache"],
            "remaining_capture_target_contract_ids": [
                "raw_mixer_state_physical_readback_contract",
            ],
            "remaining_capture_target_forbidden_flags": [],
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "target_action_consistency": True,
            "capture_step_count": 1,
            "required_artifact_contracts": [
                "physical_readback_fixture",
                "raw_mixer_state_physical_readback_contract",
            ],
            "blocked_by": [
                "operator_must_capture_and_review_physical_readback_artifact",
            ],
            "step_surface_id": "raw_mixer_state_cache",
            "step_remediation_action_id": "remediate_raw_mixer_state_cache",
            "step_target_console_surface": "all_cached_addresses",
            "step_read_mode": "cached_state",
            "step_contract_method_id": "raw_mixer_state_physical_readback_contract",
            "step_expected_artifact_count": 2,
            "step_blocker_count": 2,
            "step_forbidden_actions": [
                "do not transmit OSC, MIDI, TCP, or websocket mixer-control messages",
            ],
            "operator_sequence_count": 5,
            "focused_validation": [
                "tests/test_read_surface_physical_readback_capture_plan.py",
                "tests/test_no_wing_release_check.py",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_capture_packet_artifact_summary(
        [result]
    ) == summary["read_surface_physical_readback_capture_packet_artifact"]
    artifact = json_summary[
        "read_surface_physical_readback_capture_packet_artifact"
    ]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["operator_required"] is True
    assert artifact["execution_allowed_by_packet"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["packet_status"] == "operator_capture_required"
    assert artifact["remaining_capture_target_count"] == 1
    assert artifact["remaining_capture_target_surface_ids"] == ["raw_mixer_state_cache"]
    assert artifact["remaining_capture_target_action_ids"] == [
        "remediate_raw_mixer_state_cache"
    ]
    assert artifact["remaining_capture_target_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract"
    ]
    assert artifact["remaining_capture_target_forbidden_flags"] == []
    assert artifact["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert artifact["target_action_consistency"] is True
    assert "physical_readback_fixture" in artifact["required_artifact_contracts"]
    assert "tests/test_no_wing_release_check.py" in artifact["focused_validation"]
    assert artifact["violations"] == []
    assert (
        "## Read Surface Physical Readback Capture Packet Artifact"
        in markdown_summary
    )
    assert "execution_allowed_by_packet: `false`" in markdown_summary
    assert "remaining_capture_target_surface_ids" in markdown_summary
    assert "all_cached_addresses" in markdown_summary
    assert "do not transmit OSC" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_capture_template_artifact(
    tmp_path,
):
    result = CheckResult(
        name="read_surface_physical_readback_capture_template_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "template_path": "/tmp/read_surface_physical_readback_capture_template.json",
            "template_type": "read_surface_physical_readback_artifact_capture_template",
            "template_status": "operator_capture_template_ready",
            "source_packet_type": "read_surface_physical_readback_capture_packet",
            "source_packet_status": "operator_capture_required",
            "source_method_id": "read_surface_physical_readback_capture_plan",
            "selected_surface_id": "raw_mixer_state_cache",
            "include_all_targets": False,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_template": False,
            "accepted_as_physical_fixture_candidate": False,
            "fixture_closure_allowed_by_template": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "all_remaining_target_count": 2,
            "capture_template_count": 1,
            "capture_template_surface_ids": ["raw_mixer_state_cache"],
            "capture_template_contract_ids": [
                "raw_mixer_state_physical_readback_contract",
            ],
            "payload_template_contract_ids": [
                "raw_mixer_state_physical_readback_contract",
            ],
            "payload_template_artifact_types": [
                "raw_mixer_state_physical_readback",
            ],
            "payload_template_schema_ids": [
                "raw_mixer_state_physical_readback.v1",
            ],
            "payload_template_field_counts": [10],
            "required_operator_edits": [
                "artifact_source_id",
                "captured_at",
                "captured_by",
                "source",
                "console_model",
                "capture_scope",
                "physical_readback_payload",
                "physical_readback_payload_template.payload",
            ],
            "operator_sequence_count": 6,
            "blocked_by": [
                "operator_must_replace_capture_template_placeholders",
            ],
            "record_unsafe_flags": [],
            "focused_validation": [
                "tests/test_read_surface_physical_readback_capture_template.py",
                "tests/test_no_wing_release_check.py",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_capture_template_artifact_summary(
        [result]
    ) == summary["read_surface_physical_readback_capture_template_artifact"]
    artifact = json_summary[
        "read_surface_physical_readback_capture_template_artifact"
    ]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["operator_required"] is True
    assert artifact["execution_allowed_by_template"] is False
    assert artifact["accepted_as_physical_fixture_candidate"] is False
    assert artifact["fixture_closure_allowed_by_template"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["template_status"] == "operator_capture_template_ready"
    assert artifact["selected_surface_id"] == "raw_mixer_state_cache"
    assert artifact["capture_template_count"] == 1
    assert artifact["capture_template_surface_ids"] == ["raw_mixer_state_cache"]
    assert artifact["capture_template_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract"
    ]
    assert artifact["payload_template_contract_ids"] == [
        "raw_mixer_state_physical_readback_contract"
    ]
    assert artifact["payload_template_artifact_types"] == [
        "raw_mixer_state_physical_readback"
    ]
    assert artifact["payload_template_schema_ids"] == [
        "raw_mixer_state_physical_readback.v1"
    ]
    assert artifact["payload_template_field_counts"] == [10]
    assert "physical_readback_payload" in artifact["required_operator_edits"]
    assert (
        "physical_readback_payload_template.payload"
        in artifact["required_operator_edits"]
    )
    assert artifact["record_unsafe_flags"] == []
    assert artifact["violations"] == []
    assert (
        "## Read Surface Physical Readback Capture Template Artifact"
        in markdown_summary
    )
    assert "execution_allowed_by_template: `false`" in markdown_summary
    assert "capture_template_surface_ids" in markdown_summary
    assert "payload_template_artifact_types" in markdown_summary
    assert "raw_mixer_state_physical_readback" in markdown_summary
    assert "physical_readback_payload" in markdown_summary


def test_no_wing_release_check_summary_embeds_wing_readonly_preflight_evidence(
    tmp_path,
):
    expected_target_defaults = {
        "url": WING_READONLY_PREFLIGHT_RUNNER_URL,
        "channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "value_types": WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "artifact_root": WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    }
    result = CheckResult(
        name="wing_readonly_preflight_evidence_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "evidence_status": "operator_preflight_required",
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "accepted_as_readonly_preflight_candidate": False,
            "readiness_matrix_update_allowed": False,
            "readiness_closure_allowed_by_evidence": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "mvp_blocker_candidate_count": 0,
            "mvp_blockers_closed_by_evidence": [],
            "mvp_blocker_matrix_ids": [
                "wing_readonly_connection_proof",
                "physical_console_readback_proof",
                "operator_visible_evidence_report",
                "readiness_matrix_update_based_on_real_proof",
            ],
            "latest_valid_candidate_run_id": None,
            "artifact_operator_next_steps": [
                "Regenerate the offline expected plan before opening the backend/WING WebSocket."
            ],
            "capture_tool_path": "tools/wing_readonly_preflight_runner.py",
            "capture_tool_available": True,
            "plan_command": [
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
            ],
            "plan_command_uses_plan_only": True,
            "expected_target_defaults": expected_target_defaults,
            "runner_argparse_target_defaults": expected_target_defaults,
            "runner_argparse_target_defaults_match_contract": True,
            "expected_supported_value_types": sorted(
                WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
            ),
            "runner_supported_value_types": sorted(
                WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
            ),
            "runner_supported_value_types_match_contract": True,
            "expected_proof_runner": WING_READONLY_PREFLIGHT_PROOF_RUNNER,
            "runner_proof_runner": WING_READONLY_PREFLIGHT_PROOF_RUNNER,
            "evidence_proof_runner": WING_READONLY_PREFLIGHT_PROOF_RUNNER,
            "runner_proof_runner_match_contract": True,
            "evidence_proof_runner_match_contract": True,
            "expected_run_summary_status": WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
            "expected_run_summary_reason": WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
            "runner_run_summary_status": WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS,
            "runner_run_summary_reason": WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON,
            "evidence_expected_run_summary_status": (
                WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
            ),
            "evidence_expected_run_summary_reason": (
                WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
            ),
            "runner_run_summary_status_match_contract": True,
            "runner_run_summary_reason_match_contract": True,
            "evidence_run_summary_status_match_contract": True,
            "evidence_run_summary_reason_match_contract": True,
            "expected_physical_readback_statuses": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
            ),
            "expected_physical_readback_sources": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
            ),
            "runner_physical_readback_statuses": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
            ),
            "runner_physical_readback_sources": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
            ),
            "runner_physical_readback_statuses_match_contract": True,
            "runner_physical_readback_sources_match_contract": True,
            "evidence_physical_readback_statuses": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
            ),
            "evidence_physical_readback_sources": sorted(
                WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
            ),
            "evidence_physical_readback_statuses_match_contract": True,
            "evidence_physical_readback_sources_match_contract": True,
            "plan_command_target_defaults": expected_target_defaults,
            "plan_command_target_defaults_match_contract": True,
            "plan_command_output": WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
            "plan_command_output_matches_contract": True,
            "plan_command_channels": "3",
            "plan_command_operator_confirmed_channels": "3",
            "plan_command_confirms_session_ready": True,
            "plan_command_value_types": "fader,gain,pan",
            "plan_command_artifact_root": "artifacts/wing_readonly_preflight",
            "plan_command_prestart_summary": (
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
            ),
            "plan_command_matches_capture": True,
            "prestart_backend_port_check_required": True,
            "prestart_backend_port_check_must_pass_before_starting_backend": True,
            "prestart_backend_port_check_command": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "prestart_backend_port_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "capture_command_template": [
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
            ],
            "capture_command_url": WING_READONLY_PREFLIGHT_RUNNER_URL,
            "capture_command_target_defaults": expected_target_defaults,
            "capture_command_target_defaults_match_contract": True,
            "capture_command_channels": "3",
            "capture_command_operator_confirmed_channels": "3",
            "capture_command_requires_operator_channel_confirmation": True,
            "capture_command_operator_confirmation_matches_channels": True,
            "capture_command_confirms_session_ready": True,
            "capture_command_requires_physical_readback": True,
            "capture_command_prestart_summary": (
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
            ),
            "capture_command_prestart_summary_matches_plan": True,
            "capture_command_expected_plan": WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
            "capture_command_expected_plan_matches_contract": True,
            "capture_command_expected_plan_matches_plan_output": True,
            "review_tool_path": "tools/review_wing_readonly_preflight_artifacts.py",
            "review_tool_available": True,
            "review_command": [
                "python3",
                "tools/review_wing_readonly_preflight_artifacts.py",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--output",
                WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
                "--decision-template-output",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
            ],
            "review_command_writes_decision_template": True,
            "review_decision_template_output": (
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
            ),
            "review_decision_template_output_matches_contract": True,
            "review_exit_codes": {
                "candidate_ready": 0,
                "artifact_missing": 2,
                "invalid_artifact": 3,
            },
            "review_manual_checklist_ids": [
                "confirm_artifact_paths",
                "confirm_operator_channel_scope",
                "confirm_operator_session_ready",
                "confirm_physical_readbacks",
                "confirm_event_log_readonly_sequence",
                "confirm_no_live_authority",
            ],
            "review_manual_checklist_complete": True,
            "review_decision_template_required_fields": [
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
            ],
            "review_decision_template_contract_complete": True,
            "review_decision_template_status": (
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
            ),
            "review_decision_template_status_matches_contract": True,
            "handoff_packet_tool_path": (
                "tools/build_wing_readonly_preflight_handoff_packet.py"
            ),
            "handoff_packet_tool_available": True,
            "handoff_packet_command": [
                "python3",
                "tools/build_wing_readonly_preflight_handoff_packet.py",
                "--url",
                "ws://localhost:8765",
                "--channels",
                "3",
                "--value-types",
                "fader,gain,pan",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--plan-output",
                WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
                "--prestart-summary",
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
                "--output",
                WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
            ],
            "handoff_packet_output": (
                WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
            ),
            "handoff_packet_output_matches_contract": True,
            "handoff_packet_plan_output": (
                WING_READONLY_PREFLIGHT_PLAN_OUTPUT
            ),
            "handoff_packet_plan_output_matches_contract": True,
            "handoff_packet_plan_output_matches_plan_command": True,
            "handoff_packet_url": "ws://localhost:8765",
            "handoff_packet_target_defaults": expected_target_defaults,
            "handoff_packet_target_defaults_match_contract": True,
            "handoff_packet_artifact_root": "artifacts/wing_readonly_preflight",
            "handoff_packet_prestart_summary": (
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
            ),
            "handoff_packet_channels": "3",
            "handoff_packet_value_types": "fader,gain,pan",
            "handoff_packet_url_matches_capture": True,
            "handoff_packet_artifact_root_matches_capture": True,
            "handoff_packet_prestart_summary_matches_capture": True,
            "handoff_packet_channel_matches_capture": True,
            "handoff_packet_value_types_match_capture": True,
            "blocked_by": [
                "operator_must_run_wing_readonly_preflight_runner"
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert (
        wing_readonly_preflight_evidence_summary([result])
        == summary["wing_readonly_preflight_evidence"]
    )
    preflight = json_summary["wing_readonly_preflight_evidence"]
    assert preflight["read_only"] is True
    assert preflight["live_write"] is False
    assert preflight["queries_console"] is False
    assert preflight["executes_tests"] is False
    assert preflight["operator_required"] is True
    assert preflight["execution_allowed_by_evidence"] is False
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["mvp_blocker_matrix_ids"] == [
        "wing_readonly_connection_proof",
        "physical_console_readback_proof",
        "operator_visible_evidence_report",
        "readiness_matrix_update_based_on_real_proof",
    ]
    assert preflight["capture_tool_path"] == "tools/wing_readonly_preflight_runner.py"
    assert preflight["capture_tool_available"] is True
    assert preflight["plan_command"][1] == "tools/wing_readonly_preflight_runner.py"
    assert "--plan-only" in preflight["plan_command"]
    assert preflight["plan_command_uses_plan_only"] is True
    assert preflight["expected_target_defaults"] == expected_target_defaults
    assert preflight["runner_argparse_target_defaults"] == expected_target_defaults
    assert preflight["runner_argparse_target_defaults_match_contract"] is True
    assert preflight["expected_supported_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert preflight["runner_supported_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES
    )
    assert preflight["runner_supported_value_types_match_contract"] is True
    assert preflight["expected_proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert preflight["runner_proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert preflight["evidence_proof_runner"] == WING_READONLY_PREFLIGHT_PROOF_RUNNER
    assert preflight["runner_proof_runner_match_contract"] is True
    assert preflight["evidence_proof_runner_match_contract"] is True
    assert preflight["expected_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert preflight["expected_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert preflight["runner_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert preflight["runner_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert preflight["evidence_expected_run_summary_status"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_STATUS
    )
    assert preflight["evidence_expected_run_summary_reason"] == (
        WING_READONLY_PREFLIGHT_RUN_SUMMARY_REASON
    )
    assert preflight["runner_run_summary_status_match_contract"] is True
    assert preflight["runner_run_summary_reason_match_contract"] is True
    assert preflight["evidence_run_summary_status_match_contract"] is True
    assert preflight["evidence_run_summary_reason_match_contract"] is True
    assert preflight["expected_physical_readback_statuses"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert preflight["expected_physical_readback_sources"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )
    assert preflight["runner_physical_readback_statuses"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert preflight["runner_physical_readback_sources"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )
    assert preflight["runner_physical_readback_statuses_match_contract"] is True
    assert preflight["runner_physical_readback_sources_match_contract"] is True
    assert preflight["evidence_physical_readback_statuses"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_STATUSES
    )
    assert preflight["evidence_physical_readback_sources"] == sorted(
        WING_READONLY_PREFLIGHT_PHYSICAL_READBACK_SOURCES
    )
    assert preflight["evidence_physical_readback_statuses_match_contract"] is True
    assert preflight["evidence_physical_readback_sources_match_contract"] is True
    assert preflight["plan_command_target_defaults"] == expected_target_defaults
    assert preflight["plan_command_target_defaults_match_contract"] is True
    assert preflight["plan_command_output"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert preflight["plan_command_output_matches_contract"] is True
    assert preflight["plan_command_channels"] == "3"
    assert preflight["plan_command_operator_confirmed_channels"] == "3"
    assert preflight["plan_command_confirms_session_ready"] is True
    assert preflight["plan_command_value_types"] == "fader,gain,pan"
    assert preflight["plan_command_artifact_root"] == "artifacts/wing_readonly_preflight"
    assert preflight["plan_command_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert preflight["plan_command_matches_capture"] is True
    assert preflight["prestart_backend_port_check_required"] is True
    assert preflight["prestart_backend_port_check_must_pass_before_starting_backend"] is True
    assert preflight["prestart_backend_port_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert preflight["prestart_backend_port_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert preflight["capture_command_template"][1] == "tools/wing_readonly_preflight_runner.py"
    assert preflight["capture_command_url"] == WING_READONLY_PREFLIGHT_RUNNER_URL
    assert preflight["capture_command_target_defaults"] == expected_target_defaults
    assert preflight["capture_command_target_defaults_match_contract"] is True
    assert preflight["capture_command_channels"] == "3"
    assert preflight["capture_command_operator_confirmed_channels"] == "3"
    assert "--operator-confirm-channels" in preflight["capture_command_template"]
    assert "--operator-confirm-session-ready" in preflight["capture_command_template"]
    assert "--require-physical-readback" in preflight["capture_command_template"]
    assert "--prestart-summary" in preflight["capture_command_template"]
    assert "--expected-plan" in preflight["capture_command_template"]
    assert preflight["capture_command_requires_operator_channel_confirmation"] is True
    assert preflight["capture_command_operator_confirmation_matches_channels"] is True
    assert preflight["capture_command_confirms_session_ready"] is True
    assert preflight["capture_command_requires_physical_readback"] is True
    assert preflight["capture_command_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert preflight["capture_command_prestart_summary_matches_plan"] is True
    assert preflight["capture_command_expected_plan"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert preflight["capture_command_expected_plan_matches_contract"] is True
    assert preflight["capture_command_expected_plan_matches_plan_output"] is True
    assert preflight["review_tool_path"] == (
        "tools/review_wing_readonly_preflight_artifacts.py"
    )
    assert preflight["review_tool_available"] is True
    assert preflight["review_command"][1] == (
        "tools/review_wing_readonly_preflight_artifacts.py"
    )
    assert "--decision-template-output" in preflight["review_command"]
    assert preflight["review_command_writes_decision_template"] is True
    assert preflight["review_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert preflight["review_decision_template_output_matches_contract"] is True
    assert preflight["review_exit_codes"] == {
        "candidate_ready": 0,
        "artifact_missing": 2,
        "invalid_artifact": 3,
    }
    assert preflight["review_manual_checklist_ids"] == [
        "confirm_artifact_paths",
        "confirm_operator_channel_scope",
        "confirm_operator_session_ready",
        "confirm_physical_readbacks",
        "confirm_event_log_readonly_sequence",
        "confirm_no_live_authority",
    ]
    assert preflight["review_manual_checklist_complete"] is True
    assert preflight["review_decision_template_required_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert preflight["review_decision_template_contract_complete"] is True
    assert preflight["review_decision_template_status"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert preflight["review_decision_template_status_matches_contract"] is True
    assert preflight["handoff_packet_tool_path"] == (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
    )
    assert preflight["handoff_packet_tool_available"] is True
    assert preflight["handoff_packet_command"][1] == (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
    )
    assert preflight["handoff_packet_output"] == (
        WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
    )
    assert preflight["handoff_packet_output_matches_contract"] is True
    assert preflight["handoff_packet_plan_output"] == (
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT
    )
    assert preflight["handoff_packet_plan_output_matches_contract"] is True
    assert preflight["handoff_packet_plan_output_matches_plan_command"] is True
    assert preflight["handoff_packet_url"] == "ws://localhost:8765"
    assert preflight["handoff_packet_target_defaults"] == expected_target_defaults
    assert preflight["handoff_packet_target_defaults_match_contract"] is True
    assert preflight["handoff_packet_artifact_root"] == (
        "artifacts/wing_readonly_preflight"
    )
    assert preflight["handoff_packet_prestart_summary"] == (
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
    )
    assert preflight["handoff_packet_channels"] == "3"
    assert preflight["handoff_packet_value_types"] == "fader,gain,pan"
    assert preflight["handoff_packet_url_matches_capture"] is True
    assert preflight["handoff_packet_artifact_root_matches_capture"] is True
    assert preflight["handoff_packet_prestart_summary_matches_capture"] is True
    assert preflight["handoff_packet_channel_matches_capture"] is True
    assert preflight["handoff_packet_value_types_match_capture"] is True
    assert preflight["artifact_operator_next_steps"] == [
        "Regenerate the offline expected plan before opening the backend/WING WebSocket."
    ]
    assert "operator_must_run_wing_readonly_preflight_runner" in preflight["blocked_by"]
    assert "## WING Read-Only Preflight Evidence" in markdown_summary
    assert "readiness_closure_allowed_by_evidence: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "readiness_update_candidate_available: `false`" in markdown_summary
    assert "artifact_operator_next_steps" in markdown_summary
    assert "Regenerate the offline expected plan" in markdown_summary
    assert "capture_tool_path: `tools/wing_readonly_preflight_runner.py`" in markdown_summary
    assert "plan_command_uses_plan_only: `true`" in markdown_summary
    assert "expected_target_defaults" in markdown_summary
    assert "runner_argparse_target_defaults_match_contract: `true`" in markdown_summary
    assert "runner_supported_value_types_match_contract: `true`" in markdown_summary
    assert "runner_proof_runner_match_contract: `true`" in markdown_summary
    assert "evidence_proof_runner_match_contract: `true`" in markdown_summary
    assert "runner_run_summary_status_match_contract: `true`" in markdown_summary
    assert "runner_run_summary_reason_match_contract: `true`" in markdown_summary
    assert "evidence_run_summary_status_match_contract: `true`" in markdown_summary
    assert "evidence_run_summary_reason_match_contract: `true`" in markdown_summary
    assert "runner_physical_readback_statuses_match_contract: `true`" in markdown_summary
    assert "runner_physical_readback_sources_match_contract: `true`" in markdown_summary
    assert "evidence_physical_readback_statuses_match_contract: `true`" in markdown_summary
    assert "evidence_physical_readback_sources_match_contract: `true`" in markdown_summary
    assert "plan_command_target_defaults_match_contract: `true`" in markdown_summary
    assert "wing_readonly_preflight_plan/latest_plan.json" in markdown_summary
    assert "plan_command_output_matches_contract: `true`" in markdown_summary
    assert "plan_command_matches_capture: `true`" in markdown_summary
    assert "prestart_backend_port_check_required: `true`" in markdown_summary
    assert (
        "prestart_backend_port_check_must_pass_before_starting_backend: `true`"
        in markdown_summary
    )
    assert "prestart_backend_port_check_command" in markdown_summary
    assert "prestart_backend_port_check_failure_detail_fields" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert "tools/no_wing_release_check.py" in markdown_summary
    assert "capture_command_url: `ws://localhost:8765`" in markdown_summary
    assert "capture_command_target_defaults_match_contract: `true`" in markdown_summary
    assert "capture_command_channels: `3`" in markdown_summary
    assert "capture_command_operator_confirmed_channels: `3`" in markdown_summary
    assert "capture_command_operator_confirmation_matches_channels: `true`" in markdown_summary
    assert "--operator-confirm-channels" in markdown_summary
    assert "--operator-confirm-session-ready" in markdown_summary
    assert "--require-physical-readback" in markdown_summary
    assert "--prestart-summary" in markdown_summary
    assert "capture_command_prestart_summary_matches_plan: `true`" in markdown_summary
    assert (
        "capture_command_expected_plan_matches_contract: `true`"
        in markdown_summary
    )
    assert (
        "review_tool_path: `tools/review_wing_readonly_preflight_artifacts.py`"
        in markdown_summary
    )
    assert "WING_READONLY_PREFLIGHT_REVIEW_LATEST.md" in markdown_summary
    assert "pending_review_decision_template.json" in markdown_summary
    assert '"invalid_artifact": 3' in markdown_summary
    assert "review_command_writes_decision_template: `true`" in markdown_summary
    assert "review_decision_template_output:" in markdown_summary
    assert (
        "review_decision_template_output_matches_contract: `true`"
        in markdown_summary
    )
    assert "review_manual_checklist_complete: `true`" in markdown_summary
    assert "confirm_event_log_readonly_sequence" in markdown_summary
    assert "review_decision_template_contract_complete: `true`" in markdown_summary
    assert (
        "review_decision_template_status: "
        f"`{WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS}`"
    ) in markdown_summary
    assert (
        "review_decision_template_status_matches_contract: `true`"
        in markdown_summary
    )
    assert "artifact_source_id" in markdown_summary
    assert (
        "handoff_packet_tool_path: `tools/build_wing_readonly_preflight_handoff_packet.py`"
        in markdown_summary
    )
    assert "WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md" in markdown_summary
    assert "handoff_packet_plan_output" in markdown_summary
    assert "handoff_packet_output_matches_contract: `true`" in markdown_summary
    assert "handoff_packet_plan_output_matches_contract: `true`" in markdown_summary
    assert "handoff_packet_target_defaults_match_contract: `true`" in markdown_summary
    assert (
        "handoff_packet_plan_output_matches_plan_command: `true`"
        in markdown_summary
    )
    assert "handoff_packet_url: `ws://localhost:8765`" in markdown_summary
    assert (
        "handoff_packet_artifact_root: `artifacts/wing_readonly_preflight`"
        in markdown_summary
    )
    assert "handoff_packet_value_types: `fader,gain,pan`" in markdown_summary
    assert "handoff_packet_url_matches_capture: `true`" in markdown_summary
    assert "handoff_packet_artifact_root_matches_capture: `true`" in markdown_summary
    assert "handoff_packet_channel_matches_capture: `true`" in markdown_summary
    assert "handoff_packet_value_types_match_capture: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_wing_preflight_plan_artifact(tmp_path):
    result = CheckResult(
        name="wing_readonly_preflight_plan_artifact_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_path": "/tmp/wing_readonly_preflight_plan.json",
            "plan_type": "wing_readonly_preflight_offline_plan",
            "plan_status": "ready_for_operator_review",
            "plan_output": "/tmp/wing_readonly_preflight_plan.json",
            "expected_plan_output": "/tmp/wing_readonly_preflight_plan.json",
            "channels": [3],
            "operator_confirmed_channels": [3],
            "operator_session_confirmation_required": True,
            "operator_confirmed_session_ready": True,
            "value_types": ["fader", "gain", "pan"],
            "request_count": 5,
            "expected_request_count": 5,
            "request_sequence_matches_expected": True,
            "read_only": True,
            "live_write": False,
            "contacts_backend": False,
            "queries_console": False,
            "operator_required": True,
            "expected_plan_required_for_real_run": True,
            "physical_preflight_requires_existing_backend_connection": True,
            "backend_port_prestart_check_required": True,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "backend_port_prestart_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_summary_required": True,
            "backend_port_prestart_summary_path": (
                "artifacts/no_wing_release_check/run/summary.json"
            ),
            "backend_port_prestart_summary_sha256": "a" * 64,
            "backend_port_prestart_summary_type": "no_wing_release_check",
            "backend_port_prestart_summary_status": "passed",
            "backend_port_prestart_summary_connection_policy": (
                "no_hardware_no_network_no_live_writes"
            ),
            "backend_port_prestart_summary_live_write": False,
            "backend_port_prestart_summary_contacts_backend": False,
            "backend_port_prestart_summary_queries_console": False,
            "backend_port_prestart_summary_executes_tests": False,
            "backend_port_prestart_check_status": "passed",
            "manual_review_required_before_readiness_update": True,
            "execution_allowed_by_plan": False,
            "live_authority_granted": False,
            "selected_mvp_blocker": WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER,
            "selected_mvp_blocker_status": (
                WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
            ),
            "forbidden_request_types": ["apply_operator_proposal", "set_gain"],
            "fail_closed_conditions": [
                "backend_not_in_wing_mode",
                "operator_channel_confirmation_mismatch",
                "operator_session_confirmation_missing",
                "physical_readback_missing",
            ],
            "post_run_review_commands": [
                "python3 tools/review_wing_readonly_preflight_artifacts.py",
                "python3 tools/review_wing_readonly_preflight_decisions.py",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert wing_readonly_preflight_plan_artifact_summary([result]) == summary[
        "wing_readonly_preflight_plan_artifact"
    ]
    artifact = json_summary["wing_readonly_preflight_plan_artifact"]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["contacts_backend"] is False
    assert artifact["queries_console"] is False
    assert artifact["operator_required"] is True
    assert artifact["expected_plan_required_for_real_run"] is True
    assert artifact["physical_preflight_requires_existing_backend_connection"] is True
    assert artifact["backend_port_prestart_check_required"] is True
    assert artifact["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert artifact["backend_port_prestart_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert artifact["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert artifact["backend_port_prestart_summary_contacts_backend"] is False
    assert artifact["backend_port_prestart_summary_queries_console"] is False
    assert artifact["backend_port_prestart_summary_executes_tests"] is False
    assert artifact["manual_review_required_before_readiness_update"] is True
    assert artifact["execution_allowed_by_plan"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["plan_status"] == "ready_for_operator_review"
    assert artifact["selected_mvp_blocker"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER
    )
    assert artifact["selected_mvp_blocker_status"] == (
        WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER_STATUS
    )
    assert artifact["channels"] == [3]
    assert artifact["operator_confirmed_channels"] == [3]
    assert artifact["operator_session_confirmation_required"] is True
    assert artifact["operator_confirmed_session_ready"] is True
    assert artifact["value_types"] == ["fader", "gain", "pan"]
    assert artifact["request_count"] == 5
    assert artifact["expected_request_count"] == 5
    assert artifact["request_sequence_matches_expected"] is True
    assert artifact["forbidden_request_types"] == ["apply_operator_proposal", "set_gain"]
    assert artifact["fail_closed_conditions"] == [
        "backend_not_in_wing_mode",
        "operator_channel_confirmation_mismatch",
        "operator_session_confirmation_missing",
        "physical_readback_missing",
    ]
    assert artifact["post_run_review_commands"] == [
        "python3 tools/review_wing_readonly_preflight_artifacts.py",
        "python3 tools/review_wing_readonly_preflight_decisions.py",
    ]
    assert artifact["violations"] == []
    assert "## WING Read-Only Preflight Plan Artifact" in markdown_summary
    assert "contacts_backend: `false`" in markdown_summary
    assert "expected_plan_required_for_real_run: `true`" in markdown_summary
    assert (
        "physical_preflight_requires_existing_backend_connection: `true`"
        in markdown_summary
    )
    assert "backend_port_prestart_check_required: `true`" in markdown_summary
    assert "backend_port_prestart_check_must_pass_before_starting_backend: `true`" in markdown_summary
    assert "backend_port_prestart_summary_contacts_backend: `false`" in markdown_summary
    assert "backend_port_prestart_summary_queries_console: `false`" in markdown_summary
    assert "backend_port_prestart_summary_executes_tests: `false`" in markdown_summary
    assert "backend_port_prestart_check_failure_detail_fields" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert "tools/no_wing_release_check.py" in markdown_summary
    assert "manual_review_required_before_readiness_update: `true`" in markdown_summary
    assert "operator_session_confirmation_required: `true`" in markdown_summary
    assert "operator_confirmed_session_ready: `true`" in markdown_summary
    assert WING_READONLY_PREFLIGHT_SELECTED_MVP_BLOCKER in markdown_summary
    assert "physical_readback_missing" in markdown_summary
    assert "review_wing_readonly_preflight_decisions.py" in markdown_summary
    assert "request_sequence_matches_expected: `true`" in markdown_summary
    assert "set_gain" in markdown_summary


def test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact(
    tmp_path,
):
    result = CheckResult(
        name="wing_readonly_preflight_handoff_packet_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "packet_path": "/tmp/wing_readonly_preflight_handoff_packet.json",
            "expected_plan_output": "/tmp/wing_readonly_preflight_plan.json",
            "packet_type": "wing_readonly_preflight_handoff_packet",
            "packet_status": "operator_hardware_step_required",
            "plan_artifact_status": "passed",
            "plan_artifact_violations": [],
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
            "backend_port_prestart_check_operator_sequence_step_index": 0,
            "backend_port_prestart_check_must_be_first_operator_step": True,
            "backend_port_prestart_check_operator_sequence_step": (
                "Before starting the intended backend/WING session, run "
                "prestart_backend_port_check and resolve only confirmed "
                "unexpected listeners."
            ),
            "backend_port_prestart_check_operator_sequence_step_is_first": True,
            "backend_port_prestart_gate": {
                "required": True,
                "operator_sequence_step_index": 0,
                "operator_sequence_step_must_be_first": True,
                "operator_sequence_step": (
                    "Before starting the intended backend/WING session, run "
                    "prestart_backend_port_check and resolve only confirmed "
                    "unexpected listeners."
                ),
                "required_markers": [
                    "Before starting the intended backend/WING session",
                    "prestart_backend_port_check",
                    "unexpected listeners",
                ],
                "runs_command": "prestart_backend_port_check",
                "must_pass_before_backend_wing_session": True,
                "grants_live_authority": False,
            },
            "expected_backend_port_prestart_gate": {
                "required": True,
                "operator_sequence_step_index": 0,
                "operator_sequence_step_must_be_first": True,
                "operator_sequence_step": (
                    "Before starting the intended backend/WING session, run "
                    "prestart_backend_port_check and resolve only confirmed "
                    "unexpected listeners."
                ),
                "required_markers": [
                    "Before starting the intended backend/WING session",
                    "prestart_backend_port_check",
                    "unexpected listeners",
                ],
                "runs_command": "prestart_backend_port_check",
                "must_pass_before_backend_wing_session": True,
                "grants_live_authority": False,
            },
            "declared_expected_backend_port_prestart_gate": {
                "required": True,
                "operator_sequence_step_index": 0,
                "operator_sequence_step_must_be_first": True,
                "operator_sequence_step": (
                    "Before starting the intended backend/WING session, run "
                    "prestart_backend_port_check and resolve only confirmed "
                    "unexpected listeners."
                ),
                "required_markers": [
                    "Before starting the intended backend/WING session",
                    "prestart_backend_port_check",
                    "unexpected listeners",
                ],
                "runs_command": "prestart_backend_port_check",
                "must_pass_before_backend_wing_session": True,
                "grants_live_authority": False,
            },
            "backend_port_prestart_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "manual_review_required_before_readiness_update": True,
            "execution_allowed_by_packet": False,
            "readiness_matrix_update_allowed_by_packet": False,
            "readiness_closure_allowed_by_packet": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "next_manual_artifact_required": (
                WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
            ),
            "plan_output": "/tmp/wing_readonly_preflight_plan.json",
            "channels": [3],
            "value_types": ["fader", "gain", "pan"],
            "command_names": [
                "capture_readonly_preflight",
                "focused_validation",
                "prestart_backend_port_check",
                "preview_readonly_request_plan",
                "review_artifact_and_write_pending_decision_template",
                "review_edited_decision_json",
            ],
            "preview_command_uses_plan_only": True,
            "capture_command_requires_physical_readback": True,
            "prestart_backend_port_check_command": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "review_command_writes_decision_template": True,
            "review_decision_template_output": (
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
            ),
            "review_decision_template_output_matches_contract": True,
            "review_decision_template_output_contract_flag": True,
            "review_command_decision_template_output": (
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
            ),
            "review_command_decision_template_output_matches_packet": True,
            "review_output": WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
            "review_output_matches_contract": True,
            "review_output_contract_flag": True,
            "review_command_output": WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT,
            "review_command_output_matches_packet": True,
            "decision_review_output": WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
            "decision_review_output_matches_contract": True,
            "decision_review_output_contract_flag": True,
            "decision_review_command_output": (
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
            ),
            "decision_review_command_output_matches_packet": True,
            "required_review_checklist_ids": ["confirm_no_live_authority"],
            "forbidden_live_request_types": ["apply_operator_proposal"],
            "blockers_not_closed_by_packet": ["physical_console_readback_proof"],
            "operator_sequence_count": 10,
            "operator_sequence_has_mvp_blocker_anchor_digest_review": True,
            "operator_sequence_mvp_blocker_anchor_digest_review_step": (
                "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
                "current first open blocker before supervised proof planning."
            ),
            "operator_sequence_mvp_blocker_anchor_digest_review_step_index": 8,
            "operator_sequence_has_supervised_prestart_gate": True,
            "operator_sequence_supervised_prestart_gate_step": (
                "Only after accepted_decision_ready and a valid accepted-candidate "
                "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
            ),
            "operator_sequence_supervised_prestart_gate_step_index": 9,
            "operator_sequence_supervised_prestart_gate_is_final": True,
            "supervised_prestart_gate": {
                "required": True,
                "operator_sequence_step_index": 9,
                "operator_sequence_step_must_be_final": True,
                "operator_sequence_step": (
                    "Only after accepted_decision_ready and a valid accepted-candidate "
                    "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
                ),
                "required_markers": [
                    "accepted_decision_ready",
                    "valid accepted-candidate prestart trace",
                    "WING_SUPERVISED_FADER_GAIN_PROOF",
                ],
                "unlocks_runbook": "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
                "grants_live_authority": False,
                "requires_operator_decision_status": "accepted_decision_ready",
                "requires_accepted_candidate_prestart_trace": True,
            },
            "expected_supervised_prestart_gate": {
                "required": True,
                "operator_sequence_step_index": 9,
                "operator_sequence_step_must_be_final": True,
                "operator_sequence_step": (
                    "Only after accepted_decision_ready and a valid accepted-candidate "
                    "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
                ),
                "required_markers": [
                    "accepted_decision_ready",
                    "valid accepted-candidate prestart trace",
                    "WING_SUPERVISED_FADER_GAIN_PROOF",
                ],
                "unlocks_runbook": "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
                "grants_live_authority": False,
                "requires_operator_decision_status": "accepted_decision_ready",
                "requires_accepted_candidate_prestart_trace": True,
            },
            "required_supervised_prestart_gate_markers": [
                "accepted_decision_ready",
                "valid accepted-candidate prestart trace",
                "WING_SUPERVISED_FADER_GAIN_PROOF",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert wing_readonly_preflight_handoff_packet_artifact_summary([result]) == summary[
        "wing_readonly_preflight_handoff_packet_artifact"
    ]
    artifact = json_summary["wing_readonly_preflight_handoff_packet_artifact"]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["expected_plan_required_for_real_run"] is True
    assert artifact["physical_preflight_requires_existing_backend_connection"] is True
    assert artifact["backend_port_prestart_check_required"] is True
    assert artifact["backend_port_prestart_check_must_pass_before_starting_backend"] is True
    assert artifact["backend_port_prestart_check_operator_sequence_step_index"] == 0
    assert artifact["backend_port_prestart_check_must_be_first_operator_step"] is True
    assert artifact["backend_port_prestart_check_operator_sequence_step"] == (
        "Before starting the intended backend/WING session, run "
        "prestart_backend_port_check and resolve only confirmed unexpected listeners."
    )
    assert artifact["backend_port_prestart_check_operator_sequence_step_is_first"] is True
    assert artifact["backend_port_prestart_gate"] == artifact[
        "expected_backend_port_prestart_gate"
    ]
    assert artifact["declared_expected_backend_port_prestart_gate"] == artifact[
        "expected_backend_port_prestart_gate"
    ]
    assert artifact["backend_port_prestart_gate"]["operator_sequence_step_index"] == 0
    assert artifact["backend_port_prestart_gate"][
        "operator_sequence_step_must_be_first"
    ] is True
    assert artifact["backend_port_prestart_gate"][
        "must_pass_before_backend_wing_session"
    ] is True
    assert artifact["backend_port_prestart_gate"]["grants_live_authority"] is False
    assert artifact["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert artifact["manual_review_required_before_readiness_update"] is True
    assert artifact["execution_allowed_by_packet"] is False
    assert artifact["readiness_matrix_update_allowed_by_packet"] is False
    assert artifact["readiness_closure_allowed_by_packet"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["next_manual_artifact_required"] == (
        WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
    )
    assert artifact["packet_status"] == "operator_hardware_step_required"
    assert artifact["plan_artifact_status"] == "passed"
    assert artifact["plan_artifact_violations"] == []
    assert artifact["preview_command_uses_plan_only"] is True
    assert artifact["capture_command_requires_physical_readback"] is True
    assert artifact["prestart_backend_port_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert artifact["review_command_writes_decision_template"] is True
    assert artifact["review_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert artifact["review_decision_template_output_matches_contract"] is True
    assert artifact["review_decision_template_output_contract_flag"] is True
    assert artifact["review_command_decision_template_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )
    assert artifact["review_command_decision_template_output_matches_packet"] is True
    assert artifact["review_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert artifact["review_output_matches_contract"] is True
    assert artifact["review_output_contract_flag"] is True
    assert artifact["review_command_output"] == WING_READONLY_PREFLIGHT_REVIEW_REPORT_OUTPUT
    assert artifact["review_command_output_matches_packet"] is True
    assert artifact["decision_review_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert artifact["decision_review_output_matches_contract"] is True
    assert artifact["decision_review_output_contract_flag"] is True
    assert artifact["decision_review_command_output"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT
    )
    assert artifact["decision_review_command_output_matches_packet"] is True
    assert artifact["operator_sequence_count"] == 10
    assert artifact["operator_sequence_has_mvp_blocker_anchor_digest_review"] is True
    assert artifact["operator_sequence_mvp_blocker_anchor_digest_review_step"] == (
        "Confirm mvp_blocker_anchor and mvp_blocker_anchor_sha256 match the "
        "current first open blocker before supervised proof planning."
    )
    assert artifact["operator_sequence_mvp_blocker_anchor_digest_review_step_index"] == 8
    assert artifact["operator_sequence_has_supervised_prestart_gate"] is True
    assert artifact["operator_sequence_supervised_prestart_gate_step"] == (
        "Only after accepted_decision_ready and a valid accepted-candidate "
        "prestart trace, move to WING_SUPERVISED_FADER_GAIN_PROOF."
    )
    assert artifact["operator_sequence_supervised_prestart_gate_step_index"] == 9
    assert artifact["operator_sequence_supervised_prestart_gate_is_final"] is True
    assert artifact["supervised_prestart_gate"] == artifact[
        "expected_supervised_prestart_gate"
    ]
    assert artifact["supervised_prestart_gate"]["operator_sequence_step_index"] == 9
    assert artifact["supervised_prestart_gate"][
        "operator_sequence_step_must_be_final"
    ] is True
    assert artifact["supervised_prestart_gate"]["grants_live_authority"] is False
    assert artifact["supervised_prestart_gate"][
        "requires_accepted_candidate_prestart_trace"
    ] is True
    assert artifact["required_supervised_prestart_gate_markers"] == [
        "accepted_decision_ready",
        "valid accepted-candidate prestart trace",
        "WING_SUPERVISED_FADER_GAIN_PROOF",
    ]
    assert artifact["violations"] == []
    assert "## WING Read-Only Preflight Handoff Packet Artifact" in markdown_summary
    assert "expected_plan_required_for_real_run: `true`" in markdown_summary
    assert (
        "physical_preflight_requires_existing_backend_connection: `true`"
        in markdown_summary
    )
    assert "backend_port_prestart_check_required: `true`" in markdown_summary
    assert "backend_port_prestart_check_must_pass_before_starting_backend: `true`" in markdown_summary
    assert "backend_port_prestart_check_operator_sequence_step_index: `0`" in (
        markdown_summary
    )
    assert "backend_port_prestart_check_must_be_first_operator_step: `true`" in (
        markdown_summary
    )
    assert "backend_port_prestart_check_operator_sequence_step_is_first: `true`" in (
        markdown_summary
    )
    assert "backend_port_prestart_gate" in markdown_summary
    assert "expected_backend_port_prestart_gate" in markdown_summary
    assert "prestart_backend_port_check_command" in markdown_summary
    assert "backend_port_prestart_check_failure_detail_fields" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert "tools/no_wing_release_check.py" in markdown_summary
    assert "review_decision_template_output:" in markdown_summary
    assert "review_decision_template_output_matches_contract: `true`" in markdown_summary
    assert "review_command_decision_template_output_matches_packet: `true`" in (
        markdown_summary
    )
    assert "review_output_matches_contract: `true`" in markdown_summary
    assert "review_command_output_matches_packet: `true`" in markdown_summary
    assert "decision_review_output_matches_contract: `true`" in markdown_summary
    assert "decision_review_command_output_matches_packet: `true`" in markdown_summary
    assert "manual_review_required_before_readiness_update: `true`" in markdown_summary
    assert (
        "next_manual_artifact_required: "
        f"`{WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED}`"
    ) in markdown_summary
    assert "plan_artifact_status: `passed`" in markdown_summary
    assert "execution_allowed_by_packet: `false`" in markdown_summary
    assert "preview_command_uses_plan_only: `true`" in markdown_summary
    assert "capture_command_requires_physical_readback: `true`" in markdown_summary
    assert "operator_sequence_has_supervised_prestart_gate: `true`" in markdown_summary
    assert (
        "operator_sequence_has_mvp_blocker_anchor_digest_review: `true`"
        in markdown_summary
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in markdown_summary
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index: `8`"
        in markdown_summary
    )
    assert "operator_sequence_supervised_prestart_gate_step" in markdown_summary
    assert "operator_sequence_supervised_prestart_gate_step_index: `9`" in markdown_summary
    assert "operator_sequence_supervised_prestart_gate_is_final: `true`" in markdown_summary
    assert "required_supervised_prestart_gate_markers" in markdown_summary
    assert "supervised_prestart_gate" in markdown_summary
    assert "expected_supervised_prestart_gate" in markdown_summary
    assert '"grants_live_authority": false' in markdown_summary
    assert '"requires_accepted_candidate_prestart_trace": true' in markdown_summary
    assert "valid accepted-candidate prestart trace" in markdown_summary
    assert "physical_console_readback_proof" in markdown_summary


def test_no_wing_release_check_summary_embeds_read_surface_acceptance_template_artifact(
    tmp_path,
):
    result = CheckResult(
        name="read_surface_physical_readback_acceptance_template_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "template_path": "/tmp/read_surface_physical_readback_acceptance_template.json",
            "template_type": "read_surface_physical_readback_fixture_acceptance_template",
            "template_status": "operator_candidate_missing",
            "source_queue_status": "operator_artifact_capture_required",
            "selected_surface_id": None,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_template": False,
            "fixture_closure_allowed_by_template": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "review_candidate_count": 0,
            "template_record_count": 0,
            "candidate_trace_count": 0,
            "required_operator_edits": [
                "decision",
                "reviewed_by",
                "reviewed_at",
                "review_notes",
            ],
            "operator_sequence_count": 6,
            "blocked_by": ["operator_must_capture_missing_physical_readback_artifacts"],
            "record_unsafe_flags": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_acceptance_template_artifact_summary(
        [result]
    ) == summary["read_surface_physical_readback_acceptance_template_artifact"]
    artifact = json_summary[
        "read_surface_physical_readback_acceptance_template_artifact"
    ]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["operator_required"] is True
    assert artifact["execution_allowed_by_template"] is False
    assert artifact["fixture_closure_allowed_by_template"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["template_status"] == "operator_candidate_missing"
    assert artifact["template_record_count"] == 0
    assert artifact["candidate_trace_count"] == 0
    assert artifact["required_operator_edits"] == [
        "decision",
        "reviewed_by",
        "reviewed_at",
        "review_notes",
    ]
    assert artifact["record_unsafe_flags"] == []
    assert artifact["violations"] == []
    assert (
        "## Read Surface Physical Readback Acceptance Template Artifact"
        in markdown_summary
    )
    assert "execution_allowed_by_template: `false`" in markdown_summary
    assert "fixture_closure_allowed_by_template: `false`" in markdown_summary
    assert "candidate_trace_count: `0`" in markdown_summary
    assert "operator_must_capture_missing_physical_readback_artifacts" in markdown_summary


def test_no_wing_release_check_summary_embeds_wing_preflight_review_decision(
    tmp_path,
):
    result = CheckResult(
        name="wing_readonly_preflight_review_decision_ledger_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "ledger_status": "operator_review_required",
            "review_tool_path": "tools/review_wing_readonly_preflight_decisions.py",
            "review_tool_available": True,
            "review_command": [
                "python3",
                "tools/review_wing_readonly_preflight_decisions.py",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--decision-root",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
                "--output",
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_REPORT_OUTPUT,
            ],
            "review_exit_codes": {
                "accepted_decision_ready": 0,
                "preflight_candidate_missing": 2,
                "decision_missing_or_invalid": 3,
                "candidate_rejected": 4,
            },
            "preflight_evidence_status": "operator_preflight_required",
            "decision_search_roots": [
                WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
            ],
            "review_candidate_count": 0,
            "decision_record_count": 0,
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "manual_readiness_update_candidate_available": False,
            "operator_review_required": True,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "accepted_source_ids": [],
            "accepted_candidate_prestart_contract_valid": False,
            "accepted_candidate_prestart_source_ids": [],
            "rejected_source_ids": [],
            "invalid_record_ids": [],
            "unreviewed_candidate_source_ids": [],
            "candidate_reviews": [
                {
                    "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
                    "status": "valid_readonly_preflight_artifact",
                    "run_id": "20260606T120000Z_wing_readonly_preflight",
                    "summary_path": "artifacts/wing_readonly_preflight/run/summary.json",
                    "events_path": "artifacts/wing_readonly_preflight/run/events.jsonl",
                    "backend_port_prestart_check_required": True,
                    "backend_port_prestart_check_must_pass_before_starting_backend": True,
                    "backend_port_prestart_check_command": [
                        "PYTHONPATH=backend",
                        "python3",
                        "tools/no_wing_release_check.py",
                        "--quick",
                    ],
                    "backend_port_prestart_check_failure_detail_fields": list(
                        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                }
            ],
            "blocked_by": ["operator_must_capture_valid_readonly_preflight_candidate_first"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert (
        wing_readonly_preflight_review_decision_summary([result])
        == summary["wing_readonly_preflight_review_decision_ledger"]
    )
    decision = json_summary["wing_readonly_preflight_review_decision_ledger"]
    assert decision["read_only"] is True
    assert decision["live_write"] is False
    assert decision["operator_required"] is True
    assert decision["ledger_status"] == "operator_review_required"
    assert decision["review_tool_path"] == "tools/review_wing_readonly_preflight_decisions.py"
    assert decision["review_tool_available"] is True
    assert decision["review_command"][1] == "tools/review_wing_readonly_preflight_decisions.py"
    assert decision["review_exit_codes"]["accepted_decision_ready"] == 0
    assert decision["candidate_reviews"][0]["backend_port_prestart_check_required"] is True
    assert (
        decision["candidate_reviews"][0][
            "backend_port_prestart_check_must_pass_before_starting_backend"
        ]
        is True
    )
    assert "listener_processes" in decision["candidate_reviews"][0][
        "backend_port_prestart_check_failure_detail_fields"
    ]
    assert "listener_confirmation_checklist" in decision["candidate_reviews"][0][
        "backend_port_prestart_check_failure_detail_fields"
    ]
    assert decision["decision_search_roots"] == [
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT
    ]
    assert decision["manual_readiness_update_candidate_available"] is False
    assert decision["accepted_candidate_prestart_contract_valid"] is False
    assert decision["accepted_candidate_prestart_source_ids"] == []
    assert decision["readiness_matrix_update_allowed_by_decision"] is False
    assert decision["readiness_closure_allowed_by_decision"] is False
    assert decision["live_authority_granted"] is False
    assert decision["full_autonomous_ready"] is False
    assert "## WING Read-Only Preflight Review Decision Ledger" in markdown_summary
    assert "readiness_matrix_update_allowed_by_decision: `false`" in markdown_summary
    assert "WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md" in markdown_summary
    assert "accepted_decision_ready" in markdown_summary
    assert "candidate_reviews:" in markdown_summary
    assert "accepted_candidate_prestart_contract_valid: `false`" in markdown_summary
    assert "backend_port_prestart_check_required" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert "listener_confirmation_checklist" in markdown_summary
    assert "wing_readonly_preflight_review_decisions" in markdown_summary
    assert "manual_readiness_update_candidate_available: `false`" in markdown_summary


def test_no_wing_release_check_summary_embeds_raw_mixer_readback_contract(tmp_path):
    result = CheckResult(
        name="raw_mixer_state_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "raw_mixer_state_physical_readback.v1",
            "surface_id": "raw_mixer_state_cache",
            "remediation_action_id": "remediate_raw_mixer_state_cache",
            "next_physical_readback_surface_id": "raw_mixer_state_cache",
            "next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "fixture_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "remediation_next_physical_readback_surface_id": "raw_mixer_state_cache",
            "remediation_next_physical_readback_action_id": "remediate_raw_mixer_state_cache",
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "required_top_level_field_count": 10,
            "required_row_field_count": 5,
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert raw_mixer_readback_contract_summary([result]) == summary[
        "raw_mixer_state_physical_readback_contract"
    ]
    contract = json_summary["raw_mixer_state_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "raw_mixer_state_cache"
    assert contract["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert "## Raw Mixer State Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_raw_mixer_artifact_intake(tmp_path):
    result = CheckResult(
        name="raw_mixer_state_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "operator_artifact_missing",
            "schema_id": "raw_mixer_state_physical_readback.v1",
            "surface_id": "raw_mixer_state_cache",
            "remediation_action_id": "remediate_raw_mixer_state_cache",
            "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
            "artifact_search_roots": ["artifacts/read_surface_physical_readback/raw_mixer_state"],
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "accepted_as_physical_fixture_candidate": False,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "latest_valid_candidate_source_id": None,
            "latest_valid_candidate_captured_at": None,
            "blocked_by": ["operator_must_capture_raw_mixer_state_physical_readback_artifact"],
            "artifact_source_ids": [],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert raw_mixer_readback_artifact_intake_summary([result]) == summary[
        "raw_mixer_state_physical_readback_artifact_intake"
    ]
    intake = json_summary["raw_mixer_state_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["operator_required"] is True
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "operator_artifact_missing"
    assert intake["artifact_count"] == 0
    assert intake["accepted_as_physical_fixture_candidate"] is False
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert "## Raw Mixer State Physical Readback Artifact Intake" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "operator_review_required: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_basic_controls_readback_contract(tmp_path):
    result = CheckResult(
        name="channel_basic_controls_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "channel_basic_controls_physical_readback.v1",
            "surface_id": "channel_identity_and_basic_controls",
            "remediation_action_id": "remediate_channel_identity_and_basic_controls",
            "physical_readback_surface_id": "channel_identity_and_basic_controls",
            "physical_readback_action_id": "remediate_channel_identity_and_basic_controls",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "pan_phase_delay",
            ],
            "required_top_level_field_count": 10,
            "required_channel_field_count": 8,
            "required_identity_fields": ["source_name", "source_role", "input_source"],
            "required_control_fields": ["fader_db", "trim_db", "mute", "pan"],
            "required_coverage_flags": [
                "identity",
                "basic_controls",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_channel_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "mute_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_basic_controls_readback_contract_summary([result]) == summary[
        "channel_basic_controls_physical_readback_contract"
    ]
    contract = json_summary["channel_basic_controls_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "channel_identity_and_basic_controls"
    assert contract["remediation_action_id"] == "remediate_channel_identity_and_basic_controls"
    assert contract["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert contract["required_control_fields"] == ["fader_db", "trim_db", "mute", "pan"]
    assert "## Channel Basic Controls Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "promotion_families_requiring_surface" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_basic_controls_artifact_intake(tmp_path):
    result = CheckResult(
        name="channel_basic_controls_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "operator_artifact_missing",
            "schema_id": "channel_basic_controls_physical_readback.v1",
            "surface_id": "channel_identity_and_basic_controls",
            "remediation_action_id": "remediate_channel_identity_and_basic_controls",
            "required_contract_method_id": "channel_basic_controls_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/channel_basic_controls",
            ],
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "valid_channel_count": 0,
            "accepted_as_physical_fixture_candidate": False,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "pan_phase_delay",
            ],
            "latest_valid_candidate_source_id": None,
            "latest_valid_candidate_captured_at": None,
            "blocked_by": [
                "operator_must_capture_channel_basic_controls_physical_readback_artifact",
            ],
            "artifact_source_ids": [],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_basic_controls_readback_artifact_intake_summary([result]) == summary[
        "channel_basic_controls_physical_readback_artifact_intake"
    ]
    intake = json_summary["channel_basic_controls_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["operator_required"] is True
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "operator_artifact_missing"
    assert intake["artifact_count"] == 0
    assert intake["valid_channel_count"] == 0
    assert intake["accepted_as_physical_fixture_candidate"] is False
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["promotion_families_requiring_surface"] == [
        "fader",
        "gain",
        "mute",
        "pan_phase_delay",
    ]
    assert "## Channel Basic Controls Physical Readback Artifact Intake" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "operator_review_required: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_processing_readback_contract(tmp_path):
    result = CheckResult(
        name="channel_processing_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "channel_processing_physical_readback.v1",
            "surface_id": "channel_eq_dynamics_gate",
            "remediation_action_id": "remediate_channel_eq_dynamics_gate",
            "physical_readback_surface_id": "channel_eq_dynamics_gate",
            "physical_readback_action_id": "remediate_channel_eq_dynamics_gate",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "eq",
                "compressor",
                "gate",
                "deessing",
            ],
            "required_top_level_field_count": 10,
            "required_channel_field_count": 10,
            "required_hpf_fields": ["enabled", "frequency_hz"],
            "required_eq_fields": ["enabled", "bands"],
            "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
            "required_compressor_fields": ["enabled", "model", "gain_reduction_db"],
            "required_gate_fields": ["enabled", "model", "range_db"],
            "required_coverage_flags": [
                "hpf",
                "eq",
                "compressor",
                "gate",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_eq_band_count": 4,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_channel_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "eq_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_processing_readback_contract_summary([result]) == summary[
        "channel_processing_physical_readback_contract"
    ]
    contract = json_summary["channel_processing_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "channel_eq_dynamics_gate"
    assert contract["remediation_action_id"] == "remediate_channel_eq_dynamics_gate"
    assert contract["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
    ]
    assert contract["required_eq_band_fields"] == ["band", "frequency_hz", "gain_db", "q"]
    assert "gain_reduction_db" in contract["required_compressor_fields"]
    assert "range_db" in contract["required_gate_fields"]
    assert contract["minimum_eq_band_count"] == 4
    assert "## Channel Processing Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_eq_band_count: `4`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_channel_processing_artifact_intake(tmp_path):
    result = CheckResult(
        name="channel_processing_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "operator_artifact_missing",
            "schema_id": "channel_processing_physical_readback.v1",
            "surface_id": "channel_eq_dynamics_gate",
            "remediation_action_id": "remediate_channel_eq_dynamics_gate",
            "required_contract_method_id": "channel_processing_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/channel_processing",
            ],
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "valid_channel_count": 0,
            "accepted_as_physical_fixture_candidate": False,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "eq",
                "compressor",
                "gate",
                "deessing",
                "frequency_separation",
                "cross_channel_unmasking",
            ],
            "latest_valid_candidate_source_id": None,
            "latest_valid_candidate_captured_at": None,
            "blocked_by": [
                "operator_must_capture_channel_processing_physical_readback_artifact",
            ],
            "artifact_source_ids": [],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert channel_processing_readback_artifact_intake_summary([result]) == summary[
        "channel_processing_physical_readback_artifact_intake"
    ]
    intake = json_summary["channel_processing_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["operator_required"] is True
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "operator_artifact_missing"
    assert intake["artifact_count"] == 0
    assert intake["valid_channel_count"] == 0
    assert intake["accepted_as_physical_fixture_candidate"] is False
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["promotion_families_requiring_surface"] == [
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ]
    assert "## Channel Processing Physical Readback Artifact Intake" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "operator_review_required: `true`" in markdown_summary


def test_no_wing_release_check_processing_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="channel_processing_physical_readback_artifact_intake_schema",
        status="failed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "channel_processing_physical_readback.v1",
            "surface_id": "channel_eq_dynamics_gate",
            "remediation_action_id": "remediate_channel_eq_dynamics_gate",
            "required_contract_method_id": "channel_processing_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_channel_count": 24,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": False,
            "physical_fixture_gap_remains": False,
            "fixture_closure_allowed_by_intake": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["eq"],
            "blocked_by": [],
            "artifact_source_ids": ["unsafe-processing.json"],
            "intake_violations": [],
            "violations": ["channel_processing_artifact_intake_live_authority_not_false"],
        },
    )

    summary = channel_processing_readback_artifact_intake_summary([result])

    assert summary["status"] == "failed"
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["operator_review_required"] is True
    assert summary["physical_fixture_gap_remains"] is True
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == [
        "channel_processing_artifact_intake_live_authority_not_false"
    ]


def test_no_wing_release_check_summary_embeds_bus_group_main_dca_readback_contract(tmp_path):
    result = CheckResult(
        name="bus_group_main_dca_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "bus_group_main_dca_physical_readback.v1",
            "surface_id": "bus_group_main_dca_state",
            "remediation_action_id": "remediate_bus_group_main_dca_state",
            "physical_readback_surface_id": "bus_group_main_dca_state",
            "physical_readback_action_id": "remediate_bus_group_main_dca_state",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "dca",
                "master_bus",
            ],
            "required_top_level_field_count": 14,
            "required_bus_field_count": 9,
            "required_main_field_count": 8,
            "required_dca_field_count": 8,
            "required_bus_control_fields": ["fader_db", "mute", "pan", "width", "mono"],
            "required_main_control_fields": ["fader_db", "mute", "pan"],
            "required_dca_control_fields": ["fader_db", "mute", "monitor"],
            "required_bus_assignment_fields": [
                "dca_assignments",
                "main_send_on",
                "main_send_level_db",
                "main_send_pre",
            ],
            "required_dca_assignment_fields": ["member_channels", "member_buses"],
            "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
            "required_bus_coverage_flags": [
                "controls",
                "assignments",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_main_coverage_flags": [
                "controls",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_dca_coverage_flags": [
                "identity",
                "controls",
                "assignments",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_bus_count": 1,
            "minimum_main_count": 1,
            "minimum_dca_count": 1,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_surface_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": ["manual_write_events", "dca_write_events"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert bus_group_main_dca_readback_contract_summary([result]) == summary[
        "bus_group_main_dca_physical_readback_contract"
    ]
    contract = json_summary["bus_group_main_dca_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "bus_group_main_dca_state"
    assert contract["remediation_action_id"] == "remediate_bus_group_main_dca_state"
    assert contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
    ]
    assert contract["required_bus_control_fields"] == ["fader_db", "mute", "pan", "width", "mono"]
    assert "true_peak_dbtp" in contract["required_metering_fields"]
    assert "member_buses" in contract["required_dca_assignment_fields"]
    assert contract["minimum_bus_count"] == 1
    assert contract["minimum_main_count"] == 1
    assert contract["minimum_dca_count"] == 1
    assert "## Bus Group Main DCA Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_bus_count: `1`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_bus_group_main_dca_artifact_intake(tmp_path):
    result = CheckResult(
        name="bus_group_main_dca_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "operator_artifact_missing",
            "schema_id": "bus_group_main_dca_physical_readback.v1",
            "surface_id": "bus_group_main_dca_state",
            "remediation_action_id": "remediate_bus_group_main_dca_state",
            "required_contract_method_id": "bus_group_main_dca_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/bus_group_main_dca",
            ],
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "valid_bus_count": 0,
            "valid_main_count": 0,
            "valid_dca_count": 0,
            "accepted_as_physical_fixture_candidate": False,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "dca",
                "master_bus",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": None,
            "latest_valid_candidate_captured_at": None,
            "blocked_by": [
                "operator_must_capture_bus_group_main_dca_physical_readback_artifact",
            ],
            "artifact_source_ids": [],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert bus_group_main_dca_readback_artifact_intake_summary([result]) == summary[
        "bus_group_main_dca_physical_readback_artifact_intake"
    ]
    intake = json_summary["bus_group_main_dca_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["operator_required"] is True
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "operator_artifact_missing"
    assert intake["artifact_count"] == 0
    assert intake["valid_bus_count"] == 0
    assert intake["valid_main_count"] == 0
    assert intake["valid_dca_count"] == 0
    assert intake["accepted_as_physical_fixture_candidate"] is False
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ]
    assert "## Bus Group Main DCA Physical Readback Artifact Intake" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "operator_review_required: `true`" in markdown_summary


def test_no_wing_release_check_bus_group_main_dca_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="bus_group_main_dca_physical_readback_artifact_intake_schema",
        status="failed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "bus_group_main_dca_physical_readback.v1",
            "surface_id": "bus_group_main_dca_state",
            "remediation_action_id": "remediate_bus_group_main_dca_state",
            "required_contract_method_id": "bus_group_main_dca_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_bus_count": 8,
            "valid_main_count": 1,
            "valid_dca_count": 8,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": False,
            "physical_fixture_gap_remains": False,
            "fixture_closure_allowed_by_intake": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["dca"],
            "blocked_by": [],
            "artifact_source_ids": ["unsafe-bus.json"],
            "intake_violations": [],
            "violations": ["bus_group_main_dca_artifact_intake_live_authority_not_false"],
        },
    )

    summary = bus_group_main_dca_readback_artifact_intake_summary([result])

    assert summary["status"] == "failed"
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["operator_review_required"] is True
    assert summary["physical_fixture_gap_remains"] is True
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == [
        "bus_group_main_dca_artifact_intake_live_authority_not_false"
    ]


def test_no_wing_release_check_summary_embeds_matrix_state_readback_contract(tmp_path):
    result = CheckResult(
        name="matrix_state_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "matrix_state_physical_readback.v1",
            "surface_id": "matrix_state",
            "remediation_action_id": "remediate_matrix_state",
            "physical_readback_surface_id": "matrix_state",
            "physical_readback_action_id": "remediate_matrix_state",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "fx_send_insert",
                "routing",
            ],
            "required_top_level_field_count": 10,
            "required_matrix_field_count": 11,
            "required_control_fields": ["fader_db", "mute", "pan"],
            "required_eq_fields": ["enabled", "bands"],
            "required_eq_band_fields": ["band", "frequency_hz", "gain_db", "q"],
            "required_dynamics_fields": ["enabled", "model", "gain_reduction_db"],
            "required_insert_fields": ["pre_insert_on", "post_insert_weight"],
            "required_metering_fields": ["lufs", "true_peak_dbtp", "headroom_db"],
            "required_coverage_flags": [
                "controls",
                "eq",
                "dynamics",
                "inserts",
                "metering",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_matrix_count": 1,
            "minimum_eq_band_count": 8,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_matrix_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "matrix_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert matrix_state_readback_contract_summary([result]) == summary[
        "matrix_state_physical_readback_contract"
    ]
    contract = json_summary["matrix_state_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "matrix_state"
    assert contract["remediation_action_id"] == "remediate_matrix_state"
    assert contract["promotion_families_requiring_surface"] == [
        "bus_group_matrix",
        "fx_send_insert",
        "routing",
    ]
    assert contract["required_control_fields"] == ["fader_db", "mute", "pan"]
    assert "gain_reduction_db" in contract["required_dynamics_fields"]
    assert "post_insert_weight" in contract["required_insert_fields"]
    assert "true_peak_dbtp" in contract["required_metering_fields"]
    assert contract["minimum_matrix_count"] == 1
    assert contract["minimum_eq_band_count"] == 8
    assert "## Matrix State Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_matrix_count: `1`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_matrix_state_artifact_intake(tmp_path):
    result = CheckResult(
        name="matrix_state_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "matrix_state_physical_readback.v1",
            "surface_id": "matrix_state",
            "remediation_action_id": "remediate_matrix_state",
            "required_contract_method_id": "matrix_state_physical_readback_contract",
            "artifact_search_roots": ["artifacts/read_surface_physical_readback/matrix_state"],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_matrix_count": 2,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "bus_group_matrix",
                "fx_send_insert",
                "routing",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_matrix_state_candidate_before_fixture_closure"
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert matrix_state_readback_artifact_intake_summary([result]) == summary[
        "matrix_state_physical_readback_artifact_intake"
    ]
    intake = json_summary["matrix_state_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["operator_required"] is True
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_matrix_count"] == 2
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## Matrix State Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_matrix_count: `2`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary


def test_no_wing_release_check_matrix_state_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="matrix_state_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "matrix_state_physical_readback.v1",
            "surface_id": "matrix_state",
            "remediation_action_id": "remediate_matrix_state",
            "required_contract_method_id": "matrix_state_physical_readback_contract",
            "artifact_search_roots": [],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_matrix_count": 1,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["routing"],
            "blocked_by": [],
            "artifact_source_ids": [],
            "intake_violations": [],
            "violations": ["matrix_state_artifact_intake_live_authority_not_false"],
        },
    )

    summary = matrix_state_readback_artifact_intake_summary([result])

    assert summary is not None
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == [
        "matrix_state_artifact_intake_live_authority_not_false"
    ]


def test_no_wing_release_check_summary_embeds_fx_slots_inserts_readback_contract(tmp_path):
    result = CheckResult(
        name="fx_slots_inserts_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "fx_slots_inserts_physical_readback.v1",
            "surface_id": "fx_slots_and_inserts",
            "remediation_action_id": "remediate_fx_slots_and_inserts",
            "physical_readback_surface_id": "fx_slots_and_inserts",
            "physical_readback_action_id": "remediate_fx_slots_and_inserts",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fx_send_insert",
                "routing",
                "mix_balance",
            ],
            "required_top_level_field_count": 12,
            "required_fx_slot_field_count": 16,
            "required_fx_slot_fields": ["slot", "slot_number", "parameter_schema"],
            "required_parameter_fields": ["parameter_index", "value", "safe_range"],
            "required_parameter_safe_range_fields": ["min", "max"],
            "required_parameter_schema_fields": [
                "status",
                "physical_model_schema_verified",
            ],
            "required_fx_slot_coverage_flags": [
                "identity",
                "assignment",
                "cache_matches_physical_readback",
            ],
            "required_insert_fields": ["target_type", "position", "linked_fx_slot"],
            "required_insert_coverage_flags": [
                "assignment",
                "status",
                "scoped_diff_marker",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_fx_slot_count": 1,
            "minimum_fx_parameter_count": 4,
            "minimum_insert_count": 1,
            "allowed_sources": ["physical_readback", "wing_physical_readback"],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "allowed_insert_target_types": ["channel", "bus", "main", "matrix"],
            "allowed_insert_positions": ["pre", "post"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "fx_parameter_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert fx_slots_inserts_readback_contract_summary([result]) == summary[
        "fx_slots_inserts_physical_readback_contract"
    ]
    contract = json_summary["fx_slots_inserts_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "fx_slots_and_inserts"
    assert contract["remediation_action_id"] == "remediate_fx_slots_and_inserts"
    assert contract["promotion_families_requiring_surface"] == [
        "fx_send_insert",
        "routing",
        "mix_balance",
    ]
    assert "parameter_schema" in contract["required_fx_slot_fields"]
    assert "safe_range" in contract["required_parameter_fields"]
    assert "physical_model_schema_verified" in contract["required_parameter_schema_fields"]
    assert "scoped_diff_marker" in contract["required_insert_coverage_flags"]
    assert contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert contract["minimum_fx_slot_count"] == 1
    assert contract["minimum_fx_parameter_count"] == 4
    assert contract["minimum_insert_count"] == 1
    assert "## FX Slots Inserts Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_fx_parameter_count: `4`" in markdown_summary
    assert "required_false_safety_flags" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_fx_slots_inserts_artifact_intake(
    tmp_path,
):
    result = CheckResult(
        name="fx_slots_inserts_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "fx_slots_inserts_physical_readback.v1",
            "surface_id": "fx_slots_and_inserts",
            "remediation_action_id": "remediate_fx_slots_and_inserts",
            "required_contract_method_id": "fx_slots_inserts_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/fx_slots_and_inserts",
            ],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_fx_slot_count": 2,
            "valid_insert_count": 3,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "fx_send_insert",
                "routing",
                "mix_balance",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_fx_slots_inserts_candidate_before_fixture_closure",
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert fx_slots_inserts_readback_artifact_intake_summary([result]) == summary[
        "fx_slots_inserts_physical_readback_artifact_intake"
    ]
    intake = json_summary["fx_slots_inserts_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["live_widening_allowed"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_fx_slot_count"] == 2
    assert intake["valid_insert_count"] == 3
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## FX Slots Inserts Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_fx_slot_count: `2`" in markdown_summary
    assert "valid_insert_count: `3`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary


def test_no_wing_release_check_fx_slots_inserts_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="fx_slots_inserts_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "fx_slots_inserts_physical_readback.v1",
            "surface_id": "fx_slots_and_inserts",
            "remediation_action_id": "remediate_fx_slots_and_inserts",
            "required_contract_method_id": "fx_slots_inserts_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_fx_slot_count": 2,
            "valid_insert_count": 3,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["fx_send_insert"],
            "violations": ["claimed_live_authority"],
        },
    )

    summary = fx_slots_inserts_readback_artifact_intake_summary([result])

    assert summary is not None
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == ["claimed_live_authority"]


def test_no_wing_release_check_summary_embeds_routing_sends_readback_contract(tmp_path):
    result = CheckResult(
        name="routing_sends_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "routing_sends_physical_readback.v1",
            "surface_id": "routing_and_sends",
            "remediation_action_id": "remediate_routing_and_sends",
            "physical_readback_surface_id": "routing_and_sends",
            "physical_readback_action_id": "remediate_routing_and_sends",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "routing",
                "fx_send_insert",
                "bus_group_matrix",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "required_top_level_field_count": 18,
            "required_static_route_field_count": 13,
            "required_static_route_fields": ["start", "end", "role", "wing_routing_hint"],
            "required_bus_send_fields": ["channel", "send", "target_bus", "pre_on"],
            "required_main_send_fields": ["channel", "main", "pre"],
            "required_matrix_send_fields": ["channel", "send", "target_matrix"],
            "required_output_route_fields": ["output_group", "output_number", "source_channel"],
            "required_static_route_coverage_flags": [
                "identity",
                "static_config_match",
                "physical_route_mapping",
                "scoped_diff_marker",
            ],
            "required_send_coverage_flags": [
                "assignment",
                "level",
                "mode",
                "scoped_diff_marker",
                "cache_matches_physical_readback",
            ],
            "required_output_route_coverage_flags": [
                "assignment",
                "source",
                "scoped_diff_marker",
                "cache_matches_physical_readback",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
            ],
            "minimum_static_dante_route_count": 1,
            "minimum_channel_bus_send_count": 1,
            "minimum_channel_main_send_count": 1,
            "minimum_channel_matrix_send_count": 1,
            "minimum_output_route_count": 1,
            "allowed_sources": [
                "physical_readback",
                "wing_physical_readback",
                "dlive_physical_readback",
            ],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "manual_write_events",
                "routing_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert routing_sends_readback_contract_summary([result]) == summary[
        "routing_sends_physical_readback_contract"
    ]
    contract = json_summary["routing_sends_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["mutates_console_state"] is False
    assert contract["queries_console"] is False
    assert contract["executes_tests"] is False
    assert contract["live_widening_allowed"] is False
    assert contract["full_autonomous_ready"] is False
    assert contract["artifact_required"] is True
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "routing_and_sends"
    assert contract["remediation_action_id"] == "remediate_routing_and_sends"
    assert "routing" in contract["promotion_families_requiring_surface"]
    assert "wing_routing_hint" in contract["required_static_route_fields"]
    assert "pre_on" in contract["required_bus_send_fields"]
    assert "main" in contract["required_main_send_fields"]
    assert "target_matrix" in contract["required_matrix_send_fields"]
    assert "source_channel" in contract["required_output_route_fields"]
    assert "cache_matches_physical_readback" in contract["required_send_coverage_flags"]
    assert (
        "cache_matches_physical_readback"
        in contract["required_output_route_coverage_flags"]
    )
    assert contract["required_false_safety_flags"] == [
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    ]
    assert contract["minimum_static_dante_route_count"] == 1
    assert contract["minimum_channel_bus_send_count"] == 1
    assert contract["minimum_channel_main_send_count"] == 1
    assert contract["minimum_channel_matrix_send_count"] == 1
    assert contract["minimum_output_route_count"] == 1
    assert "## Routing Sends Physical Readback Contract" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "minimum_channel_bus_send_count: `1`" in markdown_summary
    assert "required_false_safety_flags" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_routing_sends_artifact_intake(tmp_path):
    result = CheckResult(
        name="routing_sends_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "routing_sends_physical_readback.v1",
            "surface_id": "routing_and_sends",
            "remediation_action_id": "remediate_routing_and_sends",
            "required_contract_method_id": "routing_sends_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/routing_and_sends",
            ],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_static_dante_route_count": 2,
            "valid_channel_bus_send_count": 3,
            "valid_channel_main_send_count": 4,
            "valid_channel_matrix_send_count": 5,
            "valid_output_route_count": 6,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "routing",
                "fx_send_insert",
                "bus_group_matrix",
                "scene_preset",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_routing_sends_candidate_before_fixture_closure",
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert routing_sends_readback_artifact_intake_summary([result]) == summary[
        "routing_sends_physical_readback_artifact_intake"
    ]
    intake = json_summary["routing_sends_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["live_widening_allowed"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_static_dante_route_count"] == 2
    assert intake["valid_channel_bus_send_count"] == 3
    assert intake["valid_channel_main_send_count"] == 4
    assert intake["valid_channel_matrix_send_count"] == 5
    assert intake["valid_output_route_count"] == 6
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## Routing Sends Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_static_dante_route_count: `2`" in markdown_summary
    assert "valid_channel_bus_send_count: `3`" in markdown_summary
    assert "valid_output_route_count: `6`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary


def test_no_wing_release_check_routing_sends_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="routing_sends_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "routing_sends_physical_readback.v1",
            "surface_id": "routing_and_sends",
            "remediation_action_id": "remediate_routing_and_sends",
            "required_contract_method_id": "routing_sends_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_static_dante_route_count": 2,
            "valid_channel_bus_send_count": 3,
            "valid_channel_main_send_count": 4,
            "valid_channel_matrix_send_count": 5,
            "valid_output_route_count": 6,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["routing"],
            "violations": ["claimed_live_authority"],
        },
    )

    summary = routing_sends_readback_artifact_intake_summary([result])

    assert summary is not None
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == ["claimed_live_authority"]


def test_no_wing_release_check_summary_embeds_snapshots_scenes_artifact_intake(tmp_path):
    result = CheckResult(
        name="snapshots_scenes_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "snapshots_scenes_physical_readback.v1",
            "surface_id": "snapshots_scenes",
            "remediation_action_id": "remediate_snapshots_scenes",
            "required_contract_method_id": "snapshots_scenes_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/snapshots_scenes",
            ],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_snapshot_count": 2,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "scene_preset",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_snapshots_scenes_candidate_before_fixture_closure",
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert snapshots_scenes_readback_artifact_intake_summary([result]) == summary[
        "snapshots_scenes_physical_readback_artifact_intake"
    ]
    intake = json_summary["snapshots_scenes_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["scene_load_allowed"] is False
    assert intake["scene_save_allowed"] is False
    assert intake["live_widening_allowed"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_snapshot_count"] == 2
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## Snapshots Scenes Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_snapshot_count: `2`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "scene_load_allowed: `false`" in markdown_summary
    assert "scene_save_allowed: `false`" in markdown_summary


def test_no_wing_release_check_snapshots_scenes_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="snapshots_scenes_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "snapshots_scenes_physical_readback.v1",
            "surface_id": "snapshots_scenes",
            "remediation_action_id": "remediate_snapshots_scenes",
            "required_contract_method_id": "snapshots_scenes_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_snapshot_count": 2,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": True,
            "scene_load_allowed": True,
            "scene_save_allowed": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["scene_preset"],
            "violations": ["claimed_scene_authority"],
        },
    )

    summary = snapshots_scenes_readback_artifact_intake_summary([result])

    assert summary is not None
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["scene_load_allowed"] is False
    assert summary["scene_save_allowed"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == ["claimed_scene_authority"]


def test_no_wing_release_check_summary_embeds_user_presets_readback_contract(tmp_path):
    result = CheckResult(
        name="user_presets_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "user_presets_physical_readback.v1",
            "surface_id": "user_presets",
            "remediation_action_id": "remediate_user_presets",
            "physical_readback_surface_id": "user_presets",
            "physical_readback_action_id": "remediate_user_presets",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "scene_preset",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "required_top_level_field_count": 10,
            "required_preset_fields": [
                "preset_index",
                "preset_name",
                "preset_content_fingerprint",
                "rollback_reference",
            ],
            "required_coverage_flags": [
                "catalog_identity",
                "preset_content_fingerprint",
                "rollback_reference",
            ],
            "required_false_safety_flags": [
                "live_write",
                "mutates_console_state",
                "live_widening_allowed",
                "full_autonomous_ready",
                "preset_recall_allowed",
                "preset_store_allowed",
            ],
            "minimum_preset_count": 1,
            "allowed_sources": [
                "physical_readback",
                "wing_physical_readback",
                "dlive_physical_readback",
            ],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "preset_recall_events",
                "preset_store_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert user_presets_readback_contract_summary([result]) == summary[
        "user_presets_physical_readback_contract"
    ]
    contract = json_summary["user_presets_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["preset_recall_allowed"] is False
    assert contract["preset_store_allowed"] is False
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "user_presets"
    assert "scene_preset" in contract["promotion_families_requiring_surface"]
    assert "preset_content_fingerprint" in contract["required_preset_fields"]
    assert "rollback_reference" in contract["required_coverage_flags"]
    assert "preset_store_allowed" in contract["required_false_safety_flags"]
    assert contract["minimum_preset_count"] == 1
    assert "## User Presets Physical Readback Contract" in markdown_summary
    assert "preset_recall_allowed: `false`" in markdown_summary
    assert "preset_store_allowed: `false`" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_user_presets_artifact_intake(tmp_path):
    result = CheckResult(
        name="user_presets_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "user_presets_physical_readback.v1",
            "surface_id": "user_presets",
            "remediation_action_id": "remediate_user_presets",
            "required_contract_method_id": "user_presets_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/user_presets",
            ],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_preset_count": 3,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "scene_preset",
                "routing",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_user_presets_candidate_before_fixture_closure",
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert user_presets_readback_artifact_intake_summary([result]) == summary[
        "user_presets_physical_readback_artifact_intake"
    ]
    intake = json_summary["user_presets_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["preset_recall_allowed"] is False
    assert intake["preset_store_allowed"] is False
    assert intake["live_widening_allowed"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_preset_count"] == 3
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["target_consistency"] is True
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## User Presets Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_preset_count: `3`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "preset_recall_allowed: `false`" in markdown_summary
    assert "preset_store_allowed: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_dlive_state_readback_contract(tmp_path):
    result = CheckResult(
        name="dlive_state_physical_readback_contract_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "contract_status": "contract_ready_physical_fixture_missing",
            "schema_id": "dlive_state_physical_readback.v1",
            "surface_id": "dlive_state",
            "remediation_action_id": "remediate_dlive_state",
            "physical_readback_surface_id": "dlive_state",
            "physical_readback_action_id": "remediate_dlive_state",
            "fixture_physical_gap_declared": True,
            "remediation_physical_gap_declared": True,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "routing",
                "performance_adaptation",
            ],
            "required_top_level_field_count": 12,
            "required_channel_fields": [
                "channel",
                "cached_channel_name",
                "fader_raw_14bit",
            ],
            "required_midi_state_fields": [
                "midi_base_channel",
                "raw_state_key_count",
            ],
            "required_parity_fields": [
                "wing_surface_equivalent",
                "mapping_status",
            ],
            "required_coverage_flags": [
                "channel_identity",
                "wing_parity_mapping",
            ],
            "required_false_safety_flags": [
                "live_write",
                "midi_send_allowed",
                "tcp_send_allowed",
                "dlive_write_allowed",
            ],
            "minimum_channel_count": 1,
            "allowed_sources": [
                "physical_readback",
                "dlive_physical_readback",
            ],
            "accepted_row_statuses": ["verified_from_physical_readback"],
            "forbidden_mutation_evidence_fields": [
                "midi_send_events",
                "tcp_send_events",
                "dlive_write_events",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert dlive_state_readback_contract_summary([result]) == summary[
        "dlive_state_physical_readback_contract"
    ]
    contract = json_summary["dlive_state_physical_readback_contract"]
    assert contract["read_only"] is True
    assert contract["live_write"] is False
    assert contract["midi_send_allowed"] is False
    assert contract["tcp_send_allowed"] is False
    assert contract["dlive_write_allowed"] is False
    assert contract["physical_fixture_gap_remains"] is True
    assert contract["accepted_as_physical_fixture_candidate"] is False
    assert contract["live_authority_granted"] is False
    assert contract["target_consistency"] is True
    assert contract["surface_id"] == "dlive_state"
    assert "performance_adaptation" in contract["promotion_families_requiring_surface"]
    assert "fader_raw_14bit" in contract["required_channel_fields"]
    assert "midi_base_channel" in contract["required_midi_state_fields"]
    assert "wing_surface_equivalent" in contract["required_parity_fields"]
    assert "wing_parity_mapping" in contract["required_coverage_flags"]
    assert "midi_send_allowed" in contract["required_false_safety_flags"]
    assert contract["minimum_channel_count"] == 1
    assert "## dLive State Physical Readback Contract" in markdown_summary
    assert "midi_send_allowed: `false`" in markdown_summary
    assert "tcp_send_allowed: `false`" in markdown_summary
    assert "dlive_write_allowed: `false`" in markdown_summary
    assert "physical_fixture_gap_remains: `true`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "target_consistency: `true`" in markdown_summary


def test_no_wing_release_check_summary_embeds_dlive_state_artifact_intake(tmp_path):
    result = CheckResult(
        name="dlive_state_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "dlive_state_physical_readback.v1",
            "surface_id": "dlive_state",
            "remediation_action_id": "remediate_dlive_state",
            "required_contract_method_id": "dlive_state_physical_readback_contract",
            "artifact_search_roots": [
                "artifacts/read_surface_physical_readback/dlive_state",
            ],
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_channel_count": 8,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": [
                "fader",
                "gain",
                "mute",
                "routing",
                "scene_preset",
                "mix_balance",
                "performance_adaptation",
            ],
            "latest_valid_candidate_source_id": "inline_payload_0",
            "latest_valid_candidate_captured_at": "2026-06-05T00:00:00Z",
            "blocked_by": [
                "operator_must_review_valid_dlive_state_candidate_before_fixture_closure",
            ],
            "artifact_source_ids": ["inline_payload_0"],
            "intake_violations": [],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert dlive_state_readback_artifact_intake_summary([result]) == summary[
        "dlive_state_physical_readback_artifact_intake"
    ]
    intake = json_summary["dlive_state_physical_readback_artifact_intake"]
    assert intake["read_only"] is True
    assert intake["live_write"] is False
    assert intake["mutates_console_state"] is False
    assert intake["queries_console"] is False
    assert intake["executes_tests"] is False
    assert intake["execution_allowed_by_intake"] is False
    assert intake["fixture_closure_allowed_by_intake"] is False
    assert intake["midi_send_allowed"] is False
    assert intake["tcp_send_allowed"] is False
    assert intake["dlive_write_allowed"] is False
    assert intake["dlive_apply_allowed"] is False
    assert intake["live_widening_allowed"] is False
    assert intake["full_autonomous_ready"] is False
    assert intake["intake_status"] == "valid_operator_artifact_available"
    assert intake["artifact_count"] == 1
    assert intake["valid_candidate_count"] == 1
    assert intake["invalid_candidate_count"] == 0
    assert intake["valid_channel_count"] == 8
    assert intake["accepted_as_physical_fixture_candidate"] is True
    assert intake["operator_review_required"] is True
    assert intake["physical_fixture_gap_remains"] is True
    assert intake["live_authority_granted"] is False
    assert intake["artifact_source_ids"] == ["inline_payload_0"]
    assert "## dLive State Physical Readback Artifact Intake" in markdown_summary
    assert "artifact_count: `1`" in markdown_summary
    assert "valid_channel_count: `8`" in markdown_summary
    assert "fixture_closure_allowed_by_intake: `false`" in markdown_summary
    assert "midi_send_allowed: `false`" in markdown_summary
    assert "tcp_send_allowed: `false`" in markdown_summary
    assert "dlive_write_allowed: `false`" in markdown_summary
    assert "dlive_apply_allowed: `false`" in markdown_summary


def test_no_wing_release_check_summary_embeds_artifact_review_queue(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "queue_status": "operator_review_candidates_available",
            "intake_surface_count": 10,
            "available_intake_count": 10,
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "review_candidate_count": 1,
            "missing_artifact_surface_count": 9,
            "unsafe_authority_claim_count": 0,
            "candidate_source_id_missing_count": 0,
            "target_inconsistent_surface_count": 0,
            "review_candidate_surface_ids": ["raw_mixer_state_cache"],
            "invalid_surface_ids": [],
            "missing_artifact_surface_ids": ["user_presets"],
            "unsafe_authority_surface_ids": [],
            "candidate_source_id_missing_surface_ids": [],
            "target_inconsistent_surface_ids": [],
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_queue": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": ["fader", "gain"],
            "queue_item_method_ids": [
                "raw_mixer_state_physical_readback_artifact_intake",
                "user_presets_physical_readback_artifact_intake",
            ],
            "blocked_by": [
                "operator_must_review_valid_candidates_before_fixture_closure",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_artifact_review_queue_summary([result]) == summary[
        "read_surface_physical_readback_artifact_review_queue"
    ]
    queue = json_summary["read_surface_physical_readback_artifact_review_queue"]
    assert queue["read_only"] is True
    assert queue["live_write"] is False
    assert queue["mutates_console_state"] is False
    assert queue["queries_console"] is False
    assert queue["executes_tests"] is False
    assert queue["execution_allowed_by_queue"] is False
    assert queue["fixture_closure_allowed_by_queue"] is False
    assert queue["live_widening_allowed"] is False
    assert queue["full_autonomous_ready"] is False
    assert queue["queue_status"] == "operator_review_candidates_available"
    assert queue["intake_surface_count"] == 10
    assert queue["artifact_count"] == 1
    assert queue["valid_candidate_count"] == 1
    assert queue["invalid_candidate_count"] == 0
    assert queue["review_candidate_count"] == 1
    assert queue["review_candidates_available"] is True
    assert queue["missing_artifact_surface_count"] == 9
    assert queue["unsafe_authority_claim_count"] == 0
    assert queue["candidate_source_id_missing_count"] == 0
    assert queue["target_inconsistent_surface_count"] == 0
    assert queue["operator_review_required"] is True
    assert queue["physical_fixture_gap_remains"] is True
    assert queue["live_authority_granted"] is False
    assert queue["review_candidate_surface_ids"] == ["raw_mixer_state_cache"]
    assert "## Read Surface Physical Readback Artifact Review Queue" in markdown_summary
    assert "queue_status: `operator_review_candidates_available`" in markdown_summary
    assert "fixture_closure_allowed_by_queue: `false`" in markdown_summary
    assert "review_candidate_count: `1`" in markdown_summary
    assert "review_candidates_available: `true`" in markdown_summary
    assert "full_autonomous_ready: `false`" in markdown_summary


def test_no_wing_release_check_summary_embeds_artifact_review_queue_artifact(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "queue_path": "/tmp/read_surface_physical_readback_artifact_review_queue.json",
            "queue_type": "read_surface_physical_readback_artifact_review_queue",
            "queue_status": "operator_artifact_capture_required",
            "method_id": "read_surface_physical_readback_artifact_review_queue",
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "intake_surface_count": len(release_check.INTAKE_SPECS),
            "available_intake_count": len(release_check.INTAKE_SPECS),
            "artifact_count": 0,
            "valid_candidate_count": 0,
            "invalid_candidate_count": 0,
            "review_candidate_count": 0,
            "missing_artifact_surface_count": len(release_check.INTAKE_SPECS),
            "unsafe_authority_claim_count": 0,
            "candidate_source_id_missing_count": 0,
            "target_inconsistent_surface_count": 0,
            "review_candidate_surface_ids": [],
            "invalid_surface_ids": [],
            "missing_artifact_surface_ids": [
                str(spec["surface_id"]) for spec in release_check.INTAKE_SPECS
            ],
            "unsafe_authority_surface_ids": [],
            "candidate_source_id_missing_surface_ids": [],
            "target_inconsistent_surface_ids": [],
            "promotion_families_requiring_surface": ["fader", "gain"],
            "queue_item_method_ids": [
                str(spec["method_id"]) for spec in release_check.INTAKE_SPECS
            ],
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "blocked_by": [
                "operator_must_capture_missing_physical_readback_artifacts"
            ],
            "operator_sequence_count": 5,
            "focused_validation": [
                "tests/test_read_surface_physical_readback_artifact_review_queue.py",
                "tests/test_no_wing_release_check.py",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_artifact_review_queue_artifact_summary(
        [result]
    ) == summary["read_surface_physical_readback_artifact_review_queue_artifact"]
    artifact = json_summary[
        "read_surface_physical_readback_artifact_review_queue_artifact"
    ]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["operator_required"] is True
    assert artifact["execution_allowed_by_queue"] is False
    assert artifact["fixture_closure_allowed_by_queue"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["queue_status"] == "operator_artifact_capture_required"
    assert artifact["review_candidate_count"] == 0
    assert artifact["review_candidates_available"] is False
    assert artifact["missing_artifact_surface_count"] == len(release_check.INTAKE_SPECS)
    assert artifact["physical_fixture_gap_remains"] is True
    assert artifact["violations"] == []
    assert (
        "## Read Surface Physical Readback Artifact Review Queue Artifact"
        in markdown_summary
    )
    assert "fixture_closure_allowed_by_queue: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "operator_must_capture_missing_physical_readback_artifacts" in markdown_summary


def test_no_wing_release_check_summary_embeds_fixture_acceptance_ledger(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_fixture_acceptance_ledger_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "ledger_status": "operator_acceptance_required",
            "queue_status": "operator_review_candidates_available",
            "review_candidate_count": 2,
            "acceptance_record_count": 1,
            "accepted_record_count": 1,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "candidate_trace_missing_count": 0,
            "candidate_trace_mismatch_count": 0,
            "unreviewed_candidate_count": 1,
            "accepted_surface_ids": ["raw_mixer_state_cache"],
            "rejected_surface_ids": [],
            "invalid_record_ids": [],
            "unreviewed_candidate_surface_ids": ["dlive_state"],
            "operator_acceptance_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_acceptance": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "blocked_by": [
                "operator_must_accept_or_reject_all_review_candidates",
            ],
            "violations": ["claimed_acceptance_authority"],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_fixture_acceptance_ledger_summary([result]) == summary[
        "read_surface_physical_readback_fixture_acceptance_ledger"
    ]
    ledger = json_summary["read_surface_physical_readback_fixture_acceptance_ledger"]
    assert ledger["read_only"] is True
    assert ledger["live_write"] is False
    assert ledger["mutates_console_state"] is False
    assert ledger["queries_console"] is False
    assert ledger["executes_tests"] is False
    assert ledger["execution_allowed_by_acceptance"] is False
    assert ledger["fixture_closure_allowed_by_acceptance"] is False
    assert ledger["live_widening_allowed"] is False
    assert ledger["live_authority_granted"] is False
    assert ledger["full_autonomous_ready"] is False
    assert ledger["ledger_status"] == "operator_acceptance_required"
    assert ledger["queue_status"] == "operator_review_candidates_available"
    assert ledger["review_candidate_count"] == 2
    assert ledger["acceptance_record_count"] == 1
    assert ledger["accepted_record_count"] == 1
    assert ledger["candidate_trace_missing_count"] == 0
    assert ledger["candidate_trace_mismatch_count"] == 0
    assert ledger["unreviewed_candidate_count"] == 1
    assert ledger["operator_acceptance_required"] is True
    assert ledger["operator_review_required"] is True
    assert ledger["physical_fixture_gap_remains"] is True
    assert ledger["accepted_surface_ids"] == ["raw_mixer_state_cache"]
    assert ledger["unreviewed_candidate_surface_ids"] == ["dlive_state"]
    assert ledger["violations"] == ["claimed_acceptance_authority"]
    assert "## Read Surface Physical Readback Fixture Acceptance Ledger" in markdown_summary
    assert "fixture_closure_allowed_by_acceptance: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "full_autonomous_ready: `false`" in markdown_summary
    assert "accepted_record_count: `1`" in markdown_summary
    assert "candidate_trace_missing_count: `0`" in markdown_summary
    assert "candidate_trace_mismatch_count: `0`" in markdown_summary


def test_no_wing_release_check_summary_embeds_fixture_closure_plan(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_fixture_closure_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "manual_fixture_closure_candidate_available",
            "ledger_status": "operator_acceptance_records_available",
            "review_candidate_count": 2,
            "acceptance_record_count": 2,
            "accepted_record_count": 1,
            "rejected_record_count": 1,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "closure_candidate_count": 1,
            "closure_candidate_surface_ids": ["raw_mixer_state_cache"],
            "rejected_surface_ids": ["dlive_state"],
            "invalid_record_ids": [],
            "unreviewed_candidate_surface_ids": [],
            "manual_fixture_closure_candidate_available": True,
            "manual_fixture_closure_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_update_allowed_by_plan": True,
            "fixture_closure_allowed_by_plan": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "blocked_by": [
                "manual_fixture_baseline_update_required_before_gap_closure",
            ],
            "violations": ["claimed_closure_authority"],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_fixture_closure_plan_summary([result]) == summary[
        "read_surface_physical_readback_fixture_closure_plan"
    ]
    plan = json_summary["read_surface_physical_readback_fixture_closure_plan"]
    assert plan["read_only"] is True
    assert plan["live_write"] is False
    assert plan["mutates_console_state"] is False
    assert plan["queries_console"] is False
    assert plan["executes_tests"] is False
    assert plan["execution_allowed_by_plan"] is False
    assert plan["fixture_update_allowed_by_plan"] is False
    assert plan["fixture_closure_allowed_by_plan"] is False
    assert plan["live_widening_allowed"] is False
    assert plan["live_authority_granted"] is False
    assert plan["full_autonomous_ready"] is False
    assert plan["plan_status"] == "manual_fixture_closure_candidate_available"
    assert plan["ledger_status"] == "operator_acceptance_records_available"
    assert plan["closure_candidate_count"] == 1
    assert plan["closure_candidate_surface_ids"] == ["raw_mixer_state_cache"]
    assert plan["rejected_surface_ids"] == ["dlive_state"]
    assert plan["manual_fixture_closure_candidate_available"] is True
    assert plan["manual_fixture_closure_required"] is True
    assert plan["operator_review_required"] is True
    assert plan["physical_fixture_gap_remains"] is True
    assert plan["violations"] == ["claimed_closure_authority"]
    assert "## Read Surface Physical Readback Fixture Closure Plan" in markdown_summary
    assert "fixture_update_allowed_by_plan: `false`" in markdown_summary
    assert "fixture_closure_allowed_by_plan: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary
    assert "full_autonomous_ready: `false`" in markdown_summary
    assert "closure_candidate_count: `1`" in markdown_summary


def test_no_wing_release_check_summary_embeds_fixture_closure_plan_artifact(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_fixture_closure_plan_artifact_schema",
        status="passed",
        started_at="2026-06-07T00:00:00Z",
        finished_at="2026-06-07T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_path": "/tmp/read_surface_physical_readback_fixture_closure_plan.json",
            "plan_type": "read_surface_physical_readback_fixture_closure_plan",
            "plan_status": "fixture_closure_not_ready",
            "method_id": "read_surface_physical_readback_fixture_closure_plan",
            "source_method_id": "read_surface_physical_readback_fixture_acceptance_ledger",
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "ledger_status": "operator_acceptance_required",
            "review_candidate_count": 0,
            "acceptance_record_count": 0,
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "closure_candidate_count": 0,
            "closure_candidate_surface_ids": [],
            "rejected_surface_ids": [],
            "invalid_record_ids": [],
            "unreviewed_candidate_surface_ids": [],
            "manual_fixture_closure_candidate_available": False,
            "manual_fixture_closure_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "blocked_by": [
                "accepted_physical_readback_candidate_required_before_fixture_closure"
            ],
            "next_global_step": "Operator needs accepted physical readback candidates first.",
            "operator_sequence_count": 5,
            "focused_validation": [
                "tests/test_read_surface_physical_readback_fixture_closure_plan.py",
                "tests/test_no_wing_release_check.py",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-07T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_fixture_closure_plan_artifact_summary(
        [result]
    ) == summary["read_surface_physical_readback_fixture_closure_plan_artifact"]
    artifact = json_summary[
        "read_surface_physical_readback_fixture_closure_plan_artifact"
    ]
    assert artifact["read_only"] is True
    assert artifact["live_write"] is False
    assert artifact["queries_console"] is False
    assert artifact["executes_tests"] is False
    assert artifact["operator_required"] is True
    assert artifact["execution_allowed_by_plan"] is False
    assert artifact["fixture_update_allowed_by_plan"] is False
    assert artifact["fixture_closure_allowed_by_plan"] is False
    assert artifact["live_authority_granted"] is False
    assert artifact["full_autonomous_ready"] is False
    assert artifact["plan_status"] == "fixture_closure_not_ready"
    assert artifact["closure_candidate_count"] == 0
    assert artifact["physical_fixture_gap_remains"] is True
    assert artifact["violations"] == []
    assert (
        "## Read Surface Physical Readback Fixture Closure Plan Artifact"
        in markdown_summary
    )
    assert "fixture_update_allowed_by_plan: `false`" in markdown_summary
    assert "fixture_closure_allowed_by_plan: `false`" in markdown_summary
    assert (
        "accepted_physical_readback_candidate_required_before_fixture_closure"
        in markdown_summary
    )


def test_no_wing_release_check_dlive_state_intake_summary_normalizes_authority_claims():
    result = CheckResult(
        name="dlive_state_physical_readback_artifact_intake_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "intake_status": "valid_operator_artifact_available",
            "schema_id": "dlive_state_physical_readback.v1",
            "surface_id": "dlive_state",
            "remediation_action_id": "remediate_dlive_state",
            "required_contract_method_id": "dlive_state_physical_readback_contract",
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "valid_channel_count": 8,
            "accepted_as_physical_fixture_candidate": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_intake": True,
            "midi_send_allowed": True,
            "tcp_send_allowed": True,
            "dlive_write_allowed": True,
            "dlive_apply_allowed": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "promotion_families_requiring_surface": ["fader", "gain"],
            "violations": ["claimed_dlive_authority"],
        },
    )

    summary = dlive_state_readback_artifact_intake_summary([result])

    assert summary is not None
    assert summary["accepted_as_physical_fixture_candidate"] is False
    assert summary["fixture_closure_allowed_by_intake"] is False
    assert summary["midi_send_allowed"] is False
    assert summary["tcp_send_allowed"] is False
    assert summary["dlive_write_allowed"] is False
    assert summary["dlive_apply_allowed"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == ["claimed_dlive_authority"]


def test_no_wing_release_check_summary_embeds_artifact_review_queue_multiple_candidates(tmp_path):
    result = CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "queue_status": "operator_review_candidates_available",
            "intake_surface_count": 10,
            "available_intake_count": 10,
            "artifact_count": 2,
            "valid_candidate_count": 2,
            "invalid_candidate_count": 0,
            "review_candidate_count": 2,
            "missing_artifact_surface_count": 8,
            "unsafe_authority_claim_count": 0,
            "candidate_source_id_missing_count": 0,
            "target_inconsistent_surface_count": 0,
            "review_candidate_surface_ids": ["raw_mixer_state_cache", "dlive_state"],
            "invalid_surface_ids": [],
            "missing_artifact_surface_ids": ["user_presets"],
            "unsafe_authority_surface_ids": [],
            "candidate_source_id_missing_surface_ids": [],
            "target_inconsistent_surface_ids": [],
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "fixture_closure_allowed_by_queue": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": ["fader", "gain"],
            "queue_item_method_ids": [
                "raw_mixer_state_physical_readback_artifact_intake",
                "dlive_state_physical_readback_artifact_intake",
            ],
            "blocked_by": [
                "operator_must_review_valid_candidates_before_fixture_closure",
            ],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert read_surface_physical_readback_artifact_review_queue_summary([result]) == summary[
        "read_surface_physical_readback_artifact_review_queue"
    ]
    queue = json_summary["read_surface_physical_readback_artifact_review_queue"]
    assert queue["read_only"] is True
    assert queue["live_write"] is False
    assert queue["mutates_console_state"] is False
    assert queue["queries_console"] is False
    assert queue["executes_tests"] is False
    assert queue["execution_allowed_by_queue"] is False
    assert queue["fixture_closure_allowed_by_queue"] is False
    assert queue["live_widening_allowed"] is False
    assert queue["full_autonomous_ready"] is False
    assert queue["queue_status"] == "operator_review_candidates_available"
    assert queue["review_candidate_count"] == 2
    assert queue["review_candidates_available"] is True
    assert queue["missing_artifact_surface_count"] == 8
    assert queue["operator_review_required"] is True
    assert queue["physical_fixture_gap_remains"] is True
    assert queue["live_authority_granted"] is False
    assert queue["review_candidate_surface_ids"] == ["raw_mixer_state_cache", "dlive_state"]
    assert "## Read Surface Physical Readback Artifact Review Queue" in markdown_summary
    assert "review_candidate_count: `2`" in markdown_summary
    assert "review_candidates_available: `true`" in markdown_summary
    assert "fixture_closure_allowed_by_queue: `false`" in markdown_summary
    assert "live_authority_granted: `false`" in markdown_summary


def test_no_wing_release_check_artifact_review_queue_summary_normalizes_authority_claims():
    result = CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "queue_status": "operator_review_candidates_available",
            "intake_surface_count": 10,
            "available_intake_count": 10,
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "review_candidate_count": 1,
            "unsafe_authority_claim_count": 0,
            "candidate_source_id_missing_count": 0,
            "target_inconsistent_surface_count": 0,
            "fixture_closure_allowed_by_queue": True,
            "live_authority_granted": True,
            "full_autonomous_ready": True,
            "review_candidate_surface_ids": ["raw_mixer_state_cache"],
            "blocked_by": ["unsafe_claim"],
            "violations": ["claimed_queue_authority"],
        },
    )

    summary = read_surface_physical_readback_artifact_review_queue_summary([result])

    assert summary is not None
    assert summary["review_candidates_available"] is False
    assert summary["fixture_closure_allowed_by_queue"] is False
    assert summary["live_authority_granted"] is False
    assert summary["full_autonomous_ready"] is False
    assert summary["violations"] == ["claimed_queue_authority"]


def test_no_wing_release_check_summary_embeds_next_wing_proof_plan(tmp_path):
    result = CheckResult(
        name="next_wing_proof_plan_schema",
        status="passed",
        started_at="2026-06-05T00:00:00Z",
        finished_at="2026-06-05T00:00:01Z",
        duration_sec=1.0,
        detail={
            "plan_status": "operator_proof_required",
            "step_count": 1,
            "planned_value_types": ["gain"],
            "planned_channels": [3],
            "readonly_preflight_required": True,
            "readonly_preflight_candidate_available": True,
            "readonly_preflight_evidence_status": "readonly_preflight_candidate_available",
            "readonly_preflight_blocked_by": [
                "operator_must_review_readonly_preflight_candidate_before_readiness_update"
            ],
            "readonly_preflight_prestart_check_required": True,
            "readonly_preflight_prestart_check_must_pass_before_starting_backend": True,
            "readonly_preflight_prestart_check_command": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "readonly_preflight_prestart_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "readonly_preflight_step_count": 0,
            "readonly_preflight_review_decision_required": True,
            "readonly_preflight_review_decision_accepted": True,
            "readonly_preflight_review_decision_status": "operator_review_decision_records_available",
            "readonly_preflight_review_decision_blocked_by": [
                "readiness_matrix_update_still_requires_separate_manual_action"
            ],
            "readonly_preflight_review_decision_accepted_record_count": 1,
            "readonly_preflight_review_decision_accepted_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "readonly_preflight_review_decision_candidate_prestart_contract_valid": True,
            "readonly_preflight_review_decision_candidate_prestart_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "readonly_preflight_review_decision_rejected_source_ids": [],
            "readonly_preflight_review_decision_invalid_record_count": 0,
            "readonly_preflight_review_decision_invalid_record_ids": [],
            "readonly_preflight_review_decision_duplicate_source_ids": [],
            "readonly_preflight_review_decision_conflicting_source_ids": [],
            "readonly_preflight_review_decision_unreviewed_candidate_count": 0,
            "global_blockers": ["operator_must_approve_and_execute_manually"],
            "violations": [],
        },
    )

    summary = write_summary(tmp_path, [result], generated_at="2026-06-05T00:00:00Z")
    json_summary = json.loads((tmp_path / "summary.json").read_text())
    markdown_summary = (tmp_path / "summary.md").read_text()

    assert next_wing_proof_plan_summary([result]) == summary["next_wing_proof_plan"]
    assert json_summary["next_wing_proof_plan"]["live_write"] is False
    assert json_summary["next_wing_proof_plan"]["execution_allowed_by_plan"] is False
    assert json_summary["next_wing_proof_plan"]["operator_required"] is True
    assert json_summary["next_wing_proof_plan"]["planned_value_types"] == ["gain"]
    assert json_summary["next_wing_proof_plan"]["readonly_preflight_required"] is True
    assert json_summary["next_wing_proof_plan"]["readonly_preflight_candidate_available"] is True
    assert (
        json_summary["next_wing_proof_plan"]["readonly_preflight_evidence_status"]
        == "readonly_preflight_candidate_available"
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_prestart_check_required"
        ]
        is True
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_prestart_check_must_pass_before_starting_backend"
        ]
        is True
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_prestart_check_command"
        ]
        == [
            "PYTHONPATH=backend",
            "python3",
            "tools/no_wing_release_check.py",
            "--quick",
        ]
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_prestart_check_failure_detail_fields"
        ]
        == list(WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS)
    )
    assert json_summary["next_wing_proof_plan"]["readonly_preflight_step_count"] == 0
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_required"
        ]
        is True
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_accepted"
        ]
        is True
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_status"
        ]
        == "operator_review_decision_records_available"
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_accepted_source_ids"
        ]
        == ["artifacts/wing_readonly_preflight/run/summary.json"]
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_candidate_prestart_source_ids"
        ]
        == ["artifacts/wing_readonly_preflight/run/summary.json"]
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_rejected_source_ids"
        ]
        == []
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_invalid_record_ids"
        ]
        == []
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_duplicate_source_ids"
        ]
        == []
    )
    assert (
        json_summary["next_wing_proof_plan"][
            "readonly_preflight_review_decision_conflicting_source_ids"
        ]
        == []
    )
    assert "## Next WING Proof Plan" in markdown_summary
    assert 'planned_value_types: `["gain"]`' in markdown_summary
    assert "readonly_preflight_candidate_available: `true`" in markdown_summary
    assert "readonly_preflight_prestart_check_required: `true`" in markdown_summary
    assert (
        "readonly_preflight_prestart_check_must_pass_before_starting_backend: `true`"
        in markdown_summary
    )
    assert "readonly_preflight_prestart_check_command" in markdown_summary
    assert "readonly_preflight_prestart_check_failure_detail_fields" in markdown_summary
    assert "listener_processes" in markdown_summary
    assert "tools/no_wing_release_check.py" in markdown_summary
    assert "readonly_preflight_step_count: `0`" in markdown_summary
    assert "readonly_preflight_review_decision_accepted: `true`" in markdown_summary
    assert (
        'readonly_preflight_review_decision_accepted_source_ids: '
        '`["artifacts/wing_readonly_preflight/run/summary.json"]`'
        in markdown_summary
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: `true`"
        in markdown_summary
    )
    assert "readonly_preflight_review_decision_candidate_prestart_source_ids" in markdown_summary
    assert "readonly_preflight_review_decision_rejected_source_ids: `[]`" in markdown_summary
    assert "readonly_preflight_review_decision_duplicate_source_ids: `[]`" in markdown_summary


def test_no_wing_release_check_records_command_timeout(tmp_path):
    check = CommandCheck(
        name="slow_check",
        command=(
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(10)",
        ),
        cwd=str(Path.cwd()),
        timeout_sec=2.0,
    )

    result = run_command_check(check, tmp_path)

    assert result.status == "timeout"
    assert result.returncode is None
    assert result.detail == {"timeout_sec": 2.0, "reason": "command_timeout"}
    assert "started" in Path(result.stdout_log).read_text()
    assert "Command timed out after 2.0 seconds." in Path(result.stderr_log).read_text()


def test_backend_port_check_records_listener_owner_when_busy(monkeypatch):
    listener_processes = [
        {"pid": 87210, "command": "Python", "name": "TCP *:8765 (LISTEN)"}
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        result = release_check.check_port_free(port=port)

    assert result.status == "failed"
    assert result.detail["listener_detected"] is True
    assert result.detail["listener_processes"] == listener_processes
    assert result.detail["listener_confirmation_checklist"] == list(
        release_check.BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST
    )
    assert result.detail["backend_port_prestart_check_required"] is True
    assert (
        result.detail["backend_port_prestart_check_must_pass_before_starting_backend"]
        is True
    )
    assert result.detail["backend_port_prestart_check_command"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    assert result.detail["backend_port_prestart_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert result.detail["backend_port_prestart_gate_execution_allowed"] is False
    assert (
        result.detail["backend_port_prestart_gate_readiness_closure_allowed"] is False
    )
    assert result.detail["backend_port_prestart_gate_live_authority_granted"] is False
    assert result.detail["preflight_blocked_until_listener_confirmed"] is True
    assert result.detail["intended_backend_confirmation_required"] is True
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert (
        result.detail["fail_closed_reason"]
        == "backend_port_listener_requires_operator_confirmation"
    )
    assert "operator_next_step" in result.detail
    assert "listener_process_error" not in result.detail


def test_backend_port_check_accepts_operator_confirmed_intended_listener(
    tmp_path, monkeypatch
):
    command_line = "python3 backend/server.py --mode wing"
    listener_processes = [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": command_line,
        }
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    confirmation_path = tmp_path / "backend-listener-confirmation.json"
    confirmation_path.write_text(
        json.dumps(
            {
                "type": release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ARTIFACT_TYPE,
                "status": release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS,
                "host": "127.0.0.1",
                "port": 8765,
                "confirms_intended_backend": True,
                "reviewed_listener_pid": 87210,
                "reviewed_listener_command_line": command_line,
                "operator_reviewed_at": "2026-06-13T09:15:00Z",
                "operator_name": "operator",
                "safe_to_stop_listener_by_gate": False,
                "live_authority_granted": False,
            }
        ),
        encoding="utf-8",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
        payload["port"] = port
        confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = release_check.check_port_free(
            port=port,
            listener_confirmation_path=confirmation_path,
        )

    assert result.status == "passed"
    assert result.detail["listener_detected"] is True
    assert result.detail["listener_confirmation_status"] == "accepted"
    assert result.detail["intended_backend_confirmed"] is True
    assert result.detail["preflight_blocked_until_listener_confirmed"] is False
    assert result.detail["intended_backend_confirmation_required"] is False
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert result.detail["live_authority_granted"] is False
    assert result.detail["listener_confirmation_violations"] == []
    assert "fail_closed_reason" not in result.detail


def test_backend_listener_confirmation_template_requires_operator_edit(tmp_path):
    command_line = "python3 backend/server.py --mode wing"
    template_path = tmp_path / "backend-listener-confirmation-template.json"

    payload = release_check.write_backend_listener_confirmation_template(
        template_path,
        host="127.0.0.1",
        port=8765,
        listener_processes=[
            {
                "pid": 87210,
                "command": "Python",
                "name": "TCP *:8765 (LISTEN)",
                "command_line": command_line,
            }
        ],
        generated_at="2026-06-13T10:30:00Z",
    )

    assert template_path.exists()
    persisted = json.loads(template_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["type"] == release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ARTIFACT_TYPE
    assert (
        payload["status"]
        == release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_STATUS
    )
    assert payload["confirms_intended_backend"] is False
    assert payload["reviewed_listener_pid"] == 87210
    assert payload["reviewed_listener_command_line"] == command_line
    assert payload["operator_reviewed_at"] == "REPLACE_WITH_OPERATOR_REVIEW_UTC"
    assert payload["operator_name"] == "REPLACE_WITH_OPERATOR_NAME"
    assert payload["safe_to_stop_listener_by_gate"] is False
    assert payload["live_authority_granted"] is False
    assert payload["accepted_by_template"] is False
    assert payload["execution_allowed_by_template"] is False
    assert payload["readiness_closure_allowed_by_template"] is False
    assert "set status to operator_confirmed_intended_backend" in " ".join(
        payload["required_operator_edits"]
    )
    assert payload["listener_candidates"] == [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": command_line,
        }
    ]


def test_backend_listener_confirmation_template_stays_invalid_until_edited(
    tmp_path, monkeypatch
):
    command_line = "python3 backend/server.py --mode wing"
    listener_processes = [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": command_line,
        }
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    confirmation_path = tmp_path / "backend-listener-confirmation-template.json"
    release_check.write_backend_listener_confirmation_template(
        confirmation_path,
        host="127.0.0.1",
        port=8765,
        listener_processes=listener_processes,
        generated_at="2026-06-13T10:30:00Z",
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
        payload["port"] = port
        confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = release_check.check_port_free(
            port=port,
            listener_confirmation_path=confirmation_path,
        )

    assert result.status == "failed"
    assert result.detail["listener_confirmation_status"] == "invalid"
    assert (
        "listener_confirmation_status_not_accepted"
        in result.detail["listener_confirmation_violations"]
    )
    assert (
        "listener_confirmation_intended_backend_not_true"
        in result.detail["listener_confirmation_violations"]
    )
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert result.detail["live_authority_granted"] is False


def test_backend_listener_confirmation_rejects_partially_edited_template(
    tmp_path, monkeypatch
):
    command_line = "python3 backend/server.py --mode wing"
    listener_processes = [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": command_line,
        }
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    confirmation_path = tmp_path / "backend-listener-confirmation-template.json"
    payload = release_check.write_backend_listener_confirmation_template(
        confirmation_path,
        host="127.0.0.1",
        port=8765,
        listener_processes=listener_processes,
        generated_at="2026-06-13T10:30:00Z",
    )
    payload["status"] = release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS
    payload["confirms_intended_backend"] = True

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload["port"] = port
        confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = release_check.check_port_free(
            port=port,
            listener_confirmation_path=confirmation_path,
        )

    assert result.status == "failed"
    assert result.detail["listener_confirmation_status"] == "invalid"
    violations = result.detail["listener_confirmation_violations"]
    assert "listener_confirmation_operator_reviewed_at_placeholder" in violations
    assert "listener_confirmation_operator_name_placeholder" in violations
    assert "listener_confirmation_template_status_still_pending" in violations
    assert "listener_confirmation_accepted_by_template_not_true" in violations
    assert result.detail["preflight_blocked_until_listener_confirmed"] is True
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert result.detail["live_authority_granted"] is False


def test_backend_listener_confirmation_accepts_fully_edited_template(
    tmp_path, monkeypatch
):
    command_line = "python3 backend/server.py --mode wing"
    listener_processes = [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": command_line,
        }
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    confirmation_path = tmp_path / "backend-listener-confirmation-template.json"
    payload = release_check.write_backend_listener_confirmation_template(
        confirmation_path,
        host="127.0.0.1",
        port=8765,
        listener_processes=listener_processes,
        generated_at="2026-06-13T10:30:00Z",
    )
    payload.update(
        {
            "status": release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS,
            "confirms_intended_backend": True,
            "operator_reviewed_at": "2026-06-13T10:35:00Z",
            "operator_name": "operator",
            "template_status": (
                release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_REVIEWED_STATUS
            ),
            "accepted_by_template": True,
        }
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload["port"] = port
        confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = release_check.check_port_free(
            port=port,
            listener_confirmation_path=confirmation_path,
        )

    assert result.status == "passed"
    assert result.detail["listener_confirmation_status"] == "accepted"
    assert result.detail["intended_backend_confirmed"] is True
    assert result.detail["listener_confirmation_violations"] == []
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert result.detail["live_authority_granted"] is False


def test_backend_port_check_rejects_mismatched_listener_confirmation(
    tmp_path, monkeypatch
):
    listener_processes = [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": "python3 backend/server.py --mode wing",
        }
    ]
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: (listener_processes, None),
    )
    confirmation_path = tmp_path / "backend-listener-confirmation.json"
    confirmation_path.write_text(
        json.dumps(
            {
                "type": release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ARTIFACT_TYPE,
                "status": release_check.BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS,
                "host": "127.0.0.1",
                "port": 8765,
                "confirms_intended_backend": True,
                "reviewed_listener_pid": 87210,
                "reviewed_listener_command_line": "python3 other.py",
                "operator_reviewed_at": "2026-06-13T09:15:00Z",
                "operator_name": "operator",
                "safe_to_stop_listener_by_gate": False,
                "live_authority_granted": False,
            }
        ),
        encoding="utf-8",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
        payload["port"] = port
        confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

        result = release_check.check_port_free(
            port=port,
            listener_confirmation_path=confirmation_path,
        )

    assert result.status == "failed"
    assert result.detail["listener_confirmation_status"] == "invalid"
    assert result.detail["preflight_blocked_until_listener_confirmed"] is True
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert (
        "listener_confirmation_reviewed_command_line_mismatch"
        in result.detail["listener_confirmation_violations"]
    )
    assert (
        result.detail["fail_closed_reason"]
        == "backend_port_listener_requires_operator_confirmation"
    )


def test_backend_port_check_records_listener_owner_probe_error(monkeypatch):
    monkeypatch.setattr(
        release_check,
        "_listener_processes_for_port",
        lambda port: ([], "lsof_not_available"),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        result = release_check.check_port_free(port=port)

    assert result.status == "failed"
    assert result.detail["listener_detected"] is True
    assert result.detail["listener_processes"] == []
    assert result.detail["listener_process_error"] == "lsof_not_available"
    assert result.detail["preflight_blocked_until_listener_confirmed"] is True
    assert result.detail["safe_to_stop_listener_by_gate"] is False
    assert (
        result.detail["fail_closed_reason"]
        == "backend_port_listener_requires_operator_confirmation"
    )


def test_listener_processes_for_port_parses_lsof_field_output(monkeypatch):
    def fake_run(command, capture_output, check, text, timeout):
        assert capture_output is True
        assert check is False
        assert text is True
        assert timeout == 1.0
        if command == (
            "lsof",
            "-nP",
            "-iTCP:8765",
            "-sTCP:LISTEN",
            "-F",
            "pcn",
        ):
            return release_check.subprocess.CompletedProcess(
                command,
                0,
                stdout="p87210\ncPython\nnTCP *:8765 (LISTEN)\n",
                stderr="",
            )
        assert command == ("ps", "-p", "87210", "-o", "command=")
        return release_check.subprocess.CompletedProcess(
            command,
            0,
            stdout="python3 backend/server.py --mode wing\n",
            stderr="",
        )

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)

    processes, error = release_check._listener_processes_for_port(8765)

    assert error is None
    assert processes == [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line": "python3 backend/server.py --mode wing",
        }
    ]


def test_listener_processes_for_port_records_command_line_probe_failure(monkeypatch):
    def fake_run(command, capture_output, check, text, timeout):
        assert capture_output is True
        assert check is False
        assert text is True
        assert timeout == 1.0
        if command[0] == "lsof":
            return release_check.subprocess.CompletedProcess(
                command,
                0,
                stdout="p87210\ncPython\nnTCP *:8765 (LISTEN)\n",
                stderr="",
            )
        assert command == ("ps", "-p", "87210", "-o", "command=")
        return release_check.subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="permission denied",
        )

    monkeypatch.setattr(release_check.subprocess, "run", fake_run)

    processes, error = release_check._listener_processes_for_port(8765)

    assert error is None
    assert processes == [
        {
            "pid": 87210,
            "command": "Python",
            "name": "TCP *:8765 (LISTEN)",
            "command_line_error": "ps_failed:permission denied",
        }
    ]


def test_no_wing_release_check_stops_when_backend_port_is_busy(tmp_path, monkeypatch):
    def fake_busy_port_check(
        port: int,
        host: str = "127.0.0.1",
        listener_confirmation_path: Path | None = None,
    ) -> CheckResult:
        return CheckResult(
            name="backend_port_free",
            status="failed",
            started_at="2026-05-26T00:00:00Z",
            finished_at="2026-05-26T00:00:01Z",
            duration_sec=1.0,
            detail={
                "host": host,
                "listener_detected": True,
                "listener_processes": [
                    {"pid": 87210, "command": "Python", "name": "TCP *:8765 (LISTEN)"}
                ],
                "listener_confirmation_checklist": list(
                    release_check.BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST
                ),
                "preflight_blocked_until_listener_confirmed": True,
                "intended_backend_confirmation_required": True,
                "safe_to_stop_listener_by_gate": False,
                "fail_closed_reason": "backend_port_listener_requires_operator_confirmation",
                "operator_next_step": (
                    "Stop or reconfigure the unexpected local listener only after "
                    "confirming it is not the intended Automixer backend, then rerun "
                    "the no-WING gate and WING read-only preflight."
                ),
                "policy": "port_must_not_have_unexpected_listener",
                "port": port,
            },
        )

    monkeypatch.setattr(release_check, "check_port_free", fake_busy_port_check)

    summary = run_release_checks(
        artifact_root=tmp_path,
        include_frontend=False,
        quick=True,
        backend_port=8765,
    )

    assert summary["status"] == "failed"
    assert summary["failed_checks"] == ["backend_port_free"]
    assert summary["live_write"] is False
    assert summary["results"][0]["detail"]["preflight_blocked_until_listener_confirmed"] is True
    assert summary["results"][0]["detail"]["listener_confirmation_checklist"] == list(
        release_check.BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST
    )
    assert summary["results"][0]["detail"]["intended_backend_confirmation_required"] is True
    assert summary["results"][0]["detail"]["safe_to_stop_listener_by_gate"] is False
    assert (
        summary["results"][0]["detail"]["fail_closed_reason"]
        == "backend_port_listener_requires_operator_confirmation"
    )
    assert (
        summary["results"][0]["detail"]["listener_confirmation_template_status"]
        == "pending_operator_edit"
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_acceptance_status"
        ]
        == release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_STATUS
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_accepted_by_template"
        ]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_execution_allowed_by_template"
        ]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_readiness_closure_allowed_by_template"
        ]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_operator_reviewed_at"
        ]
        == "REPLACE_WITH_OPERATOR_REVIEW_UTC"
    )
    assert (
        summary["results"][0]["detail"]["listener_confirmation_template_operator_name"]
        == "REPLACE_WITH_OPERATOR_NAME"
    )
    assert summary["results"][0]["detail"][
        "listener_confirmation_template_operator_placeholder_fields"
    ] == ["operator_reviewed_at", "operator_name"]
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_safe_to_stop_listener_by_gate"
        ]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "listener_confirmation_template_live_authority_granted"
        ]
        is False
    )
    assert summary["results"][0]["detail"][
        "backend_port_prestart_check_required"
    ] is True
    assert (
        summary["results"][0]["detail"][
            "backend_port_prestart_check_must_pass_before_starting_backend"
        ]
        is True
    )
    assert summary["results"][0]["detail"][
        "backend_port_prestart_check_command"
    ] == list(WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND)
    assert summary["results"][0]["detail"][
        "backend_port_prestart_check_failure_detail_fields"
    ] == list(WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS)
    assert (
        summary["results"][0]["detail"]["backend_port_prestart_gate_execution_allowed"]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "backend_port_prestart_gate_readiness_closure_allowed"
        ]
        is False
    )
    assert (
        summary["results"][0]["detail"][
            "backend_port_prestart_gate_live_authority_granted"
        ]
        is False
    )
    template_path = Path(
        summary["results"][0]["detail"]["listener_confirmation_template_path"]
    )
    assert template_path.exists()
    template_payload = json.loads(template_path.read_text(encoding="utf-8"))
    assert (
        template_payload["status"]
        == release_check.BACKEND_PORT_LISTENER_CONFIRMATION_TEMPLATE_STATUS
    )
    assert template_payload["confirms_intended_backend"] is False
    assert template_payload["safe_to_stop_listener_by_gate"] is False
    assert template_payload["live_authority_granted"] is False
    assert template_payload["execution_allowed_by_template"] is False
    assert template_payload["readiness_closure_allowed_by_template"] is False
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "summary.json").exists()
    assert not (run_dir / "logs").exists()


def test_no_wing_release_check_passes_backend_listener_confirmation(
    tmp_path, monkeypatch
):
    confirmation_path = tmp_path / "backend-listener-confirmation.json"
    seen_confirmation_paths = []

    def fake_backend_check(
        port: int,
        host: str = "127.0.0.1",
        listener_confirmation_path: Path | None = None,
    ) -> CheckResult:
        seen_confirmation_paths.append(listener_confirmation_path)
        return CheckResult(
            name="backend_port_free",
            status="passed",
            started_at="2026-06-13T00:00:00Z",
            finished_at="2026-06-13T00:00:01Z",
            duration_sec=1.0,
            detail={
                "host": host,
                "port": port,
                "listener_detected": True,
                "listener_confirmation_path": str(listener_confirmation_path),
                "listener_confirmation_status": "accepted",
                "intended_backend_confirmed": True,
                "preflight_blocked_until_listener_confirmed": False,
                "intended_backend_confirmation_required": False,
                "safe_to_stop_listener_by_gate": False,
                "live_authority_granted": False,
            },
        )

    monkeypatch.setattr(release_check, "check_port_free", fake_backend_check)
    monkeypatch.setattr(
        release_check,
        "build_checks",
        lambda **kwargs: [],
    )

    summary = run_release_checks(
        artifact_root=tmp_path / "artifacts",
        include_frontend=False,
        quick=True,
        backend_port=8765,
        backend_listener_confirmation=confirmation_path,
    )

    assert summary["status"] == "passed"
    assert seen_confirmation_paths == [confirmation_path]
    assert (
        summary["results"][0]["detail"]["listener_confirmation_path"]
        == str(confirmation_path)
    )
