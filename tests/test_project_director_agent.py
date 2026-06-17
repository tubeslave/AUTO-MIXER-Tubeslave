import hashlib
import importlib.util
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTOR_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "project_director_agent.py"
PROMPT_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"
README_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "README.md"
DOCS_INDEX = PROJECT_ROOT / "Docs" / "INDEX.md"

ENVIRONMENT_BLOCKED_FRONTEND_MARKERS = (
    "environment_blocked_checks",
    "environment_blocked_check_details",
    "local_dev_server_bind_denied",
    "127.0.0.1:3000",
)


def load_director_module():
    sys.path.insert(0, str(DIRECTOR_PATH.parent))
    spec = importlib.util.spec_from_file_location("project_director_agent", DIRECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_wing_mvp_20260614_index_reports():
    index_text = DOCS_INDEX.read_text(encoding="utf-8")
    return sorted(
        set(
            re.findall(
                r"`(Docs/reports/AUTOMIXER_WING_MVP_[^`]+_20260614\.md)`",
                index_text,
            )
        )
    )


def test_project_director_selects_first_non_closed_mvp_blocker():
    director = load_director_module()

    directive = director.build_directive(
        project_root=PROJECT_ROOT,
        lookback_commits=1,
        report_limit=1,
    )

    assert directive["type"] == "project_director_directive"
    assert directive["current_mvp"] == "supervised_wing_rack_proof"
    assert directive["selected_blocker"] == "WING read-only connection proof"
    assert directive["selected_blocker_status"] == "tooling_ready_waits_real_wing"
    assert directive["selected_action_source"] == "Current First Command"
    assert directive["selected_action_sha256"] == hashlib.sha256(
        directive["selected_action"].encode("utf-8")
    ).hexdigest()
    assert directive["current_first_command_baseline_coverage_required"] is True
    assert directive["current_first_command_baseline_coverage_passed"] is True
    assert directive["current_first_command_baseline_missing_markers"] == []
    assert (
        directive["current_first_command_baseline_coverage_marker_count"]
        == len(
            director.update_mvp_progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS
        )
    )
    assert "tools/no_wing_release_check.py --quick" in directive["selected_action"]
    assert "before starting the intended backend/WING session" in directive["selected_action"]
    assert "`Docs/runbooks/WING_READONLY_PREFLIGHT.md`" in directive["selected_action"]
    assert "backend_port_prestart_gate" in directive["selected_action"]
    assert "expected_backend_port_prestart_gate" in directive["selected_action"]
    assert "backend_port_prestart_gate.grants_live_authority: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_gate.must_pass_before_backend_wing_session: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in directive[
        "selected_action"
    ]
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in directive["selected_action"]
    )
    assert (
        "backend_port_prestart_gate.runs_command: prestart_backend_port_check"
        in directive["selected_action"]
    )
    assert "backend_port_prestart_gate.required_markers" in directive[
        "selected_action"
    ]
    assert "Before starting the intended backend/WING session" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_check_command" in directive["selected_action"]
    assert "backend_port_prestart_check_command_text" in directive["selected_action"]
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in directive[
        "selected_action"
    ]
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in directive["selected_action"]
    )
    assert "backend_port_prestart_check_required: true" in directive[
        "selected_action"
    ]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in directive["selected_action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_check_failure_detail_fields" in directive[
        "selected_action"
    ]
    assert "failed busy-port no-WING summary" in directive["selected_action"]
    assert "no-WING handoff summary" in directive["selected_action"]
    assert "listener_processes" in directive["selected_action"]
    assert "listener_process_error" in directive["selected_action"]
    assert "safe_to_stop_listener_by_gate" in directive["selected_action"]
    assert "listener_confirmation_template_status" in directive["selected_action"]
    assert "listener_confirmation_template_path" in directive["selected_action"]
    assert "listener_confirmation_template_acceptance_status" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_accepted_by_template" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_execution_allowed_by_template" in directive[
        "selected_action"
    ]
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template"
        in directive["selected_action"]
    )
    assert "fail_closed_reason" in directive["selected_action"]
    assert "operator_next_step" in directive["selected_action"]
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in directive["selected_action"]
    assert "listener_confirmation_template_path" in directive["selected_action"]
    assert "listener_confirmation_template_status: pending_operator_edit" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_live_authority_granted: false" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_accepted_by_template: false" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_operator_placeholder_fields" in directive[
        "selected_action"
    ]
    assert "listener_confirmation_template_required_edits" in directive[
        "selected_action"
    ]
    assert "set template_status to operator_reviewed_and_edited" in directive[
        "selected_action"
    ]
    assert "set accepted_by_template to true" in directive["selected_action"]
    assert "REPLACE_WITH_*" in directive["selected_action"]
    assert "template_status: operator_reviewed_and_edited" in directive[
        "selected_action"
    ]
    assert "accepted_by_template: true" in directive["selected_action"]
    assert "manual_review_required_before_readiness_update: true" in directive[
        "selected_action"
    ]
    assert "readiness_matrix_update_allowed_by_packet: false" in directive[
        "selected_action"
    ]
    assert "readiness_closure_allowed_by_packet: false" in directive[
        "selected_action"
    ]
    assert "read_only: true" in directive["selected_action"]
    assert "live_write: false" in directive["selected_action"]
    assert "mutates_console_state: false" in directive["selected_action"]
    assert "queries_console: false" in directive["selected_action"]
    assert "executes_tests: false" in directive["selected_action"]
    assert "operator_required: true" in directive["selected_action"]
    assert "expected_plan_required_for_real_run: true" in directive["selected_action"]
    assert "backend_port_prestart_summary_required: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_path_placeholder" in directive[
        "selected_action"
    ]
    assert "preview_command_prestart_summary" in directive["selected_action"]
    assert "preview_command_prestart_summary_matches_plan: true" in directive[
        "selected_action"
    ]
    assert "capture_command_prestart_summary" in directive["selected_action"]
    assert "capture_command_prestart_summary_matches_plan: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_type: no_wing_release_check" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_status: passed" in directive[
        "selected_action"
    ]
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in directive["selected_action"]
    assert "backend_port_prestart_summary_read_only: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_live_write: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_contacts_wing: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_mutates_console_state: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_queries_console: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_executes_tests: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_check_status: passed" in directive[
        "selected_action"
    ]
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in directive["selected_action"]
    assert (
        "physical_preflight_requires_existing_backend_connection: true"
        in directive["selected_action"]
    )
    assert "backend_port_prestart_check_required: true" in directive[
        "selected_action"
    ]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in directive["selected_action"]
    )
    assert "execution_allowed_by_packet: false" in directive["selected_action"]
    assert "live_authority_granted: false" in directive["selected_action"]
    for marker in (
        "summary_md_readiness_closure_allowed_by_packet_denies_authority: true",
        "summary_md_live_authority_granted_denies_authority: true",
        "summary_md_artifact_manifest_contract_cross_surface_readiness_closure_allowed_denies_authority: true",
        "summary_md_artifact_manifest_contract_cross_surface_live_authority_granted_denies_authority: true",
        "summary_md_artifact_manifest_readiness_closure_allowed_denies_authority: true",
        "summary_md_artifact_manifest_live_authority_granted_denies_authority: true",
        "summary_md_artifact_manifest_verifier_readiness_closure_allowed_denies_authority: true",
        "summary_md_artifact_manifest_verifier_live_authority_granted_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_readiness_closure_allowed_by_packet_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_live_authority_granted_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_closure_allowed_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_live_authority_granted_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_readiness_closure_allowed_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_live_authority_granted_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_denies_authority: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_denies_authority: true",
    ):
        assert marker in directive["selected_action"]
    for marker in (
        "artifact_manifest_verifier_read_only: true",
        "artifact_manifest_verifier_live_write: false",
        "artifact_manifest_verifier_contacts_backend: false",
        "artifact_manifest_verifier_executes_tests: false",
        "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_read_only_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_read_only_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_read_only_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_write_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_write_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_write_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_backend_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_backend_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_backend_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_executes_tests_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_executes_tests_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_executes_tests_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_required_fields_missing_empty: true",
        "artifact_manifest_verifier_check_verifier_report_required_fields_match_contract: true",
        "artifact_manifest_verifier_check_verifier_report_read_only_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_live_write_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_contacts_wing_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_contacts_backend_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_mutates_console_state_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_queries_console_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_executes_tests_matches_expected: true",
        "artifact_manifest_verifier_check_failed_checks_missing_check_entries_empty: true",
    ):
        assert marker in directive["selected_action"]
    assert "full_autonomous_ready: false" in directive["selected_action"]
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in directive["selected_action"]
    )
    assert "expected_artifacts" in directive["selected_action"]
    assert "expected_artifact_globs" in directive["selected_action"]
    assert "plan_expected_artifacts" in directive["selected_action"]
    assert "expected_artifacts_match_plan: true" in directive["selected_action"]
    assert "summary_glob_matches_plan: true" in directive["selected_action"]
    assert "event_log_glob_matches_plan: true" in directive["selected_action"]
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_check_operator_sequence_step" in directive[
        "selected_action"
    ]
    assert "prestart_backend_port_check" in directive["selected_action"]
    assert "unexpected listeners" in directive["selected_action"]
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in directive[
        "selected_action"
    ]
    assert "backend_connection_confirmation_step_index: 3" in directive[
        "selected_action"
    ]
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in directive["selected_action"]
    assert "operator_sequence_has_backend_connection_confirmation: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in directive[
        "selected_action"
    ]
    assert "accepted_candidate_prestart_contract_violations_by_source" in directive[
        "selected_action"
    ]
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in directive["selected_action"]
    )
    assert "required_supervised_proof_plan_summary_fields" in directive[
        "selected_action"
    ]
    assert "expected_required_supervised_proof_plan_summary_fields" in directive[
        "selected_action"
    ]
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in directive[
        "selected_action"
    ]
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in directive["selected_action"]
    )
    assert "operator-readable no-WING `summary.md`" in directive["selected_action"]
    assert "operator_markdown_summary_required: true" in directive["selected_action"]
    assert "operator_markdown_summary_required_field_lines" in directive[
        "selected_action"
    ]
    for marker in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert marker in directive["selected_action"]
    assert "operator_markdown_summary_required_field_lines_missing_members: []" in directive[
        "selected_action"
    ]
    assert "operator_markdown_summary_required_field_lines_match_contract: true" in directive[
        "selected_action"
    ]
    assert (
        "operator_markdown_summary_field_lines_rendered: true"
        in directive["selected_action"]
    )
    assert "operator_markdown_summary_missing_field_lines: []" in directive[
        "selected_action"
    ]
    assert "operator_markdown_summary_artifact: summary.md" in directive[
        "selected_action"
    ]
    assert "operator_markdown_summary_sha256" in directive["selected_action"]
    assert "operator_markdown_summary_sha256_available: true" in directive[
        "selected_action"
    ]
    assert "operator_markdown_summary_actual_sha256" in directive["selected_action"]
    assert "operator_markdown_summary_sha256_matches_actual: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_surface_required: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_surface_missing_fields: []" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_surface_required_field_lines" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_id" in directive["selected_action"]
    assert "artifact_manifest_contract_required: true" in directive["selected_action"]
    assert "artifact_manifest_contract_fields" in directive["selected_action"]
    assert "artifact_manifest_contract_sha256" in directive["selected_action"]
    assert "artifact_manifest_contract_sha256_available: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_sha256_matches_expected: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_cross_surface_required: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_cross_surface_required_fields" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_contract_cross_surface_match: true" in directive[
        "selected_action"
    ]
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in directive["selected_action"]
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in directive["selected_action"]
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in directive[
        "selected_action"
    ]
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in directive["selected_action"]
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in directive["selected_action"]
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_path: artifact-manifest.json" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_artifact_count: 2" in directive["selected_action"]
    assert "artifact_manifest_required_artifact_count: 2" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_artifact_count_matches_required: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_required_artifact_ids" in directive["selected_action"]
    assert "artifact_manifest_artifact_ids" in directive["selected_action"]
    assert "artifact_manifest_required_artifacts_present: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_required_artifact_paths" in directive["selected_action"]
    assert "artifact_manifest_artifact_paths" in directive["selected_action"]
    assert "artifact_manifest_required_artifact_paths_present: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_required_artifact_sha256_available: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_json_sha256" in directive["selected_action"]
    assert "artifact_manifest_summary_json_sha256_available: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_summary_md_sha256" in directive["selected_action"]
    assert "artifact_manifest_summary_md_sha256_available: true" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_readiness_closure_allowed: false" in directive[
        "selected_action"
    ]
    assert "artifact_manifest_live_authority_granted: false" in directive[
        "selected_action"
    ]
    assert "readonly_preflight_review_decision_expected_candidate_artifact_roots" in directive[
        "selected_action"
    ]
    assert "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source" in directive[
        "selected_action"
    ]
    assert "readonly_preflight_review_decision_invalid_records" in directive[
        "selected_action"
    ]
    assert (
        "next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations"
        in directive["selected_action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in directive["selected_action"]
    )
    for marker in (
        "backend_port_prestart_summary_read_only: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing: false",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state: false",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_self_check_required: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_required_fields",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_mismatches: []",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_match: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false",
        "project_director_selected_action_value_gate_self_check_required: true",
        "project_director_selected_action_value_gate_self_check_required_markers",
        "project_director_selected_action_value_gate_self_check_missing_markers: []",
        "project_director_selected_action_value_gate_self_check_markers_present: true",
        "project_director_selected_action_value_gate_self_check_readiness_closure_allowed: false",
        "project_director_selected_action_value_gate_self_check_live_authority_granted: false",
        "project_director_selected_action_value_gate_self_check_markdown_surface_required: true",
        "project_director_selected_action_value_gate_self_check_markdown_surface_required_field_lines",
        "project_director_selected_action_value_gate_self_check_markdown_surface_missing_field_lines: []",
        "project_director_selected_action_value_gate_self_check_markdown_surface_field_lines_rendered: true",
        "project_director_selected_action_value_gate_self_check_markdown_surface_readiness_closure_allowed: false",
        "project_director_selected_action_value_gate_self_check_markdown_surface_live_authority_granted: false",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_fields_present: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_required_matches_summary: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_summary: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_summary: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_summary: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_summary: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_required_is_boolean: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_expected: true",
        "artifact_manifest_verifier_check_manifest_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_required_is_boolean: true",
        "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_mismatches_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_match_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed_matches_expected: true",
        "artifact_manifest_verifier_check_verifier_report_wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted_matches_expected: true",
    ):
        assert marker in directive["selected_action"]
    assert "backend_port_prestart_gate_matches_expected: true" in directive[
        "selected_action"
    ]
    assert "declared_backend_port_prestart_gate_matches_expected: true" in directive[
        "selected_action"
    ]
    assert "supervised_prestart_gate_matches_expected: true" in directive[
        "selected_action"
    ]
    assert "declared_supervised_prestart_gate_matches_expected: true" in directive[
        "selected_action"
    ]
    assert "readonly_preflight_review_decision_expected_candidate_prestart_source_ids" in directive[
        "selected_action"
    ]
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source"
        in directive["selected_action"]
    )
    assert "Both maps must be `{}` before supervised proof steps are planned" in directive[
        "selected_action"
    ]
    assert "operator_sequence_backend_connection_confirmation_before_capture: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_backend_connection_confirmation_step" in directive[
        "selected_action"
    ]
    assert "existing backend connection" in directive["selected_action"]
    assert "intended WING Rack" in directive["selected_action"]
    assert "read-only inspection" in directive["selected_action"]
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in directive[
        "selected_action"
    ]
    assert "mvp_blocker_anchor" in directive["selected_action"]
    assert "mvp_blocker_anchor_matches_current_first_open: true" in directive[
        "selected_action"
    ]
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in directive[
        "selected_action"
    ]
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in directive[
        "selected_action"
    ]
    assert "project_director_selected_action_anchor" in directive["selected_action"]
    assert (
        "expected_project_director_selected_action_anchor"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_source: Current First Command"
        in directive["selected_action"]
    )
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in directive["selected_action"]
    )
    assert "project_director_selected_action_sha256" in directive["selected_action"]
    assert (
        "actual_project_director_selected_action_sha256"
        in directive["selected_action"]
    )
    assert (
        "expected_project_director_selected_action_sha256"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_sha256_available: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_sha256_matches_anchor: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_sha256_matches_text: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in directive["selected_action"]
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in directive["selected_action"]
    )
    assert (
        "project_director_current_first_command_baseline_missing_markers: []"
        in directive["selected_action"]
    )
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_count: 12" in directive["selected_action"]
    assert "expected_operator_sequence_count: 12" in directive["selected_action"]
    assert (
        "operator_sequence_count_matches_contract: true"
        in directive["selected_action"]
    )
    assert "operator_sequence_sha256_matches_actual: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_matches_contract: true" in directive["selected_action"]
    assert "operator_sequence_sha256_matches_contract: true" in directive[
        "selected_action"
    ]
    assert (
        "declared_expected_operator_sequence_sha256_matches_contract: true"
        in directive["selected_action"]
    )
    assert "declared_operator_sequence_matches_contract_flag: true" in directive[
        "selected_action"
    ]
    assert (
        "declared_operator_sequence_matches_contract_flag_matches_actual: true"
        in directive["selected_action"]
    )
    assert (
        "declared_operator_sequence_sha256_matches_contract_flag: true"
        in directive["selected_action"]
    )
    assert (
        "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true"
        in directive["selected_action"]
    )
    assert "declared_operator_sequence_count: 12" in directive["selected_action"]
    assert "declared_expected_operator_sequence_count: 12" in directive[
        "selected_action"
    ]
    assert "declared_operator_sequence_count_contract_flag: true" in directive[
        "selected_action"
    ]
    assert "declared_operator_sequence_count_matches_actual: true" in directive[
        "selected_action"
    ]
    assert "declared_operator_sequence_count_matches_contract: true" in directive[
        "selected_action"
    ]
    assert (
        "declared_expected_operator_sequence_count_matches_contract: true"
        in directive["selected_action"]
    )
    assert (
        "declared_operator_sequence_count_contract_flag_matches_declared_count: true"
        in directive["selected_action"]
    )
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in directive[
        "selected_action"
    ]
    assert (
        "expected_mvp_blocker_anchor_digest_review_step_index: 10"
        in directive["selected_action"]
    )
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in directive["selected_action"]
    )
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in directive["selected_action"]
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in directive[
        "selected_action"
    ]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in directive[
        "selected_action"
    ]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in directive["selected_action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in directive["selected_action"]
    )
    assert "mvp_blocker_anchor_sha256" in directive["selected_action"]
    assert "current first open blocker" in directive["selected_action"]
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_has_supervised_prestart_gate: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step" in directive[
        "selected_action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in directive[
        "selected_action"
    ]
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in directive["selected_action"]
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in directive["selected_action"]
    )
    assert "required_supervised_prestart_gate_markers" in directive[
        "selected_action"
    ]
    assert "accepted_decision_ready" in directive["selected_action"]
    assert "valid accepted-candidate prestart trace" in directive["selected_action"]
    assert "expected-artifact-bound accepted source id" in directive["selected_action"]
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in directive["selected_action"]
    assert "supervised_prestart_gate" in directive["selected_action"]
    assert "expected_supervised_prestart_gate" in directive["selected_action"]
    assert "supervised_prestart_gate.grants_live_authority: false" in directive[
        "selected_action"
    ]
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in directive[
        "selected_action"
    ]
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in directive["selected_action"]
    )
    assert "supervised_prestart_gate_required: true" in directive["selected_action"]
    assert (
        "supervised_prestart_gate_operator_sequence_step_index: 11"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_operator_sequence_step_must_be_final: true"
        in directive["selected_action"]
    )
    assert "supervised_prestart_gate_operator_sequence_step" in directive[
        "selected_action"
    ]
    assert (
        "supervised_prestart_gate_operator_sequence_step_matches_summary: true"
        in directive["selected_action"]
    )
    assert "supervised_prestart_gate_required_markers" in directive[
        "selected_action"
    ]
    assert "expected_supervised_prestart_gate_required_markers" in directive[
        "selected_action"
    ]
    assert "expected_required_supervised_prestart_gate_markers" in directive[
        "selected_action"
    ]
    assert (
        "required_supervised_prestart_gate_marker_count: 7"
        in directive["selected_action"]
    )
    assert (
        "expected_required_supervised_prestart_gate_marker_count: 7"
        in directive["selected_action"]
    )
    assert (
        "required_supervised_prestart_gate_marker_count_matches_contract: true"
        in directive["selected_action"]
    )
    assert (
        "required_supervised_prestart_gate_marker_count_matches_supervised_gate: true"
        in directive["selected_action"]
    )
    assert (
        "required_supervised_prestart_gate_markers_match_contract: true"
        in directive["selected_action"]
    )
    assert (
        "required_supervised_prestart_gate_markers_match_supervised_gate: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_required_marker_count: 7"
        in directive["selected_action"]
    )
    assert (
        "expected_supervised_prestart_gate_required_marker_count: 7"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_required_marker_count_matches_contract: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_required_marker_count_matches_summary: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_required_markers_match_contract: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_required_markers_match_summary: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in directive["selected_action"]
    )
    assert "supervised_prestart_gate_grants_live_authority: false" in directive[
        "selected_action"
    ]
    assert (
        "supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true"
        in directive["selected_action"]
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true"
        in directive["selected_action"]
    )
    assert "accepted-candidate prestart trace" in directive["selected_action"]
    assert "valid accepted-candidate request-sequence trace" in directive[
        "selected_action"
    ]
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert directive["selected_evidence_file_count"] == len(directive["selected_evidence_files"])
    missing_index_reports = [
        report
        for report in current_wing_mvp_20260614_index_reports()
        if report not in directive["selected_evidence_files"]
    ]
    assert missing_index_reports == []
    assert (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
        in directive["selected_evidence_files"]
    )
    assert (
        "tools/verify_no_wing_artifact_manifest.py"
        in directive["selected_evidence_files"]
    )
    assert (
        "backend/wing_readonly_preflight_review_decision.py"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_PACKET_20260607.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in directive["selected_evidence_files"]
    )
    for report in (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md",
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md",
        "Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md",
    ):
        assert report in directive["selected_evidence_files"]
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in directive["selected_evidence_files"]
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_VERIFIER_20260615.md"
        in directive["selected_evidence_files"]
    )
    assert directive["stop_broad_work"] is True
    assert directive["continue_goal_mode"] is True
    assert "Docs/MVP_WING_SUPERVISED.md" in directive["required_files"]
    assert "backend/wing_readonly_preflight_contract.py" in directive["required_files"]
    assert (
        "backend/wing_readonly_preflight_review_decision.py"
        in directive["required_files"]
    )
    assert "backend/supervised_wing_proof_plan.py" in directive["required_files"]
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in directive["required_files"]
    assert "tools/wing_readonly_preflight_runner.py" in directive["required_files"]
    assert "tools/review_wing_readonly_preflight_artifacts.py" in directive["required_files"]
    assert (
        "tools/review_wing_readonly_preflight_decisions.py"
        in directive["required_files"]
    )
    assert (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
        in directive["required_files"]
    )
    assert "tools/no_wing_release_check.py" in directive["required_files"]
    assert "tools/verify_no_wing_artifact_manifest.py" in directive["required_files"]
    assert any("Develop correction methods" in item for item in directive["allowed_scope"])


def test_project_director_derives_required_files_from_selected_blocker_evidence():
    director = load_director_module()

    evidence_files = director.extract_evidence_files(
        "`tools/example.py`; `backend/example_contract.py`; "
        "`Docs/runbooks/EXAMPLE.md`; `Docs/reports/example.md`; "
        "`tools/example.py`; plain_text"
    )

    assert evidence_files == [
        "tools/example.py",
        "backend/example_contract.py",
        "Docs/runbooks/EXAMPLE.md",
        "Docs/reports/example.md",
    ]
    assert director.required_file_candidates_from_evidence(evidence_files) == [
        "tools/example.py",
        "backend/example_contract.py",
        "Docs/runbooks/EXAMPLE.md",
    ]


def test_project_director_markdown_is_operator_readable():
    director = load_director_module()
    directive = {
        "generated_at": "2026-06-06T00:00:00Z",
        "current_mvp": "supervised_wing_rack_proof",
        "guard_status": "mixed_needs_review",
        "guard_score": 30,
        "stop_broad_work": True,
        "continue_goal_mode": True,
        "selected_blocker": "WING read-only connection proof",
        "selected_blocker_status": "tooling_ready_waits_real_wing",
        "selected_action_source": "Current First Command",
        "selected_action_sha256": "a" * 64,
        "current_first_command_baseline_coverage_required": True,
        "current_first_command_baseline_coverage_passed": True,
        "current_first_command_baseline_coverage_marker_count": len(
            director.update_mvp_progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS
        ),
        "current_first_command_baseline_missing_markers": [],
        "selected_action": "Run read-only preflight.",
        "selected_evidence_file_count": 1,
        "selected_evidence_files": ["tools/wing_readonly_preflight_runner.py"],
        "allowed_scope": ["Work only on the selected first non-closed MVP blocker."],
        "forbidden_scope": ["No queue-import chains."],
        "required_files": ["Docs/MVP_WING_SUPERVISED.md"],
    }

    markdown = director.render_markdown(directive)

    assert "# Project Director Agent Directive" in markdown
    assert "Selected blocker" in markdown
    assert "Continue Goal mode: `true`" in markdown
    assert "Selected action source: `Current First Command`" in markdown
    assert "Selected action sha256: `" in markdown
    assert "Current first command baseline coverage required: `true`" in markdown
    assert "Current first command baseline coverage passed: `true`" in markdown
    assert "Current first command baseline missing marker count: `0`" in markdown
    assert "Current first command baseline missing markers: `[]`" in markdown
    assert "Selected evidence file count: `1`" in markdown
    assert "## Selected Evidence Files" in markdown
    assert "tools/wing_readonly_preflight_runner.py" in markdown
    assert "No queue-import chains." in markdown
    assert "does not grant live readiness" in markdown


def test_prompt_and_readme_document_project_director_agent():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "project_director_agent.py" in prompt
    assert "project_director_latest.md" in prompt
    assert "Project Director Agent" in readme
    assert "first non-closed MVP blocker" in readme
    assert "selected_action_sha256" in prompt
    assert "current_first_command_baseline_coverage_passed" in prompt
    assert "current_first_command_baseline_missing_markers" in prompt
    assert "selected_action_sha256" in readme
    assert "current_first_command_baseline_coverage_passed" in readme
    assert "current_first_command_baseline_missing_markers" in readme
    assert "backend_port_prestart_gate" in prompt
    assert "expected_backend_port_prestart_gate" in prompt
    assert "backend_port_prestart_gate.grants_live_authority: false" in prompt
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in prompt
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in prompt
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in prompt
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in prompt
    assert "backend_port_prestart_gate.required_markers" in prompt
    assert "Before starting the intended backend/WING session" in prompt
    assert "backend_port_prestart_check_command" in prompt
    assert "backend_port_prestart_check_command_text" in prompt
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in prompt
    assert "failed busy-port no-WING summary and no-WING handoff summary both" in prompt
    assert "backend_port_prestart_check_required: true" in prompt
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in prompt
    assert "backend_port_prestart_gate_execution_allowed: false" in prompt
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in prompt
    assert "backend_port_prestart_gate_live_authority_granted: false" in prompt
    assert "backend_port_prestart_check_failure_detail_fields" in prompt
    assert "listener_processes" in prompt
    assert "listener_process_error" in prompt
    assert "safe_to_stop_listener_by_gate" in prompt
    assert "listener_confirmation_template_path" in prompt
    assert "listener_confirmation_template_acceptance_status" in prompt
    assert "operator_next_step" in prompt
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in prompt
    assert "listener_confirmation_template_accepted_by_template: false" in prompt
    assert "listener_confirmation_template_operator_placeholder_fields" in prompt
    assert "listener_confirmation_template_required_edits" in prompt
    assert "set template_status to operator_reviewed_and_edited" in prompt
    assert "set accepted_by_template to true" in prompt
    assert "REPLACE_WITH_*" in prompt
    assert "template_status: operator_reviewed_and_edited" in prompt
    assert "accepted_by_template: true" in prompt
    assert "manual_review_required_before_readiness_update: true" in prompt
    assert "readiness_matrix_update_allowed_by_packet: false" in prompt
    assert "readiness_closure_allowed_by_packet: false" in prompt
    assert "read_only: true" in prompt
    assert "live_write: false" in prompt
    assert "mutates_console_state: false" in prompt
    assert "queries_console: false" in prompt
    assert "executes_tests: false" in prompt
    assert "operator_required: true" in prompt
    assert "expected_plan_required_for_real_run: true" in prompt
    assert "physical_preflight_requires_existing_backend_connection: true" in prompt
    assert "backend_port_prestart_check_required: true" in prompt
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in prompt
    )
    assert "execution_allowed_by_packet: false" in prompt
    assert "live_authority_granted: false" in prompt
    assert "full_autonomous_ready: false" in prompt
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in prompt
    )
    assert "expected_artifacts" in prompt
    assert "expected_artifact_globs" in prompt
    assert "plan_expected_artifacts" in prompt
    assert "expected_artifacts_match_plan: true" in prompt
    assert "summary_glob_matches_plan: true" in prompt
    assert "event_log_glob_matches_plan: true" in prompt
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in prompt
    assert "backend_port_prestart_check_operator_sequence_step" in prompt
    assert "prestart_backend_port_check" in prompt
    assert "unexpected listeners" in prompt
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in prompt
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in prompt
    assert "backend_connection_confirmation_step_index: 3" in prompt
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in prompt
    assert "operator_sequence_has_backend_connection_confirmation: true" in prompt
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in prompt
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in prompt
    )
    assert "operator_sequence_backend_connection_confirmation_step" in prompt
    assert "existing backend connection" in prompt
    assert "intended WING Rack" in prompt
    assert "read-only inspection" in prompt
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in prompt
    assert "backend_port_prestart_gate" in readme
    assert "expected_backend_port_prestart_gate" in readme
    assert "backend_port_prestart_gate.grants_live_authority: false" in readme
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in readme
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in readme
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in readme
    )
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in readme
    assert "backend_port_prestart_gate.required_markers" in readme
    assert "Before starting the intended backend/WING session" in readme
    assert "backend_port_prestart_check_command" in readme
    assert "backend_port_prestart_check_command_text" in readme
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in readme
    assert "failed busy-port no-WING summary" in readme
    assert "no-WING handoff summary must both show" in readme
    assert "backend_port_prestart_check_required: true" in readme
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in readme
    assert "backend_port_prestart_gate_execution_allowed: false" in readme
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readme
    assert "backend_port_prestart_gate_live_authority_granted: false" in readme
    assert "backend_port_prestart_check_failure_detail_fields" in readme
    assert "listener_processes" in readme
    assert "listener_process_error" in readme
    assert "safe_to_stop_listener_by_gate" in readme
    assert "listener_confirmation_template_path" in readme
    assert "listener_confirmation_template_acceptance_status" in readme
    assert "operator_next_step" in readme
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in readme
    assert "listener_confirmation_template_accepted_by_template: false" in readme
    assert "listener_confirmation_template_operator_placeholder_fields" in readme
    assert "listener_confirmation_template_required_edits" in readme
    assert "set template_status to operator_reviewed_and_edited" in readme
    assert "set accepted_by_template to true" in readme
    assert "REPLACE_WITH_*" in readme
    assert "template_status: operator_reviewed_and_edited" in readme
    assert "accepted_by_template: true" in readme
    assert "manual_review_required_before_readiness_update: true" in readme
    assert "readiness_matrix_update_allowed_by_packet: false" in readme
    assert "readiness_closure_allowed_by_packet: false" in readme
    assert "read_only: true" in readme
    assert "live_write: false" in readme
    assert "mutates_console_state: false" in readme
    assert "queries_console: false" in readme
    assert "executes_tests: false" in readme
    assert "operator_required: true" in readme
    assert "expected_plan_required_for_real_run: true" in readme
    assert "physical_preflight_requires_existing_backend_connection: true" in readme
    assert "backend_port_prestart_check_required: true" in readme
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readme
    )
    assert "execution_allowed_by_packet: false" in readme
    assert "live_authority_granted: false" in readme
    assert "full_autonomous_ready: false" in readme
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in readme
    )
    assert "expected_artifacts" in readme
    assert "expected_artifact_globs" in readme
    assert "plan_expected_artifacts" in readme
    assert "expected_artifacts_match_plan: true" in readme
    assert "summary_glob_matches_plan: true" in readme
    assert "event_log_glob_matches_plan: true" in readme
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in readme
    assert "backend_port_prestart_check_operator_sequence_step" in readme
    assert "prestart_backend_port_check" in readme
    assert "unexpected listeners" in readme
    assert (
        "backend_port_prestart_check_must_be_first_operator_step: true"
        in readme
    )
    assert (
        "backend_port_prestart_check_operator_sequence_step_is_first: true"
        in readme
    )
    assert "backend_connection_confirmation_step_index: 3" in readme
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in readme
    assert "operator_sequence_has_backend_connection_confirmation: true" in readme
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in readme
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readme
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readme
    assert "existing backend connection" in readme
    assert "intended WING Rack" in readme
    assert "read-only inspection" in readme
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in readme
    assert "supervised_prestart_gate" in prompt
    assert "expected_supervised_prestart_gate" in prompt
    assert "supervised_prestart_gate.grants_live_authority: false" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in prompt
    assert "supervised_prestart_gate_required: true" in prompt
    assert "supervised_prestart_gate_operator_sequence_step_index: 11" in prompt
    assert (
        "supervised_prestart_gate_operator_sequence_step_must_be_final: true"
        in prompt
    )
    assert "supervised_prestart_gate_operator_sequence_step" in prompt
    assert "supervised_prestart_gate_operator_sequence_step_matches_summary: true" in prompt
    assert "supervised_prestart_gate_required_markers" in prompt
    assert "expected_supervised_prestart_gate_required_markers" in prompt
    assert "expected_required_supervised_prestart_gate_markers" in prompt
    assert "required_supervised_prestart_gate_marker_count: 7" in prompt
    assert "expected_required_supervised_prestart_gate_marker_count: 7" in prompt
    assert (
        "required_supervised_prestart_gate_marker_count_matches_contract: true"
        in prompt
    )
    assert (
        "required_supervised_prestart_gate_marker_count_matches_supervised_gate: true"
        in prompt
    )
    assert "required_supervised_prestart_gate_markers_match_contract: true" in prompt
    assert (
        "required_supervised_prestart_gate_markers_match_supervised_gate: true"
        in prompt
    )
    assert "supervised_prestart_gate_required_marker_count: 7" in prompt
    assert "expected_supervised_prestart_gate_required_marker_count: 7" in prompt
    assert (
        "supervised_prestart_gate_required_marker_count_matches_contract: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate_required_marker_count_matches_summary: true"
        in prompt
    )
    assert "supervised_prestart_gate_required_markers_match_contract: true" in prompt
    assert "supervised_prestart_gate_required_markers_match_summary: true" in prompt
    assert (
        "supervised_prestart_gate.requires_operator_decision_status:"
        in prompt
    )
    assert "accepted_decision_ready" in prompt
    assert "supervised_prestart_gate.unlocks_runbook:" in prompt
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in readme
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in readme
    )
    assert "supervised_prestart_gate_required: true" in readme
    assert "supervised_prestart_gate_operator_sequence_step_index: 11" in readme
    assert (
        "supervised_prestart_gate_operator_sequence_step_must_be_final: true"
        in readme
    )
    assert "supervised_prestart_gate_operator_sequence_step" in readme
    assert "supervised_prestart_gate_operator_sequence_step_matches_summary: true" in readme
    assert "supervised_prestart_gate_required_markers" in readme
    assert "expected_supervised_prestart_gate_required_markers" in readme
    assert "expected_required_supervised_prestart_gate_markers" in readme
    assert "required_supervised_prestart_gate_marker_count: 7" in readme
    assert "expected_required_supervised_prestart_gate_marker_count: 7" in readme
    assert (
        "required_supervised_prestart_gate_marker_count_matches_contract: true"
        in readme
    )
    assert (
        "required_supervised_prestart_gate_marker_count_matches_supervised_gate: true"
        in readme
    )
    assert "required_supervised_prestart_gate_markers_match_contract: true" in readme
    assert (
        "required_supervised_prestart_gate_markers_match_supervised_gate: true"
        in readme
    )
    assert "supervised_prestart_gate_required_marker_count: 7" in readme
    assert "expected_supervised_prestart_gate_required_marker_count: 7" in readme
    assert (
        "supervised_prestart_gate_required_marker_count_matches_contract: true"
        in readme
    )
    assert (
        "supervised_prestart_gate_required_marker_count_matches_summary: true"
        in readme
    )
    assert "supervised_prestart_gate_required_markers_match_contract: true" in readme
    assert "supervised_prestart_gate_required_markers_match_summary: true" in readme
    assert (
        "supervised_prestart_gate.requires_operator_decision_status:"
        in readme
    )
    assert "supervised_prestart_gate.unlocks_runbook:" in readme
    assert "mvp_blocker_anchor_matches_current_first_open: true" in prompt
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in prompt
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in prompt
    assert "project_director_selected_action_anchor" in prompt
    assert "expected_project_director_selected_action_anchor" in prompt
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in prompt
    )
    assert "project_director_selected_action_source: Current First Command" in prompt
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in prompt
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in prompt
    )
    assert "project_director_selected_action_sha256" in prompt
    assert "actual_project_director_selected_action_sha256" in prompt
    assert "expected_project_director_selected_action_sha256" in prompt
    assert "project_director_selected_action_sha256_available: true" in prompt
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in prompt
    )
    assert "project_director_selected_action_sha256_matches_anchor: true" in prompt
    assert "project_director_selected_action_sha256_matches_text: true" in prompt
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in prompt
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in prompt
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in prompt
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in prompt
    )
    assert "project_director_current_first_command_baseline_missing_markers: []" in prompt
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in prompt
    assert "operator_sequence_count: 12" in prompt
    assert "expected_operator_sequence_count: 12" in prompt
    assert "operator_sequence_count_matches_contract: true" in prompt
    assert "operator_sequence_sha256_matches_actual: true" in prompt
    assert "operator_sequence_matches_contract: true" in prompt
    assert "operator_sequence_sha256_matches_contract: true" in prompt
    assert "declared_expected_operator_sequence_sha256_matches_contract: true" in prompt
    assert "declared_operator_sequence_matches_contract_flag: true" in prompt
    assert (
        "declared_operator_sequence_matches_contract_flag_matches_actual: true"
        in prompt
    )
    assert "declared_operator_sequence_sha256_matches_contract_flag: true" in prompt
    assert (
        "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true"
        in prompt
    )
    assert "declared_operator_sequence_count: 12" in prompt
    assert "declared_expected_operator_sequence_count: 12" in prompt
    assert "declared_operator_sequence_count_contract_flag: true" in prompt
    assert "declared_operator_sequence_count_matches_actual: true" in prompt
    assert "declared_operator_sequence_count_matches_contract: true" in prompt
    assert "declared_expected_operator_sequence_count_matches_contract: true" in prompt
    assert (
        "declared_operator_sequence_count_contract_flag_matches_declared_count: true"
        in prompt
    )
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in prompt
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in prompt
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in prompt
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in prompt
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in prompt
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in prompt
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in prompt
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in prompt
    )
    assert "operator_sequence_has_supervised_prestart_gate: true" in prompt
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in prompt
    assert "operator_sequence_supervised_prestart_gate_step" in prompt
    assert "operator_sequence_supervised_prestart_gate_step_index" in prompt
    assert "expected_operator_sequence_supervised_prestart_gate_step_index" in prompt
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in prompt
    )
    assert "required_supervised_prestart_gate_markers" in prompt
    assert "required_supervised_proof_plan_summary_fields" in prompt
    assert "expected_required_supervised_proof_plan_summary_fields" in prompt
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in prompt
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in prompt
    )
    assert "operator-readable no-WING" in prompt
    assert "`summary.md` proof plan section" in prompt
    assert "operator_markdown_summary_required: true" in prompt
    assert "operator_markdown_summary_required_field_lines" in prompt
    for marker in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert marker in prompt
    assert "operator_markdown_summary_required_field_lines_missing_members: []" in prompt
    assert "operator_markdown_summary_required_field_lines_match_contract: true" in prompt
    assert "operator_markdown_summary_field_lines_rendered: true" in prompt
    assert "operator_markdown_summary_missing_field_lines: []" in prompt
    assert "operator_markdown_summary_artifact: summary.md" in prompt
    assert "operator_markdown_summary_sha256" in prompt
    assert "operator_markdown_summary_sha256_available: true" in prompt
    assert "operator_markdown_summary_actual_sha256" in prompt
    assert "operator_markdown_summary_sha256_matches_actual: true" in prompt
    assert "artifact_manifest_summary_surface_required: true" in prompt
    assert "artifact_manifest_summary_surface_missing_fields: []" in prompt
    assert "artifact_manifest_summary_surface_required_field_lines" in prompt
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in prompt
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in prompt
    assert "artifact_manifest_contract_id" in prompt
    assert "artifact_manifest_contract_required: true" in prompt
    assert "artifact_manifest_contract_fields" in prompt
    assert "artifact_manifest_contract_sha256" in prompt
    assert "artifact_manifest_contract_sha256_available: true" in prompt
    assert "artifact_manifest_contract_sha256_matches_expected: true" in prompt
    assert "artifact_manifest_contract_cross_surface_required: true" in prompt
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in prompt
    assert "artifact_manifest_contract_cross_surface_required_fields" in prompt
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in prompt
    assert "artifact_manifest_contract_cross_surface_match: true" in prompt
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in prompt
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in prompt
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in prompt
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in prompt
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in prompt
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in prompt
    assert "artifact_manifest_path: artifact-manifest.json" in prompt
    assert "artifact_manifest_artifact_count: 2" in prompt
    assert "artifact_manifest_required_artifact_count: 2" in prompt
    assert "artifact_manifest_artifact_count_matches_required: true" in prompt
    assert "artifact_manifest_required_artifact_ids" in prompt
    assert "artifact_manifest_artifact_ids" in prompt
    assert "artifact_manifest_required_artifacts_present: true" in prompt
    assert "artifact_manifest_required_artifact_paths" in prompt
    assert "artifact_manifest_artifact_paths" in prompt
    assert "artifact_manifest_required_artifact_paths_present: true" in prompt
    assert "artifact_manifest_required_artifact_sha256_available: true" in prompt
    assert "artifact_manifest_summary_json_sha256" in prompt
    assert "artifact_manifest_summary_json_sha256_available: true" in prompt
    assert "artifact_manifest_summary_md_sha256" in prompt
    assert "artifact_manifest_summary_md_sha256_available: true" in prompt
    assert "artifact_manifest_readiness_closure_allowed: false" in prompt
    assert "artifact_manifest_live_authority_granted: false" in prompt
    assert "backend_port_prestart_gate_matches_expected: true" in prompt
    assert "declared_backend_port_prestart_gate_matches_expected: true" in prompt
    assert "supervised_prestart_gate_matches_expected: true" in prompt
    assert "declared_supervised_prestart_gate_matches_expected: true" in prompt
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_source_ids"
        in prompt
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source"
        in prompt
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source"
        in prompt
    )
    assert "readonly_preflight_review_decision_invalid_records" in prompt
    assert (
        "next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations"
        in prompt
    )
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in prompt
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in prompt
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in prompt
    )
    assert "accepted_decision_ready" in prompt
    assert "valid accepted-candidate prestart trace" in prompt
    assert "expected-artifact-bound accepted source id" in prompt
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in prompt
    assert "backend_port_prestart_summary_type: no_wing_release_check" in prompt
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in prompt
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in prompt
    assert "current first open blocker" in prompt
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in prompt
    )
    assert "supervised_prestart_gate_grants_live_authority: false" in prompt
    assert (
        "supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready"
        in prompt
    )
    assert (
        "supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in prompt
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true"
        in prompt
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true"
        in prompt
    )
    assert "accepted_candidate_prestart_contract_violations_by_source: {}" in prompt
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in prompt
    )
    assert "supervised_prestart_gate" in readme
    assert "expected_supervised_prestart_gate" in readme
    assert "supervised_prestart_gate.grants_live_authority: false" in readme
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readme
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readme
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readme
    assert "project_director_selected_action_anchor" in readme
    assert "expected_project_director_selected_action_anchor" in readme
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in readme
    )
    assert "project_director_selected_action_source: Current First Command" in readme
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in readme
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in readme
    )
    assert "project_director_selected_action_sha256" in readme
    assert "actual_project_director_selected_action_sha256" in readme
    assert "expected_project_director_selected_action_sha256" in readme
    assert "project_director_selected_action_sha256_available: true" in readme
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in readme
    )
    assert "project_director_selected_action_sha256_matches_anchor: true" in readme
    assert "project_director_selected_action_sha256_matches_text: true" in readme
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in readme
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in readme
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in readme
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in readme
    )
    assert "project_director_current_first_command_baseline_missing_markers: []" in readme
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readme
    assert "operator_sequence_count: 12" in readme
    assert "expected_operator_sequence_count: 12" in readme
    assert "operator_sequence_count_matches_contract: true" in readme
    assert "operator_sequence_sha256_matches_actual: true" in readme
    assert "operator_sequence_matches_contract: true" in readme
    assert "operator_sequence_sha256_matches_contract: true" in readme
    assert "declared_expected_operator_sequence_sha256_matches_contract: true" in readme
    assert "declared_operator_sequence_matches_contract_flag: true" in readme
    assert (
        "declared_operator_sequence_matches_contract_flag_matches_actual: true"
        in readme
    )
    assert "declared_operator_sequence_sha256_matches_contract_flag: true" in readme
    assert (
        "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true"
        in readme
    )
    assert "declared_operator_sequence_count: 12" in readme
    assert "declared_expected_operator_sequence_count: 12" in readme
    assert "declared_operator_sequence_count_contract_flag: true" in readme
    assert "declared_operator_sequence_count_matches_actual: true" in readme
    assert "declared_operator_sequence_count_matches_contract: true" in readme
    assert "declared_expected_operator_sequence_count_matches_contract: true" in readme
    assert (
        "declared_operator_sequence_count_contract_flag_matches_declared_count: true"
        in readme
    )
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in readme
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in readme
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in readme
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in readme
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readme
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in readme
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in readme
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in readme
    )
    assert "operator_sequence_has_supervised_prestart_gate: true" in readme
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in readme
    assert "operator_sequence_supervised_prestart_gate_step" in readme
    assert "operator_sequence_supervised_prestart_gate_step_index" in readme
    assert "expected_operator_sequence_supervised_prestart_gate_step_index" in readme
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in readme
    )
    assert "required_supervised_prestart_gate_markers" in readme
    assert "required_supervised_proof_plan_summary_fields" in readme
    assert "expected_required_supervised_proof_plan_summary_fields" in readme
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in readme
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in readme
    )
    assert "readonly_preflight_review_decision_invalid_records" in readme
    assert "operator-readable no-WING" in readme
    assert "`summary.md` proof plan section" in readme
    assert "operator_markdown_summary_required: true" in readme
    assert "operator_markdown_summary_required_field_lines" in readme
    for marker in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert marker in readme
    assert "operator_markdown_summary_required_field_lines_missing_members: []" in readme
    assert "operator_markdown_summary_required_field_lines_match_contract: true" in readme
    assert "operator_markdown_summary_field_lines_rendered: true" in readme
    assert "operator_markdown_summary_missing_field_lines: []" in readme
    assert "operator_markdown_summary_artifact: summary.md" in readme
    assert "operator_markdown_summary_sha256" in readme
    assert "operator_markdown_summary_sha256_available: true" in readme
    assert "operator_markdown_summary_actual_sha256" in readme
    assert "operator_markdown_summary_sha256_matches_actual: true" in readme
    assert "artifact_manifest_summary_surface_required: true" in readme
    assert "artifact_manifest_summary_surface_missing_fields: []" in readme
    assert "artifact_manifest_summary_surface_required_field_lines" in readme
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in readme
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in readme
    assert "artifact_manifest_contract_id" in readme
    assert "artifact_manifest_contract_required: true" in readme
    assert "artifact_manifest_contract_fields" in readme
    assert "artifact_manifest_contract_sha256" in readme
    assert "artifact_manifest_contract_sha256_available: true" in readme
    assert "artifact_manifest_contract_sha256_matches_expected: true" in readme
    assert "artifact_manifest_contract_cross_surface_required: true" in readme
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in readme
    assert "artifact_manifest_contract_cross_surface_required_fields" in readme
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in readme
    assert "artifact_manifest_contract_cross_surface_match: true" in readme
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in readme
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in readme
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in readme
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in readme
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in readme
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in readme
    assert "artifact_manifest_path: artifact-manifest.json" in readme
    assert "artifact_manifest_artifact_count: 2" in readme
    assert "artifact_manifest_required_artifact_count: 2" in readme
    assert "artifact_manifest_artifact_count_matches_required: true" in readme
    assert "artifact_manifest_required_artifact_ids" in readme
    assert "artifact_manifest_artifact_ids" in readme
    assert "artifact_manifest_required_artifacts_present: true" in readme
    assert "artifact_manifest_required_artifact_paths" in readme
    assert "artifact_manifest_artifact_paths" in readme
    assert "artifact_manifest_required_artifact_paths_present: true" in readme
    assert "artifact_manifest_required_artifact_sha256_available: true" in readme
    assert "artifact_manifest_summary_json_sha256" in readme
    assert "artifact_manifest_summary_json_sha256_available: true" in readme
    assert "artifact_manifest_summary_md_sha256" in readme
    assert "artifact_manifest_summary_md_sha256_available: true" in readme
    assert "artifact_manifest_readiness_closure_allowed: false" in readme
    assert "artifact_manifest_live_authority_granted: false" in readme
    assert "backend_port_prestart_gate_matches_expected: true" in readme
    assert "declared_backend_port_prestart_gate_matches_expected: true" in readme
    assert "supervised_prestart_gate_matches_expected: true" in readme
    assert "declared_supervised_prestart_gate_matches_expected: true" in readme
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_source_ids"
        in readme
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source"
        in readme
    )
    assert "accepted_decision_ready" in readme
    assert "valid accepted-candidate prestart trace" in readme
    assert "expected-artifact-bound accepted source id" in readme
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in readme
    assert "backend_port_prestart_summary_type: no_wing_release_check" in readme
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in readme
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in readme
    assert "current first open blocker" in readme
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readme
    )
    assert "supervised_prestart_gate_grants_live_authority: false" in readme
    assert (
        "supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready"
        in readme
    )
    assert (
        "supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in readme
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true"
        in readme
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true"
        in readme
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true"
        in readme
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true"
        in readme
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true"
        in readme
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true"
        in readme
    )
    assert "accepted_candidate_prestart_contract_violations_by_source: {}" in readme
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in readme
    )
