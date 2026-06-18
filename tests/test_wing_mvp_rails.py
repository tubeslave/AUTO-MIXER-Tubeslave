import importlib.util
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVP_DOC = PROJECT_ROOT / "Docs" / "MVP_WING_SUPERVISED.md"
CORRECTION_ROADMAP = PROJECT_ROOT / "Docs" / "CORRECTION_METHODS_ROADMAP.md"
BLOCKERS_DOC = PROJECT_ROOT / ".ai" / "project-layer" / "wing_mvp_blockers.md"
CORRECTION_BACKLOG = PROJECT_ROOT / ".ai" / "project-layer" / "correction_methods_backlog.md"
READONLY_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_READONLY_PREFLIGHT.md"
SUPERVISED_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_SUPERVISED_FADER_GAIN_PROOF.md"
DOCS_INDEX = PROJECT_ROOT / "Docs" / "INDEX.md"
PROGRESS_TOOL = PROJECT_ROOT / "tools" / "codex_autoloop" / "update_mvp_progress.py"
RUNNER_TOOL = PROJECT_ROOT / "tools" / "wing_readonly_preflight_runner.py"
PROMPT = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"

DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS = (
    "declared_operator_sequence_count: 12",
    "declared_expected_operator_sequence_count: 12",
    "declared_operator_sequence_count_contract_flag: true",
    "declared_operator_sequence_count_matches_actual: true",
    "declared_operator_sequence_count_matches_contract: true",
    "declared_expected_operator_sequence_count_matches_contract: true",
    "declared_operator_sequence_count_contract_flag_matches_declared_count: true",
)

OPERATOR_SEQUENCE_SHA_MARKERS = (
    "operator_sequence_sha256",
    "actual_operator_sequence_sha256",
    "operator_sequence_sha256_matches_actual: true",
)

OPERATOR_SEQUENCE_CONTRACT_MARKERS = (
    "expected_operator_sequence_sha256",
    "actual_expected_operator_sequence_sha256",
    "operator_sequence_matches_contract: true",
    "operator_sequence_sha256_matches_contract: true",
    "declared_expected_operator_sequence_sha256_matches_contract: true",
    "declared_operator_sequence_matches_contract_flag: true",
    "declared_operator_sequence_matches_contract_flag_matches_actual: true",
    "declared_operator_sequence_sha256_matches_contract_flag: true",
    "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true",
)

DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS = tuple(
    marker.split(":", 1)[0] for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS
)

ENVIRONMENT_BLOCKED_FRONTEND_MARKERS = (
    "environment_blocked_checks",
    "environment_blocked_check_details",
    "local_dev_server_bind_denied",
    "127.0.0.1:3000",
)

ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS = (
    "artifact-manifest.json",
    "same no-WING artifact directory",
    "artifact_manifest_self_consistency_summary_surface_match: true",
    "artifact_manifest_self_consistency_summary_surface_json_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false",
    "artifact_manifest_self_consistency_summary_surface_live_authority_granted: false",
    "artifact_manifest_self_consistency_match: true",
    "artifact_manifest_self_consistency_mismatches: []",
    "artifact_manifest_contract_cross_surface_match: true",
    "artifact_manifest_contract_cross_surface_mismatches: []",
)

ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS = (
    "status: passed",
    "failed_checks: []",
    "read_only: true",
    "live_authority_granted: false",
    "readiness_closure_allowed: false",
)

ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS = (
    "PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py",
    "status: passed",
    "failed_checks: []",
    "failed_checks_missing_check_entries: []",
    "failed_checks_have_check_entries: true",
    "summary_json_exists: true",
    "summary_md_exists: true",
    "artifact_manifest_exists: true",
    "artifact_bundle_loadable: true",
    "manifest_type: true",
    "manifest_required_artifact_ids_match_contract: true",
    "manifest_required_artifact_paths_match_contract: true",
    "manifest_records_required_artifact_ids: true",
    "summary_json_path_matches_contract: true",
    "summary_json_sha256_matches_actual: true",
    "summary_md_path_matches_contract: true",
    "summary_md_sha256_matches_actual: true",
    "summary_json_sha256_matches_manifest: true",
    "summary_md_sha256_matches_manifest: true",
    "manifest_denies_readiness_matrix_update: true",
    "manifest_denies_readiness_closure: true",
    "manifest_denies_live_authority: true",
    "summary_denies_readiness_matrix_update: true",
    "summary_denies_readiness_closure: true",
    "summary_denies_live_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "summary_json_artifact_manifest_verifier_fields_present: true",
    "manifest_artifact_manifest_verifier_fields_present: true",
    "summary_md_artifact_manifest_verifier_fields_rendered: true",
    "manifest_artifact_manifest_verifier_command_matches_expected: true",
    "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "verifier_report_required_fields_present: true",
    "verifier_report_required_fields_missing: []",
    "verifier_report_required_fields_match_contract: true",
    "verifier_report_readiness_closure_allowed_matches_expected: true",
    "verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_mismatches: []",
    "artifact_manifest_verifier_check_summary_json_exists: true",
    "artifact_manifest_verifier_check_summary_md_exists: true",
    "artifact_manifest_verifier_check_artifact_manifest_exists: true",
    "artifact_manifest_verifier_check_artifact_bundle_loadable: true",
    "artifact_manifest_verifier_check_manifest_type: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_ids_match_contract: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_paths_match_contract: true",
    "artifact_manifest_verifier_check_manifest_records_required_artifact_ids: true",
    "artifact_manifest_verifier_check_summary_json_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_md_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_manifest_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_summary_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_fields_rendered: true",
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
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_command_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_required_fields_present: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_contract_cross_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_failed_checks_have_check_entries: true",
    "artifact_manifest_verifier_report_required_fields",
    "checks",
    "verifier_report_required_fields",
    "verifier_report_required_fields_missing",
    "verifier_report_required_fields_match_contract",
)


def load_progress_module():
    spec = importlib.util.spec_from_file_location("update_mvp_progress", PROGRESS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "wing_readonly_preflight_runner_for_mvp_rails", RUNNER_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def covered_verifier_checks(markers):
    covered: set[str] = set()
    for marker in markers:
        if not marker.endswith(": true"):
            continue
        raw_marker = marker.removesuffix(": true")
        covered.add(raw_marker)
        if raw_marker.startswith("artifact_manifest_verifier_check_"):
            covered.add(raw_marker.removeprefix("artifact_manifest_verifier_check_"))
    return covered


def current_wing_mvp_20260614_index_report_names():
    index_text = DOCS_INDEX.read_text(encoding="utf-8")
    report_paths = re.findall(
        r"`(Docs/reports/AUTOMIXER_WING_MVP_[^`]+_20260614\.md)`",
        index_text,
    )
    return sorted({Path(path).name for path in report_paths})


def test_progress_baseline_covers_runner_verifier_checks():
    progress = load_progress_module()
    runner = load_runner_module()
    missing_checks = sorted(
        set(runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS)
        - covered_verifier_checks(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)
    )
    assert missing_checks == []


def test_mvp_doc_declares_scope_and_stop_rules():
    text = MVP_DOC.read_text(encoding="utf-8")

    assert "Supervised WING Rack Proof" in text
    assert "Full autonomous live mixing" in text
    assert "Stop-The-Line Rules" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "tools/wing_readonly_preflight_runner.py" in text
    assert "tools/review_wing_readonly_preflight_artifacts.py" in text
    assert "tools/no_wing_release_check.py" in text
    assert "readiness candidates are not enough" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in text
    assert "backend_port_prestart_summary_sha256_available: true" in text
    assert (
        "backend_port_prestart_summary_sha256_available_matches_digest: true"
        in text
    )
    assert "expected_operator_sequence_count: 12" in text
    assert "operator_sequence_count_matches_contract: true" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in text
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in text
    )
    assert "backend_port_prestart_check_command" in text
    assert "backend_port_prestart_check_command_text" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "listener_confirmation_template_status: pending_operator_edit" in text
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in text
    assert "listener_confirmation_template_live_authority_granted: false" in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe packet values" in text
    assert "unsafe safety flags" in text
    assert "full_autonomous_ready" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
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
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in text
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in text
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in text
    )
    assert "grants_live_authority: true" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text


def test_blocker_table_has_exact_nine_rows_and_no_closed_baseline():
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)

    assert len(rows) == 9
    assert rows[0]["blocker"] == "WING read-only connection proof"
    assert rows[0]["status"] == "tooling_ready_waits_real_wing"
    missing_index_reports = [
        report_name
        for report_name in current_wing_mvp_20260614_index_report_names()
        if report_name not in rows[0]["evidence"]
    ]
    assert missing_index_reports == []
    assert "tools/no_wing_release_check.py" in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in rows[0]["evidence"]
    )
    for report_name in (
        "AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md",
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md",
        "AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md",
    ):
        assert report_name in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available_matches_digest: true" in rows[
        0
    ]["next action"]
    assert "expected_operator_sequence_count: 12" in rows[0]["next action"]
    assert "operator_sequence_count_matches_contract: true" in rows[0][
        "next action"
    ]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "accepted-candidate prestart trace" in rows[0]["next action"]
    assert "backend_port_prestart_gate" in rows[0]["next action"]
    assert "expected_backend_port_prestart_gate" in rows[0]["next action"]
    assert "backend_port_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate.must_pass_before_backend_wing_session: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in rows[
        0
    ]["next action"]
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in rows[0]["next action"]
    )
    assert (
        "backend_port_prestart_gate.runs_command: prestart_backend_port_check"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate.required_markers" in rows[0]["next action"]
    assert "Before starting the intended backend/WING session" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_command" in rows[0]["next action"]
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in rows[
        0
    ]["next action"]
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_failure_detail_fields" in rows[0][
        "next action"
    ]
    assert "failed busy-port no-WING summary" in rows[0]["next action"]
    assert "no-WING handoff summary" in rows[0]["next action"]
    assert "listener_processes" in rows[0]["next action"]
    assert "listener_process_error" in rows[0]["next action"]
    assert "safe_to_stop_listener_by_gate" in rows[0]["next action"]
    assert "operator_next_step" in rows[0]["next action"]
    assert "listener_confirmation_template_path" in rows[0]["next action"]
    assert "listener_confirmation_template_acceptance_status" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_status: pending_operator_edit" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in rows[
        0
    ]["next action"]
    assert "listener_confirmation_template_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_accepted_by_template: false" in rows[0][
        "next action"
    ]
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_required_edits" in rows[0]["next action"]
    assert "set template_status to operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "set accepted_by_template to true" in rows[0]["next action"]
    assert "REPLACE_WITH_*" in rows[0]["next action"]
    assert "template_status: operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "accepted_by_template: true" in rows[0]["next action"]
    assert "manual_review_required_before_readiness_update: true" in rows[0][
        "next action"
    ]
    assert "readiness_matrix_update_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "readiness_closure_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "read_only: true" in rows[0]["next action"]
    assert "live_write: false" in rows[0]["next action"]
    assert "mutates_console_state: false" in rows[0]["next action"]
    assert "queries_console: false" in rows[0]["next action"]
    assert "executes_tests: false" in rows[0]["next action"]
    assert "operator_required: true" in rows[0]["next action"]
    assert "expected_plan_required_for_real_run: true" in rows[0]["next action"]
    assert "backend_port_prestart_summary_required: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_type: no_wing_release_check" in rows[0][
        "next action"
    ]
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in rows[0]["next action"]
    assert "backend_port_prestart_summary_path_placeholder" in rows[0][
        "next action"
    ]
    assert "preview_command_prestart_summary" in rows[0]["next action"]
    assert "preview_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "capture_command_prestart_summary" in rows[0]["next action"]
    assert "capture_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_status: passed" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_read_only: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_live_write: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_wing: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_mutates_console_state: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_queries_console: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_executes_tests: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_status: passed" in rows[0]["next action"]
    assert (
        "physical_preflight_requires_existing_backend_connection: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "execution_allowed_by_packet: false" in rows[0]["next action"]
    assert "live_authority_granted: false" in rows[0]["next action"]
    assert "full_autonomous_ready: false" in rows[0]["next action"]
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in rows[0]["next action"]
    )
    assert "expected_artifacts" in rows[0]["next action"]
    assert "expected_artifact_globs" in rows[0]["next action"]
    assert "plan_expected_artifacts" in rows[0]["next action"]
    assert "expected_artifacts_match_plan: true" in rows[0]["next action"]
    assert "summary_glob_matches_plan: true" in rows[0]["next action"]
    assert "event_log_glob_matches_plan: true" in rows[0]["next action"]
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in rows[0]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_operator_sequence_step" in rows[0][
        "next action"
    ]
    assert "prestart_backend_port_check" in rows[0]["next action"]
    assert "unexpected listeners" in rows[0]["next action"]
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in rows[
        0
    ]["next action"]
    assert (
        "backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_has_backend_connection_confirmation: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in rows[0]["next action"]
    )
    assert "operator_sequence_backend_connection_confirmation_step" in rows[0][
        "next action"
    ]
    assert "existing backend connection" in rows[0]["next action"]
    assert "intended WING Rack" in rows[0]["next action"]
    assert "read-only inspection" in rows[0]["next action"]
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_index: 5"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_capture_readonly_preflight_step_index: 6"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor" in rows[0]["next action"]
    assert "mvp_blocker_anchor_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_count: 12" in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor_sha256" in rows[0]["next action"]
    assert "current first open blocker" in rows[0]["next action"]
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_supervised_prestart_gate: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in rows[0][
        "next action"
    ]
    assert "required_supervised_prestart_gate_markers" in rows[0]["next action"]
    assert "accepted_decision_ready" in rows[0]["next action"]
    assert "valid accepted-candidate prestart trace" in rows[0]["next action"]
    assert "valid accepted-candidate verifier WebSocket proof" in rows[0][
        "next action"
    ]
    assert "valid accepted-candidate artifact-root trace" in rows[0]["next action"]
    assert "valid accepted-candidate request-sequence trace" in rows[0][
        "next action"
    ]
    assert "expected-artifact-bound accepted source id" in rows[0]["next action"]
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in rows[0]["next action"]
    assert "supervised_prestart_gate" in rows[0]["next action"]
    assert "expected_supervised_prestart_gate" in rows[0]["next action"]
    assert "supervised_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in rows[0][
        "next action"
    ]
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    fader_row = next(row for row in rows if row["blocker"] == "Supervised fader write proof")
    gain_row = next(
        row for row in rows if row["blocker"] == "Supervised gain-down / trim proof"
    )
    for row in (fader_row, gain_row):
        assert row["status"] == "blocked_by_readonly_preflight"
        assert "accepted decision JSON" in row["next action"]
        assert "backend_port_prestart_gate" in row["next action"]
        assert "expected_backend_port_prestart_gate" in row["next action"]
        assert "without live authority" in row["next action"]
        assert "final handoff operator-sequence step" in row["next action"]
        assert "valid accepted-candidate prestart trace" in row["next action"]
        assert "valid accepted-candidate verifier WebSocket proof" in row[
            "next action"
        ]
        assert "valid accepted-candidate artifact-root trace" in row["next action"]
        assert "valid accepted-candidate request-sequence trace" in row["next action"]
        assert "expected-artifact-bound accepted source id" in row["next action"]
        assert "supervised_prestart_gate" in row["next action"]
        assert "expected_supervised_prestart_gate" in row["next action"]
    assert all(row["status"] != "closed" for row in rows)
    assert progress.first_open_blocker(rows)["blocker"] == "WING read-only connection proof"


def test_runbooks_keep_readonly_before_supervised_execute():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    supervised = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")

    assert "wing_readonly_preflight_runner.py" in readonly
    assert "review_wing_readonly_preflight_artifacts.py" in readonly
    assert "--require-physical-readback" in readonly
    assert "--operator-confirm-channels" in readonly
    assert "--url ws://localhost:8765" in readonly
    assert "--value-types fader,gain,pan" in readonly
    assert "--artifact-root artifacts/wing_readonly_preflight" in readonly
    assert "--value-types fader,gain \\" not in readonly
    assert "--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json" in readonly
    assert "--validate-run-inputs-only" in readonly
    assert "status: ready_for_websocket" in readonly
    assert "reason: run_inputs_validated_no_websocket" in readonly
    assert "validation_only: true" in readonly
    assert "websocket_opened: false" in readonly
    assert "contacts_backend: false" in readonly
    assert "contacts_wing: false" in readonly
    assert "expected_plan_match: true" in readonly
    assert "expected_plan_violations: []" in readonly
    assert "sent_request_types: []" in readonly
    assert "--execute" not in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "write_request_sent: false" in readonly
    assert "proposal_request_sent: false" in readonly
    assert "readiness_matrix_update_allowed: false" in readonly
    assert "readiness_closure_allowed: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "operator_confirmed_channels == channels" in readonly
    assert "strict `send`/`recv` pairs" in readonly
    assert "channel, value type, and address" in readonly
    assert "final handoff operator-sequence step" in readonly
    assert "operator_sequence_has_supervised_prestart_gate" in readonly
    assert "operator_sequence_supervised_prestart_gate_is_final" in readonly
    assert "supervised_prestart_gate" in readonly
    assert "expected_supervised_prestart_gate" in readonly
    assert "artifact_operator_next_steps" in readonly
    assert "before opening a backend/WING WebSocket" in readonly
    assert "they do not approve live writes or close readiness rows" in readonly
    assert "artifact_manifest_verifier_required: true" in readonly
    assert "artifact_manifest_verifier_command" in readonly
    assert "artifact_manifest_verifier_target_artifact_dir" in readonly
    assert "artifact_manifest_verifier_status_required: passed" in readonly
    assert "artifact_manifest_verifier_failed_checks_required: []" in readonly
    assert "artifact_manifest_verifier_required_fields_missing: []" in readonly
    assert "artifact_manifest_verifier_contacts_wing: false" in readonly
    assert "artifact_manifest_verifier_mutates_console_state: false" in readonly
    assert "artifact_manifest_verifier_queries_console: false" in readonly
    assert "artifact_manifest_verifier_readiness_closure_allowed: false" in readonly
    assert "artifact_manifest_verifier_live_authority_granted: false" in readonly
    assert "summary_json_artifact_manifest_verifier_fields_present: true" in readonly
    assert "manifest_artifact_manifest_verifier_fields_present: true" in readonly
    assert "summary_md_artifact_manifest_verifier_fields_rendered: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert "manifest_artifact_manifest_verifier_command_matches_expected: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert "verifier_report_read_only_matches_expected: true" in readonly
    assert "verifier_report_live_write_matches_expected: true" in readonly
    assert "verifier_report_contacts_wing_matches_expected: true" in readonly
    assert "verifier_report_contacts_backend_matches_expected: true" in readonly
    assert "verifier_report_mutates_console_state_matches_expected: true" in readonly
    assert "verifier_report_queries_console_matches_expected: true" in readonly
    assert "verifier_report_executes_tests_matches_expected: true" in readonly
    assert (
        "verifier_report_readiness_closure_allowed_matches_expected: true"
        in readonly
    )
    assert "verifier_report_live_authority_granted_matches_expected: true" in readonly
    assert "artifact_manifest_verifier_check_mismatches: []" in readonly
    assert "artifact_manifest_verifier_check_<check_name>: true" in readonly
    assert "artifact_manifest_verifier_check_manifest_type: true" in readonly
    assert (
        "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_check_summary_denies_live_authority: true"
        in readonly
    )
    assert "artifact_manifest_verifier_must_pass_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_passed_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_websocket_opened_after_pass: true" in readonly
    assert "artifact_manifest_verifier_operator_sequence_step_index: 1" in readonly
    assert (
        "expected_artifact_manifest_verifier_operator_sequence_step_index: 1"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_operator_sequence_step" in readonly
    assert "expected_artifact_manifest_verifier_operator_sequence_step" in readonly
    assert (
        "artifact_manifest_verifier_operator_sequence_step_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_required_markers" in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS:
        assert marker in readonly
    assert "artifact_manifest_verifier_operator_sequence_before_preview: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in readonly
    assert "backend_port_prestart_check_operator_sequence_step" in readonly
    assert "prestart_backend_port_check" in readonly
    assert "unexpected listeners" in readonly
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in readonly
    assert "backend_connection_confirmation_step_index: 3" in readonly
    assert "operator_sequence_has_backend_connection_confirmation: true" in readonly
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in readonly
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readonly
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readonly
    assert "existing backend connection" in readonly
    assert "intended WING Rack" in readonly
    assert "read-only inspection" in readonly
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in readonly
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in readonly
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in readonly
    assert "mvp_blocker_anchor" in readonly
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor_sha256" in readonly
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readonly
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readonly
    assert "operator_sequence_count: 12" in readonly
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in readonly
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in readonly
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in readonly
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in readonly
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in readonly
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readonly
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in readonly
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in readonly
    )
    assert "current first open blocker" in readonly
    assert "backend_port_prestart_gate" in readonly
    assert "expected_backend_port_prestart_gate" in readonly
    assert "backend_port_prestart_gate.grants_live_authority: false" in readonly
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in readonly
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in readonly
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in readonly
    )
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in readonly
    assert "backend_port_prestart_gate.required_markers" in readonly
    assert "Before starting the intended backend/WING session" in readonly
    assert "backend_port_prestart_check_command" in readonly
    assert "backend_port_prestart_check_command_text" in readonly
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in readonly
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary both show"
        in readonly
    )
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "backend_port_prestart_check_failure_detail_fields" in readonly
    assert "listener_processes" in readonly
    assert "listener_process_error" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "operator_next_step" in readonly
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in readonly
    assert "listener_confirmation_template_accepted_by_template: false" in readonly
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in readonly
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in readonly
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in readonly
    assert "REPLACE_WITH_*" in readonly
    assert "manual_review_required_before_readiness_update: true" in readonly
    assert "readiness_matrix_update_allowed_by_packet: false" in readonly
    assert "readiness_closure_allowed_by_packet: false" in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "mutates_console_state: false" in readonly
    assert "queries_console: false" in readonly
    assert "executes_tests: false" in readonly
    assert "operator_required: true" in readonly
    assert "expected_plan_required_for_real_run: true" in readonly
    assert "physical_preflight_requires_existing_backend_connection: true" in readonly
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "execution_allowed_by_packet: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "full_autonomous_ready: false" in readonly
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in readonly
    )
    assert "expected_artifacts" in readonly
    assert "expected_artifact_globs" in readonly
    assert "plan_expected_artifacts" in readonly
    assert "expected_artifacts_match_plan: true" in readonly
    assert "summary_glob_matches_plan: true" in readonly
    assert "event_log_glob_matches_plan: true" in readonly
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in readonly
    assert "supervised_prestart_gate.grants_live_authority: false" in readonly
    assert "required_supervised_prestart_gate_markers" in readonly
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in readonly
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in readonly
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in readonly
    )
    assert "supervised_prestart_gate.requires_operator_decision_status:" in readonly
    assert "accepted_decision_ready" in readonly
    assert "supervised_prestart_gate.unlocks_runbook:" in readonly
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in readonly
    assert "--backend-listener-confirmation" in readonly
    assert "wing_backend_port_listener_confirmation.v1" in readonly
    assert "operator_confirmed_intended_backend" in readonly
    assert "confirms_intended_backend" in readonly
    assert "reviewed_listener_pid" in readonly
    assert "reviewed_listener_command_line" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "live_authority_granted" in readonly
    assert '"artifact_root": "artifacts/wing_readonly_preflight"' in readonly
    assert "`review_id` field must remain present" in readonly
    assert (
        "keeps the record invalid even if the artifact hashes still match"
        in readonly
    )
    assert (
        "`template_status: operator_must_replace_placeholders_before_ledger_acceptance`"
        in readonly
    )
    assert "unedited pending template and fails closed" in readonly
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_sha256`, `events_sha256`"
        in readonly
    )
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_path`, `events_path`"
        in readonly
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readonly
    )
    assert "valid accepted-candidate prestart trace" in readonly
    assert "expected-artifact-bound accepted source id" in readonly
    assert "wing_operator_proposal_validation_runner.py" in supervised
    assert "--execute" in supervised
    assert "blocked until a clean read-only preflight" in supervised
    assert "valid accepted-candidate prestart trace" in supervised
    assert "valid accepted-candidate verifier WebSocket proof" in supervised
    assert "valid accepted-candidate artifact-root trace" in supervised
    assert "valid accepted-candidate request-sequence trace" in supervised
    assert "expected-artifact-bound accepted source id" in supervised
    assert "supervised_prestart_gate" in supervised
    assert "expected_supervised_prestart_gate" in supervised
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in supervised
    assert (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in supervised
    )
    assert "accepted_candidate_prestart_contract_violations_by_source" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in readonly
    )
    assert "accepted_candidate_prestart_contract_violations_by_source: {}" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in readonly
    )
    assert "negative delta" in supervised
    assert "Full autonomous" not in supervised


def test_readonly_runbook_requires_no_wing_manifest_summary_surface_sidecar_gate():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    normalized_readonly = " ".join(readonly.split())

    assert "Before opening a backend/WING WebSocket" in readonly
    assert "do not open the WING session" in normalized_readonly
    assert "do not close readiness" in readonly
    assert "approve live writes" in readonly
    assert "grant live authority" in readonly
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in readonly


def test_progress_tool_generates_latest_report_from_blocker_table(tmp_path):
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)
    current_command = progress.current_first_command(BLOCKERS_DOC)
    assert current_command is not None
    assert current_command.startswith("Use `PYTHONPATH=backend python3")
    assert "required_supervised_proof_plan_summary_fields" in current_command
    assert "expected_required_supervised_proof_plan_summary_fields" in current_command
    assert (
        "required_supervised_proof_plan_summary_fields_match_contract: true"
        in current_command
    )
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in current_command
    )
    for marker in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert marker in current_command
    assert "operator-readable no-WING `summary.md`" in current_command
    assert "operator_markdown_summary_required: true" in current_command
    assert "operator_markdown_summary_field_lines_rendered: true" in current_command
    assert "operator_markdown_summary_missing_field_lines: []" in current_command
    assert "operator_markdown_summary_artifact: summary.md" in current_command
    assert "operator_markdown_summary_sha256" in current_command
    assert "operator_markdown_summary_sha256_available: true" in current_command
    assert "operator_markdown_summary_actual_sha256" in current_command
    assert "operator_markdown_summary_sha256_matches_actual: true" in current_command
    assert "artifact_manifest_summary_surface_required: true" in current_command
    assert "artifact_manifest_summary_surface_missing_fields: []" in current_command
    assert "artifact_manifest_summary_surface_required_field_lines" in current_command
    assert (
        "artifact_manifest_summary_surface_missing_field_lines: []"
        in current_command
    )
    assert (
        "artifact_manifest_summary_surface_field_lines_rendered: true"
        in current_command
    )
    assert "artifact_manifest_contract_id" in current_command
    assert "artifact_manifest_contract_required: true" in current_command
    assert "artifact_manifest_contract_fields" in current_command
    assert "artifact_manifest_contract_sha256" in current_command
    assert "artifact_manifest_contract_sha256_available: true" in current_command
    assert (
        "artifact_manifest_contract_sha256_matches_expected: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_required: true" in current_command
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in current_command
    assert "artifact_manifest_contract_cross_surface_required_fields" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_expected_match: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in current_command
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in current_command
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in current_command
    assert "artifact_manifest_path: artifact-manifest.json" in current_command
    assert "artifact_manifest_artifact_count: 2" in current_command
    assert "artifact_manifest_required_artifact_count: 2" in current_command
    assert "artifact_manifest_artifact_count_matches_required: true" in current_command
    assert "artifact_manifest_required_artifact_ids" in current_command
    assert "artifact_manifest_artifact_ids" in current_command
    assert "artifact_manifest_required_artifacts_present: true" in current_command
    assert "artifact_manifest_required_artifact_paths" in current_command
    assert "artifact_manifest_artifact_paths" in current_command
    assert "artifact_manifest_required_artifact_paths_present: true" in current_command
    assert (
        "artifact_manifest_required_artifact_sha256_available: true"
        in current_command
    )
    assert "artifact_manifest_summary_json_sha256" in current_command
    assert "artifact_manifest_summary_json_sha256_available: true" in current_command
    assert "artifact_manifest_summary_md_sha256" in current_command
    assert "artifact_manifest_summary_md_sha256_available: true" in current_command
    assert "artifact_manifest_readiness_closure_allowed: false" in current_command
    assert "artifact_manifest_live_authority_granted: false" in current_command
    assert "Before opening a backend/WING WebSocket" in current_command
    assert "do not open the WING session" in current_command
    assert "do not close readiness" in current_command
    assert "approve live writes" in current_command
    assert "grant live authority" in current_command
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in current_command
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in current_command
    assert "backend_port_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_backend_port_prestart_gate_matches_expected: true"
        in current_command
    )
    assert "supervised_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_supervised_prestart_gate_matches_expected: true"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_source_ids"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_roots"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_roots"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert "project_director_selected_action_anchor" in current_command
    assert "expected_project_director_selected_action_anchor" in current_command
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in current_command
    )
    assert "project_director_selected_action_sha256" in current_command
    assert "actual_project_director_selected_action_sha256" in current_command
    assert "expected_project_director_selected_action_sha256" in current_command
    assert "project_director_selected_action_sha256_available: true" in current_command
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_anchor: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_text: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_missing_markers: []"
        in current_command
    )
    assert progress.current_first_command_missing_baseline_markers(current_command) == []
    assert progress.current_first_command_covers_baseline(current_command)
    missing_proof_plan_fields_command = current_command.replace(
        "required_supervised_proof_plan_summary_fields",
        "missing_supervised_proof_plan_summary_fields",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_proof_plan_fields_command
    )
    assert "required_supervised_proof_plan_summary_fields" in (
        progress.current_first_command_missing_baseline_markers(
            missing_proof_plan_fields_command
        )
    )
    failing_candidate_artifact_root_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true",
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_artifact_root_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_artifact_root_command
        )
    )
    missing_contact_query_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source",
        "readonly_preflight_review_decision_accepted_candidate_queries_console_missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_query_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_query_trace_command
        )
    )
    missing_contact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_trace_command
        )
    )
    missing_contact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_source_trace_command
        )
    )
    failing_contact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_value_gate_command
        )
    )
    failing_request_sequence_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_value_gate_command
        )
    )
    missing_request_sequence_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_source_trace_command
        )
    )
    failing_expected_artifact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_value_gate_command
        )
    )
    missing_expected_artifact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_source_trace_command
        )
    )
    missing_expected_artifact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_trace_command
        )
    )
    missing_request_sequence_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_trace_command
        )
    )
    missing_markdown_summary_command = current_command.replace(
        "operator-readable no-WING `summary.md`",
        "machine-only no-WING summary",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_summary_command
    )
    assert "operator-readable no-WING `summary.md`" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_summary_command
        )
    )
    missing_markdown_rendered_command = current_command.replace(
        "operator_markdown_summary_field_lines_rendered: true",
        "operator_markdown_summary_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_rendered_command
    )
    assert "operator_markdown_summary_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_rendered_command
        )
    )
    missing_markdown_sha_command = current_command.replace(
        "operator_markdown_summary_sha256_available: true",
        "operator_markdown_summary_sha256_available: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_command
    )
    assert "operator_markdown_summary_sha256_available: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_command
        )
    )
    missing_markdown_sha_match_command = current_command.replace(
        "operator_markdown_summary_sha256_matches_actual: true",
        "operator_markdown_summary_sha256_matches_actual: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_match_command
    )
    assert "operator_markdown_summary_sha256_matches_actual: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_match_command
        )
    )
    missing_artifact_manifest_command = current_command.replace(
        "artifact_manifest_path: artifact-manifest.json",
        "artifact_manifest_path: missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_command
    )
    assert "artifact_manifest_path: artifact-manifest.json" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_command
        )
    )
    missing_artifact_manifest_surface_command = current_command.replace(
        "artifact_manifest_summary_surface_required: true",
        "artifact_manifest_summary_surface_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_command
    )
    assert "artifact_manifest_summary_surface_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_command
        )
    )
    missing_artifact_manifest_surface_rendered_command = current_command.replace(
        "artifact_manifest_summary_surface_field_lines_rendered: true",
        "artifact_manifest_summary_surface_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_rendered_command
    )
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_rendered_command
        )
    )
    missing_artifact_manifest_contract_digest_command = current_command.replace(
        "artifact_manifest_contract_sha256_matches_expected: true",
        "artifact_manifest_contract_sha256_matches_expected: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_contract_digest_command
    )
    assert "artifact_manifest_contract_sha256_matches_expected: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_contract_digest_command
        )
    )
    missing_artifact_manifest_cross_surface_command = current_command.replace(
        "artifact_manifest_contract_cross_surface_match: true",
        "artifact_manifest_contract_cross_surface_match: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_cross_surface_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_cross_surface_command
        )
    )
    missing_artifact_manifest_count_match_command = current_command.replace(
        "artifact_manifest_artifact_count_matches_required: true",
        "artifact_manifest_artifact_count_matches_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_count_match_command
    )
    assert "artifact_manifest_artifact_count_matches_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_count_match_command
        )
    )
    missing_artifact_manifest_closure_command = current_command.replace(
        "artifact_manifest_readiness_closure_allowed: false",
        "artifact_manifest_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_closure_command
    )
    assert "artifact_manifest_readiness_closure_allowed: false" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_closure_command
        )
    )
    missing_artifact_manifest_ids_command = current_command.replace(
        "artifact_manifest_required_artifacts_present: true",
        "artifact_manifest_required_artifacts_present: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_ids_command
    )
    assert "artifact_manifest_required_artifacts_present: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_ids_command
        )
    )
    missing_director_action_digest_command = current_command.replace(
        "project_director_selected_action_sha256",
        "missing_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_director_action_digest_command
    )
    assert "project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_director_action_digest_command
        )
    )
    missing_expected_digest_command = current_command.replace(
        "expected_project_director_selected_action_sha256",
        "missing_expected_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_digest_command
    )
    assert "expected_project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_expected_digest_command
        )
    )
    report = progress.build_progress_report(
        rows,
        generated_at="2026-06-06T00:00:00Z",
        current_first_command=current_command,
    )

    assert "Total blockers: 9" in report
    assert "Closed blockers: 0" in report
    assert "Current first open blocker: WING read-only connection proof" in report
    assert "Next required action source: Current First Command" in report
    assert "Current First Command baseline coverage required: true" in report
    assert "Current First Command baseline coverage passed: true" in report
    assert (
        "Current First Command baseline coverage marker count: "
        f"{len(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)}"
        in report
    )
    assert "Current First Command baseline missing markers: []" in report
    assert (
        f"Selected evidence file count: {len(progress.extract_evidence_files(rows[0]['evidence']))}"
        in report
    )
    assert "backend_port_prestart_gate" in report
    assert "expected_backend_port_prestart_gate" in report
    assert "backend_port_prestart_gate.grants_live_authority: false" in report
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in report
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in report
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in report
    assert "backend_port_prestart_gate.required_markers" in report
    assert "Before starting the intended backend/WING session" in report
    assert "backend_port_prestart_check_command" in report
    assert "backend_port_prestart_check_command_text" in report
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in report
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in report
    )
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "backend_port_prestart_check_failure_detail_fields" in report
    assert "listener_processes" in report
    assert "listener_process_error" in report
    assert "safe_to_stop_listener_by_gate" in report
    assert "operator_next_step" in report
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in report
    assert "listener_confirmation_template_path" in report
    assert "listener_confirmation_template_acceptance_status" in report
    assert "listener_confirmation_template_accepted_by_template: false" in report
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in report
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in report
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in report
    assert "listener_confirmation_template_required_edits" in report
    assert "set template_status to operator_reviewed_and_edited" in report
    assert "set accepted_by_template to true" in report
    assert "REPLACE_WITH_*" in report
    assert "template_status: operator_reviewed_and_edited" in report
    assert "accepted_by_template: true" in report
    assert "manual_review_required_before_readiness_update: true" in report
    assert "readiness_matrix_update_allowed_by_packet: false" in report
    assert "readiness_closure_allowed_by_packet: false" in report
    assert "read_only: true" in report
    assert "live_write: false" in report
    assert "mutates_console_state: false" in report
    assert "queries_console: false" in report
    assert "executes_tests: false" in report
    assert "operator_required: true" in report
    assert "expected_plan_required_for_real_run: true" in report
    assert "backend_port_prestart_summary_type: no_wing_release_check" in report
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in report
    assert "backend_port_prestart_summary_read_only: true" in report
    assert "backend_port_prestart_summary_live_write: false" in report
    assert "backend_port_prestart_summary_contacts_backend: false" in report
    assert "backend_port_prestart_summary_contacts_wing: false" in report
    assert "backend_port_prestart_summary_mutates_console_state: false" in report
    assert "backend_port_prestart_summary_queries_console: false" in report
    assert "backend_port_prestart_summary_executes_tests: false" in report
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
        assert marker in report
    assert "physical_preflight_requires_existing_backend_connection: true" in report
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "execution_allowed_by_packet: false" in report
    assert "live_authority_granted: false" in report
    assert "full_autonomous_ready: false" in report
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in report
    )
    assert "expected_artifacts" in report
    assert "expected_artifact_globs" in report
    assert "plan_expected_artifacts" in report
    assert "expected_artifacts_match_plan: true" in report
    assert "summary_glob_matches_plan: true" in report
    assert "event_log_glob_matches_plan: true" in report
    assert "accepted_candidate_prestart_contract_violations_by_source" in report
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in report
    )
    assert "Both maps must be `{}` before supervised proof steps are planned" in report
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in report
    assert "backend_port_prestart_check_operator_sequence_step" in report
    assert "prestart_backend_port_check" in report
    assert "unexpected listeners" in report
    assert "backend_connection_confirmation_step_index: 3" in report
    assert "operator_sequence_has_backend_connection_confirmation: true" in report
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in report
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in report
    )
    assert "operator_sequence_backend_connection_confirmation_step" in report
    assert "existing backend connection" in report
    assert "intended WING Rack" in report
    assert "read-only inspection" in report
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in report
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in report
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in report
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in report
    )
    assert "mvp_blocker_anchor" in report
    assert "mvp_blocker_anchor_matches_current_first_open: true" in report
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in report
    assert "project_director_selected_action_anchor" in report
    assert "expected_project_director_selected_action_anchor" in report
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_source: Current First Command" in report
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in report
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_sha256" in report
    assert "actual_project_director_selected_action_sha256" in report
    assert "expected_project_director_selected_action_sha256" in report
    assert "project_director_selected_action_sha256_available: true" in report
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in report
    )
    assert "project_director_selected_action_sha256_matches_anchor: true" in report
    assert "project_director_selected_action_sha256_matches_text: true" in report
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in report
    )
    assert "project_director_current_first_command_baseline_missing_markers: []" in report
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in report
    assert "operator_sequence_count: 12" in report
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in report
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in report
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in report
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in report
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in report
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in report
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in report
    )
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in report
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in report
    assert "operator_sequence_has_supervised_prestart_gate: true" in report
    assert "required_supervised_prestart_gate_markers" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in report
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in report
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in report
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in report
    )
    assert "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true" in report
    assert "accepted-candidate prestart trace" in report
    assert "expected-artifact-bound accepted source id" in report
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in report
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in report
    )
    assert "final handoff operator-sequence step" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "Offline tooling, shadow" in report

    output = tmp_path / "MVP_PROGRESS_LATEST.md"
    progress.write_report(output, report)
    assert output.read_text(encoding="utf-8") == report


def test_correction_methods_roadmap_preserves_full_console_goal_without_live_widening():
    roadmap = CORRECTION_ROADMAP.read_text(encoding="utf-8")
    backlog = CORRECTION_BACKLOG.read_text(encoding="utf-8")

    for family in (
        "Gain / Trim",
        "Pan / Stereo Placement",
        "EQ",
        "Compressor",
        "FX / Sends",
        "Fader / Balance",
    ):
        assert family in roadmap

    assert "offline analysis" in roadmap
    assert "proposal-only" in roadmap
    assert "No method may skip stages." in roadmap
    assert "must not widen live writes" in roadmap
    assert "separate ADR after WING MVP" in backlog
    assert "Do not widen live writes" in backlog


def test_codex_prompt_requires_mvp_rails_before_new_work():
    text = PROMPT.read_text(encoding="utf-8")

    assert "Docs/MVP_WING_SUPERVISED.md" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/reports/MVP_PROGRESS_LATEST.md" in text
    assert "first non-closed row" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_check_command" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "failed busy-port no-WING summary and no-WING handoff summary" in text
    assert "must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe `true` values" in text
    assert "failed handoff is auditable" in text
    assert "packet safety flags" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
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
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
    assert "supervised_prestart_gate.requires_operator_decision_status:" in text
    assert "supervised_prestart_gate.unlocks_runbook:" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "required_supervised_proof_plan_summary_fields" in text
    assert "expected_required_supervised_proof_plan_summary_fields" in text
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in text
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in text
    )
    for marker in (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete",
    ):
        assert marker in text
    assert "operator-readable no-WING `summary.md`" in text
    assert "operator_markdown_summary_required: true" in text
    assert "operator_markdown_summary_field_lines_rendered: true" in text
    assert "operator_markdown_summary_missing_field_lines: []" in text
    assert "operator_markdown_summary_artifact: summary.md" in text
    assert "operator_markdown_summary_sha256" in text
    assert "operator_markdown_summary_sha256_available: true" in text
    assert "operator_markdown_summary_actual_sha256" in text
    assert "operator_markdown_summary_sha256_matches_actual: true" in text
    assert "artifact_manifest_summary_surface_required: true" in text
    assert "artifact_manifest_summary_surface_missing_fields: []" in text
    assert "artifact_manifest_summary_surface_required_field_lines" in text
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in text
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in text
    assert "artifact_manifest_contract_id" in text
    assert "artifact_manifest_contract_required: true" in text
    assert "artifact_manifest_contract_fields" in text
    assert "artifact_manifest_contract_sha256" in text
    assert "artifact_manifest_contract_sha256_available: true" in text
    assert "artifact_manifest_contract_sha256_matches_expected: true" in text
    assert "artifact_manifest_contract_cross_surface_required: true" in text
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in text
    assert "artifact_manifest_contract_cross_surface_required_fields" in text
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in text
    assert "artifact_manifest_contract_cross_surface_match: true" in text
    assert "artifact_manifest_summary_json_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_summary_md_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in text
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in text
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in text
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in text
    assert "artifact_manifest_path: artifact-manifest.json" in text
    assert "artifact_manifest_artifact_count: 2" in text
    assert "artifact_manifest_required_artifact_count: 2" in text
    assert "artifact_manifest_artifact_count_matches_required: true" in text
    assert "artifact_manifest_required_artifact_ids" in text
    assert "artifact_manifest_artifact_ids" in text
    assert "artifact_manifest_required_artifacts_present: true" in text
    assert "artifact_manifest_required_artifact_paths" in text
    assert "artifact_manifest_artifact_paths" in text
    assert "artifact_manifest_required_artifact_paths_present: true" in text
    assert "artifact_manifest_required_artifact_sha256_available: true" in text
    assert "artifact_manifest_summary_json_sha256" in text
    assert "artifact_manifest_summary_json_sha256_available: true" in text
    assert "artifact_manifest_summary_md_sha256" in text
    assert "artifact_manifest_summary_md_sha256_available: true" in text
    assert "artifact_manifest_readiness_closure_allowed: false" in text
    assert "artifact_manifest_live_authority_granted: false" in text
    assert "backend_port_prestart_gate_matches_expected: true" in text
    assert "declared_backend_port_prestart_gate_matches_expected: true" in text
    assert "supervised_prestart_gate_matches_expected: true" in text
    assert "declared_supervised_prestart_gate_matches_expected: true" in text
    assert "readonly_preflight_review_decision_expected_candidate_prestart_source_ids" in text
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source"
        in text
    )


def test_docs_index_points_to_current_wing_mvp_prestart_gate():
    text = DOCS_INDEX.read_text(encoding="utf-8")

    assert "Current WING MVP rails" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in text
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_REQUIREMENT_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_PENDING_SUMMARY_FIELDS_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_FINAL_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_DECISION_RUNBOOK_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_REQUIRED_MARKERS_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_INDEXES_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_PROJECT_GOAL_SUPERVISED_AUTOMIX_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert "final handoff operator-sequence step" in text
    assert "accepted-candidate prestart gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.runs_command" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "backend_port_prestart_check_command" in text
    assert "failed busy-port no-WING summary" in text
    assert "no-WING handoff summary must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "manual_review_required_before_readiness_update" in text
    assert "readiness_matrix_update_allowed_by_packet" in text
    assert "readiness_closure_allowed_by_packet" in text
    assert "live_write" in text
    assert "execution_allowed_by_packet" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "operator_sequence_count" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
