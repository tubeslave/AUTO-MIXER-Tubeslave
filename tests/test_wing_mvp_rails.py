import importlib.util
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
PROMPT = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"

DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS = (
    "declared_operator_sequence_count: 10",
    "declared_expected_operator_sequence_count: 10",
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


def load_progress_module():
    spec = importlib.util.spec_from_file_location("update_mvp_progress", PROGRESS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert "expected_operator_sequence_count: 10" in text
    assert "operator_sequence_count_matches_contract: true" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 9"
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
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
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
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 10" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index: 8" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8" in text
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index: 9" in text
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
    assert "expected_operator_sequence_count: 10" in rows[0]["next action"]
    assert "operator_sequence_count_matches_contract: true" in rows[0][
        "next action"
    ]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 9"
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
    assert "backend_port_prestart_summary_live_write: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in rows[0][
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
        "backend_connection_confirmation_step_index: 2"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_has_backend_connection_confirmation: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_step_index: 2"
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
        "operator_sequence_capture_readonly_preflight_step_index: 4"
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
    assert "operator_sequence_count: 10" in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index: 8" in rows[0][
        "next action"
    ]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in rows[0][
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
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
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
    assert "operator_sequence_supervised_prestart_gate_step_index: 9" in rows[0][
        "next action"
    ]
    assert "required_supervised_prestart_gate_markers" in rows[0]["next action"]
    assert "accepted_decision_ready" in rows[0]["next action"]
    assert "valid accepted-candidate prestart trace" in rows[0]["next action"]
    assert "expected-artifact-bound accepted source id" in rows[0]["next action"]
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in rows[0]["next action"]
    assert "supervised_prestart_gate" in rows[0]["next action"]
    assert "expected_supervised_prestart_gate" in rows[0]["next action"]
    assert "supervised_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in rows[0][
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
    assert "--execute" not in readonly
    assert "write_request_sent: false" in readonly
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
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in readonly
    assert "backend_port_prestart_check_operator_sequence_step" in readonly
    assert "prestart_backend_port_check" in readonly
    assert "unexpected listeners" in readonly
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in readonly
    assert "backend_connection_confirmation_step_index: 2" in readonly
    assert "operator_sequence_has_backend_connection_confirmation: true" in readonly
    assert "operator_sequence_backend_connection_confirmation_step_index: 2" in readonly
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readonly
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readonly
    assert "existing backend connection" in readonly
    assert "intended WING Rack" in readonly
    assert "read-only inspection" in readonly
    assert "operator_sequence_capture_readonly_preflight_step_index: 4" in readonly
    assert "mvp_blocker_anchor" in readonly
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor_sha256" in readonly
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readonly
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readonly
    assert "operator_sequence_count: 10" in readonly
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in readonly
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in readonly
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in readonly
    assert "mvp_blocker_anchor_digest_review_step_index: 8" in readonly
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in readonly
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in readonly
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in readonly
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readonly
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8" in readonly
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
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
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in readonly
    assert "supervised_prestart_gate.grants_live_authority: false" in readonly
    assert "required_supervised_prestart_gate_markers" in readonly
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in readonly
    assert "operator_sequence_supervised_prestart_gate_step_index: 9" in readonly
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
    assert "expected-artifact-bound accepted source id" in supervised
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
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


def test_progress_tool_generates_latest_report_from_blocker_table(tmp_path):
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)
    current_command = progress.current_first_command(BLOCKERS_DOC)
    assert current_command is not None
    assert current_command.startswith("Use `PYTHONPATH=backend python3")
    report = progress.build_progress_report(
        rows,
        generated_at="2026-06-06T00:00:00Z",
        current_first_command=current_command,
    )

    assert "Total blockers: 9" in report
    assert "Closed blockers: 0" in report
    assert "Current first open blocker: WING read-only connection proof" in report
    assert (
        "Next required action source: Current First Command + blocker table baseline"
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
    assert "backend_connection_confirmation_step_index: 2" in report
    assert "operator_sequence_has_backend_connection_confirmation: true" in report
    assert "operator_sequence_backend_connection_confirmation_step_index: 2" in report
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in report
    )
    assert "operator_sequence_backend_connection_confirmation_step" in report
    assert "existing backend connection" in report
    assert "intended WING Rack" in report
    assert "read-only inspection" in report
    assert "operator_sequence_capture_readonly_preflight_step_index: 4" in report
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in report
    )
    assert "mvp_blocker_anchor" in report
    assert "mvp_blocker_anchor_matches_current_first_open: true" in report
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in report
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in report
    assert "operator_sequence_count: 10" in report
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in report
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in report
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in report
    assert "mvp_blocker_anchor_digest_review_step_index: 8" in report
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 8" in report
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in report
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in report
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8" in report
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8"
        in report
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in report
    )
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in report
    assert "operator_sequence_supervised_prestart_gate_step_index: 9" in report
    assert "operator_sequence_has_supervised_prestart_gate: true" in report
    assert "required_supervised_prestart_gate_markers" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in report
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
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
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
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 10" in text
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
