import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRECTOR_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "project_director_agent.py"
PROMPT_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"
README_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "README.md"


def load_director_module():
    sys.path.insert(0, str(DIRECTOR_PATH.parent))
    spec = importlib.util.spec_from_file_location("project_director_agent", DIRECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert "capture_command_prestart_summary" in directive["selected_action"]
    assert "capture_command_prestart_summary_matches_plan: true" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_status: passed" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_live_write: false" in directive[
        "selected_action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in directive[
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
    assert "backend_connection_confirmation_step_index: 2" in directive[
        "selected_action"
    ]
    assert "operator_sequence_has_backend_connection_confirmation: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_backend_connection_confirmation_step_index: 2" in directive[
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
    assert "operator_sequence_capture_readonly_preflight_step_index: 4" in directive[
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
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in directive[
        "selected_action"
    ]
    assert "operator_sequence_count: 10" in directive["selected_action"]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in directive[
        "selected_action"
    ]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8" in directive[
        "selected_action"
    ]
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
    assert "operator_sequence_supervised_prestart_gate_step_index: 9" in directive[
        "selected_action"
    ]
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
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in directive[
        "selected_action"
    ]
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
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
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in directive["selected_action"]
    )
    assert "accepted-candidate prestart trace" in directive["selected_action"]
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert directive["selected_evidence_file_count"] == len(directive["selected_evidence_files"])
    assert (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
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
    assert "operator_next_step" in prompt
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
    assert "backend_connection_confirmation_step_index: 2" in prompt
    assert "operator_sequence_has_backend_connection_confirmation: true" in prompt
    assert "operator_sequence_backend_connection_confirmation_step_index: 2" in prompt
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in prompt
    )
    assert "operator_sequence_backend_connection_confirmation_step" in prompt
    assert "existing backend connection" in prompt
    assert "intended WING Rack" in prompt
    assert "read-only inspection" in prompt
    assert "operator_sequence_capture_readonly_preflight_step_index: 4" in prompt
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
    assert "operator_next_step" in readme
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
    assert "backend_connection_confirmation_step_index: 2" in readme
    assert "operator_sequence_has_backend_connection_confirmation: true" in readme
    assert "operator_sequence_backend_connection_confirmation_step_index: 2" in readme
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readme
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readme
    assert "existing backend connection" in readme
    assert "intended WING Rack" in readme
    assert "read-only inspection" in readme
    assert "operator_sequence_capture_readonly_preflight_step_index: 4" in readme
    assert "supervised_prestart_gate" in prompt
    assert "expected_supervised_prestart_gate" in prompt
    assert "supervised_prestart_gate.grants_live_authority: false" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in prompt
    assert (
        "supervised_prestart_gate.requires_operator_decision_status:"
        in prompt
    )
    assert "accepted_decision_ready" in prompt
    assert "supervised_prestart_gate.unlocks_runbook:" in prompt
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in prompt
    assert "supervised_prestart_gate.operator_sequence_step_index: 9" in readme
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in readme
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status:"
        in readme
    )
    assert "supervised_prestart_gate.unlocks_runbook:" in readme
    assert "mvp_blocker_anchor_matches_current_first_open: true" in prompt
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in prompt
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in prompt
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in prompt
    assert "operator_sequence_count: 10" in prompt
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in prompt
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in prompt
    assert "operator_sequence_has_supervised_prestart_gate: true" in prompt
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in prompt
    assert "operator_sequence_supervised_prestart_gate_step" in prompt
    assert "operator_sequence_supervised_prestart_gate_step_index" in prompt
    assert "required_supervised_prestart_gate_markers" in prompt
    assert "accepted_decision_ready" in prompt
    assert "valid accepted-candidate prestart trace" in prompt
    assert "expected-artifact-bound accepted source id" in prompt
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in prompt
    assert "current first open blocker" in prompt
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in prompt
    )
    assert "supervised_prestart_gate" in readme
    assert "expected_supervised_prestart_gate" in readme
    assert "supervised_prestart_gate.grants_live_authority: false" in readme
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readme
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readme
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readme
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readme
    assert "operator_sequence_count: 10" in readme
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readme
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in readme
    assert "operator_sequence_has_supervised_prestart_gate: true" in readme
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in readme
    assert "operator_sequence_supervised_prestart_gate_step" in readme
    assert "operator_sequence_supervised_prestart_gate_step_index" in readme
    assert "required_supervised_prestart_gate_markers" in readme
    assert "accepted_decision_ready" in readme
    assert "valid accepted-candidate prestart trace" in readme
    assert "expected-artifact-bound accepted source id" in readme
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in readme
    assert "current first open blocker" in readme
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readme
    )
