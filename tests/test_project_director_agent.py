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
    assert "backend_port_prestart_check_failure_detail_fields" in directive[
        "selected_action"
    ]
    assert "listener_processes" in directive["selected_action"]
    assert "safe_to_stop_listener_by_gate" in directive["selected_action"]
    assert "fail_closed_reason" in directive["selected_action"]
    assert "operator_next_step" in directive["selected_action"]
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
    assert "execution_allowed_by_packet: false" in directive["selected_action"]
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
    assert "backend_port_prestart_check_failure_detail_fields" in prompt
    assert "listener_processes" in prompt
    assert "safe_to_stop_listener_by_gate" in prompt
    assert "operator_next_step" in prompt
    assert "manual_review_required_before_readiness_update: true" in prompt
    assert "readiness_matrix_update_allowed_by_packet: false" in prompt
    assert "readiness_closure_allowed_by_packet: false" in prompt
    assert "read_only: true" in prompt
    assert "live_write: false" in prompt
    assert "mutates_console_state: false" in prompt
    assert "queries_console: false" in prompt
    assert "executes_tests: false" in prompt
    assert "operator_required: true" in prompt
    assert "execution_allowed_by_packet: false" in prompt
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in prompt
    assert "backend_port_prestart_check_operator_sequence_step" in prompt
    assert "prestart_backend_port_check" in prompt
    assert "unexpected listeners" in prompt
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in prompt
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in prompt
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
    assert "backend_port_prestart_check_failure_detail_fields" in readme
    assert "listener_processes" in readme
    assert "safe_to_stop_listener_by_gate" in readme
    assert "operator_next_step" in readme
    assert "manual_review_required_before_readiness_update: true" in readme
    assert "readiness_matrix_update_allowed_by_packet: false" in readme
    assert "readiness_closure_allowed_by_packet: false" in readme
    assert "read_only: true" in readme
    assert "live_write: false" in readme
    assert "mutates_console_state: false" in readme
    assert "queries_console: false" in readme
    assert "executes_tests: false" in readme
    assert "operator_required: true" in readme
    assert "execution_allowed_by_packet: false" in readme
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
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in readme
    assert "current first open blocker" in readme
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readme
    )
