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
    assert "accepted-candidate prestart trace" in directive["selected_action"]
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in directive["selected_action"]
    )
    assert directive["stop_broad_work"] is True
    assert directive["continue_goal_mode"] is True
    assert "Docs/MVP_WING_SUPERVISED.md" in directive["required_files"]
    assert "backend/wing_readonly_preflight_contract.py" in directive["required_files"]
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in directive["required_files"]
    assert "tools/wing_readonly_preflight_runner.py" in directive["required_files"]
    assert "tools/review_wing_readonly_preflight_artifacts.py" in directive["required_files"]
    assert (
        "tools/build_wing_readonly_preflight_handoff_packet.py"
        in directive["required_files"]
    )
    assert "tools/no_wing_release_check.py" in directive["required_files"]
    assert any("Develop correction methods" in item for item in directive["allowed_scope"])


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
        "allowed_scope": ["Work only on the selected first non-closed MVP blocker."],
        "forbidden_scope": ["No queue-import chains."],
        "required_files": ["Docs/MVP_WING_SUPERVISED.md"],
    }

    markdown = director.render_markdown(directive)

    assert "# Project Director Agent Directive" in markdown
    assert "Selected blocker" in markdown
    assert "Continue Goal mode: `true`" in markdown
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
    assert "backend_port_prestart_gate" in readme
    assert "expected_backend_port_prestart_gate" in readme
    assert "backend_port_prestart_gate.grants_live_authority: false" in readme
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in readme
    )
    assert "supervised_prestart_gate" in prompt
    assert "expected_supervised_prestart_gate" in prompt
    assert "supervised_prestart_gate.grants_live_authority: false" in prompt
    assert "supervised_prestart_gate" in readme
    assert "expected_supervised_prestart_gate" in readme
    assert "supervised_prestart_gate.grants_live_authority: false" in readme
