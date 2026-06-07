import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_direction_guard.py"
PROMPT_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"
README_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "README.md"


def load_guard_module():
    spec = importlib.util.spec_from_file_location("codex_direction_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direction_guard_accepts_wing_preflight_and_timeout_work():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            ("git_log", "4f13dbc Add WING read-only preflight runner"),
            ("git_log", "8f96d4d Add WING timeout fail-closed proof gate"),
            ("report", "physical readback and operator-visible evidence report"),
        ]
    )

    assert report["status"] == "on_target"
    assert report["score"] >= 70
    assert report["summary"]["target_item_count"] == 3
    assert report["summary"]["drift_item_count"] == 0
    assert "run read-only preflight" in report["next_allowed_actions"][0]


def test_direction_guard_flags_memory_queue_consumer_drift():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            ("git_log", "Add promotion queue import result consumer chain"),
            ("git_status", "M backend/handlers/product_state_handlers.py"),
            ("report", "memory-trust promotion memory queue-import handoff"),
        ]
    )

    assert report["status"] == "drifting"
    assert report["summary"]["target_item_count"] == 0
    assert report["summary"]["drift_item_count"] >= 2
    assert report["summary"]["action_required"] is True


def test_direction_guard_marks_mixed_work_for_review():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            ("git_log", "Add WING read-only preflight runner"),
            ("git_log", "Add ProductState promotion memory preflight handoff"),
        ]
    )

    assert report["status"] == "mixed_needs_review"
    assert report["summary"]["target_item_count"] == 1
    assert report["summary"]["drift_item_count"] == 1


def test_direction_guard_ignores_forbidden_policy_sections_when_scoring_drift():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            (
                "report",
                """
# Project Director Agent Directive

## Selected Work

- Selected blocker: `WING read-only connection proof`
- Selected action: Run `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.

## Forbidden Scope

- No new memory-trust chains unless they close the selected blocker.
- No new queue-import chains unless they close the selected blocker.
- No new result-consumer/report-only/read-model work unless it closes the selected blocker.

## Required Files

- `tools/wing_readonly_preflight_runner.py`
""",
            ),
            ("git_log", "Add WING read-only preflight readback identity gate"),
            ("report", "physical readback and timeout fail-closed proof context"),
        ]
    )

    assert report["status"] == "on_target"
    assert report["summary"]["drift_item_count"] == 0
    assert report["summary"]["mixed_item_count"] == 0


def test_direction_guard_ignores_generated_guard_evidence_sections_when_scoring_drift():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            (
                "report:.paperclip/reports/codex_direction_guard_final.md",
                """
codex_direction_guard_final.md
# Codex Direction Guard

## Verdict

There is WING proof progress.

## Next Allowed Actions

- Run read-only preflight and capture physical readback.

## Mixed Evidence

- `report:.paperclip/reports/old.md`: WING read-only preflight plus memory-trust queue-import result consumer.
  - target: `wing read-only`
  - drift: `memory-trust, queue-import, result consumer`
""",
            ),
            ("git_log", "Add WING read-only preflight readback identity gate"),
            ("report", "physical readback and timeout fail-closed proof context"),
        ]
    )

    assert report["status"] == "on_target"
    assert report["summary"]["drift_item_count"] == 0
    assert report["summary"]["mixed_item_count"] == 0


def test_direction_guard_still_flags_drift_outside_forbidden_policy_sections():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [
            (
                "report",
                """
# Cycle Report

## Summary

Implemented a memory-trust queue-import result consumer around the WING read-only preflight handoff.

## Forbidden Scope

- No new memory-trust chains unless they close the selected blocker.
""",
            )
        ]
    )

    assert report["status"] == "mixed_needs_review"
    assert report["summary"]["mixed_item_count"] == 1
    assert report["mixed_items"][0]["drift_matches"]


def test_direction_guard_markdown_and_json_outputs(tmp_path):
    guard = load_guard_module()
    report = guard.analyze_evidence(
        [("git_log", "Add supervised fader rollback proof artifact")]
    )

    markdown = guard.render_markdown(report)
    assert "# Codex Direction Guard" in markdown
    assert "before Codex selects a task" in markdown
    assert "Forbidden Next Actions" in markdown

    output_path = tmp_path / "guard.json"
    guard.write_report(output_path, json.dumps(report, ensure_ascii=False))
    assert json.loads(output_path.read_text(encoding="utf-8"))["type"] == "codex_direction_guard"
    assert report["mvp_blocker_table"] == ".ai/project-layer/wing_mvp_blockers.md"
    assert report["continue_goal_mode"] is True
    assert report["stop_scope"] == "broad_work_only"


def test_direction_guard_stop_the_line_marks_drift():
    guard = load_guard_module()

    report = guard.analyze_evidence(
        [("git_log", "Add promotion queue import result consumer chain")]
    )

    assert report["status"] == "drifting"
    assert report["stop_the_line"] is True


def test_codex_cycle_prompt_runs_direction_guard_before_plan_and_final_report():
    text = PROMPT_PATH.read_text(encoding="utf-8")

    assert "codex_direction_guard.py" in text
    assert "перед выбором задачи" in text.lower()
    assert "перед финальным отчетом" in text
    assert "codex_direction_guard_latest.md" in text
    assert "codex_direction_guard_final.md" in text
    assert "project_director_agent.py" in text


def test_codex_autoloop_readme_documents_direction_guard_activation_time():
    text = README_PATH.read_text(encoding="utf-8")

    assert "Direction guard activation" in text
    assert "before task selection" in text
    assert "before the final cycle report" in text
    assert "1200" in text
    assert "seconds" in text
