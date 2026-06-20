#!/usr/bin/env python3
"""Repo-local Project Director Agent for Automixer.

The director reads the MVP rail files and the direction guard, then emits the
next Codex directive. It does not edit production code, run tests, connect to
WING, or grant live authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import codex_direction_guard  # noqa: E402
import update_mvp_progress  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / ".paperclip" / "reports" / "project_director_latest.md"
EXTERNAL_VALIDATION_STATUSES = frozenset(
    {
        "tooling_ready_waits_real_wing",
        "blocked_by_readonly_preflight",
        "blocked_by_real_artifact_review",
        "waiting_for_external_validation",
    }
)
EXTERNAL_VALIDATION_ACTION_MARKERS = (
    "real wing",
    "wing rack",
    "intended backend/wing session",
    "operator-supervised",
    "operator review",
    "hardware",
    "real artifact",
    "wing_readonly_preflight",
    "wing_readonly_preflight.md",
)
POST_M104_INTAKE_GUARD_MIN_STREAK = 10
POST_M104_INTAKE_GUARD_COMPLETION_MARKERS = (
    "capture time order",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def extract_evidence_files(evidence_text: str) -> list[str]:
    return update_mvp_progress.extract_evidence_files(evidence_text)


def required_file_candidates_from_evidence(evidence_files: Sequence[str]) -> list[str]:
    return [
        path
        for path in evidence_files
        if path.startswith(("backend/", "tools/", "Docs/runbooks/"))
    ]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_scope_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def recent_commit_subjects_from_evidence(
    evidence: Sequence[tuple[str, str]],
) -> list[str]:
    subjects: list[str] = []
    for source, text in evidence:
        if source != "git_log":
            continue
        _, _, subject = text.partition(" ")
        subjects.append(subject.strip() if subject else text.strip())
    return [subject for subject in subjects if subject]


def is_wing_readonly_intake_guard_subject(subject: str) -> bool:
    return "guard wing readonly" in normalized_scope_text(subject)


def wing_readonly_intake_guard_streak(subjects: Sequence[str]) -> list[str]:
    streak: list[str] = []
    for subject in subjects:
        if not is_wing_readonly_intake_guard_subject(subject):
            break
        streak.append(subject)
    return streak


def pre_hardware_intake_hardening_loop_guard_required(
    *, external_validation_wait: bool, recent_commit_subjects: Sequence[str]
) -> bool:
    if not external_validation_wait:
        return False
    streak = wing_readonly_intake_guard_streak(recent_commit_subjects)
    if len(streak) < POST_M104_INTAKE_GUARD_MIN_STREAK:
        return False
    return any(
        marker in normalized_scope_text(subject)
        for subject in streak
        for marker in POST_M104_INTAKE_GUARD_COMPLETION_MARKERS
    )


def selected_action_requires_external_validation(*, status: str, action: str) -> bool:
    action_lc = action.lower()
    return status in EXTERNAL_VALIDATION_STATUSES and any(
        marker in action_lc for marker in EXTERNAL_VALIDATION_ACTION_MARKERS
    )


def allowed_scope_for_selected_action(
    *,
    selected_blocker: str,
    external_validation_wait: bool,
    pre_hardware_intake_loop_guard: bool = False,
) -> list[str]:
    if external_validation_wait:
        allowed_scope = [
            "Continue Goal-mode only for the selected external WING validation handoff/intake.",
            f"Work only on the selected first non-closed MVP blocker: {selected_blocker}.",
            "Maintain control-plane docs/state/rails that preserve the WING handoff and operator intake.",
            "Use real WING artifacts only when hardware is connected and an operator is supervising.",
            "Keep all live writes behind explicit supervised operator approval.",
        ]
        if pre_hardware_intake_loop_guard:
            allowed_scope.append(
                "After M104, select more WING read-only intake hardening only for a new real artifact, explicit operator request, or named unverified contract gap."
            )
        return allowed_scope
    return [
        "Continue uninterrupted Goal-mode development.",
        "Work only on the selected first non-closed MVP blocker.",
        "Develop correction methods only when they support that blocker or stay offline/proposal-only.",
        "Use docs/runbooks and real WING artifacts when hardware is connected.",
        "Keep all live writes behind explicit supervised operator approval.",
    ]


def forbidden_scope_for_selected_action(
    *, external_validation_wait: bool, pre_hardware_intake_loop_guard: bool = False
) -> list[str]:
    forbidden_scope = [
        "No new memory-trust chains unless they close the selected blocker.",
        "No new queue-import chains unless they close the selected blocker.",
        "No new result-consumer/report-only/read-model work unless it closes the selected blocker.",
        "No live EQ/dynamics/routing/mute/FX/scene/matrix/DCA/autonomous writes.",
        "Do not close readiness from offline, shadow, or candidate evidence.",
    ]
    if external_validation_wait:
        forbidden_scope.extend(
            [
                "No additional proposal-only context-gate milestone while M001 waits for external WING validation unless the selected WING blocker or operator explicitly requests it.",
                "Do not simulate progress by adding offline capability gates when the selected action requires a real WING artifact.",
            ]
        )
    if pre_hardware_intake_loop_guard:
        forbidden_scope.extend(
            [
                "No further generic WING read-only intake guard milestone after M104 without a new real WING artifact, explicit operator request, or named unverified contract gap.",
                "Do not keep improving the same WING read-only intake function as a substitute for M002 physical readback evidence.",
            ]
        )
    return forbidden_scope


def build_directive(*, project_root: Path, lookback_commits: int, report_limit: int) -> dict:
    blocker_path = project_root / ".ai" / "project-layer" / "wing_mvp_blockers.md"
    rows = update_mvp_progress.parse_blocker_table(blocker_path)
    first_open = update_mvp_progress.first_open_blocker(rows)
    current_first_command = update_mvp_progress.current_first_command(blocker_path)
    current_first_command_missing_markers = (
        update_mvp_progress.current_first_command_missing_baseline_markers(
            current_first_command
        )
    )
    evidence = codex_direction_guard.collect_repo_evidence(
        project_root,
        lookback_commits=lookback_commits,
        report_limit=report_limit,
    )
    guard_report = codex_direction_guard.analyze_evidence(evidence)
    recent_commit_subjects = recent_commit_subjects_from_evidence(evidence)

    if first_open:
        selected_blocker = first_open["blocker"]
        selected_action, selected_action_source = (
            update_mvp_progress.selected_action_text(
                blocker_next_action=first_open["next action"],
                current_first_command=current_first_command,
            )
        )
        selected_status = first_open["status"]
        selected_evidence_files = update_mvp_progress.extract_evidence_files(
            first_open["evidence"]
        )
    else:
        selected_blocker = "none"
        selected_action = "All MVP blockers are closed; prepare operator review before widening scope."
        selected_action_source = "all_blockers_closed"
        selected_status = "closed"
        selected_evidence_files = []

    baseline_coverage_required = bool(
        first_open
        and current_first_command
        and current_first_command != first_open["next action"]
    )
    external_validation_wait = selected_action_requires_external_validation(
        status=selected_status,
        action=selected_action,
    )
    intake_guard_streak = wing_readonly_intake_guard_streak(recent_commit_subjects)
    pre_hardware_intake_loop_guard = pre_hardware_intake_hardening_loop_guard_required(
        external_validation_wait=external_validation_wait,
        recent_commit_subjects=recent_commit_subjects,
    )
    stop_broad_work = bool(guard_report["stop_the_line"]) or selected_status != "closed"
    required_files = update_mvp_progress.unique_ordered(
        [
            "Docs/MVP_WING_SUPERVISED.md",
            ".ai/project-layer/wing_mvp_blockers.md",
            "Docs/CORRECTION_METHODS_ROADMAP.md",
            ".ai/project-layer/correction_methods_backlog.md",
            "Docs/runbooks/WING_READONLY_PREFLIGHT.md",
            "backend/wing_readonly_preflight_contract.py",
            "backend/wing_readonly_preflight_review_decision.py",
            "backend/supervised_wing_proof_plan.py",
            "tools/wing_readonly_preflight_runner.py",
            "tools/review_wing_readonly_preflight_artifacts.py",
            "tools/review_wing_readonly_preflight_decisions.py",
            "tools/build_wing_readonly_preflight_handoff_packet.py",
            "tools/no_wing_release_check.py",
            "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
            "Docs/reports/MVP_PROGRESS_LATEST.md",
            *required_file_candidates_from_evidence(selected_evidence_files),
        ]
    )
    return {
        "type": "project_director_directive",
        "generated_at": utc_now(),
        "current_mvp": "supervised_wing_rack_proof",
        "guard_status": guard_report["status"],
        "guard_score": guard_report["score"],
        "guard_stop_the_line": guard_report["stop_the_line"],
        "continue_goal_mode": True,
        "stop_scope": "broad_work_only",
        "selected_blocker": selected_blocker,
        "selected_blocker_status": selected_status,
        "selected_action": selected_action,
        "selected_action_source": selected_action_source,
        "selected_action_sha256": sha256_text(selected_action),
        "current_first_command_baseline_coverage_required": baseline_coverage_required,
        "current_first_command_baseline_coverage_passed": (
            bool(current_first_command) and not current_first_command_missing_markers
        ),
        "current_first_command_baseline_coverage_marker_count": len(
            update_mvp_progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS
        ),
        "current_first_command_baseline_missing_markers": (
            current_first_command_missing_markers
        ),
        "selected_evidence_file_count": len(selected_evidence_files),
        "selected_evidence_files": selected_evidence_files,
        "stop_broad_work": stop_broad_work,
        "external_validation_wait": external_validation_wait,
        "selected_action_requires_external_validation": external_validation_wait,
        "pre_hardware_intake_hardening_loop_guard": pre_hardware_intake_loop_guard,
        "pre_hardware_intake_guard_recent_streak_count": len(intake_guard_streak),
        "pre_hardware_intake_guard_min_streak": POST_M104_INTAKE_GUARD_MIN_STREAK,
        "pre_hardware_intake_guard_recent_subjects": intake_guard_streak,
        "allowed_autonomous_scope": (
            "external_validation_wait_or_named_gap_only"
            if pre_hardware_intake_loop_guard
            else
            "external_validation_handoff_or_control_plane_only"
            if external_validation_wait
            else "selected_blocker_only"
        ),
        "allowed_scope": allowed_scope_for_selected_action(
            selected_blocker=selected_blocker,
            external_validation_wait=external_validation_wait,
            pre_hardware_intake_loop_guard=pre_hardware_intake_loop_guard,
        ),
        "forbidden_scope": forbidden_scope_for_selected_action(
            external_validation_wait=external_validation_wait,
            pre_hardware_intake_loop_guard=pre_hardware_intake_loop_guard,
        ),
        "required_files": required_files,
    }


def render_markdown(directive: dict) -> str:
    missing_markers_json = json.dumps(
        directive.get("current_first_command_baseline_missing_markers", []),
        ensure_ascii=False,
    )
    lines = [
        "# Project Director Agent Directive",
        "",
        f"Generated: `{directive['generated_at']}`",
        f"Current MVP: `{directive['current_mvp']}`",
        f"Guard status: `{directive['guard_status']}`",
        f"Guard score: `{directive['guard_score']}`",
        f"Stop broad work: `{str(directive['stop_broad_work']).lower()}`",
        f"Continue Goal mode: `{str(directive['continue_goal_mode']).lower()}`",
        "External validation wait: "
        f"`{str(directive.get('external_validation_wait', False)).lower()}`",
        "Pre-hardware intake hardening loop guard: "
        f"`{str(directive.get('pre_hardware_intake_hardening_loop_guard', False)).lower()}`",
        "- Pre-hardware intake guard recent streak count: "
        f"`{directive.get('pre_hardware_intake_guard_recent_streak_count', 0)}`",
        "- Pre-hardware intake guard min streak: "
        f"`{directive.get('pre_hardware_intake_guard_min_streak', 0)}`",
        f"Allowed autonomous scope: `{directive.get('allowed_autonomous_scope', 'selected_blocker_only')}`",
        "",
        "## Selected Work",
        "",
        f"- Selected blocker: `{directive['selected_blocker']}`",
        f"- Blocker status: `{directive['selected_blocker_status']}`",
        f"- Selected action source: `{directive.get('selected_action_source', 'unknown')}`",
        f"- Selected action sha256: `{directive.get('selected_action_sha256', '')}`",
        "- Current first command baseline coverage required: "
        f"`{str(directive.get('current_first_command_baseline_coverage_required', False)).lower()}`",
        "- Current first command baseline coverage passed: "
        f"`{str(directive.get('current_first_command_baseline_coverage_passed', False)).lower()}`",
        "- Current first command baseline coverage marker count: "
        f"`{directive.get('current_first_command_baseline_coverage_marker_count', 0)}`",
        "- Current first command baseline missing marker count: "
        f"`{len(directive.get('current_first_command_baseline_missing_markers', []))}`",
        "- Current first command baseline missing markers: "
        f"`{missing_markers_json}`",
        f"- Selected action: {directive['selected_action']}",
        f"- Selected evidence file count: `{directive.get('selected_evidence_file_count', 0)}`",
        "",
        "## Allowed Scope",
        "",
    ]
    lines.extend(f"- {item}" for item in directive["allowed_scope"])
    lines.extend(["", "## Forbidden Scope", ""])
    lines.extend(f"- {item}" for item in directive["forbidden_scope"])
    lines.extend(["", "## Required Files", ""])
    lines.extend(f"- `{item}`" for item in directive["required_files"])
    if directive.get("selected_evidence_files"):
        lines.extend(["", "## Selected Evidence Files", ""])
        lines.extend(f"- `{item}`" for item in directive["selected_evidence_files"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This director is an audit/planning agent only. It does not run WING,",
            "does not write code by itself, and does not grant live readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Automixer Project Director directive.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lookback-commits", type=int, default=12)
    parser.add_argument("--report-limit", type=int, default=10)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--print", action="store_true")
    parser.add_argument(
        "--stop-the-line",
        action="store_true",
        help="Exit non-zero when broad work must stop and Codex must follow the selected blocker.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directive = build_directive(
        project_root=Path(args.project_root).resolve(),
        lookback_commits=args.lookback_commits,
        report_limit=args.report_limit,
    )
    text = (
        json.dumps(directive, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(directive)
    )
    write_output(Path(args.output), text)
    if args.print:
        print(text)
    if args.stop_the_line and directive["stop_broad_work"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
