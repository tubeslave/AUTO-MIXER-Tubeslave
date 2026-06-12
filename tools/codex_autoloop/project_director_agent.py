#!/usr/bin/env python3
"""Repo-local Project Director Agent for Automixer.

The director reads the MVP rail files and the direction guard, then emits the
next Codex directive. It does not edit production code, run tests, connect to
WING, or grant live authority.
"""

from __future__ import annotations

import argparse
import json
import re
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
EVIDENCE_PATH_RE = re.compile(r"`([^`]+)`")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def unique_ordered(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def extract_evidence_files(evidence_text: str) -> list[str]:
    return unique_ordered(
        match.group(1)
        for match in EVIDENCE_PATH_RE.finditer(evidence_text)
        if "/" in match.group(1)
    )


def required_file_candidates_from_evidence(evidence_files: Sequence[str]) -> list[str]:
    return [
        path
        for path in evidence_files
        if path.startswith(("backend/", "tools/", "Docs/runbooks/"))
    ]


def build_directive(*, project_root: Path, lookback_commits: int, report_limit: int) -> dict:
    rows = update_mvp_progress.parse_blocker_table(
        project_root / ".ai" / "project-layer" / "wing_mvp_blockers.md"
    )
    first_open = update_mvp_progress.first_open_blocker(rows)
    evidence = codex_direction_guard.collect_repo_evidence(
        project_root,
        lookback_commits=lookback_commits,
        report_limit=report_limit,
    )
    guard_report = codex_direction_guard.analyze_evidence(evidence)

    if first_open:
        selected_blocker = first_open["blocker"]
        selected_action = first_open["next action"]
        selected_status = first_open["status"]
        selected_evidence_files = extract_evidence_files(first_open["evidence"])
    else:
        selected_blocker = "none"
        selected_action = "All MVP blockers are closed; prepare operator review before widening scope."
        selected_status = "closed"
        selected_evidence_files = []

    stop_broad_work = bool(guard_report["stop_the_line"]) or selected_status != "closed"
    required_files = unique_ordered(
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
        "selected_evidence_file_count": len(selected_evidence_files),
        "selected_evidence_files": selected_evidence_files,
        "stop_broad_work": stop_broad_work,
        "allowed_scope": [
            "Continue uninterrupted Goal-mode development.",
            "Work only on the selected first non-closed MVP blocker.",
            "Develop correction methods only when they support that blocker or stay offline/proposal-only.",
            "Use docs/runbooks and real WING artifacts when hardware is connected.",
            "Keep all live writes behind explicit supervised operator approval.",
        ],
        "forbidden_scope": [
            "No new memory-trust chains unless they close the selected blocker.",
            "No new queue-import chains unless they close the selected blocker.",
            "No new result-consumer/report-only/read-model work unless it closes the selected blocker.",
            "No live EQ/dynamics/routing/mute/FX/scene/matrix/DCA/autonomous writes.",
            "Do not close readiness from offline, shadow, or candidate evidence.",
        ],
        "required_files": required_files,
    }


def render_markdown(directive: dict) -> str:
    lines = [
        "# Project Director Agent Directive",
        "",
        f"Generated: `{directive['generated_at']}`",
        f"Current MVP: `{directive['current_mvp']}`",
        f"Guard status: `{directive['guard_status']}`",
        f"Guard score: `{directive['guard_score']}`",
        f"Stop broad work: `{str(directive['stop_broad_work']).lower()}`",
        f"Continue Goal mode: `{str(directive['continue_goal_mode']).lower()}`",
        "",
        "## Selected Work",
        "",
        f"- Selected blocker: `{directive['selected_blocker']}`",
        f"- Blocker status: `{directive['selected_blocker_status']}`",
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
