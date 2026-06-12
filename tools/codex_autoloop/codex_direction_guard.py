#!/usr/bin/env python3
"""Check whether recent Codex work is moving Automixer toward the current goal.

The guard is intentionally read-only. It inspects git status, recent commit
subjects, and recent repo-local report names/content snippets, then produces a
direction report for the next Codex cycle. It never opens mixer transports and
never grants live authority.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIRS = (".paperclip/reports", "Docs/reports")
MVP_BLOCKERS_PATH = ".ai/project-layer/wing_mvp_blockers.md"
STOP_THE_LINE_STATUSES = {"drifting", "mixed_needs_review", "needs_direction"}

TARGET_PATTERNS = (
    "wing read-only",
    "wing readonly",
    "wing_readonly",
    "read-only preflight",
    "readonly preflight",
    "readonly_preflight",
    "physical readback",
    "physical_readback",
    "console readback",
    "supervised fader",
    "supervised_fader",
    "supervised gain",
    "supervised_gain",
    "gain-down",
    "gain_down",
    "rollback proof",
    "rollback",
    "emergency stop",
    "emergency_stop",
    "timeout fail-closed",
    "timeout fail closed",
    "fail-closed",
    "fail_closed",
    "operator-visible evidence",
    "operator visible evidence",
    "hardware proof",
    "wing proof",
    "readiness matrix update based on real proof",
)

DRIFT_PATTERNS = (
    "memory-trust",
    "memory_trust",
    "promotion memory",
    "promotion_memory",
    "queue-import",
    "queue_import",
    "queue import",
    "result consumer",
    "result_consumer",
    "productstate promotion memory",
    "product_state promotion",
    "operator-proposal-apply-sequence-policy-memory",
    "broad report layer",
    "report pipeline",
    "backend read model",
    "frontend read-only ux",
    "new proposal-only surface",
    "proposal-only surface",
    "shadow-only method",
)

NEXT_ALLOWED_ACTIONS = (
    (
        "Before starting the intended backend/WING session, run "
        "`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`; "
        "then run read-only preflight and capture a real artifact."
    ),
    "Review a real wing_readonly_preflight.v1 artifact before any readiness update.",
    "Prepare supervised fader/gain proof only after explicit operator approval.",
    "Preserve rollback, emergency-stop, timeout/fail-closed, and readback evidence.",
    "Update readiness matrix only from real proof, not from offline/shadow evidence.",
)

FORBIDDEN_NEXT_ACTIONS = (
    "Do not add memory-trust, queue-import, or result-consumer layers unless they close a named WING blocker.",
    "Do not add broad report/read-model/frontend layers as a substitute for hardware proof.",
    "Do not treat shadow-only/offline evidence as live readiness.",
    "Do not widen live writes beyond supervised fader/gain without a separate ADR and hardware proof.",
)

DRIFT_POLICY_SECTION_TITLES = frozenset(
    {
        "forbidden next actions",
        "forbidden scope",
    }
)
DERIVATIVE_GUARD_SECTION_TITLES = frozenset(
    {
        "target evidence",
        "drift evidence",
        "mixed evidence",
    }
)


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    text: str
    target_matches: tuple[str, ...]
    drift_matches: tuple[str, ...]

    @property
    def is_target(self) -> bool:
        return bool(self.target_matches)

    @property
    def is_drift(self) -> bool:
        return bool(self.drift_matches)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def pattern_matches(text: str, patterns: Iterable[str]) -> tuple[str, ...]:
    normalized = normalize(text)
    compact = normalized.replace("-", " ")
    matches: list[str] = []
    for pattern in patterns:
        normalized_pattern = normalize(pattern)
        compact_pattern = normalized_pattern.replace("-", " ")
        if normalized_pattern in normalized or compact_pattern in compact:
            matches.append(pattern)
    return tuple(matches)


def markdown_heading_level(line: str) -> int | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    level = len(stripped) - len(stripped.lstrip("#"))
    if level == 0 or level > 6:
        return None
    if len(stripped) == level or stripped[level] != " ":
        return None
    return level


def markdown_heading_title(line: str) -> str:
    stripped = line.lstrip()
    return stripped.lstrip("#").strip().lower()


def text_without_markdown_sections(text: str, section_titles: Iterable[str]) -> str:
    skipped_titles = set(section_titles)
    lines: list[str] = []
    skipping_section = False
    for line in text.splitlines():
        level = markdown_heading_level(line)
        if level is not None and level <= 2:
            title = markdown_heading_title(line)
            if title in skipped_titles:
                skipping_section = True
                continue
            skipping_section = False
        if not skipping_section:
            lines.append(line)
    return "\n".join(lines)


def text_without_json_policy_fields(text: str) -> str:
    stripped = text.lstrip()
    prefix = ""
    json_text = stripped
    if not stripped.startswith("{") and "\n" in stripped:
        first_line, rest = stripped.split("\n", 1)
        if rest.lstrip().startswith("{"):
            prefix = first_line
            json_text = rest.lstrip()
    if not json_text.startswith("{"):
        return text
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("forbidden_scope", None)
        payload.pop("forbidden_next_actions", None)
        filtered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return f"{prefix}\n{filtered}" if prefix else filtered
    return text


def is_generated_direction_guard_report(source: str, text: str) -> bool:
    return source.startswith("report:") and (
        "codex_direction_guard_" in source
        or "# Codex Direction Guard" in text[:300]
    )


def text_for_drift_scoring(source: str, text: str) -> str:
    ignored_sections = set(DRIFT_POLICY_SECTION_TITLES)
    if is_generated_direction_guard_report(source, text):
        ignored_sections.update(DERIVATIVE_GUARD_SECTION_TITLES)
    filtered = text_without_markdown_sections(text, ignored_sections)
    return text_without_json_policy_fields(filtered)


def classify_item(source: str, text: str) -> EvidenceItem:
    drift_text = text_for_drift_scoring(source, text)
    return EvidenceItem(
        source=source,
        text=text.strip(),
        target_matches=pattern_matches(text, TARGET_PATTERNS),
        drift_matches=pattern_matches(drift_text, DRIFT_PATTERNS),
    )


def analyze_evidence(raw_items: Iterable[tuple[str, str]]) -> dict:
    items = tuple(
        item
        for source, text in raw_items
        if text.strip()
        for item in (classify_item(source, text),)
    )
    target_items = tuple(item for item in items if item.is_target)
    drift_items = tuple(item for item in items if item.is_drift and not item.is_target)
    mixed_items = tuple(item for item in items if item.is_target and item.is_drift)

    score = 50
    score += min(40, len(target_items) * 8)
    score -= min(60, len(drift_items) * 15 + len(mixed_items) * 5)
    if not target_items:
        score -= 20
    score = max(0, min(100, score))

    if not target_items:
        status = "drifting"
        verdict = "No current evidence points at the supervised WING proof blockers."
    elif drift_items and len(drift_items) > len(target_items):
        status = "drifting"
        verdict = "Drift evidence is stronger than WING proof evidence."
    elif drift_items or mixed_items:
        status = "mixed_needs_review"
        verdict = "There is WING proof progress, but the cycle still carries drift signals."
    elif score >= 70:
        status = "on_target"
        verdict = "Recent work is aligned with supervised WING Rack MVP blockers."
    else:
        status = "needs_direction"
        verdict = "Some WING proof evidence exists, but the next step must be tightened."

    return {
        "type": "codex_direction_guard",
        "generated_at": utc_now(),
        "current_focus": "supervised_wing_rack_mvp",
        "activation_policy": {
            "before_plan": True,
            "before_final_report": True,
            "wall_clock_schedule": "every Codex autoloop cycle; LaunchAgent interval remains 21600 seconds when installed",
        },
        "mvp_blocker_table": MVP_BLOCKERS_PATH,
        "status": status,
        "score": score,
        "stop_the_line": status in STOP_THE_LINE_STATUSES,
        "continue_goal_mode": True,
        "stop_scope": "broad_work_only",
        "verdict": verdict,
        "summary": {
            "evidence_count": len(items),
            "target_item_count": len(target_items),
            "drift_item_count": len(drift_items),
            "mixed_item_count": len(mixed_items),
            "action_required": status != "on_target",
        },
        "next_allowed_actions": list(NEXT_ALLOWED_ACTIONS),
        "forbidden_next_actions": list(FORBIDDEN_NEXT_ACTIONS),
        "target_items": [asdict(item) for item in target_items[:12]],
        "drift_items": [asdict(item) for item in drift_items[:12]],
        "mixed_items": [asdict(item) for item in mixed_items[:12]],
    }


def git_bin() -> str:
    preferred = Path("/opt/homebrew/bin/git")
    if preferred.exists():
        return str(preferred)
    return os.environ.get("GIT_BIN", "git")


def run_git(project_root: Path, args: Sequence[str]) -> list[str]:
    try:
        result = subprocess.run(
            [git_bin(), *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        return [f"git {' '.join(args)} failed: {type(exc).__name__}: {exc}"]
    output = result.stdout if result.stdout.strip() else result.stderr
    return [line for line in output.splitlines() if line.strip()]


def recent_report_items(project_root: Path, *, limit: int) -> list[tuple[str, str]]:
    paths: list[Path] = []
    for report_dir in REPORT_DIRS:
        root = project_root / report_dir
        if root.is_dir():
            paths.extend(path for path in root.glob("*.md") if path.is_file())
    latest = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]

    items: list[tuple[str, str]] = []
    for path in latest:
        rel = path.relative_to(project_root)
        try:
            snippet = path.read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError as exc:
            snippet = f"could not read report: {type(exc).__name__}: {exc}"
        items.append((f"report:{rel}", f"{path.name}\n{snippet}"))
    return items


def collect_repo_evidence(project_root: Path, *, lookback_commits: int, report_limit: int) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    items.extend(("git_status", line) for line in run_git(project_root, ["status", "--short"]))
    items.extend(
        ("git_log", line)
        for line in run_git(
            project_root,
            ["log", f"--max-count={lookback_commits}", "--pretty=format:%h %s"],
        )
    )
    items.extend(recent_report_items(project_root, limit=report_limit))
    return items


def render_markdown(report: dict) -> str:
    lines = [
        "# Codex Direction Guard",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Current focus: `{report['current_focus']}`",
        f"Status: `{report['status']}`",
        f"Score: `{report['score']}`",
        "",
        "## Verdict",
        "",
        str(report["verdict"]),
        "",
        "## Activation",
        "",
        "- Runs before Codex selects a task for the cycle.",
        "- Runs again before the final cycle report.",
        "- Wall-clock activation follows the existing Codex autoloop interval: every 21600 seconds when the LaunchAgent is installed.",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Next Allowed Actions", ""])
    lines.extend(f"- {item}" for item in report["next_allowed_actions"])
    lines.extend(["", "## Forbidden Next Actions", ""])
    lines.extend(f"- {item}" for item in report["forbidden_next_actions"])

    for title, key in (
        ("Target Evidence", "target_items"),
        ("Drift Evidence", "drift_items"),
        ("Mixed Evidence", "mixed_items"),
    ):
        lines.extend(["", f"## {title}", ""])
        entries = report[key]
        if not entries:
            lines.append("- None.")
            continue
        for item in entries:
            text = " ".join(str(item["text"]).split())[:220]
            target = ", ".join(item["target_matches"]) or "-"
            drift = ", ".join(item["drift_matches"]) or "-"
            lines.append(f"- `{item['source']}`: {text}")
            lines.append(f"  - target: `{target}`")
            lines.append(f"  - drift: `{drift}`")

    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Codex direction for Automixer.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--lookback-commits", type=int, default=12)
    parser.add_argument("--report-limit", type=int, default=10)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", help="Optional output path for the guard report.")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit non-zero when direction is drifting or mixed.",
    )
    parser.add_argument(
        "--stop-the-line",
        action="store_true",
        help="Exit non-zero when broad work should stop and Codex should return to the first open MVP blocker.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    evidence = collect_repo_evidence(
        project_root,
        lookback_commits=args.lookback_commits,
        report_limit=args.report_limit,
    )
    report = analyze_evidence(evidence)
    text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        write_report(Path(args.output), text)
    else:
        print(text)
    if args.stop_the_line and report["stop_the_line"]:
        return 2
    if args.fail_on_drift and report["status"] in {"drifting", "mixed_needs_review"}:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
